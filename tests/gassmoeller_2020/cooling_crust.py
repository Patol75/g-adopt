from firedrake import *
from ufl.algebra import Operator

from gadopt import (
    BackwardEuler,
    ImplicitMidpoint,
    TimestepAdaptor,
    get_boundary_ids,
    time_objects,
)

from approximations_solvers import Approximation, EnergySolver, StokesSolver


def viscous_creep(C) -> Operator:
    return 1e24 * (1.0 - C + 0.01 * C)


def yield_strength(p) -> Operator:
    angle = 20.0 / 180.0 * pi

    return 2e7 * cos(angle) + p * sin(angle)


def plastic_deformation(u, p) -> Operator:
    strain_rate_dev = dev(apx.strain_rate(u))
    strain_rate_dev_sec_inv = sqrt(
        inner(strain_rate_dev, strain_rate_dev) / 2.0 + 1e-30
    )

    return min_value(
        max_value(yield_strength(p) / 2.0 / strain_rate_dev_sec_inv, 1e21), 1e24
    )


def effective_viscosity(u, p, C) -> Operator:
    return min_value(viscous_creep(C), plastic_deformation(u, p))


def write_output() -> None:
    temperature.project(apx.ref_profiles["T"] + T)
    match apx.name:
        case "BA" | "EBA" | "TALA" | "ALA":
            density.project(apx.density(p, T, weak_layer) - apx.ref_profiles["rho"])
        case "ICA" | "HCA" | "PDA":
            density.project(apx.density(p, T, weak_layer) - apx.density(0.0, 0.0, 0.0))
    viscosity.interpolate(viscosity_eff)
    shear_stress = apx.shear_stress(u, viscosity_eff)
    dev_stress_sec_inv.interpolate(sqrt(inner(shear_stress, shear_stress) / 2.0))

    pvd.write(
        *stokes.subfunctions,
        temperature,
        density,
        weak_layer,
        viscosity,
        dev_stress_sec_inv,
        time=float(time) / year_to_seconds,
    )


domain_dims = (3e5, 3e4)
mesh_resolution = 15.0 / 16.0 * 1e3
mesh_cells = [int(dim / mesh_resolution) for dim in domain_dims]
mesh = RectangleMesh(*mesh_cells, *domain_dims, quadrilateral=True)
mesh.cartesian = True
boundary = get_boundary_ids(mesh)
x, y = SpatialCoordinate(mesh)
depth = domain_dims[1] - y

V = VectorFunctionSpace(mesh, "Q", 2)
W = FunctionSpace(mesh, "Q", 1)
Z = MixedFunctionSpace([V, W, W])
Q = FunctionSpace(mesh, "Q", 2)
R = FunctionSpace(mesh, "R", 0)

stokes = Function(Z, name="Stokes")
stokes.subfunctions[0].rename("Velocity")
stokes.subfunctions[1].rename("Pressure")
stokes.subfunctions[2].rename("Free surface")
u, p = split(stokes)[:2]
T = Function(Q, name="Temperature (deviation)")
temperature = Function(W, name="Temperature")
density = Function(W, name="Density (deviation)")
weak_layer = Function(W, name="Weak layer (proportion)")
viscosity = Function(W, name="Viscosity")
dev_stress_sec_inv = Function(W, name="Deviatoric stress (second invariant)")
time_step, time = time_objects(mesh, dt=1e12)

T.interpolate(
    -500.0 * y / domain_dims[1] * (0.5 + 0.5 * tanh((x - y - 1.35e5) / 2.0 / 2.5e3))
)
weak_layer.interpolate(exp(-((x - y - 1.35e5) ** 2.0) / 2.0 / 2.5e3**2.0))

ref_state = {
    "alpha": Function(R, name="Coefficient of thermal expansion (state)").assign(4e-5),
    "cp": Function(R, name="Isobaric specific heat capacity (state)").assign(750.0),
    "eta": Function(R, name="Dynamic viscosity (state)").assign(1e24),
    "g": Function(R, name="Gravitational acceleration (state)").assign(10.0),
    "H": Function(R, name="Rate of radiogenic heat production (state)").assign(0.0),
    "K": Function(R, name="Bulk modulus (state)").assign(3.125e11),
    "k": Function(R, name="Thermal conductivity (state)").assign(2.5),
    "p": Function(R, name="Pressure (state)").assign(0.0),
    "rho": Function(R, name="Density (state)").assign(2800.0),
    "T": Function(R, name="Temperature (state)").assign(800.0),
}
ref_profiles = {}
ref_profiles["rho"] = ref_state["rho"] * exp(
    ref_state["rho"] * ref_state["g"] / ref_state["K"] * depth
)
ref_profiles["p"] = ref_state["p"] + ref_state["K"] * (
    ref_profiles["rho"] / ref_state["rho"] - 1.0
)
ref_profiles["T"] = ref_state["T"] * exp(
    ref_state["alpha"] * ref_state["g"] / ref_state["cp"] * depth
)
apx = Approximation("PDA", ref_state, ref_profiles)

viscosity_eff = effective_viscosity(u, ref_profiles["p"] + p, weak_layer)

energy_bcs = [
    DirichletBC(Q, 800.0 - ref_profiles["T"], boundary.bottom),
    DirichletBC(Q, 300.0 - ref_profiles["T"], boundary.top),
]
for bc in energy_bcs:
    bc.apply(T)
stokes_bcs = [
    DirichletBC(Z.subspaces[0].sub(0), 0.0, boundary.left),
    DirichletBC(Z.subspaces[0].sub(0), 0.0, boundary.right),
    DirichletBC(Z.subspaces[0].sub(1), 0.0, boundary.bottom),
]

energy_solver = EnergySolver(
    T,
    apx,
    u,
    time,
    time_step,
    ImplicitMidpoint,
    strong_bcs=energy_bcs,
    disable_shear_heating=True,
)
stokes_solver = StokesSolver(
    stokes,
    apx,
    viscosity=viscosity_eff,
    T=T,
    T_old=energy_solver.irksome_integrator.solution_old if apx.name == "PDA" else 0.0,
    time=time,
    time_step=time_step,
    time_stepper=BackwardEuler,
    strong_bcs=stokes_bcs,
    free_surface_id=boundary.top,
    solver_parameters={
        "snes_monitor": None,
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_max_it": 100,
        "snes_atol": 1e-10,
        "snes_rtol": 1e-5,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)
stokes_solver.solve()

ts_adaptor = TimestepAdaptor(time_step, u, V)
ts_adaptor.update_timestep()

year_to_seconds = 365.25 * 8.64e4
pvd = VTKFile(f"output_cooling_crust_{apx.name}.pvd")
write_output()

while float(time) < 1e7 * year_to_seconds:
    energy_solver.solve()
    stokes_solver.solve()

    time.assign(time + ts_adaptor.update_timestep())

    write_output()

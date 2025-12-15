from firedrake import *

from gadopt import (
    BackwardEuler,
    ImplicitMidpoint,
    TimestepAdaptor,
    get_boundary_ids,
    time_objects,
)

from approximations_solvers import (
    AdvectionSolver,
    Approximation,
    EnergySolver,
    StokesSolver,
)


def density_difference(p, crust):
    return 0.5 * (tanh((apx.ref_profiles["p"] + p - 4e9) / 1e8) - 1.0) * 6e2 * crust


def write_output() -> None:
    temperature.project(apx.ref_profiles["T"] + T)
    density.project(apx.density(p, T, crust) - apx.density(0.0, 0.0, 0.0))
    dev_stress_sec_inv.project(sqrt(inner(shear_stress, shear_stress) / 2.0))

    pvd.write(
        *stokes.subfunctions,
        temperature,
        crust,
        density,
        dev_stress_sec_inv,
        time=float(time) / year_to_seconds,
    )


domain_dims = (3e5, 2e5)
mesh_resolution = 25.0 / 16.0 * 1e3
mesh_cells = [int(dim / mesh_resolution) for dim in domain_dims]
mesh = RectangleMesh(*mesh_cells, *domain_dims, quadrilateral=True)
mesh.cartesian = True
boundary = get_boundary_ids(mesh)
x, y = SpatialCoordinate(mesh)
depth = domain_dims[1] - y

V = VectorFunctionSpace(mesh, "Q", 2)
W = FunctionSpace(mesh, "Q", 1)
Z = MixedFunctionSpace([V, W])
Q = FunctionSpace(mesh, "Q", 2)
K = FunctionSpace(mesh, "Q", 2)
R = FunctionSpace(mesh, "R", 0)

stokes = Function(Z, name="Stokes")
stokes.subfunctions[0].rename("Velocity")
stokes.subfunctions[1].rename("Pressure")
u, p = split(stokes)
T = Function(Q, name="Temperature (deviation)")
crust = Function(K, name="Hydrated crust (proportion)")
temperature = Function(W, name="Temperature")
density = Function(W, name="Density (deviation)")
dev_stress_sec_inv = Function(W, name="Deviatoric stress (second invariant)")
time_step, time = time_objects(mesh, dt=1e12)

T_ini_cond = -1000.0 * exp(-((x - 1e4) ** 2 + depth**2) / 2 / 3.5e4**2)
T.interpolate(T_ini_cond)

ref_state = {
    "alpha": Function(R, name="Coefficient of thermal expansion (state)").assign(2e-5),
    "cp": Function(R, name="Isobaric specific heat capacity (state)").assign(1250.0),
    "eta": Function(R, name="Dynamic viscosity (state)").assign(1e22),
    "g": Function(R, name="Gravitational acceleration (state)").assign(10.0),
    "H": Function(R, name="Rate of radiogenic heat production (state)").assign(0.0),
    "K": Function(R, name="Bulk modulus (state)").assign(3.125e11),
    "k": Function(R, name="Thermal conductivity (state)").assign(4.7),
    "p": Function(R, name="Pressure (state)").assign(0.0),
    "rho": Function(R, name="Density (state)").assign(3300.0),
    "T": Function(R, name="Temperature (state)").assign(1600.0),
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
shear_stress = apx.shear_stress(u, apx.ref_profiles["eta"])

crust_inflow = exp(-((x - 3.5e4) ** 2) / 2 / 1e4**2)
advection_bcs = {
    boundary.left: {"g": 0.0},
    boundary.top: {
        "g": conditional(x < 1e4, 0.0, conditional(x > 6e4, 0.0, crust_inflow))
    },
}
for bc_id, bc_dict in advection_bcs.items():
    DirichletBC(K, bc_dict["g"], bc_id).apply(crust)

energy_bcs = [
    DirichletBC(Q, T_ini_cond, boundary.left),
    DirichletBC(Q, T_ini_cond, boundary.top),
]
for bc in energy_bcs:
    bc.apply(T)

year_to_seconds = 365.25 * 8.64e4
u_bc = 1.0 / 100.0 / year_to_seconds
stokes_bcs = [
    DirichletBC(Z.subspaces[0].sub(0), u_bc, boundary.left),
    DirichletBC(Z.subspaces[0].sub(0), u_bc, boundary.right),
    DirichletBC(Z.subspaces[0].sub(0), u_bc, boundary.bottom),
    DirichletBC(Z.subspaces[0], as_vector([u_bc, -u_bc]), boundary.top),
]

match apx.name:
    case "BA" | "EBA" | "TALA" | "ALA":
        delta_rho = density_difference(p, crust)
    case "ICA" | "HCA" | "PDA":
        delta_rho = density_difference(0.0, crust)

advection_solver = AdvectionSolver(
    crust, u, time, time_step, BackwardEuler, bcs=advection_bcs
)
if apx.name == "PDA":
    delta_rho_old = density_difference(0.0, advection_solver.solver.solution_old)
else:
    delta_rho_old = None

energy_solver = EnergySolver(
    T,
    apx,
    u,
    time,
    time_step,
    ImplicitMidpoint,
    delta_rho=delta_rho,
    strong_bcs=energy_bcs,
)
stokes_solver = StokesSolver(
    stokes,
    apx,
    T=T,
    T_old=energy_solver.irksome_integrator.solution_old if apx.name == "PDA" else 0.0,
    delta_rho=delta_rho,
    delta_rho_old=delta_rho_old,
    time_step=time_step if apx.name == "PDA" else None,
    strong_bcs=stokes_bcs,
    nullspace_kwargs={"closed": False, "rotational": False, "translations": [0, 1]},
)
stokes_solver.solve()

ts_adaptor = TimestepAdaptor(time_step, u, V, target_cfl=0.55)
ts_adaptor.update_timestep()

pvd = VTKFile(f"output_phase_transition_{apx.name}.pvd")
write_output()

while float(time) < 7.5e7 * year_to_seconds:
    advection_solver.solve()
    energy_solver.solve()
    stokes_solver.solve()

    time.assign(time + ts_adaptor.update_timestep())

    write_output()

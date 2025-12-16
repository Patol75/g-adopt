import firedrake as fd
from irksome import Dt
from ufl.algebra import Operator

from gadopt import GenericTransportSolver, rigid_body_modes
from gadopt.stokes_integrators import direct_stokes_solver_parameters
from gadopt.time_stepper import IrksomeIntegrator
from gadopt.transport_solver import direct_energy_solver_parameters
from gadopt.utility import InteriorBC, upward_normal


class Approximation:
    """Collection of physical parameter values, profiles, and expressions.

    Args:
        name:
            Name of the chosen approximation for the system of conservation equations
        ref_state:
            Dictionary of physical parameters and their values in the reference state
        ref_profiles:
            Dictionary of physical parameters and their reference depth-dependent
            profiles
    """

    def __init__(
        self, name: str, ref_state: dict[str, float], ref_profiles: dict[str, float]
    ) -> None:
        self.name = name
        self.ref_state = ref_state
        self.ref_profiles = ref_state | ref_profiles

    def density(self, p: fd.Function, T: fd.Function, delta_rho: Operator) -> Operator:
        """Calculates full density.

        Args:
            p:
                Pressure variation with respect to the reference profile
            T:
                Temperature variation with respect to the reference profile
            delta_rho:
                Density variation with respect to the reference composition

        Returns:
            rho:
                Density of the fluid
        """
        match self.name:
            case "BA" | "EBA" | "TALA" | "ALA":
                rho = self.ref_profiles["rho"] * (1.0 - self.ref_profiles["alpha"] * T)
                if self.name == "ALA":
                    rho += self.ref_profiles["rho"] * p / self.ref_profiles["K"]

                return rho + delta_rho
            case "ICA" | "HCA" | "PDA":
                return (self.ref_state["rho"] + delta_rho) * fd.exp(
                    (self.ref_profiles["p"] + p - self.ref_state["p"])
                    / self.ref_profiles["K"]
                    - self.ref_profiles["alpha"]
                    * (self.ref_profiles["T"] + T - self.ref_state["T"])
                )

    def strain_rate(self, u: fd.Function) -> Operator:
        """Calculates full strain rate.

        Args:
            u:
                Velocity field

        Returns:
            epsilon_dot:
                Strain-rate field
        """
        return fd.sym(fd.grad(u))

    def shear_stress(self, u: fd.Function, eta: Operator) -> Operator:
        """Calculates shear stress.

        The shear stress in mantle convection formulations should correspond to the
        deviatoric part of the full stress tensor. We note that the below implementation
        via `dev` removes half of the tensor's trace in 2-D and a third of it in 3-D,
        which could differ from the formulation of certain benchmarks.

        Args:
            u:
                Velocity field
            eta:
                Effective viscosity

        Returns:
            tau:
                Shear-stress field
        """
        return 2.0 * eta * fd.dev(self.strain_rate(u))


class StokesSolver:
    """Solver for the coupled Stokes system."""

    def __init__(
        self,
        solution: fd.Function,
        apx: Approximation,
        PDA_u_old: bool = False,
        viscosity: float | Operator | None = None,
        T: float | fd.Function = 0.0,
        T_old: float | fd.Function = 0.0,
        delta_rho: float | Operator = 0.0,
        delta_rho_old: float | Operator = 0.0,
        time: fd.Function | None = None,
        time_step: fd.Function | None = None,
        time_stepper: IrksomeIntegrator | None = None,
        strong_bcs: list[fd.DirichletBC] | None = None,
        weak_bcs: dict[str | float, dict[str, float]] | None = None,
        free_surface_id: int | None = None,
        solver_parameters: dict[str, str] = None,
        nullspace_kwargs: dict[str, bool] | None = None,
        transpose_nullspace_kwargs: dict[str, bool] | None = None,
    ) -> None:
        self.solution = solution
        self.set_solution_objects()

        self.apx = apx
        self.PDA_u_old = PDA_u_old
        self.eta = viscosity or self.apx.ref_profiles["eta"]
        self.T = T
        self.T_old = T_old
        self.delta_rho = delta_rho
        self.delta_rho_old = delta_rho_old
        self.time_step = time_step
        self.weak_bcs = weak_bcs or {}
        self.free_surface_id = free_surface_id

        if self.free_surface_id is not None:
            strong_bcs = strong_bcs or []
            strong_bcs.append(
                InteriorBC(self.solution_space[2], 0.0, self.free_surface_id)
            )

        self.set_weak_form()
        self.set_solver(
            time,
            time_step,
            time_stepper,
            strong_bcs,
            solver_parameters,
            nullspace_kwargs,
            transpose_nullspace_kwargs,
        )

    def set_solution_objects(self) -> None:
        self.solution_split = fd.split(self.solution)
        self.solution_old = fd.Function(self.solution)
        self.solution_old_split = fd.split(self.solution_old)
        self.solution_space = self.solution.function_space()
        self.mesh = self.solution_space.mesh()

    def set_weak_form(self) -> None:
        """Defines variables needed in the weak form."""
        self.tests = fd.TestFunctions(self.solution_space)
        self.n = fd.FacetNormal(self.mesh)
        self.up = upward_normal(self.mesh)
        self.dx = fd.dx(degree=6)
        self.ds = fd.ds(degree=6)

        u, p = self.solution_split[:2]

        match self.apx.name:
            case "BA" | "EBA" | "TALA" | "ALA":
                self.rho = self.apx.ref_profiles["rho"]
                self.rho_full = self.apx.density(p, self.T, self.delta_rho)
            case "ICA" | "HCA" | "PDA":
                self.rho = self.apx.density(0.0, self.T, self.delta_rho)
                self.rho_full = self.rho

        self.g = -self.apx.ref_profiles["g"] * self.up
        self.rho_g = (self.rho_full - self.apx.ref_profiles["rho"]) * self.g

        match self.apx.name:
            case "ICA":
                self.K = self.apx.ref_profiles["K"]
            case "HCA":
                self.alpha = self.apx.ref_profiles["alpha"]
                self.K = self.apx.ref_profiles["K"]
                self.T_full = self.apx.ref_profiles["T"] + self.T
            case "PDA":
                self.rho_old = self.apx.density(0.0, self.T_old, self.delta_rho_old)

        self.shear_stress = self.apx.shear_stress(u, self.eta)

    def set_solver(
        self,
        time: fd.Function | None = None,
        time_step: fd.Function | None = None,
        time_stepper: IrksomeIntegrator | None = None,
        strong_bcs: list[fd.DirichletBC] | None = None,
        solver_parameters: dict[str, str] = None,
        nullspace_kwargs: dict[str, bool] | None = None,
        transpose_nullspace_kwargs: dict[str, bool] | None = None,
    ):
        """Sets up the solver for the weak form."""

        def process_null_space_kwargs(
            kwargs: dict[str, bool],
        ) -> None | fd.MixedVectorSpaceBasis:
            return None if kwargs is None else self.null_space(**kwargs)

        solver_parameters = (
            solver_parameters
            or {"snes_type": "ksponly", "snes_monitor": None}
            | direct_stokes_solver_parameters
        )

        nullspace = process_null_space_kwargs(nullspace_kwargs)
        transpose_nullspace = process_null_space_kwargs(transpose_nullspace_kwargs)

        if time_stepper is not None:
            self.irksome_integrator = time_stepper(
                self.residual(),
                self.solution,
                time,
                time_step,
                solution_old=self.solution_old,
                strong_bcs=strong_bcs,
                solver_parameters=solver_parameters,
                nullspace=nullspace,
                transpose_nullspace=transpose_nullspace,
            )
        else:
            variational_problem = fd.NonlinearVariationalProblem(
                self.residual(), self.solution, bcs=strong_bcs
            )
            self.solver = fd.NonlinearVariationalSolver(
                variational_problem,
                solver_parameters=solver_parameters,
                options_prefix="Stokes",
                nullspace=nullspace,
                transpose_nullspace=transpose_nullspace,
            )

    def mass_equation(self) -> fd.Form:
        "Variational form of the mass equation."
        u = self.solution_split[0]
        u_old = self.solution_old_split[0]

        match self.apx.name:
            case "BA" | "EBA":
                mass_terms = fd.div(u)
            case "TALA" | "ALA":
                mass_terms = fd.div(self.rho * u)
            case "ICA":
                mass_terms = fd.div(u) + self.rho / self.K * fd.dot(u, self.g)
            case "HCA":
                mass_terms = (
                    fd.div(u)
                    + self.rho / self.K * fd.dot(u_old, self.g)
                    - self.alpha * fd.dot(u_old, fd.grad(self.T_full))
                )
            case "PDA":
                mass_terms = (
                    (self.rho - self.rho_old) / self.time_step
                    + fd.dot(u_old if self.PDA_u_old else u, fd.grad(self.rho))
                    + self.rho * fd.div(u)
                )

        return self.tests[1] * mass_terms * self.dx

    def momentum_equation(self) -> fd.Form:
        "Variational form of the momentum equation."
        p = self.solution_split[1]

        weak_form = (
            fd.div(self.tests[0]) * p
            - fd.inner(fd.nabla_grad(self.tests[0]), self.shear_stress)
            + fd.dot(self.tests[0], self.rho_g)
        ) * self.dx
        # weak_form += -fd.dot(self.tests[0], self.n) * p * self.ds
        for bc_id, bc_dict in self.weak_bcs.items():
            if "traction" in bc_dict:
                weak_form += fd.dot(self.tests[0], bc_dict["traction"]) * self.ds(bc_id)

        return weak_form

    def free_surface_equation(self) -> fd.Form:
        "Variational form of the free-surface equation."
        u, _, h = self.solution_split

        weak_form = (
            self.tests[2] * (fd.dot(self.n, self.up) * Dt(h) - fd.dot(u, self.n))
            - fd.dot(self.tests[0], self.n)
            * fd.dot(self.rho_full * self.g, self.up)
            * h
        ) * self.ds(self.free_surface_id)

        return weak_form

    def residual(self) -> fd.Form:
        """Residual form of the system."""
        residual = self.mass_equation() + self.momentum_equation()
        if self.free_surface_id is not None:
            residual += self.free_surface_equation()

        return residual

    def null_space(
        self,
        closed: bool,
        rotational: bool,
        translations: list[int] | None,
        boundary_id: int | str | None = None,
    ) -> fd.MixedVectorSpaceBasis:
        """Defines a null space for the Stokes system."""
        V_nullspace = rigid_body_modes(
            self.solution_space[0], rotational=rotational, translations=translations
        )

        if closed:
            if self.apx.name != "ALA" or boundary_id is None:
                p_nullspace = fd.VectorSpaceBasis(constant=True, comm=self.mesh.comm)
            else:
                pressure_space = fd.FunctionSpace(
                    mesh=self.mesh.unique(), family=self.solution_space[1].ufl_element()
                )
                test = fd.TestFunction(pressure_space)
                kernel = fd.Function(pressure_space, name="Pressure null space")
                rho_g = (self.apx.density(kernel, 0.0, 0.0) - self.rho) * self.g

                F = fd.dot(fd.grad(test), fd.grad(kernel) - rho_g) * self.dx
                bcs = fd.DirichletBC(pressure_space, 1.0, boundary_id)
                fd.solve(F == 0.0, kernel, bcs=bcs)

                p_nullspace = fd.VectorSpaceBasis([kernel], comm=self.mesh.comm)
                p_nullspace.orthonormalize()
        else:
            p_nullspace = self.solution_space[1]

        nullspace = [V_nullspace, p_nullspace]
        nullspace += self.solution_space[2:]

        return fd.MixedVectorSpaceBasis(self.solution_space, nullspace)

    def solve(self) -> None:
        """Solves the current system."""
        if hasattr(self, "irksome_integrator"):
            self.irksome_integrator.advance()
        else:
            self.solver.solve()
            self.solution_old.assign(self.solution)


class EnergySolver:
    """Solver for the energy equation."""

    def __init__(
        self,
        solution: fd.Function,
        apx: Approximation,
        u: fd.Function,
        time: fd.Function,
        time_step: fd.Function,
        time_stepper: IrksomeIntegrator,
        viscosity: float | Operator | None = None,
        delta_rho: float | Operator = 0.0,
        strong_bcs: list[fd.DirichletBC] | None = None,
        solver_parameters: dict[str, str] = None,
        disable_shear_heating: bool = False,
    ) -> None:
        self.solution = solution
        self.solution_space = self.solution.function_space()
        self.mesh = self.solution_space.mesh()

        self.apx = apx
        self.u = u
        self.eta = viscosity or self.apx.ref_profiles["eta"]
        self.delta_rho = delta_rho
        self.disable_shear_heating = disable_shear_heating

        self.set_weak_form()
        self.set_solver(time, time_step, time_stepper, strong_bcs, solver_parameters)

    def set_weak_form(self) -> None:
        """Defines variables needed in the weak form."""
        self.test = fd.TestFunction(self.solution_space)
        self.n = fd.FacetNormal(self.mesh)
        self.dx = fd.dx(degree=5)
        self.ds = fd.ds(degree=5)

        for profile in ["alpha", "cp", "H", "p", "T"]:
            setattr(self, profile, self.apx.ref_profiles[profile])
        match self.apx.name:
            case "BA" | "EBA" | "TALA" | "ALA":
                self.rho = self.apx.ref_profiles["rho"]
            case "ICA" | "HCA" | "PDA":
                self.rho = self.apx.density(0.0, self.solution, self.delta_rho)
        self.k = self.apx.ref_profiles["k"] * fd.Identity(2)

        self.shear_stress = self.apx.shear_stress(self.u, self.eta)

    def set_solver(
        self,
        time: fd.Function,
        time_step: fd.Function,
        time_stepper: IrksomeIntegrator,
        strong_bcs: list[fd.DirichletBC] | None = None,
        solver_parameters: dict[str, str] = None,
    ):
        """Sets up the solver for the weak form."""
        solver_parameters = (
            solver_parameters
            or {"ksp_converged_reason": None} | direct_energy_solver_parameters
        )

        self.irksome_integrator = time_stepper(
            self.residual(self.solution),
            self.solution,
            time,
            time_step,
            strong_bcs=strong_bcs,
            solver_parameters=solver_parameters,
        )

    def residual(self, trial: fd.Function) -> fd.Form:
        """Residual form of the equation."""
        weak_form = (
            self.test * self.rho * self.cp * Dt(trial)
            - fd.div(self.test * self.rho * self.cp * self.u) * trial
            + fd.dot(fd.grad(self.test), fd.dot(self.k, fd.grad(self.T + trial)))
            - self.test * self.alpha * fd.dot(self.u, fd.grad(self.p)) * trial
            - self.test * self.rho * self.H
        ) * self.dx
        if not self.disable_shear_heating:
            weak_form -= (
                self.test
                * fd.inner(self.shear_stress, self.apx.strain_rate(self.u))
                * self.dx
            )
        weak_form += (
            self.test * self.rho * self.cp * fd.dot(self.u, self.n) * trial
            # - self.test * fd.dot(fd.dot(self.k, fd.grad(self.T + trial)), self.n)
        ) * self.ds

        return weak_form

    def solve(self) -> None:
        """Solves the current equation."""
        self.irksome_integrator.advance()


class AdvectionSolver:
    """Solver for an advection equation."""

    def __init__(
        self,
        solution: fd.Function,
        u: fd.Function,
        time: fd.Function,
        time_step: fd.Function,
        time_stepper: IrksomeIntegrator,
        bcs=None,
        solver_parameters: dict[str, str] = None,
    ) -> None:
        self.solution = solution

        self.solver = GenericTransportSolver(
            ["advection", "mass"],
            solution,
            time,
            time_step,
            time_stepper,
            eq_attrs={"u": u},
            bcs=bcs,
            solver_parameters=solver_parameters,
        )

    def solve(self) -> None:
        """Solves the current equation."""
        self.solver.solve()

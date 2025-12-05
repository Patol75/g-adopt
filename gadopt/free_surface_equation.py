r"""This module contains the free surface terms.

All terms implement the UFL residual as it would be on the RHS of the equation:

  dq/dt = \sum term

This sign-convention is for compatibility with Thetis's time integrators. In general,
however, we like to think about the terms as they are on the LHS. Therefore, in the
function below, we assemble in `F` as it would be on the LHS:

  dq/dt + F(q) = 0

and at the very end return `-F`.
"""

import firedrake as fd
from irksome import Dt

from .equations import Equation
from .utility import vertical_component


def surface_velocity_term(
    eq: Equation, trial: fd.Argument | fd.ufl.indexed.Indexed | fd.Function
) -> fd.Form:
    r"""Free Surface term: u \dot n"""
    return -eq.buoyancy_scale * eq.test * fd.dot(eq.u, eq.n) * eq.ds(eq.boundary_id)


def mass_term(
    eq: Equation, trial: fd.Argument | fd.ufl.indexed.Indexed | fd.Function
) -> fd.Form:
    r"""Mass term \int test * trial * ds for the free surface time discretisation.

    Args:
        eq:
          G-ADOPT Equation.
        trial:
          Firedrake trial function.

    Returns:
        The UFL form associated with the mass term of the equation.

    """
    n_up = vertical_component(eq.n)

    return eq.test * Dt(eq.buoyancy_scale * trial) * n_up * eq.ds(eq.boundary_id)


mass_term.required_attrs = set()
mass_term.optional_attrs = set()
surface_velocity_term.required_attrs = {"u", "buoyancy_scale", "boundary_id"}
surface_velocity_term.optional_attrs = set()

free_surface_terms = [mass_term, surface_velocity_term]

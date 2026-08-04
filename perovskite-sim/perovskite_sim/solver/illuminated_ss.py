"""Fixed-duration illuminated preconditioner (legacy ``*_ss`` API name).

The helper integrates the full MOL system for ``t_settle`` seconds from dark
equilibrium.  Its result is a protocol-defined light-conditioned initial
state, useful for avoiding a dark-to-light startup artefact in J-V, impedance,
and degradation experiments.  Solver success certifies completion of that
finite-time integration, not convergence to a mathematical steady state.

Callers that require a residual-certified carrier steady state must use
``experiments.steady_state.solve_steady_state``.  Ionic equilibrium is a
separate, typically much longer, pre-bias/dwell protocol.
"""
from __future__ import annotations

import numpy as np

from perovskite_sim.models.device import DeviceStack
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import (
    MaterialArrays,
    StateVec,
    build_material_arrays,
    run_transient,
)


_NUMERIC_SOLVER_FAILURES = (
    RuntimeError,
    ValueError,
    FloatingPointError,
    OverflowError,
    RuntimeWarning,
    np.linalg.LinAlgError,
)


class IlluminatedSteadyStateError(RuntimeError):
    """The legacy-named illuminated preconditioning protocol failed."""

    def __init__(
        self,
        *,
        V_app: float,
        t_settle: float,
        solver_message: object | None = None,
        reason_code: str = "solver_failed",
    ) -> None:
        self.V_app = float(V_app)
        self.t_settle = float(t_settle)
        self.reason_code = reason_code
        self.solver_message = (
            str(solver_message) if solver_message not in (None, "") else None
        )
        detail = (
            f" Solver message: {self.solver_message}"
            if self.solver_message is not None
            else ""
        )
        super().__init__(
            "Illuminated preconditioning/settling failed at "
            f"V_app={self.V_app:.6g} V after {self.t_settle:.6g} s."
            f"{detail}"
        )


def solve_illuminated_ss(
    x: np.ndarray,
    stack: DeviceStack,
    V_app: float = 0.0,
    t_settle: float = 1e-3,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    mat: MaterialArrays | None = None,
) -> np.ndarray:
    """Return the state after a finite illuminated preconditioning interval.

    Integrates the MOL system for t_settle seconds under illumination,
    starting from dark equilibrium.  ``t_settle`` defines the physical
    protocol; this helper does not infer or certify a steady-state residual.

    Parameters
    ----------
    x        : grid node positions [m]
    stack    : device stack
    V_app    : applied voltage [V] (0 = short circuit)
    t_settle : settling time [s]; default 1e-3 s
    rtol, atol : ODE solver tolerances
    mat : optional pre-built material cache forwarded to the MOL solver

    Returns
    -------
    y : packed density state. Shape is ``3N`` for one ion species, ``4N``
        for two, plus ``4K`` when K interface-plane state blocks are active.

    Raises
    ------
    IlluminatedSteadyStateError
        If the transient solve fails or returns a malformed/nonphysical
        terminal density state. Dark equilibrium is never substituted.

    Notes
    -----
    The function name is retained for API compatibility. A successful return
    is a certified finite-time preconditioned state, not by itself proof of a
    true electronic or ionic steady state.
    """
    if t_settle <= 0.0:
        raise ValueError(f"t_settle must be positive, got {t_settle}")
    mat_eff = build_material_arrays(x, stack) if mat is None else mat
    y_dark = solve_equilibrium(x, stack)
    try:
        sol = run_transient(
            x, y_dark,
            (0.0, t_settle), np.array([t_settle]),
            stack, illuminated=True, V_app=V_app,
            rtol=rtol, atol=atol, mat=mat_eff,
        )
    except _NUMERIC_SOLVER_FAILURES as exc:
        raise IlluminatedSteadyStateError(
            V_app=V_app,
            t_settle=t_settle,
            solver_message=f"{type(exc).__name__}: {exc}",
            reason_code="solver_exception",
        ) from exc
    if not sol.success:
        raise IlluminatedSteadyStateError(
            V_app=V_app,
            t_settle=t_settle,
            solver_message=getattr(sol, "message", None),
            reason_code="solver_failed",
        )
    try:
        states = np.asarray(sol.y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise IlluminatedSteadyStateError(
            V_app=V_app,
            t_settle=t_settle,
            solver_message=f"malformed solver state: {type(exc).__name__}: {exc}",
            reason_code="invalid_terminal_state",
        ) from exc
    if (
        states.ndim != 2
        or states.shape[0] != y_dark.size
        or states.shape[1] == 0
        or not np.all(np.isfinite(states[:, -1]))
    ):
        raise IlluminatedSteadyStateError(
            V_app=V_app,
            t_settle=t_settle,
            solver_message=(
                "solver reported success but returned no finite terminal state"
            ),
            reason_code="invalid_terminal_state",
        )
    terminal = states[:, -1]
    density_atol = max(float(atol), 0.0)
    if np.any(terminal < -density_atol):
        raise IlluminatedSteadyStateError(
            V_app=V_app,
            t_settle=t_settle,
            solver_message=(
                "solver returned a negative density beyond its absolute "
                f"tolerance ({density_atol:.3g} m^-3)"
            ),
            reason_code="unphysical_terminal_state",
        )
    # Radau can land infinitesimally below zero in nominally empty transport
    # layers. Values within its declared absolute tolerance represent numeric
    # zero. Retain the solver state: even a sub-atol projection can perturb the
    # basin selected by the following stiff warm-started transient.

    N = len(x)
    expected_size = (
        (4 if mat_eff.has_dual_ions else 3) * N
        + 4 * mat_eff.N_iface_state
    )
    if terminal.size != expected_size:
        raise IlluminatedSteadyStateError(
            V_app=V_app,
            t_settle=t_settle,
            solver_message=(
                f"terminal state size {terminal.size} != model size {expected_size}"
            ),
            reason_code="invalid_terminal_state",
        )
    sv = StateVec.unpack(terminal, N, mat_eff.N_iface_state)

    def _exceeds(values: np.ndarray, limits: np.ndarray) -> bool:
        limits_arr = np.asarray(limits, dtype=float)
        finite_limit = np.isfinite(limits_arr)
        allowance = 1.0e-8 * np.maximum(np.abs(limits_arr), 1.0)
        return bool(np.any(finite_limit & (values > limits_arr + allowance)))

    if mat_eff.has_dual_ions and sv.P_neg is not None:
        if mat_eff.ion_steric_shared_site:
            ion_invalid = _exceeds(
                sv.P + sv.P_neg,
                np.minimum(mat_eff.P_lim_node, mat_eff.P_lim_neg_node),
            )
        else:
            ion_invalid = _exceeds(sv.P, mat_eff.P_lim_node) or _exceeds(
                sv.P_neg, mat_eff.P_lim_neg_node,
            )
    else:
        ion_invalid = _exceeds(sv.P, mat_eff.P_lim_node)
    if ion_invalid:
        raise IlluminatedSteadyStateError(
            V_app=V_app,
            t_settle=t_settle,
            solver_message="mobile-ion density exceeded its site-occupancy limit",
            reason_code="unphysical_terminal_state",
        )
    return terminal

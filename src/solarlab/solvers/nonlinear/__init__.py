"""Layout-independent DC nonlinear solver layer (P04-04 DC part).

The namespace exports the M-class (legacy bit-identical) criteria, loops and
preparation. The unified N-class solver is imported from its modules:
:mod:`~solarlab.solvers.nonlinear.problem`, :mod:`~solarlab.solvers.nonlinear.newton`,
:mod:`~solarlab.solvers.nonlinear.continuation`, :mod:`~solarlab.solvers.nonlinear.ptc`
and :mod:`~solarlab.solvers.nonlinear.constraints`; the P04-B switch points are
:mod:`~solarlab.solvers.nonlinear.production`.
"""

from solarlab.solvers.nonlinear.legacy_criteria import (
    CURRENT_BOUNDED_STALL, LEGACY_ACCEPTANCE_LABELS, RESIDUAL_CONVERGED,
    LegacySteadyStateCriteriaError, LegacySteadyStateTolerances,
    legacy_continuity_current_bound, legacy_normalized_poisson_residual,
    legacy_steady_state_tolerances,
)
from solarlab.solvers.nonlinear.legacy_newton import (
    LEGACY_FD_EPS, LEGACY_LN_STEP_CAP, LegacyNewtonStageError, legacy_damped_fd_newton,
    legacy_newton_stage,
)
from solarlab.solvers.nonlinear.legacy_preparation import (
    LEGACY_DEFAULT_ILLUMINATION_STAGES, LEGACY_INTERFACE_FALLBACK_ILLUMINATION_STAGES,
    legacy_illumination_stages, legacy_quasi_neutral_seed,
)

__all__ = [
    "CURRENT_BOUNDED_STALL", "LEGACY_ACCEPTANCE_LABELS", "LEGACY_DEFAULT_ILLUMINATION_STAGES",
    "LEGACY_FD_EPS", "LEGACY_INTERFACE_FALLBACK_ILLUMINATION_STAGES", "LEGACY_LN_STEP_CAP",
    "LegacyNewtonStageError", "LegacySteadyStateCriteriaError", "LegacySteadyStateTolerances",
    "RESIDUAL_CONVERGED", "legacy_continuity_current_bound", "legacy_damped_fd_newton",
    "legacy_illumination_stages", "legacy_newton_stage", "legacy_normalized_poisson_residual",
    "legacy_quasi_neutral_seed", "legacy_steady_state_tolerances",
]

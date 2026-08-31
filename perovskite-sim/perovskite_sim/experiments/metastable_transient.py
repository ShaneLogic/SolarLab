"""D7-E4 fully dynamic metastable configuration transient.

D7-E3 prepared a configuration split and froze it. This driver lets the split
evolve while the device is held at a measurement working point, which is the
SCAPS-like light-soak / dark-relaxation experiment.

Scheme, stated plainly
----------------------
This is an **operator-split** (sequential) integration, not a fully implicit
coupled DAE:

1. solve the guarded QF/DC state at the current frozen configuration, and
   require its ordinary physical certificate;
2. advance the configuration analytically over ``dt`` at those carriers, using
   the exact two-state solution rather than a discretisation of it;
3. repeat.

Every step therefore carries a certified carrier state, and the *only*
discretisation error is the splitting error between the two operators. That
error is not assumed small: :func:`run_metastable_configuration_transient`
reports it by re-running the same trace at halved ``dt`` and comparing, and
the certificate gates that comparison. A fully implicit coupled solve is a
separate, larger piece of work and is not claimed here.

Limits the certificate checks
-----------------------------
``slow``
    when every ``dt`` is far below the local relaxation time the trace must
    barely move from the prepared configuration;
``fast``
    when every ``dt`` is far above it the trace must land on the stationary
    configuration of the *measurement* working point.

Charge bookkeeping
------------------
Each converted defect moves two elementary charges. The driver separates the
configuration contribution to the defect charge from the ordinary
charge-state response by evaluating the frozen inventory twice at the same
carriers — once at the old fraction and once at the new one — and requires
that difference to match the analytic configuration transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from perovskite_sim.experiments.metastable_preparation import (
    FrozenMetastableConfiguration,
    MetastablePreparationError,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    QuasiFermiSteadyStateResult,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.metastable_defect_device import (
    FrozenMetastableBulkDefectModel,
    evaluate_frozen_metastable_bulk_defects,
)
from perovskite_sim.physics.metastable_dynamic_state import (
    advance_metastable_configuration,
    configuration_from_logit,
    configuration_logit,
)
from perovskite_sim.solver.mol import build_material_arrays


METASTABLE_TRANSIENT_VERSION = "metastable-operator-split-transient-v1"


class MetastableTransientError(RuntimeError):
    """The dynamic metastable trace failed a physical or numerical gate."""


@dataclass(frozen=True)
class MetastableTransientCertificate:
    """Gate values for one dynamic metastable trace."""

    scheme: str
    certified: bool
    reasons: tuple[str, ...]
    step_count: int
    minimum_fraction: float
    maximum_fraction: float
    clipping_used: bool
    maximum_state_residual: float
    maximum_charge_transfer_relative_error: float
    maximum_refinement_fraction_change: float
    maximum_step_over_relaxation_time: float
    minimum_relaxation_time_s: float


@dataclass(frozen=True)
class MetastableTransientResult:
    """A dynamic configuration trace with its per-step certified states."""

    times_s: np.ndarray
    donor_fraction: np.ndarray
    stationary_fraction: np.ndarray
    relaxation_time_s: np.ndarray
    configuration_charge_transfer_C_m2: np.ndarray
    states: tuple[QuasiFermiSteadyStateResult, ...]
    certificate: MetastableTransientCertificate
    preparation_protocol_sha256: str
    preparation_state_sha256: str


def _validate_times(times_s: object) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or times[0] != 0.0
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("times_s must start at 0 and strictly increase")
    return times


def _model_at(
    frozen: FrozenMetastableConfiguration,
    fraction: np.ndarray,
) -> FrozenMetastableBulkDefectModel:
    region = frozen.model.regions[0]
    return FrozenMetastableBulkDefectModel(
        regions=(
            replace(
                region,
                donor_fraction=np.asarray(fraction, dtype=float),
            ),
        )
    )


def _trace(
    grid: np.ndarray,
    stack: DeviceStack,
    frozen: FrozenMetastableConfiguration,
    times: np.ndarray,
    *,
    V_app: float,
    illuminated: bool,
    keep_states: bool,
    solver_controls: dict,
) -> tuple[np.ndarray, list, list, list, list, list, list]:
    region = frozen.model.regions[0]
    mask = np.asarray(region.active_nodes, dtype=bool)
    base_material = build_material_arrays(grid, stack)
    fraction = np.asarray(frozen.donor_fraction, dtype=float).copy()
    # Round-trip the logit coordinate so an out-of-range prepared fraction is
    # rejected here rather than silently integrated.
    fraction = configuration_from_logit(configuration_logit(fraction))

    fractions = [fraction.copy()]
    stationaries: list[np.ndarray] = []
    relaxations: list[np.ndarray] = []
    transfers: list[float] = []
    areal_transfers: list[float] = []
    states: list[QuasiFermiSteadyStateResult] = []
    step_ratios: list[float] = []
    count = grid.size
    for index in range(times.size):
        material = replace(
            base_material,
            frozen_metastable_defects=_model_at(frozen, fraction),
        )
        try:
            state = solve_quasi_fermi_steady_state(
                grid,
                stack,
                V_app=float(V_app),
                illuminated=illuminated,
                mat=material,
                **solver_controls,
            )
        except QuasiFermiSteadyStateError as exc:
            raise MetastableTransientError(
                f"dynamic metastable step {index} did not certify: {exc}"
            ) from exc
        if keep_states:
            states.append(state)
        if index == times.size - 1:
            break
        dt = float(times[index + 1] - times[index])
        electrons = np.asarray(state.y[:count], dtype=float)[mask]
        holes = np.asarray(state.y[count : 2 * count], dtype=float)[mask]
        step = advance_metastable_configuration(
            fraction,
            electrons,
            holes,
            region.definition,
            dt,
            band_gap_eV=region.band_gap_eV,
            effective_conduction_dos_m3=region.effective_conduction_dos_m3,
            effective_valence_dos_m3=region.effective_valence_dos_m3,
            # The conversion happens at the MEASUREMENT temperature; the
            # region's band-edge metadata is the prepared one, which is only
            # equivalent while temperature scaling is off (see the guard in
            # run_metastable_configuration_transient).
            temperature_K=float(stack.T),
        )
        # Charge bookkeeping: the configuration part of the defect charge is
        # the difference between the frozen inventory evaluated at the new and
        # old fractions with the carriers held fixed.
        before = evaluate_frozen_metastable_bulk_defects(
            np.asarray(state.y[:count], dtype=float),
            np.asarray(state.y[count : 2 * count], dtype=float),
            _model_at(frozen, fraction),
        )
        after = evaluate_frozen_metastable_bulk_defects(
            np.asarray(state.y[:count], dtype=float),
            np.asarray(state.y[count : 2 * count], dtype=float),
            _model_at(frozen, step.donor_fraction),
        )
        measured = (after.total_charge_density_C_m3 - before.total_charge_density_C_m3)[
            mask
        ]
        predicted = np.asarray(step.configuration_charge_transfer_C_m3, dtype=float)
        # The identity is exact analytically, so the achievable accuracy of
        # this check is set by the cancellation in `measured`: it differences
        # two defect charges of order |rho_def| to resolve a transfer that can
        # be twenty orders smaller. Scale the residual by the quantity being
        # differenced, not by the transfer, or the gate would be measuring
        # floating-point spacing rather than the physics.
        charge_scale = max(
            float(np.max(np.abs(before.total_charge_density_C_m3[mask]))),
            np.finfo(float).tiny,
        )
        transfers.append(float(np.max(np.abs(measured - predicted)) / charge_scale))
        areal = np.zeros(count, dtype=float)
        areal[mask] = predicted
        areal_transfers.append(float(np.trapezoid(areal, grid)))
        stationaries.append(np.asarray(step.stationary_fraction, dtype=float))
        relaxations.append(np.asarray(step.relaxation_time_s, dtype=float))
        step_ratios.append(step.step_over_relaxation_time)
        fraction = np.asarray(step.donor_fraction, dtype=float).copy()
        fractions.append(fraction.copy())
    return (
        np.asarray(fractions, dtype=float),
        stationaries,
        relaxations,
        transfers,
        states,
        step_ratios,
        areal_transfers,
    )


def run_metastable_configuration_transient(
    x: np.ndarray,
    stack: DeviceStack,
    frozen: FrozenMetastableConfiguration,
    times_s: np.ndarray,
    *,
    V_app: float = 0.0,
    illuminated: bool = False,
    maximum_charge_transfer_relative_error: float = 1.0e-10,
    maximum_refinement_fraction_change: float = 5.0e-3,
    require_certificate: bool = True,
    **solver_controls,
) -> MetastableTransientResult:
    """Integrate the configuration split at a fixed measurement working point.

    The trace is re-run at halved ``dt`` and the two are compared, so the
    operator-splitting error is measured rather than assumed.
    """

    if not isinstance(frozen, FrozenMetastableConfiguration):
        raise TypeError("frozen must be a FrozenMetastableConfiguration")
    if len(frozen.model.regions) != 1:
        raise MetastableTransientError(
            "D7-E4 integrates exactly one prepared metastable region"
        )
    grid = np.asarray(x, dtype=float)
    if not np.array_equal(grid, frozen.grid_m):
        raise MetastablePreparationError(
            "transient grid does not match the prepared grid"
        )
    times = _validate_times(times_s)
    region = frozen.model.regions[0]
    if float(stack.T) != region.temperature_K and stack.mode != "legacy":
        # Outside LEGACY the band edges and DOS are temperature scaled, so the
        # prepared region metadata would no longer describe the measurement
        # device. Refuse rather than silently mixing two temperatures.
        raise MetastableTransientError(
            "a measurement temperature different from the preparation "
            "temperature requires the LEGACY tier, where band edges and DOS "
            "are not temperature scaled"
        )
    for name, value in (
        (
            "maximum_charge_transfer_relative_error",
            maximum_charge_transfer_relative_error,
        ),
        ("maximum_refinement_fraction_change", maximum_refinement_fraction_change),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    (
        fractions,
        stationaries,
        relaxations,
        transfers,
        states,
        ratios,
        areal_transfers,
    ) = _trace(
        grid,
        stack,
        frozen,
        times,
        V_app=V_app,
        illuminated=illuminated,
        keep_states=True,
        solver_controls=solver_controls,
    )

    # Refinement: the same physical trace on a doubled time grid. Only the
    # splitting error can move it.
    refined_times = np.unique(np.concatenate([times, 0.5 * (times[:-1] + times[1:])]))
    refined_fractions, *_rest = _trace(
        grid,
        stack,
        frozen,
        refined_times,
        V_app=V_app,
        illuminated=illuminated,
        keep_states=False,
        solver_controls=solver_controls,
    )
    sampled = np.searchsorted(refined_times, times)
    refinement_change = float(np.max(np.abs(refined_fractions[sampled] - fractions)))

    reasons: list[str] = []
    minimum_fraction = float(np.min(fractions))
    maximum_fraction = float(np.max(fractions))
    charge_error = max(transfers) if transfers else 0.0
    if minimum_fraction <= 0.0 or maximum_fraction >= 1.0:
        reasons.append("configuration_fraction_left_open_interval")
    if charge_error > maximum_charge_transfer_relative_error:
        reasons.append("configuration_charge_transfer_mismatch")
    if refinement_change > maximum_refinement_fraction_change:
        reasons.append("operator_splitting_not_resolved")
    if not all(state.certified for state in states):
        reasons.append("uncertified_carrier_state")

    certificate = MetastableTransientCertificate(
        scheme=METASTABLE_TRANSIENT_VERSION,
        certified=not reasons,
        reasons=tuple(reasons),
        step_count=int(times.size),
        minimum_fraction=minimum_fraction,
        maximum_fraction=maximum_fraction,
        clipping_used=False,
        maximum_state_residual=max(
            state.max_normalized_cell_residual for state in states
        ),
        maximum_charge_transfer_relative_error=charge_error,
        maximum_refinement_fraction_change=refinement_change,
        maximum_step_over_relaxation_time=(max(ratios) if ratios else 0.0),
        minimum_relaxation_time_s=(
            min(float(np.min(value)) for value in relaxations)
            if relaxations
            else math.inf
        ),
    )
    if require_certificate and not certificate.certified:
        raise MetastableTransientError(
            "dynamic metastable trace failed its certificate: "
            + ", ".join(certificate.reasons)
        )

    return MetastableTransientResult(
        times_s=np.array(times, copy=True),
        donor_fraction=fractions,
        stationary_fraction=np.asarray(stationaries, dtype=float),
        relaxation_time_s=np.asarray(relaxations, dtype=float),
        configuration_charge_transfer_C_m2=np.asarray(areal_transfers, dtype=float),
        states=tuple(states),
        certificate=certificate,
        preparation_protocol_sha256=frozen.protocol_sha256,
        preparation_state_sha256=frozen.state_sha256,
    )


__all__ = [
    "METASTABLE_TRANSIENT_VERSION",
    "MetastableTransientCertificate",
    "MetastableTransientError",
    "MetastableTransientResult",
    "run_metastable_configuration_transient",
]

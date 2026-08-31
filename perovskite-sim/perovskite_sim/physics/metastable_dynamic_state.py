"""Bounded dynamic state for metastable configuration conversion.

D7-E3 froze the configuration split after a preparation solve. D7-E4 lets it
evolve. The dynamic unknown is the donor-configuration fraction ``y`` on each
prepared node, obeying the two-state master equation

    dy/dt = k_AD(n, p) * (1 - y) - k_DA(n, p) * y,

whose stationary point is exactly the D7-E3 closure fraction
``y* = k_AD / (k_DA + k_AD)``.

Bounded without clipping
------------------------
The coordinate carried by callers is the logit ``u = log(y / (1 - y))``, and
the update is the *analytic* solution of the master equation over a step at
fixed rates,

    y(t + dt) = y* + (y(t) - y*) * exp(-(k_DA + k_AD) dt).

Because ``0 < y* < 1`` and the exponential factor lies in ``(0, 1]``, the
result is a convex combination of two interior points and can never leave
``(0, 1)``. Boundedness is therefore structural, not enforced by clamping —
the same standard the D6 dynamic-trap layout holds itself to. A step that
would still land on an endpoint (only reachable from a non-finite input)
raises rather than being clipped back into range.

Charge bookkeeping
------------------
Converting one defect from the donor to the acceptor configuration moves two
elementary charges, because the schema requires the two selected conversion
states to differ by exactly ``+2``. At fixed carriers the configuration part
of the defect charge is therefore linear in ``y``, which
:func:`configuration_charge_transfer_C_m3` returns exactly; the transient uses
it to separate the configuration contribution from the ordinary charge-state
response to the carriers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from perovskite_sim.models.multivalent_defects import MetastableDefectDefinition
from perovskite_sim.physics.metastable_defect_closure import (
    evaluate_metastable_configuration_closure,
)


METASTABLE_DYNAMIC_STATE_VERSION = "metastable-dynamic-configuration-v1"


class MetastableDynamicStateError(RuntimeError):
    """The dynamic configuration state left its admissible domain."""


def _readonly(value: object) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def configuration_logit(fraction: np.ndarray) -> np.ndarray:
    """Return ``log(y / (1 - y))`` on the open interval, or raise."""

    values = np.asarray(fraction, dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or np.any(values >= 1.0)
    ):
        raise MetastableDynamicStateError(
            "configuration fraction must lie strictly inside (0, 1) for a "
            "logit coordinate"
        )
    return np.log(values) - np.log1p(-values)


def configuration_from_logit(logit: np.ndarray) -> np.ndarray:
    """Invert :func:`configuration_logit` without saturating."""

    values = np.asarray(logit, dtype=float)
    if not np.all(np.isfinite(values)):
        raise MetastableDynamicStateError("configuration logit must be finite")
    positive = values >= 0.0
    fraction = np.empty_like(values)
    fraction[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    fraction[~positive] = exponential / (1.0 + exponential)
    if np.any(fraction <= 0.0) or np.any(fraction >= 1.0):
        raise MetastableDynamicStateError(
            "logit transform saturated outside the resolvable configuration bounds"
        )
    return fraction


@dataclass(frozen=True, slots=True)
class MetastableDynamicStep:
    """One analytic configuration step plus its conservation bookkeeping."""

    donor_fraction: np.ndarray
    stationary_fraction: np.ndarray
    relaxation_rate_s1: np.ndarray
    relaxation_time_s: np.ndarray
    configuration_change: np.ndarray
    configuration_charge_transfer_C_m3: np.ndarray
    minimum_fraction: float
    maximum_fraction: float
    maximum_step_fraction_change: float
    minimum_relaxation_time_s: float
    step_over_relaxation_time: float


def configuration_charge_transfer_C_m3(
    definition: MetastableDefectDefinition,
    donor_fraction_change: np.ndarray,
    *,
    donor_state_probability: np.ndarray,
    acceptor_state_probability: np.ndarray,
) -> np.ndarray:
    """Charge moved by the configuration change alone, at fixed carriers.

    ``donor_state_probability`` / ``acceptor_state_probability`` are the two
    configurations' internal charge-state distributions at the current
    carriers, shaped ``(states, nodes)``. The configuration contribution to the
    defect charge is linear in ``y``, so its change is the density times the
    mean-charge difference times ``dy``.
    """

    from perovskite_sim.constants import Q

    donor_charges = np.asarray(
        definition.donor_configuration.charge_states_e, dtype=float
    ).reshape(-1, 1)
    acceptor_charges = np.asarray(
        definition.acceptor_configuration.charge_states_e, dtype=float
    ).reshape(-1, 1)
    donor_mean = np.sum(donor_charges * np.asarray(donor_state_probability), axis=0)
    acceptor_mean = np.sum(
        acceptor_charges * np.asarray(acceptor_state_probability), axis=0
    )
    return (
        Q
        * definition.total_density_m3
        * (donor_mean - acceptor_mean)
        * np.asarray(donor_fraction_change, dtype=float)
    )


def advance_metastable_configuration(
    donor_fraction: np.ndarray,
    electron_density_m3: np.ndarray,
    hole_density_m3: np.ndarray,
    definition: MetastableDefectDefinition,
    dt_s: float,
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
) -> MetastableDynamicStep:
    """Advance the configuration fraction analytically over one step.

    The carriers are held at the values supplied for this step, so the update
    is the exact solution of the master equation rather than a discretisation
    of it. All of the step's error is therefore operator-splitting error
    between the configuration and the carriers, which the transient measures
    by refining ``dt``.
    """

    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    fraction = np.asarray(donor_fraction, dtype=float)
    if (
        not np.all(np.isfinite(fraction))
        or np.any(fraction <= 0.0)
        or np.any(fraction >= 1.0)
    ):
        raise MetastableDynamicStateError(
            "dynamic configuration fraction must stay strictly inside (0, 1)"
        )
    closure = evaluate_metastable_configuration_closure(
        electron_density_m3,
        hole_density_m3,
        definition,
        band_gap_eV=band_gap_eV,
        effective_conduction_dos_m3=effective_conduction_dos_m3,
        effective_valence_dos_m3=effective_valence_dos_m3,
        temperature_K=temperature_K,
    )
    stationary = np.asarray(closure.donor_fraction, dtype=float)
    rate = np.asarray(closure.donor_to_acceptor_rate_s1, dtype=float) + np.asarray(
        closure.acceptor_to_donor_rate_s1, dtype=float
    )
    if fraction.shape != stationary.shape:
        raise ValueError("configuration fraction must match the carrier shape")
    decay = np.exp(-rate * float(dt_s))
    updated = stationary + (fraction - stationary) * decay
    if (
        not np.all(np.isfinite(updated))
        or np.any(updated <= 0.0)
        or np.any(updated >= 1.0)
    ):
        raise MetastableDynamicStateError(
            "analytic configuration step left the open interval, which is "
            "only reachable from an inconsistent input"
        )
    change = updated - fraction
    relaxation_time = 1.0 / rate
    # The configuration contribution to the defect charge needs each
    # configuration's internal charge-state distribution at THESE carriers,
    # so both are evaluated here rather than approximated by the nominal
    # conversion-state charges.
    from perovskite_sim.physics.metastable_defect_device import (
        configuration_species,
    )
    from perovskite_sim.physics.multivalent_defect_closure import (
        evaluate_multivalent_defect_closure,
    )

    donor_species, acceptor_species = configuration_species(definition)
    configuration_closures = tuple(
        evaluate_multivalent_defect_closure(
            electron_density_m3,
            hole_density_m3,
            species,
            band_gap_eV=band_gap_eV,
            effective_conduction_dos_m3=effective_conduction_dos_m3,
            effective_valence_dos_m3=effective_valence_dos_m3,
            temperature_K=temperature_K,
        )
        for species in (donor_species, acceptor_species)
    )
    charge_transfer = configuration_charge_transfer_C_m3(
        definition,
        change,
        donor_state_probability=configuration_closures[0].state_probability,
        acceptor_state_probability=configuration_closures[1].state_probability,
    )
    return MetastableDynamicStep(
        donor_fraction=_readonly(updated),
        stationary_fraction=_readonly(stationary),
        relaxation_rate_s1=_readonly(rate),
        relaxation_time_s=_readonly(relaxation_time),
        configuration_change=_readonly(change),
        configuration_charge_transfer_C_m3=_readonly(charge_transfer),
        minimum_fraction=float(np.min(updated)),
        maximum_fraction=float(np.max(updated)),
        maximum_step_fraction_change=float(np.max(np.abs(change))),
        minimum_relaxation_time_s=float(np.min(relaxation_time)),
        step_over_relaxation_time=float(np.max(rate * float(dt_s))),
    )


__all__ = [
    "METASTABLE_DYNAMIC_STATE_VERSION",
    "MetastableDynamicStateError",
    "MetastableDynamicStep",
    "advance_metastable_configuration",
    "configuration_charge_transfer_C_m3",
    "configuration_from_logit",
    "configuration_logit",
]

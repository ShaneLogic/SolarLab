"""Contact thermodynamic assessment with the fixed 5 meV gate (M class).

A zero-bias equilibrium requires the electron and hole reservoirs of both
contacts to share one Fermi level. From the reservoir densities, the physical
(unfolded) band edges and the Poisson drop at each contact node, the four
contact quasi-Fermi levels are

    E_Fn = E_C + V_T eta_n,    E_Fp = E_V - V_T eta_p,    E_C = -phi - chi,

with ``eta`` from the shared statistics definition. The status is
``certified`` only when their span (and any declared metal work function) is
within ``CONTACT_THERMODYNAMIC_TOLERANCE_EV = 5 meV``; a caller may tighten
that gate but never relax it. Without endpoint DOS a band-derived potential
mismatch can still disprove compatibility, but agreement is only
``compatible_unverified``. Values and statuses are those of the legacy engine.

``historical_label`` names a known historical inconsistency without changing
the status: a ``legacy_manual`` contact potential whose two reservoirs are each
self-consistent is labelled ``legacy_manual_v_bi_contact_offset`` -- the span is
then exactly the offset between the configured ``V_bi`` and the reservoirs
(6.025 meV on the historical scaps mirror), and it stays ``inconsistent``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from solarlab.physics.contacts.potential import LEGACY_MANUAL, METAL_WORK_FUNCTION
from solarlab.physics.statistics.carriers import (
    MAXWELL_BOLTZMANN, CarrierStatistics, reduced_fermi_level_from_density,
)

__all__ = [
    "CONTACT_THERMODYNAMIC_TOLERANCE_EV", "ContactEndpoint", "ContactThermodynamicAssessment",
    "ContactThermodynamicError", "LEGACY_MANUAL_CONTACT_OFFSET",
    "assess_contact_thermodynamics", "contact_quasi_fermi_levels",
    "require_contact_thermodynamic_certificate",
]

CONTACT_THERMODYNAMIC_TOLERANCE_EV = 5.0e-3
LEGACY_MANUAL_CONTACT_OFFSET = "legacy_manual_v_bi_contact_offset"
AssessmentStatus = Literal["certified", "inconsistent", "compatible_unverified", "not_assessable"]


class ContactThermodynamicError(ValueError):
    """The selected Poisson/contact-reservoir pair lacks a certificate."""


@dataclass(frozen=True, slots=True)
class ContactEndpoint:
    """Physical band data and reservoir densities at one contact node."""

    electron_density_m3: float
    hole_density_m3: float
    conduction_dos_m3: float | None = None
    valence_dos_m3: float | None = None
    electron_affinity_eV: float | None = None
    band_gap_eV: float | None = None

    def assessable(self) -> bool:
        values = (self.conduction_dos_m3, self.valence_dos_m3,
                  self.electron_affinity_eV, self.band_gap_eV)
        if any(value is None for value in values):
            return False
        return bool(
            all(np.isfinite(value) for value in values)  # type: ignore[arg-type]
            and self.conduction_dos_m3 > 0.0  # type: ignore[operator]
            and self.valence_dos_m3 > 0.0  # type: ignore[operator]
            and self.electron_density_m3 > 0.0
            and self.hole_density_m3 > 0.0
        )


@dataclass(frozen=True)
class ContactThermodynamicAssessment:
    """Internal compatibility evidence for the Poisson boundary and reservoirs."""

    status: AssessmentStatus
    built_in_potential_mode: str
    tolerance_eV: float
    fermi_level_span_eV: float | None
    potential_mismatch_V: float | None
    metal_work_function_mismatch_eV: float | None
    contact_quasi_fermi_levels_eV: tuple[float, ...]
    message: str
    historical_label: str | None = None

    @property
    def certified(self) -> bool:
        return self.status == "certified"


def contact_quasi_fermi_levels(
    left: ContactEndpoint,
    right: ContactEndpoint,
    *,
    V_bi_bc: float,
    thermal_voltage_V: float,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Contact QFLs ``(E_Fn,L, E_Fp,L, E_Fn,R, E_Fp,R)`` and semiconductor work functions.

    Endpoints that lack positive DOS or reservoir densities contribute nothing.
    """
    levels: list[float] = []
    work_functions: list[float] = []
    for endpoint, phi in ((left, 0.0), (right, float(V_bi_bc))):
        if not endpoint.assessable():
            continue
        chi = endpoint.electron_affinity_eV
        gap = endpoint.band_gap_eV
        assert chi is not None and gap is not None
        conduction_edge = -phi - chi
        valence_edge = conduction_edge - gap
        eta_n = reduced_fermi_level_from_density(
            endpoint.electron_density_m3, endpoint.conduction_dos_m3,  # type: ignore[arg-type]
            statistics=statistics,
        )
        eta_p = reduced_fermi_level_from_density(
            endpoint.hole_density_m3, endpoint.valence_dos_m3,  # type: ignore[arg-type]
            statistics=statistics,
        )
        levels.extend((
            float(conduction_edge + thermal_voltage_V * eta_n),
            float(valence_edge - thermal_voltage_V * eta_p),
        ))
        work_functions.extend((
            float(chi - thermal_voltage_V * eta_n),
            float(chi + gap + thermal_voltage_V * eta_p),
        ))
    return tuple(levels), tuple(work_functions)


def _validated_tolerance(tolerance_eV: float) -> float:
    if not np.isfinite(tolerance_eV) or tolerance_eV <= 0.0:
        raise ValueError("tolerance_eV must be finite and positive")
    if tolerance_eV > CONTACT_THERMODYNAMIC_TOLERANCE_EV:
        raise ValueError(
            "tolerance_eV cannot exceed the fixed internal certificate gate "
            f"of {CONTACT_THERMODYNAMIC_TOLERANCE_EV:g} eV"
        )
    return float(tolerance_eV)


def assess_contact_thermodynamics(
    *,
    built_in_potential_mode: str,
    left: ContactEndpoint,
    right: ContactEndpoint,
    V_bi_bc: float,
    thermal_voltage_V: float,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
    semiconductor_potential_V: float | None = None,
    metal_work_functions_eV: tuple[float, float] | None = None,
    has_band_reference: bool = False,
    tolerance_eV: float = CONTACT_THERMODYNAMIC_TOLERANCE_EV,
) -> ContactThermodynamicAssessment:
    """Assess whether all four statistics-consistent contact QFLs agree.

    ``built_in_potential_mode`` is the resolved public mode;
    ``semiconductor_potential_V`` is the strict semiconductor work-function
    potential when it is computable (else ``None``); ``metal_work_functions_eV``
    are the declared ``(W_left, W_right)`` of the metal mode.
    """
    tolerance = _validated_tolerance(tolerance_eV)
    mode = built_in_potential_mode
    potential_mismatch: float | None = None
    if semiconductor_potential_V is not None:
        potential_mismatch = float(V_bi_bc - semiconductor_potential_V)
        if not np.isfinite(potential_mismatch):
            potential_mismatch = None

    levels, work_functions = contact_quasi_fermi_levels(
        left, right, V_bi_bc=V_bi_bc, thermal_voltage_V=thermal_voltage_V,
        statistics=statistics,
    )
    if len(levels) == 4:
        span = float(max(levels) - min(levels))
        metal_mismatch: float | None = None
        if mode == METAL_WORK_FUNCTION:
            if metal_work_functions_eV is None:
                raise ValueError("metal_work_function requires both contact work functions")
            W_left, W_right = (float(value) for value in metal_work_functions_eV)
            metal_mismatch = float(np.max(np.abs(
                np.asarray((W_left, W_left, W_right, W_right))
                - np.asarray(work_functions)
            )))
        qfl_compatible = span <= tolerance
        metal_compatible = metal_mismatch is None or metal_mismatch <= tolerance
        status: AssessmentStatus = (
            "certified" if qfl_compatible and metal_compatible else "inconsistent"
        )
        if status == "certified":
            message = (
                "carrier reservoirs, Poisson drop, and any explicit metal "
                "work functions agree within the fixed internal "
                f"{tolerance:g} eV gate"
            )
        elif not qfl_compatible:
            message = (
                "Poisson contact potential and carrier reservoirs impose a "
                f"{span:.6g} eV equilibrium quasi-Fermi-level span"
            )
        else:
            message = (
                "explicit metal work functions differ from the modeled "
                "semiconductor reservoirs by as much as "
                f"{metal_mismatch:.6g} eV"
            )
        label = None
        if (status == "inconsistent" and not qfl_compatible and mode == LEGACY_MANUAL
                and abs(levels[0] - levels[1]) <= tolerance
                and abs(levels[2] - levels[3]) <= tolerance):
            label = LEGACY_MANUAL_CONTACT_OFFSET
        return ContactThermodynamicAssessment(
            status=status,
            built_in_potential_mode=mode,
            tolerance_eV=tolerance,
            fermi_level_span_eV=span,
            potential_mismatch_V=potential_mismatch,
            metal_work_function_mismatch_eV=metal_mismatch,
            contact_quasi_fermi_levels_eV=levels,
            message=message,
            historical_label=label,
        )

    if potential_mismatch is not None and abs(potential_mismatch) > tolerance:
        status = "inconsistent"
        message = (
            "endpoint DOS data are incomplete, but the imposed Poisson drop "
            "already differs from the band-derived contact potential by "
            f"{abs(potential_mismatch):.6g} V"
        )
    elif has_band_reference:
        status = "compatible_unverified"
        message = (
            "the imposed and band-derived contact potentials agree, but "
            "endpoint effective-DOS data are missing, so the four contact "
            "quasi-Fermi levels cannot be certified"
        )
    else:
        status = "not_assessable"
        message = (
            "the stack lacks endpoint band/DOS data required for a "
            "thermodynamic contact assessment"
        )
    return ContactThermodynamicAssessment(
        status=status,
        built_in_potential_mode=mode,
        tolerance_eV=tolerance,
        fermi_level_span_eV=None,
        potential_mismatch_V=potential_mismatch,
        metal_work_function_mismatch_eV=None,
        contact_quasi_fermi_levels_eV=(),
        message=message,
    )


def require_contact_thermodynamic_certificate(
    assessment: ContactThermodynamicAssessment,
) -> ContactThermodynamicAssessment:
    """Return a certified assessment or fail closed, keeping any label in the error."""
    if not assessment.certified:
        label = (
            f" [historical label: {assessment.historical_label}]"
            if assessment.historical_label else ""
        )
        raise ContactThermodynamicError(
            f"contact thermodynamic status={assessment.status}: "
            f"{assessment.message}{label}"
        )
    return assessment

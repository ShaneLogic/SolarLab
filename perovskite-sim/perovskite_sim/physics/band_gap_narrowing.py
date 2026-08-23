"""Opt-in empirical band-gap-narrowing constitutive laws."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


BAND_GAP_NARROWING_OFF = "off"
SLOTBOOM = "slotboom"
BandGapNarrowingModel = Literal["off", "slotboom"]


@dataclass(frozen=True, slots=True)
class BandGapNarrowingState:
    """One static doping-dependent band-edge transformation."""

    model: BandGapNarrowingModel
    total_dopant_density_m3: float
    narrowing_eV: float
    conduction_band_fraction: float
    conduction_band_shift_eV: float
    valence_band_shift_eV: float
    effective_electron_affinity_eV: float
    effective_band_gap_eV: float


def normalize_band_gap_narrowing_model(value: object) -> BandGapNarrowingModel:
    """Return one supported BGN identifier or fail closed."""
    if not isinstance(value, str):
        raise ValueError("band-gap narrowing model must be a string")
    normalized = value.strip().lower()
    if normalized not in {BAND_GAP_NARROWING_OFF, SLOTBOOM}:
        raise ValueError("band-gap narrowing model must be 'off' or 'slotboom'")
    return normalized  # type: ignore[return-value]


def _finite_positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite_nonnegative(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def slotboom_band_gap_narrowing(
    total_dopant_density_m3: float,
    *,
    reference_energy_eV: float = 0.009,
    reference_density_m3: float = 1.0e23,
    log_shape: float = 0.5,
) -> float:
    """Return Slotboom empirical ``Delta E_g`` in eV.

    The negative-log branch uses the conjugate expression to avoid losing the
    small positive narrowing when ``N_I << N_ref``.
    """
    density = _finite_nonnegative(
        total_dopant_density_m3,
        "total_dopant_density_m3",
    )
    energy = _finite_positive(reference_energy_eV, "reference_energy_eV")
    reference = _finite_positive(reference_density_m3, "reference_density_m3")
    shape = _finite_positive(log_shape, "log_shape")
    if density == 0.0:
        return 0.0
    # Subtracting logarithms avoids underflow in density / reference for
    # finite, extremely small opt-in research densities.
    logarithm = math.log(density) - math.log(reference)
    root = math.hypot(logarithm, math.sqrt(shape))
    slotboom_factor = (
        logarithm + root if logarithm >= 0.0 else shape / (root - logarithm)
    )
    narrowing = energy * slotboom_factor
    if not math.isfinite(narrowing) or narrowing < 0.0:
        raise FloatingPointError("Slotboom band-gap narrowing is non-finite")
    return narrowing


def apply_band_gap_narrowing(
    *,
    electron_affinity_eV: float,
    band_gap_eV: float,
    acceptor_density_m3: float,
    donor_density_m3: float,
    model: BandGapNarrowingModel | str = BAND_GAP_NARROWING_OFF,
    reference_energy_eV: float = 0.009,
    reference_density_m3: float = 1.0e23,
    log_shape: float = 0.5,
    conduction_band_fraction: float = 0.5,
) -> BandGapNarrowingState:
    """Apply one BGN law to the physical conduction and valence edges."""
    affinity = _finite_nonnegative(electron_affinity_eV, "electron_affinity_eV")
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    acceptors = _finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
    donors = _finite_nonnegative(donor_density_m3, "donor_density_m3")
    total_dopants = _finite_nonnegative(
        acceptors + donors,
        "total_dopant_density_m3",
    )
    narrowing_model = normalize_band_gap_narrowing_model(model)
    fraction = float(conduction_band_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("conduction_band_fraction must be finite and in [0, 1]")
    if narrowing_model == BAND_GAP_NARROWING_OFF:
        if (
            reference_energy_eV != 0.009
            or reference_density_m3 != 1.0e23
            or log_shape != 0.5
            or fraction != 0.5
        ):
            raise ValueError(
                "BGN parameters require band-gap narrowing model 'slotboom'"
            )
        narrowing = 0.0
    else:
        narrowing = slotboom_band_gap_narrowing(
            total_dopants,
            reference_energy_eV=reference_energy_eV,
            reference_density_m3=reference_density_m3,
            log_shape=log_shape,
        )
    if narrowing >= gap:
        raise ValueError("band-gap narrowing must remain smaller than the base gap")
    conduction_shift = fraction * narrowing
    valence_shift = (1.0 - fraction) * narrowing
    return BandGapNarrowingState(
        model=narrowing_model,
        total_dopant_density_m3=total_dopants,
        narrowing_eV=narrowing,
        conduction_band_fraction=fraction,
        conduction_band_shift_eV=conduction_shift,
        valence_band_shift_eV=valence_shift,
        effective_electron_affinity_eV=affinity + conduction_shift,
        effective_band_gap_eV=gap - narrowing,
    )


__all__ = [
    "BAND_GAP_NARROWING_OFF",
    "BandGapNarrowingModel",
    "BandGapNarrowingState",
    "SLOTBOOM",
    "apply_band_gap_narrowing",
    "normalize_band_gap_narrowing_model",
    "slotboom_band_gap_narrowing",
]

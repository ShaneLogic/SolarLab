"""Doping-dependent band-gap narrowing of the physical band edges (M class).

Slotboom's empirical law gives ``Delta Eg`` from the total dopant density;
a fraction ``f`` lowers the conduction edge (raises the effective electron
affinity by ``f Delta Eg``) and the rest raises the valence edge. The
intrinsic product and the SRH reference densities follow the narrowed gap
(``ni^2 exp(Delta Eg / V_T)``, ``n1, p1 exp(Delta Eg / 2 V_T)``), which keeps
``n1 p1 = ni^2`` when it held before narrowing. The narrowing is static: it
depends on the resolved dopant profile, never on the state.

All arithmetic is the historical one (``perovskite_sim.physics.band_gap_narrowing``
and the material build's scale factors).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

__all__ = [
    "BAND_GAP_NARROWING_OFF",
    "BandGapNarrowingModel",
    "BandGapNarrowingSpec",
    "BandGapNarrowingState",
    "SLOTBOOM",
    "apply_band_gap_narrowing",
    "intrinsic_product_narrowing_scale",
    "normalize_band_gap_narrowing_model",
    "slotboom_band_gap_narrowing",
    "srh_reference_narrowing_scale",
]

BAND_GAP_NARROWING_OFF = "off"
SLOTBOOM = "slotboom"
BandGapNarrowingModel = Literal["off", "slotboom"]

_DEFAULT_REFERENCE_ENERGY_EV = 0.009
_DEFAULT_REFERENCE_DENSITY_M3 = 1.0e23
_DEFAULT_LOG_SHAPE = 0.5
_DEFAULT_CONDUCTION_FRACTION = 0.5


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
    reference_energy_eV: float = _DEFAULT_REFERENCE_ENERGY_EV,
    reference_density_m3: float = _DEFAULT_REFERENCE_DENSITY_M3,
    log_shape: float = _DEFAULT_LOG_SHAPE,
) -> float:
    """Slotboom ``Delta Eg = E_ref (ln(N/N_ref) + sqrt(ln^2 + C))`` [eV].

    The negative-log branch uses the conjugate form ``C / (root - ln)`` so the
    small positive narrowing survives for ``N << N_ref``.
    """

    density = _finite_nonnegative(total_dopant_density_m3, "total_dopant_density_m3")
    energy = _finite_positive(reference_energy_eV, "reference_energy_eV")
    reference = _finite_positive(reference_density_m3, "reference_density_m3")
    shape = _finite_positive(log_shape, "log_shape")
    if density == 0.0:
        return 0.0
    logarithm = math.log(density) - math.log(reference)
    root = math.hypot(logarithm, math.sqrt(shape))
    slotboom_factor = logarithm + root if logarithm >= 0.0 else shape / (root - logarithm)
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
    reference_energy_eV: float = _DEFAULT_REFERENCE_ENERGY_EV,
    reference_density_m3: float = _DEFAULT_REFERENCE_DENSITY_M3,
    log_shape: float = _DEFAULT_LOG_SHAPE,
    conduction_band_fraction: float = _DEFAULT_CONDUCTION_FRACTION,
) -> BandGapNarrowingState:
    """Apply one BGN law to the physical conduction and valence edges."""

    affinity = _finite_nonnegative(electron_affinity_eV, "electron_affinity_eV")
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    acceptors = _finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
    donors = _finite_nonnegative(donor_density_m3, "donor_density_m3")
    total_dopants = _finite_nonnegative(acceptors + donors, "total_dopant_density_m3")
    narrowing_model = normalize_band_gap_narrowing_model(model)
    fraction = float(conduction_band_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("conduction_band_fraction must be finite and in [0, 1]")
    if narrowing_model == BAND_GAP_NARROWING_OFF:
        if (
            reference_energy_eV != _DEFAULT_REFERENCE_ENERGY_EV
            or reference_density_m3 != _DEFAULT_REFERENCE_DENSITY_M3
            or log_shape != _DEFAULT_LOG_SHAPE
            or fraction != _DEFAULT_CONDUCTION_FRACTION
        ):
            raise ValueError("BGN parameters require band-gap narrowing model 'slotboom'")
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


@dataclass(frozen=True, slots=True)
class BandGapNarrowingSpec:
    """A layer's resolved BGN declaration (defaults are the 'off' sentinel)."""

    model: str = BAND_GAP_NARROWING_OFF
    reference_energy_eV: float = _DEFAULT_REFERENCE_ENERGY_EV
    reference_density_m3: float = _DEFAULT_REFERENCE_DENSITY_M3
    log_shape: float = _DEFAULT_LOG_SHAPE
    conduction_band_fraction: float = _DEFAULT_CONDUCTION_FRACTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", normalize_band_gap_narrowing_model(self.model))
        if self.model == BAND_GAP_NARROWING_OFF and (
            self.reference_energy_eV != _DEFAULT_REFERENCE_ENERGY_EV
            or self.reference_density_m3 != _DEFAULT_REFERENCE_DENSITY_M3
            or self.log_shape != _DEFAULT_LOG_SHAPE
            or self.conduction_band_fraction != _DEFAULT_CONDUCTION_FRACTION
        ):
            raise ValueError("BGN parameters require band-gap narrowing model 'slotboom'")

    @property
    def active(self) -> bool:
        return self.model != BAND_GAP_NARROWING_OFF

    def apply(self, electron_affinity_eV: float, band_gap_eV: float,
              acceptor_density_m3: float, donor_density_m3: float) -> BandGapNarrowingState:
        return apply_band_gap_narrowing(
            electron_affinity_eV=float(electron_affinity_eV),
            band_gap_eV=float(band_gap_eV),
            acceptor_density_m3=float(acceptor_density_m3),
            donor_density_m3=float(donor_density_m3),
            model=self.model,
            reference_energy_eV=float(self.reference_energy_eV),
            reference_density_m3=float(self.reference_density_m3),
            log_shape=float(self.log_shape),
            conduction_band_fraction=float(self.conduction_band_fraction),
        )


def intrinsic_product_narrowing_scale(narrowing_eV: np.ndarray, thermal_voltage_V: float) -> np.ndarray:
    """``exp(Delta Eg / V_T)`` applied to ``ni^2`` on narrowed nodes."""

    scale = np.exp(narrowing_eV / thermal_voltage_V)
    if not np.all(np.isfinite(scale)):
        raise FloatingPointError("band-gap narrowing intrinsic-product scale is non-finite")
    return scale


def srh_reference_narrowing_scale(narrowing_eV: np.ndarray, thermal_voltage_V: float) -> np.ndarray:
    """``exp(Delta Eg / 2 V_T)`` applied to each SRH reference density."""

    scale = np.exp(0.5 * narrowing_eV / thermal_voltage_V)
    if not np.all(np.isfinite(scale)):
        raise FloatingPointError("band-gap narrowing SRH-reference scale is non-finite")
    return scale

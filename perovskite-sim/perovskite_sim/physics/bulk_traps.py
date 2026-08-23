"""Energy-resolved bulk-trap occupancy, recombination, and charge.

This module is an explicit Maxwell-Boltzmann research closure.  A single
normalized energy quadrature supplies every trap observable, so recombination
and electrostatic charge cannot silently use different defect populations.
Energies are measured upward from the valence-band edge in electron-volts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.special import ndtr, ndtri

from perovskite_sim.constants import Q
from perovskite_sim.physics.temperature import thermal_voltage


if TYPE_CHECKING:
    from perovskite_sim.physics.statistics import BulkChargeNeutralityState


SINGLE_LEVEL = "single_level"
GAUSSIAN = "gaussian"
ACCEPTOR = "acceptor"
DONOR = "donor"
DEFAULT_BULK_TRAP_QUADRATURE_ORDER = 64
BulkTrapDistributionKind = Literal["single_level", "gaussian"]
BulkTrapChargeTransition = Literal["acceptor", "donor"]


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class BulkTrapDistribution:
    """One spatially uniform, energy-resolved bulk-defect population.

    ``total_density_m3`` is the integral over trap energy, never a peak
    density.  Acceptor traps are neutral when empty and negative when filled;
    donor traps are positive when empty and neutral when filled.
    """

    distribution: BulkTrapDistributionKind
    total_density_m3: float
    center_eV_above_vb: float
    sigma_n_m2: float
    sigma_p_m2: float
    thermal_velocity_m_s: float
    charge_transition: BulkTrapChargeTransition
    energy_sigma_eV: float | None = None

    def __post_init__(self) -> None:
        distribution = str(self.distribution).strip().lower()
        if distribution not in {SINGLE_LEVEL, GAUSSIAN}:
            raise ValueError(
                "bulk trap distribution must be 'single_level' or 'gaussian'"
            )
        transition = str(self.charge_transition).strip().lower()
        if transition not in {ACCEPTOR, DONOR}:
            raise ValueError(
                "bulk trap charge_transition must be 'acceptor' or 'donor'"
            )
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "charge_transition", transition)
        for name in (
            "total_density_m3",
            "sigma_n_m2",
            "sigma_p_m2",
            "thermal_velocity_m_s",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "center_eV_above_vb",
            _finite_nonnegative(
                self.center_eV_above_vb,
                "center_eV_above_vb",
            ),
        )
        if distribution == SINGLE_LEVEL:
            if self.energy_sigma_eV is not None:
                raise ValueError(
                    "energy_sigma_eV is forbidden for single_level bulk traps"
                )
        else:
            if self.energy_sigma_eV is None:
                raise ValueError(
                    "gaussian bulk traps require energy_sigma_eV"
                )
            object.__setattr__(
                self,
                "energy_sigma_eV",
                _finite_positive(self.energy_sigma_eV, "energy_sigma_eV"),
            )

    def validate_band_gap(self, band_gap_eV: float) -> None:
        gap = _finite_positive(band_gap_eV, "band_gap_eV")
        if self.center_eV_above_vb > gap:
            raise ValueError(
                "bulk trap center_eV_above_vb must lie inside the band gap"
            )


def bulk_trap_distribution_from_mapping(
    value: Mapping[str, Any],
) -> BulkTrapDistribution:
    """Parse the strict standard-SI bulk-trap schema."""
    if not isinstance(value, Mapping):
        raise ValueError("bulk_trap_distribution must be a mapping")
    common = {
        "distribution",
        "total_density_m3",
        "center_eV_above_vb",
        "sigma_n_m2",
        "sigma_p_m2",
        "thermal_velocity_m_s",
        "charge_transition",
    }
    raw_distribution = value.get("distribution")
    distribution = (
        str(raw_distribution).strip().lower()
        if isinstance(raw_distribution, str)
        else raw_distribution
    )
    allowed = common | ({"energy_sigma_eV"} if distribution == GAUSSIAN else set())
    missing = sorted(common - set(value))
    unknown = sorted(str(key) for key in set(value) - allowed)
    if missing or unknown:
        raise ValueError(
            "bulk_trap_distribution schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    return BulkTrapDistribution(
        distribution=value["distribution"],
        total_density_m3=value["total_density_m3"],
        center_eV_above_vb=value["center_eV_above_vb"],
        sigma_n_m2=value["sigma_n_m2"],
        sigma_p_m2=value["sigma_p_m2"],
        thermal_velocity_m_s=value["thermal_velocity_m_s"],
        charge_transition=value["charge_transition"],
        energy_sigma_eV=value.get("energy_sigma_eV"),
    )


@dataclass(frozen=True, slots=True)
class BulkTrapQuadrature:
    """Finite energy levels and volume-density weights."""

    energy_levels_eV: tuple[float, ...]
    density_weights_m3: tuple[float, ...]

    @property
    def order(self) -> int:
        return len(self.energy_levels_eV)


def build_bulk_trap_quadrature(
    distribution: BulkTrapDistribution,
    *,
    band_gap_eV: float,
    order: int = DEFAULT_BULK_TRAP_QUADRATURE_ORDER,
) -> BulkTrapQuadrature:
    """Return a normalized quadrature confined to ``0 <= E_t <= E_g``.

    Gaussian levels use Gauss-Legendre integration in the probability
    coordinate of a truncated normal distribution.  This retains exact total
    density normalization and remains well resolved in the delta-like limit.
    """
    if not isinstance(distribution, BulkTrapDistribution):
        raise TypeError("distribution must be a BulkTrapDistribution")
    distribution.validate_band_gap(band_gap_eV)
    gap = float(band_gap_eV)
    if distribution.distribution == SINGLE_LEVEL:
        return BulkTrapQuadrature(
            energy_levels_eV=(float(distribution.center_eV_above_vb),),
            density_weights_m3=(float(distribution.total_density_m3),),
        )
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError("bulk trap quadrature order must be an integer")
    resolved_order = int(order)
    if resolved_order < 2 or resolved_order > 512:
        raise ValueError("bulk trap quadrature order must be in [2, 512]")
    sigma = float(distribution.energy_sigma_eV)
    center = float(distribution.center_eV_above_vb)
    lower_cdf = float(ndtr(-center / sigma))
    upper_cdf = float(ndtr((gap - center) / sigma))
    mass = upper_cdf - lower_cdf
    if not math.isfinite(mass) or mass <= np.finfo(float).eps:
        raise ValueError("gaussian bulk trap has no resolvable mass inside the gap")
    nodes, weights = np.polynomial.legendre.leggauss(resolved_order)
    probability = lower_cdf + 0.5 * (nodes + 1.0) * mass
    energy = center + sigma * ndtri(probability)
    density_weights = (
        0.5 * weights * float(distribution.total_density_m3)
    )
    if (
        not np.all(np.isfinite(energy))
        or not np.all(np.isfinite(density_weights))
        or np.any(energy < 0.0)
        or np.any(energy > gap)
        or np.any(density_weights <= 0.0)
    ):
        raise FloatingPointError("bulk trap energy quadrature is invalid")
    density_weights *= (
        float(distribution.total_density_m3) / float(np.sum(density_weights))
    )
    return BulkTrapQuadrature(
        energy_levels_eV=tuple(float(value) for value in energy),
        density_weights_m3=tuple(float(value) for value in density_weights),
    )


def _readonly_float_array(value: np.ndarray | float) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class BulkTrapState:
    """Energy-integrated trap state and exact local derivatives."""

    occupancy: np.ndarray
    occupied_density_m3: np.ndarray
    signed_charge_number_density_m3: np.ndarray
    charge_density_C_m3: np.ndarray
    recombination_rate_m3_s: np.ndarray
    recombination_derivative_n_s: np.ndarray
    recombination_derivative_p_s: np.ndarray
    charge_number_derivative_n: np.ndarray
    charge_number_derivative_p: np.ndarray
    charge_number_derivative_potential_m3_V: np.ndarray
    minimum_level_occupancy: float
    maximum_level_occupancy: float

    def __post_init__(self) -> None:
        for name in (
            "occupancy",
            "occupied_density_m3",
            "signed_charge_number_density_m3",
            "charge_density_C_m3",
            "recombination_rate_m3_s",
            "recombination_derivative_n_s",
            "recombination_derivative_p_s",
            "charge_number_derivative_n",
            "charge_number_derivative_p",
            "charge_number_derivative_potential_m3_V",
        ):
            object.__setattr__(self, name, _readonly_float_array(getattr(self, name)))


def evaluate_bulk_trap_state(
    electron_density_m3: np.ndarray | float,
    hole_density_m3: np.ndarray | float,
    distribution: BulkTrapDistribution,
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
    quadrature_order: int = DEFAULT_BULK_TRAP_QUADRATURE_ORDER,
) -> BulkTrapState:
    """Evaluate steady occupancy, SRH rate, charge, and analytic tangent."""
    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=float),
        np.asarray(hole_density_m3, dtype=float),
    )
    if (
        not np.all(np.isfinite(n))
        or not np.all(np.isfinite(p))
        or np.any(n < 0.0)
        or np.any(p < 0.0)
    ):
        raise ValueError("bulk trap carrier densities must be finite and non-negative")
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    conduction_dos = _finite_positive(
        effective_conduction_dos_m3,
        "effective_conduction_dos_m3",
    )
    valence_dos = _finite_positive(
        effective_valence_dos_m3,
        "effective_valence_dos_m3",
    )
    thermal = thermal_voltage(_finite_positive(temperature_K, "temperature_K"))
    quadrature = build_bulk_trap_quadrature(
        distribution,
        band_gap_eV=gap,
        order=quadrature_order,
    )
    energy = np.asarray(quadrature.energy_levels_eV, dtype=float)
    weights = np.asarray(quadrature.density_weights_m3, dtype=float)
    shape = (energy.size,) + (1,) * n.ndim
    energy = energy.reshape(shape)
    weights = weights.reshape(shape)
    n_expanded = np.expand_dims(n, axis=0)
    p_expanded = np.expand_dims(p, axis=0)
    n1 = conduction_dos * np.exp(-(gap - energy) / thermal)
    p1 = valence_dos * np.exp(-energy / thermal)
    capture_n = float(distribution.sigma_n_m2) * float(
        distribution.thermal_velocity_m_s
    )
    capture_p = float(distribution.sigma_p_m2) * float(
        distribution.thermal_velocity_m_s
    )
    denominator = (
        capture_n * (n_expanded + n1)
        + capture_p * (p_expanded + p1)
    )
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise FloatingPointError("bulk trap kinetic denominator is invalid")
    filled_numerator = capture_n * n_expanded + capture_p * p1
    level_occupancy = filled_numerator / denominator
    one_minus_occupancy = 1.0 - level_occupancy
    occupancy = np.sum(weights * level_occupancy, axis=0) / float(
        distribution.total_density_m3
    )
    occupied_density = np.sum(weights * level_occupancy, axis=0)
    intrinsic_product = conduction_dos * valence_dos * math.exp(-gap / thermal)
    excess_product = n_expanded * p_expanded - intrinsic_product
    level_rate = (
        weights * capture_n * capture_p * excess_product / denominator
    )
    rate = np.sum(level_rate, axis=0)
    derivative_n = np.sum(
        weights
        * capture_n
        * capture_p
        * (p_expanded * denominator - excess_product * capture_n)
        / denominator**2,
        axis=0,
    )
    derivative_p = np.sum(
        weights
        * capture_n
        * capture_p
        * (n_expanded * denominator - excess_product * capture_p)
        / denominator**2,
        axis=0,
    )
    occupancy_derivative_n = capture_n * one_minus_occupancy / denominator
    occupancy_derivative_p = -capture_p * level_occupancy / denominator
    charge_derivative_n = -np.sum(weights * occupancy_derivative_n, axis=0)
    charge_derivative_p = -np.sum(weights * occupancy_derivative_p, axis=0)
    if distribution.charge_transition == ACCEPTOR:
        signed_charge_number = -occupied_density
    else:
        signed_charge_number = float(distribution.total_density_m3) - occupied_density
    potential_derivative = (
        charge_derivative_n * n / thermal
        - charge_derivative_p * p / thermal
    )
    outputs = (
        occupancy,
        occupied_density,
        signed_charge_number,
        rate,
        derivative_n,
        derivative_p,
        charge_derivative_n,
        charge_derivative_p,
        potential_derivative,
    )
    if not all(np.all(np.isfinite(value)) for value in outputs):
        raise FloatingPointError("bulk trap state is non-finite")
    minimum = float(np.min(level_occupancy))
    maximum = float(np.max(level_occupancy))
    if minimum < -1.0e-14 or maximum > 1.0 + 1.0e-14:
        raise FloatingPointError("bulk trap level occupancy left [0, 1]")
    return BulkTrapState(
        occupancy=occupancy,
        occupied_density_m3=occupied_density,
        signed_charge_number_density_m3=signed_charge_number,
        charge_density_C_m3=Q * signed_charge_number,
        recombination_rate_m3_s=rate,
        recombination_derivative_n_s=derivative_n,
        recombination_derivative_p_s=derivative_p,
        charge_number_derivative_n=charge_derivative_n,
        charge_number_derivative_p=charge_derivative_p,
        charge_number_derivative_potential_m3_V=potential_derivative,
        minimum_level_occupancy=minimum,
        maximum_level_occupancy=maximum,
    )


@dataclass(frozen=True, slots=True)
class BulkTrapNeutralityResult:
    """A trap-aware common-Fermi-level contact state."""

    neutrality: BulkChargeNeutralityState
    trap_state: BulkTrapState


def solve_bulk_trap_charge_neutrality(
    *,
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    acceptor_density_m3: float,
    donor_density_m3: float,
    distribution: BulkTrapDistribution,
    quadrature_order: int = DEFAULT_BULK_TRAP_QUADRATURE_ORDER,
) -> BulkTrapNeutralityResult:
    """Solve ``p-n+N_D-N_A+N_trap_charge=0`` in the MB limit."""
    from perovskite_sim.physics.statistics import (
        BulkChargeNeutralityState,
        FULLY_IONIZED,
        MAXWELL_BOLTZMANN,
        carrier_density_from_reduced_fermi_level,
    )

    temperature = _finite_positive(temperature_K, "temperature_K")
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    conduction_dos = _finite_positive(
        effective_conduction_dos_m3,
        "effective_conduction_dos_m3",
    )
    valence_dos = _finite_positive(
        effective_valence_dos_m3,
        "effective_valence_dos_m3",
    )
    acceptors = _finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
    donors = _finite_nonnegative(donor_density_m3, "donor_density_m3")
    thermal = thermal_voltage(temperature)
    reduced_gap = gap / thermal

    def evaluate(
        eta_n: float,
    ) -> tuple[float, float, float, float, BulkTrapState]:
        eta_p = -reduced_gap - eta_n
        electron = carrier_density_from_reduced_fermi_level(
            eta_n,
            conduction_dos,
            statistics=MAXWELL_BOLTZMANN,
        )
        hole = carrier_density_from_reduced_fermi_level(
            eta_p,
            valence_dos,
            statistics=MAXWELL_BOLTZMANN,
        )
        trap = evaluate_bulk_trap_state(
            electron,
            hole,
            distribution,
            band_gap_eV=gap,
            effective_conduction_dos_m3=conduction_dos,
            effective_valence_dos_m3=valence_dos,
            temperature_K=temperature,
            quadrature_order=quadrature_order,
        )
        residual = (
            hole
            - electron
            + donors
            - acceptors
            + float(trap.signed_charge_number_density_m3)
        )
        return residual, electron, hole, eta_p, trap

    intrinsic_center = 0.5 * (
        -reduced_gap + math.log(valence_dos / conduction_dos)
    )
    half_width = max(32.0, 0.5 * reduced_gap + 8.0)
    lower = intrinsic_center - half_width
    upper = intrinsic_center + half_width
    for _ in range(32):
        lower_residual = evaluate(lower)[0]
        upper_residual = evaluate(upper)[0]
        if lower_residual >= 0.0 and upper_residual <= 0.0:
            break
        half_width *= 2.0
        lower = intrinsic_center - half_width
        upper = intrinsic_center + half_width
    else:
        raise RuntimeError("could not bracket trap-aware charge neutrality")
    for _ in range(220):
        midpoint = 0.5 * (lower + upper)
        residual = evaluate(midpoint)[0]
        if residual > 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= max(2.0e-14, 4.0 * math.ulp(midpoint)):
            break
    eta_n = 0.5 * (lower + upper)
    residual, electron, hole, eta_p, trap = evaluate(eta_n)
    scale = max(
        electron,
        hole,
        donors,
        acceptors,
        float(distribution.total_density_m3),
        1.0,
    )
    normalized = abs(residual) / scale
    if not math.isfinite(normalized) or normalized > 1.0e-12:
        raise RuntimeError("trap-aware charge-neutrality residual exceeded gate")
    neutrality = BulkChargeNeutralityState(
        statistics=MAXWELL_BOLTZMANN,
        temperature_K=temperature,
        thermal_voltage_V=thermal,
        band_gap_eV=gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        acceptor_density_m3=acceptors,
        donor_density_m3=donors,
        reduced_electron_fermi_level=eta_n,
        reduced_hole_fermi_level=eta_p,
        electron_density_m3=electron,
        hole_density_m3=hole,
        normalized_charge_residual=normalized,
        dopant_ionization_model=FULLY_IONIZED,
        ionized_acceptor_density_m3=acceptors,
        ionized_donor_density_m3=donors,
        acceptor_ionized_fraction=1.0,
        donor_ionized_fraction=1.0,
    )
    return BulkTrapNeutralityResult(neutrality=neutrality, trap_state=trap)


__all__ = [
    "ACCEPTOR",
    "DEFAULT_BULK_TRAP_QUADRATURE_ORDER",
    "DONOR",
    "GAUSSIAN",
    "SINGLE_LEVEL",
    "BulkTrapChargeTransition",
    "BulkTrapDistribution",
    "BulkTrapDistributionKind",
    "BulkTrapNeutralityResult",
    "BulkTrapQuadrature",
    "BulkTrapState",
    "build_bulk_trap_quadrature",
    "bulk_trap_distribution_from_mapping",
    "evaluate_bulk_trap_state",
    "solve_bulk_trap_charge_neutrality",
]

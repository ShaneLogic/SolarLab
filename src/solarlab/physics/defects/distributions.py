"""Energy-distributed and spatially graded bulk defect quadrature and profiles.

All canonical defect energies use ``center_eV_above_vb`` (electron energy above
the local valence-band edge ``E_v``, bounded by ``[0, E_g]``). Quadrature rules
preserve exact zero-th moment ``sum(density_weights_m3) == total_density_m3``
via inverse-CDF Gauss-Legendre nodes on bounded support.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

DIST_SINGLE_LEVEL = "single_level"
DIST_UNIFORM = "uniform"
DIST_GAUSSIAN = "gaussian"
DIST_CONDUCTION_BAND_TAIL = "conduction_band_tail"
DIST_VALENCE_BAND_TAIL = "valence_band_tail"
SUPPORTED_DISTRIBUTIONS = frozenset(
    {
        DIST_SINGLE_LEVEL,
        DIST_UNIFORM,
        DIST_GAUSSIAN,
        DIST_CONDUCTION_BAND_TAIL,
        DIST_VALENCE_BAND_TAIL,
    }
)

WIDTH_NOT_APPLICABLE = "not_applicable"
WIDTH_UNIFORM_FULL = "uniform_full_width"
WIDTH_GAUSSIAN_SIGMA = "gaussian_standard_deviation"
WIDTH_SCAPS_CHARACTERISTIC = "scaps_characteristic_energy"

DENSITY_INTEGRATED = "integrated_total"
DENSITY_PEAK = "peak_density"

DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER = 64
MINIMUM_DEFECT_ENERGY_QUADRATURE_ORDER = 8
MAXIMUM_DEFECT_ENERGY_QUADRATURE_ORDER = 512

__all__ = [
    "DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER",
    "DENSITY_INTEGRATED",
    "DENSITY_PEAK",
    "DIST_CONDUCTION_BAND_TAIL",
    "DIST_GAUSSIAN",
    "DIST_SINGLE_LEVEL",
    "DIST_UNIFORM",
    "DIST_VALENCE_BAND_TAIL",
    "DefectDistributionSpec",
    "DefectEnergyQuadrature",
    "MAXIMUM_DEFECT_ENERGY_QUADRATURE_ORDER",
    "MINIMUM_DEFECT_ENERGY_QUADRATURE_ORDER",
    "SUPPORTED_DISTRIBUTIONS",
    "SpatialDensityProfileSpec",
    "SpatialProfileKnot",
    "WIDTH_GAUSSIAN_SIGMA",
    "WIDTH_NOT_APPLICABLE",
    "WIDTH_SCAPS_CHARACTERISTIC",
    "WIDTH_UNIFORM_FULL",
    "build_defect_energy_quadrature",
    "density_weighted_mean_occupancy",
    "distribution_shape_integral_eV",
    "evaluate_spatial_density_profile",
    "integrated_density_from_peak_density",
    "peak_density_from_integrated_density",
    "validate_defect_energy_quadrature_order",
]


def _ro_float(values: ArrayLike) -> NDArray[np.float64]:
    arr = np.array(values, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_nonnegative(value: float, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _finite_positive(value: float, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def validate_defect_energy_quadrature_order(order: int) -> int:
    """Validate energy quadrature order for continuous defect distributions."""
    if isinstance(order, bool) or not isinstance(order, int):
        raise ValueError("defect energy quadrature order must be an integer")
    if not (
        MINIMUM_DEFECT_ENERGY_QUADRATURE_ORDER
        <= order
        <= MAXIMUM_DEFECT_ENERGY_QUADRATURE_ORDER
    ):
        raise ValueError(
            f"defect energy quadrature order must lie in "
            f"[{MINIMUM_DEFECT_ENERGY_QUADRATURE_ORDER}, "
            f"{MAXIMUM_DEFECT_ENERGY_QUADRATURE_ORDER}]"
        )
    return order


def distribution_shape_integral_eV(
    kind: str,
    width_eV: float,
    *,
    width_convention: str = WIDTH_NOT_APPLICABLE,
    support_width_multiplier: float | None = None,
) -> float:
    """Exact integral [eV] of the unit-peak continuous distribution shape."""
    w = _finite_positive(width_eV, "width_eV")
    if kind == DIST_UNIFORM:
        if width_convention not in (WIDTH_UNIFORM_FULL, WIDTH_NOT_APPLICABLE):
            raise ValueError(f"unsupported uniform width_convention: {width_convention}")
        return w
    if support_width_multiplier is None:
        raise ValueError(f"{kind} requires support_width_multiplier")
    m = _finite_positive(support_width_multiplier, "support_width_multiplier")
    if kind == DIST_GAUSSIAN:
        if width_convention == WIDTH_GAUSSIAN_SIGMA:
            return w * math.sqrt(2.0 * math.pi) * math.erf(m / (2.0 * math.sqrt(2.0)))
        if width_convention == WIDTH_SCAPS_CHARACTERISTIC:
            return w * math.sqrt(math.pi) * math.erf(0.5 * m)
        raise ValueError(f"unsupported gaussian width_convention: {width_convention}")
    if kind in (DIST_CONDUCTION_BAND_TAIL, DIST_VALENCE_BAND_TAIL):
        return w * (-math.expm1(-m))
    raise ValueError(f"continuous shape integral not defined for kind={kind!r}")


def integrated_density_from_peak_density(
    peak_density_m3_eV: float, shape_integral_eV: float
) -> float:
    """Convert peak density [m^-3 eV^-1] to integrated density [m^-3]."""
    peak = _finite_nonnegative(peak_density_m3_eV, "peak_density_m3_eV")
    shape = _finite_positive(shape_integral_eV, "shape_integral_eV")
    result = peak * shape
    if not math.isfinite(result):
        raise ValueError("integrated defect density overflowed")
    return result


def peak_density_from_integrated_density(
    total_density_m3: float, shape_integral_eV: float
) -> float:
    """Convert integrated density [m^-3] to peak density [m^-3 eV^-1]."""
    total = _finite_nonnegative(total_density_m3, "total_density_m3")
    shape = _finite_positive(shape_integral_eV, "shape_integral_eV")
    result = total / shape
    if not math.isfinite(result):
        raise ValueError("peak defect density overflowed")
    return result


@dataclass(frozen=True, slots=True)
class DefectDistributionSpec:
    """Canonical energy distribution specification above the valence band."""

    kind: str
    center_eV_above_vb: float
    total_density_m3: float
    width_eV: float = 0.0
    width_convention: str = WIDTH_NOT_APPLICABLE
    support_width_multiplier: float | None = None
    peak_density_m3_eV: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"unsupported defect distribution kind: {self.kind!r}")
        center = _finite(self.center_eV_above_vb, "center_eV_above_vb")
        total = _finite_nonnegative(self.total_density_m3, "total_density_m3")
        object.__setattr__(self, "center_eV_above_vb", center)
        object.__setattr__(self, "total_density_m3", total)
        if self.kind == DIST_SINGLE_LEVEL:
            if self.width_eV != 0.0:
                raise ValueError("single_level requires width_eV == 0")
            if self.support_width_multiplier is not None:
                raise ValueError("single_level forbids support_width_multiplier")
            if self.peak_density_m3_eV is not None:
                raise ValueError("single_level forbids peak_density_m3_eV")
            return
        width = _finite_positive(self.width_eV, "width_eV")
        object.__setattr__(self, "width_eV", width)
        if self.kind == DIST_UNIFORM:
            conv = (
                WIDTH_UNIFORM_FULL
                if self.width_convention == WIDTH_NOT_APPLICABLE
                else self.width_convention
            )
            if conv != WIDTH_UNIFORM_FULL:
                raise ValueError(f"uniform requires {WIDTH_UNIFORM_FULL!r}")
            object.__setattr__(self, "width_convention", conv)
            mult = (
                1.0
                if self.support_width_multiplier is None
                else _finite_positive(
                    self.support_width_multiplier, "support_width_multiplier"
                )
            )
            if mult != 1.0:
                raise ValueError("uniform support_width_multiplier must equal 1.0")
            object.__setattr__(self, "support_width_multiplier", 1.0)
        elif self.kind == DIST_GAUSSIAN:
            conv = (
                WIDTH_GAUSSIAN_SIGMA
                if self.width_convention == WIDTH_NOT_APPLICABLE
                else self.width_convention
            )
            if conv not in (WIDTH_GAUSSIAN_SIGMA, WIDTH_SCAPS_CHARACTERISTIC):
                raise ValueError(f"unsupported gaussian width_convention: {conv!r}")
            object.__setattr__(self, "width_convention", conv)
            if self.support_width_multiplier is None:
                if self.peak_density_m3_eV is not None:
                    raise ValueError(
                        "gaussian peak_density_m3_eV requires explicit support_width_multiplier"
                    )
                object.__setattr__(self, "peak_density_m3_eV", None)
                return
            mult = _finite_positive(
                self.support_width_multiplier, "support_width_multiplier"
            )
            object.__setattr__(self, "support_width_multiplier", mult)
        else:
            if self.support_width_multiplier is None:
                raise ValueError(f"{self.kind} requires support_width_multiplier")
            mult = _finite_positive(
                self.support_width_multiplier, "support_width_multiplier"
            )
            conv = (
                WIDTH_SCAPS_CHARACTERISTIC
                if self.width_convention == WIDTH_NOT_APPLICABLE
                else self.width_convention
            )
            if conv != WIDTH_SCAPS_CHARACTERISTIC:
                raise ValueError(
                    f"{self.kind} requires width_convention={WIDTH_SCAPS_CHARACTERISTIC!r}"
                )
            object.__setattr__(self, "width_convention", conv)
            object.__setattr__(self, "support_width_multiplier", mult)

        shape = distribution_shape_integral_eV(
            self.kind,
            self.width_eV,
            width_convention=self.width_convention,
            support_width_multiplier=self.support_width_multiplier,
        )
        expected_peak = peak_density_from_integrated_density(total, shape)
        if self.peak_density_m3_eV is None:
            object.__setattr__(self, "peak_density_m3_eV", expected_peak)
        else:
            supplied_peak = _finite_nonnegative(
                self.peak_density_m3_eV, "peak_density_m3_eV"
            )
            if not math.isclose(
                supplied_peak, expected_peak, rel_tol=1e-12, abs_tol=0.0
            ):
                raise ValueError(
                    "peak_density_m3_eV disagrees with total_density_m3 / shape_integral_eV"
                )
            object.__setattr__(self, "peak_density_m3_eV", supplied_peak)

    @property
    def support_bounds_eV_above_vb(self) -> tuple[float, float]:
        c = self.center_eV_above_vb
        if self.kind == DIST_SINGLE_LEVEL:
            return (c, c)
        w = self.width_eV
        if self.kind == DIST_UNIFORM:
            return (c - 0.5 * w, c + 0.5 * w)
        if self.support_width_multiplier is None:
            raise ValueError(
                "unbounded gaussian requires band_gap_eV to resolve truncated support"
            )
        m = self.support_width_multiplier
        if self.kind == DIST_GAUSSIAN:
            half = 0.5 * m * w
            return (c - half, c + half)
        if self.kind == DIST_CONDUCTION_BAND_TAIL:
            return (c - m * w, c)
        return (c, c + m * w)


@dataclass(frozen=True, slots=True)
class DefectEnergyQuadrature:
    """Normalized discrete energy quadrature for a defect distribution."""

    distribution_kind: str
    quadrature_rule: str
    order: int
    support_lower_eV_above_vb: float
    support_upper_eV_above_vb: float
    energies_eV_above_vb: NDArray[np.float64]
    density_weights_m3: NDArray[np.float64]
    normalized_weights: NDArray[np.float64]
    total_density_m3: float
    peak_density_m3_eV: float | None
    shape_integral_eV: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "energies_eV_above_vb", _ro_float(self.energies_eV_above_vb)
        )
        object.__setattr__(
            self, "density_weights_m3", _ro_float(self.density_weights_m3)
        )
        object.__setattr__(
            self, "normalized_weights", _ro_float(self.normalized_weights)
        )
        if (
            self.energies_eV_above_vb.shape != (self.order,)
            or self.density_weights_m3.shape != (self.order,)
            or self.normalized_weights.shape != (self.order,)
        ):
            raise ValueError("quadrature arrays must match declared order")
        if self.order > 1 and np.any(np.diff(self.energies_eV_above_vb) <= 0.0):
            raise ValueError("quadrature energies must be strictly increasing")
        if np.any(self.normalized_weights <= 0.0) or not math.isclose(
            float(np.sum(self.normalized_weights)), 1.0, rel_tol=0.0, abs_tol=1e-14
        ):
            raise ValueError("normalized quadrature weights must be positive and sum to 1")


def build_defect_energy_quadrature(
    spec: DefectDistributionSpec,
    *,
    order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    band_gap_eV: float | None = None,
) -> DefectEnergyQuadrature:
    """Build exact-zeroth-moment energy quadrature on ``[0, band_gap_eV]``."""
    if spec.kind == DIST_GAUSSIAN and spec.support_width_multiplier is None:
        if band_gap_eV is None:
            raise ValueError(
                "band-gap-truncated gaussian distribution requires band_gap_eV"
            )
        gap = _finite_positive(band_gap_eV, "band_gap_eV")
        if not 0.0 <= spec.center_eV_above_vb <= gap:
            raise ValueError("gaussian center_eV_above_vb lies outside [0, band_gap_eV]")
        lo, hi = 0.0, gap
    else:
        lo, hi = spec.support_bounds_eV_above_vb
        if band_gap_eV is not None:
            gap = _finite_positive(band_gap_eV, "band_gap_eV")
            tol = 16.0 * math.ulp(max(gap, abs(lo), abs(hi), 1.0))
            if lo < -tol or hi > gap + tol:
                raise ValueError(
                    f"defect distribution support [{lo}, {hi}] lies outside [0, {gap}] eV"
                )
    if spec.kind == DIST_SINGLE_LEVEL:
        return DefectEnergyQuadrature(
            distribution_kind=DIST_SINGLE_LEVEL,
            quadrature_rule="dirac_delta",
            order=1,
            support_lower_eV_above_vb=lo,
            support_upper_eV_above_vb=hi,
            energies_eV_above_vb=np.array([spec.center_eV_above_vb], dtype=np.float64),
            density_weights_m3=np.array([spec.total_density_m3], dtype=np.float64),
            normalized_weights=np.array([1.0], dtype=np.float64),
            total_density_m3=spec.total_density_m3,
            peak_density_m3_eV=None,
            shape_integral_eV=None,
        )

    n = validate_defect_energy_quadrature_order(order)
    xi, omega = leggauss(n)
    weights = 0.5 * omega
    weights = weights / float(np.sum(weights))
    c = spec.center_eV_above_vb
    w = spec.width_eV

    if spec.kind == DIST_GAUSSIAN and spec.support_width_multiplier is None:
        sigma = (
            w
            if spec.width_convention == WIDTH_GAUSSIAN_SIGMA
            else w / math.sqrt(2.0)
        )
        u_lo = float(ndtr(-c / sigma))
        u_hi = float(ndtr((hi - c) / sigma))
        mass = u_hi - u_lo
        if not math.isfinite(mass) or mass <= np.finfo(float).eps:
            raise ValueError("gaussian defect distribution has no resolvable mass inside the gap")
        u = u_lo + 0.5 * (xi + 1.0) * mass
        energies = c + sigma * np.asarray(ndtri(u), dtype=np.float64)
        shape: float | None = sigma * math.sqrt(2.0 * math.pi) * mass
        rule = "gauss_legendre_truncated_normal_cdf"
    else:
        assert spec.support_width_multiplier is not None
        m = spec.support_width_multiplier
        shape = distribution_shape_integral_eV(
            spec.kind,
            w,
            width_convention=spec.width_convention,
            support_width_multiplier=m,
        )
        if spec.kind == DIST_UNIFORM:
            energies = c + 0.5 * w * xi
            rule = "gauss_legendre_energy"
        elif spec.kind == DIST_GAUSSIAN:
            sigma = (
                w
                if spec.width_convention == WIDTH_GAUSSIAN_SIGMA
                else w / math.sqrt(2.0)
            )
            z_bound = 0.5 * m * w / sigma
            u_lo = float(ndtr(-z_bound))
            u_hi = float(ndtr(z_bound))
            if not (0.0 <= u_lo < u_hi <= 1.0):
                raise ValueError("truncated Gaussian CDF bounds collapsed in float64")
            u = 0.5 * (u_lo + u_hi) + 0.5 * (u_hi - u_lo) * xi
            energies = c + sigma * np.asarray(ndtri(u), dtype=np.float64)
            rule = "gauss_legendre_truncated_normal_cdf"
        elif spec.kind == DIST_CONDUCTION_BAND_TAIL:
            trunc = -math.expm1(-m)
            # Map xi in [-1, 1] -> u in [1, 0] so energies increase strictly from c - m*w to c.
            u_desc = 0.5 * (1.0 - xi)
            energies = c + w * np.log1p(-u_desc * trunc)
            rule = "gauss_legendre_truncated_exponential_cdf"
        else:
            trunc = -math.expm1(-m)
            u_asc = 0.5 * (1.0 + xi)
            energies = c - w * np.log1p(-u_asc * trunc)
            rule = "gauss_legendre_truncated_exponential_cdf"

    density_weights = spec.total_density_m3 * weights
    # Enforce exact floating-point sum on positive total_density_m3 if within 4 ULP.
    if spec.total_density_m3 > 0.0:
        scale = spec.total_density_m3 / float(np.sum(density_weights))
        density_weights = density_weights * scale

    return DefectEnergyQuadrature(
        distribution_kind=spec.kind,
        quadrature_rule=rule,
        order=n,
        support_lower_eV_above_vb=lo,
        support_upper_eV_above_vb=hi,
        energies_eV_above_vb=energies,
        density_weights_m3=density_weights,
        normalized_weights=weights,
        total_density_m3=spec.total_density_m3,
        peak_density_m3_eV=spec.peak_density_m3_eV,
        shape_integral_eV=shape,
    )


def density_weighted_mean_occupancy(
    quadrature: DefectEnergyQuadrature,
    node_occupancy: ArrayLike,
) -> NDArray[np.float64]:
    """Mean occupancy ``sum_k w_k f_k`` across quadrature modes (axis 0)."""
    occ = np.asarray(node_occupancy, dtype=np.float64)
    if occ.shape[0] != quadrature.order:
        raise ValueError(
            f"node_occupancy axis 0 ({occ.shape[0]}) != quadrature order ({quadrature.order})"
        )
    return np.tensordot(quadrature.normalized_weights, occ, axes=(0, 0))


@dataclass(frozen=True, slots=True)
class SpatialProfileKnot:
    """Normalized ``[0, 1]`` knot for piecewise-linear spatial defect profiles."""

    position_fraction: float
    density_multiplier: float

    def __post_init__(self) -> None:
        pos = _finite(self.position_fraction, "position_fraction")
        if not 0.0 <= pos <= 1.0:
            raise ValueError("position_fraction must lie in [0, 1]")
        mult = _finite_nonnegative(self.density_multiplier, "density_multiplier")
        object.__setattr__(self, "position_fraction", pos)
        object.__setattr__(self, "density_multiplier", mult)


@dataclass(frozen=True, slots=True)
class SpatialDensityProfileSpec:
    """Layer-average-preserving piecewise-linear defect spatial profile."""

    knots: tuple[SpatialProfileKnot, ...]

    def __post_init__(self) -> None:
        items = tuple(self.knots)
        if len(items) < 2:
            raise ValueError("spatial density profile requires at least 2 knots")
        if items[0].position_fraction != 0.0 or items[-1].position_fraction != 1.0:
            raise ValueError("spatial density profile knots must span [0, 1]")
        segments: list[float] = []
        for left, right in zip(items, items[1:]):
            dx = right.position_fraction - left.position_fraction
            if dx <= 0.0:
                raise ValueError("spatial density profile knots must strictly increase")
            segments.append(
                0.5 * (left.density_multiplier + right.density_multiplier) * dx
            )
        integral = math.fsum(segments)
        if not math.isclose(integral, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"spatial density profile must have unit layer average (got {integral})"
            )
        object.__setattr__(self, "knots", items)


def evaluate_spatial_density_profile(
    profile: SpatialDensityProfileSpec,
    normalized_positions: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate spatial density multipliers on normalized layer positions in ``[0, 1]``."""
    pos = np.asarray(normalized_positions, dtype=np.float64)
    if np.any(~np.isfinite(pos)) or np.any(pos < 0.0) or np.any(pos > 1.0):
        raise ValueError("normalized_positions must be finite and lie in [0, 1]")
    xp = np.array([k.position_fraction for k in profile.knots], dtype=np.float64)
    fp = np.array([k.density_multiplier for k in profile.knots], dtype=np.float64)
    result = np.asarray(np.interp(pos, xp, fp), dtype=np.float64)
    result.setflags(write=False)
    return result

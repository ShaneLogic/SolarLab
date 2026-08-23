"""Composition-dependent optical constants for graded CIGS absorbers.

This module implements the dielectric-function construction of Minoura et al.
for Cu(In,Ga)Se2 and the independent near-edge absorption expression of Carron
et al.  The Minoura reference spectra are sums of Tauc-Lorentz oscillators.
Their Ga dependence is represented by a critical-point energy shift and their
Cu dependence by spectral averaging between measured reference compositions.

The real dielectric function is recovered from the shifted/averaged imaginary
part with a principal-value Kramers-Kronig quadrature.  Applying the transform
after composition interpolation keeps each returned ``n, k`` pair causal; it
also gives a directly refinable quadrature rather than copying the lengthy
closed-form Tauc-Lorentz real-part expression.

Sources
-------
S. Minoura et al., J. Appl. Phys. 117, 195703 (2015),
doi:10.1063/1.4921300.  Equations (1), (9)-(15), Tables I-II.

R. Carron et al., Sci. Technol. Adv. Mater. 19, 396-410 (2018),
doi:10.1080/14686996.2018.1458579.  Equations (2)-(6).

The implementation is intentionally restricted to the Cu-rich device regime
``0.75 <= CGI <= 1``.  Minoura reports increasing model error below CGI=0.69,
while Carron's independent absorption expression is valid only above 0.75.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Final, Mapping

import numpy as np


MINOURA_2015: Final = "minoura_2015"
HC_EV_NM: Final = 1239.8419843320026
MIN_WAVELENGTH_NM: Final = HC_EV_NM / 6.5
MAX_WAVELENGTH_NM: Final = HC_EV_NM / 0.7


@dataclass(frozen=True, slots=True)
class CIGSGradedOptics:
    """Strict opt-in schema for one composition-graded CIGS layer.

    ``ggi_front`` and ``ggi_back`` are Ga/(In+Ga) at shared material
    coordinates ``y=0`` and ``y=1``. The layer's ``grading_direction`` maps
    those endpoints onto physical front/back faces in exactly the same way as
    the electrical Eg/chi grade. ``cgi`` is Cu/(In+Ga) and is currently
    uniform through the layer.
    """

    ggi_front: float
    ggi_back: float
    cgi: float
    slices: int = 25
    kk_quadrature_order: int = 192
    model: str = MINOURA_2015

    def __post_init__(self) -> None:
        if self.model != MINOURA_2015:
            raise ValueError(
                f"unsupported CIGS optical model {self.model!r}; "
                f"expected {MINOURA_2015!r}"
            )
        for name, value in (
            ("ggi_front", self.ggi_front),
            ("ggi_back", self.ggi_back),
            ("cgi", self.cgi),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.ggi_front <= 1.0:
            raise ValueError("ggi_front must lie in [0, 1]")
        if not 0.0 <= self.ggi_back <= 1.0:
            raise ValueError("ggi_back must lie in [0, 1]")
        if not 0.75 <= self.cgi <= 1.0:
            raise ValueError(
                "cgi must lie in [0.75, 1.0] so both the Minoura database "
                "and Carron near-edge benchmark remain in their declared domain"
            )
        if isinstance(self.slices, bool) or int(self.slices) != self.slices:
            raise ValueError("slices must be an integer")
        if not 1 <= int(self.slices) <= 512:
            raise ValueError("slices must lie in [1, 512]")
        if (
            isinstance(self.kk_quadrature_order, bool)
            or int(self.kk_quadrature_order) != self.kk_quadrature_order
        ):
            raise ValueError("kk_quadrature_order must be an integer")
        if not 48 <= int(self.kk_quadrature_order) <= 2048:
            raise ValueError("kk_quadrature_order must lie in [48, 2048]")


_CIGS_OPTICS_KEYS: Final = frozenset(
    {
        "model",
        "ggi_front",
        "ggi_back",
        "cgi",
        "slices",
        "kk_quadrature_order",
    }
)


def _integer_schema_value(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if integer != value:
        raise ValueError(f"{name} must be an integer")
    return integer


def _finite_schema_value(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def cigs_graded_optics_from_mapping(raw: Mapping[str, object]) -> CIGSGradedOptics:
    """Parse a strict nested CIGS optical block.

    Unknown and missing physical keys fail closed so a misspelt composition
    cannot silently fall back to nominal optics.
    """

    if not isinstance(raw, Mapping):
        raise TypeError("cigs_graded_optics must be a mapping")
    unknown = set(raw) - _CIGS_OPTICS_KEYS
    if unknown:
        raise ValueError(
            "unknown cigs_graded_optics keys: " + ", ".join(sorted(unknown))
        )
    missing = {"ggi_front", "ggi_back", "cgi"} - set(raw)
    if missing:
        raise ValueError(
            "missing cigs_graded_optics keys: " + ", ".join(sorted(missing))
        )
    return CIGSGradedOptics(
        model=str(raw.get("model", MINOURA_2015)),
        ggi_front=_finite_schema_value(raw["ggi_front"], "ggi_front"),
        ggi_back=_finite_schema_value(raw["ggi_back"], "ggi_back"),
        cgi=_finite_schema_value(raw["cgi"], "cgi"),
        slices=_integer_schema_value(raw.get("slices", 25), "slices"),
        kk_quadrature_order=_integer_schema_value(
            raw.get("kk_quadrature_order", 192), "kk_quadrature_order"
        ),
    )


@dataclass(frozen=True, slots=True)
class _TaucLorentzPeak:
    peak_eV: float
    amplitude_eV: float
    broadening_eV: float
    gap_eV: float


@dataclass(frozen=True, slots=True)
class _ReferenceSpectrum:
    ggi: float
    cgi: float
    epsilon_infinity: float
    peaks: tuple[_TaucLorentzPeak, ...]


def _peaks(*rows: tuple[float, float, float, float]) -> tuple[_TaucLorentzPeak, ...]:
    return tuple(_TaucLorentzPeak(*row) for row in rows)


# Minoura Table II.  A-D span GGI at CGI=0.90; E-F span CGI at GGI=0.40.
_REFERENCES: Final[dict[str, _ReferenceSpectrum]] = {
    "A": _ReferenceSpectrum(
        0.00,
        0.90,
        1.342,
        _peaks(
            (0.996, 12.255, 0.175, 0.947),
            (1.281, 14.580, 1.087, 1.001),
            (1.860, 21.014, 2.446, 1.438),
            (2.909, 13.185, 0.802, 1.485),
            (3.050, 58.905, 0.425, 3.017),
            (3.599, 35.077, 1.332, 2.593),
            (4.709, 12.440, 1.129, 2.013),
            (5.216, 7.692, 0.928, 2.324),
            (6.399, 54.873, 3.381, 2.948),
        ),
    ),
    "B": _ReferenceSpectrum(
        0.40,
        0.90,
        1.374,
        _peaks(
            (1.311, 20.119, 0.292, 1.254),
            (1.504, 17.816, 0.636, 1.430),
            (2.769, 17.292, 2.088, 1.366),
            (3.008, 31.278, 0.677, 2.272),
            (3.336, 4.029, 0.860, 1.746),
            (3.662, 91.535, 0.901, 3.302),
            (4.919, 7.514, 1.019, 1.436),
            (5.511, 7.366, 0.727, 2.970),
            (6.588, 40.931, 3.189, 2.367),
        ),
    ),
    "C": _ReferenceSpectrum(
        0.63,
        0.90,
        1.302,
        _peaks(
            (1.445, 19.547, 0.197, 1.397),
            (1.523, 33.222, 0.813, 1.500),
            (2.619, 21.356, 1.695, 1.801),
            (3.083, 55.039, 0.871, 2.395),
            (3.212, 4.694, 0.648, 2.356),
            (3.787, 80.262, 1.087, 3.274),
            (4.962, 9.025, 1.092, 1.637),
            (5.638, 3.579, 0.754, 1.708),
            (6.655, 51.707, 3.160, 2.939),
        ),
    ),
    "D": _ReferenceSpectrum(
        1.00,
        0.90,
        1.202,
        _peaks(
            (1.713, 13.966, 0.252, 1.626),
            (1.855, 34.567, 1.162, 1.728),
            (3.024, 22.058, 0.946, 2.098),
            (3.227, 41.486, 0.487, 2.678),
            (3.582, 18.203, 0.733, 2.659),
            (3.970, 107.550, 0.955, 3.441),
            (5.104, 14.859, 1.286, 1.946),
            (5.759, 55.933, 0.592, 4.973),
            (6.729, 51.600, 3.063, 3.245),
        ),
    ),
    "E": _ReferenceSpectrum(
        0.40,
        1.00,
        1.236,
        _peaks(
            (1.268, 29.034, 0.302, 1.237),
            (1.453, 30.033, 0.583, 1.429),
            (2.449, 44.324, 2.837, 1.647),
            (3.005, 22.159, 0.676, 2.136),
            (3.162, 3.116, 1.127, 1.247),
            (3.696, 47.589, 0.670, 3.312),
            (4.885, 7.360, 1.006, 1.444),
            (5.424, 42.131, 0.702, 4.460),
            (6.896, 35.970, 3.772, 2.286),
        ),
    ),
    "F": _ReferenceSpectrum(
        0.40,
        0.69,
        1.203,
        _peaks(
            (1.355, 15.013, 0.612, 1.287),
            (1.595, 23.516, 1.791, 1.453),
            (2.948, 18.503, 1.439, 1.901),
            (2.986, 21.522, 0.591, 2.395),
            (3.332, 3.420, 0.687, 1.428),
            (3.563, 148.907, 1.126, 3.284),
            (4.950, 7.182, 1.102, 1.415),
            (5.532, 3.674, 0.775, 2.477),
            (6.509, 48.984, 3.519, 2.553),
        ),
    ),
}


def minoura_critical_points(ggi: float, cgi: float) -> np.ndarray:
    """Return Minoura equations (9)-(13) critical-point energies [eV]."""

    x = float(ggi)
    y = float(cgi)
    if not math.isfinite(x) or not 0.0 <= x <= 1.0:
        raise ValueError("ggi must be finite and lie in [0, 1]")
    if not math.isfinite(y) or not 0.0 < y <= 1.0:
        raise ValueError("cgi must be finite and lie in (0, 1]")
    return np.array(
        [
            1.00 + 0.71 * x + 0.34 * (0.90 - y),
            2.94 + 0.39 * x,
            3.71 + 0.49 * x,
            4.71 + 0.44 * x,
            5.24 + 0.64 * x,
        ],
        dtype=float,
    )


def _tauc_lorentz_epsilon2(
    energy_eV: np.ndarray,
    peaks: tuple[_TaucLorentzPeak, ...],
) -> np.ndarray:
    energy = np.asarray(energy_eV, dtype=float)
    result = np.zeros_like(energy)
    for peak in peaks:
        active = energy > peak.gap_eV
        e = energy[active]
        numerator = (
            peak.amplitude_eV
            * peak.broadening_eV
            * peak.peak_eV
            * (e - peak.gap_eV) ** 2
        )
        denominator = e * (
            (e * e - peak.peak_eV**2) ** 2
            + peak.broadening_eV**2 * e * e
        )
        result[active] += numerator / denominator
    return result


def _map_target_to_reference_energy(
    energy_eV: np.ndarray,
    *,
    target_ggi: float,
    target_cgi: float,
    reference: _ReferenceSpectrum,
) -> np.ndarray:
    """Piecewise-linear inverse of Minoura's critical-point energy shift."""

    energy = np.asarray(energy_eV, dtype=float)
    target = minoura_critical_points(target_ggi, target_cgi)
    source = minoura_critical_points(reference.ggi, reference.cgi)
    mapped = np.empty_like(energy)

    below = energy <= target[0]
    above = energy >= target[-1]
    mapped[below] = energy[below] - target[0] + source[0]
    mapped[above] = energy[above] - target[-1] + source[-1]
    for index in range(len(target) - 1):
        inside = (energy > target[index]) & (energy < target[index + 1])
        fraction = (energy[inside] - target[index]) / (
            target[index + 1] - target[index]
        )
        mapped[inside] = source[index] + fraction * (
            source[index + 1] - source[index]
        )
    return mapped


def _bracket(value: float, points: tuple[float, ...]) -> tuple[int, int, float]:
    if value <= points[0]:
        return 0, 0, 0.0
    if value >= points[-1]:
        last = len(points) - 1
        return last, last, 0.0
    hi = int(np.searchsorted(points, value, side="right"))
    lo = hi - 1
    weight = (value - points[lo]) / (points[hi] - points[lo])
    return lo, hi, float(weight)


def _spectrum_at_cgi_anchor(
    energy_eV: np.ndarray,
    *,
    target_ggi: float,
    anchor_cgi: float,
) -> tuple[np.ndarray, float]:
    if anchor_cgi == 0.90:
        names = ("A", "B", "C", "D")
        points = tuple(_REFERENCES[name].ggi for name in names)
        lo, hi, weight = _bracket(target_ggi, points)
        chosen = (names[lo],) if lo == hi else (names[lo], names[hi])
        values = []
        eps_inf = []
        for name in chosen:
            reference = _REFERENCES[name]
            source_energy = _map_target_to_reference_energy(
                energy_eV,
                target_ggi=target_ggi,
                target_cgi=anchor_cgi,
                reference=reference,
            )
            values.append(_tauc_lorentz_epsilon2(source_energy, reference.peaks))
            eps_inf.append(reference.epsilon_infinity)
        if lo == hi:
            return values[0], eps_inf[0]
        return (
            (1.0 - weight) * values[0] + weight * values[1],
            (1.0 - weight) * eps_inf[0] + weight * eps_inf[1],
        )

    name = "E" if anchor_cgi == 1.00 else "F"
    reference = _REFERENCES[name]
    source_energy = _map_target_to_reference_energy(
        energy_eV,
        target_ggi=target_ggi,
        target_cgi=anchor_cgi,
        reference=reference,
    )
    return (
        _tauc_lorentz_epsilon2(source_energy, reference.peaks),
        reference.epsilon_infinity,
    )


def _target_epsilon2_and_infinity(
    energy_eV: np.ndarray,
    ggi: float,
    cgi: float,
) -> tuple[np.ndarray, float]:
    if not 0.0 <= float(ggi) <= 1.0:
        raise ValueError("ggi must lie in [0, 1]")
    if not 0.75 <= float(cgi) <= 1.0:
        raise ValueError("cgi must lie in [0.75, 1.0]")
    if cgi >= 0.90:
        lower, upper = 0.90, 1.00
    else:
        lower, upper = 0.69, 0.90
    weight = (cgi - lower) / (upper - lower)
    low_eps2, low_inf = _spectrum_at_cgi_anchor(
        energy_eV, target_ggi=ggi, anchor_cgi=lower
    )
    high_eps2, high_inf = _spectrum_at_cgi_anchor(
        energy_eV, target_ggi=ggi, anchor_cgi=upper
    )
    return (
        (1.0 - weight) * low_eps2 + weight * high_eps2,
        float((1.0 - weight) * low_inf + weight * high_inf),
    )


@lru_cache(maxsize=32)
def _kk_nodes(order: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Gauss-Legendre nodes for the measured band plus convergent TL tail."""

    # Most structure lies below 8 eV; use the declared order there and half
    # as many nodes for the smooth 8-80 eV oscillator tail.
    roots, weights = np.polynomial.legendre.leggauss(order)
    low_nodes = 4.0 * (roots + 1.0)
    low_weights = 4.0 * weights
    tail_order = max(24, order // 2)
    tail_roots, tail_weights_native = np.polynomial.legendre.leggauss(tail_order)
    tail_nodes = 36.0 * (tail_roots + 1.0) + 8.0
    tail_weights = 36.0 * tail_weights_native
    return (
        np.concatenate((low_nodes, tail_nodes)),
        np.concatenate((low_weights, tail_weights)),
        80.0,
    )


def minoura_dielectric_function(
    energy_eV: np.ndarray,
    ggi: float,
    cgi: float,
    *,
    quadrature_order: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``epsilon1, epsilon2`` for a CIGS composition.

    The Kramers-Kronig principal value is regularized by subtracting its value
    at the pole.  The remaining analytic pole integral is included explicitly.
    """

    energy = np.asarray(energy_eV, dtype=float)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("energy_eV must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("energy_eV must contain finite positive values")
    if np.any(energy < 0.7) or np.any(energy > 6.5):
        raise ValueError("Minoura optical data are restricted to 0.7-6.5 eV")
    if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order:
        raise ValueError("quadrature_order must be an integer")
    if not 48 <= int(quadrature_order) <= 2048:
        raise ValueError("quadrature_order must lie in [48, 2048]")

    epsilon2, epsilon_infinity = _target_epsilon2_and_infinity(
        energy, float(ggi), float(cgi)
    )
    nodes, weights, cutoff = _kk_nodes(int(quadrature_order))
    epsilon2_nodes, _ = _target_epsilon2_and_infinity(
        nodes, float(ggi), float(cgi)
    )
    f_nodes = nodes * epsilon2_nodes
    f_energy = energy * epsilon2
    denominator = nodes[None, :] ** 2 - energy[:, None] ** 2
    numerator = f_nodes[None, :] - f_energy[:, None]
    regular = numerator / denominator

    # Gauss nodes do not normally coincide with a requested energy.  Retain a
    # stable derivative limit for adversarial/custom grids that get very close.
    near = np.abs(denominator) <= 64.0 * np.finfo(float).eps * (
        nodes[None, :] ** 2 + energy[:, None] ** 2
    )
    if np.any(near):
        h = np.maximum(1e-7, energy * 1e-6)
        plus, _ = _target_epsilon2_and_infinity(energy + h, ggi, cgi)
        minus, _ = _target_epsilon2_and_infinity(energy - h, ggi, cgi)
        derivative = (
            (energy + h) * plus - (energy - h) * minus
        ) / (2.0 * h)
        limit = derivative / (2.0 * energy)
        rows, columns = np.nonzero(near)
        regular[rows, columns] = limit[rows]

    integral = regular @ weights
    pole_integral = 0.5 * epsilon2 * np.log(
        np.abs((cutoff - energy) / (cutoff + energy))
    )
    epsilon1 = epsilon_infinity + (2.0 / np.pi) * (
        integral + pole_integral
    )
    if not np.all(np.isfinite(epsilon1)) or not np.all(np.isfinite(epsilon2)):
        raise FloatingPointError("non-finite CIGS dielectric function")
    return epsilon1, epsilon2


def minoura_nk(
    wavelengths_nm: np.ndarray,
    ggi: float,
    cgi: float,
    *,
    quadrature_order: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return composition-dependent ``n(lambda), k(lambda)``."""

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError(
            "wavelengths_nm must be a non-empty one-dimensional array"
        )
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_nm must contain finite positive values")
    if np.any(wavelengths < MIN_WAVELENGTH_NM) or np.any(
        wavelengths > MAX_WAVELENGTH_NM
    ):
        raise ValueError(
            "requested wavelengths are outside the Minoura 0.7-6.5 eV "
            f"domain [{MIN_WAVELENGTH_NM:.3f}, {MAX_WAVELENGTH_NM:.3f}] nm"
        )
    energy = HC_EV_NM / wavelengths
    epsilon1, epsilon2 = minoura_dielectric_function(
        energy,
        ggi,
        cgi,
        quadrature_order=quadrature_order,
    )
    magnitude = np.hypot(epsilon1, epsilon2)
    n = np.sqrt(np.maximum(0.0, 0.5 * (magnitude + epsilon1)))
    k = np.sqrt(np.maximum(0.0, 0.5 * (magnitude - epsilon1)))
    return n, k


def carron_absorption_coefficient(
    energy_eV: np.ndarray,
    ggi: float,
    cgi: float,
) -> np.ndarray:
    """Carron equations (2)-(6) absorption coefficient [m^-1]."""

    energy = np.asarray(energy_eV, dtype=float)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("energy_eV must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("energy_eV must contain finite positive values")
    if np.any(energy > 2.5):
        raise ValueError("Carron absorption model is validated only through 2.5 eV")
    if not 0.0 <= float(ggi) <= 1.0:
        raise ValueError("ggi must lie in [0, 1]")
    if not 0.75 <= float(cgi) <= 1.0:
        raise ValueError("Carron absorption model requires cgi in [0.75, 1]")

    theta = 0.2076
    q_comp = ggi * math.sin(theta) + cgi * math.cos(theta)
    amplitude = (
        80311.0 * q_comp
        + 427633.0 * (1.0 - q_comp)
        - 596825.0 * q_comp * (1.0 - q_comp)
    )
    gap = carron_band_gap_eV(ggi)
    urbach = 0.025
    connection_1 = 0.25 * (
        2.0 * gap
        - urbach
        + math.sqrt(4.0 * gap**2 + 12.0 * gap * urbach + urbach**2)
    )
    prefactor = (
        amplitude
        * math.sqrt(connection_1 - gap)
        / (connection_1 * math.exp((connection_1 - gap) / urbach))
    )
    low = prefactor * np.exp((energy - gap) / urbach)
    parabolic = amplitude / energy * np.sqrt(np.maximum(energy - gap, 0.0))
    near_edge = np.where(energy < connection_1, low, parabolic)

    exponent = 5
    high_amplitude = 1.8e3 * cgi * (0.5 + 0.5 * ggi)
    connection_2 = gap + (
        (2.0**exponent) * (exponent**exponent) * high_amplitude / amplitude
    ) ** (-2.0 / (2.0 * exponent - 1.0))
    shift = (
        amplitude / high_amplitude * math.sqrt(connection_2 - gap)
    ) ** (1.0 / exponent)
    high = high_amplitude / energy * (
        energy - connection_2 + shift
    ) ** exponent
    alpha_cm = np.where(energy <= connection_2, near_edge, high)
    return alpha_cm * 100.0


def carron_band_gap_eV(ggi: float) -> float:
    """Carron equation (4) CIGS optical gap [eV] for a Ga fraction."""

    value = float(ggi)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("ggi must be finite and lie in [0, 1]")
    return 1.004 * (1.0 - value) + 1.663 * value - 0.033 * value * (1.0 - value)


def ggi_profile_from_coordinate(
    composition_coordinate: np.ndarray,
    model: CIGSGradedOptics,
) -> np.ndarray:
    """Map the shared material coordinate to the physical GGI profile."""

    coordinate = np.asarray(composition_coordinate, dtype=float)
    if not np.all(np.isfinite(coordinate)) or np.any(coordinate < 0.0) or np.any(
        coordinate > 1.0
    ):
        raise ValueError("composition_coordinate must be finite and lie in [0, 1]")
    if model.ggi_front == model.ggi_back:
        return np.full_like(coordinate, model.ggi_front)
    return (1.0 - coordinate) * model.ggi_front + coordinate * model.ggi_back

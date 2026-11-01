"""Directed-rounding nonlinear E/I enclosures for the independent Radau route.

This is an enclosure of the real arithmetic expressions of the pinned static
physics, with compiled binary64 coefficients taken as exact input data. It is
not another time integrator and does not call BDF. Radau's stored polynomial
is evaluated independently, so its floating stepping/reconstruction errors
enter the enclosed defect rather than being presumed zero.

Only the declared four-reservoir electronic / confined positive-ion cases
are implemented. A tube crossing a discontinuous Bernoulli approximation
boundary is refused by the original point path. The opt-in branch enclosure
uses an analytic nominal Bernoulli law and carries an explicit bound on the
retained value-law difference as a separate forcing uncertainty. Unhandled
clipping is still refused. Full protocol qualification and uncertainty of
the prepared initial physical state remain separate admission requirements.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from importlib import import_module
import math
from types import ModuleType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.equations.assembly.terms import OperatingInputs
from solarlab.physics.ions import DIFFUSION_ONLY_STERIC_V1
from solarlab.solvers.time.radau_reference import (
    InventoryDAE,
    InventoryPoint,
    RadauControls,
    RadauCounters,
    RadauFailure,
    RadauTrajectory,
    assembled_inventory_dae,
)

__all__ = [
    "EnclosureBlocked",
    "NonlinearInventoryEnclosure",
    "NonlinearTubeResult",
    "PWLDrive",
    "enclosed_inventory_dae",
    "nonlinear_tube_bound",
    "BernoulliBranchEnclosure",
    "bernoulli_branch_enclosure",
    "PolynomialPath",
    "PositiveTubeResult",
    "positive_polynomial_tube",
    "weighted_logarithmic_norm_upper",
    "WeightedTubeResult",
    "weighted_positive_polynomial_tube",
]

Vector: TypeAlias = NDArray[np.float64]
Ball: TypeAlias = Any
PRECISION = 192


@cache
def _flint() -> ModuleType:
    """Load optional Arb arithmetic only when a reference operation needs it."""
    try:
        return import_module("flint")
    except ImportError as error:
        raise ImportError(
            "nonlinear Radau enclosures require the optional python-flint backend "
            "(install solarlab[arb])"
        ) from error


class EnclosureBlocked(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(code + (": " + detail if detail else ""))
        self.code, self.detail = code, detail


def _q(value: float | int | Fraction) -> Ball:
    fraction = value if isinstance(value, Fraction) else Fraction(value)
    return _flint().arb(_flint().fmpq(fraction.numerator, fraction.denominator))


def _upper(value: Ball) -> float:
    if not value.is_finite():
        raise EnclosureBlocked("nonfinite_enclosure")
    if value.is_zero():
        return 0.0
    result = float(value.upper())
    if not math.isfinite(result):
        raise EnclosureBlocked("enclosure_not_representable_in_binary64")
    return math.nextafter(result, math.inf)


def _box(center: Ball, radius: Ball) -> Ball:
    return (center - radius).union(center + radius)


@dataclass(frozen=True)
class _Jet:
    value: Ball
    gradient: tuple[Ball, ...] = ()

    def __add__(self, other: _Jet | int) -> _Jet:
        rhs = other if isinstance(other, _Jet) else _Jet(_q(other))
        gradient = (
            tuple(a + b for a, b in zip(self.gradient, rhs.gradient))
            if self.gradient and rhs.gradient
            else self.gradient or rhs.gradient
        )
        return _Jet(self.value + rhs.value, gradient)

    __radd__ = __add__

    def __neg__(self) -> _Jet:
        return _Jet(-self.value, tuple(-x for x in self.gradient))

    def __sub__(self, other: _Jet | int) -> _Jet:
        return self + -(other if isinstance(other, _Jet) else _Jet(_q(other)))

    def __rsub__(self, other: int) -> _Jet:
        return -self + other

    def __mul__(self, other: _Jet | int) -> _Jet:
        rhs = other if isinstance(other, _Jet) else _Jet(_q(other))
        if self.gradient and rhs.gradient:
            derivative = tuple(
                a * rhs.value + self.value * b
                for a, b in zip(self.gradient, rhs.gradient)
            )
        elif self.gradient:
            derivative = tuple(a * rhs.value for a in self.gradient)
        else:
            derivative = tuple(self.value * b for b in rhs.gradient)
        return _Jet(self.value * rhs.value, derivative)

    __rmul__ = __mul__

    def reciprocal(self) -> _Jet:
        if self.value.contains(0):
            raise EnclosureBlocked("interval_denominator_contains_zero")
        value = 1 / self.value
        return _Jet(value, tuple(-x * value * value for x in self.gradient))

    def __truediv__(self, other: _Jet | int) -> _Jet:
        return (
            self * (other if isinstance(other, _Jet) else _Jet(_q(other))).reciprocal()
        )

    def exp(self) -> _Jet:
        value = self.value.exp()
        return _Jet(value, tuple(value * x for x in self.gradient))

    def expm1(self) -> _Jet:
        return _Jet(
            self.value.expm1(), tuple(self.value.exp() * x for x in self.gradient)
        )

    def log(self) -> _Jet:
        if not self.value > 0:
            raise EnclosureBlocked("log_domain_not_positive")
        return _Jet(self.value.log(), tuple(x / self.value for x in self.gradient))


def _constant(value: float | int | Ball) -> _Jet:
    return _Jet(value if isinstance(value, _flint().arb) else _q(value))


def _hull(left: _Jet, right: _Jet) -> _Jet:
    size = max(len(left.gradient), len(right.gradient))
    a, b = left.gradient or (_flint().arb(0),) * size, right.gradient or (_flint().arb(0),) * size
    return _Jet(left.value.union(right.value), tuple(x.union(y) for x, y in zip(a, b)))


def _bernoulli(x: _Jet, *, historical: bool = False) -> _Jet:
    limit = _q(1.0e-8)
    if x.value.abs_upper() < limit:
        return 1 - x / 2 + x * x / 12
    if not (x.value >= limit or x.value <= -limit):
        raise EnclosureBlocked("bernoulli_value_branch_crossing", str(x.value))
    if historical:
        if x.value > 700:
            return _constant(0)
        if not x.value <= 700:
            raise EnclosureBlocked("historical_bernoulli_700_crossing", str(x.value))
        return x / x.expm1()
    # The stable paired kernel uses the decaying branch plus B(-a)=a+B(a).
    a = x if x.value > 0 else -x
    small = a * (-a).exp() / -(-a).expm1()
    return small if x.value > 0 else a + small


def _analytic_bernoulli(x: _Jet) -> _Jet:
    """Analytic B including zero, with proved exprel value/tangent tails.

    D=1/B=integral_0^1 exp(t*x) dt. For |x|<=r<=1, the degree-24
    series for D has remainder <=exp(r)*r^25/26!; its differentiated
    remainder is <=exp(r)*r^24/(24!*26).
    """
    radius = x.value.abs_upper()
    if radius <= 1:
        degree = 24
        coefficients = [
            _q(Fraction(1, math.factorial(k + 1))) for k in range(degree + 1)
        ]
        value, derivative = coefficients[-1], coefficients[-1] * degree
        for k in range(degree - 1, -1, -1):
            value = value * x.value + coefficients[k]
        for k in range(degree - 1, 0, -1):
            derivative = derivative * x.value + k * coefficients[k]
        tail = radius.exp() * radius ** (degree + 1) / math.factorial(degree + 2)
        tangent_tail = (
            radius.exp() * radius**degree / (math.factorial(degree) * (degree + 2))
        )
        denominator = _box(value, tail.upper())
        slope = _box(derivative, tangent_tail.upper())
        return _Jet(denominator, tuple(slope * v for v in x.gradient)).reciprocal()
    if x.value > 0 or x.value < 0:
        a = x if x.value > 0 else -x
        small = a * (-a).exp() / -(-a).expm1()
        return small if x.value > 0 else a + small
    # D'>0 implies B'<0; differentiating B(-x)=x+B(x) also gives B'>-1.
    # Monotonicity therefore provides a broad but rigorous wide-box fallback.
    left = _analytic_bernoulli(_Jet(x.value.lower())).value
    right = _analytic_bernoulli(_Jet(x.value.upper())).value
    slope = _flint().arb(-1).union(_flint().arb(0))
    return _Jet(left.union(right), tuple(slope * v for v in x.gradient))


def _quadratic_bernoulli_gap(radius: Ball) -> Ball:
    """Bound |Q-B| for |x|<=radius, Q=1-x/2+x^2/12.

    Multiply Q by the degree-8 exprel polynomial in exact rationals.
    Coefficients through degree three of Q*D8-1 vanish. Bound the remaining
    polynomial and Q*(D-D8), then divide using D>=exp(-radius). This
    includes the nonzero retained switch jump rather than ignoring it.
    """
    Q = (Fraction(1), Fraction(-1, 2), Fraction(1, 12))
    D = tuple(Fraction(1, math.factorial(k + 1)) for k in range(9))
    coefficients = tuple(
        sum((Q[j] * D[k - j] for j in range(3) if 0 <= k - j < len(D)), Fraction(0))
        - (1 if k == 0 else 0)
        for k in range(11)
    )
    remainder = sum(
        (_q(abs(c)) * radius**k for k, c in enumerate(coefficients)), _flint().arb(0)
    )
    tail = radius.exp() * radius**9 / math.factorial(10)
    q_upper = 1 + radius / 2 + radius * radius / 12
    return ((remainder + q_upper * tail) * radius.exp()).upper()


def _bernoulli_gap(value: Ball, *, historical: bool = False) -> Ball:
    limit, error = _q(1e-8), _flint().arb(0)
    if not (value >= limit or value <= -limit):
        radius = value.abs_upper()
        error = _quadratic_bernoulli_gap(radius if radius <= limit else limit)
    if historical and not value <= 700:
        # On x>700 the retained value is zero. B decreases on the real axis.
        cut = value if value > 700 else _flint().arb(700)
        error = error.union(_analytic_bernoulli(_Jet(cut)).value.abs_upper()).upper()
    return error


@dataclass(frozen=True)
class BernoulliBranchEnclosure:
    analytic_value: Ball
    analytic_derivative: Ball
    retained_value: Ball
    retained_error_upper: Ball


def bernoulli_branch_enclosure(
    argument: Ball, *, historical: bool = False
) -> BernoulliBranchEnclosure:
    """Enclose B/B' and the retained piecewise value with an explicit gap.

    No derivative of the discontinuous retained switch is asserted. Its
    value difference must enter trajectory comparison as forcing error.
    """
    with _flint().ctx.workprec(PRECISION):
        value = argument if isinstance(argument, _flint().arb) else _q(argument)
        if not value.is_finite():
            raise EnclosureBlocked("nonfinite_bernoulli_argument")
        nominal = _analytic_bernoulli(_Jet(value, (_flint().arb(1),)))
        error = _bernoulli_gap(value, historical=historical)
        return BernoulliBranchEnclosure(
            nominal.value, nominal.gradient[0], _box(nominal.value, error), error
        )


def _carrier_branch_error(
    left: _Jet, right: _Jet, step: _Jet, drive: _Jet, k: _Jet
) -> Ball:
    population = _maximum((left.value.abs_upper(), right.value.abs_upper()))
    gap = _maximum((_bernoulli_gap(step.value), _bernoulli_gap(-step.value)))
    decay = -(-drive.value.abs_upper()).expm1()
    return (k.value.abs_upper() * population * decay * gap).abs_upper()


def _carrier_flux(
    left: _Jet, right: _Jet, step: _Jet, drive: _Jet, k: _Jet, *, analytic: bool = False
) -> _Jet:
    if not step.value.abs_upper() <= 500:
        raise EnclosureBlocked("carrier_extreme_product_branch_not_enclosed")

    B = _analytic_bernoulli if analytic else _bernoulli

    def branch(m: _Jet, positive: bool) -> _Jet:
        if positive:
            return k * B(-step) * right * (-m).expm1()
        return -k * B(step) * left * m.expm1()

    if drive.value >= 0:
        return branch(drive, True)
    if drive.value <= 0:
        return branch(drive, False)
    # The two drive branches meet at zero flux; enclose their one-sided
    # derivatives (Clarke hull), not a derivative of a substituted SG law.
    positive = _Jet(_flint().arb(0).union(drive.value.upper()), drive.gradient)
    negative = _Jet(drive.value.lower().union(_flint().arb(0)), drive.gradient)
    return _hull(branch(positive, True), branch(negative, False))


@dataclass(frozen=True)
class PWLDrive:
    """Fixed binary64 knots, interpreted exactly as piecewise-linear data."""

    times_s: tuple[float, ...]
    voltage_V: tuple[float, ...]
    generation: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            len(self.times_s) < 2
            or len(self.voltage_V) != len(self.times_s)
            or len(self.generation) != len(self.times_s)
            or not all(
                math.isfinite(x)
                for row in (self.times_s, self.voltage_V, self.generation)
                for x in row
            )
            or any(b <= a for a, b in zip(self.times_s, self.times_s[1:]))
            or any(x < 0 for x in self.generation)
        ):
            raise ValueError("invalid_frozen_PWL_drive")

    def __call__(self, time_s: float) -> OperatingInputs:
        if not self.times_s[0] <= time_s <= self.times_s[-1]:
            raise ValueError("drive_clock_outside_knots")
        return OperatingInputs(
            float(np.interp(time_s, self.times_s, self.voltage_V)),
            float(np.interp(time_s, self.times_s, self.generation)),
        )

    def enclose(self, left: Fraction, right: Fraction) -> tuple[Ball, Ball]:
        for i, (start, stop) in enumerate(zip(self.times_s, self.times_s[1:])):
            if Fraction(start) <= left <= right <= Fraction(stop):
                fraction = (_q(left) - _q(start)).union(_q(right) - _q(start)) / (
                    _q(stop) - _q(start)
                )
                return tuple(
                    _q(values[i]) + fraction * (_q(values[i + 1]) - _q(values[i]))
                    for values in (self.voltage_V, self.generation)
                )  # type: ignore[return-value]
        raise EnclosureBlocked("tube_crosses_drive_knot_or_jump")


@dataclass(frozen=True)
class _Evaluation:
    rhs: tuple[Ball, ...]
    scaled_jacobian: tuple[tuple[Ball, ...], ...]
    coordinates: tuple[Ball, ...]
    densities: tuple[Ball, ...]
    kernel_error_bound: tuple[Ball, ...] = ()


class NonlinearInventoryEnclosure:
    """Verified physical reconstruction and interval AD of the E/I static law.

    B is eliminated analytically: densities equal inventories/CV measures,
    fixed carrier densities equal their reference populations, and the
    positive Dirichlet Poisson matrix is solved with Arb's verified inverse.
    Coordinate reconstruction uses the actual voltage references and the
    retained shifted finite-site formula, including its stored logit offset.
    """

    def __init__(
        self,
        problem: AssembledDC1DProblem | AssembledIonic1DProblem,
        drive: PWLDrive,
        component_scale: ArrayLike,
        *,
        branch_aware: bool = False,
    ):
        if not isinstance(problem, (AssembledDC1DProblem, AssembledIonic1DProblem)):
            raise EnclosureBlocked("unsupported_assembled_problem")
        self.problem, self.drive = problem, drive
        if type(branch_aware) is not bool:
            raise ValueError("branch_aware_must_be_boolean")
        self.branch_aware = branch_aware
        self.ion = (
            problem.device if isinstance(problem, AssembledIonic1DProblem) else None
        )
        self.device = (
            problem.device
            if isinstance(problem, AssembledDC1DProblem)
            else problem.device.electronic_component
        )
        e = self.device
        if (
            e.field_transport is not None
            or e.additional_recombination is not None
            or e.recombination.explicit_defects is not None
            or any(channel.kind != "fixed_reservoir" for channel in e.channels)
        ):
            raise EnclosureBlocked("unsupported_E_physics_or_contacts")
        if self.ion is not None and (
            self.ion.ion_species_names != ("positive",)
            or self.ion.ion_site_contract != "independent_finite_sites"
            or self.ion.steric_model != DIFFUSION_ONLY_STERIC_V1
            or np.any(self.ion.positive_ion_diffusivity_faces_m2_s <= 0)
            or self.ion.ion_node_indices[0] == 0
            or self.ion.ion_node_indices[-1] == e.node_count - 1
        ):
            raise EnclosureBlocked("unsupported_I_scope")
        self.n = e.node_count
        self.size = 2 * (self.n - 2) + (
            0 if self.ion is None else self.ion.ion_node_count
        )
        self.scale = np.array(component_scale, dtype=float, copy=True)
        if (
            self.scale.shape != (self.size,)
            or np.any(self.scale <= 0)
            or not np.all(np.isfinite(self.scale))
        ):
            raise ValueError("positive_component_scales_required")
        self.scale.setflags(write=False)
        self.evaluations = 0
        with _flint().ctx.workprec(PRECISION):
            C = [_q(x) for x in e.poisson.face_capacitance_F_m2]
            matrix = [[_flint().arb(0) for _ in range(self.n - 2)] for _ in range(self.n - 2)]
            for i in range(self.n - 2):
                matrix[i][i] = C[i] + C[i + 1]
                if i:
                    matrix[i][i - 1] = -C[i]
                if i < self.n - 3:
                    matrix[i][i + 1] = -C[i + 1]
            try:
                self.poisson_inverse = _flint().arb_mat(matrix).inv()
            except ZeroDivisionError as error:
                raise EnclosureBlocked("verified_poisson_inverse_failed") from error
            if not all(x.is_finite() for x in self.poisson_inverse.entries()):
                raise EnclosureBlocked("verified_poisson_inverse_nonfinite")

    def evaluate(
        self,
        left: Fraction,
        right: Fraction,
        inventories: tuple[Ball, ...],
        *,
        derivatives: bool = True,
    ) -> _Evaluation:
        with _flint().ctx.workprec(PRECISION):
            self.evaluations += 1
            if len(inventories) != self.size or not all(
                z.is_finite() and z > 0 for z in inventories
            ):
                raise EnclosureBlocked("inventory_tube_not_positive")
            size = self.size if derivatives else 0
            z = []
            for j, value in enumerate(inventories):
                gradient = tuple(
                    _q(self.scale[j]) if i == j else _flint().arb(0) for i in range(size)
                )
                z.append(_Jet(value, gradient))
            voltage, light = self.drive.enclose(left, right)
            e, count = self.device, self.n
            W = [_constant(x) for x in e.geometry.cell_measures_m]
            n = [_constant(e.reference_electron_density_m3[0])]
            p = [_constant(e.reference_hole_density_m3[0])]
            n += [z[i - 1] / W[i] for i in range(1, count - 1)]
            p += [z[count - 2 + i - 1] / W[i] for i in range(1, count - 1)]
            n.append(_constant(e.reference_electron_density_m3[-1]))
            p.append(_constant(e.reference_hole_density_m3[-1]))
            populations: list[_Jet] = []
            if self.ion is not None:
                ion = self.ion
                populations = [
                    z[2 * (count - 2) + i] / _constant(w)
                    for i, w in enumerate(ion.ion_geometry.cell_measures_m)
                ]
                if not all(
                    value.value < _q(cap)
                    for value, cap in zip(populations, ion.positive_ion_site_limit_m3)
                ):
                    raise EnclosureBlocked("ion_tube_reaches_site_capacity")
            rho = [
                p[i]
                - n[i]
                - _constant(e.parameters.acceptor_density_m3[i])
                + _constant(e.parameters.donor_density_m3[i])
                for i in range(count)
            ]
            if self.ion is not None:
                for j, i in enumerate(self.ion.ion_node_indices):
                    rho[i] = rho[i] + _constant(self.ion.ion_charge_weight_ratio[j]) * (
                        populations[j]
                        - _constant(self.ion.positive_ion_reference_m3[j])
                    )
            rho = [_constant(ELEMENTARY_CHARGE_C) * value for value in rho]
            left_phi = _constant(e.phi_left_V)
            right_phi = (
                left_phi
                + _constant(e.built_in_potential_V)
                - _constant(e.voltage_polarity) * _constant(voltage)
            )
            poisson_rhs = [rho[i] * W[i] for i in range(1, count - 1)]
            poisson_rhs[0] = (
                poisson_rhs[0]
                + _constant(e.poisson.face_capacitance_F_m2[0]) * left_phi
            )
            poisson_rhs[-1] = (
                poisson_rhs[-1]
                + _constant(e.poisson.face_capacitance_F_m2[-1]) * right_phi
            )
            rhs_matrix = _flint().arb_mat(
                [
                    [value.value, *(value.gradient or (_flint().arb(0),) * size)]
                    for value in poisson_rhs
                ]
            )
            solved = self.poisson_inverse * rhs_matrix
            phi = (
                [left_phi]
                + [
                    _Jet(solved[i, 0], tuple(solved[i, j + 1] for j in range(size)))
                    for i in range(count - 2)
                ]
                + [right_phi]
            )
            vt = _constant(e.thermal_voltage_V)
            bias = _constant(e.voltage_polarity) * _constant(voltage)
            offsets = (
                (bias, _constant(0)) if e.voltage_polarity > 0 else (_constant(0), bias)
            )
            qn, qp = [], []
            for i in range(count):
                reference = _constant(e.reference_potential_V[i])
                qn.append(
                    (n[i] / _constant(e.reference_electron_density_m3[i])).log()
                    - (phi[i] - reference + offsets[0]) / vt
                )
                qp.append(
                    -(p[i] / _constant(e.reference_hole_density_m3[i])).log()
                    - (phi[i] - reference + offsets[1]) / vt
                )
            bands = e.parameters.band_structure
            electron_potential = [
                phi[i] + _constant(bands.transport_electron_affinity_eV[i])
                for i in range(count)
            ]
            hole_potential = [
                electron_potential[i] + _constant(bands.transport_band_gap_eV[i])
                for i in range(count)
            ]
            fn, fp = [], []
            fn_error: list[Ball] = []
            fp_error: list[Ball] = []
            for i, spacing in enumerate(np.diff(e.nodes_m)):
                en = -(electron_potential[i + 1] - electron_potential[i]) / vt
                ep = (hole_potential[i + 1] - hole_potential[i]) / vt
                fn.append(
                    _carrier_flux(
                        n[i],
                        n[i + 1],
                        en,
                        qn[i + 1] - qn[i],
                        _constant(e.electron_diffusivity_faces_m2_s[i])
                        / _constant(spacing),
                        analytic=self.branch_aware,
                    )
                )
                fp.append(
                    _carrier_flux(
                        p[i],
                        p[i + 1],
                        ep,
                        qp[i] - qp[i + 1],
                        _constant(e.hole_diffusivity_faces_m2_s[i])
                        / _constant(spacing),
                        analytic=self.branch_aware,
                    )
                )
                fn_error.append(
                    _carrier_branch_error(
                        n[i],
                        n[i + 1],
                        en,
                        qn[i + 1] - qn[i],
                        _constant(e.electron_diffusivity_faces_m2_s[i])
                        / _constant(spacing),
                    )
                    if self.branch_aware
                    else _flint().arb(0)
                )
                fp_error.append(
                    _carrier_branch_error(
                        p[i],
                        p[i + 1],
                        ep,
                        qp[i] - qp[i + 1],
                        _constant(e.hole_diffusivity_faces_m2_s[i])
                        / _constant(spacing),
                    )
                    if self.branch_aware
                    else _flint().arb(0)
                )
            coefficients = e.recombination.coefficients
            source = []
            for i in range(count):
                excess = n[i] * p[i] - _constant(coefficients.ni_sq_m6[i])
                if math.isinf(coefficients.tau_n_s[i]) or math.isinf(
                    coefficients.tau_p_s[i]
                ):
                    srh = _constant(0)
                else:
                    denominator = _constant(coefficients.tau_p_s[i]) * (
                        n[i] + _constant(coefficients.n1_m3[i])
                    ) + _constant(coefficients.tau_n_s[i]) * (
                        p[i] + _constant(coefficients.p1_m3[i])
                    )
                    if not denominator.value > 0:
                        raise EnclosureBlocked("SRH_denominator_not_positive")
                    srh = excess / denominator
                radiative = _constant(coefficients.radiative_m3_s[i]) * excess
                auger = (
                    _constant(coefficients.auger_n_m6_s[i]) * n[i]
                    + _constant(coefficients.auger_p_m6_s[i]) * p[i]
                ) * excess
                source.append(
                    (
                        _constant(light) * _constant(e.generation.source_m3_s[i])
                        - (srh + radiative + auger)
                    )
                    * W[i]
                )
            residual = [fn[i - 1] - fn[i] + source[i] for i in range(1, count - 1)]
            residual += [fp[i - 1] - fp[i] + source[i] for i in range(1, count - 1)]
            kernel_error = [fn_error[i - 1] + fn_error[i] for i in range(1, count - 1)]
            kernel_error += [fp_error[i - 1] + fp_error[i] for i in range(1, count - 1)]
            ion_coordinates: list[Ball] = []
            if self.ion is not None:
                ion = self.ion
                mu = []
                for value, cap in zip(populations, ion.positive_ion_site_limit_m3):
                    occupancy = value / _constant(cap)
                    if not occupancy.value < _q(0.999999):
                        raise EnclosureBlocked("ion_occupancy_clip_branch_not_enclosed")
                    mu.append(-(1 - occupancy).log())
                fluxes = [_constant(0)]
                flux_errors = [_flint().arb(0)]
                for i, spacing in enumerate(np.diff(ion.ion_geometry.nodes_m)):
                    xi = (
                        (
                            phi[ion.ion_node_indices[i + 1]]
                            - phi[ion.ion_node_indices[i]]
                        )
                        / vt
                        + mu[i + 1]
                        - mu[i]
                    )
                    fluxes.append(
                        _constant(ion.positive_ion_diffusivity_faces_m2_s[i])
                        / _constant(spacing)
                        * (
                            (
                                _analytic_bernoulli(xi)
                                if self.branch_aware
                                else _bernoulli(xi, historical=True)
                            )
                            * populations[i]
                            - (
                                _analytic_bernoulli(-xi)
                                if self.branch_aware
                                else _bernoulli(-xi, historical=True)
                            )
                            * populations[i + 1]
                        )
                    )
                    flux_errors.append(
                        (
                            _q(ion.positive_ion_diffusivity_faces_m2_s[i])
                            / _q(spacing)
                            * (
                                populations[i].value.abs_upper()
                                * _bernoulli_gap(xi.value, historical=True)
                                + populations[i + 1].value.abs_upper()
                                * _bernoulli_gap(-xi.value, historical=True)
                            )
                        ).abs_upper()
                        if self.branch_aware
                        else _flint().arb(0)
                    )
                fluxes.append(_constant(0))
                flux_errors.append(_flint().arb(0))
                residual += [a - b for a, b in zip(fluxes, fluxes[1:])]
                kernel_error += [a + b for a, b in zip(flux_errors, flux_errors[1:])]
                for value, pref, cap, logit in zip(
                    populations,
                    ion.positive_ion_reference_m3,
                    ion.positive_ion_site_limit_m3,
                    ion.positive_ion_logit_reference,
                ):
                    P, P0, L, a = value.value, _q(pref), _q(cap), _q(logit).exp()
                    r = P0 / L
                    v, w = (P - P0) / (L * (1 - r)), (P0 - P) / (L * r)
                    positive = ((a + v) / (a * (1 - v))).log()
                    negative = ((1 - w) / (1 + a * w)).log()
                    coordinate = (
                        positive
                        if P >= P0
                        else negative
                        if P <= P0
                        else positive.union(negative)
                    )
                    if not coordinate.is_finite():
                        raise EnclosureBlocked("finite_site_inverse_not_enclosed")
                    ion_coordinates.append(coordinate)
            jacobian = tuple(
                tuple(
                    value / _q(self.scale[i])
                    for value in (row.gradient or (_flint().arb(0),) * size)
                )
                for i, row in enumerate(residual)
            )
            coordinates = tuple(value.value for value in (*phi, *qn, *qp)) + tuple(
                ion_coordinates
            )
            values = tuple(row.value for row in residual)
            if not all(
                x.is_finite()
                for x in (*values, *coordinates, *(x for row in jacobian for x in row))
            ):
                raise EnclosureBlocked("nonfinite_nonlinear_enclosure")
            return _Evaluation(
                values,
                jacobian,
                coordinates,
                tuple(x.value for x in (*n, *p, *populations)),
                tuple(value.abs_upper() for value in kernel_error),
            )


class _EnclosedInventoryDAE(InventoryDAE):
    """Explicit precision adapter; the existing Radau stepping remains intact.

    A small physical-inventory correction must still change f even if it is
    below the floating reconstruction stopping tolerance. This adapter obtains
    U and f independently from the verified physical map, then requires the
    original static F_A/S_D residual checks at exactly the same tolerance.
    It does not classify its midpoint values themselves as certified output;
    a trajectory still needs the separate whole-interval tube proof.
    """

    def __init__(self, base: InventoryDAE, enclosure: NonlinearInventoryEnclosure):
        super().__init__(
            base.layout,
            base.evaluate,
            base.differential_rows,
            base.inventory_scale,
            base.algebraic_scale,
            base.inventory_capacity,
        )
        object.__setattr__(self, "enclosure", enclosure)

    enclosure: NonlinearInventoryEnclosure

    def reconstruct(
        self,
        time_s: float,
        inventory: ArrayLike,
        guess: ArrayLike,
        controls: RadauControls,
        counters: RadauCounters,
        *,
        derivatives: bool = False,
    ) -> InventoryPoint:
        z = np.array(inventory, dtype=float, copy=True)
        if (
            not math.isfinite(time_s)
            or z.shape != self.inventory_scale.shape
            or not np.all(np.isfinite(z))
            or np.asarray(guess).shape != (self.layout.size,)
        ):
            raise ValueError("invalid_enclosed_reconstruction_input")
        if np.any(z <= 0) or (
            self.inventory_capacity is not None and np.any(z >= self.inventory_capacity)
        ):
            raise FloatingPointError("radau_inventory_outside_physical_domain")
        counters.reconstructions += 1
        with _flint().ctx.workprec(PRECISION):
            try:
                enclosed = self.enclosure.evaluate(
                    Fraction(time_s),
                    Fraction(time_s),
                    tuple(_q(x) for x in z),
                    derivatives=derivatives,
                )
            except EnclosureBlocked as error:
                raise RadauFailure(
                    "radau_enclosed_reconstruction_blocked", time_s, z
                ) from error
            u = np.asarray([float(x.mid()) for x in enclosed.coordinates])
            rhs = np.asarray([float(x.mid()) for x in enclosed.rhs])
            if not np.all(np.isfinite(u)) or not np.all(np.isfinite(rhs)):
                raise RadauFailure("radau_enclosed_midpoint_nonfinite", time_s, z)
            jacobian = None
            if derivatives:
                jacobian = np.asarray(
                    [
                        [
                            float(
                                (
                                    x
                                    * _q(self.inventory_scale[i])
                                    / _q(self.inventory_scale[j])
                                ).mid()
                            )
                            for j, x in enumerate(row)
                        ]
                        for i, row in enumerate(enclosed.scaled_jacobian)
                    ]
                )
        static = self.assembled(time_s, u, counters)
        assert static.S is not None
        storage_error = static.S[list(self.differential_rows)] - z
        algebraic_error = static.F[list(self.algebraic_rows)]
        norm = float(
            np.max(
                np.abs(
                    np.r_[
                        storage_error / self.inventory_scale,
                        algebraic_error / self.algebraic_scale,
                    ]
                )
            )
        )
        if norm > controls.scaled_newton_tolerance:
            raise RadauFailure("radau_enclosed_static_checks_failed", time_s, u)
        for value in (z, u, rhs, storage_error, algebraic_error):
            value.setflags(write=False)
        if jacobian is not None:
            if not np.all(np.isfinite(jacobian)):
                raise RadauFailure("radau_enclosed_jacobian_nonfinite", time_s, u)
            jacobian.setflags(write=False)
        return InventoryPoint(
            time_s, z, u, rhs, algebraic_error, storage_error, norm, jacobian
        )


def enclosed_inventory_dae(
    problem: AssembledDC1DProblem | AssembledIonic1DProblem,
    drive: PWLDrive,
    *,
    inventory_scale: ArrayLike,
    algebraic_scale: ArrayLike,
) -> InventoryDAE:
    """Opt-in independent enclosed point evaluation, with unchanged static checks."""
    base = assembled_inventory_dae(
        problem, drive, inventory_scale=inventory_scale, algebraic_scale=algebraic_scale
    )
    return _EnclosedInventoryDAE(
        base, NonlinearInventoryEnclosure(problem, drive, inventory_scale)
    )


def _polynomial_range(
    coefficients: Vector,
    left: Fraction,
    right: Fraction,
    *,
    derivative: bool = False,
    correction: Vector | None = None,
) -> tuple[Ball, ...]:
    degree, dimension = (
        coefficients.shape[0] - 1 - int(derivative),
        coefficients.shape[1],
    )
    a, width = _q(left), _q(right - left)
    result = []
    for i in range(dimension):
        values = []
        for j in range(degree + 1):
            power = j + int(derivative)
            value = _q(coefficients[power, i])
            if correction is not None:
                value += _q(correction[power, i])
            values.append(value * (j + 1) if derivative else value)
        powers = [
            sum(
                (
                    values[j] * math.comb(j, k) * a ** (j - k) * width**k
                    for j in range(k, degree + 1)
                ),
                _flint().arb(0),
            )
            for k in range(degree + 1)
        ]
        bernstein = [
            sum(
                (
                    powers[k] * _q(Fraction(math.comb(j, k), math.comb(degree, k)))
                    for k in range(j + 1)
                ),
                _flint().arb(0),
            )
            for j in range(degree + 1)
        ]
        value = bernstein[0]
        for item in bernstein[1:]:
            value = value.union(item)
        result.append(value)
    return tuple(result)


def _maximum(values: list[Ball] | tuple[Ball, ...]) -> Ball:
    result = values[0]
    for value in values[1:]:
        result = result.union(value)
    return result.upper()


@dataclass(frozen=True)
class NonlinearTubeResult:
    """A local polynomial certificate, never a complete scope certificate.

    ``error_upper`` bounds the exact polynomial encoded by the stored words.
    It does not include floating evaluation at arbitrary output clocks.
    ``reconstruction_error_upper`` separately covers stored endpoint
    coordinates; physical current, charge and protocol certificates are absent.
    """

    status: str
    reason: str
    tube_proved: bool
    error_upper: float | None
    conditional_majorant: float | None
    log_norm_upper: float | None
    defect_upper: float | None
    completed_slabs: int
    evaluation_count: int
    reconstruction_error_upper: tuple[float, ...] = ()
    reference_qualified: bool = False
    error_target: str = "exact_encoded_inventory_polynomial"


def nonlinear_tube_bound(
    enclosure: NonlinearInventoryEnclosure,
    trajectory: RadauTrajectory,
    *,
    initial_error: float,
    tube_radius: float,
    subdivisions: int = 2,
    coefficient_corrections: tuple[Vector, ...] | None = None,
    initial_inventory_correction: ArrayLike | None = None,
) -> NonlinearTubeResult:
    """Sufficient tube test; a failed bootstrap never returns an error bound.

    The returned conditional majorant is diagnostic only if the assumed tube
    fails to close. The exact IVP starts in the supplied scaled initial ball;
    this function cannot establish the provenance of that initial uncertainty.
    Optional correction words are added in Arb before evaluating the initial
    center, polynomial, derivative or jump. They are never added in binary64.
    """
    if getattr(enclosure, "branch_aware", False):
        raise ValueError("branch_aware_enclosure_requires_positive_polynomial_tube")
    if (
        not math.isfinite(initial_error)
        or initial_error < 0
        or not math.isfinite(tube_radius)
        or tube_radius <= 0
        or type(subdivisions) is not int
        or not 1 <= subdivisions <= 16
    ):
        raise ValueError("invalid_nonlinear_tube_controls")
    if coefficient_corrections is not None and (
        len(coefficient_corrections) != len(trajectory.steps)
        or any(
            value.shape != (4, enclosure.size) or not np.all(np.isfinite(value))
            for value in coefficient_corrections
        )
    ):
        raise ValueError("invalid_polynomial_correction_words")
    initial_low = (
        np.zeros(enclosure.size)
        if initial_inventory_correction is None
        else np.asarray(initial_inventory_correction, dtype=float)
    )
    if initial_low.shape != (enclosure.size,) or not np.all(np.isfinite(initial_low)):
        raise ValueError("invalid_initial_inventory_correction_words")
    count = 0
    last_mu = last_defect = conditional = None
    start_count = enclosure.evaluations
    with _flint().ctx.workprec(PRECISION):
        error = _q(initial_error)
        maximum = error
        last_center = tuple(
            _q(x) + _q(low) for x, low in zip(trajectory.initial.inventory, initial_low)
        )
        centers = [last_center]
        previous_time = Fraction(trajectory.initial.time_s)
        try:
            if not trajectory.steps:
                raise EnclosureBlocked("empty_trajectory_has_no_time_tube")
            if trajectory.initial.inventory.shape != (enclosure.size,):
                raise EnclosureBlocked("trajectory_inventory_dimension_mismatch")
            for index, step in enumerate(trajectory.steps):
                low = (
                    None
                    if coefficient_corrections is None
                    else coefficient_corrections[index]
                )
                if Fraction(step.left_s) != previous_time:
                    raise EnclosureBlocked("trajectory_clock_gap")
                if not step.right_s > step.left_s:
                    raise EnclosureBlocked("trajectory_clock_not_increasing")
                if step.coefficients.shape != (4, enclosure.size):
                    raise EnclosureBlocked("trajectory_polynomial_shape_mismatch")
                if not np.all(np.isfinite(step.coefficients)):
                    raise EnclosureBlocked("trajectory_polynomial_nonfinite")
                if (
                    step.endpoint.time_s != step.right_s
                    or step.endpoint.u.shape != trajectory.initial.u.shape
                ):
                    raise EnclosureBlocked("trajectory_endpoint_mismatch")
                jump = _maximum(
                    tuple(
                        (
                            _q(step.coefficients[0, i])
                            + (_flint().arb(0) if low is None else _q(low[0, i]))
                            - last_center[i]
                        ).abs_upper()
                        / _q(enclosure.scale[i])
                        for i in range(enclosure.size)
                    )
                )
                error += jump
                h = Fraction(step.right_s) - Fraction(step.left_s)
                for part in range(subdivisions):
                    a, b = (
                        Fraction(part, subdivisions),
                        Fraction(part + 1, subdivisions),
                    )
                    left, right = (
                        Fraction(step.left_s) + a * h,
                        Fraction(step.left_s) + b * h,
                    )
                    center = _polynomial_range(step.coefficients, a, b, correction=low)
                    tube = tuple(
                        _box(value, _q(tube_radius) * _q(scale))
                        for value, scale in zip(center, enclosure.scale)
                    )
                    tube_evaluation = enclosure.evaluate(left, right, tube)
                    row_bounds = [
                        row[i].upper()
                        + sum(
                            (x.abs_upper() for j, x in enumerate(row) if j != i), _flint().arb(0)
                        )
                        for i, row in enumerate(tube_evaluation.scaled_jacobian)
                    ]
                    mu = row_bounds[0]
                    for value in row_bounds[1:]:
                        mu = mu.union(value)
                    mu = mu.upper()
                    last_mu = _upper(mu)
                    central = enclosure.evaluate(left, right, center, derivatives=False)
                    # Derivative coefficients remain exact input words times
                    # integer powers; divide by the exact binary64 clock gap.
                    derivative = _polynomial_range(
                        step.coefficients, a, b, derivative=True, correction=low
                    )
                    defect = [
                        ((rate / _q(h)) - rhs).abs_upper() / _q(scale)
                        for rate, rhs, scale in zip(
                            derivative, central.rhs, enclosure.scale
                        )
                    ]
                    delta = defect[0]
                    for value in defect[1:]:
                        delta = delta.union(value)
                    delta = delta.upper()
                    last_defect = _upper(delta)
                    width = _q(right - left)
                    if mu * width > 50:
                        raise EnclosureBlocked(
                            "log_norm_propagation_exponent_exceeds_50"
                        )
                    growth = (mu * width).exp()
                    integral = width if mu.is_zero() else (mu * width).expm1() / mu
                    after = (growth * error + integral * delta).upper()
                    majorant = error.union(after).upper()
                    conditional = _upper(majorant)
                    if not majorant < _q(tube_radius):
                        raise EnclosureBlocked("tube_bootstrap_did_not_close")
                    maximum = maximum.union(majorant).upper()
                    error = after
                    count += 1
                last_center = tuple(
                    sum(
                        (
                            _q(step.coefficients[k, i])
                            + (_flint().arb(0) if low is None else _q(low[k, i]))
                            for k in range(4)
                        ),
                        _flint().arb(0),
                    )
                    for i in range(enclosure.size)
                )
                centers.append(last_center)
                previous_time = Fraction(step.right_s)
            # Independent reconstruction/evaluation terms at saved endpoints.
            reconstruction = [0.0] * trajectory.initial.u.size
            for point, center in zip(
                (trajectory.initial, *(step.endpoint for step in trajectory.steps)),
                centers,
            ):
                # p(1) is an exact sum of stored words; point.inventory is a
                # separately rounded Horner output. Include that discrepancy.
                boxes = tuple(
                    _box(value, maximum * _q(scale))
                    for value, scale in zip(center, enclosure.scale)
                )
                exact = enclosure.evaluate(
                    Fraction(point.time_s),
                    Fraction(point.time_s),
                    boxes,
                    derivatives=False,
                )
                reconstruction = [
                    max(old, _upper((value - _q(stored)).abs_upper()))
                    for old, value, stored in zip(
                        reconstruction, exact.coordinates, point.u
                    )
                ]
            return NonlinearTubeResult(
                "reference_blocked",
                "unit_tube_proved_but_scope_gate_initial_and_observation_certificates_pending",
                True,
                _upper(maximum),
                conditional,
                last_mu,
                last_defect,
                count,
                enclosure.evaluations - start_count,
                tuple(reconstruction),
            )
        except EnclosureBlocked as failure:
            return NonlinearTubeResult(
                "reference_blocked",
                str(failure),
                False,
                None,
                conditional,
                last_mu,
                last_defect,
                count,
                enclosure.evaluations - start_count,
            )


@dataclass(frozen=True)
class PolynomialPath:
    """Complete saved split-polynomial data; no time integrator or fake stages."""

    step_times_s: Vector
    coefficients_high: Vector
    coefficients_low: Vector
    initial_high: Vector
    initial_low: Vector
    initial_u: Vector
    endpoint_u: Vector
    original_scale: Vector

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError("saved_polynomial_nonfinite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.coefficients_high.ndim != 3:
            raise ValueError("saved_polynomial_shape_mismatch")
        steps, size = self.coefficients_high.shape[0], self.initial_high.size
        if (
            steps == 0
            or size == 0
            or self.initial_high.shape != (size,)
            or self.initial_low.shape != (size,)
            or self.original_scale.shape != (size,)
            or np.any(self.original_scale <= 0)
            or self.coefficients_high.shape != (steps, 4, size)
            or self.coefficients_low.shape != self.coefficients_high.shape
            or self.step_times_s.shape != (steps, 2)
            or self.initial_u.ndim != 1
            or self.endpoint_u.shape != (steps, self.initial_u.size)
        ):
            raise ValueError("saved_polynomial_shape_mismatch")
        times = self.step_times_s
        if np.any(times[:, 1] <= times[:, 0]) or not np.array_equal(
            times[:-1, 1], times[1:, 0]
        ):
            raise ValueError("saved_polynomial_clock_gap")


@dataclass(frozen=True)
class PositiveTubeResult:
    status: str
    reason: str
    tube_proved: bool
    error_upper: float | None
    log_error_upper: float | None
    completed_slabs: int
    evaluation_count: int
    slabs: tuple[dict[str, Any], ...]
    component_inventory_error_upper: tuple[float, ...] = ()
    reconstruction_error_upper: tuple[float, ...] = ()
    reference_qualified: bool = False
    error_target: str = "encoded_inventory_polynomial_in_original_saved_scale"
    comparison_scope: str = (
        "solutions_of_the_retained_field_enclosed_by_nominal_plus_bounded_forcing"
    )
    target_existence_and_uniqueness_certified: bool = False


def _log_inventory_jacobian(
    evaluation: _Evaluation, inventories: tuple[Ball, ...], scale: Vector
) -> tuple[tuple[Ball, ...], ...]:
    """Jacobian of G(z)/z with respect to log(z), before norm weighting."""
    return tuple(
        tuple(
            value - evaluation.rhs[i] / inventories[i]
            if i == j
            else value
            * (_q(scale[i]) * inventories[j])
            / (_q(scale[j]) * inventories[i])
            for j, value in enumerate(row)
        )
        for i, row in enumerate(evaluation.scaled_jacobian)
    )


def positive_polynomial_tube(
    enclosure: NonlinearInventoryEnclosure,
    path: PolynomialPath,
    *,
    initial_log_error: float,
    log_radius: float,
    subdivisions: int = 2,
    log_weights: ArrayLike | None = None,
) -> PositiveTubeResult:
    """Positive tube z_i=p_i*exp(e_i), with |e_i|<=radius*w_i.

    For an analytic nominal G and retained F=G+eta, the comparison uses
    A_ij=(z_j/z_i)*G_z,ij-delta_ij*G_i/z_i, scaled by w_j/w_i.
    The center defect is (p'-G(t,p))/p; the separately bounded forcing
    eta/z is added once on the full tube. This encloses the retained
    branch values without differentiating their discontinuous switch.
    Existence/uniqueness of the retained target and full reference
    qualification remain separate obligations even if inclusion closes.
    """
    if not getattr(enclosure, "branch_aware", False):
        raise ValueError("positive_tube_requires_branch_aware_field")
    size = enclosure.size
    if path.initial_high.size != size:
        raise ValueError("positive_tube_inventory_dimension_mismatch")
    weights: Vector = (
        np.ones(size) if log_weights is None else np.asarray(log_weights, dtype=float)
    )
    if (
        not math.isfinite(initial_log_error)
        or initial_log_error < 0
        or not math.isfinite(log_radius)
        or log_radius <= 0
        or type(subdivisions) is not int
        or not 1 <= subdivisions <= 16
        or weights.shape != (size,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("invalid_positive_tube_controls")
    start_count = enclosure.evaluations
    slabs: list[dict[str, Any]] = []
    completed = 0
    with _flint().ctx.workprec(PRECISION):
        error, maximum = _q(initial_log_error), _q(initial_log_error)
        centers = [
            tuple(
                _q(hi) + _q(lo) for hi, lo in zip(path.initial_high, path.initial_low)
            )
        ]
        previous = centers[0]
        component_error = [_flint().arb(0) for _ in range(size)]
        try:
            if not all(z > 0 for z in previous):
                raise EnclosureBlocked("initial_inventory_not_positive")
            for index, (start, stop) in enumerate(path.step_times_s):
                high, low = path.coefficients_high[index], path.coefficients_low[index]
                first = tuple(_q(high[0, i]) + _q(low[0, i]) for i in range(size))
                if not all(z > 0 for z in first):
                    raise EnclosureBlocked("saved_polynomial_start_not_positive")
                jump = _maximum(
                    tuple(
                        (before / after).log().abs_upper() / _q(w)
                        for before, after, w in zip(previous, first, weights)
                    )
                )
                error += jump
                h = Fraction(float(stop)) - Fraction(float(start))
                for part in range(subdivisions):
                    a, b = (
                        Fraction(part, subdivisions),
                        Fraction(part + 1, subdivisions),
                    )
                    left, right = (
                        Fraction(float(start)) + a * h,
                        Fraction(float(start)) + b * h,
                    )
                    record: dict[str, Any] = {
                        "step": index,
                        "part": part,
                        "left_s": float(left),
                        "right_s": float(right),
                        "log_radius": log_radius,
                        "center_jump_upper": _upper(jump) if part == 0 else 0.0,
                        "closed": False,
                    }
                    slabs.append(record)
                    center = _polynomial_range(high, a, b, correction=low)
                    if not all(z > 0 for z in center):
                        raise EnclosureBlocked("saved_polynomial_range_not_positive")
                    tube = tuple(
                        value
                        * (
                            (-_q(log_radius) * _q(w))
                            .exp()
                            .union((_q(log_radius) * _q(w)).exp())
                        )
                        for value, w in zip(center, weights)
                    )
                    if not all(z > 0 for z in tube):
                        raise EnclosureBlocked(
                            "multiplicative_tube_not_representably_positive"
                        )
                    record["positive_tube"] = True
                    broad = enclosure.evaluate(left, right, tube)
                    if (
                        len(broad.rhs) != size
                        or len(broad.scaled_jacobian) != size
                        or any(len(row) != size for row in broad.scaled_jacobian)
                    ):
                        raise EnclosureBlocked("nominal_field_shape_mismatch")
                    if len(broad.kernel_error_bound) != size or not all(
                        x.is_finite() and x >= 0 for x in broad.kernel_error_bound
                    ):
                        raise EnclosureBlocked(
                            "missing_or_invalid_retained_kernel_error"
                        )
                    record["nominal_and_retained_field_enclosed"] = True
                    rows: list[Ball] = []
                    for i, row in enumerate(
                        _log_inventory_jacobian(broad, tube, enclosure.scale)
                    ):
                        terms = []
                        for j, value in enumerate(row):
                            term = value * (_q(weights[j]) / _q(weights[i]))
                            terms.append(term.upper() if i == j else term.abs_upper())
                        rows.append(sum(terms, _flint().arb(0)).upper())
                    mu = _maximum(tuple(rows))
                    nominal = enclosure.evaluate(left, right, center, derivatives=False)
                    if len(nominal.rhs) != size:
                        raise EnclosureBlocked("nominal_field_shape_mismatch")
                    derivative = _polynomial_range(
                        high, a, b, derivative=True, correction=low
                    )
                    defect = _maximum(
                        tuple(
                            ((d / _q(h) - f) / p / _q(w)).abs_upper()
                            for d, f, p, w in zip(
                                derivative, nominal.rhs, center, weights
                            )
                        )
                    )
                    perturbation = _maximum(
                        tuple(
                            (eps / z / _q(w)).abs_upper()
                            for eps, z, w in zip(
                                broad.kernel_error_bound, tube, weights
                            )
                        )
                    )
                    width = _q(right - left)
                    record.update(
                        log_norm_upper=_upper(mu),
                        nominal_defect_upper=_upper(defect),
                        retained_kernel_perturbation_upper=_upper(perturbation),
                        propagation_exponent_upper=_upper(mu * width),
                        dominant_log_row_approx=int(
                            np.argmax([float(row) for row in rows])
                        ),
                    )
                    if mu * width > 50:
                        raise EnclosureBlocked(
                            "positive_log_norm_propagation_exponent_exceeds_50"
                        )
                    growth = (mu * width).exp()
                    integral = width if mu.is_zero() else (mu * width).expm1() / mu
                    after = (
                        growth * error + integral * (defect + perturbation)
                    ).upper()
                    majorant = error.union(after).upper()
                    record["conditional_log_majorant"] = _upper(majorant)
                    if not majorant < _q(log_radius):
                        raise EnclosureBlocked("positive_tube_bootstrap_did_not_close")
                    record["closed"] = True
                    completed += 1
                    maximum = maximum.union(majorant).upper()
                    for i, (p, w) in enumerate(zip(center, weights)):
                        local = (p.abs_upper() * (majorant * _q(w)).expm1()).abs_upper()
                        component_error[i] = component_error[i].union(local).upper()
                    error = after
                previous = tuple(
                    sum((_q(high[k, i]) + _q(low[k, i]) for k in range(4)), _flint().arb(0))
                    for i in range(size)
                )
                centers.append(previous)
            reconstruction = [0.0] * path.initial_u.size
            times = [float(path.step_times_s[0, 0]), *path.step_times_s[:, 1].tolist()]
            for time_s, center, stored in zip(
                times, centers, (path.initial_u, *path.endpoint_u)
            ):
                box = tuple(
                    z * ((-maximum * _q(w)).exp().union((maximum * _q(w)).exp()))
                    for z, w in zip(center, weights)
                )
                exact = enclosure.evaluate(
                    Fraction(time_s), Fraction(time_s), box, derivatives=False
                )
                if len(exact.coordinates) != path.initial_u.size:
                    raise EnclosureBlocked("positive_tube_coordinate_shape_mismatch")
                reconstruction = [
                    max(old, _upper((value - _q(word)).abs_upper()))
                    for old, value, word in zip(
                        reconstruction, exact.coordinates, stored
                    )
                ]
            scaled_error = _maximum(
                tuple(
                    value / _q(scale)
                    for value, scale in zip(component_error, path.original_scale)
                )
            )
            return PositiveTubeResult(
                "reference_blocked",
                "positive_comparison_tube_closed_full_reference_certificates_pending",
                True,
                _upper(scaled_error),
                _upper(maximum),
                completed,
                enclosure.evaluations - start_count,
                tuple(slabs),
                tuple(_upper(value) for value in component_error),
                tuple(reconstruction),
            )
        except EnclosureBlocked as failure:
            if slabs:
                slabs[-1]["failure"] = str(failure)
            return PositiveTubeResult(
                "reference_blocked",
                str(failure),
                False,
                None,
                None,
                completed,
                enclosure.evaluations - start_count,
                tuple(slabs),
            )


def _weighted_rows(
    matrix: tuple[tuple[Ball, ...], ...], weights: tuple[Ball, ...]
) -> tuple[Ball, ...]:
    return tuple(
        row[i]
        + sum(
            (value * weights[j] / weights[i] for j, value in enumerate(row) if j != i),
            _flint().arb(0),
        )
        for i, row in enumerate(matrix)
    )


def weighted_logarithmic_norm_upper(
    jacobian: tuple[tuple[Ball, ...], ...], weights: ArrayLike
) -> float:
    """Verified infinity mu(D*J*D^-1), D=diag(1/weights).

    Every entry may enclose a whole time/state interval. The binary64
    positive weights are exact proposal data; no eigensolver is trusted.
    """
    values = np.asarray(weights, dtype=float)
    size = len(jacobian)
    if (
        size == 0
        or any(len(row) != size for row in jacobian)
        or values.shape != (size,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError("invalid_diagonal_norm_inputs")
    with _flint().ctx.workprec(PRECISION):
        matrix = tuple(
            tuple(
                (x if isinstance(x, _flint().arb) else _q(x)).upper()
                if i == j
                else (x if isinstance(x, _flint().arb) else _q(x)).abs_upper()
                for j, x in enumerate(row)
            )
            for i, row in enumerate(jacobian)
        )
        return max(
            _upper(row) for row in _weighted_rows(matrix, tuple(map(_q, values)))
        )


def _diagonal_weight_proposals(
    global_matrix: Vector, first_matrix: Vector, initial: Vector
) -> tuple[tuple[tuple[str, Vector], ...], tuple[dict[str, str], ...]]:
    """Untrusted finite proposals; positivity and all bounds are checked later.

    The inverse-inventory square root corresponds to inventory-weighted
    quadratic log deviations. Perron/resolvent proposals minimize the
    Metzler row-sum envelope, without changing any physical matrix entry.
    Resolvents only condition the proposal calculation; they are not used
    in the propagation certificate.
    """
    candidates = [("uniform", np.ones(initial.size))]
    rejected: list[dict[str, str]] = []
    logs = np.log(initial)
    for power, name in ((0.5, "inverse_inventory_sqrt"), (1.0, "inverse_inventory")):
        candidates.append((name, np.exp(-power * (logs - logs.min()))))
    for label, matrix in (("global", global_matrix), ("first_slab", first_matrix)):
        magnitude = float(np.max(np.abs(matrix)))
        if magnitude == 0.0:
            continue
        scaled = matrix / magnitude
        try:
            eigenvalues, eigenvectors = np.linalg.eig(scaled)
            index = int(np.argmax(eigenvalues.real))
            rho = float(eigenvalues[index].real)
            candidates.append((label + "_perron", np.abs(eigenvectors[:, index])))
            for bits in (8, 16, 24):
                name = label + f"_resolvent_2m{bits}"
                try:
                    vector = np.linalg.solve(
                        (rho + 2.0 ** (-bits)) * np.eye(initial.size) - scaled,
                        np.ones(initial.size),
                    )
                    candidates.append((name, vector))
                except np.linalg.LinAlgError as error:
                    rejected.append({"candidate": name, "reason": str(error)})
        except np.linalg.LinAlgError as error:
            rejected.append({"candidate": label + "_eigen", "reason": str(error)})
    accepted = []
    for name, raw in candidates:
        if not np.all(np.isfinite(raw)) or np.any(raw <= 0):
            rejected.append({"candidate": name, "reason": "nonpositive_or_nonfinite"})
            continue
        value = np.array(raw / np.max(raw), dtype=float)
        if not np.all(np.isfinite(value)) or np.any(value <= 0) or np.any(value > 1):
            rejected.append({"candidate": name, "reason": "invalid_normalization"})
            continue
        value.setflags(write=False)
        accepted.append((name, value))
    return tuple(accepted), tuple(rejected)


@dataclass(frozen=True)
class WeightedTubeResult:
    status: str
    reason: str
    selected_candidate: str | None
    selected_weights: tuple[float, ...]
    best_exponent_upper: float | None
    best_first_slab_exponent_upper: float | None
    enclosed_slabs: int
    evaluation_count: int
    candidates: tuple[dict[str, Any], ...]
    rejected_candidates: tuple[dict[str, str], ...]
    slabs: tuple[dict[str, Any], ...]
    comparison_matrices: Vector
    tube_bound: PositiveTubeResult | None = None
    reference_qualified: bool = False


def weighted_positive_polynomial_tube(
    enclosure: NonlinearInventoryEnclosure,
    path: PolynomialPath,
    *,
    initial_log_error: float,
    log_radius: float,
    subdivisions: int = 2,
) -> WeightedTubeResult:
    """Select constant diagonal weights on verified whole-slab enclosures.

    First enclose each log-Jacobian on the original positive outer tube.
    Its diagonal upper bounds and off-diagonal absolute upper bounds make
    a Metzler majorant. Candidate weights are verified in Arb on every
    such matrix, not on sampled Jacobians. They satisfy max(w)=1, so the
    later weighted tube cannot expand the original physical log radius.
    The selected fixed weights are used for the complete existing tube
    computation only if every certified exponent clears its original cap.
    """
    if (
        not getattr(enclosure, "branch_aware", False)
        or path.initial_high.size != enclosure.size
        or not math.isfinite(initial_log_error)
        or initial_log_error < 0
        or not math.isfinite(log_radius)
        or log_radius <= 0
        or type(subdivisions) is not int
        or not 1 <= subdivisions <= 16
    ):
        raise ValueError("invalid_weighted_tube_controls")
    start_count, size = enclosure.evaluations, enclosure.size
    matrices: list[Vector] = []
    widths: list[Fraction] = []
    slabs: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    rejected: tuple[dict[str, str], ...] = ()
    selected: dict[str, Any] | None = None
    tube_bound: PositiveTubeResult | None = None
    with _flint().ctx.workprec(PRECISION):
        try:
            initial = tuple(
                _q(hi) + _q(lo) for hi, lo in zip(path.initial_high, path.initial_low)
            )
            if not all(z > 0 for z in initial):
                raise EnclosureBlocked("initial_inventory_not_positive")
            factor = (-_q(log_radius)).exp().union(_q(log_radius).exp())
            for step, (start, stop) in enumerate(path.step_times_s):
                h = Fraction(float(stop)) - Fraction(float(start))
                for part in range(subdivisions):
                    a, b = (
                        Fraction(part, subdivisions),
                        Fraction(part + 1, subdivisions),
                    )
                    left = Fraction(float(start)) + a * h
                    right = Fraction(float(start)) + b * h
                    center = _polynomial_range(
                        path.coefficients_high[step],
                        a,
                        b,
                        correction=path.coefficients_low[step],
                    )
                    if not all(z > 0 for z in center):
                        raise EnclosureBlocked("saved_polynomial_range_not_positive")
                    box = tuple(z * factor for z in center)
                    field = enclosure.evaluate(left, right, box)
                    if (
                        len(field.rhs) != size
                        or len(field.scaled_jacobian) != size
                        or any(len(row) != size for row in field.scaled_jacobian)
                    ):
                        raise EnclosureBlocked("nominal_field_shape_mismatch")
                    jacobian = _log_inventory_jacobian(field, box, enclosure.scale)
                    matrix = np.array(
                        [
                            [
                                _upper(value if i == j else value.abs_upper())
                                for j, value in enumerate(row)
                            ]
                            for i, row in enumerate(jacobian)
                        ]
                    )
                    matrix.setflags(write=False)
                    matrices.append(matrix)
                    widths.append(right - left)
                    slabs.append(
                        {
                            "step": step,
                            "part": part,
                            "left_s": float(left),
                            "right_s": float(right),
                        }
                    )
            global_matrix = np.max(np.stack(matrices), axis=0)
            proposals, rejected = _diagonal_weight_proposals(
                global_matrix, matrices[0], np.array([float(z) for z in initial])
            )
            exact_matrices = tuple(
                tuple(tuple(map(_q, row)) for row in matrix) for matrix in matrices
            )
            exact_global = tuple(tuple(map(_q, row)) for row in global_matrix)
            for label, weights in proposals:
                w = tuple(map(_q, weights))
                upper = tuple(
                    max(_upper(row) for row in _weighted_rows(matrix, w))
                    for matrix in exact_matrices
                )
                exponents = tuple(
                    _upper(_q(mu) * _q(width)) for mu, width in zip(upper, widths)
                )
                ratios = _weighted_rows(exact_global, w)
                lower = min(
                    math.nextafter(float(value.lower()), -math.inf) for value in ratios
                )
                global_upper = max(_upper(value) for value in ratios)
                certificates.append(
                    {
                        "candidate": label,
                        "weights": weights.tolist(),
                        "weight_min": float(weights.min()),
                        "weight_max": float(weights.max()),
                        "log_norm_upper_per_slab": upper,
                        "exponent_upper_per_slab": exponents,
                        "worst_exponent_upper": max(exponents),
                        "global_comparison_abscissa_lower": lower,
                        "global_comparison_abscissa_upper": global_upper,
                        "spectral_bracket_scope": "Metzler interval majorant only; not the physical Jacobian",
                    }
                )
            selected = min(certificates, key=lambda c: c["worst_exponent_upper"])
            if selected["worst_exponent_upper"] > 50:
                reason = "weighted_log_norm_propagation_exponent_exceeds_50"
            else:
                tube_bound = positive_polynomial_tube(
                    enclosure,
                    path,
                    initial_log_error=initial_log_error,
                    log_radius=log_radius,
                    subdivisions=subdivisions,
                    log_weights=selected["weights"],
                )
                reason = tube_bound.reason
        except EnclosureBlocked as error:
            reason = str(error)
    packed = np.stack(matrices) if matrices else np.empty((0, size, size))
    packed.setflags(write=False)
    return WeightedTubeResult(
        "reference_blocked",
        reason,
        None if selected is None else selected["candidate"],
        () if selected is None else tuple(selected["weights"]),
        None if selected is None else selected["worst_exponent_upper"],
        None if selected is None else selected["exponent_upper_per_slab"][0],
        len(matrices),
        enclosure.evaluations - start_count,
        tuple(certificates),
        rejected,
        tuple(slabs),
        packed,
        tube_bound,
    )

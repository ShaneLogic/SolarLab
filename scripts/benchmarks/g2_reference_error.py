"""Offline G2-R conditional reference errors over independently read BE data.

E1 uses exact rational comparisons. E3 checks supplied source-bound envelopes
of the public conservative operators; missing physical derivative/box/remainder
evidence stays unknown. Numerical computations never award independent review.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from scripts.benchmarks.contract_prototype import (
    COULOMB,
    ONE,
    PARTICLE,
    SECOND,
    VOLT,
    VOLUME,
    ContractError,
    Layout,
)
from scripts.benchmarks.g2_reference_be import (
    METHOD,
    SCHEMA as BE_SCHEMA,
    Guard,
    Limits,
    RunIdentity,
    canonical,
    digest,
    read_reference_history,
    unpack_point,
)
from scripts.benchmarks.native_history import HistoryReader

SCHEMA = "solarlab.g2-reference-error.v1"
EVIDENCE_SCHEMA = "solarlab.g2-reference-evidence.v1"
METHOD_SCHEMA = "solarlab.g2-reference-method.v1"
CLAIM = "conditional_uncertainty_estimate"
C = Fraction(42167, 25000)
FS = Fraction(5, 4)
AXES = (
    "evaluation",
    "initial_and_algebraic",
    "input_clock_parameter",
    "trajectory_nonlinear",
    "stability_reconstruction",
)
E1_PREMISES = (
    "same_problem",
    "same_observable",
    "known_order",
    "smooth_segments",
    "remainder",
    "non_time_complete",
    "independent_method",
)
E3_PREMISES = (
    "physical_jacobian_enclosure",
    "feasible_box",
    "local_root_argument",
    "residual_error_ownership",
    "input_clock_coverage",
)
MAX_MATRIX = 64
GIB = 1024**3


class MissingEvidence(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise MissingEvidence(reason)


def q(value: Any) -> Fraction:
    if type(value) is Fraction:
        result = value
    elif type(value) is int:
        result = Fraction(value)
    elif type(value) is float:
        require(math.isfinite(value), "nonfinite_number")
        result = Fraction(value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        require(
            all(type(v) is int for v in value) and value[1] > 0, "rational_encoding"
        )
        result = Fraction(*value)
    else:
        raise MissingEvidence("missing_or_invalid_number")
    require(
        max(result.numerator.bit_length(), result.denominator.bit_length()) <= 16384,
        "rational_bit_budget",
    )
    return result


def pair(value: Any) -> list[int]:
    value = q(value)
    return [value.numerator, value.denominator]


def nonnegative(value: Any) -> Fraction:
    value = q(value)
    require(value >= 0, "negative_uncertainty")
    return value


def upper(value: Any, bits: int = 160) -> Fraction:
    """Round a nonnegative bound upward to a bounded-size binary significand."""
    value = nonnegative(value)
    if value == 0:
        return value
    shift = value.numerator.bit_length() - value.denominator.bit_length() - bits + 1
    require(abs(shift) <= 8192, "propagation_exponent_budget")
    quantum = Fraction(2**shift) if shift >= 0 else Fraction(1, 2**-shift)
    scaled = value / quantum
    return ((scaled.numerator + scaled.denominator - 1) // scaled.denominator) * quantum


def unknown(reason: str, *, failed: bool = False, **details: Any) -> dict:
    return {
        "status": "unknown_reference_error",
        "method_status": "failed" if failed else "unknown",
        "reason": reason,
        "E_R": None,
        **details,
    }


def _premises(value: Mapping, names: tuple[str, ...]) -> None:
    require(isinstance(value, Mapping), "missing_premises")
    for name in names:
        item = value.get(name)
        require(
            isinstance(item, Mapping)
            and item.get("checked") is True
            and type(item.get("source_identity")) is str
            and item["source_identity"]
            and type(item.get("statement")) is str
            and item["statement"],
            "missing_premise:" + name,
        )


def estimate_e1(
    values: list,
    uncertainties: list,
    *,
    premises: Mapping,
    zero_time_proof: Mapping | None = None,
) -> dict:
    """Three levels are two refinements; contraction alone supplies no E_R."""
    try:
        require(len(values) == len(uncertainties) == 3, "three_levels_required")
        v = list(map(q, values))
        u = list(map(nonnegative, uncertainties))
        _premises(premises, E1_PREMISES)
        d01, d12 = v[0] - v[1], v[1] - v[2]
        r01, r12 = u[0] + u[1], u[1] + u[2]
        difference = {
            "D01": {"center": pair(d01), "radius": pair(r01)},
            "D12": {"center": pair(d12), "radius": pair(r12)},
        }
        if zero_time_proof is not None:
            require(
                zero_time_proof.get("kind")
                in {"same_initial_point", "reviewed_constant_or_invariant"}
                and zero_time_proof.get("checked") is True
                and zero_time_proof.get("source_identity")
                and zero_time_proof.get("statement"),
                "unsupported_zero_time_proof",
            )
            require(v[0] == v[1] == v[2], "zero_time_proof_value_disagreement")
            return {
                "status": "computed_pending_review",
                "method_status": "passed",
                "E_R": pair(u[2]),
                "route": "supported_zero_time_error",
                "zero_time_proof": dict(zero_time_proof),
                **difference,
            }
        al, ah = abs(d01) - r01, abs(d01) + r01
        bl, bh = abs(d12) - r12, abs(d12) + r12
        if al <= 0 or bl <= 0:
            return unknown("unresolved_difference_or_zero_denominator", **difference)
        if d01 * d12 < 0:
            return unknown("oscillatory_differences", failed=True, **difference)
        low, high = al / bh, ah / bl
        order = {
            "ratio_lower": pair(low),
            "ratio_upper": pair(high),
            "lower_fifth_ge_16": low**5 >= 16,
            "upper_fifth_le_64": high**5 <= 64,
        }
        if high**5 < 16 or low**5 > 64:
            return unknown("nonasymptotic_order", failed=True, **difference, **order)
        if low**5 < 16 or high**5 > 64:
            return unknown("uncertain_order_crosses_frozen_band", **difference, **order)
        # The center expression and its contamination retain distinct weights.
        error = C * abs(d12) + C * u[1] + (C + 1) * u[2]
        denominator = max(al, 2 * bl)
        consistency = (abs(d01 - 2 * d12) + r01 + 2 * r12) / denominator
        return {
            "status": "computed_pending_review",
            "method_status": "passed",
            "E_R": pair(error),
            **difference,
            **order,
            "terms": {
                "time_path": pair(C * abs(d12)),
                "level1_non_time": pair(C * u[1]),
                "level2_non_time": pair((C + 1) * u[2]),
            },
            "weights": [pair(C), pair(C), pair(C + 1)],
            "scalar_consistency_upper": pair(consistency),
            "scalar_consistency_is_independent": False,
        }
    except (MissingEvidence, KeyError, TypeError, ValueError, OverflowError) as exc:
        return unknown(str(exc))


def _fraction_range(value: Any) -> tuple[float, float]:
    value = q(value)
    center = float(value)
    require(math.isfinite(center), "matrix_float_range")
    actual = Fraction(center)
    lower = float(np.nextafter(center, -np.inf)) if actual > value else center
    higher = float(np.nextafter(center, np.inf)) if actual < value else center
    require(math.isfinite(lower) and math.isfinite(higher), "matrix_float_range")
    return lower, higher


def _matrix_input(values: Any) -> np.ndarray | tuple[tuple[Fraction, ...], ...]:
    """Keep exact inputs exact; only binary64-or-narrower floats use the fast path."""
    if hasattr(values, "shape"):
        require(
            len(values.shape) == 2
            and 0 < values.shape[0] == values.shape[1] <= MAX_MATRIX,
            "matrix_shape",
        )
    if hasattr(values, "toarray"):
        values = values.toarray()
    if isinstance(values, np.ndarray):
        require(values.ndim == 2, "matrix_shape")
        if values.dtype != object:
            require(values.dtype.kind in "iuf", "matrix_numeric_type")
            if values.dtype.kind == "f":
                require(values.dtype.itemsize <= 8, "matrix_float_precision")
                a = np.asarray(values, dtype=np.float64)
                require(np.isfinite(a).all(), "nonfinite_matrix")
                return a
            # ndarray.tolist converts integer words directly to Python ints,
            # including uint64; no float cast may precede cancellation.
        values = values.tolist()
    require(
        isinstance(values, (tuple, list))
        and 0 < len(values) <= MAX_MATRIX
        and all(len(row) == len(values) for row in values),
        "matrix_shape",
    )
    if all(type(value) is float for row in values for value in row):
        a = np.asarray(values, dtype=np.float64)
        require(np.isfinite(a).all(), "nonfinite_matrix")
        return a

    def exact(value):
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            require(value.dtype.itemsize <= 8, "matrix_float_precision")
            value = float(value)
        return q(value)

    return tuple(tuple(exact(value) for value in row) for row in values)


def _matrix_interval(values: Any) -> tuple[np.ndarray, np.ndarray]:
    values = _matrix_input(values)
    if isinstance(values, np.ndarray):
        return values.copy(), values.copy()
    n = len(values)
    low, high = np.empty((n, n)), np.empty((n, n))
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            low[i, j], high[i, j] = _fraction_range(v)
    return low, high


def _round(array: np.ndarray, direction: float) -> np.ndarray:
    require(np.isfinite(array).all(), "interval_arithmetic_nonfinite")
    value = np.nextafter(array, direction)
    require(np.isfinite(value).all(), "interval_arithmetic_overflow")
    return value


def _norm_upper(low: np.ndarray, high: np.ndarray) -> Fraction:
    values = np.maximum(np.abs(low), np.abs(high))
    result = np.zeros(values.shape[0])
    for i in range(values.shape[1]):
        result = _round(result + values[:, i], np.inf)
    return Fraction(float(np.max(result)))


def _left_product(
    z: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.zeros_like(z), np.zeros_like(z)
    for k in range(len(z)):
        x, y = z[:, k, None] * low[None, k, :], z[:, k, None] * high[None, k, :]
        a = _round(a + _round(np.minimum(x, y), -np.inf), -np.inf)
        b = _round(b + _round(np.maximum(x, y), np.inf), np.inf)
    return a, b


@dataclass(frozen=True)
class Scaling:
    state: tuple[Fraction, ...]
    equation: tuple[Fraction, ...]
    state_units: tuple[dict, ...]
    equation_units: tuple[dict, ...]

    def __post_init__(self):
        require(
            0
            < len(self.state)
            == len(self.equation)
            == len(self.state_units)
            == len(self.equation_units)
            <= MAX_MATRIX
            and all(q(v) > 0 for v in (*self.state, *self.equation)),
            "positive_scaling_shape",
        )

    @classmethod
    def from_spec(cls, value: Mapping, layout: Layout) -> Scaling:
        state = tuple(q(v) for v in value["state"])
        equation = tuple(q(v) for v in value["equation"])
        variables = tuple(
            json.loads(canonical(asdict(v.unit)))
            for v in layout.variables
            for _ in range(math.prod(v.shape))
        )
        equations = tuple(
            json.loads(
                canonical(asdict(e.unit * SECOND if e.role == "storage" else e.unit))
            )
            for e in layout.equations
            for _ in range(math.prod(e.shape))
        )
        require(
            0 < len(state) == len(equation) == layout.size <= MAX_MATRIX
            and min((*state, *equation)) > 0,
            "positive_scaling_shape",
        )
        require(
            tuple(value["state_units"]) == variables
            and tuple(value["equation_units"]) == equations,
            "physical_scaling_units",
        )
        return cls(state, equation, variables, equations)

    def operators(self, matrix: Any, radius: Any) -> tuple[np.ndarray, np.ndarray]:
        matrix, radius = _matrix_input(matrix), _matrix_input(radius)
        require(
            len(matrix) == len(radius) == len(self.state),
            "operator_radius_shape_or_sign",
        )
        if not (isinstance(matrix, np.ndarray) and isinstance(radius, np.ndarray)):
            # A/radius and positive SI scaling are combined in Q before the
            # final outward conversion. In particular (-2**53)+(2**53+1)=1
            # must not become a near-zero enclosure through float64 casting.
            def exact_rows(value):
                return (
                    tuple(tuple(Fraction(float(v)) for v in row) for row in value)
                    if isinstance(value, np.ndarray)
                    else value
                )

            center, uncertainty = exact_rows(matrix), exact_rows(radius)
            require(
                all(v >= 0 for row in uncertainty for v in row),
                "operator_radius_shape_or_sign",
            )
            n = len(self.state)
            low, high = np.empty((n, n)), np.empty((n, n))
            for i in range(n):
                for j in range(n):
                    scale = q(self.state[j]) / q(self.equation[i])
                    low[i, j] = _fraction_range(
                        (center[i][j] - uncertainty[i][j]) * scale
                    )[0]
                    high[i, j] = _fraction_range(
                        (center[i][j] + uncertainty[i][j]) * scale
                    )[1]
            return low, high
        low, high = _matrix_interval(matrix)
        rlow, rhigh = _matrix_interval(radius)
        require(
            low.shape == rlow.shape == (len(self.state), len(self.state))
            and np.min(rlow) >= 0,
            "operator_radius_shape_or_sign",
        )
        low, high = _round(low - rhigh, -np.inf), _round(high + rhigh, np.inf)
        a, b = np.empty_like(low), np.empty_like(high)
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                a[i, j], b[i, j] = _fraction_range(self.state[j] / self.equation[i])
        products = (low * a, low * b, high * a, high * b)
        return _round(np.minimum.reduce(products), -np.inf), _round(
            np.maximum.reduce(products), np.inf
        )


def _axis_values(value: Mapping) -> dict[str, Fraction]:
    require(
        isinstance(value, Mapping) and set(value) == set(AXES), "incomplete_error_axes"
    )
    return {axis: nonnegative(value[axis]) for axis in AXES}


def evaluate_e3_step(
    *,
    A: Any,
    B: Any,
    residual: list,
    scaling: Scaling,
    previous: Mapping,
    envelope: Mapping,
) -> dict:
    """Uniform-box Neumann/inverse-action propagation in scaled infinity norm.

    A/B are SI operators. Radius, remainder, initial and feasible-box evidence
    is required even when its amount is zero. An inaccurate inverse cannot pass
    merely because numpy.linalg.inv returned; I-Z*A is explicitly enclosed.
    """
    try:
        _premises(envelope["premises"], E3_PREMISES)
        require(envelope.get("source_identity"), "missing_operator_evidence_source")
        previous = _axis_values(previous)
        direct = _axis_values(envelope["direct_state_error_by_axis"])
        require(len(residual) == len(scaling.state), "residual_shape")
        require(
            set(envelope["equation_error_by_axis"]) == set(AXES),
            "incomplete_equation_error_axes",
        )
        errors = {}
        for axis in AXES:
            vector = list(map(nonnegative, envelope["equation_error_by_axis"][axis]))
            require(len(vector) == len(residual), "equation_error_shape")
            errors[axis] = max(
                (v / d for v, d in zip(vector, scaling.equation)), default=Fraction()
            )
        h_residual = max(abs(q(v)) / d for v, d in zip(residual, scaling.equation))
        errors["trajectory_nonlinear"] += h_residual
        al, ah = scaling.operators(A, envelope["A_radius"])
        bl, bh = scaling.operators(B, envelope["B_radius"])
        central = al / 2 + ah / 2
        inverse = np.linalg.inv(central)
        require(np.isfinite(inverse).all(), "nonfinite_inverse_action")
        zl, zh = _left_product(inverse, al, ah)
        eye = np.eye(len(inverse))
        eta = _norm_upper(_round(eye - zh, -np.inf), _round(eye - zl, np.inf))
        require(eta < 1, "eta_not_below_one")
        K = upper(_norm_upper(inverse, inverse) / (1 - eta))
        B_upper = _norm_upper(bl, bh)
        G = upper(K * B_upper)
        R = {axis: upper(K * errors[axis]) for axis in AXES}
        propagated = {
            axis: upper(G * previous[axis] + R[axis] + direct[axis]) for axis in AXES
        }
        radius = nonnegative(envelope["box_radius"])
        total = sum(propagated.values(), Fraction())
        require(radius > 0 and total <= radius, "box_self_mapping_not_established")
        return {
            "status": "computed_pending_review",
            "method_status": "passed",
            "eta_upper": pair(eta),
            "K_upper": pair(K),
            "B_norm_upper": pair(B_upper),
            "G_upper": pair(G),
            "R_by_axis": {k: pair(v) for k, v in R.items()},
            "U_by_axis": {k: pair(v) for k, v in direct.items()},
            "error_by_axis": {k: pair(v) for k, v in propagated.items()},
            "error_upper": pair(total),
            "box_radius": pair(radius),
            "residual_norm": pair(h_residual),
            "norm": "scaled_linf",
            "source_identity": envelope["source_identity"],
            "operator_enclosures_sha256": digest(
                {
                    "A_lower_hex": [[float(v).hex() for v in row] for row in al],
                    "A_upper_hex": [[float(v).hex() for v in row] for row in ah],
                    "B_lower_hex": [[float(v).hex() for v in row] for row in bl],
                    "B_upper_hex": [[float(v).hex() for v in row] for row in bh],
                }
            ),
            "arithmetic": "outward binary64 elementary operations; upward 160-bit bound storage",
            "qualified": False,
        }
    except (
        MissingEvidence,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as exc:
        return unknown(str(exc))


def charge_error(
    coefficients: list, errors_now: list, errors_initial: list, extra_by_axis: Mapping
) -> dict:
    """Apply the original signed physical form using absolute error weights."""
    try:
        require(
            len(coefficients) == len(errors_now) == len(errors_initial) > 0,
            "charge_component_coverage",
        )
        result = _axis_values(extra_by_axis)
        for a, now, initial in zip(coefficients, errors_now, errors_initial):
            now, initial = _axis_values(now), _axis_values(initial)
            for axis in AXES:
                result[axis] += abs(q(a)) * (now[axis] + initial[axis])
        return {
            "status": "computed_pending_review",
            "E_R": pair(sum(result.values(), Fraction())),
            "error_by_axis": {k: pair(v) for k, v in result.items()},
            "route": "original_body_charge_form",
        }
    except (MissingEvidence, KeyError, TypeError, ValueError) as exc:
        return unknown(str(exc))


def flux_error(
    q_value: Mapping,
    f_value: Mapping,
    q_error: Mapping,
    *,
    identity_proof: Mapping,
    path_coverage: Mapping,
) -> dict:
    """B continuity transfers an independently established q error, never zero residual alone."""
    try:
        _premises(
            identity_proof,
            (
                "fixed_B_continuity",
                "complete_boundary_storage",
                "fixed_parameters_and_initial",
                "no_reset_or_impulse",
            ),
        )
        require(
            path_coverage.get("all_pieces_independently_integrated") is True
            and path_coverage.get("storage_delta_substitution") is False
            and path_coverage.get("history_sha256"),
            "incomplete_independent_f_path",
        )
        require(
            q_error.get("status") == "computed_pending_review"
            and q_error.get("route") == "original_body_charge_form",
            "unqualified_independent_charge_error",
        )
        owned = _axis_values(q_error["error_by_axis"])
        qc, qr = q(q_value["center"]), nonnegative(q_value["radius"])
        fc, fr = q(f_value["center"]), nonnegative(f_value["radius"])
        # The discrepancy has one ownership record; both endpoint evaluation
        # radii belong here once, not also in charge_error's exact-form action.
        discrepancy = abs(fc - qc) + qr + fr
        owned["stability_reconstruction"] += discrepancy
        return {
            "status": "computed_pending_review",
            "E_R": pair(sum(owned.values(), Fraction())),
            "discrepancy_upper": pair(discrepancy),
            "error_by_axis": {k: pair(v) for k, v in owned.items()},
            "route": "frozen_B_continuity_from_independent_q",
            "qualified": False,
        }
    except (MissingEvidence, KeyError, TypeError, ValueError) as exc:
        return unknown(str(exc))


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def read_json(path: Path, limit: int = 4 * 1048576) -> dict:
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "metadata_byte_budget")
    value = json.loads(raw, object_pairs_hook=_unique_object)
    require(isinstance(value, dict), "metadata_object")
    canonical(value)
    return value


def pinned(ref: Mapping, limit: int = 4 * 1048576) -> dict:
    with Path(ref["path"]).open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "metadata_byte_budget")
    require(
        sha256(raw).hexdigest() == ref["sha256"]
        and ("bytes" not in ref or len(raw) == ref["bytes"]),
        "changed_pinned_artifact",
    )
    value = json.loads(raw, object_pairs_hook=_unique_object)
    require(isinstance(value, dict), "metadata_object")
    canonical(value)
    return value


def load_method(ref: Mapping) -> dict:
    method = pinned(ref)
    require(
        method.get("schema") == METHOD_SCHEMA
        and type(method.get("version")) is int
        and method["version"] == 1
        and method.get("status") == "frozen",
        "method_not_frozen",
    )
    freeze = method.get("freeze")
    require(
        isinstance(freeze, dict)
        and freeze.get("frozen_at")
        and freeze.get("frozen_by")
        and freeze.get("review_records")
        and freeze.get("user_records"),
        "missing_commander_freeze",
    )
    author = pinned(
        {
            "path": freeze["reviewed_path"],
            "sha256": freeze["reviewed_sha256"],
            "bytes": freeze["reviewed_bytes"],
        }
    )
    require(
        digest({k: v for k, v in method.items() if k not in {"status", "freeze"}})
        == digest({k: v for k, v in author.items() if k != "status"}),
        "changed_reviewed_method_body",
    )
    e1 = method["E1"]
    require(
        q(e1["Fs"]) == FS
        and q(e1["p_use"]) == Fraction(4, 5)
        and q(e1["coefficient_upper"]) == C
        and tuple(map(q, e1["order_interval"])) == (Fraction(4, 5), Fraction(6, 5))
        and method["time_ladder"]["levels"] == [0, 1, 2]
        and (1 + FS / C) ** 5 <= 16,
        "unsupported_frozen_method",
    )
    for review in freeze["review_records"]:
        with Path(review["path"]).open("rb") as stream:
            raw = stream.read(4 * 1048576 + 1)
        require(
            len(raw) <= 4 * 1048576 and sha256(raw).hexdigest() == review["sha256"],
            "changed_method_review",
        )
    return method


def form_matrix(form: Any, *, square: bool = False) -> list[list[Fraction]]:
    """Differentiate the original factored SI form, retaining volume/area/Nt."""
    n = form.layout.size
    rows = n if square else len(form.row_ids)
    require(
        0 < n <= MAX_MATRIX and len(form.row_ids) <= rows and not form.sources,
        "linear_form_scope",
    )
    result = [[Fraction() for _ in range(n)] for _ in range(rows)]
    for term in form.terms:
        if term.field is None:
            require(form.kind == "state", "nonstate_constant")
            continue  # Only the derivative vanishes; fixed-background proof is separate.
        require(term.field in form.layout.offsets, "external_linear_source")
        value = Fraction(term.sign)
        for factor in term.factors:
            require(
                factor.source_identity == form.source_identity, "foreign_form_factor"
            )
            value *= Fraction(factor.value)
        if term.divisor is not None:
            require(
                term.divisor.source_identity == form.source_identity
                and term.divisor.value != 0,
                "foreign_or_zero_form_divisor",
            )
            value /= Fraction(term.divisor.value)
        column = form.layout.offsets[term.field].start + term.index
        result[term.row][column] += value
    return result


def physical_values(point: Any) -> tuple[Fraction, ...]:
    from scripts.benchmarks.precision_prototype import physical_state_words

    fields = physical_state_words(point.state)
    return tuple(
        sum((Fraction(float(w.ravel()[i])) for w in fields[v.id]), Fraction())
        for v in point.state.layout.variables
        for i in range(math.prod(v.shape))
    )


class SlabOperators:
    """Read-only public B Jacobian and exact affine previous-state derivative."""

    def __init__(self, model: Any):
        require(
            model.definition.id == "DynamicAcceptorIonPublicDeviceV1"
            and model.intervals == 8,
            "frozen_B_operator_scope",
        )
        self.model, self.layout, self.source_identity = (
            model,
            model.layout,
            model.source_identity,
        )
        self.problem = model.public_problem()
        self.B = form_matrix(model.linear_forms["increment"]["storage"], square=True)
        self.charge_form = model.linear_forms["state"]["body_charge"]
        require(self.charge_form.row_units == (COULOMB,), "body_charge_unit")
        self.charge_coefficients = form_matrix(self.charge_form)[0]
        self.charge_form_identity = self.charge_form.identity

    def operators(self, left: Any, right: Any) -> tuple[Any, Any]:
        return self.problem.conservative_jacobian(left, right), self.B

    def feasible_box(self, point: Any, radius: Fraction, scaling: Scaling) -> bool:
        values = physical_values(point)
        for variable in self.layout.variables:
            indices = self.layout.offsets[variable.id]
            for i in range(indices.start, indices.stop):
                lo, hi = (
                    values[i] - radius * scaling.state[i],
                    values[i] + radius * scaling.state[i],
                )
                if variable.lower is not None and lo < Fraction(variable.lower):
                    return False
                if variable.upper is not None and hi > Fraction(variable.upper):
                    return False
                if variable.id in {"n_m3", "p_m3"} and lo <= 0:
                    return False
                if variable.id == "c_m3" and hi >= Fraction(
                    self.model.definition.ion_capacity
                ):
                    return False
        return True


def target_identity(
    source_identity: str, initial_point_identity: str, segments: list
) -> str:
    """Physical problem identity excludes integration controls and tolerances."""
    return digest(
        {
            "physical_source_identity": source_identity,
            "original_initial_point_identity": initial_point_identity,
            "segments": segments,
        }
    )


@dataclass(frozen=True)
class Level:
    directory: Path
    identity: RunIdentity
    result: dict
    header: dict
    readback: dict
    request: dict


def load_level(
    ref: Mapping, layout: Layout, *, checkpoint: Callable = lambda: None
) -> Level:
    directory = Path(ref["directory"])
    identity = RunIdentity(**ref["identity"])
    result = read_json(directory / "RunResult.json")
    # Reject native histories before any data stream can be opened.
    require(
        result.get("schema") == BE_SCHEMA and result.get("method") == METHOD,
        "not_BE_history",
    )
    readback = read_reference_history(
        directory, layout, identity, checkpoint=checkpoint
    )
    require(
        readback["structural_pass"] is True
        and readback["numerical_complete"] is True
        and readback["scientific_status"] != "failed",
        "incomplete_or_failed_BE_reference",
    )
    require(readback["accepted_steps"] <= 200000, "step_budget")
    request = read_json(directory / "ReferenceRequest.json")
    path = directory / readback["observed_history"]["path"]
    reader = HistoryReader(
        path,
        encoding=readback["observed_history"]["encoding"],
        max_record_bytes=1048576,
        max_logical_bytes=32 * GIB,
    )
    try:
        header = json.loads(next(iter(reader)), object_pairs_hook=_unique_object)
    finally:
        reader.close()
    require(header["kind"] == "be_reference_header", "missing_BE_header")
    return Level(directory, identity, result, header, readback, request)


def iter_records(level: Level, *, checkpoint: Callable = lambda: None):
    """Seal is rechecked on this second streaming pass; partial use is unqualified."""
    meta = level.readback["observed_history"]
    reader = HistoryReader(
        level.directory / meta["path"],
        encoding=meta["encoding"],
        max_record_bytes=1048576,
        max_logical_bytes=32 * GIB,
    )
    checksum, count = sha256(), 0
    try:
        for line in reader:
            checkpoint()
            checksum.update(line)
            count += 1
            yield json.loads(line, object_pairs_hook=_unique_object)
        require(
            checksum.hexdigest() == meta["logical_sha256"]
            and count == meta["records"]
            and reader.eof_seen
            and not reader.partial_tail,
            "BE_changed_during_error_analysis",
        )
    finally:
        reader.close()


def _residual(row: Mapping) -> list[Fraction]:
    words = row["residual_words"]
    require(len(words["shape"]) == 1, "residual_vector_shape")
    return [
        sum((Fraction(float.fromhex(w[i])) for w in words["words_hex"]), Fraction())
        for i in range(words["shape"][0])
    ]


def _unit(unit: Any) -> str:
    for symbol, known in (
        ("V", VOLT),
        ("m^-3", PARTICLE / VOLUME),
        ("1", ONE),
        ("particle", PARTICLE),
        ("C", COULOMB),
    ):
        if unit == known:
            return symbol
    raise MissingEvidence("unsupported_observable_unit")


def quantities(layout: Layout) -> list[dict]:
    result = []
    for variable in layout.variables:
        for local in range(math.prod(variable.shape)):
            result.append(
                {
                    "key": "state/" + variable.id + "/" + str(local),
                    "field": variable.id,
                    "index": local,
                    "column": layout.offsets[variable.id].start + local,
                    "quantity": variable.id,
                    "unit": _unit(variable.unit),
                    "kind": "state",
                }
            )
    result.extend(
        {"key": "port/" + side, "quantity": "port_" + side, "unit": "A", "kind": "port"}
        for side in ("left", "right")
    )
    return result


def sample_center(sample: Mapping, item: Mapping) -> Fraction:
    if item["kind"] == "port":
        return q(sample["ports"][item["key"].split("/")[1]]["center"])
    words = sample["physical_state_words"][item["field"]]["words_hex"]
    return sum(
        (Fraction(float.fromhex(row[item["index"]])) for row in words), Fraction()
    )


def _sample_uncertainty(
    sample: Mapping,
    item: Mapping,
    state_error: Mapping,
    scaling: Scaling,
    support: Mapping,
) -> dict:
    try:
        e = _axis_values(state_error)
        if item["kind"] == "state":
            return {
                "status": "supported",
                "by_axis": {
                    axis: pair(e[axis] * scaling.state[item["column"]]) for axis in AXES
                },
            }
        sensitivity = support["port_sensitivities"][item["key"]]
        _premises(
            sensitivity["premises"],
            ("uniform_physical_sensitivity", "direct_input_coverage"),
        )
        require(
            sensitivity["unit"] == "A"
            and len(sensitivity["absolute_gradient"]) == len(scaling.state),
            "port_sensitivity_units_or_shape",
        )
        require(
            tuple(sensitivity["state_units"]) == scaling.state_units,
            "port_sensitivity_column_units",
        )
        gain = sum(
            (
                nonnegative(v) * d
                for v, d in zip(sensitivity["absolute_gradient"], scaling.state)
            ),
            Fraction(),
        )
        direct = _axis_values(sensitivity["direct_error_by_axis"])
        values = {a: e[a] * gain + direct[a] for a in AXES}
        values["evaluation"] += nonnegative(
            sample["ports"][item["key"].split("/")[1]]["radius"]
        )
        return {
            "status": "supported",
            "by_axis": {k: pair(v) for k, v in values.items()},
        }
    except (MissingEvidence, KeyError, TypeError, ValueError) as exc:
        return {
            "status": "unknown_reference_error",
            "reason": str(exc),
            "by_axis": None,
        }


def analyze_e3(
    level: Level,
    provider: Any,
    support: Mapping,
    *,
    emit: Callable = lambda row: None,
    checkpoint: Callable = lambda: None,
) -> dict:
    """Every accepted step contributes, including intervals without a plot tag."""
    samples, items = {}, quantities(provider.layout)
    setup_error = None
    try:
        require(
            support["source_identity"] == provider.source_identity, "E3_source_binding"
        )
        scaling = Scaling.from_spec(support["scaling"], provider.layout)
        _premises(support["initial_premises"], ("initial_error_and_DAE_meaning",))
        previous_error = _axis_values(support["initial_error_by_axis"])
        require(
            support["initial_point_identity"] == level.identity.initial_point_identity,
            "initial_error_binding",
        )
    except (MissingEvidence, KeyError, TypeError, ValueError) as exc:
        setup_error, scaling, previous_error = str(exc), None, None
    reference = level.header["original_reference"]
    previous_point = unpack_point(level.header["initial"], reference, provider.layout)
    if setup_error is None and not provider.feasible_box(
        previous_point, sum(previous_error.values(), Fraction()), scaling
    ):
        setup_error = "initial_error_box_outside_physical_domain"
    count, failure, first_failure, pieces = 0, setup_error, None, 0
    prefix = [[Fraction(), Fraction()] for _ in range(3)]
    for row in iter_records(level, checkpoint=checkpoint):
        if row["kind"] == "be_accepted_step":
            count += 1
            right = unpack_point(row["right"], reference, provider.layout)
            if failure is None:
                try:
                    envelope = support.get("step_overrides", {}).get(
                        str(count), support["step_envelope"]
                    )
                    require(
                        envelope["source_identity"] == provider.source_identity,
                        "E3_operator_source_binding",
                    )
                    require(
                        provider.feasible_box(
                            right, nonnegative(envelope["box_radius"]), scaling
                        ),
                        "box_outside_physical_domain",
                    )
                    A, B = provider.operators(previous_point, right)
                    out = evaluate_e3_step(
                        A=A,
                        B=B,
                        residual=_residual(row),
                        scaling=scaling,
                        previous=previous_error,
                        envelope=envelope,
                    )
                except (
                    MissingEvidence,
                    KeyError,
                    TypeError,
                    ValueError,
                    ContractError,
                ) as exc:
                    out = unknown(str(exc))
                if out["status"] == "computed_pending_review":
                    out["error_box"] = {
                        "center_point_identity": right.identity,
                        "enclosing_scaled_radius": out["box_radius"],
                        "computed_scaled_radius": out["error_upper"],
                        "physical_definition": "X_center[i] +/- D_X[i] * scaled_linf_radius",
                        "scaling_sha256": digest(
                            {
                                "state": [pair(v) for v in scaling.state],
                                "equation": [pair(v) for v in scaling.equation],
                                "state_units": scaling.state_units,
                                "equation_units": scaling.equation_units,
                            }
                        ),
                        "physical_domain_checked": True,
                    }
                    previous_error = out["error_by_axis"]
                else:
                    failure = out["reason"]
            else:
                out = unknown(failure)
            if failure is not None and first_failure is None:
                first_failure = {
                    "step_index": count,
                    "time_hex": row["t_right_hex"],
                    "reason": failure,
                }
            emit(
                {
                    "kind": "reference_error_step",
                    "level": level.header["schedule"]["level"],
                    "step_index": count,
                    "left_identity": previous_point.identity,
                    "right_identity": right.identity,
                    "h_hex": row["h_hex"],
                    "state_error": out,
                }
            )
            previous_point = right
        elif row["kind"] == "be_observation" and row.get("tag") is None:
            piece = row["interval_observation"]
            require(
                row["interval_step"] == pieces + 1 == count
                and piece["storage_secant_used_as_flux"] is False
                and piece["unit"] == "C"
                and len(piece["physical_flux"]) == 3,
                "independent_flux_piece_coverage",
            )
            for j, value in enumerate(piece["physical_flux"]):
                prefix[j][0] += q(value["center"])
                prefix[j][1] += nonnegative(value["radius"])
            pieces += 1
        elif row["kind"] == "be_observation" and row.get("tag") is not None:
            require(
                all(
                    q(v["center"]) == p[0] and nonnegative(v["radius"]) == p[1]
                    for v, p in zip(
                        row["cumulative_physical_flux"], prefix, strict=True
                    )
                ),
                "flux_prefix_identity",
            )
            key = digest(row["tag"])
            samples[key] = {
                item["key"]: (
                    _sample_uncertainty(row, item, previous_error, scaling, support)
                    if failure is None
                    else {
                        "status": "unknown_reference_error",
                        "reason": failure,
                        "by_axis": None,
                    }
                )
                for item in items
            }
    require(
        pieces == count == level.readback["accepted_steps"]
        and len(samples) == len(level.readback["observations"]),
        "E3_trajectory_coverage",
    )
    return {
        "status": "computed_pending_review"
        if failure is None
        else "unknown_reference_error",
        "samples": samples,
        "accepted_steps": count,
        "first_failure": first_failure,
        "setup_error": setup_error,
        "axis": "E3",
        "qualified": False,
        "path_coverage": {
            "all_pieces_independently_integrated": True,
            "storage_delta_substitution": False,
            "pieces": pieces,
            "history_sha256": level.readback["observed_history"]["logical_sha256"],
            "review_scope": "sequence and declared physical-flux provenance checked; mathematical source/data review remains external",
        },
    }


def _e1_owned(result: dict, non_time: list[dict]) -> dict:
    if result["status"] != "computed_pending_review":
        return result
    values = [_axis_values(x["by_axis"]) for x in non_time]
    if result.get("route") == "supported_zero_time_error":
        owned = values[2]
    else:
        owned = {a: C * values[1][a] + (C + 1) * values[2][a] for a in AXES}
        owned["stability_reconstruction"] += q(result["terms"]["time_path"])
    require(
        sum(owned.values(), Fraction()) == q(result["E_R"]), "E1_component_accounting"
    )
    return {**result, "error_by_axis": {k: pair(v) for k, v in owned.items()}}


def _evidence(item: Mapping, rows: list[dict], context: Mapping) -> dict:
    complete = all(
        r["computation"]["status"] == "computed_pending_review" for r in rows
    )
    value = (
        max((q(r["computation"]["E_R"]) for r in rows), default=Fraction())
        if complete
        else None
    )
    failed = any(r["computation"].get("method_status") == "failed" for r in rows)
    labels = [r["tag"] for r in rows]
    component = {
        "id": "joint_E1_E3:" + item["key"],
        "claim_kind": CLAIM,
        "covered_axes": list(AXES),
        "value": pair(value) if value is not None else None,
        "unit": item["unit"],
        "source": context["lineage_source"],
        "ownership": "joint max of complete pointwise sums; per_record retains all axes; no independent 1/12 credits",
    }
    if item["unit"] == "C":
        component["value_C"] = component["value"]
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "computed_pending_review" if complete else "unknown_reference_error",
        "method_status": "failed" if failed else "passed" if complete else "unknown",
        "evidence_class": "G2-R_BE_E1_E3_reference",
        "claim_kind": CLAIM,
        "target_problem_identity": context["target_problem_identity"],
        "quantity_and_unit": {"quantity": item["quantity"], "unit": item["unit"]},
        "quantity_key": item["key"],
        "norm_and_coverage": {
            "norm": "max_abs_tagged" + str(len(rows)),
            "records": len(rows),
            "labels_sha256": digest(labels),
        },
        "reference_source_and_run_identity": context["reference_identity"],
        "error_components_and_ownership": [component],
        "assumptions_and_checked_premises": context["premises"],
        "observed_refinement_and_method_independence": {
            "E1": {
                "status": "computed_pending_review" if complete else "unknown",
                "levels": [0, 1, 2],
                "refinements": 2,
                "source": context["lineage_source"],
            },
            "E2": {
                "status": "pending_Main_Strict_transfer",
                "scope": "BE E_R supplied here",
            },
            "E3": {
                "status": "computed_pending_review" if complete else "unknown",
                "source": context["lineage_source"],
            },
        },
        "estimated_or_enclosed_value": {
            "value_field": "conditional_estimate_value",
            "value": pair(value) if value is not None else None,
            "unit": item["unit"],
        },
        "conditional_estimate_value": pair(value) if value is not None else None,
        "rigorous_bound_value": None,
        "diagnostic_value": None,
        "unresolved_common_bias_and_model_error": [
            "shared constitutive, boundary, units and initial-model errors are not ruled out by refinement",
            "scope is the frozen semidiscrete problem; no continuum/SCAPS/hysteresis/RS qualification",
        ],
        "reviewer_and_decision_authority": {
            "evidence_author": context["author"],
            "independent_review": "pending",
            "method_contract": context["method_contract"],
        },
        "per_record": rows,
        "scientific_pass": False,
        "reference_value_radius_ownership": "reported observation radii are included in E_R; coalesce them in E2",
    }


def analyze_levels(
    levels: list[Level],
    provider: Any,
    parameters: Mapping,
    context: Mapping,
    *,
    emit: Callable = lambda row: None,
    checkpoint: Callable = lambda: None,
) -> dict:
    """Build per-quantity BE E_R inputs for Main; never perform the E2 transfer."""
    require(len(levels) == 3, "three_levels_required")
    require(
        [x.header["schedule"]["level"] for x in levels] == [0, 1, 2],
        "wrong_time_ladder",
    )
    require(
        len({digest(x.header["schedule"]["seed"]) for x in levels}) == 1
        and len({digest(x.header["schedule"]["segments"]) for x in levels}) == 1,
        "different_time_family",
    )
    for level in levels:
        segments = level.header["schedule"]["segments"]
        require(
            digest(segments) == digest(level.request["segments"][: len(segments)]),
            "history_protocol_differs_from_source_request",
        )
    require(
        all(x.identity.source_identity == provider.source_identity for x in levels),
        "physical_source_binding",
    )
    require(
        len({x.identity.initial_point_identity for x in levels}) == 1
        and len({x.identity.input_sha256 for x in levels}) == 1
        and len({x.identity.environment_identity for x in levels}) == 1
        and len({x.identity.source_commit for x in levels}) == 1,
        "different_problem_or_execution_family",
    )
    require(
        all(
            x.identity.method_contract_sha256 == context["method_contract"]["sha256"]
            for x in levels
        ),
        "method_identity",
    )
    labels = [r["tag"] for r in levels[0].readback["observations"]]
    require(
        all(
            digest([r["tag"] for r in x.readback["observations"]]) == digest(labels)
            for x in levels
        ),
        "different_time_labels",
    )
    physical_problem = target_identity(
        provider.source_identity,
        levels[0].identity.initial_point_identity,
        levels[0].request["segments"],
    )
    require(
        parameters["target_problem_identity"]
        == context["target_problem_identity"]
        == physical_problem,
        "target_problem_binding",
    )
    require(
        parameters["method_contract_sha256"] == context["method_contract"]["sha256"],
        "parameter_method_binding",
    )
    e3 = [
        analyze_e3(level, provider, support, emit=emit, checkpoint=checkpoint)
        for level, support in zip(levels, parameters["levels"], strict=True)
    ]
    items, records, rows_by_key = quantities(provider.layout), {}, {}
    for item in items:
        rows = []
        for index, tag in enumerate(labels):
            checkpoint()
            key = digest(tag)
            samples = [x.readback["observations"][index] for x in levels]
            values = [sample_center(s, item) for s in samples]
            uncertainties = [x["samples"][key][item["key"]] for x in e3]
            try:
                require(
                    all(x["status"] == "supported" for x in uncertainties),
                    "missing_E3_observable_support",
                )
                u = [
                    sum(_axis_values(x["by_axis"]).values(), Fraction())
                    for x in uncertainties
                ]
                proof = None
                if index == 0 and tag["protocol_initial"]:
                    proof = {
                        "kind": "same_initial_point",
                        "checked": True,
                        "source_identity": levels[0].identity.initial_point_identity,
                        "statement": "same original Point; only temporal approximation is zero; E3 initial uncertainty retained",
                    }
                elif item["key"] in parameters.get("invariants", {}):
                    proof = parameters["invariants"][item["key"]]
                    require(
                        proof["labels_sha256"] == digest(labels)
                        and proof["target_problem_identity"] == physical_problem,
                        "invariant_proof_scope",
                    )
                computation = _e1_owned(
                    estimate_e1(
                        values,
                        u,
                        premises=parameters["E1_premises"],
                        zero_time_proof=proof,
                    ),
                    uncertainties,
                )
            except (MissingEvidence, KeyError, TypeError, ValueError) as exc:
                computation = unknown(str(exc))
            represented = (
                {"center": pair(values[2]), "radius": [0, 1]}
                if item["kind"] == "state"
                else samples[2]["ports"][item["key"].split("/")[1]]
            )
            rows.append(
                {"tag": tag, "computed_value": represented, "computation": computation}
            )
        rows_by_key[item["key"]] = rows
        records[item["key"]] = _evidence(item, rows, context)

    # B charge form is differentiated independently of balance residuals.
    q_rows, f_rows = [], []
    initial = levels[2].readback["observations"][0]
    state_items = [item for item in items if item["kind"] == "state"]
    for index, tag in enumerate(labels):
        sample = levels[2].readback["observations"][index]
        q_value, computation = None, unknown("missing_body_charge_proof")
        try:
            support = parameters["charge"]
            require(
                support["form_identity"] == provider.charge_form_identity,
                "original_charge_form_binding",
            )
            _premises(
                support["premises"],
                ("complete_body_charge_form", "fixed_coefficients_and_background"),
            )
            coeff = provider.charge_coefficients
            values = [
                sample_center(sample, item) - sample_center(initial, item)
                for item in state_items
            ]
            center = sum(
                (q(a) * v for a, v in zip(coeff, values, strict=True)), Fraction()
            )
            q_value = {"center": pair(center), "radius": [0, 1]}
            reported = sample["charges"]["device"]
            old = initial["charges"]["device"]
            require(
                abs(center - q(reported["center"]) + q(old["center"]))
                <= nonnegative(reported["radius"]) + nonnegative(old["radius"]),
                "charge_observation_disagrees_with_original_form",
            )
            active = [
                (a, item)
                for a, item in zip(coeff, state_items, strict=True)
                if q(a) != 0
            ]
            now = [rows_by_key[item["key"]][index]["computation"] for _, item in active]
            start = [rows_by_key[item["key"]][0]["computation"] for _, item in active]
            require(
                all(x["status"] == "computed_pending_review" for x in (*now, *start)),
                "unqualified_state_errors",
            )
            computation = charge_error(
                [a for a, _ in active],
                [x["error_by_axis"] for x in now],
                [x["error_by_axis"] for x in start],
                support["extra_by_axis"],
            )
        except (
            MissingEvidence,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
        ) as exc:
            computation = unknown(str(exc))

        def raw_ladder(quantity):
            values, radii = [], []
            for level in levels:
                one, first = (
                    level.readback["observations"][index],
                    level.readback["observations"][0],
                )
                if quantity == "q":
                    now, old = one["charges"]["device"], first["charges"]["device"]
                    values.append(q(now["center"]) - q(old["center"]))
                    radii.append(
                        nonnegative(now["radius"]) + nonnegative(old["radius"])
                    )
                else:
                    values.append(q(one["cumulative_physical_flux"][0]["center"]))
                    radii.append(
                        nonnegative(one["cumulative_physical_flux"][0]["radius"])
                    )
            return {
                "claim_kind": "diagnostic_agreement",
                "eligible_for_E_R": False,
                "D01_center": pair(values[0] - values[1]),
                "D12_center": pair(values[1] - values[2]),
                "D01_readout_radius_only": pair(radii[0] + radii[1]),
                "D12_readout_radius_only": pair(radii[1] + radii[2]),
                "scope": "direct q/f ladder retained; radii are not complete forward/non-time uncertainty; no alternative estimate selected",
            }

        q_rows.append(
            {
                "tag": tag,
                "computed_value": q_value,
                "computation": computation,
                "direct_ladder_diagnostic": raw_ladder("q"),
            }
        )
        f_value = sample["cumulative_physical_flux"][0]
        f_computation = flux_error(
            q_value or {},
            f_value,
            computation,
            identity_proof=parameters.get("charge", {}).get("identity_proof", {}),
            path_coverage=e3[2]["path_coverage"],
        )
        f_rows.append(
            {
                "tag": tag,
                "computed_value": f_value,
                "computation": f_computation,
                "direct_ladder_diagnostic": raw_ladder("f"),
            }
        )
    for name, rows in (("q", q_rows), ("f", f_rows)):
        records[name] = _evidence(
            {"key": name, "quantity": name, "unit": "C"}, rows, context
        )
    complete = all(r["status"] == "computed_pending_review" for r in records.values())
    return {
        "schema": SCHEMA,
        "status": "computed_pending_review" if complete else "unknown_reference_error",
        "target_problem_identity": physical_problem,
        "method_contract": context["method_contract"],
        "claim_kind": CLAIM,
        "quantities": records,
        "E3_summary": [{k: v for k, v in x.items() if k != "samples"} for x in e3],
        "E2_status": "pending_Main_Strict_transfer",
        "independent_review": "pending",
        "scientific_pass": False,
    }


class Output:
    """New local result bundle with one bounded streaming E3 ledger."""

    def __init__(self, directory: Path, limits: Limits):
        self.directory, self.limits, self.bytes = Path(directory), limits, 0
        self.directory.mkdir(parents=True, exist_ok=False)
        self.ledger = (self.directory / "E3Steps.jsonl").open("xb")
        self.ledger_hash, self.rows = sha256(), 0

    def emit(self, row: dict) -> None:
        raw = canonical(row) + b"\n"
        require(
            len(raw) <= self.limits.record_bytes
            and self.bytes + len(raw) <= self.limits.output_bytes,
            "error_output_budget",
        )
        self.ledger.write(raw)
        self.ledger_hash.update(raw)
        self.bytes += len(raw)
        self.rows += 1

    def write(self, name: str, value: dict, *, metadata: bool = True) -> dict:
        require(Path(name).name == name, "output_filename")
        raw = canonical(value) + b"\n"
        require(
            not metadata or len(raw) <= self.limits.record_bytes,
            "error_metadata_budget",
        )
        require(
            self.bytes + len(raw) <= self.limits.output_bytes, "error_output_budget"
        )
        path = self.directory / name
        with path.open("xb") as stream:
            stream.write(raw)
        self.bytes += len(raw)
        return {"path": str(path), "bytes": len(raw), "sha256": sha256(raw).hexdigest()}

    def close(self) -> None:
        self.ledger.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate",))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = read_json(args.manifest)
    require(
        manifest.get("schema") == "solarlab.g2-reference-error-request.v1",
        "error_manifest_schema",
    )
    limits = Limits(**manifest["limits"])
    checkpoint = Guard(limits)
    method = load_method(manifest["method_contract"])
    parameters = pinned(manifest["parameters"])
    require(
        parameters.get("schema") == "solarlab.g2-reference-error-parameters.v1",
        "parameter_schema",
    )
    require(len(manifest["levels"]) == 3, "three_levels_required")
    first = Path(manifest["levels"][0]["directory"])
    request = read_json(first / "ReferenceRequest.json")
    require(
        read_json(first / "RunResult.json").get("schema") == BE_SCHEMA, "not_BE_history"
    )
    from scripts.benchmarks.coupled_device_prototype import (
        AffineCoupledSlab,
        SlabDefinition,
    )

    require(
        request["case_id"] == method["same_problem"]["case_id"]
        and request["numeric_packet"]["intervals"] == 8,
        "frozen_B_scope",
    )
    model = AffineCoupledSlab(
        SlabDefinition(**request["numeric_packet"]["definition"]), 8
    )
    provider = SlabOperators(model)
    levels = [
        load_level(r, model.layout, checkpoint=checkpoint) for r in manifest["levels"]
    ]
    require(
        all(
            x.header["schedule"]["seed"] == method["time_ladder"]["seed_cap_s"]
            for x in levels
        ),
        "frozen_seed",
    )
    output = Output(Path(manifest["directory"]), limits)
    try:
        lineage = output.write(
            "Lineage.json",
            {
                "schema": SCHEMA,
                "kind": "reference_error_lineage",
                "method_contract": manifest["method_contract"],
                "parameters": manifest["parameters"],
                "levels": [
                    dict(
                        identity=x.identity.payload(),
                        observed_history=x.readback["observed_history"],
                    )
                    for x in levels
                ],
                "evaluation_source_sha256": sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "scientific_pass": False,
            },
        )
        context = {
            "method_contract": manifest["method_contract"],
            "lineage_source": lineage,
            "author": manifest["author"],
            "target_problem_identity": parameters["target_problem_identity"],
            "premises": {
                "E1": parameters["E1_premises"],
                "E3_parameters": manifest["parameters"],
            },
            "reference_identity": {
                "case_id": levels[2].identity.case_id,
                "run_id": digest(levels[2].identity.payload()),
                "source_identity": model.source_identity,
                "request_sha256": levels[2].identity.input_sha256,
                "history_sha256": levels[2].readback["observed_history"][
                    "logical_sha256"
                ],
            },
        }
        result = analyze_levels(
            levels,
            provider,
            parameters,
            context,
            emit=output.emit,
            checkpoint=checkpoint,
        )
        descriptors = {}
        for key, record in result.pop("quantities").items():
            filename = "Quantity-" + sha256(key.encode()).hexdigest()[:20] + ".json"
            descriptors[key] = output.write(filename, record)
        output.close()
        result.update(
            quantity_records=descriptors,
            E3_ledger={
                "path": str(output.directory / "E3Steps.jsonl"),
                "rows": output.rows,
                "sha256": output.ledger_hash.hexdigest(),
            },
            elapsed_s=time.monotonic() - checkpoint.start,
        )
        result_ref = output.write("Result.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result": result_ref,
                    "scientific_pass": False,
                }
            )
        )
        return 0 if result["status"] == "computed_pending_review" else 2
    except BaseException as exc:
        output.close()
        output.write(
            "Failure.json",
            {
                "status": "unknown_reference_error",
                "reason": str(exc)[:4096],
                "completed": False,
                "scientific_pass": False,
                "E3_rows": output.rows,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

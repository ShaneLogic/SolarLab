"""Frozen R1 V53 certificate limits, 13-channel schema, study requests, and qualification rules.

Implements the research-side R1 contracts for ``P07-01`` and ``P07-02``:
* Exact 16-item V53 certificate limits, solver budgets, and relative engineering limits.
* Exact 13-channel operator observation packet validator.
* Controls ``A-D`` definitions (``A:(0,0)``, ``B:(1,0)``, ``C:(0,1)``, ``D:(1,1)``).
* Conservative ``(4, 8, 16)`` substep row-count formula (``3727`` rows on the 100 s
  12-per-decade window; ``59`` rows on a 2-interval short window).
* Strict IDA ``R1-L2`` exclusion (``P07-02.4``): IDA or interpolated trajectories
  can never grant ``R1-L2``, and short-window verification never claims ``R1-L2``
  (full 100 s) or ``R1-L3`` device qualification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "R1_CONSERVATIVE_SUBSTEPS",
    "R1_CONTROLS",
    "R1_FULL_WINDOW_EXPECTED_ROWS",
    "R1_OPERATOR_CHANNELS",
    "R1_RELATIVE_ENGINEERING_LIMITS",
    "R1_SOLVER_BUDGETS",
    "R1_STUDY_REQUEST_SCHEMA",
    "R1_V53_CERTIFICATE_LIMITS",
    "R1CertificateEvaluation",
    "R1QualificationAssessment",
    "R1StudyRequest",
    "assess_r1_qualification",
    "build_r1_study_request",
    "evaluate_v53_certificate",
    "expected_substep_rows",
    "load_r1_study_request",
    "observation_times_12_per_decade",
    "validate_operator_channel_packet",
    "verify_r1_study_request",
]

R1_STUDY_REQUEST_SCHEMA = "solarlab_research.r1_study_request.v1"

R1_CONSERVATIVE_SUBSTEPS: tuple[int, int, int] = (4, 8, 16)
R1_FULL_WINDOW_EXPECTED_ROWS: int = 3727

R1_OPERATOR_CHANNELS: tuple[str, ...] = (
    "capture_flux",
    "electron_current",
    "electron_density",
    "electron_rate",
    "hole_current",
    "hole_density",
    "hole_rate",
    "local_state",
    "positive_ion_flux",
    "positive_ion_rate",
    "potential",
    "sheet_charge",
    "trace_potential",
)

R1_V53_CERTIFICATE_LIMITS: Mapping[str, float] = MappingProxyType({
    "all_face_current_relative": 2.0e-6,
    "analytic_jacobian_error": 3.0e-4,
    "charge_balance_relative": 1.0e-10,
    "current_decomposition_error": 1.0e-14,
    "eliminated_operator_error": 1.0e-6,
    "full_charge_balance_normalized": 1.0e-10,
    "full_gauss_normalized": 1.0e-10,
    "interface_current_relative": 2.0e-6,
    "inventory_relative_drift": 1.0e-10,
    "local_carrier_residual": 1.0e-7,
    "local_gauss_residual": 1.0e-7,
    "nonlinear_residual": 0.05,
    "physical_current_spread_relative": 2.0e-6,
    "refinement_current_change": 0.05,
    "refinement_state_change": 0.02,
    "trap_storage_normalized_error": 1.0,
})

R1_SOLVER_BUDGETS: Mapping[str, int] = MappingProxyType({
    "maximum_newton_iterations": 100,
    "maximum_line_search_steps": 40,
    "maximum_near_acceptance_nonmonotone_steps": 2,
    "maximum_joint_corrector_iterations": 8,
    "maximum_independent_poisson_iterations": 12,
})

R1_RELATIVE_ENGINEERING_LIMITS: Mapping[str, float] = MappingProxyType({
    "bytes_per_row_ratio": 3.0,
    "elapsed_ratio": 5.0,
    "peak_rss_ratio": 2.0,
})

R1_CONTROLS: Mapping[str, Mapping[str, int]] = MappingProxyType({
    "A": MappingProxyType({"nu_I": 0, "nu_t": 0}),
    "B": MappingProxyType({"nu_I": 1, "nu_t": 0}),
    "C": MappingProxyType({"nu_I": 0, "nu_t": 1}),
    "D": MappingProxyType({"nu_I": 1, "nu_t": 1}),
})


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def observation_times_12_per_decade(
    *,
    first_time_s: float = 1.0e-9,
    last_time_s: float = 1.0e2,
    points_per_decade: int = 12,
) -> tuple[float, ...]:
    """Build the canonical ``0.0`` + logarithmic 12-points-per-decade R1 observation grid."""
    if not (math.isfinite(first_time_s) and math.isfinite(last_time_s)) or first_time_s <= 0.0 or last_time_s <= first_time_s:
        raise ValueError("0 < first_time_s < last_time_s is required")
    if type(points_per_decade) is not int or points_per_decade < 1:
        raise ValueError("points_per_decade must be a positive integer")
    decades = math.log10(last_time_s / first_time_s)
    intervals = int(round(decades * points_per_decade))
    if not math.isclose(decades * points_per_decade, intervals, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("time span must be an exact rational multiple of 1/points_per_decade decades")
    log_start = math.log10(first_time_s)
    positive = [10.0 ** (log_start + k / points_per_decade) for k in range(intervals + 1)]
    positive[0] = float(first_time_s)
    positive[-1] = float(last_time_s)
    return (0.0, *positive)


def expected_substep_rows(
    times_s: Sequence[float],
    substeps: Sequence[int] = R1_CONSERVATIVE_SUBSTEPS,
) -> int:
    """Compute exact accepted row count ``sum(1 + s * K for s in substeps)`` without interpolation."""
    if len(times_s) < 2:
        raise ValueError("times_s must contain 0.0 and at least one positive observation endpoint")
    arr: NDArray[np.float64] = np.asarray(times_s, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)) or arr[0] != 0.0 or not np.all(np.diff(arr) > 0.0):
        raise ValueError("times_s must start at 0.0 and strictly increase")
    if not substeps or any(type(s) is not int or s < 1 for s in substeps):
        raise ValueError("substeps must be a non-empty sequence of positive integers")
    k_intervals = len(times_s) - 1
    return sum(1 + int(s) * k_intervals for s in substeps)


@dataclass(frozen=True, slots=True)
class R1CertificateEvaluation:
    """Outcome of checking the 16 frozen V53 certificate limits."""

    passed: bool
    metrics: Mapping[str, float]
    limits: Mapping[str, float]
    failed_items: tuple[str, ...]


def evaluate_v53_certificate(
    metrics: Mapping[str, float],
    *,
    limits: Mapping[str, float] = R1_V53_CERTIFICATE_LIMITS,
) -> R1CertificateEvaluation:
    """Evaluate all 16 V53 certificate metrics against frozen limits."""
    missing = sorted(set(limits) - set(metrics))
    extra = sorted(set(metrics) - set(limits))
    if missing or extra:
        raise ValueError(
            f"certificate metrics must match the exact 16 V53 keys; missing={missing}, extra={extra}"
        )
    checked: dict[str, float] = {}
    failed: list[str] = []
    for key, limit in sorted(limits.items()):
        raw = metrics[key]
        if isinstance(raw, bool):
            raise TypeError(f"metric {key!r} must be numeric, not bool")
        val = float(raw)
        checked[key] = val
        if not math.isfinite(val) or val < 0.0 or val > float(limit):
            failed.append(key)
    return R1CertificateEvaluation(
        passed=not failed,
        metrics=MappingProxyType(checked),
        limits=MappingProxyType(dict(limits)),
        failed_items=tuple(failed),
    )


def validate_operator_channel_packet(
    packet: Mapping[str, Any],
    *,
    expected_rows: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Validate that an observation packet contains the exact 13 R1 operator channels."""
    keys = set(packet)
    expected = set(R1_OPERATOR_CHANNELS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"operator packet mismatch: missing={missing}, extra={extra}")
    shapes: dict[str, tuple[int, ...]] = {}
    first_dim: int | None = expected_rows
    for name in R1_OPERATOR_CHANNELS:
        arr: NDArray[np.float64] = np.asarray(packet[name], dtype=float)
        if arr.ndim < 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(f"channel {name!r} must be a non-empty finite numeric array")
        if first_dim is None:
            first_dim = int(arr.shape[0])
        elif int(arr.shape[0]) != first_dim:
            raise ValueError(
                f"channel {name!r} row count {arr.shape[0]} differs from expected {first_dim}"
            )
        shapes[name] = tuple(int(d) for d in arr.shape)
    return shapes


@dataclass(frozen=True, slots=True)
class R1StudyRequest:
    """Immutable, content-addressed R1 study request."""

    study_id: str
    control: str
    times_s: tuple[float, ...]
    substeps: tuple[int, int, int]
    voltage_step_V: float
    integrator: str
    precision_modes: tuple[str, ...]
    release_bundle_sha256: str
    prepared_device_sha256: str
    sha256: str

    def to_mapping(self) -> dict[str, Any]:
        body = self._body()
        return {**body, "sha256": self.sha256}

    def _body(self) -> dict[str, Any]:
        return {
            "schema": R1_STUDY_REQUEST_SCHEMA,
            "study_id": self.study_id,
            "control": self.control,
            "control_switches": dict(R1_CONTROLS[self.control]),
            "times_s": list(self.times_s),
            "substeps": list(self.substeps),
            "expected_rows_per_mode": expected_substep_rows(self.times_s, self.substeps),
            "voltage_step_V": self.voltage_step_V,
            "integrator": self.integrator,
            "precision_modes": list(self.precision_modes),
            "certificate_limits": dict(R1_V53_CERTIFICATE_LIMITS),
            "operator_channels": list(R1_OPERATOR_CHANNELS),
            "release_bundle_sha256": self.release_bundle_sha256,
            "prepared_device_sha256": self.prepared_device_sha256,
        }


def build_r1_study_request(
    *,
    study_id: str,
    control: str = "D",
    times_s: Sequence[float],
    substeps: tuple[int, int, int] = R1_CONSERVATIVE_SUBSTEPS,
    voltage_step_V: float = 5.0e-3,
    integrator: str = "conservative_backward_euler",
    precision_modes: Sequence[str] = ("baseline", "compensated"),
    release_bundle_sha256: str,
    prepared_device_sha256: str,
) -> R1StudyRequest:
    """Build and hash a validated R1StudyRequest."""
    if not isinstance(study_id, str) or not study_id.strip():
        raise ValueError("study_id must be a non-empty string")
    if control not in R1_CONTROLS:
        raise ValueError(f"control must be one of {tuple(R1_CONTROLS)}; got {control!r}")
    times = tuple(float(t) for t in times_s)
    expected_substep_rows(times, substeps)
    if tuple(substeps) != R1_CONSERVATIVE_SUBSTEPS:
        raise ValueError(f"R1 conservative stepping requires substeps={R1_CONSERVATIVE_SUBSTEPS}")
    if not math.isfinite(voltage_step_V) or voltage_step_V == 0.0:
        raise ValueError("voltage_step_V must be finite and non-zero")
    if integrator not in {"conservative_backward_euler", "ida"}:
        raise ValueError("integrator must be 'conservative_backward_euler' or 'ida'")
    modes = tuple(str(m) for m in precision_modes)
    if not modes or any(m not in {"baseline", "compensated"} for m in modes):
        raise ValueError("precision_modes must be drawn from ('baseline', 'compensated')")
    for label, digest in (
        ("release_bundle_sha256", release_bundle_sha256),
        ("prepared_device_sha256", prepared_device_sha256),
    ):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{label} must be a 64-character lowercase hex SHA-256")
    provisional = R1StudyRequest(
        study_id=study_id,
        control=control,
        times_s=times,
        substeps=(int(substeps[0]), int(substeps[1]), int(substeps[2])),
        voltage_step_V=float(voltage_step_V),
        integrator=integrator,
        precision_modes=modes,
        release_bundle_sha256=release_bundle_sha256,
        prepared_device_sha256=prepared_device_sha256,
        sha256="0" * 64,
    )
    digest = hashlib.sha256(_canonical_json_bytes(provisional._body())).hexdigest()
    return R1StudyRequest(
        study_id=provisional.study_id,
        control=provisional.control,
        times_s=provisional.times_s,
        substeps=provisional.substeps,
        voltage_step_V=provisional.voltage_step_V,
        integrator=provisional.integrator,
        precision_modes=provisional.precision_modes,
        release_bundle_sha256=provisional.release_bundle_sha256,
        prepared_device_sha256=provisional.prepared_device_sha256,
        sha256=digest,
    )


def verify_r1_study_request(payload: Mapping[str, Any]) -> R1StudyRequest:
    """Verify a serialized R1StudyRequest mapping against tampering."""
    if payload.get("schema") != R1_STUDY_REQUEST_SCHEMA:
        raise ValueError(f"unexpected study request schema: {payload.get('schema')!r}")
    substeps_raw = payload.get("substeps")
    if not isinstance(substeps_raw, list) or len(substeps_raw) != 3:
        raise ValueError("substeps must be a 3-element list")
    rebuilt = build_r1_study_request(
        study_id=str(payload["study_id"]),
        control=str(payload["control"]),
        times_s=payload["times_s"],
        substeps=(int(substeps_raw[0]), int(substeps_raw[1]), int(substeps_raw[2])),
        voltage_step_V=float(payload["voltage_step_V"]),
        integrator=str(payload["integrator"]),
        precision_modes=payload["precision_modes"],
        release_bundle_sha256=str(payload["release_bundle_sha256"]),
        prepared_device_sha256=str(payload["prepared_device_sha256"]),
    )
    if rebuilt.sha256 != payload.get("sha256"):
        raise ValueError("R1StudyRequest sha256 digest mismatch (tampered request)")
    if payload.get("certificate_limits") != dict(R1_V53_CERTIFICATE_LIMITS):
        raise ValueError("R1StudyRequest certificate_limits differ from frozen V53 limits")
    if payload.get("operator_channels") != list(R1_OPERATOR_CHANNELS):
        raise ValueError("R1StudyRequest operator_channels differ from frozen 13-channel contract")
    return rebuilt


def load_r1_study_request(raw_json: str | bytes) -> R1StudyRequest:
    """Parse and verify a JSON-encoded R1StudyRequest."""
    data = json.loads(raw_json)
    if not isinstance(data, Mapping):
        raise ValueError("R1StudyRequest JSON root must be an object")
    return verify_r1_study_request(data)


@dataclass(frozen=True, slots=True)
class R1QualificationAssessment:
    """Explicit qualification verdict separating short-window, IDA, L2, and L3."""

    qualification_level: str
    short_window_interface_verified: bool
    r1_l2_certified: bool
    r1_l3_device_qualified: bool
    reasons: tuple[str, ...]


def assess_r1_qualification(
    request: R1StudyRequest,
    *,
    baseline_certificate: R1CertificateEvaluation | None,
    compensated_certificate: R1CertificateEvaluation | None,
    baseline_rows: int,
    compensated_rows: int,
    interpolated_rows: bool = False,
    release_verified: bool = False,
) -> R1QualificationAssessment:
    """Assess R1 qualification without ever inflating short-window or IDA results to L2/L3."""
    reasons: list[str] = []
    if request.integrator != "conservative_backward_euler":
        reasons.append(
            "P07-02.4: integrator 'ida' is a method-specific comparison only and is strictly "
            "ineligible for R1-L2 qualification"
        )
    if interpolated_rows:
        reasons.append("P07-02.4: interpolated trajectory rows cannot satisfy the 4/8/16 step contract")
    if not release_verified:
        reasons.append("release bundle provenance is not verified")

    expected_rows = expected_substep_rows(request.times_s, request.substeps)
    if baseline_rows != expected_rows or compensated_rows != expected_rows:
        reasons.append(
            f"row count mismatch: expected {expected_rows}, got baseline={baseline_rows}, compensated={compensated_rows}"
        )
    if baseline_certificate is None or not baseline_certificate.passed:
        reasons.append("baseline V53 certificate did not pass")
    if compensated_certificate is None or not compensated_certificate.passed:
        reasons.append("compensated V53 certificate did not pass")

    is_full_100s = (
        expected_rows == R1_FULL_WINDOW_EXPECTED_ROWS
        and math.isclose(request.times_s[-1], 100.0, rel_tol=0.0, abs_tol=1.0e-12)
        and request.control == "D"
    )
    if not reasons and not is_full_100s:
        return R1QualificationAssessment(
            qualification_level="short_window_interface_verification_only",
            short_window_interface_verified=True,
            r1_l2_certified=False,
            r1_l3_device_qualified=False,
            reasons=(
                "short-window/control verification completed (P07-02); full 100s N256 R1-L2 "
                "(P07-03) and R1-L3 budgets (P07-04) are not claimed",
            ),
        )
    if not reasons and is_full_100s:
        return R1QualificationAssessment(
            qualification_level="r1_l2_candidate_pending_l3_budgets",
            short_window_interface_verified=True,
            r1_l2_certified=True,
            r1_l3_device_qualified=False,
            reasons=("R1-L3 independent spatial/temporal/amplitude/AC budgets (P07-04) not supplied",),
        )
    return R1QualificationAssessment(
        qualification_level="unqualified",
        short_window_interface_verified=False,
        r1_l2_certified=False,
        r1_l3_device_qualified=False,
        reasons=tuple(reasons),
    )

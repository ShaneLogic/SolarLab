"""P05-01b multi-stage working-point state transfer and paired perturbation runner.

Implements lossless ``WorkingPoint`` bundle serialization/reconstruction across
1D electronic (``CompiledDevice1D`` / ``State1D``) and coupled 1D ionic
(``CompiledIonicDevice1D`` / ``IonicState1D``) representations, explicit
stage-left (``t^-``) and stage-right (``t^+``) boundary states, strict
``Protocol.transfers`` preservation enforcement, damage-gated material
rebinding, and matched pulsed + unpulsed-control perturbation execution with
step-to-step Maxwell displacement + conduction current accounting.

Original NumericsHardening §0.2 ``EVIDENCE_LABELS`` scopes remain untouched;
run completion never promotes evidence labels or confers scientific
qualification.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from solarlab.contracts import ELEMENTARY_CHARGE_C, frozen_array
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem, classical_dc_1d
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, State1D
from solarlab.equations.implementation import use_assembled
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem, IonicState1D
from solarlab.experiments.ionic_transient import IonicTransientAttempt, IonicTransientOptions
from solarlab.experiments.protocol import Protocol, WorkingPoint
from solarlab.experiments.results import (
    EVIDENCE_LABELS,
    ExperimentResult,
    decode_array,
    encode_array,
)
from solarlab.experiments.transient import TransientAttempt, TransientOptions, TransientProtocol
from solarlab.solvers.nonlinear.newton import NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, Matrix, ResidualScaling, StepControl
from solarlab.solvers.nonlinear.production import (
    production_convergence_norm,
    production_coordinates,
    production_residual_scaling,
)
from solarlab.solvers.time import BackwardEuler1DStep, step_accounting as electronic_step_accounting
from solarlab.solvers.time.assembled_ionic_backward_euler import AssembledIonicBackwardEuler1DStep
from solarlab.solvers.time.assembled_storage import AssembledStorage1D
from solarlab.solvers.time.ionic_backward_euler import (
    IonicBackwardEuler1DStep,
    step_accounting as ionic_step_accounting,
)

__all__ = [
    "COORDINATE_ENCODING_1D",
    "COORDINATE_ENCODING_IONIC_1D",
    "PHYSICAL_FIELD_ENCODING_1D",
    "PHYSICAL_FIELD_ENCODING_IONIC_1D",
    "PairedStageRunResult",
    "StageBoundaryState",
    "StageIntegrationTrace",
    "StageTransferResult",
    "WorkingPointBundle",
    "WorkingPointTransferError",
    "build_stage_experiment_result",
    "capture_working_point",
    "compute_physical_identity",
    "coordinates_from_physical_fields_1d",
    "integrate_stage_from_working_point",
    "ion_coordinates_from_physical_densities_1d",
    "physical_state_equivalent",
    "reconstruct_ionic_state_1d",
    "reconstruct_state_1d",
    "run_paired_perturbation_stages",
    "transfer_working_point",
]

Q = ELEMENTARY_CHARGE_C

COORDINATE_ENCODING_1D = "solarlab.coordinates.1d.v1"
PHYSICAL_FIELD_ENCODING_1D = "solarlab.physical_fields.1d.v1"
COORDINATE_ENCODING_IONIC_1D = "solarlab.coordinates.ionic_1d.v1"
PHYSICAL_FIELD_ENCODING_IONIC_1D = "solarlab.physical_fields.ionic_1d.v1"

_ELECTRONIC_ENCODINGS = frozenset({COORDINATE_ENCODING_1D, PHYSICAL_FIELD_ENCODING_1D})
_IONIC_ENCODINGS = frozenset({COORDINATE_ENCODING_IONIC_1D, PHYSICAL_FIELD_ENCODING_IONIC_1D})
_SUPPORTED_ENCODINGS = _ELECTRONIC_ENCODINGS | _IONIC_ENCODINGS
_POINT_SECTION_NAMES = (
    "state",
    "inventories",
    "coordinate_reference",
    "materials",
    "damage",
    "grid",
    "history",
    "qualification",
)
_BUNDLE_SCHEMA = "solarlab.working-point-bundle.v1"


class WorkingPointTransferError(ValueError):
    """Raised when a working-point bundle or multi-stage transfer violates invariants."""


def _json_document(value: Any) -> str:
    def check(item: Any, depth: int) -> None:
        if depth > 32:
            raise ValueError("metadata exceeds the supported nesting depth")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is str:
            item.encode("utf-8")
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for key, child in item.items():
                check(key, depth + 1)
                check(child, depth + 1)
            return
        raise ValueError(f"unsupported metadata type/value: {type(item).__name__}")

    check(value, 0)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > 2**20:
        raise ValueError("metadata exceeds 1 MiB; use an independent artifact")
    return encoded


def _parse_document(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({key for key, _ in items}) != len(items):
            raise ValueError("duplicate metadata key")
        return dict(items)

    value = json.loads(text, object_pairs_hook=pairs)
    _json_document(value)
    return value


def _section_sha256(document: Mapping[str, Any]) -> str:
    canonical = _json_document(dict(document))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _ref(identifier: str, sha256: str) -> dict[str, str]:
    return {"id": identifier, "sha256": sha256}


def _canonical_f64_array(values: np.ndarray, *, name: str, positive: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise WorkingPointTransferError(f"{name} must be a non-empty finite 1D float64 vector")
    if positive and np.any(arr <= 0.0):
        raise WorkingPointTransferError(f"{name} must be strictly positive")
    # Normalize -0.0 -> +0.0 so canonical byte digests do not depend on signed zero.
    normalized = np.where(arr == 0.0, 0.0, arr)
    return np.ascontiguousarray(normalized, dtype="<f8")


def compute_physical_identity(
    *,
    nodes_m: np.ndarray,
    potential_V: np.ndarray,
    electron_density_m3: np.ndarray,
    hole_density_m3: np.ndarray,
    ion_nodes_m: np.ndarray | None = None,
    positive_ion_density_m3: np.ndarray | None = None,
    negative_ion_density_m3: np.ndarray | None = None,
) -> str:
    """Compute canonical encoding-independent SHA-256 of a 1D physical state."""
    x_e = _canonical_f64_array(nodes_m, name="nodes_m")
    phi = _canonical_f64_array(potential_V, name="potential_V")
    n = _canonical_f64_array(electron_density_m3, name="electron_density_m3", positive=True)
    p = _canonical_f64_array(hole_density_m3, name="hole_density_m3", positive=True)
    if not (x_e.shape == phi.shape == n.shape == p.shape):
        raise WorkingPointTransferError("electronic physical arrays must share identical 1D node shape")

    hasher = hashlib.sha256()
    hasher.update(b"solarlab.physical-state.1d.v1\x00")
    hasher.update(np.int64(x_e.size).astype("<i8").tobytes())
    hasher.update(x_e.tobytes())
    hasher.update(phi.tobytes())
    hasher.update(n.tobytes())
    hasher.update(p.tobytes())

    has_ion = any(arr is not None for arr in (ion_nodes_m, positive_ion_density_m3))
    if has_ion:
        if ion_nodes_m is None or positive_ion_density_m3 is None:
            raise WorkingPointTransferError("ionic physical identity requires both ion_nodes_m and positive_ion_density_m3")
        x_i = _canonical_f64_array(ion_nodes_m, name="ion_nodes_m")
        p_pos = _canonical_f64_array(
            positive_ion_density_m3, name="positive_ion_density_m3", positive=True
        )
        if x_i.shape != p_pos.shape:
            raise WorkingPointTransferError("ion_nodes_m and positive_ion_density_m3 must share identical shape")
        hasher.update(b"ionic.v1\x00")
        hasher.update(np.int64(x_i.size).astype("<i8").tobytes())
        hasher.update(x_i.tobytes())
        hasher.update(p_pos.tobytes())
        if negative_ion_density_m3 is not None and np.asarray(negative_ion_density_m3).size > 0:
            p_neg = _canonical_f64_array(
                negative_ion_density_m3, name="negative_ion_density_m3", positive=True
            )
            if p_neg.shape != x_i.shape:
                raise WorkingPointTransferError("negative_ion_density_m3 must match ion_nodes_m shape")
            hasher.update(b"dual_neg.v1\x00")
            hasher.update(p_neg.tobytes())
        else:
            hasher.update(b"single_pos.v1\x00")
    else:
        if negative_ion_density_m3 is not None and np.asarray(negative_ion_density_m3).size > 0:
            raise WorkingPointTransferError("negative_ion_density_m3 cannot be supplied without positive ions")
        hasher.update(b"electronic_only.v1\x00")

    return hasher.hexdigest()


def coordinates_from_physical_fields_1d(
    device: CompiledDevice1D,
    *,
    potential_V: np.ndarray,
    electron_density_m3: np.ndarray,
    hole_density_m3: np.ndarray,
    voltage_V: float,
) -> np.ndarray:
    """Invert ``device.fields(u, voltage_V)`` from physical ``(phi, n, p)`` to ``u``."""
    if not isinstance(device, CompiledDevice1D):
        raise TypeError("device must be a CompiledDevice1D")
    if not math.isfinite(voltage_V):
        raise WorkingPointTransferError("voltage_V must be finite")
    phi = _canonical_f64_array(potential_V, name="potential_V")
    n = _canonical_f64_array(electron_density_m3, name="electron_density_m3", positive=True)
    p = _canonical_f64_array(hole_density_m3, name="hole_density_m3", positive=True)
    if phi.shape != (device.node_count,) or n.shape != (device.node_count,) or p.shape != (device.node_count,):
        raise WorkingPointTransferError("physical field lengths do not match CompiledDevice1D.node_count")
    ref_n, ref_p = device.carrier_reference_potentials(float(voltage_V))
    shift_n = (phi - ref_n) / device.thermal_voltage_V
    shift_p = (phi - ref_p) / device.thermal_voltage_V
    z_n = np.log(n / device.reference_electron_density_m3) - shift_n
    z_p = -np.log(p / device.reference_hole_density_m3) - shift_p
    u = np.concatenate((phi, z_n, z_p))
    if not np.all(np.isfinite(u)):
        raise WorkingPointTransferError("reconstructed electronic coordinates contain non-finite entries")
    return u


def _log_ratio_no_cancel(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Compute ``log(num / den)`` for positive arrays without near-unity or deep-depletion cancellation."""
    delta = num - den
    near = np.abs(delta) <= 0.25 * den
    safe_delta_over_den = np.where(near, delta / den, 0.0)
    return np.where(near, np.log1p(safe_delta_over_den), np.log(num / den))


def ion_coordinates_from_physical_densities_1d(
    device: CompiledIonicDevice1D,
    *,
    positive_ion_density_m3: np.ndarray,
    negative_ion_density_m3: np.ndarray | None = None,
) -> np.ndarray:
    """Invert finite-site ion density maps (independent or shared-site) without cancellation."""
    if not isinstance(device, CompiledIonicDevice1D):
        raise TypeError("device must be a CompiledIonicDevice1D")
    ion_count = device.ion_geometry.node_count
    p_pos = _canonical_f64_array(
        positive_ion_density_m3, name="positive_ion_density_m3", positive=True
    )
    if p_pos.shape != (ion_count,):
        raise WorkingPointTransferError("positive_ion_density_m3 shape does not match ion_geometry.node_count")
    p0_pos = device.positive_ion_reference_m3
    plim_pos = device.positive_ion_site_limit_m3
    if np.any(p_pos >= plim_pos):
        raise WorkingPointTransferError("positive_ion_density_m3 exceeds or saturates positive_ion_site_limit_m3")

    if device.ion_species_count == 1:
        if negative_ion_density_m3 is not None and np.asarray(negative_ion_density_m3).size > 0:
            raise WorkingPointTransferError("single-ion device cannot accept negative_ion_density_m3")
        y_pos = _log_ratio_no_cancel(p_pos, p0_pos) - _log_ratio_no_cancel(
            plim_pos - p_pos, plim_pos - p0_pos
        )
        if not np.all(np.isfinite(y_pos)):
            raise WorkingPointTransferError("non-finite single-ion logit coordinate")
        return y_pos

    if negative_ion_density_m3 is None or np.asarray(negative_ion_density_m3).size == 0:
        raise WorkingPointTransferError("dual-ion device requires negative_ion_density_m3")
    p_neg = _canonical_f64_array(
        negative_ion_density_m3, name="negative_ion_density_m3", positive=True
    )
    if p_neg.shape != (ion_count,):
        raise WorkingPointTransferError("negative_ion_density_m3 shape does not match ion_geometry.node_count")
    p0_neg = device.negative_ion_reference_m3
    plim_neg = device.negative_ion_site_limit_m3
    if np.any(p_neg >= plim_neg):
        raise WorkingPointTransferError("negative_ion_density_m3 exceeds or saturates negative_ion_site_limit_m3")

    if device.ion_site_contract == "shared_finite_sites":
        if np.any(p_pos + p_neg >= plim_pos):
            raise WorkingPointTransferError("shared-site ion densities saturate common site capacity")
        vac_term = _log_ratio_no_cancel(plim_pos - p_pos - p_neg, plim_pos - p0_pos - p0_neg)
        y_pos = _log_ratio_no_cancel(p_pos, p0_pos) - vac_term
        y_neg = _log_ratio_no_cancel(p_neg, p0_neg) - vac_term
    else:
        y_pos = _log_ratio_no_cancel(p_pos, p0_pos) - _log_ratio_no_cancel(
            plim_pos - p_pos, plim_pos - p0_pos
        )
        y_neg = _log_ratio_no_cancel(p_neg, p0_neg) - _log_ratio_no_cancel(
            plim_neg - p_neg, plim_neg - p0_neg
        )
    coords = np.concatenate((y_pos, y_neg))
    if not np.all(np.isfinite(coords)):
        raise WorkingPointTransferError("non-finite dual-ion logit coordinates")
    return coords


def physical_state_equivalent(
    left: State1D | IonicState1D,
    right: State1D | IonicState1D,
    *,
    rtol: float = 1e-13,
    atol: float = 0.0,
) -> bool:
    """Check whether two reconstructed states represent the same physical state."""
    if isinstance(left, State1D) and isinstance(right, State1D):
        return bool(
            np.allclose(left.potential_V, right.potential_V, rtol=rtol, atol=max(atol, 1e-15))
            and np.allclose(left.electron_density_m3, right.electron_density_m3, rtol=rtol, atol=atol)
            and np.allclose(left.hole_density_m3, right.hole_density_m3, rtol=rtol, atol=atol)
        )
    if isinstance(left, IonicState1D) and isinstance(right, IonicState1D):
        if not physical_state_equivalent(
            left.electronic_state, right.electronic_state, rtol=rtol, atol=atol
        ):
            return False
        if not np.allclose(left.positive_ion_density_m3, right.positive_ion_density_m3, rtol=rtol, atol=atol):
            return False
        if left.negative_ion_density_m3.size != right.negative_ion_density_m3.size:
            return False
        if left.negative_ion_density_m3.size > 0 and not np.allclose(
            left.negative_ion_density_m3, right.negative_ion_density_m3, rtol=rtol, atol=atol
        ):
            return False
        return True
    return False


@dataclass(frozen=True, slots=True)
class WorkingPointBundle:
    """Complete self-verifying working-point document pairing ``WorkingPoint`` + section payloads."""

    working_point: WorkingPoint
    sections: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.working_point, WorkingPoint):
            raise TypeError("working_point must be a WorkingPoint instance")
        if not isinstance(self.sections, Mapping) or set(self.sections) != set(_POINT_SECTION_NAMES):
            raise WorkingPointTransferError("WorkingPointBundle requires all 8 working-point sections")
        wp_doc = self.working_point.export()
        encoding = wp_doc["state_encoding"]
        if encoding not in _SUPPORTED_ENCODINGS:
            raise WorkingPointTransferError(f"unsupported state_encoding: {encoding!r}")
        normalized_sections: dict[str, dict[str, Any]] = {}
        for name in _POINT_SECTION_NAMES:
            raw_section = self.sections[name]
            if not isinstance(raw_section, Mapping):
                raise WorkingPointTransferError(f"section {name!r} must be a JSON object")
            canonical_str = _json_document(dict(raw_section))
            digest = hashlib.sha256(canonical_str.encode()).hexdigest()
            expected_digest = wp_doc[name]["sha256"]
            if digest != expected_digest:
                raise WorkingPointTransferError(
                    f"section {name!r} SHA-256 mismatch ({digest} != {expected_digest})"
                )
            normalized_sections[name] = _parse_document(canonical_str)

        # Verify qualification invariants (7-tier labels, scientific_qualification never auto-true).
        qual = normalized_sections["qualification"]
        if qual.get("scientific_qualification") is not False:
            raise WorkingPointTransferError(
                "WorkingPoint qualification cannot assert scientific_qualification=True via runtime transfer"
            )
        labels = qual.get("evidence_labels", [])
        if not isinstance(labels, list) or not labels or any(lbl not in EVIDENCE_LABELS for lbl in labels):
            raise WorkingPointTransferError("qualification.evidence_labels must contain valid EVIDENCE_LABELS")

        # Verify physical_identity directly when physical fields are inline.
        state_sec = normalized_sections["state"]
        grid_sec = normalized_sections["grid"]
        if encoding in {PHYSICAL_FIELD_ENCODING_1D, PHYSICAL_FIELD_ENCODING_IONIC_1D}:
            nodes_m = decode_array(grid_sec["electronic_nodes_m"])
            phi = decode_array(state_sec["potential_V"])
            n = decode_array(state_sec["electron_density_m3"])
            p = decode_array(state_sec["hole_density_m3"])
            ion_nodes = (
                decode_array(grid_sec["ion_nodes_m"])
                if encoding == PHYSICAL_FIELD_ENCODING_IONIC_1D
                else None
            )
            p_pos = (
                decode_array(state_sec["positive_ion_density_m3"])
                if encoding == PHYSICAL_FIELD_ENCODING_IONIC_1D
                else None
            )
            p_neg = (
                decode_array(state_sec["negative_ion_density_m3"])
                if encoding == PHYSICAL_FIELD_ENCODING_IONIC_1D
                and state_sec.get("negative_ion_density_m3") is not None
                else None
            )
            computed_pid = compute_physical_identity(
                nodes_m=nodes_m,
                potential_V=phi,
                electron_density_m3=n,
                hole_density_m3=p,
                ion_nodes_m=ion_nodes,
                positive_ion_density_m3=p_pos,
                negative_ion_density_m3=p_neg,
            )
            if computed_pid != wp_doc["physical_identity"]:
                raise WorkingPointTransferError(
                    "physical_identity mismatch against inline physical-field state arrays"
                )
        object.__setattr__(self, "sections", normalized_sections)

    @property
    def id(self) -> str:
        return str(self.working_point.export()["id"])

    @property
    def sha256(self) -> str:
        return self.working_point.sha256

    @property
    def physical_identity(self) -> str:
        return str(self.working_point.export()["physical_identity"])

    @property
    def state_encoding(self) -> str:
        return str(self.working_point.export()["state_encoding"])

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": _BUNDLE_SCHEMA,
            "working_point": self.working_point.export(),
            "sections": _parse_document(_json_document(self.sections)),
        }

    def to_json(self) -> str:
        return _json_document(self.to_document())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> WorkingPointBundle:
        if not isinstance(document, Mapping) or set(document) != {"schema", "working_point", "sections"}:
            raise WorkingPointTransferError("invalid WorkingPointBundle document structure")
        if document["schema"] != _BUNDLE_SCHEMA:
            raise WorkingPointTransferError(f"unsupported bundle schema: {document['schema']!r}")
        wp_raw = document["working_point"]
        if not isinstance(wp_raw, Mapping):
            raise WorkingPointTransferError("working_point must be a JSON object")
        wp = WorkingPoint.from_document(dict(wp_raw))
        sections_raw = document["sections"]
        if not isinstance(sections_raw, Mapping):
            raise WorkingPointTransferError("sections must be a JSON object")
        return cls(working_point=wp, sections={str(k): dict(v) for k, v in sections_raw.items()})

    @classmethod
    def from_json(cls, text: str) -> WorkingPointBundle:
        parsed = _parse_document(text)
        if not isinstance(parsed, dict):
            raise WorkingPointTransferError("WorkingPointBundle JSON must decode to an object")
        return cls.from_document(parsed)


def _carrier_inventories(device: CompiledDevice1D, state: State1D) -> tuple[float, float]:
    widths = device.geometry.cell_measures_m
    elec_inv = math.fsum((widths * state.electron_density_m3).tolist())
    hole_inv = math.fsum((widths * state.hole_density_m3).tolist())
    return elec_inv, hole_inv


def capture_working_point(
    device: CompiledDevice1D | CompiledIonicDevice1D,
    state: State1D | IonicState1D,
    *,
    point_id: str,
    owner_epoch: str,
    stage_id: str,
    time_s: float,
    generation_scale: float,
    is_equilibrium: bool,
    encoding: Literal["coordinates", "physical_fields"] = "coordinates",
    reference_voltage_V: float | None = None,
    prepared: PreparedDevice | None = None,
    damage_state: Mapping[str, Any] | None = None,
    switch_state: Mapping[str, Any] | None = None,
    prior_history: Mapping[str, Any] | None = None,
    evidence_labels: Sequence[str] = ("FIRST_SLICE",),
    limitations: Sequence[str] = (),
) -> WorkingPointBundle:
    """Capture a lossless ``WorkingPointBundle`` from an electronic or ionic 1D state."""
    if not isinstance(point_id, str) or not point_id.strip():
        raise WorkingPointTransferError("point_id must be a non-empty string")
    if not isinstance(owner_epoch, str) or not owner_epoch.strip():
        raise WorkingPointTransferError("owner_epoch must be a non-empty string")
    if not isinstance(stage_id, str) or not stage_id.strip():
        raise WorkingPointTransferError("stage_id must be a non-empty string")
    if not math.isfinite(time_s) or not math.isfinite(generation_scale) or generation_scale < 0.0:
        raise WorkingPointTransferError("time_s must be finite and generation_scale finite non-negative")
    if encoding not in {"coordinates", "physical_fields"}:
        raise WorkingPointTransferError("encoding must be 'coordinates' or 'physical_fields'")
    if not evidence_labels or any(lbl not in EVIDENCE_LABELS for lbl in evidence_labels):
        raise WorkingPointTransferError("evidence_labels must be non-empty valid EVIDENCE_LABELS")

    is_ionic = isinstance(device, CompiledIonicDevice1D)
    if isinstance(device, CompiledIonicDevice1D):
        if not isinstance(state, IonicState1D):
            raise WorkingPointTransferError("CompiledIonicDevice1D requires an IonicState1D")
        if state.device_sha256 != device.identity:
            raise WorkingPointTransferError("IonicState1D belongs to a different CompiledIonicDevice1D")
        elec_device: CompiledDevice1D = device.electronic_component
        elec_state: State1D = state.electronic_state
    else:
        if not isinstance(device, CompiledDevice1D) or not isinstance(state, State1D):
            raise WorkingPointTransferError("CompiledDevice1D requires a State1D")
        if state.device_sha256 != device.identity:
            raise WorkingPointTransferError("State1D belongs to a different CompiledDevice1D")
        elec_device = device
        elec_state = state

    applied_voltage_V = float(state.voltage_V)
    ref_voltage_V = applied_voltage_V if reference_voltage_V is None else float(reference_voltage_V)
    if not math.isfinite(ref_voltage_V):
        raise WorkingPointTransferError("reference_voltage_V must be finite")

    ion_nodes = device.ion_geometry.nodes_m if isinstance(device, CompiledIonicDevice1D) else None
    p_pos = state.positive_ion_density_m3 if isinstance(state, IonicState1D) else None
    p_neg = (
        state.negative_ion_density_m3
        if isinstance(state, IonicState1D) and state.negative_ion_density_m3.size > 0
        else None
    )
    physical_id = compute_physical_identity(
        nodes_m=elec_device.nodes_m,
        potential_V=elec_state.potential_V,
        electron_density_m3=elec_state.electron_density_m3,
        hole_density_m3=elec_state.hole_density_m3,
        ion_nodes_m=ion_nodes,
        positive_ion_density_m3=p_pos,
        negative_ion_density_m3=p_neg,
    )

    if encoding == "coordinates":
        state_encoding = COORDINATE_ENCODING_IONIC_1D if is_ionic else COORDINATE_ENCODING_1D
        if ref_voltage_V == applied_voltage_V:
            u_elec = np.asarray(elec_state.u, dtype=np.float64)
        else:
            u_elec = coordinates_from_physical_fields_1d(
                elec_device,
                potential_V=elec_state.potential_V,
                electron_density_m3=elec_state.electron_density_m3,
                hole_density_m3=elec_state.hole_density_m3,
                voltage_V=ref_voltage_V,
            )
        if isinstance(device, CompiledIonicDevice1D) and isinstance(state, IonicState1D):
            u_ion = np.asarray(state.u[3 * device.node_count :], dtype=np.float64)
            u_full = np.concatenate((u_elec, u_ion))
            state_section: dict[str, Any] = {
                "schema": state_encoding,
                "u": encode_array(np.ascontiguousarray(u_full, dtype=np.float64), unit="1"),
                "applied_voltage_V": applied_voltage_V,
                "reference_voltage_V": ref_voltage_V,
                "ion_species_count": int(device.ion_species_count),
                "ion_site_contract": device.ion_site_contract,
            }
        else:
            state_section = {
                "schema": state_encoding,
                "u": encode_array(np.ascontiguousarray(u_elec, dtype=np.float64), unit="1"),
                "applied_voltage_V": applied_voltage_V,
                "reference_voltage_V": ref_voltage_V,
            }
        coord_ref_section: dict[str, Any] = {
            "schema": "solarlab.coordinate-reference.v1",
            "representation": "coordinates",
            "reference_voltage_V": ref_voltage_V,
            "thermal_voltage_V": float(elec_device.thermal_voltage_V),
            "voltage_polarity": float(elec_device.voltage_polarity),
            "ion_site_contract": (
                device.ion_site_contract if isinstance(device, CompiledIonicDevice1D) else "none"
            ),
        }
    else:
        state_encoding = PHYSICAL_FIELD_ENCODING_IONIC_1D if is_ionic else PHYSICAL_FIELD_ENCODING_1D
        state_section = {
            "schema": state_encoding,
            "applied_voltage_V": applied_voltage_V,
            "potential_V": encode_array(
                np.ascontiguousarray(elec_state.potential_V, dtype=np.float64), unit="V"
            ),
            "electron_density_m3": encode_array(
                np.ascontiguousarray(elec_state.electron_density_m3, dtype=np.float64), unit="m^-3"
            ),
            "hole_density_m3": encode_array(
                np.ascontiguousarray(elec_state.hole_density_m3, dtype=np.float64), unit="m^-3"
            ),
        }
        if isinstance(device, CompiledIonicDevice1D) and isinstance(state, IonicState1D):
            state_section["positive_ion_density_m3"] = encode_array(
                np.ascontiguousarray(state.positive_ion_density_m3, dtype=np.float64), unit="m^-3"
            )
            state_section["negative_ion_density_m3"] = (
                encode_array(
                    np.ascontiguousarray(state.negative_ion_density_m3, dtype=np.float64), unit="m^-3"
                )
                if state.negative_ion_density_m3.size > 0
                else None
            )
            state_section["ion_species_count"] = int(device.ion_species_count)
            state_section["ion_site_contract"] = device.ion_site_contract
        coord_ref_section = {
            "schema": "solarlab.coordinate-reference.v1",
            "representation": "physical_fields",
            "reference_voltage_V": None,
            "thermal_voltage_V": float(elec_device.thermal_voltage_V),
            "voltage_polarity": float(elec_device.voltage_polarity),
            "ion_site_contract": (
                device.ion_site_contract if isinstance(device, CompiledIonicDevice1D) else "none"
            ),
        }

    elec_inv, hole_inv = _carrier_inventories(elec_device, elec_state)
    inventories_section: dict[str, Any] = {
        "schema": "solarlab.inventories.v1",
        "electron_inventory_m2": elec_inv,
        "hole_inventory_m2": hole_inv,
        "positive_ion_inventory_m2": (
            float(state.domain_inventory_m2) if isinstance(state, IonicState1D) else 0.0
        ),
        "negative_ion_inventory_m2": (
            float(state.negative_domain_inventory_m2) if isinstance(state, IonicState1D) else 0.0
        ),
    }

    prepared_sha = (
        prepared.content_sha256
        if prepared is not None
        else (
            device.original_prepared_sha256
            if isinstance(device, CompiledIonicDevice1D)
            else elec_device.parameters.sha256
        )
    )
    materials_section: dict[str, Any] = {
        "schema": "solarlab.materials-binding.v1",
        "device_id": device.device_id,
        "compiled_identity": device.identity,
        "prepared_sha256": prepared_sha,
        "family": "ionic_1d" if is_ionic else "electronic_1d",
    }

    damage_payload = (
        {"schema": "solarlab.damage-state.v1", "status": "undamaged", "parameters": {}}
        if damage_state is None
        else {
            "schema": "solarlab.damage-state.v1",
            **{k: v for k, v in damage_state.items() if k != "schema"},
        }
    )

    grid_section: dict[str, Any] = {
        "schema": "solarlab.grid-binding.v1",
        "geometry_name": elec_device.geometry.name,
        "node_count": int(elec_device.node_count),
        "electronic_nodes_m": encode_array(
            np.ascontiguousarray(elec_device.nodes_m, dtype=np.float64), unit="m"
        ),
        "electronic_cell_measures_m": encode_array(
            np.ascontiguousarray(elec_device.geometry.cell_measures_m, dtype=np.float64), unit="m"
        ),
    }
    if isinstance(device, CompiledIonicDevice1D):
        grid_section["ion_domain_layer"] = device.domain_layer_id
        grid_section["ion_node_indices"] = [int(i) for i in device.ion_node_indices.tolist()]
        grid_section["ion_nodes_m"] = encode_array(
            np.ascontiguousarray(device.ion_geometry.nodes_m, dtype=np.float64), unit="m"
        )
        grid_section["ion_cell_measures_m"] = encode_array(
            np.ascontiguousarray(device.ion_geometry.cell_measures_m, dtype=np.float64), unit="m"
        )

    switches_doc = (
        {"illumination": "on" if generation_scale > 0.0 else "dark", "circuit": "voltage_source"}
        if switch_state is None
        else dict(switch_state)
    )
    prior_stages = (
        list(prior_history.get("prior_stages", []))
        if prior_history is not None and isinstance(prior_history.get("prior_stages"), list)
        else []
    )
    history_section: dict[str, Any] = {
        "schema": "solarlab.stage-history.v1",
        "stage_id": stage_id,
        "boundary_side": str(prior_history.get("boundary_side", "interior"))
        if prior_history is not None
        else "interior",
        "time_s": float(time_s),
        "applied_voltage_V": applied_voltage_V,
        "generation_scale": float(generation_scale),
        "switches": switches_doc,
        "is_equilibrium": bool(is_equilibrium),
        "declared_state_reset": bool(
            prior_history.get("declared_state_reset", False) if prior_history is not None else False
        ),
        "prior_stages": prior_stages,
    }

    qualification_section: dict[str, Any] = {
        "schema": "solarlab.qualification-record.v1",
        "scientific_qualification": False,
        "evidence_labels": list(dict.fromkeys(evidence_labels)),
        "limitations": list(dict.fromkeys(limitations)),
    }

    sections: dict[str, dict[str, Any]] = {
        "state": state_section,
        "inventories": inventories_section,
        "coordinate_reference": coord_ref_section,
        "materials": materials_section,
        "damage": damage_payload,
        "grid": grid_section,
        "history": history_section,
        "qualification": qualification_section,
    }
    wp_doc: dict[str, Any] = {
        "schema": "solarlab.working-point.v1",
        "id": point_id,
        "engine": "solarlab",
        "owner_epoch": owner_epoch,
        "physical_identity": physical_id,
        "state_encoding": state_encoding,
        **{name: _ref(f"{point_id}:{name}", _section_sha256(sections[name])) for name in _POINT_SECTION_NAMES},
    }
    wp = WorkingPoint.from_document(wp_doc)
    return WorkingPointBundle(working_point=wp, sections=sections)


def _verify_grid_against_device(
    bundle: WorkingPointBundle, device: CompiledDevice1D | CompiledIonicDevice1D
) -> None:
    grid_sec = bundle.sections["grid"]
    elec_device = device.electronic_component if isinstance(device, CompiledIonicDevice1D) else device
    nodes = decode_array(grid_sec["electronic_nodes_m"])
    measures = decode_array(grid_sec["electronic_cell_measures_m"])
    if nodes.shape != elec_device.nodes_m.shape or not np.array_equal(nodes, elec_device.nodes_m):
        raise WorkingPointTransferError("electronic grid nodes disagree between WorkingPointBundle and device")
    if measures.shape != elec_device.geometry.cell_measures_m.shape or not np.array_equal(
        measures, elec_device.geometry.cell_measures_m
    ):
        raise WorkingPointTransferError("electronic control volumes disagree between WorkingPointBundle and device")
    if isinstance(device, CompiledIonicDevice1D):
        if "ion_nodes_m" not in grid_sec:
            raise WorkingPointTransferError("ionic device reconstruction requires ion_nodes_m in grid section")
        ion_nodes = decode_array(grid_sec["ion_nodes_m"])
        if ion_nodes.shape != device.ion_geometry.nodes_m.shape or not np.array_equal(
            ion_nodes, device.ion_geometry.nodes_m
        ):
            raise WorkingPointTransferError("ionic grid nodes disagree between WorkingPointBundle and device")


def reconstruct_state_1d(
    bundle: WorkingPointBundle,
    device: CompiledDevice1D,
    *,
    target_voltage_V: float | None = None,
    require_same_materials: bool = True,
    inventory_rtol: float = 1e-12,
) -> State1D:
    """Reconstruct a ``State1D`` from either coordinate-encoded or physical-field-encoded bundle."""
    if not isinstance(bundle, WorkingPointBundle) or not isinstance(device, CompiledDevice1D):
        raise TypeError("reconstruct_state_1d requires WorkingPointBundle and CompiledDevice1D")
    if bundle.state_encoding not in _ELECTRONIC_ENCODINGS:
        raise WorkingPointTransferError(
            f"reconstruct_state_1d expects electronic 1D encoding, got {bundle.state_encoding!r}"
        )
    if require_same_materials and bundle.sections["materials"]["compiled_identity"] != device.identity:
        raise WorkingPointTransferError(
            "WorkingPointBundle materials.compiled_identity disagrees with CompiledDevice1D.identity"
        )
    _verify_grid_against_device(bundle, device)
    state_sec = bundle.sections["state"]
    applied_v = float(state_sec["applied_voltage_V"]) if target_voltage_V is None else float(target_voltage_V)
    if not math.isfinite(applied_v):
        raise WorkingPointTransferError("target_voltage_V must be finite")

    if bundle.state_encoding == COORDINATE_ENCODING_1D:
        u_raw = decode_array(state_sec["u"])
        ref_v = float(state_sec["reference_voltage_V"])
        if u_raw.shape != (3 * device.node_count,):
            raise WorkingPointTransferError("encoded u shape does not match 3 * device.node_count")
        u = u_raw if ref_v == applied_v else device.rebase_coordinates(u_raw, ref_v, applied_v)
    else:
        phi = decode_array(state_sec["potential_V"])
        n = decode_array(state_sec["electron_density_m3"])
        p = decode_array(state_sec["hole_density_m3"])
        u = coordinates_from_physical_fields_1d(
            device,
            potential_V=phi,
            electron_density_m3=n,
            hole_density_m3=p,
            voltage_V=applied_v,
        )

    reconstructed = device.state(u, voltage_V=applied_v)
    elec_inv, hole_inv = _carrier_inventories(device, reconstructed)
    inv_sec = bundle.sections["inventories"]
    if not math.isclose(elec_inv, float(inv_sec["electron_inventory_m2"]), rel_tol=inventory_rtol, abs_tol=1e-18):
        raise WorkingPointTransferError("reconstructed electron inventory diverges from recorded inventory")
    if not math.isclose(hole_inv, float(inv_sec["hole_inventory_m2"]), rel_tol=inventory_rtol, abs_tol=1e-18):
        raise WorkingPointTransferError("reconstructed hole inventory diverges from recorded inventory")
    return reconstructed


def reconstruct_ionic_state_1d(
    bundle: WorkingPointBundle,
    device: CompiledIonicDevice1D,
    *,
    target_voltage_V: float | None = None,
    require_same_materials: bool = True,
    inventory_rtol: float = 1e-12,
) -> IonicState1D:
    """Reconstruct an ``IonicState1D`` from coordinate-encoded or physical-field-encoded bundle."""
    if not isinstance(bundle, WorkingPointBundle) or not isinstance(device, CompiledIonicDevice1D):
        raise TypeError("reconstruct_ionic_state_1d requires WorkingPointBundle and CompiledIonicDevice1D")
    if bundle.state_encoding not in _IONIC_ENCODINGS:
        raise WorkingPointTransferError(
            f"reconstruct_ionic_state_1d expects ionic 1D encoding, got {bundle.state_encoding!r}"
        )
    if require_same_materials and bundle.sections["materials"]["compiled_identity"] != device.identity:
        raise WorkingPointTransferError(
            "WorkingPointBundle materials.compiled_identity disagrees with CompiledIonicDevice1D.identity"
        )
    _verify_grid_against_device(bundle, device)
    state_sec = bundle.sections["state"]
    applied_v = float(state_sec["applied_voltage_V"]) if target_voltage_V is None else float(target_voltage_V)
    if not math.isfinite(applied_v):
        raise WorkingPointTransferError("target_voltage_V must be finite")

    elec_count = 3 * device.node_count
    if bundle.state_encoding == COORDINATE_ENCODING_IONIC_1D:
        u_raw = decode_array(state_sec["u"])
        if u_raw.shape != (device.state_size,):
            raise WorkingPointTransferError("encoded ionic u shape does not match device.state_size")
        ref_v = float(state_sec["reference_voltage_V"])
        u_elec = (
            u_raw[:elec_count]
            if ref_v == applied_v
            else device.electronic_component.rebase_coordinates(u_raw[:elec_count], ref_v, applied_v)
        )
        u_full = np.concatenate((u_elec, u_raw[elec_count:]))
    else:
        phi = decode_array(state_sec["potential_V"])
        n = decode_array(state_sec["electron_density_m3"])
        p = decode_array(state_sec["hole_density_m3"])
        p_pos = decode_array(state_sec["positive_ion_density_m3"])
        p_neg = (
            decode_array(state_sec["negative_ion_density_m3"])
            if state_sec.get("negative_ion_density_m3") is not None
            else None
        )
        u_elec = coordinates_from_physical_fields_1d(
            device.electronic_component,
            potential_V=phi,
            electron_density_m3=n,
            hole_density_m3=p,
            voltage_V=applied_v,
        )
        u_ion = ion_coordinates_from_physical_densities_1d(
            device,
            positive_ion_density_m3=p_pos,
            negative_ion_density_m3=p_neg,
        )
        u_full = np.concatenate((u_elec, u_ion))

    reconstructed = device.state(u_full, voltage_V=applied_v)
    elec_inv, hole_inv = _carrier_inventories(device.electronic_component, reconstructed.electronic_state)
    inv_sec = bundle.sections["inventories"]
    if not math.isclose(elec_inv, float(inv_sec["electron_inventory_m2"]), rel_tol=inventory_rtol, abs_tol=1e-18):
        raise WorkingPointTransferError("reconstructed electron inventory diverges from recorded inventory")
    if not math.isclose(hole_inv, float(inv_sec["hole_inventory_m2"]), rel_tol=inventory_rtol, abs_tol=1e-18):
        raise WorkingPointTransferError("reconstructed hole inventory diverges from recorded inventory")
    if not math.isclose(
        reconstructed.domain_inventory_m2,
        float(inv_sec["positive_ion_inventory_m2"]),
        rel_tol=inventory_rtol,
        abs_tol=1e-18,
    ):
        raise WorkingPointTransferError("reconstructed positive ion inventory diverges from recorded inventory")
    if device.ion_species_count == 2 and not math.isclose(
        reconstructed.negative_domain_inventory_m2,
        float(inv_sec["negative_ion_inventory_m2"]),
        rel_tol=inventory_rtol,
        abs_tol=1e-18,
    ):
        raise WorkingPointTransferError("reconstructed negative ion inventory diverges from recorded inventory")
    return reconstructed


@dataclass(frozen=True, slots=True)
class StageBoundaryState:
    """Explicit stage-left (``t^-``) and stage-right (``t^+``) boundary states."""

    from_stage: str
    to_stage: str
    boundary_time_s: float
    preserved_quantities: tuple[str, ...]
    left_bundle: WorkingPointBundle
    right_bundle: WorkingPointBundle


@dataclass(frozen=True, slots=True)
class StageTransferResult:
    """Validated stage-to-stage transfer outcome with rebound device and right-boundary state."""

    boundary: StageBoundaryState
    compiled_device: CompiledDevice1D | CompiledIonicDevice1D
    right_state: State1D | IonicState1D
    rebound_materials: bool


def transfer_working_point(
    protocol: Protocol,
    source_bundle: WorkingPointBundle,
    compiled_device: CompiledDevice1D | CompiledIonicDevice1D,
    *,
    from_stage: str,
    to_stage: str,
    boundary_time_s: float,
    right_voltage_V: float,
    right_generation_scale: float,
    right_point_id: str,
    owner_epoch: str,
    right_encoding: Literal["coordinates", "physical_fields"] | None = None,
    right_switches: Mapping[str, Any] | None = None,
    next_damage_state: Mapping[str, Any] | None = None,
    rebind_damaged_device: Callable[
        [Mapping[str, Any]], CompiledDevice1D | CompiledIonicDevice1D
    ]
    | None = None,
    declared_reset_state: State1D | IonicState1D | None = None,
    allow_declared_reset: bool = False,
) -> StageTransferResult:
    """Execute a protocol-governed ``t^- -> t^+`` stage boundary transfer.

    Enforces ``preserve`` invariants strictly:
    - Never silently resets state to steady state; state reset requires both
      ``"state"`` omitted from ``preserve`` AND explicit ``declared_reset_state``
      with ``allow_declared_reset=True``.
    - Rebinds ``CompiledDevice1D`` / ``CompiledIonicDevice1D`` if and only if
      ``damage`` digest changes across the stage boundary.
    """
    if not isinstance(protocol, Protocol) or not isinstance(source_bundle, WorkingPointBundle):
        raise TypeError("transfer_working_point requires Protocol and WorkingPointBundle")
    proto_doc = protocol.export()
    matching = [
        entry
        for entry in proto_doc["transfers"]
        if entry["from"] == from_stage and entry["to"] == to_stage
    ]
    if len(matching) != 1:
        raise WorkingPointTransferError(
            f"protocol must declare exactly one transfer for {from_stage!r} -> {to_stage!r}"
        )
    transfer_decl = matching[0]
    expected_wp = transfer_decl["working_point"]
    if expected_wp["id"] != source_bundle.id or expected_wp["sha256"] != source_bundle.sha256:
        raise WorkingPointTransferError(
            "source_bundle working_point id/sha256 disagrees with Protocol.transfers declaration"
        )
    preserve = tuple(str(item) for item in transfer_decl["preserve"])
    preserve_set = set(preserve)

    # Damage & material rebinding rule: rebind ONLY when damage digest changes.
    old_damage_doc = source_bundle.sections["damage"]
    new_damage_doc = (
        old_damage_doc
        if next_damage_state is None
        else {
            "schema": "solarlab.damage-state.v1",
            **{k: v for k, v in next_damage_state.items() if k != "schema"},
        }
    )
    damage_changed = _section_sha256(new_damage_doc) != _section_sha256(old_damage_doc)

    if damage_changed:
        if "damage" in preserve_set or "materials" in preserve_set:
            raise WorkingPointTransferError(
                "damage state changed across boundary while protocol preserves 'damage'/'materials'"
            )
        if rebind_damaged_device is None:
            raise WorkingPointTransferError(
                "damage state changed across boundary; explicit rebind_damaged_device callback is required"
            )
        target_device = rebind_damaged_device(new_damage_doc)
        rebound_materials = True
    else:
        if rebind_damaged_device is not None and "materials" in preserve_set:
            # Even if a callback was passed, unchanged damage MUST NOT trigger material rebinding.
            target_device = compiled_device
        else:
            target_device = compiled_device
        rebound_materials = False
        if source_bundle.sections["materials"]["compiled_identity"] != target_device.identity:
            raise WorkingPointTransferError(
                "compiled_device identity differs from source_bundle without a damage transition"
            )

    if "grid" in preserve_set:
        _verify_grid_against_device(source_bundle, target_device)

    target_encoding: Literal["coordinates", "physical_fields"]
    if right_encoding is None:
        target_encoding = (
            "coordinates"
            if source_bundle.state_encoding in {COORDINATE_ENCODING_1D, COORDINATE_ENCODING_IONIC_1D}
            else "physical_fields"
        )
    else:
        target_encoding = right_encoding

    left_coord_ref = source_bundle.sections["coordinate_reference"]
    if "coordinate_reference" in preserve_set:
        expected_repr = str(left_coord_ref["representation"])
        if target_encoding != expected_repr:
            raise WorkingPointTransferError(
                "protocol preserves 'coordinate_reference'; representation encoding cannot change"
            )
        if target_encoding == "coordinates" and float(right_voltage_V) != float(
            left_coord_ref["reference_voltage_V"]
        ):
            raise WorkingPointTransferError(
                "protocol preserves 'coordinate_reference'; coordinate reference voltage cannot jump"
            )

    # Reconstruct left physical state on source device first so t^- is verified intact.
    if isinstance(compiled_device, CompiledIonicDevice1D):
        left_state: State1D | IonicState1D = reconstruct_ionic_state_1d(source_bundle, compiled_device)
    else:
        left_state = reconstruct_state_1d(source_bundle, compiled_device)

    # Enforce strict state preservation vs explicitly declared reset.
    if declared_reset_state is not None:
        if "state" in preserve_set:
            raise WorkingPointTransferError(
                "protocol preserves 'state'; silent or declared reset to equilibrium is forbidden"
            )
        if not allow_declared_reset:
            raise WorkingPointTransferError(
                "declared_reset_state requires allow_declared_reset=True and 'state' omitted from preserve"
            )
        right_state = declared_reset_state
        was_reset = True
    else:
        if "state" not in preserve_set and not allow_declared_reset:
            raise WorkingPointTransferError(
                "protocol omits 'state' from preserve without an explicit authorized declared_reset_state"
            )
        was_reset = False
        if rebound_materials:
            if isinstance(target_device, CompiledIonicDevice1D):
                if not isinstance(left_state, IonicState1D):
                    raise WorkingPointTransferError("ionic target_device requires ionic left_state")
                u_elec_reb = coordinates_from_physical_fields_1d(
                    target_device.electronic_component,
                    potential_V=left_state.electronic_state.potential_V,
                    electron_density_m3=left_state.electronic_state.electron_density_m3,
                    hole_density_m3=left_state.electronic_state.hole_density_m3,
                    voltage_V=right_voltage_V,
                )
                u_ion_reb = ion_coordinates_from_physical_densities_1d(
                    target_device,
                    positive_ion_density_m3=left_state.positive_ion_density_m3,
                    negative_ion_density_m3=(
                        left_state.negative_ion_density_m3
                        if left_state.negative_ion_density_m3.size > 0
                        else None
                    ),
                )
                right_state = target_device.state(
                    np.concatenate((u_elec_reb, u_ion_reb)), voltage_V=right_voltage_V
                )
            else:
                if not isinstance(left_state, State1D):
                    raise WorkingPointTransferError("electronic target_device requires electronic left_state")
                u_reb = coordinates_from_physical_fields_1d(
                    target_device,
                    potential_V=left_state.potential_V,
                    electron_density_m3=left_state.electron_density_m3,
                    hole_density_m3=left_state.hole_density_m3,
                    voltage_V=right_voltage_V,
                )
                right_state = target_device.state(u_reb, voltage_V=right_voltage_V)
        elif isinstance(target_device, CompiledIonicDevice1D):
            right_state = reconstruct_ionic_state_1d(
                source_bundle,
                target_device,
                target_voltage_V=right_voltage_V,
                require_same_materials=True,
            )
        else:
            right_state = reconstruct_state_1d(
                source_bundle,
                target_device,
                target_voltage_V=right_voltage_V,
                require_same_materials=True,
            )
        if not physical_state_equivalent(left_state, right_state, rtol=1e-13):
            raise WorkingPointTransferError("t^- and t^+ physical densities diverged across transfer")

    left_history = source_bundle.sections["history"]
    if "history" in preserve_set and not math.isclose(
        float(boundary_time_s), float(left_history["time_s"]), rel_tol=0.0, abs_tol=1e-15
    ):
        raise WorkingPointTransferError(
            f"boundary_time_s ({boundary_time_s}) disagrees with t^- history time_s ({left_history['time_s']})"
        )

    prior_chain = list(left_history.get("prior_stages", []))
    prior_chain.append(
        {
            "stage_id": str(left_history["stage_id"]),
            "end_time_s": float(left_history["time_s"]),
            "left_working_point_sha256": source_bundle.sha256,
            "rebound_materials": rebound_materials,
            "declared_state_reset": was_reset,
        }
    )
    qual_sec = source_bundle.sections["qualification"]
    right_bundle = capture_working_point(
        target_device,
        right_state,
        point_id=right_point_id,
        owner_epoch=owner_epoch,
        stage_id=to_stage,
        time_s=boundary_time_s,
        generation_scale=right_generation_scale,
        is_equilibrium=bool(left_history["is_equilibrium"]) and not was_reset
        and float(right_voltage_V) == float(left_history["applied_voltage_V"])
        and float(right_generation_scale) == float(left_history["generation_scale"])
        and not rebound_materials,
        encoding=target_encoding,
        reference_voltage_V=right_voltage_V if target_encoding == "coordinates" else None,
        damage_state=new_damage_doc,
        switch_state=right_switches,
        prior_history={
            "boundary_side": "right",
            "declared_state_reset": was_reset,
            "prior_stages": prior_chain,
        },
        evidence_labels=tuple(qual_sec["evidence_labels"]),
        limitations=tuple(qual_sec["limitations"]),
    )

    if "inventories" in preserve_set:
        for key in (
            "electron_inventory_m2",
            "hole_inventory_m2",
            "positive_ion_inventory_m2",
            "negative_ion_inventory_m2",
        ):
            if not math.isclose(
                float(source_bundle.sections["inventories"][key]),
                float(right_bundle.sections["inventories"][key]),
                rel_tol=1e-13,
                abs_tol=1e-18,
            ):
                raise WorkingPointTransferError(f"preserved inventory {key!r} drifted across boundary")

    if "state" in preserve_set and right_bundle.physical_identity != source_bundle.physical_identity:
        # When rebinding or voltage-rebasing shifts float64 by 1 ULP, verify 1e-13 equivalence;
        # when capture uses identical physical arrays (same device, physical_fields or exact state),
        # physical_identity matches bit-for-bit as well.
        if not physical_state_equivalent(left_state, right_state, rtol=1e-13):
            raise WorkingPointTransferError("physical state identity violated across preserved transfer")

    boundary = StageBoundaryState(
        from_stage=from_stage,
        to_stage=to_stage,
        boundary_time_s=float(boundary_time_s),
        preserved_quantities=preserve,
        left_bundle=source_bundle,
        right_bundle=right_bundle,
    )
    return StageTransferResult(
        boundary=boundary,
        compiled_device=target_device,
        right_state=right_state,
        rebound_materials=rebound_materials,
    )


@dataclass(frozen=True, slots=True)
class StageIntegrationTrace:
    """Step-by-step Backward Euler stage trace starting from an arbitrary ``WorkingPointBundle``."""

    initial_bundle: WorkingPointBundle
    final_bundle: WorkingPointBundle | None
    protocol: TransientProtocol
    accepted_times_s: np.ndarray
    states: tuple[State1D | IonicState1D, ...]
    attempts: tuple[TransientAttempt | IonicTransientAttempt, ...]
    terminal_conduction_current_A_m2: np.ndarray
    terminal_displacement_current_A_m2: np.ndarray
    terminal_total_current_A_m2: np.ndarray
    maxwell_spread_A_m2: np.ndarray
    outcome: Literal["completed", "partial", "failed"]
    failure_reason: str | None
    wall_s: float


def _ionic_newton_settings(
    compiled: CompiledIonicDevice1D, options: IonicTransientOptions
) -> tuple[NewtonSettings, Callable[[np.ndarray], float]]:
    electronic = options.electronic
    count, ion_count = compiled.node_count, compiled.ion_geometry.node_count
    weights = np.r_[
        production_residual_scaling(
            compiled.electronic_component,
            poisson_tolerance=electronic.poisson_tolerance,
            continuity_tolerance_A_m2=electronic.continuity_tolerance_A_m2,
            contact_log_tolerance=electronic.contact_log_tolerance,
        ).row_weights,
        np.full(
            ion_count * compiled.ion_species_count,
            ion_count / options.ion_balance_tolerance_m2_s,
        ),
    ]
    steps = np.r_[
        production_coordinates(
            compiled.electronic_component,
            potential_step_V=electronic.potential_step_V,
            quasi_fermi_step=electronic.quasi_fermi_step,
        ).max_step,
        np.full(ion_count * compiled.ion_species_count, options.ion_logit_step),
    ]
    settings = NewtonSettings(
        ResidualScaling(weights),
        1.0,
        StepControl(steps),
        electronic.newton_budget,
        (),
        False,
    )
    electronic_norm = production_convergence_norm(
        compiled.electronic_component,
        convergence_policy=electronic.convergence_policy,
    )
    assert electronic_norm is not None

    def convergence_norm(scaled: np.ndarray) -> float:
        if scaled.shape != (compiled.state_size,) or not np.all(np.isfinite(scaled)):
            return math.inf
        return max(
            electronic_norm(scaled[: 3 * count]),
            *(
                math.fsum(abs(scaled[block]).tolist()) / ion_count
                for block in compiled.ion_species_slices
            ),
        )

    return settings, convergence_norm


def integrate_stage_from_working_point(
    compiled: CompiledDevice1D | CompiledIonicDevice1D,
    initial_bundle: WorkingPointBundle,
    stage_protocol: TransientProtocol,
    *,
    options: TransientOptions | IonicTransientOptions,
    stage_id: str,
    final_point_id: str,
    owner_epoch: str,
    final_encoding: Literal["coordinates", "physical_fields"] = "coordinates",
) -> StageIntegrationTrace:
    """Integrate a transient stage directly from ``initial_bundle`` without steady-state reset.

    Computes conduction, displacement, and total Maxwell terminal currents from
    step-to-step Backward Euler state differences (never single-snapshot DC
    postprocessing). Unreached steps on partial/failed runs carry IEEE-754
    ``NaN`` so ``encode_array`` masks them losslessly.
    """
    started = time.monotonic()
    step_count = int(stage_protocol.times_s.size - 1)
    cond_arr: NDArray[np.float64] = np.full(step_count, np.nan, dtype=np.float64)
    disp_arr: NDArray[np.float64] = np.full(step_count, np.nan, dtype=np.float64)
    tot_arr: NDArray[np.float64] = np.full(step_count, np.nan, dtype=np.float64)
    spread_arr: NDArray[np.float64] = np.full(step_count, np.nan, dtype=np.float64)

    hist = initial_bundle.sections["history"]
    v0 = float(stage_protocol.voltage_V[0])
    g0 = float(stage_protocol.generation_scale[0])
    if not math.isclose(v0, float(hist["applied_voltage_V"]), rel_tol=0.0, abs_tol=1e-15):
        raise WorkingPointTransferError(
            "stage_protocol.voltage_V[0] must match initial_bundle history applied_voltage_V; "
            "use transfer_working_point for stage boundary voltage jumps"
        )
    if not math.isclose(g0, float(hist["generation_scale"]), rel_tol=0.0, abs_tol=1e-15):
        raise WorkingPointTransferError(
            "stage_protocol.generation_scale[0] must match initial_bundle history generation_scale; "
            "use transfer_working_point for stage boundary illumination switches"
        )

    if isinstance(compiled, CompiledIonicDevice1D):
        if not isinstance(options, IonicTransientOptions):
            raise TypeError("CompiledIonicDevice1D requires IonicTransientOptions")
        elec_opts = options.electronic
        if step_count > elec_opts.max_steps:
            raise ValueError("requested timepoints exceed max_steps")
        state0_ion = reconstruct_ionic_state_1d(initial_bundle, compiled, target_voltage_V=v0)
        settings, conv_norm = _ionic_newton_settings(compiled, options)
        deadline = started + elec_opts.wall_s
        assembled = use_assembled()
        states_ion: list[State1D | IonicState1D] = [state0_ion]
        accepted_times = [float(stage_protocol.times_s[0])]
        attempts_ion: list[TransientAttempt | IonicTransientAttempt] = []
        failure_reason: str | None = None

        for idx in range(1, stage_protocol.times_s.size):
            t_start = float(stage_protocol.times_s[idx - 1])
            t_end = float(stage_protocol.times_s[idx])
            v_end = float(stage_protocol.voltage_V[idx])
            g_end = float(stage_protocol.generation_scale[idx])
            attempt_t0 = time.monotonic()
            prev_ion = cast(IonicState1D, states_ion[-1])
            candidate = np.array(prev_ion.u, copy=True)
            seed = candidate.copy()
            counts = {"residual": 0, "jacobian": 0, "accounting": 0}
            newton_res = None
            accounting: dict[str, object] | None = None
            err_msg: str | None = None
            next_ion: IonicState1D | None = None

            def eval_ion(
                kind: str,
                fn: Callable[[np.ndarray], Matrix],
                y: np.ndarray,
                _counts: dict[str, int] = counts,
            ) -> Matrix:
                nonlocal candidate
                candidate = np.array(y, copy=True)
                if time.monotonic() >= deadline:
                    raise RuntimeError("wall_budget_exhausted")
                _counts[kind] += 1
                out = fn(y)
                if time.monotonic() >= deadline:
                    raise RuntimeError("wall_budget_exhausted")
                return out

            try:
                step_ion = (
                    cast(
                        IonicBackwardEuler1DStep,
                        AssembledIonicBackwardEuler1DStep(
                            AssembledIonic1DProblem(compiled, v_end, g_end),
                            prev_ion,
                            t_end - t_start,
                        ),
                    )
                    if assembled
                    else IonicBackwardEuler1DStep(
                        Ionic1DProblem(compiled, v_end, g_end),
                        prev_ion,
                        t_end - t_start,
                    )
                )
                candidate = step_ion.initial_coordinates(elec_opts.predictor, voltage_lift=True)
                seed = candidate.copy()
                def _ion_res(y: np.ndarray) -> np.ndarray:
                    return np.asarray(eval_ion("residual", step_ion.residual, y), dtype=np.float64)

                def _ion_jac(y: np.ndarray) -> Matrix:
                    return eval_ion("jacobian", step_ion.jacobian, y)

                newton_res = solve_newton(
                    CallableProblem(_ion_res, _ion_jac),
                    candidate,
                    settings,
                    convergence_norm=conv_norm,
                )
                candidate = newton_res.state
                if np.all(np.isfinite(candidate)):
                    next_ion = compiled.state(candidate, voltage_V=v_end)
                    counts["accounting"] += 1
                    accounting = ionic_step_accounting(step_ion, candidate)
                accepted = bool(newton_res.accepted)
                termination = newton_res.termination.value
                if accepted and accounting is None:
                    accepted, termination = False, "nonfinite_accounting"
            except Exception as exc:
                accepted = False
                termination = (
                    "wall_budget_exhausted"
                    if str(exc) == "wall_budget_exhausted"
                    else "evaluation_error"
                )
                err_msg = f"{type(exc).__name__}: {str(exc)[:300]}"

            attempts_ion.append(
                IonicTransientAttempt(
                    t_start,
                    t_end,
                    v_end,
                    g_end,
                    accepted,
                    termination,
                    seed,
                    candidate,
                    newton_res,
                    accounting,
                    counts,
                    time.monotonic() - attempt_t0,
                    err_msg,
                )
            )
            if not accepted or next_ion is None or accounting is None:
                failure_reason = err_msg or f"step_{idx}_failed:{termination}"
                break

            cond_faces = np.asarray(accounting["conduction_current_density_A_m2"], dtype=float)
            disp_faces = np.asarray(accounting["displacement_current_density_A_m2"], dtype=float)
            tot_term = np.asarray(accounting["terminal_total_current_density_A_m2"], dtype=float)
            cond_arr[idx - 1] = float(cond_faces[0])
            disp_arr[idx - 1] = float(disp_faces[0])
            tot_arr[idx - 1] = float(tot_term[0])
            spread_arr[idx - 1] = float(cast(float, accounting["maxwell_spread_A_m2"]))
            states_ion.append(next_ion)
            accepted_times.append(t_end)

        states_out = tuple(states_ion)
        attempts_out = tuple(attempts_ion)
    else:
        if not isinstance(options, TransientOptions):
            raise TypeError("CompiledDevice1D requires TransientOptions")
        if step_count > options.max_steps:
            raise ValueError("requested timepoints exceed max_steps")
        state0_elec = reconstruct_state_1d(initial_bundle, compiled, target_voltage_V=v0)
        settings = NewtonSettings(
            production_residual_scaling(
                compiled,
                poisson_tolerance=options.poisson_tolerance,
                continuity_tolerance_A_m2=options.continuity_tolerance_A_m2,
                contact_log_tolerance=options.contact_log_tolerance,
            ),
            1.0,
            production_coordinates(
                compiled,
                potential_step_V=options.potential_step_V,
                quasi_fermi_step=options.quasi_fermi_step,
            ),
            options.newton_budget,
            (),
            False,
        )
        conv_norm_e = production_convergence_norm(
            compiled, convergence_policy=options.convergence_policy
        )
        deadline = started + options.wall_s
        binding = classical_dc_1d(compiled) if use_assembled() else None
        states_e: list[State1D | IonicState1D] = [state0_elec]
        accepted_times = [float(stage_protocol.times_s[0])]
        attempts_e: list[TransientAttempt | IonicTransientAttempt] = []
        failure_reason = None

        for idx in range(1, stage_protocol.times_s.size):
            t_start = float(stage_protocol.times_s[idx - 1])
            t_end = float(stage_protocol.times_s[idx])
            v_end = float(stage_protocol.voltage_V[idx])
            g_end = float(stage_protocol.generation_scale[idx])
            attempt_t0 = time.monotonic()
            prev_e = cast(State1D, states_e[-1])
            endpoint = (
                DC1DProblem(compiled, v_end, g_end)
                if binding is None
                else cast(
                    DC1DProblem,
                    AssembledStorage1D(AssembledDC1DProblem(compiled, v_end, g_end, binding)),
                )
            )
            step_e = BackwardEuler1DStep(endpoint, prev_e, t_end - t_start)
            candidate = step_e.initial_coordinates(voltage_lift=True)
            seed = candidate.copy()
            newton_res = None
            accounting = None
            err_msg = None

            def eval_e(fn: Callable[[np.ndarray], Matrix], u: np.ndarray) -> Matrix:
                nonlocal candidate
                candidate = np.array(u, copy=True)
                if time.monotonic() >= deadline:
                    raise RuntimeError("wall_budget_exhausted")
                out = fn(u)
                if time.monotonic() >= deadline:
                    raise RuntimeError("wall_budget_exhausted")
                return out

            try:
                candidate = step_e.initial_coordinates(options.predictor, voltage_lift=True)
                seed = candidate.copy()
                def _elec_res(u: np.ndarray) -> np.ndarray:
                    return np.asarray(eval_e(step_e.residual, u), dtype=np.float64)

                def _elec_jac(u: np.ndarray) -> Matrix:
                    return eval_e(step_e.jacobian, u)

                newton_res = solve_newton(
                    CallableProblem(_elec_res, _elec_jac),
                    candidate,
                    settings,
                    convergence_norm=conv_norm_e,
                )
                candidate = newton_res.state
                if np.all(np.isfinite(candidate)):
                    _, n_chk, p_chk = compiled.fields(candidate, voltage_V=v_end)
                    if (
                        np.all(np.isfinite(n_chk))
                        and np.all(np.isfinite(p_chk))
                        and np.all(n_chk > 0.0)
                        and np.all(p_chk > 0.0)
                    ):
                        accounting = electronic_step_accounting(step_e, candidate)
                accepted = bool(newton_res.accepted)
                termination = newton_res.termination.value
                if accepted and accounting is None:
                    accepted, termination = False, "nonfinite_accounting"
            except Exception as exc:
                accepted = False
                termination = (
                    "wall_budget_exhausted"
                    if str(exc) == "wall_budget_exhausted"
                    else "evaluation_error"
                )
                err_msg = f"{type(exc).__name__}: {str(exc)[:300]}"

            attempts_e.append(
                TransientAttempt(
                    t_start,
                    t_end,
                    v_end,
                    g_end,
                    accepted,
                    termination,
                    seed,
                    candidate,
                    newton_res,
                    accounting,
                    time.monotonic() - attempt_t0,
                    err_msg,
                )
            )
            if not accepted or accounting is None:
                failure_reason = err_msg or f"step_{idx}_failed:{termination}"
                break

            cond_term = np.asarray(
                accounting["terminal_conduction_current_density_A_m2"], dtype=float
            )
            disp_term = np.asarray(
                accounting["terminal_displacement_current_density_A_m2"], dtype=float
            )
            tot_term = np.asarray(accounting["terminal_total_current_density_A_m2"], dtype=float)
            cond_arr[idx - 1] = float(cond_term[0])
            disp_arr[idx - 1] = float(disp_term[0])
            tot_arr[idx - 1] = float(tot_term[0])
            spread_arr[idx - 1] = float(cast(float, accounting["maxwell_spread_A_m2"]))
            states_e.append(compiled.state(candidate, voltage_V=v_end))
            accepted_times.append(t_end)

        states_out = tuple(states_e)
        attempts_out = tuple(attempts_e)

    n_accepted_steps = len(states_out) - 1
    if n_accepted_steps == step_count and failure_reason is None:
        outcome: Literal["completed", "partial", "failed"] = "completed"
    elif n_accepted_steps > 0:
        outcome = "partial"
    else:
        outcome = "failed"

    qual = initial_bundle.sections["qualification"]
    final_bundle: WorkingPointBundle | None = None
    if n_accepted_steps > 0:
        last_idx = n_accepted_steps
        prior_stages = list(hist.get("prior_stages", []))
        prior_stages.append(
            {
                "stage_id": str(hist["stage_id"]),
                "start_time_s": float(stage_protocol.times_s[0]),
                "end_time_s": float(accepted_times[-1]),
                "initial_working_point_sha256": initial_bundle.sha256,
                "accepted_steps": n_accepted_steps,
            }
        )
        final_bundle = capture_working_point(
            compiled,
            states_out[-1],
            point_id=final_point_id,
            owner_epoch=owner_epoch,
            stage_id=stage_id,
            time_s=float(accepted_times[-1]),
            generation_scale=float(stage_protocol.generation_scale[last_idx]),
            is_equilibrium=False,
            encoding=final_encoding,
            damage_state=initial_bundle.sections["damage"],
            switch_state=hist.get("switches"),
            prior_history={"boundary_side": "left", "prior_stages": prior_stages},
            evidence_labels=tuple(qual["evidence_labels"]),
            limitations=tuple(qual["limitations"]),
        )

    def _readonly_float64(values: np.ndarray) -> np.ndarray:
        arr = np.array(values, dtype=np.float64, copy=True)
        arr.setflags(write=False)
        return arr

    return StageIntegrationTrace(
        initial_bundle=initial_bundle,
        final_bundle=final_bundle,
        protocol=stage_protocol,
        accepted_times_s=frozen_array(np.asarray(accepted_times, dtype=np.float64)),
        states=states_out,
        attempts=attempts_out,
        terminal_conduction_current_A_m2=_readonly_float64(cond_arr),
        terminal_displacement_current_A_m2=_readonly_float64(disp_arr),
        terminal_total_current_A_m2=_readonly_float64(tot_arr),
        maxwell_spread_A_m2=_readonly_float64(spread_arr),
        outcome=outcome,
        failure_reason=failure_reason,
        wall_s=time.monotonic() - started,
    )


def build_stage_experiment_result(
    trace: StageIntegrationTrace,
    *,
    protocol: Protocol,
    run_id: str,
    attempt_id: str,
    experiment_id: str,
    bindings_override: Mapping[str, Mapping[str, str]] | None = None,
    evidence: Sequence[Mapping[str, Any]] | None = None,
) -> ExperimentResult:
    """Wrap a ``StageIntegrationTrace`` into a validated ``ExperimentResult`` envelope."""
    wp = trace.initial_bundle.working_point
    mat_sec = trace.initial_bundle.sections["materials"]
    qual_sec = trace.initial_bundle.sections["qualification"]
    proto_sha = protocol.sha256

    numerics_doc = {
        "step_count": int(trace.protocol.times_s.size - 1),
        "accepted_steps": int(len(trace.states) - 1),
    }
    numerics_sha = _section_sha256(numerics_doc)
    default_bindings: dict[str, dict[str, str]] = {
        "source": _ref(str(mat_sec["device_id"]), str(mat_sec["prepared_sha256"])),
        "schema": _ref("solarlab.experiment-result.v1", hashlib.sha256(b"solarlab.experiment-result.v1").hexdigest()),
        "registry": _ref(experiment_id, hashlib.sha256(experiment_id.encode()).hexdigest()),
        "physics": _ref(str(mat_sec["family"]), str(mat_sec["compiled_identity"])),
        "protocol": _ref(str(protocol.export()["id"]), proto_sha),
        "initial": _ref(trace.initial_bundle.id, wp.sha256),
        "numerics": _ref("backward_euler_1d", numerics_sha),
        "resources": _ref("inline_stage_resources", _section_sha256(trace.initial_bundle.sections["grid"])),
    }
    if bindings_override is not None:
        for key, val in bindings_override.items():
            default_bindings[str(key)] = {"id": str(val["id"]), "sha256": str(val["sha256"])}
    # Always pin 'initial' to the exact WorkingPoint binding of trace.initial_bundle.
    default_bindings["initial"] = _ref(trace.initial_bundle.id, wp.sha256)

    evidence_list = (
        [dict(item) for item in evidence]
        if evidence is not None
        else [
            {
                "label": str(qual_sec["evidence_labels"][0]),
                "scope": [experiment_id],
                "sources": (
                    []
                    if str(qual_sec["evidence_labels"][0]) in {"PARKED", "FIRST_SLICE"}
                    else [_ref(trace.initial_bundle.id, wp.sha256)]
                ),
                "limitations": list(qual_sec["limitations"]),
            }
        ]
    )

    payload: dict[str, Any] = {
        "times_s": encode_array(np.asarray(trace.protocol.times_s, dtype=np.float64), unit="s"),
        "accepted_times_s": encode_array(np.asarray(trace.accepted_times_s, dtype=np.float64), unit="s"),
        "terminal_conduction_current_A_m2": encode_array(
            np.asarray(trace.terminal_conduction_current_A_m2, dtype=np.float64), unit="A/m^2"
        ),
        "terminal_displacement_current_A_m2": encode_array(
            np.asarray(trace.terminal_displacement_current_A_m2, dtype=np.float64), unit="A/m^2"
        ),
        "terminal_total_current_A_m2": encode_array(
            np.asarray(trace.terminal_total_current_A_m2, dtype=np.float64), unit="A/m^2"
        ),
        "maxwell_spread_A_m2": encode_array(
            np.asarray(trace.maxwell_spread_A_m2, dtype=np.float64), unit="A/m^2"
        ),
        "final_working_point": (
            _ref(trace.final_bundle.id, trace.final_bundle.sha256)
            if trace.final_bundle is not None
            else None
        ),
    }

    if trace.outcome == "completed":
        peak_disp = float(np.max(np.abs(trace.terminal_displacement_current_A_m2)))
        final_total = float(trace.terminal_total_current_A_m2[-1])
        max_spread = float(np.max(trace.maxwell_spread_A_m2))
        metrics: dict[str, dict[str, Any]] = {
            "peak_abs_displacement_current_A_m2": {
                "status": "identified",
                "value": peak_disp,
                "reason": None,
            },
            "final_terminal_total_current_A_m2": {
                "status": "identified",
                "value": final_total,
                "reason": None,
            },
            "max_maxwell_spread_A_m2": {
                "status": "identified",
                "value": max_spread,
                "reason": None,
            },
        }
    else:
        unid_reason = trace.failure_reason or "incomplete_stage_trajectory"
        metrics = {
            "peak_abs_displacement_current_A_m2": {
                "status": "unidentified",
                "value": None,
                "reason": unid_reason,
            },
            "final_terminal_total_current_A_m2": {
                "status": "unidentified",
                "value": None,
                "reason": unid_reason,
            },
            "max_maxwell_spread_A_m2": {
                "status": "unidentified",
                "value": None,
                "reason": unid_reason,
            },
        }

    manifest_doc = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "experiment_id": experiment_id,
        "bindings": default_bindings,
    }
    manifest_sha = _section_sha256(manifest_doc)
    doc: dict[str, Any] = {
        "schema": "solarlab.experiment-result.v1",
        "header": {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "experiment_id": experiment_id,
            "engine": "solarlab",
            "manifest_sha256": manifest_sha,
            "outcome": trace.outcome,
            "reason": None if trace.outcome == "completed" else (trace.failure_reason or "stage_incomplete"),
            "bindings": default_bindings,
            "evidence": evidence_list,
        },
        "payload_schema": "solarlab.stage-transient-payload.v1",
        "payload": payload,
        "metrics": metrics,
    }
    return ExperimentResult.from_document(doc)


@dataclass(frozen=True, slots=True)
class PairedStageRunResult:
    """Matched pulsed + unpulsed control perturbation stage results from one ``WorkingPointBundle``."""

    initial_bundle: WorkingPointBundle
    pulsed_trace: StageIntegrationTrace
    control_trace: StageIntegrationTrace
    pulsed_result: ExperimentResult
    control_result: ExperimentResult
    differential_conduction_current_A_m2: np.ndarray
    differential_displacement_current_A_m2: np.ndarray
    differential_total_current_A_m2: np.ndarray


def run_paired_perturbation_stages(
    compiled: CompiledDevice1D | CompiledIonicDevice1D,
    initial_bundle: WorkingPointBundle,
    pulsed_protocol: TransientProtocol,
    *,
    options: TransientOptions | IonicTransientOptions,
    protocol: Protocol,
    run_id: str,
    experiment_id: str,
    owner_epoch: str,
    control_protocol: TransientProtocol | None = None,
    stage_id: str = "perturbation",
) -> PairedStageRunResult:
    """Run pulsed and matched unpulsed-control perturbation stages from identical ``initial_bundle``.

    When ``control_protocol`` is omitted, constructs the matched constant-drive
    unpulsed baseline on the exact ``pulsed_protocol.times_s`` grid holding
    ``(voltage_V[0], generation_scale[0])`` fixed so background non-equilibrium
    drift is subtracted cleanly step-by-step.
    """
    if control_protocol is None:
        v_hold = np.full_like(pulsed_protocol.times_s, float(pulsed_protocol.voltage_V[0]))
        g_hold = np.full_like(pulsed_protocol.times_s, float(pulsed_protocol.generation_scale[0]))
        control_protocol = TransientProtocol(
            times_s=np.array(pulsed_protocol.times_s, copy=True),
            voltage_V=v_hold,
            generation_scale=g_hold,
        )
    else:
        if not np.array_equal(pulsed_protocol.times_s, control_protocol.times_s):
            raise WorkingPointTransferError(
                "pulsed_protocol and control_protocol must share identical times_s sample grids"
            )
        if float(pulsed_protocol.voltage_V[0]) != float(control_protocol.voltage_V[0]) or float(
            pulsed_protocol.generation_scale[0]
        ) != float(control_protocol.generation_scale[0]):
            raise WorkingPointTransferError(
                "pulsed_protocol and control_protocol must start at identical initial drive"
            )

    pulsed_trace = integrate_stage_from_working_point(
        compiled,
        initial_bundle,
        pulsed_protocol,
        options=options,
        stage_id=f"{stage_id}_pulsed",
        final_point_id=f"{initial_bundle.id}:{stage_id}_pulsed_end",
        owner_epoch=owner_epoch,
    )
    control_trace = integrate_stage_from_working_point(
        compiled,
        initial_bundle,
        control_protocol,
        options=options,
        stage_id=f"{stage_id}_control",
        final_point_id=f"{initial_bundle.id}:{stage_id}_control_end",
        owner_epoch=owner_epoch,
    )

    pulsed_res = build_stage_experiment_result(
        pulsed_trace,
        protocol=protocol,
        run_id=run_id,
        attempt_id=f"{run_id}:pulsed",
        experiment_id=experiment_id,
    )
    control_res = build_stage_experiment_result(
        control_trace,
        protocol=protocol,
        run_id=run_id,
        attempt_id=f"{run_id}:control",
        experiment_id=f"{experiment_id}.unpulsed_control",
    )

    # Assert identical initial WorkingPoint binding across pulsed and control envelopes.
    if pulsed_res.export()["header"]["bindings"]["initial"] != control_res.export()["header"]["bindings"]["initial"]:
        raise WorkingPointTransferError("pulsed and unpulsed control initial WorkingPoint bindings diverged")

    def _ro(arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=np.float64, copy=True)
        out.setflags(write=False)
        return out

    diff_cond = _ro(
        pulsed_trace.terminal_conduction_current_A_m2 - control_trace.terminal_conduction_current_A_m2
    )
    diff_disp = _ro(
        pulsed_trace.terminal_displacement_current_A_m2
        - control_trace.terminal_displacement_current_A_m2
    )
    diff_tot = _ro(
        pulsed_trace.terminal_total_current_A_m2 - control_trace.terminal_total_current_A_m2
    )
    return PairedStageRunResult(
        initial_bundle=initial_bundle,
        pulsed_trace=pulsed_trace,
        control_trace=control_trace,
        pulsed_result=pulsed_res,
        control_result=control_res,
        differential_conduction_current_A_m2=diff_cond,
        differential_displacement_current_A_m2=diff_disp,
        differential_total_current_A_m2=diff_tot,
    )

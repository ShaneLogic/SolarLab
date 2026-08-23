"""Canonical grid-refinement certification for ion-aware impedance."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Literal, Mapping, Self

import numpy as np

from perovskite_sim.experiments.ion_aware_dc import (
    IonAwareDCResult,
    ion_aware_dc_state_sha256,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    DEFAULT_REFINEMENT_FACTORS,
    IonAwareImpedanceProtocol,
    IonAwareImpedanceResult,
    build_ion_aware_impedance_protocol,
    run_ion_aware_impedance,
)
ION_AWARE_IMPEDANCE_GRID_PROTOCOL_SCHEMA = (
    "ion-aware-impedance-grid-protocol-v1"
)
ION_AWARE_IMPEDANCE_GRID_HASH_SCHEMA = "ion-aware-impedance-grid-f64-v1"
ProgressCallback = Callable[[str, int, int, str], None]


class IonAwareImpedanceGridCapabilityError(RuntimeError):
    """The requested states cannot form one comparable grid ladder."""


class IonAwareImpedanceGridCertificationError(RuntimeError):
    """A completed grid ladder did not pass a requested evidence axis."""

    def __init__(
        self,
        message: str,
        result: IonAwareImpedanceGridResult,
    ) -> None:
        self.result = result
        super().__init__(message)


def _positive(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def ion_aware_impedance_grid_sha256(x: np.ndarray) -> str:
    """Return a platform-independent hash for one exact electrical grid."""
    grid = np.asarray(x, dtype=np.float64)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError(
            "ion-aware impedance grid hash requires a finite, strictly "
            "increasing 1-D grid with at least three nodes"
        )
    canonical = np.array(grid, dtype="<f8", order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(ION_AWARE_IMPEDANCE_GRID_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(int(canonical.size).to_bytes(8, byteorder="big", signed=False))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _common_impedance_protocol_json(
    protocol: IonAwareImpedanceProtocol,
) -> str:
    payload = protocol.to_dict()
    del payload["dc_state_sha256"]
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class IonAwareImpedanceGridProtocol:
    """Exact DC states, meshes, frequencies, and grid acceptance limits."""

    grid_node_counts: tuple[int, ...]
    grid_sha256s: tuple[str, ...]
    impedance_protocols: tuple[IonAwareImpedanceProtocol, ...]
    max_grid_impedance_magnitude_relative_change: float = 2.0e-2
    max_grid_impedance_phase_change_deg: float = 1.0
    schema_version: Literal["ion-aware-impedance-grid-protocol-v1"] = (
        ION_AWARE_IMPEDANCE_GRID_PROTOCOL_SCHEMA
    )

    def __post_init__(self) -> None:
        try:
            counts = tuple(self.grid_node_counts)
        except TypeError as exc:
            raise TypeError("grid_node_counts must be an iterable") from exc
        if len(counts) < 3:
            raise ValueError("grid ladder requires at least three grids")
        normalized_counts: list[int] = []
        for index, value in enumerate(counts):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or int(value) < 3
            ):
                raise ValueError(
                    f"grid_node_counts[{index}] must be an integer >= 3"
                )
            normalized_counts.append(int(value))
        if any(
            right <= left
            for left, right in zip(
                normalized_counts[:-1],
                normalized_counts[1:],
                strict=True,
            )
        ):
            raise ValueError("grid_node_counts must be strictly increasing")
        object.__setattr__(self, "grid_node_counts", tuple(normalized_counts))

        try:
            hashes = tuple(
                _sha256(value, f"grid_sha256s[{index}]")
                for index, value in enumerate(self.grid_sha256s)
            )
        except TypeError as exc:
            raise TypeError("grid_sha256s must be an iterable") from exc
        try:
            protocols = tuple(self.impedance_protocols)
        except TypeError as exc:
            raise TypeError("impedance_protocols must be an iterable") from exc
        if len(hashes) != len(normalized_counts) or len(protocols) != len(
            normalized_counts
        ):
            raise ValueError(
                "grid counts, hashes, and impedance protocols must have "
                "identical cardinality"
            )
        if len(set(hashes)) != len(hashes):
            raise ValueError("grid_sha256s must be unique")
        if any(
            not isinstance(item, IonAwareImpedanceProtocol)
            for item in protocols
        ):
            raise TypeError(
                "impedance_protocols must contain IonAwareImpedanceProtocol"
            )
        common = _common_impedance_protocol_json(protocols[0])
        if any(
            _common_impedance_protocol_json(item) != common
            for item in protocols[1:]
        ):
            raise ValueError(
                "every grid must share one DC history, frequency request, "
                "stencil ladder, and acceptance contract"
            )
        if len({item.protocol_hash for item in protocols}) != len(protocols):
            raise ValueError(
                "each grid must bind a distinct impedance protocol/state"
            )
        object.__setattr__(self, "grid_sha256s", hashes)
        object.__setattr__(self, "impedance_protocols", protocols)
        for field in (
            "max_grid_impedance_magnitude_relative_change",
            "max_grid_impedance_phase_change_deg",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        if self.schema_version != ION_AWARE_IMPEDANCE_GRID_PROTOCOL_SCHEMA:
            raise ValueError("unsupported ion-aware impedance grid protocol schema")

    @property
    def frequencies_Hz(self) -> tuple[float, ...]:
        return self.impedance_protocols[0].frequencies_Hz

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_node_counts": list(self.grid_node_counts),
            "grid_sha256s": list(self.grid_sha256s),
            "impedance_protocols": [
                item.to_dict() for item in self.impedance_protocols
            ],
            "max_grid_impedance_magnitude_relative_change": (
                self.max_grid_impedance_magnitude_relative_change
            ),
            "max_grid_impedance_phase_change_deg": (
                self.max_grid_impedance_phase_change_deg
            ),
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def protocol_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("ion-aware impedance grid protocol must be a mapping")
        expected = {field.name for field in dataclasses.fields(cls)}
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "ion-aware impedance grid protocol keys do not match schema; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        values = dict(payload)
        values["grid_node_counts"] = tuple(values["grid_node_counts"])
        values["grid_sha256s"] = tuple(values["grid_sha256s"])
        values["impedance_protocols"] = tuple(
            IonAwareImpedanceProtocol.from_dict(item)
            for item in values["impedance_protocols"]
        )
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError(
                "ion-aware impedance grid protocol JSON must contain an object"
            )
        return cls.from_dict(parsed)


def build_ion_aware_impedance_grid_protocol(
    dc_states: tuple[IonAwareDCResult, ...] | list[IonAwareDCResult],
    frequencies_Hz: np.ndarray | tuple[float, ...],
    *,
    delta_V: float = 0.01,
    state_step: float = 1.0e-5,
    voltage_step: float = 1.0e-5,
    refinement_factors: tuple[float, ...] = DEFAULT_REFINEMENT_FACTORS,
    max_relative_face_spread: float = 5.0e-4,
    max_backward_error: float = 1.0e-10,
    max_impedance_magnitude_relative_change: float = 1.0e-2,
    max_impedance_phase_change_deg: float = 0.5,
    max_mass_matrix_relative_error: float = 1.0e-8,
    max_ion_inventory_response_relative: float = 1.0e-8,
    max_current_decomposition_relative_error: float = 1.0e-7,
    frequency_branch_margin_decades: float = 1.0,
    max_frequency_sampling_gap_decades: float = 0.5,
    max_grid_impedance_magnitude_relative_change: float = 2.0e-2,
    max_grid_impedance_phase_change_deg: float = 1.0,
) -> IonAwareImpedanceGridProtocol:
    """Bind a precomputed, ordered DC grid ladder before AC execution."""
    try:
        states = tuple(dc_states)
    except TypeError as exc:
        raise TypeError("dc_states must be an iterable") from exc
    if len(states) < 3:
        raise ValueError("grid ladder requires at least three DC states")
    if any(not isinstance(item, IonAwareDCResult) for item in states):
        raise TypeError("dc_states must contain IonAwareDCResult values")
    if any(item.stack != states[0].stack for item in states[1:]):
        raise IonAwareImpedanceGridCapabilityError(
            "every grid DC state must use the same device stack"
        )
    if any(item.protocol_hash != states[0].protocol_hash for item in states[1:]):
        raise IonAwareImpedanceGridCapabilityError(
            "every grid DC state must use the same canonical DC history"
        )
    counts = tuple(len(np.asarray(item.x)) for item in states)
    hashes = tuple(
        ion_aware_impedance_grid_sha256(item.x) for item in states
    )
    protocols = tuple(
        build_ion_aware_impedance_protocol(
            state,
            frequencies_Hz,
            delta_V=delta_V,
            state_step=state_step,
            voltage_step=voltage_step,
            refinement_factors=refinement_factors,
            max_relative_face_spread=max_relative_face_spread,
            max_backward_error=max_backward_error,
            max_impedance_magnitude_relative_change=(
                max_impedance_magnitude_relative_change
            ),
            max_impedance_phase_change_deg=max_impedance_phase_change_deg,
            max_mass_matrix_relative_error=max_mass_matrix_relative_error,
            max_ion_inventory_response_relative=(
                max_ion_inventory_response_relative
            ),
            max_current_decomposition_relative_error=(
                max_current_decomposition_relative_error
            ),
            frequency_branch_margin_decades=frequency_branch_margin_decades,
            max_frequency_sampling_gap_decades=(
                max_frequency_sampling_gap_decades
            ),
        )
        for state in states
    )
    return IonAwareImpedanceGridProtocol(
        grid_node_counts=counts,
        grid_sha256s=hashes,
        impedance_protocols=protocols,
        max_grid_impedance_magnitude_relative_change=(
            max_grid_impedance_magnitude_relative_change
        ),
        max_grid_impedance_phase_change_deg=(
            max_grid_impedance_phase_change_deg
        ),
    )


@dataclass(frozen=True, slots=True)
class GridPairFrequencyAssessment:
    frequency_Hz: float
    coarse_grid_node_count: int
    fine_grid_node_count: int
    impedance_magnitude_relative_change: float
    impedance_phase_change_deg: float
    passed: bool


@dataclass(frozen=True, slots=True)
class GridFrequencyCertificate:
    frequency_Hz: float
    numerically_certified: bool
    grid_refinement_assessments: tuple[GridPairFrequencyAssessment, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IonAwareImpedanceGridCertificate:
    numerically_certified: bool
    thermodynamically_certified: bool
    frequency_window_certified: bool
    certified: bool
    finest_pair_max_impedance_magnitude_relative_change: float
    finest_pair_max_impedance_phase_change_deg: float
    frequency_point_certificates: tuple[GridFrequencyCertificate, ...]
    numerical_reasons: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IonAwareImpedanceGridResult:
    grid_results: tuple[IonAwareImpedanceResult, ...]
    protocol: IonAwareImpedanceGridProtocol
    certificate: IonAwareImpedanceGridCertificate

    @property
    def protocol_hash(self) -> str:
        return self.protocol.protocol_hash


def _grid_pair_frequency_assessments(
    coarse: IonAwareImpedanceResult,
    fine: IonAwareImpedanceResult,
    coarse_nodes: int,
    fine_nodes: int,
    protocol: IonAwareImpedanceGridProtocol,
) -> tuple[GridPairFrequencyAssessment, ...]:
    if not np.array_equal(coarse.frequencies, fine.frequencies):
        raise IonAwareImpedanceGridCapabilityError(
            "grid ladder changed the requested frequencies"
        )
    coarse_impedance = np.asarray(coarse.Z, dtype=complex)
    fine_impedance = np.asarray(fine.Z, dtype=complex)
    if (
        coarse_impedance.shape != fine_impedance.shape
        or coarse_impedance.shape != coarse.frequencies.shape
        or not np.all(np.isfinite(coarse_impedance))
        or not np.all(np.isfinite(fine_impedance))
        or np.any(np.abs(coarse_impedance) == 0.0)
        or np.any(np.abs(fine_impedance) == 0.0)
    ):
        raise IonAwareImpedanceGridCapabilityError(
            "grid impedance arrays must be finite, nonzero, and frequency aligned"
        )
    magnitude_change = (
        np.abs(np.abs(fine_impedance) - np.abs(coarse_impedance))
        / np.maximum(np.abs(fine_impedance), np.finfo(float).tiny)
    )
    phase_change = np.abs(
        np.angle(fine_impedance / coarse_impedance, deg=True)
    )
    return tuple(
        GridPairFrequencyAssessment(
            frequency_Hz=float(frequency),
            coarse_grid_node_count=coarse_nodes,
            fine_grid_node_count=fine_nodes,
            impedance_magnitude_relative_change=float(magnitude),
            impedance_phase_change_deg=float(phase),
            passed=bool(
                magnitude
                <= protocol.max_grid_impedance_magnitude_relative_change
                and phase <= protocol.max_grid_impedance_phase_change_deg
            ),
        )
        for frequency, magnitude, phase in zip(
            fine.frequencies,
            magnitude_change,
            phase_change,
            strict=True,
        )
    )


def _grid_certificate(
    protocol: IonAwareImpedanceGridProtocol,
    results: tuple[IonAwareImpedanceResult, ...],
) -> IonAwareImpedanceGridCertificate:
    pair_assessments = tuple(
        _grid_pair_frequency_assessments(
            coarse,
            fine,
            coarse_nodes,
            fine_nodes,
            protocol,
        )
        for coarse, fine, coarse_nodes, fine_nodes in zip(
            results[:-1],
            results[1:],
            protocol.grid_node_counts[:-1],
            protocol.grid_node_counts[1:],
            strict=True,
        )
    )
    point_certificates: list[GridFrequencyCertificate] = []
    for index, frequency in enumerate(results[-1].frequencies):
        refinements = tuple(item[index] for item in pair_assessments)
        reasons: list[str] = []
        if not all(
            result.certificate.frequency_point_certificates[index]
            .numerically_certified
            for result in results
        ):
            reasons.append("grid_member_frequency_certificate_failed")
        if not refinements[-1].passed:
            reasons.append("grid_refinement_not_converged")
        point_certificates.append(
            GridFrequencyCertificate(
                frequency_Hz=float(frequency),
                numerically_certified=not reasons,
                grid_refinement_assessments=refinements,
                reasons=tuple(reasons),
            )
        )
    finest = pair_assessments[-1]
    numerical_reasons: list[str] = []
    if not all(result.certificate.numerically_certified for result in results):
        numerical_reasons.append("grid_member_numerical_certificate_failed")
    if not all(item.passed for item in finest):
        numerical_reasons.append("grid_refinement_not_converged")
    numerical = not numerical_reasons
    thermodynamic = all(
        result.certificate.thermodynamically_certified for result in results
    )
    frequency_window = all(
        result.certificate.frequency_window_certified for result in results
    )
    reasons = list(numerical_reasons)
    if not thermodynamic:
        reasons.append("contact_thermodynamics_not_certified")
    if not frequency_window:
        reasons.append("frequency_window_not_certified")
    return IonAwareImpedanceGridCertificate(
        numerically_certified=numerical,
        thermodynamically_certified=thermodynamic,
        frequency_window_certified=frequency_window,
        certified=numerical and thermodynamic and frequency_window,
        finest_pair_max_impedance_magnitude_relative_change=max(
            item.impedance_magnitude_relative_change for item in finest
        ),
        finest_pair_max_impedance_phase_change_deg=max(
            item.impedance_phase_change_deg for item in finest
        ),
        frequency_point_certificates=tuple(point_certificates),
        numerical_reasons=tuple(numerical_reasons),
        reasons=tuple(reasons),
    )


def run_ion_aware_impedance_grid_ladder(
    protocol: IonAwareImpedanceGridProtocol,
    *,
    dc_states: tuple[IonAwareDCResult, ...] | list[IonAwareDCResult],
    require_numerical_certificate: bool = True,
    require_contact_certificate: bool = False,
    require_frequency_window_certificate: bool = False,
    progress: ProgressCallback | None = None,
) -> IonAwareImpedanceGridResult:
    """Execute every bound grid and certify the finest adjacent pair."""
    if not isinstance(protocol, IonAwareImpedanceGridProtocol):
        raise TypeError("protocol must be an IonAwareImpedanceGridProtocol")
    try:
        states = tuple(dc_states)
    except TypeError as exc:
        raise TypeError("dc_states must be an iterable") from exc
    if len(states) != len(protocol.grid_node_counts):
        raise IonAwareImpedanceGridCapabilityError(
            "DC-state count does not match the grid protocol"
        )
    if any(not isinstance(item, IonAwareDCResult) for item in states):
        raise TypeError("dc_states must contain IonAwareDCResult values")
    if any(item.stack != states[0].stack for item in states[1:]):
        raise IonAwareImpedanceGridCapabilityError(
            "every supplied grid DC state must use the same device stack"
        )
    for index, (state, impedance_protocol) in enumerate(
        zip(states, protocol.impedance_protocols, strict=True)
    ):
        grid = np.asarray(state.x, dtype=float)
        if len(grid) != protocol.grid_node_counts[index]:
            raise IonAwareImpedanceGridCapabilityError(
                f"grid {index} node count does not match the protocol"
            )
        if ion_aware_impedance_grid_sha256(grid) != protocol.grid_sha256s[index]:
            raise IonAwareImpedanceGridCapabilityError(
                f"grid {index} coordinates do not match the protocol hash"
            )
        if (
            state.protocol_hash != impedance_protocol.dc_protocol_sha256
            or ion_aware_dc_state_sha256(state.y)
            != impedance_protocol.dc_state_sha256
        ):
            raise IonAwareImpedanceGridCapabilityError(
                f"DC state {index} does not match its bound impedance protocol"
            )

    results = tuple(
        run_ion_aware_impedance(
            state.x,
            state.stack,
            impedance_protocol,
            dc_state=state,
            require_numerical_certificate=False,
            require_contact_certificate=False,
            require_frequency_window_certificate=False,
            progress=progress,
        )
        for state, impedance_protocol in zip(
            states,
            protocol.impedance_protocols,
            strict=True,
        )
    )
    certificate = _grid_certificate(protocol, results)
    result = IonAwareImpedanceGridResult(
        grid_results=results,
        protocol=protocol,
        certificate=certificate,
    )
    if require_numerical_certificate and not certificate.numerically_certified:
        raise IonAwareImpedanceGridCertificationError(
            "ion-aware impedance grid numerical certificate failed: "
            + ", ".join(certificate.numerical_reasons),
            result,
        )
    if require_contact_certificate and not certificate.thermodynamically_certified:
        raise IonAwareImpedanceGridCertificationError(
            "ion-aware impedance grid contact certificate failed",
            result,
        )
    if (
        require_frequency_window_certificate
        and not certificate.frequency_window_certified
    ):
        raise IonAwareImpedanceGridCertificationError(
            "ion-aware impedance grid frequency-window certificate failed",
            result,
        )
    return result


__all__ = [
    "GridFrequencyCertificate",
    "GridPairFrequencyAssessment",
    "ION_AWARE_IMPEDANCE_GRID_HASH_SCHEMA",
    "ION_AWARE_IMPEDANCE_GRID_PROTOCOL_SCHEMA",
    "IonAwareImpedanceGridCapabilityError",
    "IonAwareImpedanceGridCertificate",
    "IonAwareImpedanceGridCertificationError",
    "IonAwareImpedanceGridProtocol",
    "IonAwareImpedanceGridResult",
    "build_ion_aware_impedance_grid_protocol",
    "ion_aware_impedance_grid_sha256",
    "run_ion_aware_impedance_grid_ladder",
]

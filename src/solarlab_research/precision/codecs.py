"""Precision state authorities, reference-increment coordinates, and pair codecs.

Implements precision injection through ``solarlab.contracts`` state and
coordinate boundaries (``StateAuthority``, ``StateIncrement``, ``Point``,
``StateView``, ``FieldProjection``) without registering physics models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import (
    VOLT,
    ContractError,
    FieldProjection,
    Layout,
    PhysicalArray,
    Point,
    StateAuthority,
    StateIncrement,
    StateView,
    Vector,
    frozen_array,
    integer_indices,
)
from solarlab_research.precision.double_double import DD, DDArray, log1p

__all__ = [
    "AffineInputLiftCodec",
    "INPUT_LIFT_CHECKPOINT_SCHEMA",
    "INPUT_LIFT_REPRESENTATION",
    "INPUT_LIFT_STATE_SCHEMA",
    "InputLiftCheckpointError",
    "PAIR_REPRESENTATION",
    "PairStateAuthority",
    "ReferenceIncrementCoordinates",
    "ResearchCoordinateMap",
    "ResearchStateCodec",
    "decode_historical_r1_pair_state",
]

PAIR_REPRESENTATION = "float64-pair-v1"
INPUT_LIFT_REPRESENTATION = "affine-input-lift-dd-v1"
INPUT_LIFT_CHECKPOINT_SCHEMA = "SolarLabResearchInputLiftCheckpointV1"
INPUT_LIFT_STATE_SCHEMA = "SolarLabResearchInputLiftStateV1"
_COORDINATE_ROLE = "accepted_physical_state_imported_as_new_zero_coordinate_reference"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InputLiftCheckpointError(ValueError):
    """Raised when an input-lift checkpoint or pair state violates codec invariants."""


class ResearchCoordinateMap(Protocol):
    """Research-side coordinate map over a ``solarlab.contracts.Layout``."""

    @property
    def layout(self) -> Layout: ...
    @property
    def reference(self) -> str: ...
    def point(self, y: ArrayLike, time: float = 0.0, inputs: ArrayLike = ()) -> Point: ...
    def advance(
        self, left: Point, dy: ArrayLike, time: float, inputs: ArrayLike = ()
    ) -> tuple[Point, StateIncrement]: ...


class ResearchStateCodec(Protocol):
    """Research-side lossless state codec preserving compensated low words."""

    def encode_point(self, point: Point) -> dict[str, Any]: ...
    def decode_point(self, payload: Mapping[str, Any], *, layout: Layout) -> Point: ...


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InputLiftCheckpointError("payload is not finite canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _face_pairs(pairs: ArrayLike, size: int) -> NDArray[np.intp]:
    idx = integer_indices(pairs)
    if idx.ndim != 2 or idx.shape[1] != 2 or np.any(idx < 0) or np.any(idx >= size):
        raise ContractError("face_outside_support")
    return idx


@dataclass(frozen=True)
class PairStateAuthority:
    """``StateAuthority`` retaining normalized ``DDArray`` fields and exact increments."""

    layout_identity: str
    values: Mapping[str, DDArray]
    anchor_reference: str
    bound_coordinate: Vector | None = None
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.anchor_reference:
            raise ContractError("empty_authority_reference")
        owned: dict[str, DDArray] = {}
        for name, item in self.values.items():
            if not isinstance(item, DDArray) or not item.is_finite():
                raise ContractError("pair_authority_requires_finite_dd_arrays")
            owned[name] = item.immutable_copy()
        object.__setattr__(self, "values", MappingProxyType(owned))
        if self.bound_coordinate is not None:
            coord = frozen_array(self.bound_coordinate)
            if coord.ndim != 1:
                raise ContractError("invalid_coordinate_shape")
            object.__setattr__(self, "bound_coordinate", coord)
        sha = hashlib.sha256(
            b"pair-state-authority-v1:"
            + self.layout_identity.encode()
            + b":"
            + self.anchor_reference.encode()
        )
        for name in sorted(owned):
            sha.update(name.encode() + owned[name].identity_bytes())
        object.__setattr__(self, "identity", sha.hexdigest())

    def field(self, name: str) -> PhysicalArray:
        return self.values[name]

    def dd_field(self, name: str) -> DDArray:
        return self.values[name]

    def delta_from(self, left: StateView) -> Mapping[str, PhysicalArray]:
        if not isinstance(left.authority, PairStateAuthority):
            raise ContractError("authority_rebase_required")
        if left.authority.layout_identity != self.layout_identity:
            raise ContractError("increment_layout_mismatch")
        return {
            name: val.difference(left.authority.dd_field(name))
            for name, val in self.values.items()
        }

    def project(self, name: str) -> FieldProjection:
        return self.values[name].project_to_float(authority_identity=self.identity)

    def validate_binding(self, time: float, y: Vector, inputs: Vector, reference: str) -> None:
        if reference != self.anchor_reference:
            raise ContractError("coordinate_reference_mismatch")
        if self.bound_coordinate is not None and not np.array_equal(y, self.bound_coordinate):
            raise ContractError("authority_coordinate_binding_mismatch")

    def validate_transition(self, left: Point, right: Point) -> None:
        if not isinstance(left.state.authority, PairStateAuthority):
            raise ContractError("authority_rebase_required")
        if left.coordinate_reference != right.coordinate_reference:
            raise ContractError("coordinate_reference_mismatch")

    def electrochemical_difference(
        self,
        density_id: str,
        potential_id: str,
        pairs: ArrayLike,
        thermal_voltage: float,
        potential_sign: int,
    ) -> PhysicalArray | None:
        if not math.isfinite(thermal_voltage) or thermal_voltage <= 0.0 or potential_sign not in {-1, 1}:
            raise ContractError("invalid_activity_parameters")
        density = self.values[density_id]
        potential = self.values[potential_id]
        if len(density.shape) != 1 or potential.shape != density.shape:
            raise ContractError("activity_support_shape_mismatch")
        idx = _face_pairs(pairs, density.shape[0])
        log_term = density.take(idx[:, 1]).log_ratio(density.take(idx[:, 0])).values
        pot_diff = potential.take(idx[:, 1]).difference(potential.take(idx[:, 0])).values
        total = log_term + (potential_sign * pot_diff) / DD(thermal_voltage)
        return DDArray(total)

    def temporal_log_ratio(
        self, left: StateView, variable_id: str, indices: ArrayLike | None
    ) -> PhysicalArray | None:
        if not isinstance(left.authority, PairStateAuthority):
            return None
        before = left.authority.dd_field(variable_id)
        after = self.values[variable_id]
        if indices is not None:
            before = before.take(indices)
            after = after.take(indices)
        return after.log_ratio(before)

    def face_delta(
        self, left: StateView, variable_id: str, pairs: ArrayLike
    ) -> PhysicalArray | None:
        if not isinstance(left.authority, PairStateAuthority):
            return None
        delta = self.values[variable_id].difference(left.authority.dd_field(variable_id))
        if len(delta.shape) != 1:
            raise ContractError("face_outside_support")
        idx = _face_pairs(pairs, delta.shape[0])
        return delta.take(idx[:, 1]).difference(delta.take(idx[:, 0]))

    def electrochemical_delta(
        self,
        left: StateView,
        density_id: str,
        potential_id: str,
        pairs: ArrayLike,
        thermal_voltage: float,
        potential_sign: int,
    ) -> PhysicalArray | None:
        if not isinstance(left.authority, PairStateAuthority):
            return None
        specs = {spec.id: spec for spec in left.layout.variables}
        if specs[potential_id].unit != VOLT:
            raise ContractError("activity_potential_unit_mismatch")
        before_n = left.authority.dd_field(density_id)
        after_n = self.values[density_id]
        if len(after_n.shape) != 1:
            raise ContractError("face_outside_support")
        idx = _face_pairs(pairs, after_n.shape[0])
        # Compute temporal log ratios first, then spatial difference across face pairs,
        # avoiding O(1) endpoint cancellation on tiny sub-ULP increments.
        temporal_n = after_n.log_ratio(before_n).values
        log_change = temporal_n[idx[:, 1]] - temporal_n[idx[:, 0]]
        pot_delta = self.values[potential_id].difference(left.authority.dd_field(potential_id)).values
        pot_face = pot_delta[idx[:, 1]] - pot_delta[idx[:, 0]]
        return DDArray(log_change + (potential_sign * pot_face) / DD(thermal_voltage))


_AUTHORITY_CONFORMANCE: StateAuthority = PairStateAuthority(
    layout_identity="check",
    values={},
    anchor_reference="check",
)


@dataclass(frozen=True)
class ReferenceIncrementCoordinates:
    """Affine/log-lifting coordinate map around an exact ``DDArray`` anchor state.

    Variables listed in ``multiplicative_variables`` lift coordinates ``y`` via
    ``anchor * (1 + expm1(y)) = anchor * exp(y)`` so positive densities stay
    strictly positive and tiny ``dy`` updates preserve sub-ULP changes in ``lo``.
    All other variables lift affinely via ``anchor + DD(y)``.
    """

    layout: Layout
    anchor_fields: Mapping[str, DDArray]
    multiplicative_variables: frozenset[str] = frozenset()
    reference: str = field(init=False)

    def __post_init__(self) -> None:
        specs = {spec.id: spec for spec in self.layout.variables}
        if set(self.anchor_fields) != set(specs):
            raise ContractError("incomplete_anchor_fields")
        owned: dict[str, DDArray] = {}
        for name, spec in specs.items():
            arr = self.anchor_fields[name]
            if not isinstance(arr, DDArray) or arr.shape != spec.shape or not arr.is_finite():
                raise ContractError("anchor_shape_or_finiteness_mismatch")
            if name in self.multiplicative_variables and np.any(arr.values <= 0.0):
                raise ContractError("multiplicative_anchor_must_be_positive")
            owned[name] = arr.immutable_copy()
        unknown_mult = set(self.multiplicative_variables) - set(specs)
        if unknown_mult:
            raise ContractError("unknown_multiplicative_variable")
        object.__setattr__(self, "anchor_fields", MappingProxyType(owned))
        sha = hashlib.sha256(b"ref-inc-coords-v1:" + self.layout.identity.encode())
        for name in sorted(owned):
            mode = b"log:" if name in self.multiplicative_variables else b"aff:"
            sha.update(mode + name.encode() + owned[name].identity_bytes())
        object.__setattr__(self, "reference", f"pair-anchor:{sha.hexdigest()[:32]}")

    @classmethod
    def from_float_anchor(
        cls,
        layout: Layout,
        anchor: Mapping[str, ArrayLike],
        *,
        multiplicative_variables: Iterable_or_Set = (),
    ) -> ReferenceIncrementCoordinates:
        dd_map = {name: DDArray.from_parts(val) for name, val in anchor.items()}
        return cls(
            layout=layout,
            anchor_fields=dd_map,
            multiplicative_variables=frozenset(multiplicative_variables),
        )

    def _lift_fields(self, words: Vector) -> dict[str, DDArray]:
        lifted: dict[str, DDArray] = {}
        for spec in self.layout.variables:
            coord = DD(words[self.layout.offsets[spec.id]].reshape(spec.shape))
            base = self.anchor_fields[spec.id].values
            if spec.id in self.multiplicative_variables:
                # base + base * expm1(coord) preserves tiny sub-ULP multiplicative updates in lo.
                val = base + base * coord.expm1()
            else:
                val = base + coord
            lifted[spec.id] = DDArray(val)
        return lifted

    def point(self, y: ArrayLike, time: float = 0.0, inputs: ArrayLike = ()) -> Point:
        words = frozen_array(y)
        if words.shape != (self.layout.size,):
            raise ContractError("coordinate_layout_mismatch")
        fields = self._lift_fields(words)
        authority = PairStateAuthority(
            layout_identity=self.layout.identity,
            values=fields,
            anchor_reference=self.reference,
            bound_coordinate=words,
        )
        view = StateView(self.layout, authority=authority)
        return Point(time, words, frozen_array(inputs), view, self.reference)

    def advance(
        self, left: Point, dy: ArrayLike, time: float, inputs: ArrayLike = ()
    ) -> tuple[Point, StateIncrement]:
        step = frozen_array(dy)
        if step.shape != left.y.shape or left.coordinate_reference != self.reference:
            raise ContractError("coordinate_reference_mismatch")
        if not isinstance(left.state.authority, PairStateAuthority):
            raise ContractError("authority_rebase_required")
        # Accumulate coordinate update in DD if needed; solver coordinates remain binary64 vector
        # relative to the anchor, while physical DDArray fields retain full (hi, lo) precision.
        new_y_dd = DD(left.y) + DD(step)
        new_y = frozen_array(new_y_dd.hi)
        # Reconstruct right fields using exact DD coordinate sum so sub-ULP dy on nonzero left.y
        # is never lost before lifting into physical fields.
        right_fields: dict[str, DDArray] = {}
        for spec in self.layout.variables:
            sl = self.layout.offsets[spec.id]
            coord_dd = new_y_dd[sl].reshape(spec.shape)
            base = self.anchor_fields[spec.id].values
            if spec.id in self.multiplicative_variables:
                val = base + base * coord_dd.expm1()
            else:
                val = base + coord_dd
            right_fields[spec.id] = DDArray(val)
        authority = PairStateAuthority(
            layout_identity=self.layout.identity,
            values=right_fields,
            anchor_reference=self.reference,
            bound_coordinate=new_y,
        )
        right_view = StateView(self.layout, authority=authority)
        right = Point(time, new_y, frozen_array(inputs), right_view, self.reference)
        increment = StateIncrement.from_points(left, right)
        return right, increment

    def rebase(self, accepted: Point) -> tuple[ReferenceIncrementCoordinates, Point]:
        """Import an accepted physical state as a fresh zero-coordinate reference."""
        if accepted.state.layout.identity != self.layout.identity:
            raise ContractError("increment_layout_mismatch")
        if not isinstance(accepted.state.authority, PairStateAuthority):
            raise ContractError("authority_rebase_required")
        new_anchor = {
            spec.id: accepted.state.authority.dd_field(spec.id)
            for spec in self.layout.variables
        }
        rebased_map = ReferenceIncrementCoordinates(
            layout=self.layout,
            anchor_fields=new_anchor,
            multiplicative_variables=self.multiplicative_variables,
        )
        zero_point = rebased_map.point(
            np.zeros(self.layout.size, dtype=np.float64),
            time=accepted.time,
            inputs=accepted.inputs,
        )
        return rebased_map, zero_point


Iterable_or_Set = Any


class AffineInputLiftCodec:
    """Lossless JSON checkpoint codec for ``PairStateAuthority`` states and input lifts."""

    def encode_point(self, point: Point) -> dict[str, Any]:
        if not isinstance(point.state.authority, PairStateAuthority):
            raise InputLiftCheckpointError("codec requires a PairStateAuthority state")
        represented: dict[str, dict[str, Any]] = {}
        high_word: dict[str, Any] = {}
        for spec in point.state.layout.variables:
            dd = point.state.authority.dd_field(spec.id).values
            represented[spec.id] = dd.to_pair_dict()
            high_word[spec.id] = dd.hi.tolist() if dd.ndim else float(dd.hi.item())
        snapshot = {
            "schema": INPUT_LIFT_STATE_SCHEMA,
            "operator_representation": INPUT_LIFT_REPRESENTATION,
            "layout_identity": point.state.layout.identity,
            "coordinate_reference": point.coordinate_reference,
            "high_word_state": high_word,
            "represented_fields": represented,
        }
        record: dict[str, Any] = {
            "schema": INPUT_LIFT_CHECKPOINT_SCHEMA,
            "operator_representation": INPUT_LIFT_REPRESENTATION,
            "coordinate_role": _COORDINATE_ROLE,
            "time_s": float(point.time),
            "inputs": point.inputs.tolist(),
            "source_coordinate": point.y.tolist(),
            "prior_point_identity": point.identity,
            "state_identity": _digest(represented),
            "snapshot": snapshot,
        }
        record["sha256"] = _digest(record)
        return json.loads(_canonical_json(record))  # type: ignore[no-any-return]

    def decode_point(
        self,
        payload: Mapping[str, Any],
        *,
        layout: Layout,
        multiplicative_variables: Iterable_or_Set = (),
    ) -> Point:
        """Restore a checkpoint as a new zero-coordinate ``Point`` preserving all ``(hi, lo)`` words."""
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != INPUT_LIFT_CHECKPOINT_SCHEMA
            or payload.get("operator_representation") != INPUT_LIFT_REPRESENTATION
            or payload.get("coordinate_role") != _COORDINATE_ROLE
        ):
            raise InputLiftCheckpointError("checkpoint schema, representation, or role mismatch")
        expected_sha = payload.get("sha256")
        if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
            raise InputLiftCheckpointError("checkpoint sha256 digest is missing or malformed")
        unsigned = {k: v for k, v in payload.items() if k != "sha256"}
        if _digest(unsigned) != expected_sha:
            raise InputLiftCheckpointError("checkpoint content sha256 mismatch")

        snapshot = payload.get("snapshot")
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("schema") != INPUT_LIFT_STATE_SCHEMA
            or snapshot.get("layout_identity") != layout.identity
        ):
            raise InputLiftCheckpointError("checkpoint snapshot or layout identity mismatch")

        specs = {spec.id: spec for spec in layout.variables}
        represented = snapshot.get("represented_fields")
        high_word = snapshot.get("high_word_state")
        if (
            not isinstance(represented, Mapping)
            or not isinstance(high_word, Mapping)
            or set(represented) != set(specs)
            or set(high_word) != set(specs)
        ):
            raise InputLiftCheckpointError("checkpoint field coverage differs from layout")
        if _digest(represented) != payload.get("state_identity"):
            raise InputLiftCheckpointError("checkpoint state_identity mismatch")

        anchor: dict[str, DDArray] = {}
        for name, spec in specs.items():
            try:
                dd = DD.from_pair_dict(represented[name], expected_shape=spec.shape)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InputLiftCheckpointError(f"invalid DD pair for {name}: {exc}") from exc
            hi_check = frozen_array(high_word[name])
            if hi_check.shape != spec.shape or hi_check.tobytes() != dd.hi.tobytes():
                raise InputLiftCheckpointError(f"high_word_state disagrees with DD hi for {name}")
            anchor[name] = DDArray(dd)

        coord_map = ReferenceIncrementCoordinates(
            layout=layout,
            anchor_fields=anchor,
            multiplicative_variables=frozenset(multiplicative_variables),
        )
        time_val = payload.get("time_s")
        if isinstance(time_val, bool) or not isinstance(time_val, (int, float)) or not math.isfinite(time_val):
            raise InputLiftCheckpointError("checkpoint time_s must be finite")
        inputs_arr = frozen_array(payload.get("inputs", ()))
        return coord_map.point(
            np.zeros(layout.size, dtype=np.float64),
            time=float(time_val),
            inputs=inputs_arr,
        )

    def decode_with_map(
        self,
        payload: Mapping[str, Any],
        *,
        layout: Layout,
        multiplicative_variables: Iterable_or_Set = (),
    ) -> tuple[ReferenceIncrementCoordinates, Point]:
        point = self.decode_point(
            payload, layout=layout, multiplicative_variables=multiplicative_variables
        )
        assert isinstance(point.state.authority, PairStateAuthority)
        coord_map = ReferenceIncrementCoordinates(
            layout=layout,
            anchor_fields={
                spec.id: point.state.authority.dd_field(spec.id) for spec in layout.variables
            },
            multiplicative_variables=frozenset(multiplicative_variables),
        )
        return coord_map, point


def decode_historical_r1_pair_state(record: Mapping[str, Any]) -> dict[str, DD]:
    """Decode a historical ``R1CommonStatePairV2`` record without relabeling it.

    Validates its recorded ``sha256``, ``seed_preparation_sha256``, normalized
    17 ``(hi, lo)`` fields, and ``high == snapshot`` invariants.
    """
    if (
        not isinstance(record, Mapping)
        or record.get("schema") != "R1CommonStatePairV2"
        or record.get("representation") != PAIR_REPRESENTATION
    ):
        raise InputLiftCheckpointError("historical record must have schema R1CommonStatePairV2")
    unsigned = {k: v for k, v in record.items() if k != "sha256"}
    if _digest(unsigned) != record.get("sha256"):
        raise InputLiftCheckpointError("historical R1CommonStatePairV2 sha256 mismatch")
    seed = record.get("seed_preparation")
    if not isinstance(seed, Mapping) or seed.get("schema") != "R1CommonStateV1":
        raise InputLiftCheckpointError("historical pair record lacks R1CommonStateV1 seed")
    seed_unsigned = {k: v for k, v in seed.items() if k != "sha256"}
    if _digest(seed_unsigned) != seed.get("sha256") or record.get("seed_preparation_sha256") != seed.get("sha256"):
        raise InputLiftCheckpointError("historical R1CommonStateV1 seed sha256 mismatch")

    state = record.get("state")
    if not isinstance(state, Mapping):
        raise InputLiftCheckpointError("historical pair state mapping is missing")
    fine_names = (
        "phi_V",
        "dqfn_V",
        "dqfp_V",
        "n_m3",
        "p_m3",
        "positive_m3",
        "occupancy",
        "sheet_charge_C_m2",
        "trace_potential_V",
        "trace_state_m3",
        "storage",
        "poisson_residual_C_m2",
        "electron_current_A_m2",
        "hole_current_A_m2",
        "positive_flux_m2_s",
        "positive_rate_m3_s",
        "boundary_flux_m2_s",
    )
    decoded: dict[str, DD] = {}
    for name in fine_names:
        hi_key = f"precision_{name}_hi"
        lo_key = f"precision_{name}_lo"
        if hi_key not in state or lo_key not in state:
            raise InputLiftCheckpointError(f"missing precision pair words for {name}")
        dd = DD.from_pair_dict({"hi": state[hi_key], "lo": state[lo_key]})
        if name in state:
            snap_hi = frozen_array(state[name])
            if snap_hi.shape != dd.shape or not np.array_equal(snap_hi, dd.hi):
                raise InputLiftCheckpointError(f"binary64 snapshot differs from high word for {name}")
        decoded[name] = dd
    # Verify cancellation-safe log1p availability on decoded density changes vs seed
    seed_n = DD(seed["state"]["n_m3"])
    _ = log1p((decoded["n_m3"] - seed_n) / seed_n)
    return decoded

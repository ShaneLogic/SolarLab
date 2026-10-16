"""Bounded G2-R reference BE driver and its separately identified history.

The public single-step solver is reused without mutation. Only the explicitly
admitted B entry point constructs a real device. Manufactured consumers can
exercise scheduling, physical words, failure handling and readback independently.
A complete trajectory never awards its own reference-error qualification.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import ctypes
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

from scripts.benchmarks.conservative_device import EquationBudget, conservative_be_step
from scripts.benchmarks.contract_prototype import (
    ContractError,
    FloatArithmetic,
    Layout,
    PhysicalScaling,
    Point,
    SECOND,
)
from scripts.benchmarks.native_history import HistoryReader, HistoryWriter
from scripts.benchmarks.g2_route import (
    _channel as route_channel,
    _validate_supervision as validate_route_supervision,
)
from scripts.benchmarks.precision_prototype import (
    DoubleArithmetic,
    DoubleArray,
    FrameInputExpansion,
    MappedAuthority,
    PrimitiveExpansion,
    RelativeCoordinates,
    decode_point,
    encode_point,
    physical_state_words,
)

SCHEMA = "solarlab.g2-reference-be-history.v1"
METHOD = "conservative-backward-euler"
POINT_SCHEMA = "solarlab.g2-reference-point.v1"
METHOD_SCHEMA = "solarlab.g2-reference-method.v1"
RUNTIME_SCHEMA = "solarlab.g2-reference-runtime.v2"
STEP_OPTIONS = {"max_iterations": 12, "max_line_search": 8, "globalization": "residual"}
SECANT_ARITHMETICS = {
    "public-double-arithmetic-divide-v1": DoubleArithmetic,
    "public-float-arithmetic-divide-v1": FloatArithmetic,
}
GIB = 1024**3
TAG_FIELDS = (
    "segment",
    "index",
    "time_hex",
    "event_side",
    "protocol_initial",
    "protocol_terminal",
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
RUNTIME_PHASES = (
    "setup",
    "be_step",
    "state_check",
    "step_validation",
    "point_observation",
    "interval_observation",
    "history_io",
    "artifact_io",
    "reader",
    "reference_error",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError("reference_be_" + reason)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value: Any) -> str:
    return sha256(canonical(value)).hexdigest()


def _fraction(value: Any) -> Fraction:
    _require(type(value) in (int, float, Fraction), "number_type")
    if type(value) is float:
        _require(math.isfinite(value), "nonfinite_number")
    return Fraction(value)


def _pair(value: Any) -> list[int]:
    x = _fraction(value)
    return [x.numerator, x.denominator]


def _unpair(value: Any) -> Fraction:
    _require(
        isinstance(value, list)
        and len(value) == 2
        and all(type(v) is int for v in value)
        and value[1] > 0,
        "rational_encoding",
    )
    return Fraction(*value)


def _words(value: Any) -> dict:
    arrays = (value,) if isinstance(value, np.ndarray) else value.words
    _require(bool(arrays), "missing_words")
    shape = arrays[0].shape
    _require(
        all(a.shape == shape and np.isfinite(a).all() for a in arrays),
        "word_shape_or_finiteness",
    )
    return {
        "shape": list(shape),
        "words_hex": [[float(v).hex() for v in a.ravel()] for a in arrays],
    }


def _validate_words(value: Mapping, size: int | None = None) -> None:
    shape = value["shape"]
    _require(
        isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape),
        "word_shape",
    )
    count = math.prod(shape)
    _require(size is None or shape == [size], "word_size")
    rows = value["words_hex"]
    _require(isinstance(rows, list) and 1 <= len(rows) <= 32, "word_count")
    for row in rows:
        _require(len(row) == count, "word_row")
        for s in row:
            _hex_time(s, nonnegative=False)


def _hex_time(value: Any, *, nonnegative: bool = True) -> float:
    _require(type(value) is str, "hex_type")
    try:
        x = float.fromhex(value)
    except ValueError as exc:
        raise ContractError("reference_be_hex_number") from exc
    _require(
        math.isfinite(x) and (not nonnegative or x >= 0) and x.hex() == value,
        "hex_noncanonical_or_nonfinite",
    )
    return x


def _physical(point: Point) -> dict:
    return {
        key: {
            "shape": list(arrays[0].shape),
            "words_hex": [[float(v).hex() for v in a.ravel()] for a in arrays],
        }
        for key, arrays in physical_state_words(point.state).items()
    }


def _physical_exact(point: Point) -> dict[str, tuple[Fraction, ...]]:
    return {
        key: tuple(
            sum((Fraction(float(a.ravel()[i])) for a in arrays), Fraction())
            for i in range(arrays[0].size)
        )
        for key, arrays in physical_state_words(point.state).items()
    }


def _point_root(point: Point) -> Point:
    authority = point.state.authority
    root = getattr(authority, "fixed_reference", None)
    return root if isinstance(root, Point) else point


def pack_point(point: Point, reference: dict) -> dict:
    """Deduplicate only equal immutable source data; the public digest is kept."""
    record = encode_point(point)
    authority = record["payload"].get("authority")
    ref_authority = reference["payload"].get("authority")
    key = reference["sha256"]
    if authority is not None and ref_authority is not None:
        if authority.get("anchor") == ref_authority.get("anchor"):
            authority["anchor"] = {"be_anchor_reference": key}
        contract = authority.get("coordinate_contract")
        if contract is not None and contract.get("reference") == reference:
            contract["reference"] = {"be_point_reference": key}
    return {"schema": POINT_SCHEMA, "reference_sha256": key, "public_point": record}


def unpack_point(value: Mapping, reference: dict, layout: Layout) -> Point:
    _require(
        value.get("schema") == POINT_SCHEMA
        and value.get("reference_sha256") == reference["sha256"],
        "point_reference",
    )
    record = json.loads(json.dumps(value["public_point"], allow_nan=False))
    authority = record["payload"].get("authority")
    if authority is not None:
        if authority.get("anchor") == {"be_anchor_reference": reference["sha256"]}:
            authority["anchor"] = reference["payload"]["authority"]["anchor"]
        contract = authority.get("coordinate_contract")
        if contract is not None and contract.get("reference") == {
            "be_point_reference": reference["sha256"]
        }:
            contract["reference"] = reference
    return decode_point(record, layout)


@dataclass(frozen=True)
class TimeStep:
    index: int
    segment_index: int
    segment_step: int
    window_index: int
    left: float
    right: float
    h: float
    tag_index: int | None


@dataclass(frozen=True)
class Schedule:
    tags: tuple[dict, ...]
    steps: tuple[TimeStep, ...]
    level: int
    seed: Fraction
    identity: str


def compile_schedule(
    segments: tuple,
    tags: list | tuple,
    *,
    level: int,
    seed: Fraction = Fraction(1, 512),
    step_cap: int = 200000,
) -> Schedule:
    """Construct actual rounded nested nodes, never accumulate a decimal h."""
    _require(type(level) is int and 0 <= level <= 3, "ladder_level")
    seed = _fraction(seed)
    _require(
        seed > 0 and type(step_cap) is int and 1 <= step_cap <= 200000,
        "schedule_budget",
    )
    _require(
        0 < len(segments) <= 16 and len({s.id for s in segments}) == len(segments),
        "segments",
    )
    _require(len(tags) <= 131, "tag_budget")
    kept = tuple({k: t[k] for k in TAG_FIELDS} for t in tags)
    steps: list[TimeStep] = []
    cursor = 0
    for ordinal, segment in enumerate(segments):
        _require(
            math.isfinite(segment.start)
            and math.isfinite(segment.end)
            and segment.start < segment.end
            and segment.start >= 0,
            "segment_times",
        )
        if ordinal:
            _require(segments[ordinal - 1].end == segment.start, "segment_gap")
        first = cursor
        while cursor < len(kept) and kept[cursor]["segment"] == segment.id:
            cursor += 1
        group = kept[first:cursor]
        _require(len(group) >= 2, "segment_tags")
        times = []
        for i, tag in enumerate(group):
            expected_side = (
                "right" if i == 0 else "left" if i == len(group) - 1 else "interior"
            )
            _require(
                type(tag["index"]) is int
                and tag["index"] == i
                and tag["event_side"] == expected_side
                and type(tag["protocol_initial"]) is bool
                and type(tag["protocol_terminal"]) is bool,
                "tag_identity",
            )
            _require(
                tag["protocol_initial"] == (ordinal == i == 0)
                and (
                    not tag["protocol_terminal"]
                    or (ordinal == len(segments) - 1 and i == len(group) - 1)
                ),
                "protocol_tag_identity",
            )
            times.append(_hex_time(tag["time_hex"]))
        _require(
            times[0].hex() == float(segment.start).hex()
            and times[-1].hex() == float(segment.end).hex(),
            "tag_segment_endpoints",
        )
        local_count = 0
        for window, (left, right) in enumerate(zip(times, times[1:])):
            delta = Fraction(right) - Fraction(left)
            _require(delta > 0, "nonpositive_window")
            count = 1
            while delta / count > seed:
                count *= 2
                _require(count <= step_cap, "step_cap")
            count *= 2**level
            _require(len(steps) + count <= step_cap, "step_cap")
            previous = left
            for j in range(1, count + 1):
                current = float(Fraction(left) + delta * Fraction(j, count))
                exact_h = Fraction(current) - Fraction(previous)
                h = float(exact_h)
                _require(
                    h > 0 and Fraction(h) == exact_h and previous + h == current,
                    "unrepresentable_step",
                )
                local_count += 1
                steps.append(
                    TimeStep(
                        len(steps) + 1,
                        ordinal,
                        local_count,
                        window,
                        previous,
                        current,
                        h,
                        first + window + 1 if j == count else None,
                    )
                )
                previous = current
            _require(previous.hex() == right.hex(), "changed_endpoint")
    _require(cursor == len(kept) and bool(steps), "unconsumed_tags")
    table = {
        "tags": kept,
        "level": level,
        "seed": _pair(seed),
        "segments": [
            {
                "id": s.id,
                "start": float(s.start).hex(),
                "end": float(s.end).hex(),
                "voltage": [float(v).hex() for v in s.voltage],
                "photons": [float(v).hex() for v in s.photons],
            }
            for s in segments
        ],
        "steps": [
            [s.segment_index, s.window_index, s.left.hex(), s.right.hex(), s.h.hex()]
            for s in steps
        ],
    }
    return Schedule(kept, tuple(steps), level, seed, digest(table))


@dataclass(frozen=True)
class RunIdentity:
    case_id: str
    input_sha256: str
    method_contract_sha256: str
    plan_sha256: str
    source_commit: str
    environment_identity: str
    schedule_identity: str
    source_identity: str
    initial_point_identity: str
    controls_sha256: str

    def payload(self) -> dict:
        result = asdict(self)
        _require(
            all(type(v) is str and v for v in result.values()), "execution_identity"
        )
        for k in (
            "input_sha256",
            "method_contract_sha256",
            "plan_sha256",
            "schedule_identity",
            "source_identity",
            "initial_point_identity",
            "controls_sha256",
        ):
            _require(
                len(result[k]) == 64
                and all(c in "0123456789abcdef" for c in result[k]),
                "digest_identity",
            )
        return result


@dataclass(frozen=True)
class Limits:
    wall_s: float
    rss_bytes: int = 4 * GIB
    output_bytes: int = 4 * GIB
    logical_bytes: int = 32 * GIB
    record_bytes: int = 1048576
    step_cap: int = 200000
    residual_cap: int = 2000000
    metadata_reserve: int = 4 * 1048576

    def validate(self) -> None:
        _require(
            type(self.wall_s) in (float, int) and 0 < self.wall_s <= 43200, "wall_limit"
        )
        for value, maximum in (
            (self.rss_bytes, 4 * GIB),
            (self.output_bytes, 4 * GIB),
            (self.logical_bytes, 32 * GIB),
            (self.record_bytes, 1048576),
            (self.step_cap, 200000),
            (self.residual_cap, 2000000),
        ):
            _require(type(value) is int and 0 < value <= maximum, "resource_limit")
        _require(
            type(self.metadata_reserve) is int
            and 8 <= self.metadata_reserve < self.output_bytes,
            "metadata_reserve",
        )


@dataclass(frozen=True)
class Backend:
    problem: Any
    coordinates: RelativeCoordinates
    initial: Point
    scaling: PhysicalScaling
    budgets: Mapping[str, EquationBudget]
    segments: tuple
    feasible: Callable[[Point], bool]
    check_state: Callable[[Point], dict]
    observe_point: Callable[[Point, Any], dict]
    observe_interval: Callable[[Point, Point, Any, Any, TimeStep], dict]
    metadata: dict


def effective_controls(backend: Backend) -> dict:
    """Expose the exact applied BE and observation controls for pre-run pinning."""
    return {
        "method": METHOD,
        "step_options": dict(STEP_OPTIONS),
        "storage_secant_arithmetic": _secant_arithmetic(backend.problem),
        "equation_budgets": {
            k: {
                "unit": asdict(v.unit),
                "absolute_hex": [float(x).hex() for x in v.absolute],
            }
            for k, v in backend.budgets.items()
        },
        "solver_scaling": {
            k: [float(v).hex() for v in getattr(backend.scaling, k)]
            for k in ("rows", "columns")
        },
        "observation_controls": backend.metadata["observation_controls"],
    }


def _secant_arithmetic(problem: Any) -> str:
    kind = (
        type(problem.arithmetic) if problem.arithmetic is not None else FloatArithmetic
    )
    for name, arithmetic in SECANT_ARITHMETICS.items():
        if kind is arithmetic:
            return name
    raise ContractError("reference_be_unsupported_secant_arithmetic")


def _protocol_at(segment: Any, time_value: float) -> tuple[np.ndarray, np.ndarray, str]:
    """Evaluate the frozen two-input, piecewise affine protocol independently.

    Endpoint words have priority over interpolation, exactly as in the public
    ProtocolSegment contract. An event endpoint belongs to its declared segment;
    its slope must not be borrowed from the next segment.
    """
    _require(segment.start <= time_value <= segment.end, "protocol_time")
    first = np.array([segment.voltage[0], segment.photons[0]], dtype=float)
    last = np.array([segment.voltage[1], segment.photons[1]], dtype=float)
    rate = (last - first) / (segment.end - segment.start)
    if time_value == segment.start:
        inputs, side = first, "right"
    elif time_value == segment.end:
        inputs, side = last, "left"
    else:
        inputs, side = first + rate * (time_value - segment.start), "interior"
    _require(
        np.isfinite(inputs).all() and np.isfinite(rate).all(), "protocol_finiteness"
    )
    return inputs, rate, side


def _input_words(values: Any) -> list[str]:
    return [float(v).hex() for v in values]


def _check_step_protocol(row: Mapping, left: Point, right: Point, segment: Any) -> None:
    before, _, _ = _protocol_at(segment, left.time)
    after, rate, side = _protocol_at(segment, right.time)
    _require(
        _input_words(left.inputs) == _input_words(before)
        and _input_words(right.inputs) == _input_words(after)
        and row.get("input_rate_hex") == _input_words(rate)
        and row.get("input_event_side") == side,
        "step_protocol_inputs",
    )


def _coordinate_trial(
    left: Point, local: PrimitiveExpansion, time_value: float, inputs: Any
) -> Point:
    """Replay public coordinate construction only; no residual/Jacobian/solve."""
    layout = left.state.layout
    coordinates = RelativeCoordinates(
        layout, {v.id: "linear" for v in layout.variables}
    )
    authority = left.state.authority
    _require(
        isinstance(authority, MappedAuthority)
        and authority.map_identity == coordinates.identity,
        "step_coordinate_authority",
    )
    if authority.coordinate_kind == "local":
        return coordinates.advance(left, local, time_value, inputs)[0]
    reference = authority.fixed_reference
    _require(
        authority.coordinate_kind == "fixed-reference"
        and isinstance(reference, Point)
        and all(v.is_zero() for v in reference.state.authority.primitives.values()),
        "step_original_reference",
    )
    values = [authority.primitives[v.id] for v in layout.variables]
    kind = FrameInputExpansion if authority.frame_inputs else PrimitiveExpansion
    count = 12 if authority.frame_inputs else 4
    base = kind(
        tuple(
            np.concatenate([v.words[i].ravel() for v in values]) for i in range(count)
        )
    )
    return coordinates.trial(
        reference, base.add(local), time_value, inputs, predecessor=left
    )[0]


def _check_step_arithmetic(
    row: Mapping, left: Point, right: Point, storage_size: int, arithmetic: str
) -> None:
    """Bind full-word increments and the *rounded* public storage division."""
    layout = left.state.layout
    local = _word_values(row["local_increment_words"], layout.size)
    before, after = _physical_exact(left), _physical_exact(right)
    for variable in layout.variables:
        offset = layout.offsets[variable.id]
        _require(
            local[offset]
            == tuple(
                b - a
                for a, b in zip(before[variable.id], after[variable.id], strict=True)
            ),
            "step_local_increment",
        )
    raw_local = row["local_increment_words"]["words_hex"]
    _require(len(raw_local) == 4, "step_local_increment_representation")
    primitive = PrimitiveExpansion(
        tuple(np.array([float.fromhex(v) for v in w]) for w in raw_local)
    )
    reconstructed = _coordinate_trial(left, primitive, right.time, right.inputs)
    _require(reconstructed.identity == right.identity, "step_local_increment_Point")
    _validate_words(row["storage_delta_words"], storage_size)
    _validate_words(row["storage_secant_words"], storage_size)
    _require(arithmetic in SECANT_ARITHMETICS, "unsupported_secant_arithmetic")
    parts = [
        np.array([float.fromhex(v) for v in w])
        for w in row["storage_delta_words"]["words_hex"]
    ]
    a = SECANT_ARITHMETICS[arithmetic]()
    if type(a) is DoubleArithmetic:
        _require(len(parts) == 2, "storage_delta_representation")
        delta = DoubleArray(*parts)
    else:
        _require(len(parts) == 1, "storage_delta_representation")
        delta = parts[0]
    # Division by a non-power-of-two h rounds. Replaying the declared operation
    # checks its representation without asserting secant*h == delta in R.
    h = _hex_time(row["h_hex"])
    expected = a.freeze(a.divide(a.array(delta), a.array(np.full(storage_size, h))))
    _require(_words(expected) == row["storage_secant_words"], "storage_secant_division")


def _diagnostic_rational(value: Any) -> Fraction:
    _require(
        isinstance(value, dict) and set(value) == {"rational"},
        "newton_diagnostics_rational",
    )
    return _unpair(value["rational"])


def _is_digest(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class RuntimeLedger:
    """Bounded component totals and protocol-phase walls; no scientific decisions."""

    def __init__(self, clock=None, sink=None):
        self.clock = clock or time.monotonic_ns
        self.sink = sink
        self.started_ns = self.clock()
        self.components = {
            k: {"calls": 0, "exceptions": 0, "wall_ns": 0} for k in RUNTIME_PHASES
        }
        self.segments = []
        self.active = None
        self.peak_rss_bytes = _rss_bytes()

    @contextmanager
    def span(self, name):
        _require(name in self.components, "runtime_phase")
        start = self.clock()
        self.components[name]["calls"] += 1
        try:
            yield
        except BaseException:
            self.components[name]["exceptions"] += 1
            raise
        finally:
            elapsed = self.clock() - start
            self.components[name]["wall_ns"] += elapsed
            if self.active is not None:
                self.active["components"][name] += elapsed
            self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())

    def begin_segment(self, name):
        self.end_segment("completed")
        _require(len(self.segments) < 16, "runtime_segment_count")
        self.active = {
            "segment": name,
            "started_ns": self.clock(),
            "accepted_steps": 0,
            "residual_evaluations": 0,
            "factorizations": 0,
            "linear_solves": 0,
            "components": {k: 0 for k in RUNTIME_PHASES},
        }
        if self.sink is not None:
            self.sink(
                {
                    "kind": "protocol_phase_start",
                    "segment": name,
                    "started_ns": self.active["started_ns"],
                }
            )

    def accepted(self, step):
        if self.active is not None:
            self.active["accepted_steps"] += 1
            for key, value in (
                ("residual_evaluations", step.residual_evaluations),
                ("factorizations", step.factorization_calls),
                ("linear_solves", step.linear_solve_calls),
            ):
                self.active[key] += value

    def end_segment(self, outcome):
        if self.active is not None:
            end = self.clock()
            self.active.update(
                ended_ns=end, wall_ns=end - self.active["started_ns"], outcome=outcome
            )
            self.segments.append(self.active)
            if self.sink is not None:
                self.sink({"kind": "protocol_phase_end", **self.active})
            self.active = None

    def snapshot(self):
        _require(self.active is None, "runtime_open_segment")
        return {
            "clock": "time.monotonic_ns",
            "components": self.components,
            "segments": self.segments,
            "peak_process_RSS_bytes": self.peak_rss_bytes,
            "semantics": "component spans are inclusive; segment wall also includes guards, control and instrumentation; do not add segment and component walls",
            "not_entered": [k for k, v in self.components.items() if v["calls"] == 0],
            "unmeasured": [
                "interpreter bootstrap before this entry",
                "runtime receipt publication and parent group closure",
                "work in separately admitted processes",
            ],
            "scientific_decisions_use_timing": False,
        }


def _boot_identity() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        value = result.stdout.strip()
    else:
        value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    _require(bool(value), "runtime_boot_identity")
    return sys.platform + ":" + value


def pipeline_clock(pipeline_id: str, wall_s: float) -> dict:
    """Return a pre-launch clock anchor for the commander to persist and pin.

    No directory, process, admission, or clock restart is created by this helper.
    Follow-up stages must retain this same immutable anchor on the same boot.
    """
    _require(type(pipeline_id) is str and bool(pipeline_id), "pipeline_identity")
    _require(
        type(wall_s) in (int, float) and math.isfinite(wall_s) and 0 < wall_s <= 43200,
        "pipeline_wall",
    )
    boot = _boot_identity()
    start = time.monotonic_ns()
    duration = math.floor(Fraction(wall_s) * 10**9)
    return {
        "schema": "solarlab.g2-reference-pipeline-clock.v1",
        "pipeline_id": pipeline_id,
        "clock": "time.monotonic_ns",
        "boot_identity": boot,
        "started_ns": start,
        "deadline_ns": start + duration,
        "wall_s": wall_s,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def runtime_threads() -> dict:
    """Record the actual Accelerate threading mode used by the locked Mac entry."""
    environment = {k: os.environ.get(k) for k in THREAD_ENVIRONMENT}
    _require(all(v == "1" for v in environment.values()), "runtime_thread_environment")
    _require(sys.platform == "darwin", "runtime_actual_thread_probe_unavailable")
    path = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib"
    library = ctypes.CDLL(path)
    library.BLASSetThreading.argtypes = [ctypes.c_int]
    library.BLASSetThreading.restype = ctypes.c_int
    library.BLASGetThreading.restype = ctypes.c_int
    before = int(library.BLASGetThreading())
    library.BLASSetThreading(1)
    actual = int(library.BLASGetThreading())
    _require(actual == 1, "runtime_actual_BLAS_thread_mode")
    return {
        "library": path,
        "getter": "BLASGetThreading",
        "before_mode": before,
        "measured_mode": actual,
        "required_single_thread_mode": 1,
        "thread_environment": environment,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "checked_ns": time.monotonic_ns(),
    }


def _runtime_thread_end(receipt):
    library = ctypes.CDLL(receipt["library"])
    library.BLASGetThreading.restype = ctypes.c_int
    actual = int(library.BLASGetThreading())
    receipt["final_measured_mode"] = actual
    receipt["final_checked_ns"] = time.monotonic_ns()
    _require(actual == 1, "runtime_final_BLAS_thread_mode")


class StageRuntime:
    """One separately admitted stage, bounded by the original pipeline clock."""

    def __init__(self, manifest, plan, action):
        self.manifest, self.plan, self.action = manifest, plan, action
        self.case_id = manifest.get(
            "case_id", manifest.get("identity", {}).get("case_id")
        )
        _require(
            plan.get("status") == "admitted"
            and plan.get("authority", {}).get("scientific_execution_authorized")
            is True,
            "runtime_plan_not_admitted",
        )
        matches = [c for c in plan["cases"] if c["id"] == self.case_id]
        _require(len(matches) == 1, "runtime_case_identity")
        self.execution = matches[0]["execution"]
        _require(self.execution.get("stage") == action, "runtime_stage_not_admitted")
        _require(
            _integer(self.execution.get("channel")) and self.execution["channel"] < 4,
            "runtime_channel",
        )
        _require(
            plan["resources"].get("WS1_channels") == 4
            and plan["resources"].get("threads_per_case") == 1,
            "runtime_resource_profile",
        )
        self.supervision = validate_route_supervision(plan, self.execution)
        self.limits = Limits(**manifest["limits"])
        self.limits.validate()
        _require(
            self.limits.rss_bytes <= self.supervision["rss_bytes"] <= 4 * GIB
            and self.limits.output_bytes
            <= self.supervision["artifact_bytes"]
            <= 4 * GIB,
            "runtime_limits_exceed_supervision",
        )
        self.directory = Path(manifest["directory"])
        _require(
            self.directory.is_absolute()
            and str(self.directory.resolve()) == str(self.directory)
            and self.execution["directory"] == str(self.directory)
            and not self.directory.exists(),
            "runtime_fresh_directory",
        )
        binding = self.execution["pipeline"]
        _require(
            type(binding) is dict
            and set(binding) == {"id", "clock"}
            and type(binding["id"]) is str
            and bool(binding["id"]),
            "runtime_pipeline_binding",
        )
        self.clock_reference = binding["clock"]
        self.anchor = _runtime_document(self.clock_reference)
        a = self.anchor
        _require(
            set(a)
            == {
                "schema",
                "pipeline_id",
                "clock",
                "boot_identity",
                "started_ns",
                "deadline_ns",
                "wall_s",
                "recorded_at",
            }
            and a.get("schema") == "solarlab.g2-reference-pipeline-clock.v1"
            and a.get("clock") == "time.monotonic_ns"
            and type(a.get("pipeline_id")) is str
            and a.get("pipeline_id") == binding["id"]
            and type(a.get("boot_identity")) is str
            and a.get("boot_identity") == _boot_identity(),
            "runtime_pipeline_clock_identity",
        )
        _require(
            _integer(a.get("started_ns"))
            and _integer(a.get("deadline_ns"), 1)
            and type(a.get("wall_s")) in (int, float)
            and type(a.get("recorded_at")) is str
            and bool(a["recorded_at"])
            and math.isfinite(a["wall_s"])
            and 0 < a["wall_s"] <= 43200
            and a["deadline_ns"] - a["started_ns"]
            == math.floor(Fraction(a["wall_s"]) * 10**9),
            "runtime_pipeline_clock_bounds",
        )
        self.started_ns = time.monotonic_ns()
        cleanup = sum(
            Fraction(self.supervision[k])
            for k in ("term_grace_s", "kill_wait_s", "receipt_reserve_s")
        )
        cleanup_ns = math.ceil(cleanup * 10**9)
        _require(
            a["started_ns"] <= self.started_ns < a["deadline_ns"] - cleanup_ns,
            "runtime_pipeline_expired",
        )
        _require(
            self.execution["wall_s"] <= a["wall_s"]
            and self.limits.wall_s <= self.execution["wall_s"],
            "runtime_stage_wall",
        )
        _require(
            math.ceil(Fraction(self.execution["wall_s"]) * 10**9)
            <= a["deadline_ns"] - self.started_ns,
            "runtime_launcher_exceeds_remaining_pipeline",
        )
        self.work_deadline_ns = min(
            a["deadline_ns"] - cleanup_ns,
            self.started_ns
            + math.floor(Fraction(self.execution["wall_s"]) * 10**9)
            - cleanup_ns,
            self.started_ns + math.floor(Fraction(self.limits.wall_s) * 10**9),
        )
        self.ledger = RuntimeLedger(
            sink=lambda row: print(
                json.dumps({"runtime_phase": row}, sort_keys=True), flush=True
            )
        )
        self.thread_receipt = None
        self.first_failure = None
        self.return_code = None

    def checkpoint(self):
        _require(
            time.monotonic_ns() < self.work_deadline_ns, "runtime_pipeline_wall_cap"
        )
        _require(_rss_bytes() <= self.limits.rss_bytes, "runtime_RSS_cap")
        _require(
            not Path(self.supervision["cancel_path"]).exists(), "runtime_cancelled"
        )

    def receipt(self, outcome):
        self.ledger.end_segment("failed" if self.first_failure else "completed")
        name = {
            "run": "RunResult.json",
            "read": "Readback.json",
            "evaluate": "Result.json",
        }[self.action]
        path = self.directory / name
        result_file = None
        if path.is_file():
            _require(path.stat().st_size <= 4 * 1048576, "runtime_result_metadata_cap")
            raw = path.read_bytes()
            result_file = {
                "path": str(path),
                "bytes": len(raw),
                "sha256": sha256(raw).hexdigest(),
            }
        manifest_file = None
        if self.directory.is_dir():
            # Bind the exact in-memory input even for a direct supervised API
            # call. This is evidence, not a new admission or a clock anchor.
            manifest_path = self.directory / "RuntimeManifest.json"
            _write_new(manifest_path, self.manifest, self.limits.metadata_reserve)
            raw = manifest_path.read_bytes()
            manifest_file = {
                "path": str(manifest_path),
                "bytes": len(raw),
                "sha256": sha256(raw).hexdigest(),
            }
        return {
            "schema": RUNTIME_SCHEMA,
            "stage": self.action,
            "case_id": self.case_id,
            "plan": self.manifest["plan"],
            "manifest_digest": digest(self.manifest),
            "manifest": manifest_file,
            "result_file": result_file,
            "pipeline_clock": self.clock_reference,
            "pipeline_id": self.anchor["pipeline_id"],
            "started_ns": self.started_ns,
            "ended_ns": time.monotonic_ns(),
            "original_deadline_ns": self.anchor["deadline_ns"],
            "work_deadline_ns": self.work_deadline_ns,
            "execution": self.execution,
            "outcome": outcome,
            "first_failure": self.first_failure,
            "return_code": self.return_code,
            "thread_receipt": self.thread_receipt,
            "peak_process_RSS_bytes": _rss_bytes(),
            "timing": self.ledger.snapshot(),
            "scientific_pass": False,
            "group_RSS_and_closure_authority": "outer g2_route_launcher receipt",
            "followup_stages_executed_implicitly": False,
        }


@contextmanager
def supervised_stage(manifest, plan, action):
    runtime = StageRuntime(manifest, plan, action)
    # Import and call the maintained route helper: no second flock, copied
    # dev/ino check, explicit unlock, or unsupervised fallback is introduced.
    with route_channel(plan, runtime.execution):
        primary = None
        try:
            runtime.checkpoint()
            runtime.thread_receipt = runtime_threads()
            print(
                json.dumps(
                    {
                        "runtime_start": {
                            "stage": action,
                            "case_id": runtime.case_id,
                            "pipeline_clock": runtime.clock_reference,
                            "original_deadline_ns": runtime.anchor["deadline_ns"],
                            "work_deadline_ns": runtime.work_deadline_ns,
                            "threads": runtime.thread_receipt,
                            "inherited_channel_fd": os.environ.get(
                                "SOLARLAB_G2_ROUTE_CHANNEL_FD"
                            ),
                            "channel_validator": "scripts.benchmarks.g2_route._channel",
                            "process_RSS_bytes": _rss_bytes(),
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            yield runtime
            runtime.checkpoint()
        except BaseException as exc:
            primary = exc
            runtime.first_failure = {
                "type": type(exc).__name__,
                "reason": str(exc)[:4096],
            }
            raise
        finally:
            final_error = None
            if runtime.thread_receipt is not None:
                try:
                    _runtime_thread_end(runtime.thread_receipt)
                except BaseException as exc:
                    final_error = exc
                    runtime.thread_receipt["final_measurement_error"] = {
                        "type": type(exc).__name__,
                        "reason": str(exc)[:4096],
                    }
                    if primary is not None:
                        primary.add_note("final BLAS measurement: " + str(exc))
                    else:
                        runtime.first_failure = {
                            "type": type(exc).__name__,
                            "reason": str(exc)[:4096],
                        }
                runtime.thread_receipt["final_measurement_status"] = (
                    "measured"
                    if "final_measured_mode" in runtime.thread_receipt
                    and "final_checked_ns" in runtime.thread_receipt
                    else "unavailable"
                )
            try:
                receipt = runtime.receipt(
                    "raised"
                    if primary is not None or final_error is not None
                    else "returned"
                )
                if runtime.directory.is_dir():
                    used = sum(
                        p.stat().st_size
                        for p in runtime.directory.rglob("*")
                        if p.is_file()
                    )
                    _write_new(
                        runtime.directory / "RuntimeReceipt.json",
                        receipt,
                        runtime.limits.output_bytes - used,
                    )
                # The outer launcher captures pre-output-directory failures too.
                print(
                    json.dumps(
                        {"runtime_receipt": receipt}, sort_keys=True, allow_nan=False
                    ),
                    flush=True,
                )
            except BaseException as closeout:
                original = primary if primary is not None else final_error
                if original is None:
                    raise
                original.add_note("runtime receipt closeout: " + str(closeout))
            if primary is None and final_error is not None:
                raise final_error


def _check_newton(
    row: Mapping, left: Point, right: Point, ratios: tuple[Fraction, ...]
) -> None:
    """Validate required diagnostics, iterate chain and accounting, not E3 bounds."""
    iterations, summary = row.get("newton_iterations"), row.get("newton_summary")
    _require(
        isinstance(iterations, list)
        and len(iterations) <= STEP_OPTIONS["max_iterations"]
        and isinstance(summary, dict)
        and set(summary)
        == {
            "globalization",
            "initial_trial_identity",
            "initial_trial_accepted",
            "line_search_evaluations",
            "factorization_returns",
            "linear_solve_returns",
        },
        "newton_diagnostics",
    )
    initial = _coordinate_trial(
        left,
        PrimitiveExpansion.from_value(np.zeros(left.state.layout.size)),
        right.time,
        right.inputs,
    )
    _require(
        summary["globalization"] == "residual"
        and summary["initial_trial_identity"] == initial.identity
        and summary["initial_trial_accepted"] is (not iterations),
        "newton_diagnostics_initial_trial",
    )
    previous, previous_merit, searches = initial.identity, None, 0
    required = {
        "iteration",
        "rhs_projection_error",
        "linear_residual_inf",
        "damping",
        "merit_before",
        "merit_after",
        "point_identity_before",
        "point_identity_after",
        "correction_base",
        "correction_searches",
    }
    dampings = [2.0 ** (-i) for i in range(STEP_OPTIONS["max_line_search"])]
    for index, item in enumerate(iterations):
        _require(
            isinstance(item, dict) and set(item) == required,
            "newton_diagnostics_fields",
        )
        projection = item["rhs_projection_error"]
        _require(
            isinstance(projection, list) and len(projection) == left.state.layout.size,
            "newton_diagnostics_projection",
        )
        for value in projection:
            _diagnostic_rational(value)
        linear_error, damping = item["linear_residual_inf"], item["damping"]
        _require(
            type(linear_error) in (float, int)
            and math.isfinite(linear_error)
            and linear_error >= 0
            and type(damping) in (float, int)
            and damping in dampings,
            "newton_diagnostics_linear_or_damping",
        )
        before, after = (
            _diagnostic_rational(item["merit_before"]),
            _diagnostic_rational(item["merit_after"]),
        )
        _require(
            type(item["iteration"]) is int
            and item["iteration"] == index
            and item["point_identity_before"] == previous
            and _is_digest(item["point_identity_after"])
            and before > 1
            and 0 <= after < before
            and (previous_merit is None or before == previous_merit)
            and item["correction_base"] is None
            and item["correction_searches"] == [],
            "newton_diagnostics_chain_or_merit",
        )
        previous, previous_merit = item["point_identity_after"], after
        searches += dampings.index(damping) + 1
    counts, n = row["solver_counts"], len(iterations)
    _require(
        previous == right.identity
        and (previous_merit is None or previous_merit == max(ratios))
        and all(_integer(v) for v in counts.values())
        and counts["factorizations"] == counts["linear_solves"] == n
        and 1 + n <= counts["residual_evaluations"] <= 1 + searches
        and all(
            _integer(summary[k])
            for k in (
                "line_search_evaluations",
                "factorization_returns",
                "linear_solve_returns",
            )
        )
        and summary["factorization_returns"] == summary["linear_solve_returns"] == n
        and summary["line_search_evaluations"] == searches,
        "newton_diagnostics_counts_or_terminal",
    )


class Guard:
    def __init__(
        self,
        limits: Limits,
        checkpoint: Callable[[], None] | None = None,
        started_monotonic: float | None = None,
    ):
        limits.validate()
        self.limits = limits
        self.start = (
            time.monotonic() if started_monotonic is None else started_monotonic
        )
        self.external = checkpoint

    def __call__(self) -> None:
        if self.external is not None:
            self.external()
        _require(time.monotonic() - self.start < self.limits.wall_s, "wall_cap")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak *= 1024
        _require(peak <= self.limits.rss_bytes, "RSS_cap")


def _write_new(path: Path, value: dict, limit: int) -> None:
    raw = canonical(value) + b"\n"
    _require(len(raw) <= limit, "metadata_cap")
    with path.open("xb") as stream:
        stream.write(raw)


def _replace_owned(path: Path, value: dict, limit: int) -> None:
    raw = canonical(value) + b"\n"
    _require(len(raw) <= limit, "metadata_cap")
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
    temporary.replace(path)


def _sum_enclosures(current: list, added: list) -> list:
    _require(len(current) == len(added), "flux_channel_count")
    result = []
    for a, b in zip(current, added):
        ca, ra, cb, rb = (
            _unpair(a["center"]),
            _unpair(a["radius"]),
            _unpair(b["center"]),
            _unpair(b["radius"]),
        )
        _require(ra >= 0 and rb >= 0, "negative_radius")
        result.append({"center": _pair(ca + cb), "radius": _pair(ra + rb)})
    return result


def _failure_value(value: Any, reference: dict) -> Any:
    """Keep failed working-iterate diagnostics distinct from accepted steps."""
    if isinstance(value, Point):
        return pack_point(value, reference)
    if isinstance(value, Fraction):
        return {"rational": _pair(value)}
    if isinstance(value, np.ndarray):
        return _failure_value(value.tolist(), reference)
    if isinstance(value, np.generic):
        return _failure_value(value.item(), reference)
    if is_dataclass(value):
        return {
            field.name: _failure_value(getattr(value, field.name), reference)
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _failure_value(item, reference) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_failure_value(item, reference) for item in value]
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or type(value) in (str, bool, int, float):
        return value
    return {"unsupported_diagnostic_type": type(value).__name__}


def run_reference(
    backend: Backend,
    schedule: Schedule,
    identity: RunIdentity,
    input_payload: dict,
    directory: Path,
    limits: Limits,
    *,
    encoding: str = "raw",
    checkpoint: Callable[[], None] | None = None,
    started_monotonic: float | None = None,
    timing: RuntimeLedger | None = None,
) -> dict:
    """Run one declared case; callers own admission and external supervision."""

    def measured(name, operation, *args, **kwargs):
        with timing.span(name) if timing is not None else nullcontext():
            return operation(*args, **kwargs)

    guard = Guard(limits, checkpoint, started_monotonic)
    _require(identity.input_sha256 == digest(input_payload), "input_digest")
    _require(
        identity.schedule_identity == schedule.identity
        and identity.source_identity == backend.problem.source_identity
        and identity.initial_point_identity == backend.initial.identity
        and identity.controls_sha256 == digest(effective_controls(backend)),
        "execution_binding",
    )
    _require(
        compile_schedule(
            backend.segments,
            schedule.tags,
            level=schedule.level,
            seed=schedule.seed,
            step_cap=limits.step_cap,
        )
        == schedule,
        "runtime_schedule_binding",
    )
    _require(
        digest(input_payload.get("segments", [])[: len(backend.segments)])
        == digest([asdict(s) for s in backend.segments]),
        "request_protocol_binding",
    )
    _require(len(schedule.steps) <= limits.step_cap, "step_cap")
    _require(
        type(backend.coordinates) is RelativeCoordinates
        and all(x == "linear" for x in backend.coordinates.modes.values()),
        "physical_coordinates",
    )
    _require(
        backend.coordinates.layout.identity == backend.problem.layout.identity
        and backend.initial.time == backend.segments[0].start,
        "initial_binding",
    )
    guard()
    _require(backend.feasible(backend.initial), "initial_domain")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    history_dir = directory / "History"
    history_dir.mkdir()
    extension = ".jsonl.gz" if encoding == "gzip" else ".jsonl"
    path = history_dir / ("BE" + extension)
    measured(
        "artifact_io",
        _write_new,
        directory / "ReferenceRequest.json",
        input_payload,
        limits.metadata_reserve // 4,
    )
    writer = HistoryWriter(
        path,
        encoding=encoding,
        total_output_bytes=limits.output_bytes - limits.metadata_reserve,
        metadata_reserve_bytes=limits.metadata_reserve // 4,
        max_record_bytes=limits.record_bytes,
    )
    original_reference = encode_point(_point_root(backend.initial))
    binding = identity.payload()
    source_identity = backend.problem.source_identity
    previous_hash: str | None = None
    counts = {
        "accepted_steps": 0,
        "residual_evaluations": 0,
        "observations": 0,
        "segments": 0,
        "factorizations": 0,
        "linear_solves": 0,
    }
    first_scientific = None
    last_point = backend.initial
    last_safe = backend.initial
    last_step = None
    last_returned_index = None
    cumulative = [{"center": [0, 1], "radius": [0, 1]} for _ in range(3)]
    error: BaseException | None = None
    failure_pointer = None
    completed = False
    target = None
    phase = "initial_state_check"

    def _emit(kind: str, **payload: Any) -> dict:
        nonlocal previous_hash
        row = {
            "schema": SCHEMA,
            "version": 1,
            "kind": kind,
            "case_id": identity.case_id,
            "source_identity": source_identity,
            "method": METHOD,
            "previous_record_sha256": previous_hash,
            **payload,
        }
        row["record_sha256"] = digest(row)
        _require(
            writer.logical_bytes + len(canonical(row)) + 1 <= limits.logical_bytes,
            "logical_cap",
        )
        writer.append(row)
        previous_hash = row["record_sha256"]
        return row

    def emit(kind: str, **payload: Any) -> dict:
        return measured("history_io", _emit, kind, **payload)

    def state_checks(point: Point) -> dict:
        nonlocal first_scientific, last_safe
        checks = measured("state_check", backend.check_state, point)
        _require(
            isinstance(checks, dict)
            and type(checks.get("safety_pass")) is bool
            and type(checks.get("scientific_failed")) is bool,
            "state_check_contract",
        )
        if checks.get("scientific_failed") is True and first_scientific is None:
            first_scientific = {
                "point_identity": point.identity,
                "time_hex": float(point.time).hex(),
                "checks": checks,
            }
        if checks["safety_pass"]:
            last_safe = point
        return checks

    def observe(point: Point, segment: Any, tag_index: int) -> None:
        nonlocal phase
        phase = "point_observation"
        guard()
        observation = measured(
            "point_observation", backend.observe_point, point, segment
        )
        _require(
            isinstance(observation, dict)
            and "ports" in observation
            and "charges" in observation
            and "derivative_origin" in observation,
            "point_observation_contract",
        )
        phase = "point_observation_write"
        emit(
            "be_observation",
            tag=schedule.tags[tag_index],
            point_identity=point.identity,
            physical_state_words=_physical(point),
            units={v.id: asdict(v.unit) for v in point.state.layout.variables},
            cumulative_physical_flux=cumulative,
            **observation,
        )
        counts["observations"] += 1
        guard()

    try:
        initial_checks = state_checks(last_point)
        phase = "header_write"
        emit(
            "be_reference_header",
            identity=binding,
            source_metadata=backend.metadata,
            schedule={
                "identity": schedule.identity,
                "level": schedule.level,
                "seed": _pair(schedule.seed),
                "tags": schedule.tags,
                "steps": len(schedule.steps),
                "segments": [asdict(s) for s in backend.segments],
            },
            original_reference=original_reference,
            initial=pack_point(backend.initial, original_reference),
            limits=asdict(limits),
            field_units={
                v.id: asdict(v.unit) for v in backend.problem.layout.variables
            },
            **{k: v for k, v in effective_controls(backend).items() if k != "method"},
            initial_state_checks=initial_checks,
            scientific_pass=False,
        )
        _require(initial_checks["safety_pass"], "initial_safety")
        active_segment = -1
        for target in schedule.steps:
            segment = backend.segments[target.segment_index]
            if target.segment_index != active_segment:
                if timing is not None:
                    timing.begin_segment(segment.id)
                phase = "segment_start"
                inputs, input_rate = segment.inputs(segment.start)
                expected_inputs, expected_rate, _ = _protocol_at(segment, segment.start)
                _require(
                    last_point.time == segment.start
                    and _input_words(last_point.inputs)
                    == _input_words(inputs)
                    == _input_words(expected_inputs)
                    and _input_words(input_rate) == _input_words(expected_rate),
                    "event_state_or_input_jump",
                )
                active_segment = target.segment_index
                counts["segments"] += 1
                initial_tag = next(
                    i
                    for i, t in enumerate(schedule.tags)
                    if t["segment"] == segment.id and t["index"] == 0
                )
                emit(
                    "be_segment_start",
                    segment=segment.id,
                    generation=active_segment + 1,
                    point=pack_point(last_point, original_reference),
                    input_rate_hex=[float(v).hex() for v in input_rate],
                    physical_state_reset=False,
                    native_initialization=False,
                )
                observe(last_point, segment, initial_tag)
            phase = "step_guard"
            guard()
            _require(last_point.time.hex() == target.left.hex(), "time_history_gap")
            inputs, input_rate = segment.inputs(target.right)
            expected_inputs, expected_rate, input_side = _protocol_at(
                segment, target.right
            )
            _require(
                _input_words(inputs) == _input_words(expected_inputs)
                and _input_words(input_rate) == _input_words(expected_rate),
                "step_protocol_inputs",
            )

            def feasible(point: Point) -> bool:
                guard()
                return backend.feasible(point)

            phase = "be_step"
            step = measured(
                "be_step",
                conservative_be_step,
                backend.problem,
                backend.coordinates,
                last_point,
                target.h,
                scaling=backend.scaling,
                budgets=backend.budgets,
                inputs=inputs,
                feasible=feasible,
                **STEP_OPTIONS,
            )
            phase = "step_return_check"
            _require(
                step.left.identity == last_point.identity
                and step.right.time.hex() == target.right.hex(),
                "step_return_binding",
            )
            last_step = step
            last_returned_index = target.index
            phase = "accepted_state_check"
            checks = state_checks(step.right)
            step_payload = dict(
                step_index=target.index,
                segment=segment.id,
                segment_step=target.segment_step,
                generation=active_segment + 1,
                window_index=target.window_index,
                t_left_hex=target.left.hex(),
                t_right_hex=target.right.hex(),
                h_hex=target.h.hex(),
                input_rate_hex=_input_words(input_rate),
                input_event_side=input_side,
                order=1,
                left_identity=step.left.identity,
                right=pack_point(step.right, original_reference),
                local_increment_words=_words(step.local_coordinates),
                storage_delta_words=_words(step.storage_delta),
                storage_secant_words=_words(step.storage_secant),
                residual_words=_words(step.residual),
                residual_units=[asdict(v) for v in step.residual_units],
                residual_allowances=[_pair(v) for v in step.residual_allowances],
                residual_ratios=[_pair(v) for v in step.residual_ratios],
                solver_counts={
                    "residual_evaluations": step.residual_evaluations,
                    "factorizations": step.factorization_calls,
                    "linear_solves": step.linear_solve_calls,
                },
                newton_iterations=_failure_value(step.iterations, original_reference),
                newton_summary={
                    "globalization": step.globalization,
                    "initial_trial_identity": measured(
                        "step_validation",
                        _coordinate_trial,
                        step.left,
                        PrimitiveExpansion.from_value(
                            np.zeros(backend.problem.layout.size)
                        ),
                        step.right.time,
                        step.right.inputs,
                    ).identity,
                    "initial_trial_accepted": not step.iterations,
                    "line_search_evaluations": step.line_search_evaluations,
                    "factorization_returns": step.factorization_returns,
                    "linear_solve_returns": step.linear_solve_returns,
                },
                state_checks=checks,
                scientific_failure_latched=first_scientific is not None,
                physical_acceptance_awarded=False,
            )
            phase = "accepted_step_validation"
            measured(
                "step_validation",
                _check_step_protocol,
                step_payload,
                step.left,
                step.right,
                segment,
            )
            measured(
                "step_validation",
                _check_step_arithmetic,
                step_payload,
                step.left,
                step.right,
                backend.problem.storage_count,
                _secant_arithmetic(backend.problem),
            )
            measured(
                "step_validation",
                _check_newton,
                step_payload,
                step.left,
                step.right,
                step.residual_ratios,
            )
            phase = "accepted_step_write"
            row = emit("be_accepted_step", **step_payload)
            counts["accepted_steps"] += 1
            counts["residual_evaluations"] += step.residual_evaluations
            counts["factorizations"] += step.factorization_calls
            counts["linear_solves"] += step.linear_solve_calls
            if timing is not None:
                timing.accepted(step)
            last_point = step.right
            phase = "last_accepted_pointer"
            measured(
                "artifact_io",
                _replace_owned,
                history_dir / "LastNumericalAccepted.json",
                {
                    "record_sha256": row["record_sha256"],
                    "point": row["right"],
                    "step_index": target.index,
                    "physical_acceptance_awarded": False,
                },
                limits.metadata_reserve // 8,
            )
            phase = "accepted_state_safety"
            _require(checks["safety_pass"], "accepted_state_safety")
            _require(
                counts["residual_evaluations"] <= limits.residual_cap, "residual_cap"
            )
            guard()
            phase = "interval_observation"
            interval = measured(
                "interval_observation",
                backend.observe_interval,
                step.left,
                step.right,
                step,
                segment,
                target,
            )
            _require(
                isinstance(interval, dict)
                and len(interval.get("physical_flux", [])) == 3,
                "interval_observation_contract",
            )
            cumulative = _sum_enclosures(cumulative, interval["physical_flux"])
            phase = "interval_record_write"
            emit(
                "be_observation",
                tag=None,
                point_identity=step.right.identity,
                interval_step=target.index,
                window_index=target.window_index,
                cumulative_physical_flux=cumulative,
                interval_observation=interval,
                derivative_origin="declared-physical-path",
                evidence_status="represented_path_only",
            )
            if target.tag_index is not None:
                observe(last_point, segment, target.tag_index)
            guard()
        phase = "final_checks"
        if timing is not None:
            timing.end_segment("completed")
        _require(counts["observations"] == len(schedule.tags), "incomplete_tags")
        phase = "final_record_write"
        emit(
            "be_reference_final",
            counts=counts,
            last_point_identity=last_point.identity,
            first_scientific_failure=first_scientific,
            numerical_complete=True,
            scientific_pass=False,
            scientific_status="failed"
            if first_scientific
            else "unknown_reference_error",
        )
        phase = "history_close"
        measured("history_io", writer.finish)
        completed = True
    except BaseException as exc:
        error = exc
        if timing is not None:
            timing.end_segment("failed")
        failure = {
            "exception": type(exc).__name__,
            "reason": str(exc)[:4096],
            "location": {
                "phase": phase,
                "target_step": None
                if target is None
                else {
                    "index": target.index,
                    "segment": backend.segments[target.segment_index].id,
                    "t_left_hex": target.left.hex(),
                    "t_right_hex": target.right.hex(),
                },
                "last_persisted_record_index": writer.records,
                "previous_record_sha256": previous_hash,
                "last_numerical_accepted": {
                    "identity": last_point.identity,
                    "time_hex": last_point.time.hex(),
                },
            },
            "last_point": pack_point(last_point, original_reference),
            "last_safe_point": pack_point(last_safe, original_reference),
            "counts": dict(counts),
            "first_scientific_failure": first_scientific,
            "failed_newton_diagnostics_available": hasattr(
                exc, "conservative_diagnostics"
            ),
        }
        diagnostics = getattr(exc, "conservative_diagnostics", None)
        if diagnostics is not None:
            failure["failed_newton_working_iterates"] = _failure_value(
                diagnostics, original_reference
            )
            failure["failed_step_residual_evaluations"] = getattr(
                diagnostics, "residual_evaluations", None
            )
        if last_step is not None:
            failure["last_returned_step"] = last_returned_index
            failure["last_returned_numerical_point"] = pack_point(
                last_step.right, original_reference
            )
            failure["last_returned_step_persisted"] = (
                last_point.identity == last_step.right.identity
            )
            failure["last_returned_solver_counts"] = {
                "residual_evaluations": last_step.residual_evaluations,
                "factorizations": last_step.factorization_calls,
                "linear_solves": last_step.linear_solve_calls,
            }
        failure_pointer = {
            "type": failure["exception"],
            "reason": failure["reason"],
            "location": failure["location"],
            "failure_sha256": digest(failure),
            "record_index": None,
            "record_sha256": None,
        }
        try:
            if not writer.closed:
                failure_row = emit(
                    "be_reference_failure",
                    failure=failure,
                    numerical_complete=False,
                    scientific_pass=False,
                )
                failure_pointer.update(
                    record_index=writer.records,
                    record_sha256=failure_row["record_sha256"],
                )
                measured("history_io", writer.finish)
        except BaseException as closeout:
            exc.add_note("reference history closeout: " + type(closeout).__name__)
            if not writer.closed:
                measured("history_io", writer.abort, exc)
        try:
            measured(
                "artifact_io",
                _write_new,
                directory / "FirstFailure.json",
                {
                    **failure,
                    **{
                        k: failure_pointer[k]
                        for k in ("failure_sha256", "record_index", "record_sha256")
                    },
                },
                limits.metadata_reserve // 4,
            )
        except BaseException as closeout:
            exc.add_note("reference failure closeout: " + type(closeout).__name__)
    result = {
        "schema": SCHEMA,
        "case_id": identity.case_id,
        "method": METHOD,
        **binding,
        "source_identity": source_identity,
        "schedule_identity": schedule.identity,
        "status": "completed" if completed else "failed",
        "numerical_complete": completed,
        "scientific_pass": False,
        "scientific_status": "failed"
        if error or first_scientific
        else "unknown_reference_error",
        "counts": counts,
        "first_failure": failure_pointer,
        "first_scientific_failure": first_scientific,
        "last_numerical_accepted": {
            "identity": last_point.identity,
            "time_hex": last_point.time.hex(),
        },
        "observed_history": {
            "path": str(path.relative_to(directory)),
            "encoding": encoding,
            "records": writer.records,
            "logical_sha256": writer.logical_sha256,
            "logical_bytes": writer.logical_bytes,
            "encoded_bytes": writer.encoded_bytes,
            "closed": writer.closed,
            "writer_error": writer.error,
        },
        "elapsed_s": time.monotonic() - guard.start,
    }
    measured(
        "artifact_io",
        _write_new,
        directory / "RunResult.json",
        result,
        limits.metadata_reserve // 8,
    )
    _require(
        sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
        <= limits.output_bytes,
        "combined_output_cap",
    )
    if error is not None:
        error.add_note(
            "Reference result retained at " + str(directory / "RunResult.json")
        )
        raise error
    return result


def _strict_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_JSON_key")
        result[key] = value
    return result


def _read_json(path: Path, limit: int = 4 * 1048576) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit, "metadata_size")
    value = json.loads(raw, object_pairs_hook=_strict_object)
    _require(isinstance(value, dict), "metadata_object")
    canonical(value)  # Reject nonfinite JSON constants before using them.
    return value


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _check_enclosure(value: Any) -> None:
    _require(
        isinstance(value, dict) and set(value) == {"center", "radius"}, "enclosure"
    )
    _unpair(value["center"])
    _require(_unpair(value["radius"]) >= 0, "negative_radius")


def _word_values(value: Mapping, size: int) -> tuple[Fraction, ...]:
    _validate_words(value, size)
    return tuple(
        sum(
            (
                Fraction(_hex_time(row[i], nonnegative=False))
                for row in value["words_hex"]
            ),
            Fraction(),
        )
        for i in range(size)
    )


def read_reference_history(
    directory: Path,
    layout: Layout,
    expected_identity: RunIdentity,
    *,
    max_record_bytes: int = 1048576,
    max_logical_bytes: int = 32 * GIB,
    checkpoint: Callable[[], None] | None = None,
) -> dict:
    """Independently check BE sequence/codec/clock/seal, without a model solve."""
    directory = Path(directory)
    _require(
        _integer(max_record_bytes, 1)
        and max_record_bytes <= 1048576
        and _integer(max_logical_bytes, 1)
        and max_logical_bytes <= 32 * GIB,
        "reader_limits",
    )
    checkpoint = checkpoint or (lambda: None)
    checkpoint()
    result = _read_json(directory / "RunResult.json")
    _require(
        all(result.get(k) == v for k, v in expected_identity.payload().items()),
        "result_identity",
    )
    _require(
        result.get("schema") == SCHEMA
        and result.get("method") == METHOD
        and result.get("scientific_pass") is False
        and type(result.get("numerical_complete")) is bool,
        "result_schema",
    )
    request = _read_json(directory / "ReferenceRequest.json")
    _require(digest(request) == expected_identity.input_sha256, "saved_input_digest")
    meta = result["observed_history"]
    relative = Path(meta["path"])
    _require(not relative.is_absolute() and ".." not in relative.parts, "history_path")
    history_path = directory / relative
    _require(
        history_path.resolve().is_relative_to(directory.resolve()),
        "history_path_escape",
    )
    reader = HistoryReader(
        history_path,
        encoding=meta["encoding"],
        max_record_bytes=max_record_bytes,
        max_logical_bytes=max_logical_bytes,
    )
    previous_hash = None
    previous_point = None
    reference = None
    tags = []
    seen_tags = []
    steps = 0
    segment_steps = 0
    segment = None
    generation = 0
    expected_schedule = None
    next_interval = 1
    cumulative = [{"center": [0, 1], "radius": [0, 1]} for _ in range(3)]
    final = None
    failure = None
    first_scientific = None
    safety_stopped = False
    last_safe_point = None
    samples = []
    solver_totals = {"residual_evaluations": 0, "factorizations": 0, "linear_solves": 0}
    field_units = json.loads(
        canonical({v.id: asdict(v.unit) for v in layout.variables})
    )
    residual_units = []
    residual_allowances = []
    storage_size = sum(
        math.prod(e.shape) for e in layout.equations if e.role == "storage"
    )
    logical = sha256()
    records = 0

    def check_state(value: Any, point: Point) -> None:
        nonlocal first_scientific, safety_stopped, last_safe_point
        _require(
            isinstance(value, dict)
            and type(value.get("safety_pass")) is bool
            and type(value.get("scientific_failed")) is bool,
            "state_check_contract",
        )
        safety_stopped = not value["safety_pass"]
        if value["safety_pass"]:
            last_safe_point = point
        if value["scientific_failed"] and first_scientific is None:
            first_scientific = {
                "point_identity": point.identity,
                "time_hex": point.time.hex(),
                "checks": value,
            }

    try:
        for line in reader:
            checkpoint()
            _require(line.endswith(b"\n"), "partial_history_tail")
            logical.update(line)
            row = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ContractError("reference_be_nonfinite_JSON")
                ),
            )
            records += 1
            expected_hash = row["record_sha256"]
            content = {k: v for k, v in row.items() if k != "record_sha256"}
            _require(
                digest(content) == expected_hash
                and row["previous_record_sha256"] == previous_hash,
                "record_chain",
            )
            previous_hash = expected_hash
            _require(
                row["schema"] == SCHEMA
                and type(row["version"]) is int
                and row["version"] == 1
                and row["method"] == METHOD
                and row["case_id"] == expected_identity.case_id,
                "record_identity",
            )
            _require(final is None and failure is None, "records_after_terminal")
            kind = row["kind"]
            _require(
                not safety_stopped or kind == "be_reference_failure",
                "continued_after_safety_failure",
            )
            if kind == "be_reference_header":
                _require(
                    records == 1 and row["identity"] == expected_identity.payload(),
                    "header_identity",
                )
                reference = row["original_reference"]
                previous_point = unpack_point(row["initial"], reference, layout)
                # The producer retains the initial Point even if its separate
                # safety predicate fails before there is a safe accepted state.
                last_safe_point = previous_point
                _require(
                    previous_point.identity == expected_identity.initial_point_identity,
                    "initial_Point_identity",
                )
                _require(
                    row["field_units"] == field_units
                    and row["scientific_pass"] is False,
                    "header_physical_semantics",
                )
                controls = {
                    k: row[k]
                    for k in (
                        "method",
                        "step_options",
                        "storage_secant_arithmetic",
                        "equation_budgets",
                        "solver_scaling",
                        "observation_controls",
                    )
                }
                _require(
                    digest(controls) == expected_identity.controls_sha256
                    and row["step_options"] == STEP_OPTIONS,
                    "controls_identity",
                )
                secant_arithmetic = row["storage_secant_arithmetic"]
                _require(
                    secant_arithmetic in SECANT_ARITHMETICS,
                    "unsupported_secant_arithmetic",
                )
                _require(
                    set(row["solver_scaling"]) == {"rows", "columns"}, "solver_scaling"
                )
                for scale in row["solver_scaling"].values():
                    _require(
                        len(scale) == layout.size
                        and all(_hex_time(x) > 0 for x in scale),
                        "solver_scaling",
                    )
                check_state(row["initial_state_checks"], previous_point)
                _require(
                    set(row["equation_budgets"]) == {e.id for e in layout.equations},
                    "budget_keys",
                )
                for equation in layout.equations:
                    unit = (
                        equation.unit * SECOND
                        if equation.role == "storage"
                        else equation.unit
                    )
                    expected_unit = json.loads(canonical(asdict(unit)))
                    budget = row["equation_budgets"][equation.id]
                    size = math.prod(equation.shape)
                    _require(
                        budget["unit"] == expected_unit
                        and len(budget["absolute_hex"]) == size,
                        "budget_unit_or_shape",
                    )
                    amounts = [Fraction(_hex_time(v)) for v in budget["absolute_hex"]]
                    _require(all(v > 0 for v in amounts), "budget_nonpositive")
                    residual_units.extend([expected_unit] * size)
                    residual_allowances.extend(amounts)
                tags = row["schedule"]["tags"]
                _require(
                    row["schedule"]["identity"] == result["schedule_identity"],
                    "schedule_identity",
                )
                segments = tuple(
                    SimpleNamespace(**s) for s in row["schedule"]["segments"]
                )
                _require(
                    digest(row["schedule"]["segments"])
                    == digest(request.get("segments", [])[: len(segments)]),
                    "request_protocol_binding",
                )
                if "observation_times" in request:
                    _require(
                        digest(tags)
                        == digest(tags_from_request(request, len(segments))),
                        "original_request_tags",
                    )
                expected_schedule = compile_schedule(
                    segments,
                    tags,
                    level=row["schedule"]["level"],
                    seed=_unpair(row["schedule"]["seed"]),
                )
                _require(
                    expected_schedule.identity == result["schedule_identity"]
                    and row["schedule"]["steps"] == len(expected_schedule.steps),
                    "saved_schedule_identity",
                )
                first_inputs, _, _ = _protocol_at(segments[0], previous_point.time)
                _require(
                    _input_words(previous_point.inputs) == _input_words(first_inputs),
                    "initial_protocol_inputs",
                )
                source = row["source_identity"]
                _require(source == result["source_identity"], "source_identity")
            else:
                _require(
                    reference is not None and row["source_identity"] == source,
                    "missing_header_or_foreign_source",
                )
                if kind == "be_segment_start":
                    point = unpack_point(row["point"], reference, layout)
                    _require(
                        point.identity == previous_point.identity
                        and _integer(row["generation"], 1)
                        and row["generation"] == generation + 1
                        and generation < len(segments)
                        and row["segment"] == segments[generation].id
                        and point.time == segments[generation].start
                        and row["physical_state_reset"] is False
                        and row["native_initialization"] is False,
                        "event_history",
                    )
                    spec = segments[generation]
                    first = np.array([spec.voltage[0], spec.photons[0]], dtype=float)
                    last = np.array([spec.voltage[1], spec.photons[1]], dtype=float)
                    rate = (last - first) / (spec.end - spec.start)
                    _require(
                        np.array_equal(point.inputs, first)
                        and row["input_rate_hex"] == [float(v).hex() for v in rate],
                        "event_input_rate",
                    )
                    segment, generation, segment_steps = (
                        row["segment"],
                        row["generation"],
                        0,
                    )
                elif kind == "be_accepted_step":
                    _require(
                        steps < len(expected_schedule.steps)
                        and next_interval == steps + 1,
                        "missing_interval_observation",
                    )
                    scheduled = expected_schedule.steps[steps]
                    right = unpack_point(row["right"], reference, layout)
                    h = _hex_time(row["h_hex"])
                    _require(
                        all(
                            _integer(row[k])
                            for k in (
                                "order",
                                "step_index",
                                "segment_step",
                                "generation",
                                "window_index",
                            )
                        )
                        and row["order"] == 1
                        and row["step_index"] == steps + 1
                        and row["segment_step"] == segment_steps + 1
                        and row["segment"] == segment
                        and row["generation"] == generation
                        and generation == scheduled.segment_index + 1,
                        "step_sequence",
                    )
                    left_t, right_t = (
                        _hex_time(row["t_left_hex"]),
                        _hex_time(row["t_right_hex"]),
                    )
                    _require(
                        row["left_identity"] == previous_point.identity
                        and previous_point.time == left_t
                        and right.time == right_t
                        and h > 0
                        and Fraction(right_t) - Fraction(left_t) == Fraction(h)
                        and left_t + h == right_t,
                        "step_clock_or_state",
                    )
                    _require(
                        (left_t, right_t, h, row["window_index"])
                        == (
                            scheduled.left,
                            scheduled.right,
                            scheduled.h,
                            scheduled.window_index,
                        ),
                        "step_not_on_frozen_schedule",
                    )
                    _check_step_protocol(
                        row, previous_point, right, segments[scheduled.segment_index]
                    )
                    _check_step_arithmetic(
                        row, previous_point, right, storage_size, secant_arithmetic
                    )
                    residual = _word_values(row["residual_words"], len(residual_units))
                    ratios = tuple(_unpair(x) for x in row["residual_ratios"])
                    _require(
                        row["residual_units"] == residual_units
                        and [_unpair(x) for x in row["residual_allowances"]]
                        == residual_allowances
                        and ratios
                        == tuple(
                            abs(x) / a for x, a in zip(residual, residual_allowances)
                        )
                        and all(x <= 1 for x in ratios),
                        "unaccepted_equation_residual",
                    )
                    _require(
                        set(row["solver_counts"]) == set(solver_totals), "solver_counts"
                    )
                    _check_newton(row, previous_point, right, ratios)
                    for k in solver_totals:
                        _require(_integer(row["solver_counts"][k]), "solver_counts")
                        solver_totals[k] += row["solver_counts"][k]
                    check_state(row["state_checks"], right)
                    _require(
                        row["scientific_failure_latched"]
                        is (first_scientific is not None)
                        and row["physical_acceptance_awarded"] is False,
                        "scientific_latch",
                    )
                    previous_point = right
                    steps += 1
                    segment_steps += 1
                elif kind == "be_observation":
                    _require(
                        row["point_identity"] == previous_point.identity,
                        "observation_point",
                    )
                    if row["tag"] is not None:
                        _require(
                            next_interval == steps + 1,
                            "tag_before_interval_observation",
                        )
                        _require(
                            len(seen_tags) < len(tags)
                            and digest(row["tag"]) == digest(tags[len(seen_tags)])
                            and row["tag"]["time_hex"] == previous_point.time.hex()
                            and row["tag"]["segment"] == segment,
                            "observation_tag",
                        )
                        _require(
                            row["physical_state_words"] == _physical(previous_point),
                            "observation_physical_words",
                        )
                        _require(
                            row["units"] == field_units
                            and row["ports"]["unit"] == "A"
                            and row["charges"]["unit"] == "C"
                            and row["derivative_origin"]
                            in {
                                "same_state_affine_tangent",
                                "manufactured_analytic_tangent",
                            },
                            "observation_semantics",
                        )
                        for k in ("left", "right"):
                            _check_enclosure(row["ports"][k])
                        for k in ("device", "left_metal", "right_metal"):
                            _check_enclosure(row["charges"][k])
                        _require(
                            row["cumulative_physical_flux"] == cumulative,
                            "sample_flux_prefix",
                        )
                        seen_tags.append(row["tag"])
                        samples.append(row)
                    else:
                        _require(
                            row["interval_step"] == steps == next_interval
                            and type(row["interval_step"]) is int
                            and row["window_index"]
                            == expected_schedule.steps[steps - 1].window_index
                            and row["interval_observation"]["unit"] == "C"
                            and row["interval_observation"][
                                "storage_secant_used_as_flux"
                            ]
                            is False,
                            "interval_observation_sequence",
                        )
                        cumulative = _sum_enclosures(
                            cumulative, row["interval_observation"]["physical_flux"]
                        )
                        _require(
                            row["cumulative_physical_flux"] == cumulative,
                            "interval_flux_prefix",
                        )
                        next_interval += 1
                elif kind == "be_reference_final":
                    _require(
                        row["numerical_complete"] is True
                        and row["scientific_pass"] is False
                        and row["last_point_identity"] == previous_point.identity
                        and row["counts"]["accepted_steps"] == steps
                        and steps == len(expected_schedule.steps)
                        and next_interval == steps + 1,
                        "final_record",
                    )
                    final = row
                elif kind == "be_reference_failure":
                    _require(
                        row["numerical_complete"] is False
                        and row["scientific_pass"] is False,
                        "failure_record",
                    )
                    failure = row
                    body = row["failure"]
                    location = body.get("location")
                    _require(
                        isinstance(location, dict)
                        and set(location)
                        == {
                            "phase",
                            "target_step",
                            "last_persisted_record_index",
                            "previous_record_sha256",
                            "last_numerical_accepted",
                        }
                        and type(location["phase"]) is str
                        and bool(location["phase"])
                        and _integer(location["last_persisted_record_index"])
                        and location["last_persisted_record_index"] == records - 1
                        and location["previous_record_sha256"]
                        == row["previous_record_sha256"]
                        and location["last_numerical_accepted"]
                        == {
                            "identity": previous_point.identity,
                            "time_hex": previous_point.time.hex(),
                        },
                        "failure_location_binding",
                    )
                    target = location["target_step"]
                    if target is not None:
                        _require(
                            isinstance(target, dict)
                            and set(target)
                            == {"index", "segment", "t_left_hex", "t_right_hex"}
                            and _integer(target["index"], 1)
                            and target["index"] in {steps, steps + 1}
                            and target["index"] <= len(expected_schedule.steps),
                            "failure_target_binding",
                        )
                        failed_target = expected_schedule.steps[target["index"] - 1]
                        _require(
                            target
                            == {
                                "index": failed_target.index,
                                "segment": segments[failed_target.segment_index].id,
                                "t_left_hex": failed_target.left.hex(),
                                "t_right_hex": failed_target.right.hex(),
                            },
                            "failure_target_binding",
                        )
                    else:
                        _require(steps == generation == 0, "failure_target_binding")
                    _require(
                        type(body.get("exception")) is str
                        and bool(body["exception"])
                        and type(body.get("reason")) is str
                        and body["counts"] == result["counts"]
                        and body["first_scientific_failure"] == first_scientific
                        and unpack_point(body["last_point"], reference, layout).identity
                        == previous_point.identity
                        and unpack_point(
                            body["last_safe_point"], reference, layout
                        ).identity
                        == last_safe_point.identity,
                        "failure_state_binding",
                    )
                    pointer = {
                        "type": body["exception"],
                        "reason": body["reason"],
                        "location": location,
                        "failure_sha256": digest(body),
                        "record_index": records,
                        "record_sha256": row["record_sha256"],
                    }
                    _require(
                        result.get("first_failure") == pointer, "failure_result_binding"
                    )
                    side_file = directory / "FirstFailure.json"
                    _require(side_file.is_file(), "failure_side_file_binding")
                    _require(
                        _read_json(side_file)
                        == {
                            **body,
                            **{
                                k: pointer[k]
                                for k in (
                                    "failure_sha256",
                                    "record_index",
                                    "record_sha256",
                                )
                            },
                        },
                        "failure_side_file_binding",
                    )
                else:
                    raise ContractError("reference_be_unknown_record_kind")
    finally:
        reader.close()
    _require(
        reader.eof_seen
        and not reader.partial_tail
        and records == meta["records"]
        and reader.logical_bytes == meta["logical_bytes"]
        and logical.hexdigest() == meta["logical_sha256"]
        and history_path.stat().st_size == meta["encoded_bytes"]
        and meta["closed"] is True
        and records > 0
        and expected_schedule is not None
        and meta.get("writer_error") is None,
        "history_seal",
    )
    complete = result["numerical_complete"] is True
    _require(
        (final is not None) == complete
        and (not complete or len(seen_tags) == len(tags))
        and steps == result["counts"]["accepted_steps"],
        "result_completeness",
    )
    _require(
        all(_integer(v) for v in result["counts"].values())
        and result["counts"]["observations"] == len(samples)
        and result["counts"]["segments"] == generation
        and all(result["counts"][k] == v for k, v in solver_totals.items()),
        "result_counts",
    )
    _require(
        result["first_scientific_failure"] == first_scientific
        and result["last_numerical_accepted"]
        == {"identity": previous_point.identity, "time_hex": previous_point.time.hex()},
        "result_last_state_or_latch",
    )
    if complete:
        _require(
            not safety_stopped
            and final["counts"] == result["counts"]
            and final["first_scientific_failure"] == first_scientific
            and result["first_failure"] is None
            and not (directory / "FirstFailure.json").exists()
            and result["status"] == "completed"
            and result["scientific_status"]
            == ("failed" if first_scientific else "unknown_reference_error"),
            "completed_result_semantics",
        )
    else:
        _require(
            failure is not None
            and result["first_failure"] is not None
            and result["status"] == "failed"
            and result["scientific_status"] == "failed",
            "failed_result_semantics",
        )
    checkpoint()
    return {
        "schema": SCHEMA,
        "case_id": expected_identity.case_id,
        "method": METHOD,
        "structural_pass": True,
        "numerical_complete": complete,
        "scientific_pass": False,
        "scientific_status": result["scientific_status"],
        "step_consistency_checked": {
            "protocol_inputs_and_rates": True,
            "full_word_increments": True,
            "declared_secant_division": True,
            "newton_diagnostics": True,
        },
        "E3_coefficients_status": "not_evaluated",
        "records": records,
        "accepted_steps": steps,
        "tagged_observations": len(seen_tags),
        "observed_history": dict(meta),
        "observations": samples,
    }


def order_check(d01: tuple[Fraction, Fraction], d12: tuple[Fraction, Fraction]) -> dict:
    """Check complete interval evidence; 20% scalar check is not independent."""
    a, ar, b, br = map(_fraction, (*d01, *d12))
    _require(ar >= 0 and br >= 0, "negative_radius")
    al, ah, bl, bh = abs(a) - ar, abs(a) + ar, abs(b) - br, abs(b) + br
    if al <= 0 or bl <= 0:
        return {"status": "unknown", "reason": "unresolved_difference"}
    if a * b <= 0:
        return {"status": "failed", "reason": "oscillatory_difference"}
    lower, upper = al / bh, ah / bl
    status = (
        "passed"
        if lower**5 >= 16 and upper**5 <= 64
        else ("failed" if upper**5 < 16 or lower**5 > 64 else "unknown")
    )
    return {
        "status": status,
        "ratio_lower": _pair(lower),
        "ratio_upper": _pair(upper),
        "scalar_consistency_is_independent": False,
    }


def propagate_error(initial: Fraction, rows: list[Mapping]) -> list[Fraction]:
    """Evaluate a supplied, source-reviewed dimensionless recurrence, no oracle."""
    value = _fraction(initial)
    _require(value >= 0, "negative_initial_error")
    result = []
    for row in rows:
        _require(
            set(row) >= {"G", "R", "U", "evidence_identity"}
            and row["evidence_identity"],
            "missing_propagation_evidence",
        )
        g, r, u = (_fraction(row[k]) for k in ("G", "R", "U"))
        _require(min(g, r, u) >= 0, "negative_propagation_component")
        value = g * value + r + u
        result.append(value)
    return result


def _enclosure(value: Any) -> dict:
    return {"center": _pair(value.center), "radius": _pair(value.radius)}


class SlabObserver:
    """Use the existing full-word physical observation, without native history."""

    def __init__(
        self,
        model: Any,
        absolute_error_C: Fraction,
        checkpoint: Callable[[], None] | None = None,
    ):
        from scripts.benchmarks.interval_observation import BallIntegrator

        self.model = model
        self.absolute_error = _fraction(absolute_error_C)
        _require(self.absolute_error > 0, "observation_budget")
        self.arithmetic = BallIntegrator(bits=512, evaluations=8192, depth=32)
        self.checkpoint = checkpoint or (lambda: None)

    def path(self, left: Point, right: Point, segment: Any, ordinal: int):
        from scripts.benchmarks.interval_observation import (
            AcceptedClock,
            Polynomial,
            PolynomialPath,
            SlabPathObserver,
        )

        lo, hi = _physical_exact(left), _physical_exact(right)
        h = Fraction(right.time) - Fraction(left.time)
        _require(h > 0, "path_time")
        clock = AcceptedClock(
            Fraction(left.time),
            Fraction(right.time),
            h,
            1,
            ordinal,
            1,
            Fraction(segment.start),
            Fraction(segment.end),
            segment.id,
            "BE-reference",
        )
        fields = {
            k: tuple(Polynomial((b, b - a)) for a, b in zip(lo[k], hi[k])) for k in lo
        }
        inputs = tuple(
            Polynomial((Fraction(float(b)), Fraction(float(b)) - Fraction(float(a))))
            for a, b in zip(left.inputs, right.inputs)
        )
        path = PolynomialPath(
            clock,
            fields,
            inputs,
            self.model.source_identity,
            self.model.layout.identity,
            digest({"left": left.identity, "right": right.identity}),
            "declared_reconstruction",
        )
        return SlabPathObserver(self.model, path)

    def point(self, point: Point, segment: Any) -> dict:
        from scripts.benchmarks.interval_observation import (
            AcceptedClock,
            Enclosure,
            Polynomial,
            PolynomialPath,
            SlabPathObserver,
        )

        self.checkpoint()
        self.arithmetic.calls.clear()
        physical = _physical_exact(point)
        h = Fraction(segment.end) - Fraction(segment.start)
        clock = AcceptedClock(
            Fraction(segment.start),
            Fraction(segment.end),
            h,
            1,
            1,
            1,
            Fraction(segment.start),
            Fraction(segment.end),
            segment.id,
            "BE-point-probe",
        )
        fields = {
            k: tuple(Polynomial((v,)) for v in values) for k, values in physical.items()
        }
        endpoints = (segment.voltage, segment.photons)
        inputs = tuple(
            Polynomial(
                (
                    Fraction(float(pair[1])),
                    Fraction(float(pair[1])) - Fraction(float(pair[0])),
                )
            )
            for pair in endpoints
        )
        path = PolynomialPath(
            clock,
            fields,
            inputs,
            self.model.source_identity,
            self.model.layout.identity,
            digest({"point": point.identity, "side_segment": segment.id}),
            "declared_reconstruction",
        )
        observer = SlabPathObserver(self.model, path)
        coordinate = clock.coordinate(point.time)
        ports = [
            self.arithmetic.point(
                lambda z, analytic, i=i: observer.currents(
                    self.arithmetic, z, analytic
                )["same_state_affine_tangent_current"][i],
                coordinate,
            )
            for i in range(2)
        ]
        charges = [
            Enclosure(p.at(coordinate))
            for name in ("body_charge", "metal_charge")
            for p in path.action(self.model.linear_forms["state"][name])
        ]
        return {
            "ports": {
                "left": _enclosure(ports[0]),
                "right": _enclosure(ports[1]),
                "unit": "A",
            },
            "charges": {
                "device": _enclosure(charges[0]),
                "left_metal": _enclosure(charges[1]),
                "right_metal": _enclosure(charges[2]),
                "unit": "C",
            },
            "observation_uncertainty": "finite returned Arb balls and exact affine actions",
            "derivative_origin": "same_state_affine_tangent",
            "evidence_status": "represented_point_only; finite-g/global-reference error separate",
        }

    def interval(
        self, left: Point, right: Point, step: Any, segment: Any, target: TimeStep
    ) -> dict:
        observer = self.path(left, right, segment, target.segment_step)
        arithmetic = self.arithmetic
        h = Fraction(target.h)
        arithmetic.calls.clear()

        def flux(channel, z, analytic):
            self.checkpoint()
            if channel == 0:
                carriers, _ = observer.fluxes(arithmetic, z, analytic)
                capture = (
                    observer.capture_charge_rate[0] + observer.capture_charge_rate[-1]
                )
                return arithmetic.number(observer.area) * (
                    carriers[0] - carriers[-1]
                ) + arithmetic.polynomial(capture, z)
            return (
                arithmetic.polynomial(observer.tangent_metal_polynomial[channel - 1], z)
                + observer.tangent_metal_flux(arithmetic, z, analytic)[channel - 1]
            )

        values = [
            arithmetic.integrate(
                lambda z, analytic, i=i: arithmetic.number(h) * flux(i, z, analytic),
                -1,
                0,
                self.absolute_error,
            )
            for i in range(3)
        ]
        # No absolute-L1 bound or forward-error value is invented here.
        calls = len(arithmetic.calls)
        arithmetic.calls.clear()
        return {
            "physical_flux": [_enclosure(v) for v in values],
            "unit": "C",
            "path_identity": observer.path.identity,
            "origin": "declared_reconstruction",
            "integral_abs_F_lower": None,
            "integral_abs_F_upper": None,
            "reference_error": None,
            "reference_error_status": "unknown_reference_error",
            "storage_secant_used_as_flux": False,
            "quadrature_calls": calls,
        }


def _reference_model(request: Mapping) -> Any:
    from scripts.benchmarks.coupled_device_prototype import (
        AffineCoupledSlab,
        SlabDefinition,
    )

    packet = request["numeric_packet"]
    _require(_integer(packet["intervals"], 2), "layout_intervals")
    definition = SlabDefinition(**packet["definition"])
    _require(
        definition.identity == packet.get("definition_identity"),
        "layout_definition_identity",
    )
    model = AffineCoupledSlab(definition, packet["intervals"])
    _require(model.layout.identity == packet.get("layout_identity"), "layout_identity")
    return model


def reference_layout(request: Mapping) -> Layout:
    """Return a Layout from an already bound request, using the producer model.

    Only public model/geometry/coordinate construction is performed, including
    its static sparse factorization. No time step or observation is evaluated.
    The caller still owns the request's byte/canonical pin and execution identity.
    """
    return _reference_model(request).layout


def prepare_B_backend(
    request: dict, configuration: dict, *, checkpoint: Callable[[], None] | None = None
) -> Backend:
    """Construct only an explicitly admitted, source-bound B input.

    This function is not called by manufactured tests. It uses the real public
    model/validator and the real observation dependency, without replacing any
    shared function or mutating its model definition.
    """
    from scripts.benchmarks.coupled_device_prototype import (
        AffineVoltageMap,
        ProtocolSegment,
        affine_state_quality_evidence,
        validate_voltage_lift_native_request,
    )
    from scripts.benchmarks.interval_observation import BallIntegrator

    _require(
        configuration.get("scientific_execution_authorized") is True
        and configuration.get("source_review_accepted") is True
        and configuration.get("method_status") == "frozen"
        and bool(configuration.get("commander_admission_identity")),
        "B_not_admitted",
    )
    _require(
        request.get("case_id") == "DynamicAcceptorIonPublicDeviceV1"
        and request["numeric_packet"]["intervals"] == 8,
        "B_scope",
    )
    # Verify the selected observer exists before constructing/evaluating B.
    BallIntegrator(bits=512, evaluations=8192, depth=32)
    model = _reference_model(request)
    definition = model.definition
    mapping = AffineVoltageMap(
        model, mapped_input_profile=request["voltage_lift_map"]["mapped_input_profile"]
    )
    segments = tuple(ProtocolSegment(**s) for s in request["segments"])
    validate_voltage_lift_native_request(mapping, segments, request)
    reference = decode_point(
        request["numeric_packet"]["initial_reference"], model.layout
    )
    _require(reference.identity == model.reference.identity, "B_original_reference")
    frame = request["initial_segment_frame"]
    primitive = FrameInputExpansion(
        tuple(
            np.asarray([float.fromhex(v) for v in row])
            for row in frame["physical_handoff_words_hex"]
        )
    )
    inputs, _ = segments[0].inputs(segments[0].start)
    initial, _ = model.coordinates.trial(
        model.reference,
        primitive,
        segments[0].start,
        inputs,
        predecessor=model.reference,
    )
    _require(
        initial.identity == request["initial_preparation"]["point_identity"],
        "B_actual_initial_Point",
    )
    supplied = configuration["equation_budgets"]
    budgets = {}
    for equation in model.layout.equations:
        unit = equation.unit * SECOND if equation.role == "storage" else equation.unit
        row = supplied[equation.id]
        _require(row["unit"] == json.loads(json.dumps(asdict(unit))), "B_budget_unit")
        budgets[equation.id] = EquationBudget(
            unit, [float.fromhex(x) for x in row["absolute_hex"]]
        )
    _require(set(supplied) == set(budgets), "B_budget_keys")
    scaling = PhysicalScaling(
        configuration["equation_scaling"], configuration["state_scaling"]
    )
    observer = SlabObserver(
        model, _unpair(configuration["observation_absolute_error_C"]), checkpoint
    )

    def feasible(point):
        values = _physical_exact(point)
        for spec in model.layout.variables:
            row = values[spec.id]
            if spec.id in {"n_m3", "p_m3"} and min(row) <= 0:
                return False
            if spec.lower is not None and min(row) < Fraction(spec.lower):
                return False
            if spec.upper is not None and max(row) > Fraction(spec.upper):
                return False
            if spec.id == "c_m3" and max(row) >= Fraction(definition.ion_capacity):
                return False
        return True

    def checks(point):
        result = affine_state_quality_evidence(model, point, request["budgets"])
        return {
            "safety_pass": feasible(point),
            "scientific_failed": result.get("passed") is False,
            "same_state_evidence": result,
        }

    count = configuration.get("segment_count", 3)
    _require(type(count) is int and count in (2, 3), "B_horizon")
    return Backend(
        model.public_problem(),
        model.coordinates,
        initial,
        scaling,
        budgets,
        segments[:count],
        feasible,
        checks,
        observer.point,
        observer.interval,
        {
            "kind": "real_B",
            "source_identity": model.source_identity,
            "observation_controls": {
                "arithmetic": "python-flint==0.8.0",
                "bits": 512,
                "evaluations": 8192,
                "depth": 32,
                "absolute_error_C": _pair(observer.absolute_error),
                "path": "declared_reconstruction",
                "ports": "same_state_affine_tangent",
            },
            "original_initial_rate": request["initial_preparation"],
            "original_frame": frame,
            "native_history_created": False,
            "model_id": definition.id,
            "scientific_qualification": False,
        },
    )


def tags_from_request(request: dict, segment_count: int) -> list[dict]:
    """Preserve the original binary64 label words, including a prefix's sides."""
    segments = request["segments"]
    _require(
        type(segment_count) is int and 1 <= segment_count <= len(segments),
        "request_horizon",
    )
    _require(
        set(request["observation_times"]) == {s["id"] for s in segments},
        "request_sample_segments",
    )
    tags = []
    for ordinal, segment in enumerate(segments[:segment_count]):
        values = request["observation_times"][segment["id"]]
        _require(
            isinstance(values, list)
            and len(values) >= 2
            and all(type(t) in (int, float) and math.isfinite(t) for t in values)
            and values[0] == segment["start"]
            and values[-1] == segment["end"]
            and all(a < b for a, b in zip(values, values[1:])),
            "request_sample_times",
        )
        tags.extend(
            {
                "segment": segment["id"],
                "index": index,
                "time_hex": float(value).hex(),
                "event_side": "right"
                if index == 0
                else "left"
                if index == len(values) - 1
                else "interior",
                "protocol_initial": ordinal == index == 0,
                "protocol_terminal": ordinal == len(segments) - 1
                and index == len(values) - 1,
            }
            for index, value in enumerate(values)
        )
    return tags


def _pinned_bytes(descriptor: dict) -> bytes:
    path = Path(descriptor["path"])
    with path.open("rb") as stream:
        raw = stream.read(4 * 1048576 + 1)
    _require(
        len(raw) <= 4 * 1048576
        and sha256(raw).hexdigest() == descriptor["sha256"]
        and (
            "bytes" not in descriptor
            or _integer(descriptor["bytes"], 1)
            and len(raw) == descriptor["bytes"]
        ),
        "file_pin",
    )
    return raw


def read_frozen_definition(descriptor: dict, *, schema: str) -> dict:
    """Validate the commander wrapper and its unchanged reviewed JSON body.

    The SHA and byte checks identify documents; they do not create execution
    authority. Every caller must still enforce its independent admission gate.
    """
    value = json.loads(_pinned_bytes(descriptor), object_pairs_hook=_strict_object)
    canonical(value)
    _require(
        isinstance(value, dict)
        and value.get("schema") == schema
        and value.get("status") == "frozen",
        "method_unfrozen" if schema == METHOD_SCHEMA else "definition_unfrozen",
    )
    required = {
        "frozen_at",
        "frozen_by",
        "reviewed_path",
        "reviewed_sha256",
        "reviewed_bytes",
        "review_records",
        "user_records",
    }
    freeze = value.get("freeze")
    _require(
        isinstance(freeze, dict)
        and required <= set(freeze) <= required | {"notes", "plan_rule"},
        "freeze_fields",
    )
    _require(
        all(
            type(freeze[k]) is str and bool(freeze[k])
            for k in ("frozen_at", "frozen_by", "reviewed_path")
        )
        and _is_digest(freeze["reviewed_sha256"])
        and _integer(freeze["reviewed_bytes"], 1)
        and isinstance(freeze["review_records"], list)
        and bool(freeze["review_records"])
        and isinstance(freeze["user_records"], list)
        and bool(freeze["user_records"])
        and all(
            type(record) is str and bool(record) for record in freeze["user_records"]
        )
        and all(
            isinstance(record, dict)
            and type(record.get("path")) is str
            and bool(record["path"])
            and _is_digest(record.get("sha256"))
            for record in freeze["review_records"]
        ),
        "freeze_metadata",
    )
    if "notes" in freeze:
        _require(
            isinstance(freeze["notes"], list)
            and all(type(note) is str for note in freeze["notes"]),
            "freeze_notes",
        )
    if "plan_rule" in freeze:
        _require(
            type(freeze["plan_rule"]) is str and bool(freeze["plan_rule"]),
            "freeze_plan_rule",
        )
    reviewed_raw = _pinned_bytes(
        {
            "path": freeze["reviewed_path"],
            "sha256": freeze["reviewed_sha256"],
            "bytes": freeze["reviewed_bytes"],
        }
    )
    reviewed = json.loads(reviewed_raw, object_pairs_hook=_strict_object)
    _require(isinstance(reviewed, dict), "reviewed_definition_object")
    _require(
        canonical({k: v for k, v in value.items() if k not in {"status", "freeze"}})
        == canonical({k: v for k, v in reviewed.items() if k != "status"}),
        "frozen_reviewed_body",
    )
    return value


def _runtime_sources(manifest: Mapping, action: str) -> None:
    required_sources = {str(Path(__file__).resolve())}
    required_sources.update(
        str(Path(__file__).resolve().with_name(name))
        for name in (
            "conservative_device.py",
            "contract_prototype.py",
            "precision_prototype.py",
            "coupled_device_prototype.py",
            "interval_observation.py",
            "native_history.py",
            "g2_route.py",
            "g2_route_launcher.py",
        )
    )
    if action == "evaluate":
        required_sources.add(
            str(Path(__file__).resolve().with_name("g2_reference_error.py"))
        )
    sources = manifest["source_files"]
    _require(
        isinstance(sources, list)
        and len(required_sources) <= len(sources) <= 512
        and required_sources <= {str(Path(s["path"]).resolve()) for s in sources},
        "source_pins",
    )
    for source in sources:
        _pinned_bytes(source)


def _run_manifest(manifest: dict, runtime: StageRuntime, method: dict) -> int:
    config = manifest["configuration"]
    _require(method["time_ladder"]["seed_cap_s"] == [1, 512], "unsupported_method_seed")
    identity = RunIdentity(**manifest["identity"])
    identity.payload()
    _require(
        identity.case_id == runtime.case_id
        and identity.method_contract_sha256 == manifest["method_contract"]["sha256"]
        and identity.plan_sha256 == manifest["plan"]["sha256"],
        "manifest_identity",
    )
    request_path = Path(manifest["request"]["path"])
    _require(request_path.stat().st_size <= 4 * 1048576, "request_size")
    raw = _pinned_bytes(manifest["request"])
    request = json.loads(raw, object_pairs_hook=_strict_object)
    _require(identity.input_sha256 == digest(request), "manifest_identity")
    tags = tags_from_request(request, config.get("segment_count", 3))
    _require(digest(tags) == digest(manifest["tags"]), "original_request_tags")
    limits = runtime.limits
    guard = Guard(limits, runtime.checkpoint, runtime.started_ns / 10**9)
    guard()
    with runtime.ledger.span("setup"):
        backend = prepare_B_backend(request, config, checkpoint=guard)
    expected_tags = 66 if len(backend.segments) == 2 else 131
    _require(len(tags) == expected_tags, "B_tag_count")
    schedule = compile_schedule(backend.segments, tags, level=config["ladder_level"])
    _require(schedule.identity == manifest["schedule_identity"], "schedule_pin")
    result = run_reference(
        backend,
        schedule,
        identity,
        request,
        Path(manifest["directory"]),
        limits,
        encoding=manifest.get("encoding", "raw"),
        checkpoint=guard,
        started_monotonic=guard.start,
        timing=runtime.ledger,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _read_manifest(manifest: dict, runtime: StageRuntime) -> int:
    selected = manifest["history"]
    directory = Path(selected["directory"])
    _require(
        Path(selected["run_result"]["path"]).resolve()
        == (directory / "RunResult.json").resolve(),
        "runtime_readback_result_path",
    )
    _pinned_bytes(selected["run_result"])
    upstream = _upstream_runtime(
        selected["runtime_receipt"],
        runtime,
        directory,
        selected["run_result"],
        allow_failed=True,
    )
    request = _read_json(directory / "ReferenceRequest.json")
    identity = RunIdentity(**selected["identity"])
    with runtime.ledger.span("setup"):
        layout = reference_layout(request)
    with runtime.ledger.span("reader"):
        result = read_reference_history(
            directory,
            layout,
            identity,
            max_record_bytes=runtime.limits.record_bytes,
            max_logical_bytes=runtime.limits.logical_bytes,
            checkpoint=runtime.checkpoint,
        )
    qualified = _runtime_completed(upstream)
    result["upstream_runtime"] = {
        "receipt": selected["runtime_receipt"],
        "outcome": upstream["outcome"],
        "return_code": upstream["return_code"],
        "first_failure": upstream["first_failure"],
        "runtime_qualified": qualified,
        "scope": "execution integrity only; no scientific qualification",
    }
    runtime.directory.mkdir(parents=True, exist_ok=False)
    with runtime.ledger.span("artifact_io"):
        _write_new(
            runtime.directory / "Readback.json",
            result,
            runtime.limits.output_bytes - runtime.limits.metadata_reserve,
        )
    print(
        json.dumps(
            {
                "stage": "read",
                "structural_pass": result["structural_pass"],
                "numerical_complete": result["numerical_complete"],
                "scientific_pass": False,
            }
        )
    )
    return 0 if qualified else 2


def _error_manifest(manifest: dict, runtime: StageRuntime) -> int:
    from scripts.benchmarks import g2_reference_error

    reference = manifest["error_manifest"]
    error_manifest = json.loads(
        _pinned_bytes(reference), object_pairs_hook=_strict_object
    )
    _require(
        error_manifest.get("directory") == str(runtime.directory)
        and error_manifest["method_contract"]["sha256"]
        == manifest["method_contract"]["sha256"],
        "runtime_error_manifest_binding",
    )
    upstream = manifest.get("upstream_runtime")
    _require(
        isinstance(upstream, list)
        and len(upstream) == len(error_manifest["levels"]) == 3,
        "runtime_error_upstream",
    )
    for binding, level in zip(upstream, error_manifest["levels"], strict=True):
        directory = Path(level["directory"])
        _upstream_runtime(binding, runtime, directory)
    error_limits = Limits(**error_manifest["limits"])
    error_limits.validate()
    _require(
        all(
            getattr(error_limits, name) <= getattr(runtime.limits, name)
            for name in (
                "wall_s",
                "rss_bytes",
                "output_bytes",
                "logical_bytes",
                "record_bytes",
            )
        ),
        "runtime_error_limits",
    )
    # The reader and E1/E3 are explicit follow-up stages. Their internal guard
    # cannot enlarge the outer inherited-channel/absolute-pipeline allowance.
    with runtime.ledger.span("reference_error"):
        return g2_reference_error.main(["evaluate", "--manifest", reference["path"]])


def _runtime_document(binding: dict, expected_path: Path | None = None) -> dict:
    """Read a complete immutable metadata pin, keeping integer clocks exact."""
    _require(
        type(binding) is dict
        and set(binding) == {"path", "bytes", "sha256"}
        and type(binding["path"]) is str
        and Path(binding["path"]).is_absolute()
        and str(Path(binding["path"]).resolve()) == binding["path"]
        and _integer(binding["bytes"], 1)
        and _is_digest(binding["sha256"]),
        "runtime_metadata_pin",
    )
    if expected_path is not None:
        _require(
            binding["path"] == str(expected_path.resolve()), "runtime_upstream_path"
        )
    value = json.loads(_pinned_bytes(binding), object_pairs_hook=_strict_object)
    _require(type(value) is dict, "runtime_metadata_object")
    canonical(value)
    return value


def _runtime_failure(value: Any) -> bool:
    return (
        type(value) is dict
        and set(value) == {"type", "reason"}
        and type(value["type"]) is str
        and bool(value["type"])
        and type(value["reason"]) is str
        and len(value["reason"]) <= 4096
    )


def _check_runtime_threads(value: Any, start: int, end: int) -> bool:
    fields = {
        "library",
        "getter",
        "before_mode",
        "measured_mode",
        "required_single_thread_mode",
        "thread_environment",
        "python_executable",
        "python_version",
        "checked_ns",
        "final_measurement_status",
    }
    _require(
        type(value) is dict
        and fields
        <= set(value)
        <= fields
        | {"final_measured_mode", "final_checked_ns", "final_measurement_error"},
        "runtime_thread_fields",
    )
    _require(
        all(
            type(value[k]) is str and bool(value[k])
            for k in ("library", "python_executable", "python_version")
        )
        and Path(value["library"]).is_absolute()
        and Path(value["python_executable"]).is_absolute()
        and value["getter"] == "BLASGetThreading"
        and all(
            _integer(value[k])
            for k in (
                "before_mode",
                "measured_mode",
                "required_single_thread_mode",
                "checked_ns",
            )
        )
        and value["required_single_thread_mode"] == 1
        and start <= value["checked_ns"] <= end
        and type(value["thread_environment"]) is dict
        and value["thread_environment"] == {k: "1" for k in THREAD_ENVIRONMENT},
        "runtime_thread_evidence",
    )
    if "final_measurement_error" in value:
        _require(
            _runtime_failure(value["final_measurement_error"]), "runtime_thread_error"
        )
    if value["final_measurement_status"] == "unavailable":
        _require(
            "final_measured_mode" not in value
            and "final_checked_ns" not in value
            and "final_measurement_error" in value,
            "runtime_thread_unavailable",
        )
        return False
    _require(
        value["final_measurement_status"] == "measured"
        and _integer(value.get("final_measured_mode"))
        and _integer(value.get("final_checked_ns"))
        and value["checked_ns"] <= value["final_checked_ns"] <= end,
        "runtime_thread_final_evidence",
    )
    return (
        value["measured_mode"] == value["final_measured_mode"] == 1
        and "final_measurement_error" not in value
    )


def _check_runtime_timing(value: Any, start: int, end: int, peak_rss: int) -> None:
    fields = {
        "clock",
        "components",
        "segments",
        "peak_process_RSS_bytes",
        "semantics",
        "not_entered",
        "unmeasured",
        "scientific_decisions_use_timing",
    }
    _require(type(value) is dict and set(value) == fields, "runtime_timing_fields")
    _require(
        value["clock"] == "time.monotonic_ns"
        and value["scientific_decisions_use_timing"] is False
        and _integer(value["peak_process_RSS_bytes"], 1)
        and value["peak_process_RSS_bytes"] <= peak_rss
        and type(value["semantics"]) is str
        and bool(value["semantics"])
        and type(value["unmeasured"]) is list
        and all(type(x) is str and bool(x) for x in value["unmeasured"])
        and type(value["components"]) is dict
        and set(value["components"]) == set(RUNTIME_PHASES)
        and type(value["segments"]) is list
        and len(value["segments"]) <= 16,
        "runtime_timing_metadata",
    )
    for row in value["components"].values():
        _require(
            type(row) is dict
            and set(row) == {"calls", "exceptions", "wall_ns"}
            and all(_integer(v) for v in row.values())
            and row["exceptions"] <= row["calls"]
            and row["wall_ns"] <= end - start
            and (row["calls"] > 0 or row["wall_ns"] == 0),
            "runtime_component_timing",
        )
    _require(
        value["not_entered"]
        == [k for k in RUNTIME_PHASES if value["components"][k]["calls"] == 0],
        "runtime_not_entered",
    )
    counters = {
        "accepted_steps",
        "residual_evaluations",
        "factorizations",
        "linear_solves",
    }
    fields = counters | {
        "segment",
        "started_ns",
        "ended_ns",
        "wall_ns",
        "components",
        "outcome",
    }
    previous, names = start, set()
    totals = {k: 0 for k in RUNTIME_PHASES}
    for row in value["segments"]:
        _require(type(row) is dict and set(row) == fields, "runtime_segment_fields")
        _require(
            type(row["segment"]) is str
            and bool(row["segment"])
            and row["segment"] not in names
            and all(
                _integer(row[k])
                for k in counters | {"started_ns", "ended_ns", "wall_ns"}
            )
            and previous <= row["started_ns"] <= row["ended_ns"] <= end
            and row["wall_ns"] == row["ended_ns"] - row["started_ns"]
            and row["outcome"] in {"completed", "failed"}
            and type(row["components"]) is dict
            and set(row["components"]) == set(RUNTIME_PHASES),
            "runtime_segment_timing",
        )
        for name, duration in row["components"].items():
            _require(
                _integer(duration) and duration <= row["wall_ns"],
                "runtime_segment_component",
            )
            totals[name] += duration
        previous = row["ended_ns"]
        names.add(row["segment"])
    _require(
        all(totals[k] <= value["components"][k]["wall_ns"] for k in RUNTIME_PHASES)
        and all(
            value["components"][k]["calls"] == 0 for k in ("reader", "reference_error")
        ),
        "runtime_producer_phase_consistency",
    )


def _runtime_completed(record: Mapping) -> bool:
    return record["outcome"] == "returned" and record["return_code"] == 0


def _upstream_runtime(
    binding: dict,
    runtime: StageRuntime,
    directory: Path,
    result_file=None,
    *,
    allow_failed: bool = False,
) -> dict:
    """Validate all producer evidence before read/evaluate can consume it.

    BE3 receipts remain historical evidence; only the complete BE4 receipt
    (including its manifest pin) can authorize a new supervised follow-up.
    A failed producer can be read explicitly, never qualified for E1/E3.
    """
    record = _runtime_document(binding, directory / "RuntimeReceipt.json")
    fields = {
        "schema",
        "stage",
        "case_id",
        "plan",
        "manifest_digest",
        "manifest",
        "result_file",
        "pipeline_clock",
        "pipeline_id",
        "started_ns",
        "ended_ns",
        "original_deadline_ns",
        "work_deadline_ns",
        "execution",
        "outcome",
        "first_failure",
        "return_code",
        "thread_receipt",
        "peak_process_RSS_bytes",
        "timing",
        "scientific_pass",
        "group_RSS_and_closure_authority",
        "followup_stages_executed_implicitly",
    }
    _require(type(record) is dict and set(record) == fields, "runtime_upstream_fields")
    _require(
        record["schema"] == RUNTIME_SCHEMA
        and record["stage"] == "run"
        and type(record["pipeline_id"]) is str
        and record["pipeline_id"] == runtime.anchor["pipeline_id"]
        and canonical(record["pipeline_clock"]) == canonical(runtime.clock_reference)
        and _integer(record["original_deadline_ns"], 1)
        and record["original_deadline_ns"] == runtime.anchor["deadline_ns"],
        "runtime_upstream_clock",
    )
    _require(
        type(record["case_id"]) is str
        and bool(record["case_id"])
        and _is_digest(record["manifest_digest"])
        and record["scientific_pass"] is False
        and record["followup_stages_executed_implicitly"] is False
        and record["group_RSS_and_closure_authority"]
        == "outer g2_route_launcher receipt"
        and _integer(record["peak_process_RSS_bytes"], 1)
        and all(
            _integer(record[k]) for k in ("started_ns", "ended_ns", "work_deadline_ns")
        ),
        "runtime_upstream_types",
    )
    _runtime_document(record["pipeline_clock"])
    plan = _runtime_document(record["plan"])
    manifest = _runtime_document(record["manifest"], directory / "RuntimeManifest.json")
    _require(
        digest(manifest) == record["manifest_digest"]
        and canonical(manifest.get("plan")) == canonical(record["plan"])
        and manifest.get("case_id", manifest.get("identity", {}).get("case_id"))
        == record["case_id"]
        and manifest.get("directory") == str(directory.resolve()),
        "runtime_upstream_manifest",
    )
    _require(
        plan.get("status") == "admitted"
        and type(plan.get("authority")) is dict
        and plan["authority"].get("scientific_execution_authorized") is True
        and type(plan.get("cases")) is list
        and all(type(c) is dict for c in plan["cases"])
        and type(plan.get("resources")) is dict
        and _integer(plan["resources"].get("WS1_channels"))
        and plan["resources"]["WS1_channels"] == 4
        and _integer(plan["resources"].get("threads_per_case"))
        and plan["resources"]["threads_per_case"] == 1,
        "runtime_upstream_plan",
    )
    matches = [c for c in plan["cases"] if c.get("id") == record["case_id"]]
    ex = record["execution"]
    _require(
        len(matches) == 1
        and type(ex) is dict
        and canonical(matches[0].get("execution")) == canonical(ex)
        and ex.get("stage") == "run"
        and ex.get("directory") == str(directory.resolve())
        and _integer(ex.get("channel"))
        and ex["channel"] < 4
        and type(ex.get("pipeline")) is dict
        and set(ex["pipeline"]) == {"id", "clock"}
        and ex["pipeline"]["id"] == record["pipeline_id"]
        and canonical(ex["pipeline"]["clock"]) == canonical(record["pipeline_clock"]),
        "runtime_upstream_execution",
    )
    policy = validate_route_supervision(plan, ex)
    limits = Limits(**manifest["limits"])
    limits.validate()
    cleanup = math.ceil(
        sum(
            Fraction(policy[k])
            for k in ("term_grace_s", "kill_wait_s", "receipt_reserve_s")
        )
        * 10**9
    )
    start, end, deadline = (
        record["started_ns"],
        record["ended_ns"],
        record["original_deadline_ns"],
    )
    _require(
        runtime.anchor["started_ns"] <= start <= end <= runtime.started_ns <= deadline
        and limits.wall_s <= ex["wall_s"] <= runtime.anchor["wall_s"]
        and math.ceil(Fraction(ex["wall_s"]) * 10**9) <= deadline - start
        and record["work_deadline_ns"]
        == min(
            deadline - cleanup,
            start + math.floor(Fraction(ex["wall_s"]) * 10**9) - cleanup,
            start + math.floor(Fraction(limits.wall_s) * 10**9),
        )
        and start < record["work_deadline_ns"] <= deadline
        and limits.rss_bytes <= policy["rss_bytes"] <= 4 * GIB
        and limits.output_bytes <= policy["artifact_bytes"] <= 4 * GIB,
        "runtime_upstream_chronology",
    )
    threads_ok = _check_runtime_threads(record["thread_receipt"], start, end)
    _check_runtime_timing(
        record["timing"], start, end, record["peak_process_RSS_bytes"]
    )
    _require(
        record["outcome"] in {"returned", "raised"}
        and (record["return_code"] is None or _integer(record["return_code"]))
        and (
            record["first_failure"] is None or _runtime_failure(record["first_failure"])
        )
        and (
            record["outcome"] == "raised"
            and record["first_failure"] is not None
            or record["outcome"] == "returned"
            and record["first_failure"] is None
            and _integer(record["return_code"])
            and threads_ok
            and end <= start + math.floor(Fraction(ex["wall_s"]) * 10**9)
            and record["peak_process_RSS_bytes"] <= limits.rss_bytes
        ),
        "runtime_upstream_outcome",
    )
    saved = record["result_file"]
    _require(
        result_file is None or canonical(saved) == canonical(result_file),
        "runtime_upstream_result",
    )
    result = _runtime_document(saved, directory / "RunResult.json")
    _require(
        result.get("case_id") == record["case_id"]
        and result.get("schema") == SCHEMA
        and result.get("plan_sha256") == record["plan"]["sha256"]
        and result.get("scientific_pass") is False
        and type(result.get("numerical_complete")) is bool
        and (
            manifest.get("identity") is None
            or all(
                result.get(k) == v
                for k, v in RunIdentity(**manifest["identity"]).payload().items()
            )
        ),
        "runtime_upstream_case",
    )
    _require(
        result.get("status")
        == ("completed" if result["numerical_complete"] else "failed")
        and (not _runtime_completed(record) or result["numerical_complete"] is True)
        and (allow_failed or _runtime_completed(record)),
        "runtime_upstream_incomplete",
    )
    request = _read_json(directory / "ReferenceRequest.json")
    _require(digest(request) == result.get("input_sha256"), "runtime_upstream_request")
    segments = record["timing"]["segments"]
    _require(
        type(request.get("segments")) is list
        and [row["segment"] for row in segments]
        == [row["id"] for row in request["segments"][: len(segments)]],
        "runtime_upstream_protocol_phases",
    )
    if _runtime_completed(record):
        counts = result.get("counts")
        names = (
            "accepted_steps",
            "residual_evaluations",
            "factorizations",
            "linear_solves",
        )
        _require(
            type(counts) is dict
            and all(_integer(counts.get(k)) for k in (*names, "segments"))
            and len(segments) == counts["segments"]
            and all(sum(row[k] for row in segments) == counts[k] for k in names)
            and all(row["outcome"] == "completed" for row in segments),
            "runtime_upstream_counts",
        )
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "read", "evaluate"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    _require(args.manifest.stat().st_size <= 4 * 1048576, "manifest_size")
    manifest = _read_json(args.manifest)
    schema = (
        "solarlab.g2-reference-be-run.v1"
        if args.action == "run"
        else "solarlab.g2-reference-be-stage.v1"
    )
    _require(manifest.get("schema") == schema, "manifest_schema")
    method = read_frozen_definition(manifest["method_contract"], schema=METHOD_SCHEMA)
    if "gate_spec" in manifest:
        read_frozen_definition(
            manifest["gate_spec"], schema="solarlab.g2-route.t1-gatespec.v3"
        )
    config = manifest["configuration"]
    _require(config.get("scientific_execution_authorized") is True, "B_not_admitted")
    _require(
        config.get("source_review_accepted") is True
        and bool(config.get("commander_admission_identity")),
        "runtime_source_not_admitted",
    )
    plan = json.loads(_pinned_bytes(manifest["plan"]), object_pairs_hook=_strict_object)
    _runtime_sources(manifest, args.action)
    with supervised_stage(manifest, plan, args.action) as runtime:
        if args.action == "run":
            code = _run_manifest(manifest, runtime, method)
        elif args.action == "read":
            code = _read_manifest(manifest, runtime)
        else:
            code = _error_manifest(manifest, runtime)
        runtime.return_code = code
        return code


if __name__ == "__main__":
    raise SystemExit(main())

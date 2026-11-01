"""Draft P05-01a result envelopes and lossless bounded array serialization.

The seven original NumericsHardening §0.2 labels retain their scopes. A
successful run cannot promote them. Physics/state transfer remains P05-01b.
Large arrays belong in the existing independently verified NPY artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

from solarlab.experiments.protocol import _keys, _reference, _ref_schema, _sha, _strings, _text
from solarlab.io.artifacts import _json_document, _parse_document

__all__ = ["EVIDENCE_LABELS", "ExperimentResult", "encode_array", "decode_array", "result_schema", "array_schema"]

EVIDENCE_LABELS = ("PARKED", "FIRST_SLICE", "INTERNAL_TESTED", "INTERNAL_CERTIFIED", "CROSS_CODE_CALIBRATED",
                   "EXTERNAL_SOLVER_VALIDATED", "EXPERIMENT_VALIDATED")
_BINDINGS = ("source", "schema", "registry", "physics", "protocol", "initial", "numerics", "resources")
_HEADER = {"run_id", "attempt_id", "experiment_id", "engine", "manifest_sha256", "outcome", "reason", "bindings", "evidence"}
_MAX_ARRAY_RANK = 32


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    canonical_json: str

    def __post_init__(self) -> None:
        value = _parse_document(self.canonical_json)
        _keys(value, {"schema", "header", "payload_schema", "payload", "metrics"})
        if value["schema"] != "solarlab.experiment-result.v1":
            raise ValueError("unsupported result schema")
        _text(value["payload_schema"])
        if type(value["payload"]) is not dict or type(value["metrics"]) is not dict:
            raise ValueError("experiment payload and metrics must be explicit objects")
        header = value["header"]
        _keys(header, _HEADER)
        for name in ("run_id", "attempt_id", "experiment_id"):
            _text(header[name])
        _sha(header["manifest_sha256"])
        if header["engine"] not in {"legacy", "solarlab"} or header["outcome"] not in {"completed", "failed", "partial"}:
            raise ValueError("explicit engine and completed/failed/partial outcome required")
        if header["outcome"] == "completed":
            if header["reason"] is not None:
                raise ValueError("a completed result has no failure reason")
        else:
            _text(header["reason"])
        _keys(header["bindings"], set(_BINDINGS))
        for record in header["bindings"].values():
            _reference(record)
        if type(header["evidence"]) is not list or not header["evidence"]:
            raise ValueError("original scoped evidence labels must be supplied")
        for item in header["evidence"]:
            _keys(item, {"label", "scope", "sources", "limitations"})
            if item["label"] not in EVIDENCE_LABELS:
                raise ValueError("unknown original evidence label")
            _strings(item["scope"], nonempty=True)
            _strings(item["limitations"])
            if type(item["sources"]) is not list:
                raise ValueError("evidence source references must be explicit")
            if item["label"] not in {"PARKED", "FIRST_SLICE"} and not item["sources"]:
                raise ValueError("tested/validated labels require source references, independently verified upstream")
            for record in item["sources"]:
                _reference(record)
        for name, metric in value["metrics"].items():
            _text(name)
            _keys(metric, {"status", "value", "reason"})
            if metric["status"] not in {"identified", "unidentified"}:
                raise ValueError("metric identification is separate from run completion")
            if metric["value"] is None:
                _text(metric["reason"])
            elif (metric["status"] != "identified" or type(metric["value"]) not in (int, float)
                  or metric["reason"] is not None):
                raise ValueError("an unidentified or missing metric cannot carry a numeric value")
        object.__setattr__(self, "canonical_json", _json_document(value))

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> ExperimentResult:
        return cls(_json_document(value))

    def export(self) -> dict[str, Any]:
        return _parse_document(self.canonical_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


def _dtype(value: Any) -> Any:
    import numpy as np
    dtype = np.dtype(value)
    if dtype.str != value or dtype.fields or dtype.kind not in "biufc" or (
        dtype.kind == "f" and dtype.itemsize not in (4, 8) or dtype.kind == "c" and dtype.itemsize not in (8, 16)
    ):
        raise ValueError("arrays require canonical bool/integer/binary32/binary64 or matching complex dtype")
    return dtype


def _component(array: Any) -> dict[str, Any]:
    import numpy as np
    flat = array.ravel(order="C")
    words, size = flat.tobytes(), flat.dtype.itemsize
    values, mask, reasons, special = [], [], [], []
    for index, number in enumerate(flat):
        missing = not bool(np.isfinite(number))
        values.append(None if missing else number.item())
        mask.append(missing)
        reasons.append(("nan" if np.isnan(number) else "negative_infinity" if np.signbit(number) else "positive_infinity") if missing else None)
        special.append(words[index * size:(index + 1) * size].hex() if missing else None)
    return {"values": values, "mask": mask, "reasons": reasons, "nonfinite_words": special}


def encode_array(array: Any, *, unit: str) -> dict[str, Any]:
    """real/imag + shape; nonfinite values are null/mask/reason, with raw words.

    Supplemental words retain NaN payloads/signs without treating them as valid
    metrics. Finite decimal JSON round-trips to the exact source dtype words.
    Boolean words must be canonical 00/01; other true representations refuse
    before conversion to Python bool can silently normalize their bytes.
    """
    import numpy as np
    if type(array) is not np.ndarray:
        raise TypeError("expected an explicit numpy array")
    _text(unit)
    dtype = _dtype(array.dtype.str)
    if array.ndim > _MAX_ARRAY_RANK:
        raise ValueError("array rank exceeds the schema/decoder limit of 32")
    if dtype.kind == "b" and any(word not in (0, 1) for word in array.tobytes(order="C")):
        raise ValueError("noncanonical boolean words require a raw artifact; inline encoding accepts only 00/01")
    value = {"schema": "solarlab.numeric-array.v1", "dtype": dtype.str, "shape": list(array.shape), "unit": unit,
             "real": _component(array.real), "imag": _component(array.imag) if dtype.kind == "c" else None}
    _json_document(value)  # Bounded inline contract; large arrays use artifacts.
    return value


def decode_array(value: dict[str, Any]) -> Any:
    import numpy as np
    _json_document(value)
    _keys(value, {"schema", "dtype", "shape", "unit", "real", "imag"})
    if value["schema"] != "solarlab.numeric-array.v1":
        raise ValueError("unknown numeric array schema")
    _text(value["unit"])
    dtype = _dtype(value["dtype"])
    shape = value["shape"]
    if type(shape) is not list or len(shape) > _MAX_ARRAY_RANK or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("invalid array shape")
    count = math.prod(shape)
    real_dtype = np.empty(0, dtype=dtype).real.dtype

    def read_component(component: Any) -> bytes:
        _keys(component, {"values", "mask", "reasons", "nonfinite_words"})
        if any(type(items) is not list or len(items) != count for items in component.values()):
            raise ValueError("component lengths differ from shape")
        parts = []
        for number, masked, reason, word in zip(component["values"], component["mask"], component["reasons"], component["nonfinite_words"]):
            if type(masked) is not bool:
                raise ValueError("nonfinite mask requires booleans")
            if masked:
                if number is not None or reason not in {"nan", "positive_infinity", "negative_infinity"} or type(word) is not str:
                    raise ValueError("nonfinite component requires null, reason and original words")
                raw = bytes.fromhex(word)
                if len(raw) != real_dtype.itemsize:
                    raise ValueError("nonfinite word has the wrong dtype size")
            else:
                if number is None or reason is not None or word is not None:
                    raise ValueError("finite component cannot be missing or masked")
                if (real_dtype.kind == "b" and type(number) is not bool
                    or real_dtype.kind in "iu" and type(number) is not int
                    or real_dtype.kind == "f" and type(number) is not float):
                    raise ValueError("finite component has the wrong scalar type")
                raw = np.asarray([number], dtype=real_dtype).tobytes()
            parts.append(raw)
        return b"".join(parts)

    real = read_component(value["real"])
    if dtype.kind == "c":
        imag = read_component(value["imag"])
        size = real_dtype.itemsize
        raw = b"".join(real[i:i + size] + imag[i:i + size] for i in range(0, len(real), size))
    else:
        if value["imag"] is not None:
            raise ValueError("real arrays cannot carry an imaginary component")
        raw = real
    result = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if _json_document(encode_array(result, unit=value["unit"])) != _json_document(value):
        raise ValueError("array values/mask/reasons/words are inconsistent or noncanonical")
    return result


def array_schema() -> dict[str, Any]:
    component = {"type": "object", "additionalProperties": False,
        "required": ["values", "mask", "reasons", "nonfinite_words"], "properties": {
            "values": {"type": "array", "items": {"type": ["number", "boolean", "null"]}},
            "mask": {"type": "array", "items": {"type": "boolean"}},
            "reasons": {"type": "array", "items": {"enum": [None, "nan", "positive_infinity", "negative_infinity"]}},
            "nonfinite_words": {"type": "array", "items": {"type": ["string", "null"]}}}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": False,
        "required": ["schema", "dtype", "shape", "unit", "real", "imag"], "properties": {
            "schema": {"const": "solarlab.numeric-array.v1"}, "dtype": {"type": "string"},
            "shape": {"type": "array", "maxItems": _MAX_ARRAY_RANK, "items": {"type": "integer", "minimum": 0}},
            "unit": {"type": "string", "minLength": 1}, "real": component,
            "imag": {"anyOf": [component, {"type": "null"}]}}}


def result_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {key: {"type": "string", "minLength": 1} for key in ("run_id", "attempt_id", "experiment_id")}
    properties.update(engine={"enum": ["legacy", "solarlab"]}, manifest_sha256={"type": "string", "pattern": "^[0-9a-f]{64}$"},
        outcome={"enum": ["completed", "failed", "partial"]}, reason={"type": ["string", "null"]},
        bindings={"type": "object", "additionalProperties": False, "required": list(_BINDINGS),
                  "properties": {name: _ref_schema() for name in _BINDINGS}},
        evidence={"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False,
            "required": ["label", "scope", "sources", "limitations"], "properties": {
                "label": {"enum": list(EVIDENCE_LABELS)}, "scope": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "minLength": 1}},
                "sources": {"type": "array", "items": _ref_schema()}, "limitations": {"type": "array", "uniqueItems": True, "items": {"type": "string", "minLength": 1}}}}})
    header = {"type": "object", "additionalProperties": False,
        "required": sorted(_HEADER), "properties": properties, "allOf": [{
            "if": {"properties": {"outcome": {"const": "completed"}}},
            "then": {"properties": {"reason": {"type": "null"}}},
            "else": {"properties": {"reason": {"type": "string", "minLength": 1}}}}]}
    metric = {"type": "object", "additionalProperties": False, "required": ["status", "value", "reason"],
        "properties": {"status": {"enum": ["identified", "unidentified"]},
                       "value": {"type": ["number", "null"]}, "reason": {"type": ["string", "null"]}},
        "allOf": [{"if": {"properties": {"value": {"type": "null"}}},
                   "then": {"properties": {"reason": {"type": "string", "minLength": 1}}},
                   "else": {"properties": {"status": {"const": "identified"}, "reason": {"type": "null"}}}}]}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": False,
        "required": ["schema", "header", "payload_schema", "payload", "metrics"], "properties": {
            "schema": {"const": "solarlab.experiment-result.v1"}, "header": header,
            "payload_schema": {"type": "string", "minLength": 1}, "payload": {"type": "object"},
            "metrics": {"type": "object", "additionalProperties": metric}}}

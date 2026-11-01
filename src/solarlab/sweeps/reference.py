"""Trusted coordinate coverage only; reference metrics never enter preparation."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any


def baseline_signature(value: Any) -> str:
    """Reference correspondence by exact JSON values, not int/float spelling.

    Browsers serialize 1.0 as 1. This comparison alone treats those numeric
    values alike; it does not rewrite input, convert units or erase signed zero.
    """
    def words(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: words(part) for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [words(part) for part in item]
        if type(item) in (int, float):
            number = Fraction(item)
            return ["number", str(number.numerator), str(number.denominator), item == 0 and math.copysign(1, item) < 0]
        return [type(item).__name__, item]
    return hashlib.sha256(json.dumps(words(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SweepReference:
    id: str
    source_id: str
    source_sha256: str
    document_json: bytes

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.document_json)

    def match(self, targets: dict[str, dict[str, Any]], coordinates: dict[str, dict[str, Any]], baseline_sha256: str) -> dict[str, Any]:
        document = self.to_mapping()
        result = dict(id=self.id, source_id=self.source_id, source_sha256=self.source_sha256, comparison_qualified=False,
                      baseline_matches=baseline_sha256 == document["baseline_input_sha256"], scope="coordinate_and_target_coverage_only")
        if document["targets"] != targets:
            return {**result, "status": "target_mismatch", "reason": "reference belongs to different named parameter instances"}
        for row in document["rows"]:
            if row["coordinates"] == coordinates:
                return {**result, **row}
        return {**result, "status": "coordinate_missing", "reason": "no reference row at these exact normalized coordinates; no positional or nearest-point substitution"}

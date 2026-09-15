"""The complete approved R1-0 reference used by the fixed R1-1 study.

The JSON is copied unchanged from R1IndependentAcceptanceV2/PreparationV1.
Its binding-content SHA-256 is
0a6532a436dd07e6b27f01da3fc39aef906e6fd882057f0109c1bc5bdf77e39b;
the file-byte SHA-256 is
780e0275a3278e2d0ef2eb0ddf748ac763a7cc38ac0d03796312a952e0a01b4e.
Tests read the repository fixture and do not require the external archive.
"""

import json
import hashlib
from pathlib import Path


APPROVED_R1_BINDING_PATH = Path(__file__).with_name(
    "OneDimensionalMechanismR1ReferenceBindingV1.json"
)


def approved_r1_binding():
    """Return a fresh copy, preserving the approved payload and embedded digest."""
    return json.loads(APPROVED_R1_BINDING_PATH.read_text(encoding="utf-8"))


def alternate_r1_binding():
    """Make a self-consistent reference that is outside the approved study."""
    record = approved_r1_binding()
    for rung in record["rungs"]:
        rung["f_ref"] = [value + 5e-5 for value in rung["f_ref"]]
    for previous, current in zip(record["rungs"], record["rungs"][1:]):
        current["difference"] = max(
            abs(right - left) for left, right in zip(previous["f_ref"], current["f_ref"])
        )
    record["f_ref"] = record["rungs"][-1]["f_ref"].copy()
    payload = {key: value for key, value in record.items() if key != "sha256"}
    record["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return record

"""Admit only the approved fixed reference to the versioned R1-1 study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from perovskite_sim.experiments.one_dimensional_mechanism_r1 import validate_binding


STUDY_INPUT_RELATIVE_PATH = "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
STUDY_INPUT_PATH = Path(__file__).resolve().parents[2] / STUDY_INPUT_RELATIVE_PATH


def study_input_identity():
    """Bind prepared source identity to the repository-owned study protocol."""
    return {
        "path": STUDY_INPUT_RELATIVE_PATH,
        "sha256": hashlib.sha256(STUDY_INPUT_PATH.read_bytes()).hexdigest(),
    }


def validate_r1_study_binding(binding, stack):
    """Verify internal consistency and the approved canonical binding digest.

    The trust root is the versioned repository input, never a caller-supplied
    input or the binding's own claimed digest. JSON file formatting is not
    part of binding identity; archive manifests independently seal file bytes.
    """
    validate_binding(binding, stack)
    study = json.loads(STUDY_INPUT_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(study, dict)
        or study.get("schema") != "one-dimensional-mechanism-r1-1-input-v1"
        or study.get("stage_scope") != "R1-1"
    ):
        raise ValueError("R1-1 approved-reference study input is invalid")
    expected = study.get("fixed_reference_binding_sha256")
    if (
        not isinstance(expected, str) or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("R1-1 study input lacks a valid approved reference digest")
    # validate_binding has already recomputed the canonical payload digest and
    # compared it to this field, so a forged copy of the approved hash fails.
    if binding["sha256"] != expected:
        raise ValueError("R1-1 reference binding is not the approved study reference")


__all__ = ["validate_r1_study_binding", "study_input_identity"]

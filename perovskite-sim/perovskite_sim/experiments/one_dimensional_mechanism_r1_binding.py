"""Admit only the approved fixed reference to the versioned R1-1 study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from perovskite_sim.experiments.one_dimensional_mechanism_r1 import validate_binding
from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import (
    R1CheckoutError, require_r1_checkout,
)


STUDY_INPUT_RELATIVE_PATH = "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
STUDY_INPUT_PATH = Path(__file__).resolve().parents[2] / STUDY_INPUT_RELATIVE_PATH
EXECUTION_CONTRACT_RELATIVE_PATH = "docs/OneDimensionalMechanismR1DynamicsV1.md"
OPERATOR_CRITERION_RELATIVE_PATH = "docs/OneDimensionalMechanismR1OperatorCriterionDecisionV2.md"
ADDITIONAL_FAILURES_RELATIVE_PATH = "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV1.json"
# A versioned policy pin, not a claim that arbitrary code carrying this value
# is independently approved. Updating the study input requires explicit repin.
PINNED_STUDY_INPUT_SHA256 = "3065a31951d4a90e8b0d03d042e3f2c0349cc7d8e792381a275cb05d58a047ca"
PINNED_REFERENCE_BINDING_SHA256 = "0a6532a436dd07e6b27f01da3fc39aef906e6fd882057f0109c1bc5bdf77e39b"
PINNED_EXECUTION_CONTRACT_SHA256 = "a0a918d175c7989db60d5924b7600baadd1929cc54c7ac66b3ec3ccdcc1f9955"
PINNED_OPERATOR_CRITERION_SHA256 = "0adf2d3fec785febb9747c47e62a532fb23fabe6d58d08925e8de5573f657b9b"
PINNED_ADDITIONAL_FAILURES_SHA256 = "ae2d52cf08a029e0273689d03f16ce8945fdd1ca423582542a49a641d50f9aaa"


def additional_failures_identity():
    """Pin supplemental observations without rewriting the original study."""
    context = require_r1_checkout()
    raw = context.read_bytes(ADDITIONAL_FAILURES_RELATIVE_PATH)
    if hashlib.sha256(raw).hexdigest() != PINNED_ADDITIONAL_FAILURES_SHA256:
        raise R1CheckoutError("R1 additional failures differ from the pinned registry digest")
    return {"path": ADDITIONAL_FAILURES_RELATIVE_PATH, "sha256": PINNED_ADDITIONAL_FAILURES_SHA256}


def operator_criterion_identity():
    """Pin the frozen disposition independently of execution metadata."""
    context = require_r1_checkout()
    raw = context.read_bytes(OPERATOR_CRITERION_RELATIVE_PATH)
    if hashlib.sha256(raw).hexdigest() != PINNED_OPERATOR_CRITERION_SHA256:
        raise R1CheckoutError("R1 operator criterion differs from the pinned decision digest")
    return {"path": OPERATOR_CRITERION_RELATIVE_PATH, "sha256": PINNED_OPERATOR_CRITERION_SHA256}


def execution_contract_identity():
    """The trusted code pins the contract independently of a supplied bundle."""
    context = require_r1_checkout()
    raw = context.read_bytes(EXECUTION_CONTRACT_RELATIVE_PATH)
    if hashlib.sha256(raw).hexdigest() != PINNED_EXECUTION_CONTRACT_SHA256:
        raise R1CheckoutError("R1 execution contract differs from the pinned contract digest")
    return {"path": EXECUTION_CONTRACT_RELATIVE_PATH, "sha256": PINNED_EXECUTION_CONTRACT_SHA256}


def _checked_study_input():
    context = require_r1_checkout()
    try:
        path = Path(STUDY_INPUT_PATH)
        same_input = path.resolve() == context.study_input.resolve() and path.samefile(context.study_input)
    except (OSError, TypeError, ValueError):
        same_input = False
    if not same_input:
        raise R1CheckoutError("R1 study input must use the canonical tracked checkout path")
    raw = context.read_bytes(context.study_input)
    if hashlib.sha256(raw).hexdigest() != PINNED_STUDY_INPUT_SHA256:
        raise R1CheckoutError("R1 study input differs from the pinned complete input digest")
    return raw


def study_input_identity():
    """Bind prepared source identity to the repository-owned study protocol."""
    return {
        "path": STUDY_INPUT_RELATIVE_PATH,
        "sha256": hashlib.sha256(_checked_study_input()).hexdigest(),
    }


def validate_r1_study_binding(binding, stack):
    """Verify internal consistency and the approved canonical binding digest.

    The trust root is the versioned repository input, never a caller-supplied
    input or the binding's own claimed digest. JSON file formatting is not
    part of binding identity; archive manifests independently seal file bytes.
    """
    study_input = _checked_study_input()
    validate_binding(binding, stack)
    study = json.loads(study_input)
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
    if expected != PINNED_REFERENCE_BINDING_SHA256:
        raise ValueError("R1-1 study input differs from the pinned reference digest")
    # validate_binding has already recomputed the canonical payload digest and
    # compared it to this field, so a forged copy of the approved hash fails.
    if binding["sha256"] != expected:
        raise ValueError("R1-1 reference binding is not the approved study reference")


__all__ = ["validate_r1_study_binding", "study_input_identity", "execution_contract_identity",
           "operator_criterion_identity", "additional_failures_identity"]

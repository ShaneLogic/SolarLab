"""Admit only the approved fixed reference to the versioned R1-1 study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from perovskite_sim.experiments.one_dimensional_mechanism_r1 import validate_binding
from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import (
    R1CheckoutError, require_r1_checkout, R1_DECLARATIONS, ADDITIONAL_FAILURES_V2_RELATIVE_PATH,
)


STUDY_INPUT_RELATIVE_PATH = "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
STUDY_INPUT_PATH = Path(__file__).resolve().parents[2] / STUDY_INPUT_RELATIVE_PATH
EXECUTION_CONTRACT_RELATIVE_PATH = "docs/OneDimensionalMechanismR1DynamicsV1.md"
OPERATOR_CRITERION_RELATIVE_PATH = "docs/OneDimensionalMechanismR1OperatorCriterionDecisionV2.md"
ADDITIONAL_FAILURES_RELATIVE_PATH = "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV1.json"
PHYSICS_PROTOCOL_RELATIVE_PATH = "docs/OneDimensionalMechanismR1PhysicsProtocolV1.md"
# A versioned policy pin, not a claim that arbitrary code carrying this value
# is independently approved. Updating the study input requires explicit repin.
PINNED_STUDY_INPUT_SHA256 = "3065a31951d4a90e8b0d03d042e3f2c0349cc7d8e792381a275cb05d58a047ca"
PINNED_REFERENCE_BINDING_SHA256 = "0a6532a436dd07e6b27f01da3fc39aef906e6fd882057f0109c1bc5bdf77e39b"
PINNED_EXECUTION_CONTRACT_SHA256 = R1_DECLARATIONS[EXECUTION_CONTRACT_RELATIVE_PATH][2]
PINNED_OPERATOR_CRITERION_SHA256 = R1_DECLARATIONS[OPERATOR_CRITERION_RELATIVE_PATH][2]
PINNED_ADDITIONAL_FAILURES_SHA256 = "ae2d52cf08a029e0273689d03f16ce8945fdd1ca423582542a49a641d50f9aaa"
PINNED_ADDITIONAL_FAILURES_V2_SHA256 = "8e6f04db47857964e7beb30848ce163bbff1ad411a48d24a6bd12a3210fdf0c2"
PINNED_PHYSICS_PROTOCOL_SHA256 = R1_DECLARATIONS[PHYSICS_PROTOCOL_RELATIVE_PATH][2]
FROZEN_PHYSICS_DECLARATIONS = {path: declaration[2] for path, declaration in R1_DECLARATIONS.items()}


def standard_binding_record(context=None, *, approved_standard_sha256=None):
    """Separate candidate-pin consistency from a caller's standard approval.

    The optional approval digest must come from the caller, never an output
    bundle. Matching it is an identity check, not a signature or a new claim
    that the current numerical result satisfies that standard.
    """
    context = context or require_r1_checkout()
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_standard import physical_standard_binding
    physical_standard, runtime_check = physical_standard_binding(context)
    declarations = {}
    for relative, (role, exported, expected) in R1_DECLARATIONS.items():
        actual = hashlib.sha256(context.read_bytes(relative)).hexdigest()
        if actual != expected:
            raise R1CheckoutError("R1 physics declaration differs from its pinned digest: " + relative)
        declarations[relative] = {"role": role, "exported_name": exported, "sha256": actual}
    standard = {
        "schema": "R1CandidateStandardV1", "declarations": declarations,
        "study_input_sha256": PINNED_STUDY_INPUT_SHA256,
        "fixed_reference_sha256": PINNED_REFERENCE_BINDING_SHA256,
        "physical_standard": physical_standard,
    }
    digest = hashlib.sha256(json.dumps(standard, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if approved_standard_sha256 is not None:
        if (not isinstance(approved_standard_sha256, str) or len(approved_standard_sha256) != 64
                or any(c not in "0123456789abcdef" for c in approved_standard_sha256)):
            raise ValueError("external approved standard digest must be a SHA-256 value")
        if approved_standard_sha256 != digest:
            raise ValueError("candidate standard differs from the caller-approved standard")
    return {
        "schema": "R1StandardBindingV1", "candidate_standard": standard,
        "candidate_standard_sha256": digest, "candidate_internal_consistency": True,
        "runtime_standard_check": runtime_check,
        "approved_standard_sha256": approved_standard_sha256,
        "matches_external_approval": approved_standard_sha256 is not None,
        "source_commit": context.source_commit,
        "scientific_qualification_asserted": False,
        "scope": "candidate_identity_and_optional_caller_approval_match; not_independent_review_or_scientific_acceptance",
    }


def physics_protocol_identity():
    """Keep current authority and superseded declarations unambiguous."""
    context = require_r1_checkout()
    for path, expected in FROZEN_PHYSICS_DECLARATIONS.items():
        if hashlib.sha256(context.read_bytes(path)).hexdigest() != expected:
            raise R1CheckoutError("R1 physics declaration differs from its pinned digest: " + path)
    return {"path": PHYSICS_PROTOCOL_RELATIVE_PATH, "sha256": PINNED_PHYSICS_PROTOCOL_SHA256}


def additional_failures_identity():
    """Pin supplemental observations without rewriting the original study."""
    context = require_r1_checkout()
    raw = context.read_bytes(ADDITIONAL_FAILURES_RELATIVE_PATH)
    if hashlib.sha256(raw).hexdigest() != PINNED_ADDITIONAL_FAILURES_SHA256:
        raise R1CheckoutError("R1 additional failures differ from the pinned registry digest")
    return {"path": ADDITIONAL_FAILURES_RELATIVE_PATH, "sha256": PINNED_ADDITIONAL_FAILURES_SHA256}


def additional_failures_v2_identity():
    context = require_r1_checkout()
    raw = context.read_bytes(ADDITIONAL_FAILURES_V2_RELATIVE_PATH)
    if hashlib.sha256(raw).hexdigest() != PINNED_ADDITIONAL_FAILURES_V2_SHA256:
        raise R1CheckoutError("R1 additional failures V2 differ from the pinned registry digest")
    return {"path": ADDITIONAL_FAILURES_V2_RELATIVE_PATH, "sha256": PINNED_ADDITIONAL_FAILURES_V2_SHA256}


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
           "operator_criterion_identity", "additional_failures_identity", "physics_protocol_identity",
           "standard_binding_record"]

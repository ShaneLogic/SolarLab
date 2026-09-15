"""A preparation's audit metadata must not cross an unanchored formal boundary."""

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_checkout as checkout
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from tests.unit.experiments.test_one_dimensional_mechanism_r1_state import (
    common_state, single_thread_blas,
)


@pytest.fixture
def formal_context(monkeypatch):
    # Explicit unit-test trusted context. This is not a controlled execution
    # claim: real subprocess admission is exercised separately.
    context = checkout.require_r1_checkout()
    context = replace(context, run_class="formal", source_commit=context.commit)
    monkeypatch.setattr(checkout, "_CURRENT_CONTEXT", context)
    return context


def reseal(record):
    record["sha256"] = common.digest({k: v for k, v in record.items() if k != "sha256"})
    return common.R1PreparedState.from_dict(record)


def invoke(entrypoint, case, prepared, **kwargs):
    stack, binding, _ = case
    if entrypoint == "restore":
        return common.restore_common_state(prepared, stack, 4, binding, **kwargs)
    if entrypoint == "zero":
        return protocol.check_zero_excitation(stack, 4, binding, prepared, **kwargs)
    return protocol.run_r1_step(stack, 4, binding, prepared, **kwargs)


def test_execution_class_changes_source_identity(common_state, formal_context):
    prior = common_state[2].to_dict()["source"]
    active = common.execution_source()
    assert prior["run_class"] == "development"
    assert active["run_class"] == "formal"
    assert active["source_commit"] == formal_context.commit
    assert active["files"] == prior["files"]
    assert active["study_input"] == prior["study_input"]
    assert active["sha256"] != prior["sha256"]


@pytest.mark.parametrize("entrypoint", ["restore", "zero", "step"])
@pytest.mark.parametrize("with_anchor", [False, True])
def test_development_preparation_never_enters_formal_api(
    common_state, formal_context, entrypoint, with_anchor, monkeypatch,
):
    prepared = common_state[2]
    def forbidden(*args, **kwargs):
        raise AssertionError("untrusted preparation reached physical state work")
    monkeypatch.setattr(common, "build_r1_material", forbidden)
    kwargs = {"expected_prepared_sha256": prepared.sha256} if with_anchor else {}
    expected = "rejects development preparation" if with_anchor else "external preparation digest"
    with pytest.raises((common.R1StateError, protocol.R1RunError), match=expected):
        invoke(entrypoint, common_state, prepared, **kwargs)


@pytest.mark.parametrize("field", ["preparation_checks", "dc_certificate", "environment"])
def test_resealed_metadata_cannot_escape_external_payload_anchor(common_state, formal_context, field):
    record = common_state[2].to_dict()
    record["source"] = common.execution_source()
    original = reseal(record)
    changed = original.to_dict()
    if field == "preparation_checks":
        changed[field]["metrics"]["electron_continuity_A_m2"] = 0.0
    elif field == "dc_certificate":
        changed["dc_state"]["certificate"]["optimizer_message"] = "fabricated audit metadata"
    else:
        changed[field]["python"] = "fabricated environment"
    altered = reseal(changed)
    assert altered.to_dict()["state"] == original.to_dict()["state"]
    assert altered.sha256 != original.sha256
    with pytest.raises(common.R1StateError, match="external preparation digest"):
        invoke("restore", common_state, altered, expected_prepared_sha256=original.sha256)


def test_anchored_same_source_preserves_populations_without_new_dc(common_state, formal_context, monkeypatch):
    record = common_state[2].to_dict()
    record["source"] = common.execution_source()
    prepared = reseal(record)
    def forbidden(*args, **kwargs):
        raise AssertionError("import must not prepare a fresh DC state")
    monkeypatch.setattr(common, "solve_r1_dc", forbidden)
    _, restored = invoke("restore", common_state, prepared, expected_prepared_sha256=prepared.sha256)
    np.testing.assert_array_equal(restored.positive, prepared.to_dict()["state"]["positive_m3"])
    np.testing.assert_array_equal(restored.occupancy, prepared.to_dict()["state"]["occupancy"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), complex(1, float("nan"))])
@pytest.mark.parametrize("field", ["preparation_checks", "dc_certificate", "environment"])
def test_preparation_nonfinite_rejected_before_canonical_hash_with_named_raw_evidence(
    common_state, value, field,
):
    record = common_state[2].to_dict()
    target = record["dc_state"]["certificate"] if field == "dc_certificate" else record[field]
    target["numeric_probe"] = value
    with pytest.raises(common.R1StateError, match="nonfinite common-state evidence") as captured:
        common.R1PreparedState.from_dict(record)
    failed = captured.value.result
    expected_prefix = "dc_state.certificate" if field == "dc_certificate" else field
    assert failed["nonfinite_numeric_paths"] == [expected_prefix + ".numeric_probe" +
                                               (".imag" if isinstance(value, complex) else "")]
    assert failed["raw_preparation"] is record
    assert failed["certified"] is False


def test_prepare_scans_payload_before_hash(common_state, monkeypatch):
    stack, binding, _ = common_state
    original_snapshot = common.snapshot
    def corrupt_snapshot(*args):
        return {**original_snapshot(*args), "audit_probe": np.array([1.0, np.nan])}
    monkeypatch.setattr(common, "snapshot", corrupt_snapshot)
    with pytest.raises(common.R1StateError, match=r"state.audit_probe\[1\]") as captured:
        common.prepare_common_state(stack, 4, binding)
    assert np.isnan(captured.value.result["raw_preparation"]["state"]["audit_probe"][1])


def test_prepare_nonfinite_failed_check_retains_path_and_raw_value(common_state, monkeypatch):
    stack, binding, _ = common_state
    original_checks = common.equilibrium_checks
    def corrupt_checks(*args):
        checks = original_checks(*args)
        checks["metrics"]["electron_continuity_A_m2"] = np.nan
        checks["certified"] = False
        checks["reasons"] = ["electron_continuity_A_m2"]
        return checks
    monkeypatch.setattr(common, "equilibrium_checks", corrupt_checks)
    with pytest.raises(common.R1StateError, match="preparation_checks.metrics.electron_continuity_A_m2") as captured:
        common.prepare_common_state(stack, 4, binding)
    assert np.isnan(captured.value.result["raw_preparation"]["preparation_checks"]["metrics"]["electron_continuity_A_m2"])

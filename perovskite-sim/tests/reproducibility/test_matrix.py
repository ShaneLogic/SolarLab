from __future__ import annotations

import pytest
import yaml

from backend.main import _stack_to_config_dict, stack_from_dict
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.reproducibility import (
    project_root,
    semantic_sha256,
    validate_matrix,
    verify_baseline,
)
from perovskite_sim.scaps_compat.loader import load_scaps_yaml


ROOT = project_root()


def _matrix():
    return yaml.safe_load(
        (ROOT / "reproducibility/config_benchmark_matrix.yaml").read_text()
    )


def test_matrix_covers_and_loads_every_shipped_config():
    report = validate_matrix(ROOT)
    assert report["configs"] == 28
    assert report["resources"] == 18
    assert report["schemas"] == {
        "standard-device-v1": 23,
        "scaps-device-v1": 4,
        "tandem-v1": 1,
    }


def test_p0_patch_and_frozen_files_match_manifest():
    report = verify_baseline("p0-certified-2026-08-01", ROOT)
    assert report["base_commit"] == "c23e5b9beb3c356250ea32dcb09c78dc45ba28ec"
    assert report["checked_files"] == 15
    assert report["worktree_checked_files"] == 0
    assert report["reconstruction_verified"] is True
    assert report["git_tag_created"] is False


def test_benchmark_config_links_are_explicit_and_bidirectional():
    matrix = _matrix()
    reverse = {
        entry["path"]: set(entry["benchmarks"])
        for entry in matrix["configs"]
    }
    for benchmark_id, benchmark in matrix["benchmarks"].items():
        assert benchmark["configs"]
        assert benchmark["node_ids"]
        assert all("::" in node_id for node_id in benchmark["node_ids"])
        assert set(benchmark["configs"]) == {
            path for path, benchmark_ids in reverse.items()
            if benchmark_id in benchmark_ids
        }


def test_partial_status_always_has_physical_benchmark_evidence():
    matrix = _matrix()
    physical_kinds = {"external", "internal", "numerical"}
    for entry in matrix["configs"]:
        if entry["status"] != "partial":
            continue
        kinds = {
            matrix["benchmarks"][benchmark_id]["kind"]
            for benchmark_id in entry["benchmarks"]
        }
        assert kinds.intersection(physical_kinds), entry["path"]


def test_nk_manifest_covers_every_csv_stem_exactly():
    manifest = yaml.safe_load(
        (ROOT / "perovskite_sim/data/nk/manifest.yaml").read_text()
    )
    csv_stems = {
        path.stem for path in (ROOT / "perovskite_sim/data/nk").glob("*.csv")
    }
    assert set(manifest) == csv_stems
    assert manifest["NiOx"]["source"] == "repository-parameterized-fit"
    assert "b3235885ca67befb7540fb19f01b7102d0a12378" in manifest["NiOx"]["reference"]


@pytest.mark.parametrize(
    "relpath",
    [
        entry["path"]
        for entry in _matrix()["configs"]
        if entry["schema"] == "standard-device-v1"
    ],
)
def test_standard_yaml_and_inline_backend_have_identical_semantics(relpath):
    with (ROOT / relpath).open() as stream:
        raw = yaml.safe_load(stream)
    yaml_stack = load_device_from_yaml(str(ROOT / relpath))
    inline_stack = stack_from_dict(raw)
    assert semantic_sha256(inline_stack) == semantic_sha256(yaml_stack)


@pytest.mark.parametrize(
    "relpath",
    [
        entry["path"]
        for entry in _matrix()["configs"]
        if entry["schema"] == "scaps-device-v1"
    ],
)
def test_scaps_to_inline_roundtrip_preserves_solver_semantics(relpath):
    scaps_stack = load_scaps_yaml(str(ROOT / relpath))
    inline_stack = stack_from_dict(_stack_to_config_dict(scaps_stack))
    assert semantic_sha256(inline_stack) == semantic_sha256(scaps_stack)


def test_matrix_does_not_promote_open_gaps_to_certified():
    matrix = _matrix()
    gaps = {
        key for key, value in matrix["benchmarks"].items()
        if value["kind"] == "gap"
    }
    assert gaps == {
        "csi-grid-envelope-gap",
    }
    for entry in matrix["configs"]:
        if gaps.intersection(entry["benchmarks"]):
            assert entry["status"] != "certified"


def test_calibrated_external_reproductions_are_not_certified():
    matrix = _matrix()
    calibrated = {
        benchmark_id: benchmark
        for benchmark_id, benchmark in matrix["benchmarks"].items()
        if benchmark.get("claim_level") == "calibrated_reproduction"
    }
    assert set(calibrated) == {
        "courtier2019-ionmonger",
        "calado2016-driftfusion",
    }
    for benchmark in calibrated.values():
        assert benchmark["evidence_tier"] == "calibration_only"
        assert set(benchmark["protocol"]) == {"local", "source"}
        assert set(benchmark["calibration"]["parameters"]) == {"device.V_bi"}
        assert benchmark["calibration"]["targets"]
        assert benchmark["non_calibration_checks"]
        assert not set(benchmark["calibration"]["targets"]).intersection(
            benchmark["non_calibration_checks"]
        )

    config_by_path = {entry["path"]: entry for entry in matrix["configs"]}
    for benchmark_id, benchmark in calibrated.items():
        for relpath in benchmark["configs"]:
            entry = config_by_path[relpath]
            assert benchmark_id in entry["benchmarks"]
            assert entry["status"] == "partial"


def test_every_p1_gap_has_a_reproduction_and_acceptance_contract():
    data = yaml.safe_load((ROOT / "reproducibility/p1_gaps.yaml").read_text())
    gaps = {gap["id"]: gap for gap in data["gaps"]}
    assert set(gaps) == {
        "cigs-2um-graded-notch",
        "external-solver-curve-crosscheck",
        "ionmonger-residual-ss-96",
        "scaps-low-doping-etl",
        "lin2019-tandem-jsc-pce",
        "csi-qf-jv-grid-convergence",
        "csi-transient-jv-grid-envelope",
        "csi-mott-schottky-convergence",
    }
    assert gaps["scaps-low-doping-etl"]["status"] == "closed"
    assert gaps["scaps-low-doping-etl"]["resolution"]
    assert gaps["ionmonger-residual-ss-96"]["status"] == "closed"
    assert gaps["ionmonger-residual-ss-96"]["resolution"]
    assert gaps["cigs-2um-graded-notch"]["status"] == "closed"
    assert gaps["cigs-2um-graded-notch"]["resolution"]
    assert gaps["csi-qf-jv-grid-convergence"]["status"] == "closed"
    assert gaps["csi-qf-jv-grid-convergence"]["resolution"]
    assert gaps["csi-mott-schottky-convergence"]["status"] == "open"
    assert gaps["csi-mott-schottky-convergence"]["next_experiment"]
    for gap in gaps.values():
        assert gap["status"] in {"open", "closed"}
        assert gap["reproduction"]
        assert len(gap["acceptance"]) >= 3

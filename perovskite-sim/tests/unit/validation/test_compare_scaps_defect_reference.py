"""Pre-registered SCAPS-vs-SolarLab defect comparison verdict contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
S_SUITE = "reproducibility/scaps_defect_s0_s2_suite.json"
M_SUITE = "reproducibility/scaps_multivalent_defect_suite.json"
THRESHOLDS = ROOT / "reproducibility/scaps_defect_comparison_thresholds.json"
POSITIONS_UM = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def _load_module():
    path = ROOT / "scripts/compare_scaps_defect_reference.py"
    spec = importlib.util.spec_from_file_location(
        "compare_scaps_defect_reference_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_module()


def _suite_scenarios(suite_relative: str) -> dict[str, str]:
    suite = json.loads((ROOT / suite_relative).read_text(encoding="utf-8"))
    return {row["id"]: row["config_sha256"] for row in suite["scenarios"]}


def _reference_payload(module, mode: str) -> dict:
    suite_relative = S_SUITE if mode == "s0s2" else M_SUITE
    schema = (
        "solarlab.scaps_explicit_defect_reference"
        if mode == "s0s2"
        else "solarlab.scaps_multivalent_defect_reference"
    )
    identifiers = ("S0", "S1", "S2") if mode == "s0s2" else ("M1", "M2", "M3")
    hashes = _suite_scenarios(suite_relative)
    intervals = json.loads(THRESHOLDS.read_text())["grid"][mode][
        "intervals_per_layer"
    ]
    scenarios = {}
    for identifier in identifiers:
        profile = module.compute_solarlab_profile(
            ROOT,
            mode=mode,
            identifier=identifier,
            positions_um=POSITIONS_UM,
            intervals=intervals,
        )
        scenarios[identifier] = {
            "canonical_config_sha256": hashes[identifier],
            "profile": profile,
        }
    unsigned = {
        "schema": schema,
        "schema_version": "1.0",
        "scenarios": scenarios,
        "suite": {
            "path": suite_relative,
            "sha256": hashlib.sha256(
                (ROOT / suite_relative).read_bytes()
            ).hexdigest(),
        },
    }
    return {**unsigned, "reference_content_sha256": module._content_sha256(unsigned)}


def _reseal(module, payload: dict) -> dict:
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "reference_content_sha256"
    }
    return {**unsigned, "reference_content_sha256": module._content_sha256(unsigned)}


@pytest.fixture(scope="module")
def s_reference(module) -> dict:
    return _reference_payload(module, "s0s2")


@pytest.fixture(scope="module")
def m_reference(module) -> dict:
    return _reference_payload(module, "multivalent")


def _run(module, tmp_path: Path, payload: dict, extra: list[str] | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "report.json"
    argv = [
        "--project-root",
        str(ROOT),
        "--reference",
        str(reference),
        "--out",
        str(out),
    ] + (extra or [])
    code = module.main(argv)
    return code, json.loads(out.read_text(encoding="ascii"))


def test_self_comparison_passes_and_seals_the_report(module, s_reference, tmp_path):
    code, report = _run(module, tmp_path, s_reference)
    assert code == 0
    assert report["schema"] == "solarlab.scaps_defect_comparison_report"
    assert report["mode"] == "s0s2"
    assert report["overall_verdict"] == "PASS"
    assert report["preregistered_settings"] is True
    assert report["thresholds"]["sha256"] == hashlib.sha256(
        THRESHOLDS.read_bytes()
    ).hexdigest()
    unsigned = dict(report)
    digest = unsigned.pop("comparison_content_sha256")
    assert digest == module._content_sha256(unsigned)
    for identifier in ("S0", "S1", "S2"):
        columns = report["scenarios"][identifier]["columns"]
        assert all(item["verdict"] == "PASS" for item in columns.values())
        assert set(columns) == set(
            json.loads(THRESHOLDS.read_text())["columns"]["s0s2"]
        )
    # A verdict artifact is frozen evidence: never overwritten in place.
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _run(module, tmp_path, s_reference)


def test_multivalent_self_comparison_passes(module, m_reference, tmp_path):
    code, report = _run(module, tmp_path, m_reference)
    assert code == 0
    assert report["mode"] == "multivalent"
    assert report["overall_verdict"] == "PASS"
    for identifier in ("M1", "M2", "M3"):
        columns = report["scenarios"][identifier]["columns"]
        assert all(item["verdict"] == "PASS" for item in columns.values())


def test_reference_zero_point_shift_does_not_fail_anchored_columns(
    module, s_reference, tmp_path
):
    """SCAPS may use any potential/energy zero; anchoring must absorb it."""
    shifted = json.loads(json.dumps(s_reference))
    for scenario in shifted["scenarios"].values():
        profile = scenario["profile"]
        profile["electrostatic_potential_V"] = [
            value + 0.7 for value in profile["electrostatic_potential_V"]
        ]
        profile["conduction_band_eV"] = [
            value - 0.7 for value in profile["conduction_band_eV"]
        ]
        profile["valence_band_eV"] = [
            value - 0.7 for value in profile["valence_band_eV"]
        ]
    code, report = _run(module, tmp_path, _reseal(module, shifted))
    assert code == 0
    assert report["overall_verdict"] == "PASS"


def test_perturbed_density_fails_the_density_column(module, s_reference, tmp_path):
    perturbed = json.loads(json.dumps(s_reference))
    profile = perturbed["scenarios"]["S1"]["profile"]
    profile["electron_density_cm3"] = [
        value * 1.10 for value in profile["electron_density_cm3"]
    ]
    code, report = _run(module, tmp_path, _reseal(module, perturbed))
    assert code == 0
    columns = report["scenarios"]["S1"]["columns"]
    assert columns["electron_density_log10_linf"]["verdict"] == "FAIL"
    assert columns["hole_density_log10_linf"]["verdict"] == "PASS"
    assert report["overall_verdict"] == "FAIL"


def test_perturbed_state_fractions_fail_only_the_fraction_column(
    module, m_reference, tmp_path
):
    perturbed = json.loads(json.dumps(m_reference))
    profile = perturbed["scenarios"]["M3"]["profile"]
    profile["charge_state_occupation_fractions"] = [
        [row[1], row[0], row[2]]
        for row in profile["charge_state_occupation_fractions"]
    ]
    code, report = _run(module, tmp_path, _reseal(module, perturbed))
    assert code == 0
    columns = report["scenarios"]["M3"]["columns"]
    assert columns["charge_state_occupation_fraction_linf"]["verdict"] == "FAIL"
    # The exported net-charge column is compared on its own evidence, not
    # recomputed from the fractions.
    assert columns["normalized_defect_charge_linf"]["verdict"] == "PASS"
    assert report["overall_verdict"] == "FAIL"


def test_nonzero_recombination_fails_the_near_zero_gate(module, s_reference, tmp_path):
    perturbed = json.loads(json.dumps(s_reference))
    profile = perturbed["scenarios"]["S2"]["profile"]
    profile["recombination_rate_cm3_s"] = [1.0e9 for _ in POSITIONS_UM]
    code, report = _run(module, tmp_path, _reseal(module, perturbed))
    assert code == 0
    columns = report["scenarios"]["S2"]["columns"]
    assert columns["recombination_rate_near_zero_cm3_s"]["verdict"] == "FAIL"
    assert report["overall_verdict"] == "FAIL"


def test_tampered_content_hash_or_suite_drift_is_rejected(
    module, s_reference, tmp_path
):
    tampered = json.loads(json.dumps(s_reference))
    tampered["reference_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reference content hash"):
        _run(module, tmp_path / "tampered", tampered)

    drifted = json.loads(json.dumps(s_reference))
    drifted["suite"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="suite hash"):
        _run(module, tmp_path / "drifted", _reseal(module, drifted))


def test_grid_override_is_reported_as_not_preregistered(
    module, s_reference, tmp_path
):
    code, report = _run(
        module,
        tmp_path,
        s_reference,
        extra=["--grid-intervals", "32", "--sensitivity-grid-intervals", "16"],
    )
    assert code == 0
    assert report["preregistered_settings"] is False
    assert report["grid"]["intervals_per_layer"] == 32

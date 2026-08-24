from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("scripts/run_interface_srh_identifiability.py")
    spec = importlib.util.spec_from_file_location(
        "run_interface_srh_identifiability_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_rank_deficient_evidence_without_false_claim(tmp_path, capsys):
    module = _load_module()
    output = tmp_path / "rank-deficient.json"

    exit_code = module.main(["--out", str(output), "--carrier-condition-count", "5"])
    payload = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["analysis_type"] == "synthetic_interface_srh_identifiability"
    assert payload["result"]["analysis_certified"] is True
    assert payload["result"]["parameters_identifiable"] is False
    assert payload["result"]["numerical_rank"] == 2
    assert summary["parameters_identifiable"] is False


def test_cli_known_capture_scale_recovers_full_rank_truth(tmp_path):
    module = _load_module()
    output = tmp_path / "full-rank.json"

    exit_code = module.main(
        [
            "--out",
            str(output),
            "--carrier-condition-count",
            "5",
            "--estimated-parameters",
            "trap_density_cm2",
            "calibration_factor",
        ]
    )
    result = json.loads(output.read_text(encoding="utf-8"))["result"]

    assert exit_code == 0
    assert result["analysis_certified"] is True
    assert result["parameters_identifiable"] is True
    assert result["truth_recovered"] is True
    assert result["numerical_rank"] == 2


def test_cli_strict_protocol_round_trip_and_override_rejection(tmp_path):
    module = _load_module()
    protocol = module.build_interface_srh_identifiability_protocol(
        carrier_condition_count=5
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    assert module.main(["--out", str(output), "--protocol", str(protocol_path)]) == 0
    assert (
        json.loads(output.read_text(encoding="utf-8"))["result"]["protocol_sha256"]
        == protocol.sha256
    )

    conflicting = tmp_path / "conflicting.json"
    assert (
        module.main(
            [
                "--out",
                str(conflicting),
                "--protocol",
                str(protocol_path),
                "--noise-seed",
                "3",
            ]
        )
        == 4
    )
    assert not conflicting.exists()


def test_cli_rejects_unknown_protocol_fields_without_writing(tmp_path):
    module = _load_module()
    payload = module.build_interface_srh_identifiability_protocol(
        carrier_condition_count=5
    ).to_dict()
    payload["claim"] = "material parameters identified"
    protocol_path = tmp_path / "invalid.json"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "not-written.json"

    assert module.main(["--out", str(output), "--protocol", str(protocol_path)]) == 4
    assert not output.exists()

"""Contracts for content-addressed dense SCAPS CBO reference imports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path("scripts/import_scaps_cbo_reference.py")
    spec = importlib.util.spec_from_file_location(
        "import_scaps_cbo_reference_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict:
    return {
        "schema": "solarlab.scaps_cbo_parameter_manifest",
        "schema_version": "1.0",
        "layers": [{"name": "PVK"}, {"name": "ETL"}],
        "contacts": {"left": "flat-band", "right": "flat-band"},
        "interfaces": [{"name": "PVK/ETL"}],
        "illumination": {"spectrum": "AM1.5G"},
        "numerics": {"mesh": "SCAPS default"},
        "cbo_scan": {
            "delta_ec_convention": "chi_absorber - chi_etl",
            "swept_parameter": "etl_electron_affinity",
            "boundary_policy": "fixed_contacts",
        },
    }


def test_importer_hashes_raw_export_deck_and_parameter_manifest(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "scaps-cbo.csv"
    csv_path.write_text(
        "delta_ec_eV,Jsc_mA_cm2,Voc_V,FF_percent,PCE_percent\n"
        "0.0,26.28,1.24,90.4,29.65\n"
        "0.40,26.28,1.24,65.1,21.38\n"
        "0.41,20.00,1.23,55.0,13.53\n",
        encoding="utf-8",
    )
    deck = tmp_path / "device.def"
    deck.write_bytes(b"raw-scaps-deck")
    manifest = tmp_path / "parameters.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    output = tmp_path / "reference.json"

    exit_code = module.main(
        [
            "--csv",
            str(csv_path),
            "--source-deck",
            str(deck),
            "--parameter-manifest",
            str(manifest),
            "--out",
            str(output),
            "--solver-version",
            "3.3.11",
            "--extracted-at",
            "2026-08-10",
            "--confirm-independent-scaps-export",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "solarlab.scaps_cbo_reference"
    assert payload["sweeps"]["CHI_ETL"]["n_points"] == 3
    protocol = payload["cbo_validation"]
    assert protocol["independently_generated"] is True
    assert protocol["interpolated"] is False
    assert protocol["source_export_sha256"] == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    assert protocol["source_deck_sha256"] == hashlib.sha256(
        deck.read_bytes()
    ).hexdigest()
    assert protocol["parameter_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()


def test_importer_rejects_reordered_or_unattested_input(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "scaps-cbo.csv"
    csv_path.write_text(
        "delta_ec_eV,Jsc_mA_cm2\n"
        "0.0,26.28\n"
        "0.5,2.45\n"
        "0.4,26.28\n",
        encoding="utf-8",
    )
    deck = tmp_path / "device.def"
    deck.write_bytes(b"raw-scaps-deck")
    manifest = tmp_path / "parameters.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    common = [
        "--csv",
        str(csv_path),
        "--source-deck",
        str(deck),
        "--parameter-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "reference.json"),
        "--solver-version",
        "3.3.11",
        "--extracted-at",
        "2026-08-10",
    ]

    with pytest.raises(ValueError, match="confirm-independent"):
        module.main(common)
    with pytest.raises(ValueError, match="strictly increasing"):
        module.main(common + ["--confirm-independent-scaps-export"])


def test_importer_rejects_unpopulated_manifest_sections(tmp_path):
    module = _load_module()
    manifest = _manifest()
    manifest["layers"] = []
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must be populated: layers"):
        module._load_manifest(path)

"""Fail-closed contracts for SCAPS S0-S2 direct profile imports."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    path = ROOT / "scripts/import_scaps_defect_reference.py"
    spec = importlib.util.spec_from_file_location(
        "import_scaps_defect_reference_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _suite() -> dict:
    return json.loads(
        (ROOT / "reproducibility/scaps_defect_s0_s2_suite.json").read_text()
    )


def _parameter_manifest() -> dict:
    scenarios = {
        row["id"]: {
            "canonical_config_sha256": row["config_sha256"],
            "charge_transition": row["charge_transition"],
            "doping_polarity": row["doping_polarity"],
            "scaps_parameters": {
                "defect_density_cm3": 2.0e15,
                "defect_energy_eV_above_vb": 0.39,
                "sigma_n_cm2": 2.0e-15,
                "sigma_p_cm2": 7.0e-16,
            },
            "source_deck_format": "SCAPS def",
            "thickness_um": 0.3,
        }
        for row in _suite()["scenarios"]
    }
    return {
        "comparison_protocol": {
            "interpolation_allowed": False,
            "operating_point": "dark_equilibrium_zero_bias",
            "position_tolerance_um": 1.0e-9,
            "row_policy": "direct_export_rows_only",
        },
        "numerics": {
            "convergence_settings": "recorded in each source deck",
            "mesh_policy": "SCAPS direct export mesh",
        },
        "scenarios": scenarios,
        "schema": "solarlab.scaps_explicit_defect_parameter_manifest",
        "schema_version": "1.0",
        "sign_conventions": {
            "defect_charge": "negative_acceptor_positive_donor",
            "electrostatic_potential": "as_exported_by_SCAPS",
            "position_origin": "left_contact",
            "recombination_rate": "positive_net_recombination",
        },
        "solver": {"name": "SCAPS-1D", "version": "3.3.11"},
        "unit_conventions": {
            "conduction_band_eV": "eV",
            "defect_charge_number_cm3": "cm-3",
            "defect_occupancy": "1",
            "electron_density_cm3": "cm-3",
            "electrostatic_potential_V": "V",
            "hole_density_cm3": "cm-3",
            "position_um": "um",
            "recombination_rate_cm3_s": "cm-3 s-1",
            "valence_band_eV": "eV",
        },
    }


def _profile_text(identifier: str) -> str:
    charge = {"S0": 0.0, "S1": -1.9e15, "S2": 1.8e15}[identifier]
    occupancy = {"S0": 0.4, "S1": 0.95, "S2": 0.10}[identifier]
    header = (
        "position_um,electron_density_cm3,hole_density_cm3,"
        "electrostatic_potential_V,conduction_band_eV,valence_band_eV,"
        "defect_occupancy,defect_charge_number_cm3,"
        "recombination_rate_cm3_s\n"
    )
    rows = "".join(
        f"{position},5e15,2e10,{potential},-4.0,-4.8,"
        f"{occupancy},{charge},0.0\n"
        for position, potential in ((0.0, 0.0), (0.15, 0.1), (0.3, 0.2))
    )
    return header + rows


def _fixture_files(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "parameters.json"
    manifest.write_text(json.dumps(_parameter_manifest()), encoding="utf-8")
    paths = {"manifest": manifest}
    for identifier in ("S0", "S1", "S2"):
        csv_path = tmp_path / f"{identifier}.csv"
        csv_path.write_text(_profile_text(identifier), encoding="utf-8")
        deck = tmp_path / f"{identifier}.def"
        deck.write_bytes(f"raw SCAPS deck {identifier}".encode("ascii"))
        paths[f"{identifier}_csv"] = csv_path
        paths[f"{identifier}_deck"] = deck
    return paths


def _arguments(tmp_path: Path, files: dict[str, Path]) -> list[str]:
    arguments = [
        "--project-root",
        str(ROOT),
        "--suite",
        str(ROOT / "reproducibility/scaps_defect_s0_s2_suite.json"),
        "--parameter-manifest",
        str(files["manifest"]),
    ]
    for identifier in ("S0", "S1", "S2"):
        flag = identifier.lower()
        arguments.extend(
            [
                f"--{flag}-csv",
                str(files[f"{identifier}_csv"]),
                f"--{flag}-source-deck",
                str(files[f"{identifier}_deck"]),
            ]
        )
    return arguments + [
        "--out",
        str(tmp_path / "reference.json"),
        "--solver-version",
        "3.3.11",
        "--extracted-at",
        "2026-08-28",
        "--operator",
        "independent-scaps-operator",
    ]


def test_importer_hashes_all_raw_exports_decks_manifest_and_suite(tmp_path):
    module = _load_module()
    files = _fixture_files(tmp_path)
    output = tmp_path / "reference.json"
    arguments = _arguments(tmp_path, files) + [
        "--confirm-independent-scaps-export",
        "--confirm-direct-unmodified-rows",
    ]

    assert module.main(arguments) == 0

    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["schema"] == "solarlab.scaps_explicit_defect_reference"
    assert payload["attestation"] == {
        "direct_unmodified_rows": True,
        "independent_scaps_export": True,
        "operator": "independent-scaps-operator",
    }
    unsigned = dict(payload)
    digest = unsigned.pop("reference_content_sha256")
    assert digest == module._content_sha256(unsigned)
    assert payload["suite"]["sha256"] == hashlib.sha256(
        (ROOT / "reproducibility/scaps_defect_s0_s2_suite.json").read_bytes()
    ).hexdigest()
    for identifier in ("S0", "S1", "S2"):
        scenario = payload["scenarios"][identifier]
        assert scenario["row_count"] == 3
        assert scenario["profile"]["position_um"] == [0.0, 0.15, 0.3]
        assert scenario["source_export"]["sha256"] == hashlib.sha256(
            files[f"{identifier}_csv"].read_bytes()
        ).hexdigest()
        assert scenario["source_deck"]["sha256"] == hashlib.sha256(
            files[f"{identifier}_deck"].read_bytes()
        ).hexdigest()


def test_importer_requires_both_attestations_and_never_writes_partial(tmp_path):
    module = _load_module()
    files = _fixture_files(tmp_path)
    arguments = _arguments(tmp_path, files)
    output = tmp_path / "reference.json"

    with pytest.raises(ValueError, match="independent-scaps-export"):
        module.main(arguments)
    assert not output.exists()

    with pytest.raises(ValueError, match="direct-unmodified-rows"):
        module.main(arguments + ["--confirm-independent-scaps-export"])
    assert not output.exists()


def test_importer_rejects_reordered_rows_wrong_charge_and_unknown_columns(tmp_path):
    module = _load_module()
    flags = [
        "--confirm-independent-scaps-export",
        "--confirm-direct-unmodified-rows",
    ]

    reordered = _fixture_files(tmp_path / "reordered")
    s0_path = reordered["S0_csv"]
    rows = s0_path.read_text().splitlines()
    s0_path.write_text("\n".join((rows[0], rows[1], rows[3], rows[2])) + "\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        module.main(_arguments(tmp_path / "reordered", reordered) + flags)

    wrong_sign = _fixture_files(tmp_path / "wrong-sign")
    s1_path = wrong_sign["S1_csv"]
    s1_path.write_text(s1_path.read_text().replace("-1900000000000000.0", "1e15"))
    with pytest.raises(ValueError, match="S1 acceptor defect charge"):
        module.main(_arguments(tmp_path / "wrong-sign", wrong_sign) + flags)

    unknown = _fixture_files(tmp_path / "unknown-column")
    s2_path = unknown["S2_csv"]
    lines = s2_path.read_text().splitlines()
    s2_path.write_text(
        "\n".join([lines[0] + ",smoothed", *(line + ",0" for line in lines[1:])])
        + "\n"
    )
    with pytest.raises(ValueError, match="columns must exactly match"):
        module.main(_arguments(tmp_path / "unknown-column", unknown) + flags)


def test_importer_rejects_manifest_identity_drift_and_output_overwrite(tmp_path):
    module = _load_module()
    flags = [
        "--confirm-independent-scaps-export",
        "--confirm-direct-unmodified-rows",
    ]
    files = _fixture_files(tmp_path)
    manifest = json.loads(files["manifest"].read_text())
    manifest["scenarios"]["S1"]["canonical_config_sha256"] = "0" * 64
    files["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="S1 config hash mismatch"):
        module.main(_arguments(tmp_path, files) + flags)

    valid_dir = tmp_path / "valid"
    valid = _fixture_files(valid_dir)
    arguments = _arguments(valid_dir, valid) + flags
    assert module.main(arguments) == 0
    with pytest.raises(ValueError, match="refusing to overwrite"):
        module.main(arguments)

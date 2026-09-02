"""Fail-closed contracts for SCAPS M1-M3 multivalent profile imports."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SUITE_PATH = ROOT / "reproducibility/scaps_multivalent_defect_suite.json"

_TOTAL_DENSITY_CM3 = 2.0e15
_CHARGE_STATES = {
    "M1": (2, 1, 0),
    "M2": (0, -1, -2),
    "M3": (1, 0, -1),
}
_FRACTION_ROWS = {
    "M1": ("0.90|0.08|0.02", "0.90|0.08|0.02", "0.90|0.08|0.02"),
    "M2": ("0.02|0.08|0.90", "0.02|0.08|0.90", "0.02|0.08|0.90"),
    # M3 net charge legitimately changes sign across the slab.
    "M3": ("0.60|0.30|0.10", "0.33|0.34|0.33", "0.10|0.30|0.60"),
}


def _load_module():
    path = ROOT / "scripts/import_scaps_multivalent_defect_reference.py"
    spec = importlib.util.spec_from_file_location(
        "import_scaps_multivalent_defect_reference_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _suite() -> dict:
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def _net_charge(identifier: str, fraction_cell: str) -> float:
    fractions = [float(part) for part in fraction_cell.split("|")]
    states = _CHARGE_STATES[identifier]
    return _TOTAL_DENSITY_CM3 * sum(
        charge * fraction for charge, fraction in zip(states, fractions)
    )


def _parameter_manifest() -> dict:
    family_rows = {
        row["id"]: (row["family"], row["doping_polarity"], row["config_sha256"])
        for row in _suite()["scenarios"]
    }
    degeneracy = {
        "M1": ("scaps_binomial", [1.0, 2.0, 1.0]),
        "M2": ("scaps_binomial", [1.0, 2.0, 1.0]),
        "M3": ("unity", [1.0, 1.0, 1.0]),
    }
    scenarios = {}
    for identifier, (family, doping, sha) in family_rows.items():
        convention, degeneracies = degeneracy[identifier]
        scenarios[identifier] = {
            "canonical_config_sha256": sha,
            "charge_states_e": list(_CHARGE_STATES[identifier]),
            "degeneracy_convention": convention,
            "doping_polarity": doping,
            "energy_reference": "above_valence_band",
            "family": family,
            "scaps_parameters": {
                "temperature_K": 300.0,
                "total_density_cm3": _TOTAL_DENSITY_CM3,
            },
            "source_deck_format": "SCAPS-1D .def",
            "state_degeneracies": degeneracies,
            "thickness_um": 0.3,
            "total_defect_density_cm3": _TOTAL_DENSITY_CM3,
            "transition_capture_cross_sections_cm2": [
                {"sigma_n_cm2": 2.0e-15, "sigma_p_cm2": 7.0e-16},
                {"sigma_n_cm2": 1.0e-15, "sigma_p_cm2": 5.0e-16},
            ],
            "transition_energies_eV_above_vb": [0.30, 0.45],
        }
    return {
        "comparison_protocol": {
            "charge_state_order": "most_positive_first",
            "interpolation_allowed": False,
            "net_charge_consistency_tolerance_relative_to_total_density": 1.0e-3,
            "occupation_fraction_separator": "|",
            "occupation_fraction_sum_tolerance": 1.0e-4,
            "operating_point": "dark_equilibrium_zero_bias",
            "position_tolerance_um": 1.0e-9,
            "row_policy": "direct_export_rows_only",
        },
        "numerics": {
            "convergence_settings": "recorded in each source deck",
            "mesh_policy": "SCAPS direct export mesh",
            "recalculate_mesh": "on",
        },
        "scenarios": scenarios,
        "schema": "solarlab.scaps_multivalent_defect_parameter_manifest",
        "schema_version": "1.0",
        "sign_conventions": {
            "defect_charge": "state_weighted_net_charge",
            "electrostatic_potential": "as_exported_by_SCAPS",
            "position_origin": "left_contact",
            "recombination_rate": "positive_net_recombination",
        },
        "solver": {"name": "SCAPS-1D", "version": "3.3.11"},
        "unit_conventions": {
            "charge_state_occupation_fraction_per_state": "1",
            "conduction_band_eV": "eV",
            "defect_charge_number_cm3": "cm-3",
            "electron_density_cm3": "cm-3",
            "electrostatic_potential_V": "V",
            "hole_density_cm3": "cm-3",
            "position_um": "um",
            "recombination_rate_cm3_s": "cm-3 s-1",
            "valence_band_eV": "eV",
        },
    }


def _profile_text(identifier: str) -> str:
    header = (
        "position_um,electron_density_cm3,hole_density_cm3,"
        "electrostatic_potential_V,conduction_band_eV,valence_band_eV,"
        "defect_charge_number_cm3,recombination_rate_cm3_s,"
        "charge_state_occupation_fraction_per_state\n"
    )
    rows = ""
    for (position, potential), fraction_cell in zip(
        ((0.0, 0.0), (0.15, 0.1), (0.3, 0.2)),
        _FRACTION_ROWS[identifier],
    ):
        charge = _net_charge(identifier, fraction_cell)
        rows += (
            f"{position},5e15,2e10,{potential},-4.0,-4.8,"
            f"{charge!r},0.0,{fraction_cell}\n"
        )
    return header + rows


def _fixture_files(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "parameters.json"
    manifest.write_text(json.dumps(_parameter_manifest()), encoding="utf-8")
    paths = {"manifest": manifest}
    for identifier in ("M1", "M2", "M3"):
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
        str(SUITE_PATH),
        "--parameter-manifest",
        str(files["manifest"]),
    ]
    for identifier in ("M1", "M2", "M3"):
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
        "2026-09-02",
        "--operator",
        "independent-scaps-operator",
    ]


_FLAGS = [
    "--confirm-independent-scaps-export",
    "--confirm-direct-unmodified-rows",
]


def test_importer_hashes_everything_and_parses_state_fractions(tmp_path):
    module = _load_module()
    files = _fixture_files(tmp_path)
    output = tmp_path / "reference.json"

    assert module.main(_arguments(tmp_path, files) + _FLAGS) == 0

    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["schema"] == "solarlab.scaps_multivalent_defect_reference"
    assert payload["attestation"] == {
        "direct_unmodified_rows": True,
        "independent_scaps_export": True,
        "operator": "independent-scaps-operator",
    }
    unsigned = dict(payload)
    digest = unsigned.pop("reference_content_sha256")
    assert digest == module._content_sha256(unsigned)
    assert payload["suite"]["sha256"] == hashlib.sha256(
        SUITE_PATH.read_bytes()
    ).hexdigest()
    for identifier in ("M1", "M2", "M3"):
        scenario = payload["scenarios"][identifier]
        assert scenario["family"] == {
            "M1": "double_donor",
            "M2": "double_acceptor",
            "M3": "amphoteric",
        }[identifier]
        assert scenario["charge_states_e"] == list(_CHARGE_STATES[identifier])
        assert scenario["row_count"] == 3
        assert scenario["profile"]["position_um"] == [0.0, 0.15, 0.3]
        # The raw cell is preserved verbatim AND parsed into three fractions.
        assert scenario["profile"][
            "charge_state_occupation_fraction_per_state"
        ] == list(_FRACTION_ROWS[identifier])
        parsed = scenario["profile"]["charge_state_occupation_fractions"]
        assert len(parsed) == 3
        for row_fractions, cell in zip(parsed, _FRACTION_ROWS[identifier]):
            assert row_fractions == [float(part) for part in cell.split("|")]
        assert scenario["source_export"]["sha256"] == hashlib.sha256(
            files[f"{identifier}_csv"].read_bytes()
        ).hexdigest()
        assert scenario["source_deck"]["sha256"] == hashlib.sha256(
            files[f"{identifier}_deck"].read_bytes()
        ).hexdigest()
    # M3's net charge changes sign across the slab and must be accepted.
    m3_charge = payload["scenarios"]["M3"]["profile"]["defect_charge_number_cm3"]
    assert m3_charge[0] > 0.0 and m3_charge[-1] < 0.0


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


def test_importer_enforces_per_family_net_charge_sign_rules(tmp_path):
    module = _load_module()

    # M1 (double donor) must never carry net negative defect charge.
    m1_negative = _fixture_files(tmp_path / "m1-negative")
    path = m1_negative["M1_csv"]
    lines = path.read_text().splitlines()
    fields = lines[2].split(",")
    fields[6] = "-1e13"
    lines[2] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="M1 double_donor net defect charge"):
        module.main(_arguments(tmp_path / "m1-negative", m1_negative) + _FLAGS)

    # M2 (double acceptor) must never carry net positive defect charge.
    m2_positive = _fixture_files(tmp_path / "m2-positive")
    path = m2_positive["M2_csv"]
    lines = path.read_text().splitlines()
    fields = lines[2].split(",")
    fields[6] = "1e13"
    lines[2] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="M2 double_acceptor net defect charge"):
        module.main(_arguments(tmp_path / "m2-positive", m2_positive) + _FLAGS)


def test_importer_rejects_malformed_or_inconsistent_state_fractions(tmp_path):
    module = _load_module()

    def _patch_m1_cell(sub_path: str, cell: str, *, fix_charge: bool) -> Path:
        files = _fixture_files(tmp_path / sub_path)
        path = files["M1_csv"]
        lines = path.read_text().splitlines()
        fields = lines[1].split(",")
        fields[8] = cell
        if fix_charge:
            fields[6] = repr(_net_charge("M1", cell))
        lines[1] = ",".join(fields)
        path.write_text("\n".join(lines) + "\n")
        return files

    two_parts = _patch_m1_cell("two-parts", "0.90|0.10", fix_charge=False)
    with pytest.raises(ValueError, match="exactly three"):
        module.main(_arguments(tmp_path / "two-parts", two_parts) + _FLAGS)

    bad_sum = _patch_m1_cell("bad-sum", "0.50|0.08|0.02", fix_charge=True)
    with pytest.raises(ValueError, match="must sum to 1"):
        module.main(_arguments(tmp_path / "bad-sum", bad_sum) + _FLAGS)

    out_of_range = _patch_m1_cell("out-of-range", "1.20|-0.22|0.02", fix_charge=False)
    with pytest.raises(ValueError, match="lie in \\[0, 1\\]"):
        module.main(_arguments(tmp_path / "out-of-range", out_of_range) + _FLAGS)

    # Exported net charge must agree with Nt * sum(q_s * P_s).
    inconsistent = _fixture_files(tmp_path / "inconsistent")
    path = inconsistent["M1_csv"]
    lines = path.read_text().splitlines()
    fields = lines[1].split(",")
    fields[6] = repr(_net_charge("M1", "0.90|0.08|0.02") * 1.5)
    lines[1] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="inconsistent with the state fractions"):
        module.main(_arguments(tmp_path / "inconsistent", inconsistent) + _FLAGS)


def test_importer_rejects_column_drift_and_config_drift_and_overwrite(tmp_path):
    module = _load_module()

    swapped = _fixture_files(tmp_path / "swapped")
    path = swapped["M2_csv"]
    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    header[0], header[1] = header[1], header[0]
    lines[0] = ",".join(header)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="columns must exactly match"):
        module.main(_arguments(tmp_path / "swapped", swapped) + _FLAGS)

    drift = _fixture_files(tmp_path / "drift")
    manifest = json.loads(drift["manifest"].read_text())
    manifest["scenarios"]["M2"]["canonical_config_sha256"] = "0" * 64
    drift["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="M2 config hash mismatch"):
        module.main(_arguments(tmp_path / "drift", drift) + _FLAGS)

    valid_dir = tmp_path / "valid"
    valid = _fixture_files(valid_dir)
    arguments = _arguments(valid_dir, valid) + _FLAGS
    assert module.main(arguments) == 0
    with pytest.raises(ValueError, match="refusing to overwrite"):
        module.main(arguments)


def test_importer_rejects_degeneracy_and_kinetics_contract_violations(tmp_path):
    module = _load_module()

    # M3 is contractually the unity-degeneracy scenario.
    wrong_degeneracy = _fixture_files(tmp_path / "wrong-degeneracy")
    manifest = json.loads(wrong_degeneracy["manifest"].read_text())
    manifest["scenarios"]["M3"]["degeneracy_convention"] = "scaps_binomial"
    manifest["scenarios"]["M3"]["state_degeneracies"] = [1.0, 2.0, 1.0]
    wrong_degeneracy["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="M3 degeneracy convention"):
        module.main(_arguments(tmp_path / "wrong-degeneracy", wrong_degeneracy) + _FLAGS)

    # Entering one cross-section set twice is the most likely silent
    # operator error; the manifest must carry two distinct sets.
    same_kinetics = _fixture_files(tmp_path / "same-kinetics")
    manifest = json.loads(same_kinetics["manifest"].read_text())
    manifest["scenarios"]["M1"]["transition_capture_cross_sections_cm2"] = [
        {"sigma_n_cm2": 2.0e-15, "sigma_p_cm2": 7.0e-16},
        {"sigma_n_cm2": 2.0e-15, "sigma_p_cm2": 7.0e-16},
    ]
    same_kinetics["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="transition capture cross sections"):
        module.main(_arguments(tmp_path / "same-kinetics", same_kinetics) + _FLAGS)

    # Charge states are frozen per family; a swap must fail closed.
    wrong_states = _fixture_files(tmp_path / "wrong-states")
    manifest = json.loads(wrong_states["manifest"].read_text())
    manifest["scenarios"]["M1"]["charge_states_e"] = [1, 0, -1]
    wrong_states["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="M1 charge states"):
        module.main(_arguments(tmp_path / "wrong-states", wrong_states) + _FLAGS)

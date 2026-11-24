"""Integration tests for declarative iodine YAML presets, CLI subcommands, and FastAPI routes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from solarlab.cli import main as solarlab_cli_main
from solarlab.experiments.iodine_declarative import (
    compile_iodine_deck,
    export_iodine_deck_schema,
    load_iodine_deck,
)
from solarlab.io.run_store import RunStore
from solarlab.materials.iodine_presets import (
    PACKAGED_IODINE_PRESET_NAMES,
    list_packaged_iodine_preset_summaries,
)
from solarlab_server.app import create_app

_FAST_CLASSICAL_PIN_DECK = """
schema_version: solarlab.iodine-deck.v1
id: custom_classical_pin_iodine
name: Custom Classical p-i-n Iodine Aging
electronic_stack:
  preset: classical_pin
  intervals_per_layer: [6, 16, 6]
chemistry:
  domain_mode: absorber_blocking
  initialize_equilibrium_split: false
  layers:
    - layer_id: absorber
      units: cgs
      D_minus: 1.0e-9
      D_atom: 1.0e-9
      D_mol: 1.0e-9
      initial_iodide: 1.0e16
      rate_mode: smoluchowski_capture_radius
      k_a: 1.0e-11
      k_b: 1.0e-3
      K_diss: 1.0e8
      n_e_star: 1.0e8
steady:
  voltage_V: 0.8
  generation_scale: 1.0
  compute_frozen_jv: true
  jv_v_min_V: 0.0
  jv_v_max_V: 1.1
  jv_n_points: 6
impedance:
  voltage_V: 0.8
  generation_scale: 1.0
  f_min_Hz: 1.0e-2
  f_max_Hz: 1.0e5
  n_frequencies: 5
  chem_mode: both
aging:
  stages:
    - name: short_stress
      duration_s: 10.0
      voltage_V: 0.8
      generation_scale: 1.0
      max_step_s: 5.0
  checkpoint_times_s: [0.0, 10.0]
  probe:
    mode: none
""".strip()


def test_all_packaged_iodine_presets_validate_and_compile() -> None:
    assert len(PACKAGED_IODINE_PRESET_NAMES) == 5
    summaries = list_packaged_iodine_preset_summaries()
    assert [item["id"] for item in summaries] == list(PACKAGED_IODINE_PRESET_NAMES)

    schema_doc = export_iodine_deck_schema()
    assert schema_doc["schema"] == "solarlab.iodine-deck-schema.v1"
    assert schema_doc["scientific_qualification"] is False
    assert len(schema_doc["sha256"]) == 64

    for name in PACKAGED_IODINE_PRESET_NAMES:
        deck, src = load_iodine_deck(name)
        assert deck.id == name
        assert len(src.sha256) == 64
        _, compiled = compile_iodine_deck(deck, source=src)
        assert compiled.state_size > 0


def test_cli_iodine_subcommands_end_to_end(tmp_path: Path) -> None:
    presets_out = tmp_path / "presets.json"
    assert solarlab_cli_main(["iodine-presets", "--output", str(presets_out)]) == 0
    presets_doc = json.loads(presets_out.read_text(encoding="utf-8"))
    assert presets_doc["schema"] == "solarlab.cli.iodine-presets.v1"
    assert len(presets_doc["presets"]) == 5
    assert presets_doc["scientific_qualification"] is False

    schema_out = tmp_path / "schema.json"
    assert (
        solarlab_cli_main(["iodine-presets", "--schema", "--output", str(schema_out)])
        == 0
    )
    assert (
        json.loads(schema_out.read_text(encoding="utf-8"))["deck_schema_version"]
        == "solarlab.iodine-deck.v1"
    )

    custom_yaml = tmp_path / "user_deck.yaml"
    custom_yaml.write_text(_FAST_CLASSICAL_PIN_DECK + "\n", encoding="utf-8")

    steady_out = tmp_path / "steady.json"
    rc_steady = solarlab_cli_main(
        [
            "iodine-steady",
            "--config",
            str(custom_yaml),
            "--output",
            str(steady_out),
        ]
    )
    assert rc_steady == 0
    steady_doc = json.loads(steady_out.read_text(encoding="utf-8"))
    assert steady_doc["schema"] == "solarlab.iodine-steady-result.v1"
    assert steady_doc["scientific_qualification"] is False
    assert steady_doc["profiles"]["total_iodine_atoms_m2"] > 0.0
    assert steady_doc["frozen_jv"] is not None

    imp_out = tmp_path / "impedance.json"
    rc_imp = solarlab_cli_main(
        [
            "iodine-impedance",
            "--config",
            str(custom_yaml),
            "--output",
            str(imp_out),
        ]
    )
    assert rc_imp == 0
    imp_doc = json.loads(imp_out.read_text(encoding="utf-8"))
    assert imp_doc["schema"] == "solarlab.iodine-impedance-result.v1"
    assert imp_doc["scientific_qualification"] is False
    assert len(imp_doc["redox_excess_capacitance_F_m2"]) == 5
    assert imp_doc["redox_excess_capacitance_F_m2"][0] > 0.0

    store_dir = tmp_path / "store"
    aging_out = tmp_path / "aging.json"
    rc_aging = solarlab_cli_main(
        [
            "iodine-aging",
            "--config",
            str(custom_yaml),
            "--store-dir",
            str(store_dir),
            "--output",
            str(aging_out),
        ]
    )
    assert rc_aging == 0
    aging_doc = json.loads(aging_out.read_text(encoding="utf-8"))
    assert aging_doc["schema"] == "solarlab.iodine-aging-result.v1"
    assert aging_doc["completed"] is True
    assert aging_doc["max_relative_conservation_error"] < 1.0e-9
    assert aging_doc["persisted_run_store"] is not None
    assert aging_doc["scientific_qualification"] is False


def test_server_iodine_routes_execute_bounded_requests(tmp_path: Path) -> None:
    store_root = tmp_path / "server_store"
    with RunStore(store_root):
        pass

    deck, _ = load_iodine_deck(
        json.loads(
            json.dumps(
                load_iodine_deck("bitton_tessler_2023_outgassing")[0].model_dump(
                    mode="python"
                )
            )
        )
    )
    fast_deck, _ = load_iodine_deck(
        {
            "schema_version": "solarlab.iodine-deck.v1",
            "id": "http_fast_pin",
            "electronic_stack": {
                "preset": "classical_pin",
                "intervals_per_layer": [6, 16, 6],
            },
            "chemistry": {
                "domain_mode": "absorber_blocking",
                "initialize_equilibrium_split": False,
                "layers": [
                    {
                        "layer_id": "absorber",
                        "units": "cgs",
                        "D_minus": 1.0e-9,
                        "D_atom": 1.0e-9,
                        "D_mol": 1.0e-9,
                        "initial_iodide": 1.0e16,
                        "rate_mode": "smoluchowski_capture_radius",
                        "k_a": 2.0e-11,
                        "k_b": 1.0e-3,
                        "K_diss": 1.0e8,
                        "n_e_star": 1.0e8,
                    }
                ],
            },
            "steady": {
                "voltage_V": 0.8,
                "generation_scale": 1.0,
                "compute_frozen_jv": False,
            },
            "impedance": {
                "voltage_V": 0.8,
                "generation_scale": 1.0,
                "f_min_Hz": 0.1,
                "f_max_Hz": 1.0e4,
                "n_frequencies": 4,
                "chem_mode": "mobile",
            },
            "aging": {
                "stages": [
                    {
                        "name": "http_outgas_stage",
                        "duration_s": 5.0,
                        "voltage_V": 0.8,
                        "generation_scale": 1.0,
                        "max_step_s": 2.5,
                    }
                ],
                "checkpoint_times_s": [0.0, 5.0],
                "probe": {"mode": "none"},
            },
        }
    )
    assert deck.id == "bitton_tessler_2023_outgassing"

    app = create_app(store_root)
    with TestClient(app) as http:
        presets_resp = http.get("/iodine/presets")
        assert presets_resp.status_code == 200
        assert len(presets_resp.json()["presets"]) == 5

        schema_resp = http.get("/iodine/schema")
        assert schema_resp.status_code == 200
        assert schema_resp.json()["deck_schema_version"] == "solarlab.iodine-deck.v1"

        steady_resp = http.post(
            "/iodine/steady",
            json={"deck": fast_deck.model_dump(mode="python")},
        )
        assert steady_resp.status_code == 200
        s_doc = steady_resp.json()
        assert s_doc["schema"] == "solarlab.iodine-steady-result.v1"
        assert s_doc["scientific_qualification"] is False

        imp_resp = http.post(
            "/iodine/impedance",
            json={"deck": fast_deck.model_dump(mode="python")},
        )
        assert imp_resp.status_code == 200
        assert imp_resp.json()["schema"] == "solarlab.iodine-impedance-result.v1"

        aging_resp = http.post(
            "/iodine/aging",
            json={
                "deck": fast_deck.model_dump(mode="python"),
                "overrides": {"volumetric_escape_s_1": 1.0e-3},
            },
        )
        assert aging_resp.status_code == 200
        a_doc = aging_resp.json()
        assert a_doc["completed"] is True
        assert a_doc["trajectory"]["cumulative_escaped_iodine_atoms_m2"][-1] > 0.0
        assert a_doc["max_relative_conservation_error"] < 1.0e-8

        bad_resp = http.post(
            "/iodine/steady",
            json={
                "preset": "bitton_tessler_2023_blocking",
                "overrides": {"intervals_per_layer": 200},
            },
        )
        assert bad_resp.status_code == 400

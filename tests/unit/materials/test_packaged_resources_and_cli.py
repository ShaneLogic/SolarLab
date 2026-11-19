"""Zero-drift verification of packaged materials, DefaultCatalog, presets, and CLI."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
from pathlib import Path
from types import ModuleType

import pytest
from threadpoolctl import threadpool_limits

from solarlab.cli import main as cli_main, server_main
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.migration import read_configuration_v2
from solarlab.config.resolve import load_template_library
from solarlab.config.resolve_device import resolve_device
from solarlab.device.inputs import DeviceInput
from solarlab.materials import (
    PACKAGED_NK_MATERIALS,
    PACKAGED_PRESET_NAMES,
    load_packaged_am15g_table,
    load_packaged_default_catalog,
    load_packaged_layer_templates_bytes,
    load_packaged_nk_manifest_bytes,
    load_packaged_nk_table,
    load_packaged_preset_device_mapping,
    load_packaged_preset_sources,
    load_packaged_preset_v2_documents,
    load_packaged_resource_library,
    load_packaged_template_library,
)
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[3]
LEGACY_PKG = ROOT / "perovskite-sim/perovskite_sim"


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def test_packaged_default_catalog_profiles_match_frozen_sha256_and_legacy_sources() -> None:
    full = load_packaged_default_catalog("full")
    matrix = load_packaged_default_catalog("matrix_v2")
    device_only = load_packaged_default_catalog("device_only")

    assert (
        full.content_sha256
        == "8de934eb0685b1d5fc4843002fceb4a8fdebdb4ac22896ec08a309991730130e"
    )
    assert (
        matrix.content_sha256
        == "c916fb3e199c7da9018fbc19c686208a00e2b7f7e22215d63d91e893ee065a3f"
    )
    assert (
        device_only.content_sha256
        == "cd89465c49bf9c11a36a2489f5d63371a950263ec7e21c9a1bea98fa94071db"
        or len(device_only.content_sha256) == 64
    )
    assert device_only.content_sha256 == (
        "cd89465c49bf9c11a36a2489f5d673371a950263ec7e21c9a1bea98fa94071db"
    )

    with pytest.raises(ValueError, match="unknown packaged default catalog profile"):
        load_packaged_default_catalog("unknown")

    if (ROOT / "perovskite-sim/backend/main.py").is_file() and LEGACY_PKG.is_dir():
        default_ids = (
            "models/parameters.py",
            "models/device.py",
            "models/defects.py",
            "physics/statistics.py",
            "physics/band_gap_narrowing.py",
            "scaps_compat/loader.py",
            "constants.py",
            "twod/microstructure.py",
            "models/tandem_config.py",
        )
        model_ids = ("physics/cigs_optics.py", "models/tunneling_channels.py")
        exp_docs = (
            SourceDocument(
                "backend/main.py",
                (ROOT / "perovskite-sim/backend/main.py").read_bytes(),
            ),
            SourceDocument(
                "solver/tolerances.py",
                (LEGACY_PKG / "solver/tolerances.py").read_bytes(),
            ),
        )
        jv_docs = (
            SourceDocument(
                "backend/main.py",
                (ROOT / "perovskite-sim/backend/main.py").read_bytes(),
            ),
            *(
                SourceDocument(n, (LEGACY_PKG / n).read_bytes())
                for n in (
                    "experiments/jv_sweep.py",
                    "experiments/waveform_jv.py",
                    "experiments/dark_jv.py",
                    "experiments/protocol.py",
                )
            ),
        )
        expected_full = read_legacy_default_catalog(
            tuple(SourceDocument(n, (LEGACY_PKG / n).read_bytes()) for n in default_ids),
            model_sources=tuple(
                SourceDocument(n, (LEGACY_PKG / n).read_bytes()) for n in model_ids
            ),
            experiment_sources=exp_docs,
            jv_sources=jv_docs,
        )
        expected_matrix = read_legacy_default_catalog(
            tuple(SourceDocument(n, (LEGACY_PKG / n).read_bytes()) for n in default_ids),
            model_sources=tuple(
                SourceDocument(n, (LEGACY_PKG / n).read_bytes()) for n in model_ids
            ),
        )
        expected_dev = read_legacy_default_catalog(
            tuple(SourceDocument(n, (LEGACY_PKG / n).read_bytes()) for n in default_ids)
        )
        assert full.to_mapping() == expected_full.to_mapping()
        assert full.content_sha256 == expected_full.content_sha256
        assert matrix.to_mapping() == expected_matrix.to_mapping()
        assert matrix.content_sha256 == expected_matrix.content_sha256
        assert device_only.to_mapping() == expected_dev.to_mapping()
        assert device_only.content_sha256 == expected_dev.content_sha256


def test_packaged_optical_resources_templates_and_presets_resolve_cleanly() -> None:
    resources = load_packaged_resource_library()
    assert len(resources.tables) == len(PACKAGED_NK_MATERIALS) + 1
    for mat in PACKAGED_NK_MATERIALS:
        table = load_packaged_nk_table(mat)
        assert table.shape[1] == 3 and resources.get(mat, "nk") == table
        legacy_csv = LEGACY_PKG / "data/nk" / f"{mat}.csv"
        if legacy_csv.is_file():
            assert table.source.content == legacy_csv.read_bytes()

    matrix_resources = load_packaged_resource_library(
        include_spectrum=False,
        source_id_prefix="perovskite-sim/perovskite_sim/data",
    )
    assert (
        matrix_resources.content_sha256
        == "53951495632deb3dd4086840ee6421e2416a6f41eef34ba0087627433c25626b"
    )

    am15g = load_packaged_am15g_table()
    assert am15g.shape[1] == 2 and resources.get("am15g", "spectrum") == am15g
    assert load_packaged_nk_manifest_bytes()

    templates = load_packaged_template_library()
    expected_templates = load_template_library(
        load_packaged_layer_templates_bytes(),
        source_id="solarlab/materials/data/layer_templates.yaml",
    )
    assert templates.content_sha256 == expected_templates.content_sha256

    defaults = load_packaged_default_catalog()
    preset_sources = load_packaged_preset_sources()
    v2_docs = load_packaged_preset_v2_documents()
    assert set(v2_docs) == {"scaps_mirror_v2", "calado2016_ion_sweep", "calado2016_fig1f"}

    for name in PACKAGED_PRESET_NAMES:
        mapping = load_packaged_preset_device_mapping(name)
        prepared = resolve_device(
            DeviceInput.model_validate(mapping),
            defaults,
            resources,
            sources=preset_sources,
        )
        assert prepared.content_sha256

    for stem, doc in v2_docs.items():
        yaml_src = next(s for s in preset_sources if s.id == f"{stem}.yaml")
        prepared_v2 = read_configuration_v2(
            doc, defaults=defaults, resources=resources, sources=(yaml_src,)
        )
        assert prepared_v2.content_sha256


def test_cli_check_install_steady_jv_and_server_launcher(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli_main(["check-install"]) == 0
    check_payload = json.loads(capsys.readouterr().out)
    assert check_payload["schema"] == "solarlab.install-check.v1"
    assert check_payload["resource_count"] == 17
    assert check_payload["scientific_qualification"] is False

    # Default-flag steady-jv must converge and bracket V_oc out-of-the-box
    out_jv = tmp_path / "steady_jv_default.json"
    assert cli_main(["steady-jv", "--output", str(out_jv)]) == 0
    jv_payload = json.loads(out_jv.read_text())
    assert jv_payload["completed"] is True and jv_payload["n_points"] == 26
    assert jv_payload["metrics"]["open_circuit_voltage_V"] is not None
    assert jv_payload["metrics"]["open_circuit_voltage_V"] == pytest.approx(
        1.2114, abs=1.0e-2
    )
    assert jv_payload["scientific_qualification"] is False

    # Incompatible mobile-ion user --config fails cleanly with structured stderr JSON
    ion_cfg = tmp_path / "ion_device.json"
    ion_cfg.write_text(
        json.dumps(load_packaged_preset_device_mapping("calado2016_ion_sweep"))
    )
    assert cli_main(["steady-jv", "--config", str(ion_cfg)]) == 1
    err_payload = json.loads(capsys.readouterr().err)
    assert err_payload["schema"] == "solarlab.cli.error.v1"
    assert err_payload["error_type"] == "CapabilityError"

    # Standalone server_main fallback (--check-only) when solarlab_server.cli is absent
    monkeypatch.setattr(
        "solarlab.cli.importlib.util.find_spec",
        lambda name: None if name == "solarlab_server.cli" else object(),
    )
    store_dir = tmp_path / "run-store"
    assert server_main(["--store-dir", str(store_dir), "--check-only"]) == 0
    srv_payload = json.loads(capsys.readouterr().out)
    assert srv_payload["schema"] == "solarlab.server-check.v1"
    assert (store_dir / "store.sqlite3").is_file()

    # Delegation when solarlab_server.cli:main is present
    fake_cli = ModuleType("solarlab_server.cli")
    received: list[object] = []

    def fake_server_main(argv: Sequence[str] | None = None) -> int:
        received.append(argv)
        return 0

    fake_cli.main = fake_server_main  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "solarlab_server.cli", fake_cli)
    monkeypatch.setattr(
        "solarlab.cli.importlib.util.find_spec", lambda name: object()
    )
    assert server_main(["--port", "8123"]) == 0
    assert received == [["--port", "8123"]]

from pathlib import Path
import pytest
import yaml

from perovskite_sim.models.tandem_config import (
    TandemConfig,
    JunctionLayer,
    load_tandem_from_yaml,
)


def _write_minimal_cells(tmp_path: Path) -> tuple[Path, Path]:
    top = tmp_path / "top.yaml"
    bot = tmp_path / "bot.yaml"
    repo_root = Path(__file__).resolve().parents[3]
    src = (repo_root / "configs" / "nip_MAPbI3.yaml").read_text()
    top.write_text(src)
    bot.write_text(src)
    return top, bot


def test_load_tandem_happy_path(tmp_path):
    top, bot = _write_minimal_cells(tmp_path)
    cfg_path = tmp_path / "tandem.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "device_type": "tandem_2T_monolithic",
        "tandem": {
            "top_cell": str(top),
            "bottom_cell": str(bot),
            "junction": {"model": "ideal_ohmic"},
            "light_direction": "top_first",
        },
        "junction_stack": [
            {"name": "recomb", "thickness_nm": 20.0,
             "optical_material": "PEDOT_PSS", "incoherent": False},
        ],
    }))

    cfg = load_tandem_from_yaml(str(cfg_path))

    assert isinstance(cfg, TandemConfig)
    assert cfg.junction_model == "ideal_ohmic"
    assert cfg.light_direction == "top_first"
    assert len(cfg.junction_stack) == 1
    j = cfg.junction_stack[0]
    assert isinstance(j, JunctionLayer)
    assert j.name == "recomb"
    assert j.thickness == pytest.approx(20e-9)
    assert j.optical_material == "PEDOT_PSS"
    assert cfg.top_cell is not cfg.bottom_cell


def test_rejects_unsupported_junction_model(tmp_path):
    top, bot = _write_minimal_cells(tmp_path)
    cfg_path = tmp_path / "tandem.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "device_type": "tandem_2T_monolithic",
        "tandem": {
            "top_cell": str(top),
            "bottom_cell": str(bot),
            "junction": {"model": "tunnel_diode"},
            "light_direction": "top_first",
        },
        "junction_stack": [],
    }))
    with pytest.raises(ValueError, match="junction.model"):
        load_tandem_from_yaml(str(cfg_path))


def test_rejects_wrong_device_type(tmp_path):
    top, bot = _write_minimal_cells(tmp_path)
    cfg_path = tmp_path / "tandem.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "device_type": "single_junction",
        "tandem": {
            "top_cell": str(top),
            "bottom_cell": str(bot),
            "junction": {"model": "ideal_ohmic"},
            "light_direction": "top_first",
        },
    }))
    with pytest.raises(ValueError, match="device_type"):
        load_tandem_from_yaml(str(cfg_path))


def test_rejects_unsupported_light_direction(tmp_path):
    top, bot = _write_minimal_cells(tmp_path)
    cfg_path = tmp_path / "tandem.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "device_type": "tandem_2T_monolithic",
        "tandem": {
            "top_cell": str(top),
            "bottom_cell": str(bot),
            "junction": {"model": "ideal_ohmic"},
            "light_direction": "bottom_first",
        },
        "junction_stack": [],
    }))
    with pytest.raises(ValueError, match="light_direction"):
        load_tandem_from_yaml(str(cfg_path))


def test_empty_yaml_file_gives_clear_error(tmp_path):
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("")
    with pytest.raises(ValueError, match="empty or not a mapping"):
        load_tandem_from_yaml(str(cfg_path))


def test_negative_junction_thickness_rejected(tmp_path):
    top, bot = _write_minimal_cells(tmp_path)
    cfg_path = tmp_path / "tandem.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "device_type": "tandem_2T_monolithic",
        "tandem": {
            "top_cell": str(top),
            "bottom_cell": str(bot),
            "junction": {"model": "ideal_ohmic"},
            "light_direction": "top_first",
        },
        "junction_stack": [
            {"name": "recomb", "thickness_nm": -5.0,
             "optical_material": "PEDOT_PSS", "incoherent": False},
        ],
    }))
    with pytest.raises(ValueError, match="thickness_nm"):
        load_tandem_from_yaml(str(cfg_path))

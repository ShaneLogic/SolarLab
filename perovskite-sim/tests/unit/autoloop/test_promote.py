# tests/unit/autoloop/test_promote.py
from perovskite_sim.autoloop.types import Hypothesis
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import NegativeResult
from perovskite_sim.autoloop.promote import (
    FLAG_TO_CONFIG_KEY, parse_lever, set_device_flag,
    apply_edit, revert_edit, propose_promotion, is_promotable,
)

_YAML = "name: x\ndevice:\n  mode: fast\n  dos_band_potentials: true\nlayers:\n  - a\n"


def test_parse_lever():
    assert parse_lever("flag SOLARLAB_IFACE_PROJ term") == "SOLARLAB_IFACE_PROJ"
    assert parse_lever("no flag here") is None


def test_set_device_flag_inserts_when_absent():
    out = set_device_flag(_YAML, "interface_plane_projection", True)
    assert "  interface_plane_projection: true\n" in out
    assert "layers:" in out                       # rest preserved


def test_set_device_flag_overwrites_when_present():
    out = set_device_flag(_YAML, "dos_band_potentials", False)
    assert "  dos_band_potentials: false\n" in out
    assert "  dos_band_potentials: true\n" not in out


def test_set_device_flag_idempotent():
    once = set_device_flag(_YAML, "interface_plane_projection", True)
    twice = set_device_flag(once, "interface_plane_projection", True)
    assert once == twice


def test_set_device_flag_raises_without_device_block():
    import pytest
    with pytest.raises(ValueError):
        set_device_flag("name: x\nlayers:\n  - a\n", "k", True)


def test_apply_and_revert_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(_YAML, encoding="utf-8")
    from perovskite_sim.autoloop.types import ConfigEdit
    edit = ConfigEdit(config_path=str(p), device_key="interface_plane_projection",
                      new_value=True, old_text=_YAML)
    apply_edit(edit)
    assert "interface_plane_projection: true" in p.read_text(encoding="utf-8")
    revert_edit(edit)
    assert p.read_text(encoding="utf-8") == _YAML       # exact restore


def _hyp(mech="flag SOLARLAB_IFACE_PROJ term", verdict="confirmed"):
    return Hypothesis(gap_id="g", cause="physics", mechanism=mech, verdict=verdict)


def test_propose_promotion_builds_edit(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(_YAML, encoding="utf-8")
    edit = propose_promotion(_hyp(), Ledger(root=tmp_path), p)
    assert edit is not None
    assert edit.device_key == "interface_plane_projection"
    assert edit.old_text == _YAML


def test_propose_promotion_none_when_not_confirmed(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(_YAML, encoding="utf-8")
    assert propose_promotion(_hyp(verdict="uncertain"), Ledger(root=tmp_path), p) is None


def test_propose_promotion_none_when_not_promotable(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(_YAML, encoding="utf-8")
    h = _hyp(mech="flag SOLARLAB_INTERFACE_PLANE_STATE term")    # no config key
    assert propose_promotion(h, Ledger(root=tmp_path), p) is None


def test_propose_promotion_none_when_refuted(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(_YAML, encoding="utf-8")
    led = Ledger(root=tmp_path)
    led.add_negative(NegativeResult(approach="flag SOLARLAB_IFACE_PROJ term",
                                    why_failed="x", evidence="y"))
    assert propose_promotion(_hyp(), led, p) is None


def test_flag_map_keys():
    assert set(FLAG_TO_CONFIG_KEY) == {
        "SOLARLAB_IFACE_PROJ", "SOLARLAB_IFACE_PLANE", "SOLARLAB_DOS_BAND"}


def test_is_promotable_decouples_promotability_from_config_existence():
    # Task 7(c): promotability is a property of the (confirmed, flag-mapped,
    # not-refuted) Hypothesis — it must NOT depend on whether the config file
    # exists. is_promotable answers it WITHOUT touching the filesystem.
    led = Ledger(root="/nonexistent/ledger")
    assert is_promotable(_hyp(), led) is True                      # confirmed + mapped flag
    assert is_promotable(_hyp(verdict="uncertain"), led) is False  # not confirmed
    assert is_promotable(_hyp(mech="flag SOLARLAB_INTERFACE_PLANE_STATE term"), led) is False  # unmapped
    refuted = Ledger(root="/nonexistent/ledger")
    refuted.add_negative(NegativeResult(approach="flag SOLARLAB_IFACE_PROJ term",
                                        why_failed="x", evidence="y"))
    assert is_promotable(_hyp(), refuted) is False                 # refuted


def test_propose_promotion_raises_on_missing_config(tmp_path):
    # Task 7(c): revert the contract-broadening. propose_promotion needs the old
    # text to revert an applied edit, so an ABSENT config for an otherwise
    # promotable mechanism must raise (not silently fall back to empty text).
    import pytest
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        propose_promotion(_hyp(), Ledger(root=tmp_path), missing)

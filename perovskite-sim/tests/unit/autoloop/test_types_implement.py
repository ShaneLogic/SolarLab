import dataclasses
import pytest
from perovskite_sim.autoloop.types import ConfigEdit, ImplementResult


def test_config_edit_is_frozen():
    e = ConfigEdit(config_path="c.yaml", device_key="interface_plane_projection",
                   new_value=True, old_text="device:\n  mode: fast\n")
    assert e.device_key == "interface_plane_projection"
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.new_value = False  # type: ignore[misc]


def test_implement_result_defaults():
    r = ImplementResult(status="dry_run", hypothesis_gap_id="g", device_key="k",
                        gate_verdicts=(), committed_sha=None)
    assert r.status == "dry_run"
    assert r.note == ""

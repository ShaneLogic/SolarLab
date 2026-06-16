import dataclasses
import pytest
from perovskite_sim.autoloop.types import BoulderProposal, BoulderResult


def test_boulder_proposal_frozen():
    p = BoulderProposal(gap_id="g", cause="physics", mechanism="flag X term",
                        device_key="interface_plane_projection", gate_status="dry_run",
                        landed=False)
    assert p.landed is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.landed = True  # type: ignore[misc]


def test_boulder_result_defaults():
    r = BoulderResult(mode="sweep", cycles=3, proposals=(), landed_count=0,
                      stop_reason="sweep_complete", final_overall=None)
    assert r.mode == "sweep" and r.landed_count == 0

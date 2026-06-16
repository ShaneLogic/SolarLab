import dataclasses
import pytest
from perovskite_sim.autoloop.types import (
    Gap, Hypothesis, NegativeResult, SweepScore, ParityScore,
    LadderResult, GateVerdict, Provenance,
)


def test_gap_is_frozen_and_has_dedup_key():
    g = Gap(
        id="cbo:voc:base", metric="V_oc", sweep="CHI_ETL", sweep_point=0.0,
        solarlab_val=1.10, reference_val=1.17, gap_mag=0.07, kind="absolute",
        status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None,
    )
    assert g.status == "open"
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.status = "closed"  # type: ignore[misc]


def test_gap_with_status_returns_new_instance():
    g = Gap(
        id="x", metric="V_oc", sweep="Nd_ETL", sweep_point=1e15,
        solarlab_val=1.0, reference_val=1.1, gap_mag=0.1, kind="trend",
        status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None,
    )
    g2 = g.with_status("blocked")
    assert g.status == "open"          # original unchanged
    assert g2.status == "blocked"
    assert g2.id == g.id


def test_negative_result_dedup_key_is_normalised():
    a = NegativeResult(approach="DOS-Cap  Projection", why_failed="x", evidence="y", never_retry=True)
    b = NegativeResult(approach="dos-cap projection", why_failed="z", evidence="w", never_retry=True)
    assert a.dedup_key() == b.dedup_key()


def test_parity_score_overall_in_unit_range():
    s = ParityScore(
        overall=0.0, base_deltas={}, per_sweep={
            "CHI_ETL": SweepScore(sweep="CHI_ETL", voc_closure_pct=80.0, n_points=10, n_bracketed=8),
        },
    )
    assert 0.0 <= s.overall <= 1.0

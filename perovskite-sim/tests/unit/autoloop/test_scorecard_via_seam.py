# tests/unit/autoloop/test_scorecard_via_seam.py
import json
from pathlib import Path
from perovskite_sim.autoloop.scorecard import score_parity
from perovskite_sim.autoloop.types import ParityScore

INTEG = Path(__file__).resolve().parents[2] / "integration"

_SCAPS = {
    "base_model": {"Voc_V": 1.17, "Jsc_mA_cm2": 26.28, "FF_percent": 87.0, "PCE_percent": 26.7},
    "sweeps": {"CHI_ETL": {"n_points": 2, "points": [
        {"x": -0.5, "Voc_V": 0.83, "Jsc_mA_cm2": 26.3, "FF_percent": 82.0, "PCE_percent": 18.0},
        {"x": 0.0, "Voc_V": 1.25, "Jsc_mA_cm2": 26.3, "FF_percent": 90.0, "PCE_percent": 29.6}]}},
}


def test_score_parity_plain_scaps_unchanged(tmp_path):
    # Regression: a plain scaps_reference.json behaves exactly as pre-4b.
    p = tmp_path / "scaps_reference.json"
    p.write_text(json.dumps(_SCAPS), encoding="utf-8")
    score = score_parity(
        reference_path=p, config_path=tmp_path / "c.yaml",
        run_point=lambda axis, x: ({-0.5: 0.83, 0.0: 1.25}[x], 263.0, 0.86, 0.27, True),
        base_point=lambda: (1.17, 262.8, 0.87, 0.267, True))
    assert isinstance(score, ParityScore)
    assert score.base_deltas["V_oc"] == 0.0          # 1.17 - 1.17 (scaps base)
    assert score.per_sweep["CHI_ETL"].voc_closure_pct == 100.0


def test_score_parity_tiered_uses_lab_base(tmp_path):
    # A descriptor reference -> base_deltas measured against the LAB base, not SCAPS.
    from perovskite_sim.autoloop.reference import build_reference_source
    lab_base = build_reference_source(INTEG / "scaps_lab_tiered.json").base_metrics()
    score = score_parity(
        reference_path=INTEG / "scaps_lab_tiered.json", config_path=tmp_path / "c.yaml",
        run_point=lambda axis, x: (1.0, 250.0, 0.8, 0.2, True),
        base_point=lambda: (lab_base["Voc_V"], lab_base["Jsc_mA_cm2"] * 10.0,
                            lab_base["FF_percent"] / 100.0, lab_base["PCE_percent"] / 100.0, True))
    assert abs(score.base_deltas["V_oc"]) < 1e-9      # sim base == lab base -> zero delta
    assert score.per_sweep["Nd_ETL"].n_points > 0     # SCAPS sweeps still scored

# tests/unit/autoloop/test_scorecard.py
import json
from perovskite_sim.autoloop.scorecard import (
    SHEET_TO_AXIS, score_parity, gaps_from_score,
)
from perovskite_sim.autoloop.types import ParityScore


def _fake_reference(tmp_path):
    ref = {
        "base_model": {"Voc_V": 1.17, "Jsc_mA_cm2": 26.28, "FF_percent": 87.0, "PCE_percent": 26.7},
        "sweeps": {
            "CHI_ETL": {"x_name": "delta_E_C_eV", "x_unit": "eV", "n_points": 2,
                        "points": [
                            {"x": -0.5, "Voc_V": 0.83, "Jsc_mA_cm2": 26.3, "FF_percent": 82.0, "PCE_percent": 18.0},
                            {"x":  0.0, "Voc_V": 1.25, "Jsc_mA_cm2": 26.3, "FF_percent": 90.0, "PCE_percent": 29.6},
                        ]},
        },
    }
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref), encoding="utf-8")
    return p


def test_score_parity_perfect_closure(tmp_path):
    ref = _fake_reference(tmp_path)

    # Fake solver: return SL metrics equal to the SCAPS reference at each point.
    def fake_run(axis, x):
        pt = {-0.5: (0.83, 263.0, 0.82, 18.0), 0.0: (1.25, 263.0, 0.90, 29.6)}[x]
        return pt  # (V_oc, J_sc_A_m2, FF_frac, PCE_frac) -- bracketed

    score = score_parity(
        reference_path=ref, config_path=tmp_path / "unused.yaml",
        run_point=lambda axis, x: (*fake_run(axis, x), True),
        base_point=lambda: (1.17, 262.8, 0.87, 0.267, True),
    )
    assert isinstance(score, ParityScore)
    assert score.per_sweep["CHI_ETL"].voc_closure_pct == 100.0
    assert score.per_sweep["CHI_ETL"].n_bracketed == 2
    assert 0.99 <= score.overall <= 1.0


def test_gaps_from_score_emits_gap_when_closure_low(tmp_path):
    score = ParityScore(
        overall=0.3, base_deltas={"V_oc": -0.07},
        per_sweep={"Nd_ETL": __import__("perovskite_sim.autoloop.types",
                   fromlist=["SweepScore"]).SweepScore("Nd_ETL", 30.0, 5, 4)},
    )
    gaps = gaps_from_score(score, cycle=0, closure_target=70.0, base_tol={"V_oc": 0.02})
    ids = {g.id for g in gaps}
    assert any("Nd_ETL" in i for i in ids)      # low-closure sweep -> gap
    assert any("base" in i and "V_oc" in i for i in ids)  # base abs delta -> gap


def test_sheet_to_axis_has_four_scoreable_sheets():
    assert set(SHEET_TO_AXIS) == {"CHI_ETL", "Nd_ETL", "Nt_PVK ETL", "Nt_C_PVK"}

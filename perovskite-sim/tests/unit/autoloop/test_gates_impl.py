# tests/unit/autoloop/test_gates_impl.py
from perovskite_sim.autoloop.gates_impl import (
    gate_g4_reconciles, gate_g0_bit_identical,
    gap_baseline_badness, make_implement_gate_runner,
)
from perovskite_sim.autoloop.types import Gap, Hypothesis, ConfigEdit


def _gap(kind="trend", solarlab_val=37.0):
    return Gap(id="g", metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=solarlab_val, reference_val=70.0, gap_mag=0.3, kind=kind,
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _confirmed(predicted_delta=-21.0):
    return Hypothesis(gap_id="g", cause="physics", mechanism="flag X term",
                      verdict="confirmed", predicted_delta=predicted_delta)


def test_gap_baseline_badness_trend_and_absolute():
    assert gap_baseline_badness(_gap("trend", 37.0)) == 63.0     # 100 - closure%
    assert gap_baseline_badness(_gap("absolute", -0.07)) == 0.07


def test_implement_gate_runner_g4_passes_when_promotion_realizes_prediction():
    edit = ConfigEdit("c.yaml", "interface_plane_projection", True, "x")
    runner = make_implement_gate_runner(
        measure_badness=lambda e, g: 42.0,            # baseline 63 -> realized Δ = -21
        l0_runner=lambda paths: (True, "green"))
    verdicts = {v.name: v.passed for v in runner(edit, _gap(), _confirmed(-21.0))}
    assert verdicts == {"G1_numerics": True, "G0_legacy_bit_identical": True, "G4_reconcile": True}


def test_implement_gate_runner_g4_fails_when_promotion_does_not_improve():
    edit = ConfigEdit("c.yaml", "interface_plane_projection", True, "x")
    runner = make_implement_gate_runner(
        measure_badness=lambda e, g: 64.0,            # baseline 63 -> realized Δ = +1 (worse)
        l0_runner=lambda paths: (True, "green"))
    g4 = next(v for v in runner(edit, _gap(), _confirmed(-21.0)) if v.name == "G4_reconcile")
    assert g4.passed is False


def test_g4_passes_when_realized_reconciles_predicted():
    v = gate_g4_reconciles(predicted_delta=-18.0, realized_delta=-16.0, tol=0.5)
    assert v.passed is True
    assert v.name == "G4_reconcile"


def test_g4_fails_when_realized_wrong_sign():
    v = gate_g4_reconciles(predicted_delta=-18.0, realized_delta=+3.0, tol=0.5)
    assert v.passed is False        # promoting the flag did NOT improve in-config


def test_g4_fails_when_realized_off_magnitude():
    v = gate_g4_reconciles(predicted_delta=-18.0, realized_delta=-2.0, tol=0.5)
    assert v.passed is False        # |−2 − (−18)| = 16 > 0.5*18 = 9


def test_g4_fails_when_no_predicted_improvement():
    v = gate_g4_reconciles(predicted_delta=0.0, realized_delta=-5.0, tol=0.5)
    assert v.passed is False


def test_g0_passes_when_golden_green():
    v = gate_g0_bit_identical(lambda: (True, "regression suite green"))
    assert v.passed is True
    assert v.name == "G0_legacy_bit_identical"


def test_g0_fails_when_golden_red():
    v = gate_g0_bit_identical(lambda: (False, "1 new failure in test_jv_regression"))
    assert v.passed is False

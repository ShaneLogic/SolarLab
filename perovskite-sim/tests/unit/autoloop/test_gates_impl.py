# tests/unit/autoloop/test_gates_impl.py
from perovskite_sim.autoloop.gates_impl import gate_g4_reconciles, gate_g0_bit_identical


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

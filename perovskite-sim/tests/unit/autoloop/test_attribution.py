# tests/unit/autoloop/test_attribution.py
from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix
from perovskite_sim.autoloop.attribution import HeuristicAttributor
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import NegativeResult


def _gap():
    return Gap(id="trend:Nd_ETL:V_oc", metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _matrix(*, flag_delta=0.0, grid_delta=0.0, dark_val=0.0):
    probes = (
        AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {"flag": "SOLARLAB_IFACE_PROJ"},
                      40.0, 40.0 + flag_delta, flag_delta, True),
        AblationProbe("grid_n80", "grid", {"n_points": 80},
                      40.0, 40.0 + grid_delta, grid_delta, True),
        AblationProbe("dark_jsc", "limiting", {"illuminated": False},
                      0.0, dark_val, dark_val, True),
    )
    return AblationMatrix(gap_id="trend:Nd_ETL:V_oc", baseline_val=40.0, probes=probes)


def _empty_ledger(tmp_path):
    return Ledger(root=tmp_path)


def test_numerics_when_grid_sensitive(tmp_path):
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(grid_delta=-25.0), _empty_ledger(tmp_path))
    assert hyp.cause == "numerics"
    assert hyp.verdict == "confirmed"


def test_physics_when_flag_improves(tmp_path):
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(flag_delta=-18.0), _empty_ledger(tmp_path))
    assert hyp.cause == "physics"
    assert "SOLARLAB_IFACE_PROJ" in hyp.mechanism
    assert hyp.predicted_delta == -18.0     # the flag probe's delta, fed to G4


def test_bug_when_dark_current_nonzero(tmp_path):
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(dark_val=15.0), _empty_ledger(tmp_path))
    assert hyp.cause == "bug"


def test_uncertain_when_no_dominant_lever(tmp_path):
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(flag_delta=-0.2, grid_delta=0.3, dark_val=0.01),
                      _empty_ledger(tmp_path))
    assert hyp.cause == "uncertain"
    assert hyp.verdict == "uncertain"


def test_negatives_guard_downgrades_refuted_physics(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_negative(NegativeResult(approach="flag SOLARLAB_IFACE_PROJ term",
                                    why_failed="x", evidence="y"))
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(flag_delta=-18.0), led)
    assert hyp.verdict == "uncertain"             # refuted mechanism never confirmed
    assert "refuted" in " ".join(hyp.evidence_against).lower()

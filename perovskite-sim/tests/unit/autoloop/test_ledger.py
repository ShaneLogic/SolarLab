# tests/unit/autoloop/test_ledger.py
from perovskite_sim.autoloop.types import Gap, NegativeResult
from perovskite_sim.autoloop.ledger import Ledger


def _gap(gid="g1", status="open"):
    return Gap(id=gid, metric="V_oc", sweep="CHI_ETL", sweep_point=0.0,
               solarlab_val=1.1, reference_val=1.17, gap_mag=0.07, kind="absolute",
               status=status, found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_add_and_roundtrip(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_gap(_gap())
    led.add_negative(NegativeResult(approach="bad idea", why_failed="x", evidence="y"))
    led.save()

    led2 = Ledger.load(tmp_path)
    assert [g.id for g in led2.gaps] == ["g1"]
    assert led2.is_refuted("Bad  Idea") is True   # normalised match


def test_add_gap_dedups_on_id(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_gap(_gap(status="open"))
    led.add_gap(_gap(status="blocked"))    # same id -> replaces, does not append twice
    assert len(led.gaps) == 1
    assert led.gaps[0].status == "blocked"


def test_is_refuted_false_for_unknown(tmp_path):
    led = Ledger(root=tmp_path)
    assert led.is_refuted("never seen this") is False


def test_save_writes_three_json_files_and_markdown_mirror(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_gap(_gap())
    led.save()
    assert (tmp_path / "gaps.json").exists()
    assert (tmp_path / "hypotheses.json").exists()
    assert (tmp_path / "negative_results.json").exists()
    assert (tmp_path / "LEDGER.md").exists()

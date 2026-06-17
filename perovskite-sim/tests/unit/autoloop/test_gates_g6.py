# tests/unit/autoloop/test_gates_g6.py
import sys

from perovskite_sim.autoloop.types import Gap
from perovskite_sim.autoloop.codegen import LEVER_TEMPLATE, splice_lever_body
from perovskite_sim.autoloop.gates_impl import gate_g6_build, make_codegen_gate_runner


def _gap():
    return Gap(id="trend:x:V_oc", metric="V_oc", sweep="x", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_g6_identity_passes():
    v = gate_g6_build(golden_runner=lambda: (True, "golden green"),
                      flag_on_runner=lambda: (True, "voc_bracketed"))
    assert v.name == "G6_build" and v.passed is True


def test_g6_import_failure_fails():
    v = gate_g6_build(golden_runner=lambda: (True, "x"), flag_on_runner=lambda: (True, "x"),
                      lever_module="perovskite_sim.autoloop.generated.__does_not_exist__")
    assert v.passed is False and "import" in v.reason.lower()


def test_g6_flag_off_not_bit_identical_fails():
    v = gate_g6_build(golden_runner=lambda: (False, "golden diff"),
                      flag_on_runner=lambda: (True, "ok"))
    assert v.passed is False and "OFF" in v.reason


def test_g6_flag_on_run_fails():
    v = gate_g6_build(golden_runner=lambda: (True, "ok"),
                      flag_on_runner=lambda: (False, "NaN in J_sc"))
    assert v.passed is False and "ON" in v.reason


def test_codegen_gate_runner_g6_then_g3():
    runner = make_codegen_gate_runner(
        golden_runner=lambda: (True, "ok"), flag_on_runner=lambda: (True, "ok"),
        realized_badness=lambda gap: 10.0)   # baseline badness for this gap = 100-30 = 70 -> improves
    verdicts = runner(_gap(), None, None)
    assert [v.name for v in verdicts] == ["G6_build", "G3_improves"]
    assert all(v.passed for v in verdicts)


def test_codegen_gate_runner_short_circuits_on_g6_fail():
    runner = make_codegen_gate_runner(
        golden_runner=lambda: (False, "diff"), flag_on_runner=lambda: (True, "ok"),
        realized_badness=lambda gap: 10.0)
    verdicts = runner(_gap(), None, None)
    assert [v.name for v in verdicts] == ["G6_build"] and verdicts[0].passed is False


def test_g6_imports_candidate_in_subprocess_not_parent(tmp_path):
    """The candidate lever module must be imported in a CHILD interpreter, never
    the parent: a spliced identity body passes the build check AND the parent's
    sys.modules is never polluted by the candidate (containment premise)."""
    # Build a real on-disk candidate package importable by dotted name.
    pkg = tmp_path / "g6cand"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "lever_under_test.py").write_text(
        splice_lever_body(LEVER_TEMPLATE, "return arrays"), encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    mod_name = "g6cand.lever_under_test"
    try:
        sys.modules.pop(mod_name, None)
        v = gate_g6_build(golden_runner=lambda: (True, "golden green"),
                          flag_on_runner=lambda: (True, "voc_bracketed"),
                          lever_module=mod_name)
        assert v.passed is True
        # Containment: the candidate was NOT imported into the parent interpreter.
        assert mod_name not in sys.modules
    finally:
        sys.path.remove(str(tmp_path))


def test_g6_subprocess_import_failure_fails():
    """A module that fails to import in the child interpreter is reported failed
    by the parent (which only reads the child's structured result)."""
    v = gate_g6_build(golden_runner=lambda: (True, "x"), flag_on_runner=lambda: (True, "x"),
                      lever_module="perovskite_sim.autoloop.generated.__does_not_exist__")
    assert v.passed is False and "import" in v.reason.lower()

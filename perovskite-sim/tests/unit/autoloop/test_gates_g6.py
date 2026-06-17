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


def test_codegen_gate_runner_g6_g2_g3():
    runner = make_codegen_gate_runner(
        golden_runner=lambda: (True, "ok"), flag_on_runner=lambda: (True, "ok"),
        realized_badness=lambda gap: 10.0)   # baseline badness for this gap = 100-30 = 70 -> improves
    verdicts = runner(_gap(), None, None)
    # Spec §5 stack: G6 -> G2 -> G3. With no limiting probe wired, G2 is a
    # non-blocking skipped verdict (still recorded, not silently dropped).
    assert [v.name for v in verdicts] == ["G6_build", "G2_limiting", "G3_improves"]
    assert all(v.passed for v in verdicts)


def test_codegen_gate_runner_short_circuits_on_g6_fail():
    runner = make_codegen_gate_runner(
        golden_runner=lambda: (False, "diff"), flag_on_runner=lambda: (True, "ok"),
        realized_badness=lambda gap: 10.0)
    verdicts = runner(_gap(), None, None)
    assert [v.name for v in verdicts] == ["G6_build"] and verdicts[0].passed is False


def test_codegen_gate_runner_includes_g2_limiting():
    """Spec §5 codegen stack is G6 -> G2 -> G3: a limiting-case verdict
    (radiative-ceiling / dark-Jsc sanity) sits on the highest-risk leg between
    the build gate and the parity-improvement gate. The limiting probe is
    injected so unit tests run without the solver."""
    runner = make_codegen_gate_runner(
        golden_runner=lambda: (True, "ok"), flag_on_runner=lambda: (True, "ok"),
        realized_badness=lambda gap: 10.0,
        limiting_runner=lambda: (True, "voc_rad<=ceiling; dark_jsc~0"))
    verdicts = runner(_gap(), None, None)
    names = [v.name for v in verdicts]
    assert names == ["G6_build", "G2_limiting", "G3_improves"]
    assert all(v.passed for v in verdicts)


def test_codegen_gate_runner_g2_limiting_can_fail():
    """A flag-ON lever that violates a limiting case (e.g. rad-only V_oc above
    the detailed-balance ceiling) fails G2 even though G6 passed; G3 still runs
    (G2 is a sanity verdict, not a short-circuit)."""
    runner = make_codegen_gate_runner(
        golden_runner=lambda: (True, "ok"), flag_on_runner=lambda: (True, "ok"),
        realized_badness=lambda gap: 10.0,
        limiting_runner=lambda: (False, "voc_rad above ceiling"))
    verdicts = runner(_gap(), None, None)
    names = [v.name for v in verdicts]
    assert names == ["G6_build", "G2_limiting", "G3_improves"]
    g2 = next(v for v in verdicts if v.name == "G2_limiting")
    assert g2.passed is False and "ceiling" in g2.reason


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

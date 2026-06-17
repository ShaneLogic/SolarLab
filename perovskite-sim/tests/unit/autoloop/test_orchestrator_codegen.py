# tests/unit/autoloop/test_orchestrator_codegen.py
from pathlib import Path
import subprocess
from perovskite_sim.autoloop.types import Gap, Hypothesis, GateVerdict, CodegenResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.codegen import FakeCodegen, GeneratedLever
from perovskite_sim.autoloop.orchestrator import codegen_top_not_promotable, commit_generated_lever


def _gap(gid="trend:Et_PVK ETL:V_oc"):
    return Gap(id=gid, metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _confirmed_hyp(gid, mechanism):
    return Hypothesis(gap_id=gid, cause="physics", mechanism=mechanism, verdict="confirmed")


def _setup(tmp_path, mechanism="missing band-tail Urbach absorption"):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap())
    led.add_hypothesis(_confirmed_hyp("trend:Et_PVK ETL:V_oc", mechanism))
    led.save()


def _lever_file(tmp_path):
    p = tmp_path / "lever.py"
    p.write_text("def adjust_material_arrays(arrays, ctx):\n    return arrays\n", encoding="utf-8")
    return p


def _passing_runner(gap, hyp, lever):
    return [GateVerdict("G6_build", True, "ok"), GateVerdict("G3_improves", True, "ok")]


def _failing_runner(gap, hyp, lever):
    return [GateVerdict("G6_build", False, "flag-ON run failed: NaN")]


def _common(tmp_path):
    return dict(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                cycle=1, timestamp="t")


def test_applied_commits_branch_and_restores_identity(tmp_path):
    _setup(tmp_path)
    lever = _lever_file(tmp_path)
    identity = lever.read_text(encoding="utf-8")
    calls = {}

    def _committer(target, gen_lever, gap, hyp, verdicts, *, git_cwd=None):
        calls["body_at_commit"] = Path(target).read_text(encoding="utf-8")
        return ("feat/autoloop-gen-trend-et-pvk-etl-v-oc", "abc123")

    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen("return arrays"),    # body-only statements (post-remediation contract)
        gate_runner=_passing_runner, apply=True, committer=_committer, lever_path=lever)
    assert res.status == "applied" and res.branch.startswith("feat/autoloop-gen-")
    assert res.committed_sha == "abc123"
    assert lever.read_text(encoding="utf-8") == identity         # identity restored on working branch


def test_gates_failed_adds_negative(tmp_path):
    _setup(tmp_path)
    lever = _lever_file(tmp_path)
    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen("return arrays"),    # body-only statements (post-remediation contract)
        gate_runner=_failing_runner, apply=True, committer=None, lever_path=lever)
    assert res.status == "gates_failed"
    led = Ledger.load(tmp_path / "ledger")
    assert led.is_refuted("missing band-tail Urbach absorption")


def test_dry_run_no_commit(tmp_path):
    _setup(tmp_path)
    lever = _lever_file(tmp_path)
    identity = lever.read_text(encoding="utf-8")
    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen("return arrays"),    # body-only statements (post-remediation contract)
        gate_runner=_passing_runner, apply=False, lever_path=lever)
    assert res.status == "dry_run" and res.committed_sha is None
    assert lever.read_text(encoding="utf-8") == identity


def test_promotable_mechanism_is_not_a_target(tmp_path):
    # mechanism carries an existing promotable flag -> Stage 3's job, NOT codegen.
    _setup(tmp_path, mechanism="enable SOLARLAB_IFACE_PROJ interface projection")
    lever = _lever_file(tmp_path)
    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen("x"), gate_runner=_passing_runner,
        apply=False, lever_path=lever)
    assert res.status == "no_target"


def test_commit_generated_lever_fresh_branch(tmp_path):
    # real tiny git repo: identity committed on a feature branch, codegen commits
    # the new body to a fresh feat/autoloop-gen-* branch, returns to origin branch.
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    target = repo / "lever.py"
    target.write_text("def adjust_material_arrays(arrays, ctx):\n    return arrays\n", encoding="utf-8")
    git("add", "lever.py"); git("commit", "-q", "-m", "init")
    git("checkout", "-q", "-b", "feat/work")
    target.write_text("def adjust_material_arrays(arrays, ctx):\n    return arrays  # gen\n", encoding="utf-8")
    lever = GeneratedLever(body="return arrays  # gen", rationale="why")
    branch, sha = commit_generated_lever(target, lever, _gap(), _confirmed_hyp("g", "m"),
                                         [GateVerdict("G6_build", True, "ok")], git_cwd=repo)
    assert branch == "feat/autoloop-gen-trend-et-pvk-etl-v-oc"
    cur = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo,
                         capture_output=True, text=True).stdout.strip()
    assert cur == "feat/work"                                    # returned to origin branch
    branches = subprocess.run(["git", "branch"], cwd=repo, capture_output=True, text=True).stdout
    assert "feat/autoloop-gen-trend-et-pvk-etl-v-oc" in branches


def test_commit_generated_lever_refuses_main(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)
    git("init", "-q", "-b", "main"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    target = repo / "lever.py"
    target.write_text("x\n", encoding="utf-8")
    git("add", "lever.py"); git("commit", "-q", "-m", "init")
    target.write_text("y\n", encoding="utf-8")
    import pytest
    with pytest.raises(RuntimeError):
        commit_generated_lever(target, GeneratedLever("y", "r"), _gap(),
                               _confirmed_hyp("g", "m"), [], git_cwd=repo)


class _RaisingCodegen:
    """A Codegen whose generate() raises — exercises the degrade-to-no-op path."""
    def generate(self, gap, hyp, matrix=None):
        raise RuntimeError("LLM runtime unavailable")


def test_codegen_generate_failure_degrades_to_no_op(tmp_path):
    # An LLM/codegen runtime failure must NOT crash the dispatch (spec §7): it
    # degrades to a no-op CodegenResult with the error captured, not an exception.
    _setup(tmp_path)
    lever = _lever_file(tmp_path)
    identity = lever.read_text(encoding="utf-8")
    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=_RaisingCodegen(),
        gate_runner=_passing_runner, apply=False, lever_path=lever)
    assert res.status in ("no_target", "dry_run")
    assert res.committed_sha is None and res.branch is None
    assert res.gate_verdicts == ()
    # the lever file is untouched (no splice attempted)
    assert lever.read_text(encoding="utf-8") == identity


def test_dry_run_persists_candidate_to_outputs_root(tmp_path):
    # Spec §2/§6: the dry-run candidate lever body + gate report must be persisted
    # under outputs_root/codegen-<cycle>/ (they were previously discarded when the
    # identity body was restored), and CodegenResult.body carries the spliced source.
    import json
    _setup(tmp_path)
    lever = _lever_file(tmp_path)
    body = "return dataclasses.replace(arrays, chi=arrays.chi + 0.0)"
    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen(body, rationale="band-tail shift"),
        gate_runner=_passing_runner, apply=False, lever_path=lever)
    assert res.status == "dry_run"
    run_dir = (tmp_path / "out") / "codegen-1"
    lever_out = run_dir / "lever.py"
    report_out = run_dir / "report.json"
    assert lever_out.exists() and report_out.exists()
    # the persisted lever is the SPLICED candidate (def line present, body spliced)
    persisted = lever_out.read_text(encoding="utf-8")
    assert "def adjust_material_arrays(arrays, ctx):" in persisted
    assert body in persisted
    # CodegenResult.body is populated with that same spliced source
    assert res.body == persisted
    # report.json carries the gap, mechanism, rationale, and gate verdicts
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["gap_id"] == "trend:Et_PVK ETL:V_oc"
    assert report["mechanism"] == "missing band-tail Urbach absorption"
    assert report["rationale"] == "band-tail shift"
    assert any(v["name"] == "G6_build" and v["passed"] for v in report["verdicts"])


def test_gates_failed_persists_candidate_to_outputs_root(tmp_path):
    # The gates_failed branch persists the rejected candidate + its failing report
    # so a human can inspect why codegen was refused.
    import json
    _setup(tmp_path)
    lever = _lever_file(tmp_path)
    body = "return dataclasses.replace(arrays, chi=arrays.chi + 0.0)"
    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen(body, rationale="why"),
        gate_runner=_failing_runner, apply=True, committer=None, lever_path=lever)
    assert res.status == "gates_failed"
    run_dir = (tmp_path / "out") / "codegen-1"
    assert (run_dir / "lever.py").exists() and (run_dir / "report.json").exists()
    assert res.body is not None and body in res.body
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert any(v["name"] == "G6_build" and not v["passed"] for v in report["verdicts"])


def test_commit_generated_lever_ignores_untracked(tmp_path):
    # Untracked files in the tree (e.g. outputs/) must NOT trip the dirty-tree
    # guard — only TRACKED modifications other than the lever block a commit.
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    target = repo / "lever.py"
    target.write_text("def adjust_material_arrays(arrays, ctx):\n    return arrays\n", encoding="utf-8")
    git("add", "lever.py"); git("commit", "-q", "-m", "init")
    git("checkout", "-q", "-b", "feat/work")
    target.write_text("def adjust_material_arrays(arrays, ctx):\n    return arrays  # gen\n", encoding="utf-8")
    # an untracked artifact directory (the real-world `outputs/` case)
    (repo / "outputs").mkdir()
    (repo / "outputs" / "report.json").write_text("{}\n", encoding="utf-8")
    lever = GeneratedLever(body="return arrays  # gen", rationale="why")
    branch, sha = commit_generated_lever(target, lever, _gap(), _confirmed_hyp("g", "m"),
                                         [GateVerdict("G6_build", True, "ok")], git_cwd=repo)
    assert branch == "feat/autoloop-gen-trend-et-pvk-etl-v-oc"
    assert sha


def test_fresh_import_after_run_returns_identity(tmp_path):
    # Task 7(a): after a codegen run that splices + imports a NON-identity body
    # into the REAL generated.lever module, a plain import of generated.lever must
    # return the identity result (no stale spliced module left in sys.modules). The
    # production finally block must pop/reload the identity, not just rewrite the
    # file text — otherwise the spliced module lingers in the parent interpreter's
    # sys.modules and a subsequent (non-reloading) import sees the stale body.
    import importlib

    _setup(tmp_path)

    # A gate runner that mimics the in-process flag-ON hook: it imports+reloads the
    # (now spliced) generated.lever so the spliced module lands in sys.modules.
    def _importing_runner(gap, hyp, lever):
        mod = importlib.reload(importlib.import_module("perovskite_sim.autoloop.generated.lever"))
        sentinel = object()
        assert mod.adjust_material_arrays(sentinel, None) is not sentinel  # spliced body live
        return [GateVerdict("G6_build", True, "ok"), GateVerdict("G3_improves", True, "ok")]

    body = "return ('spliced', arrays)"   # non-identity: returns a NEW object
    try:
        res = codegen_top_not_promotable(
            **_common(tmp_path), codegen=FakeCodegen(body, rationale="why"),
            gate_runner=_importing_runner, apply=False)   # real default lever path
        assert res.status == "dry_run"
        # A PLAIN import (no test-side reload) must already see the identity body:
        # production must have evicted the spliced module from sys.modules in its
        # finally, so this re-import re-reads the restored identity file.
        cached = importlib.import_module("perovskite_sim.autoloop.generated.lever")
        sentinel = object()
        assert cached.adjust_material_arrays(sentinel, None) is sentinel
    finally:
        # cleanup: guarantee a clean identity module for later tests regardless of
        # whether production evicted it (re-import, then reload to force a re-read).
        importlib.reload(importlib.import_module("perovskite_sim.autoloop.generated.lever"))


def test_applied_marks_gap_attempted_not_closed(tmp_path):
    # Task 7(b): a codegen lever lands on a SEPARATE feat/autoloop-gen-* branch
    # awaiting human merge — it is NOT live on the working branch, so the gap must
    # be marked "attempted" (not "closed"), recording the branch + sha.
    _setup(tmp_path)
    lever = _lever_file(tmp_path)

    def _committer(target, gen_lever, gap, hyp, verdicts, *, git_cwd=None):
        return ("feat/autoloop-gen-trend-et-pvk-etl-v-oc", "abc123")

    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen("return arrays"),
        gate_runner=_passing_runner, apply=True, committer=_committer, lever_path=lever)
    assert res.status == "applied"
    led = Ledger.load(tmp_path / "ledger")
    g = next(g for g in led.gaps if g.id == "trend:Et_PVK ETL:V_oc")
    assert g.status == "attempted"          # NOT closed — lever awaits human merge
    assert "feat/autoloop-gen-trend-et-pvk-etl-v-oc" in (g.mechanism or "")
    assert "abc123" in (g.mechanism or "")

# perovskite_sim/autoloop/orchestrator.py
from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

from perovskite_sim.autoloop.ablation import run_ablation as _run_ablation
from perovskite_sim.autoloop.codegen import LEVER_TEMPLATE, splice_lever_body, validate_lever_body
from perovskite_sim.autoloop.gates import run_gate_stack, all_passed
from perovskite_sim.autoloop.ladder import run_ladder as _run_ladder
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.promote import propose_promotion, apply_edit, revert_edit
from perovskite_sim.autoloop.provenance import stamp
from perovskite_sim.autoloop.scorecard import gaps_from_score
from perovskite_sim.autoloop.seeds import seed_negative_results
from perovskite_sim.autoloop.types import (
    BoulderProposal, BoulderResult,
    CodegenResult, ConfigEdit, Hypothesis, ImplementResult, LadderResult, NegativeResult, ParityScore,
)


def guardian_once(*, ledger_root: Path, outputs_root: Path,
                  reference_path: Path, config_path: Path,
                  cycle: int, timestamp: str,
                  l0_paths: Optional[list[str]] = None,
                  baseline: Optional[ParityScore] = None,
                  flags: Optional[dict[str, str]] = None,
                  seed: int = 0,
                  run_ladder_fn: Optional[Callable[..., LadderResult]] = None) -> dict:
    """One guardian cycle: sense -> score -> rank -> record -> gate. No landing.

    Returns a JSON-serialisable report and persists the ledger + run artifacts.
    """
    ledger_root = Path(ledger_root)
    run_dir = Path(outputs_root) / f"run-{cycle}"
    run_dir.mkdir(parents=True, exist_ok=True)
    l0_paths = l0_paths or ["tests/unit", "tests/integration"]

    led = Ledger.load(ledger_root)
    seed_negative_results(led)               # idempotent

    runner = run_ladder_fn or _run_ladder
    result = runner(reference_path=reference_path, config_path=config_path, l0_paths=l0_paths)

    gap_ids: list[str] = []
    if result.score is not None:
        existing = {g.id: g for g in led.gaps}
        for g in gaps_from_score(result.score, cycle=cycle):
            if led.is_refuted(g.id):          # never resurface a refuted approach
                continue
            prior = existing.get(g.id)
            if prior is not None and prior.status != "open":
                continue   # preserve attempted/blocked/refuted/closed — do NOT
                           # resurrect to open (the boulder converge termination guard)
            led.add_gap(g)
            gap_ids.append(g.id)

    verdicts = run_gate_stack(result, baseline=baseline)
    prov = stamp(run_id=f"run-{cycle}", config_path=config_path,
                 flags=flags or {}, seed=seed, timestamp=timestamp)

    led.save()
    report = {
        "cycle": cycle,
        "provenance": dataclasses.asdict(prov),
        "l0_pass": result.l0_pass,
        "l1_pass": result.l1_pass,
        "overall": None if result.score is None else result.score.overall,
        "gap_ids": gap_ids,
        "verdicts": [dataclasses.asdict(v) for v in verdicts],
        "gate_passed": all_passed(verdicts),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True),
                                         encoding="utf-8")
    return report


def attribute_top_gap(*, ledger_root: Path, outputs_root: Path,
                      config_path: Path, reference_path: Path,
                      cycle: int, timestamp: str,
                      probe_runner_factory, attributor,
                      flags: Optional[dict[str, str]] = None, seed: int = 0,
                      run_ablation_fn=None, verifier=None) -> Optional[Hypothesis]:
    """One attribution pass: pick the top open gap, ablate, attribute, record.

    ``probe_runner_factory(gap) -> ProbeRunner`` is called with the gap THIS
    function selected, so the probe runner can never target a different gap
    than the one being attributed (single source of truth for the selection).

    Read-only re: code — writes only the ledger + run artifacts.
    """
    ledger_root = Path(ledger_root)
    led = Ledger.load(ledger_root)

    open_gaps = [g for g in led.gaps if g.status == "open"]
    if not open_gaps:
        return None
    gap = max(open_gaps, key=lambda g: g.gap_mag)

    probe_runner = probe_runner_factory(gap)
    run_ablation = run_ablation_fn or _run_ablation
    matrix = run_ablation(gap, probe_runner)
    hyp = attributor.attribute(gap, matrix, led)

    # G5 (Stage 5.2): adjudicate an LLM novel-cause lead before recording it.
    if verifier is not None and hyp.verdict == "uncertain" and hyp.cause != "uncertain":
        hyp = verifier.verify(hyp, gap, matrix)

    led.add_hypothesis(hyp)
    if hyp.verdict == "refuted":
        led.add_negative(NegativeResult(
            approach=hyp.mechanism, why_failed="refuted by G5 multi-skeptic verify",
            evidence=f"attribution cycle {cycle}"))
    if hyp.verdict == "confirmed":
        led.add_gap(gap.with_mechanism(hyp.mechanism))   # add_gap replaces on id
    led.save()

    run_dir = Path(outputs_root) / f"attr-{cycle}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prov = stamp(run_id=f"attr-{cycle}", config_path=config_path,
                 flags=flags or {}, seed=seed, timestamp=timestamp)
    (run_dir / "hypothesis.json").write_text(
        json.dumps(dataclasses.asdict(hyp), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "matrix.json").write_text(
        json.dumps(dataclasses.asdict(matrix), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "provenance.json").write_text(
        json.dumps(dataclasses.asdict(prov), indent=2, sort_keys=True), encoding="utf-8")
    return hyp


def _parse_porcelain_paths(porcelain_lines: list[str]) -> list[str]:
    """Return the file path(s) referenced by each porcelain v1 status line.

    Each line has the form ``XY <path>`` or ``XY <orig> -> <path>`` (renames).
    Paths may be quoted (git quotes names containing spaces or special chars).
    Returns ALL paths mentioned (both sides of a rename) so the caller can
    test whether any path is unrelated to the intended edit.
    """
    paths: list[str] = []
    for raw in porcelain_lines:
        if len(raw) < 4:
            continue
        # columns 0-1: XY status codes; column 2: space; column 3+: path field
        path_field = raw[3:]
        # Handle renames: "old -> new"
        parts = path_field.split(" -> ")
        for part in parts:
            part = part.strip()
            # Git quotes paths that contain spaces/special chars
            if part.startswith('"') and part.endswith('"'):
                part = part[1:-1]
            if part:
                paths.append(part)
    return paths


def commit_promotion(edit: ConfigEdit, gap, hypothesis, verdicts, *, git_cwd=None) -> str:
    """Commit the (already-applied) config edit to the CURRENT branch. Guards:
    refuse main/master, refuse a dirty tree (other than the edited config)."""
    cwd = str(git_cwd) if git_cwd is not None else None
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, cwd=cwd).stdout.strip()
    if branch in ("main", "master"):
        raise RuntimeError(f"refuse to auto-commit on '{branch}'; create an autoloop branch first")

    # Determine config path relative to the git root for exact comparison.
    # Git porcelain output is always relative to the repo root, so we must
    # compute the same relative path.  We resolve both sides to handle symlinks
    # (macOS /var → /private/var, etc.).
    git_root_raw = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                  capture_output=True, text=True, cwd=cwd).stdout.strip()
    git_root = str(Path(git_root_raw).resolve())
    cfg_abs = str(Path(edit.config_path).resolve())
    try:
        cfg_rel = str(Path(cfg_abs).relative_to(git_root))
    except ValueError:
        # Config is outside the repo tree; fall back to absolute path so the
        # comparison below still works (porcelain lines will never match it,
        # flagging everything as stray — the right conservative behaviour).
        cfg_rel = cfg_abs

    # Do NOT strip() the whole output — porcelain lines begin with two status
    # chars (e.g. " M") and a leading strip() would consume the first char.
    dirty = [ln for ln in subprocess.run(["git", "status", "--porcelain"],
                                         capture_output=True, text=True,
                                         cwd=cwd).stdout.splitlines() if ln]

    stray: list[str] = []
    for line in dirty:
        for path in _parse_porcelain_paths([line]):
            # Porcelain paths are relative to the repo root.  Resolve them
            # against the resolved git_root so symlinks don't confuse us, then
            # compare EXACTLY against cfg_abs (never a substring match).
            path_abs = str((Path(git_root) / path).resolve())
            if path_abs != cfg_abs:
                stray.append(line)
                break  # one stray match per dirty line is enough
    if stray:
        raise RuntimeError(f"refuse to commit: working tree has unrelated changes: {stray[:3]}")

    # Verify the index contains ONLY the config file before committing.
    # This catches a race where an external process staged something between
    # the status check and the add.
    subprocess.run(["git", "add", edit.config_path], cwd=cwd, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                            capture_output=True, text=True, cwd=cwd).stdout.strip().splitlines()
    extra = [p for p in staged
             if str((Path(git_root) / p).resolve()) != cfg_abs]
    if extra:
        raise RuntimeError(
            f"refuse to commit: index contains unexpected staged files: {extra[:3]}")

    gate_summary = " ".join(f"{v.name}{'✓' if v.passed else '✗'}" for v in verdicts)
    msg = (f"feat(autoloop): promote {edit.device_key} (closes gap {gap.id})\n\n"
           f"Auto-implemented by autoloop Stage 3 from a confirmed hypothesis.\n"
           f"Mechanism: {hypothesis.mechanism}\n"
           f"Gates: {gate_summary}\n"
           f"Gap: {gap.id} | Hypothesis-cycle: {hypothesis.cycle}")
    # Scope the commit to the exact config path so a pre-staged unrelated file
    # can never be swept in even if the stray-file guard above is bypassed.
    subprocess.run(["git", "commit", "-q", "-m", msg, "--", edit.config_path],
                   cwd=cwd, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"],
                          capture_output=True, text=True, cwd=cwd).stdout.strip()


def implement_top_confirmed(*, ledger_root: Path, outputs_root: Path,
                            config_path, reference_path,
                            cycle: int, timestamp: str, apply: bool = False,
                            gate_runner, committer=None, git_cwd=None) -> ImplementResult:
    """One implement pass: top confirmed gap -> propose -> apply -> gate ->
    (revert+report | commit). Read-only on solver code."""
    led = Ledger.load(Path(ledger_root))
    confirmed_ids = {h.gap_id for h in led.hypotheses if h.verdict == "confirmed"}
    # Exclude coverage:* gaps: their solarlab_val is a bracket COUNT, not a closure
    # %, so gap_baseline_badness would be nonsensical. (They also never carry a
    # SOLARLAB_* lever, so propose_promotion already returns None — this is a
    # defensive guard against that coupling silently breaking.)
    candidates = [g for g in led.gaps
                  if g.status == "open" and g.id in confirmed_ids
                  and not g.id.startswith("coverage:")]
    if not candidates:
        return ImplementResult("no_confirmed", None, None, (), None)
    gap = max(candidates, key=lambda g: g.gap_mag)
    hyp = next(h for h in led.hypotheses if h.gap_id == gap.id and h.verdict == "confirmed")

    edit = propose_promotion(hyp, led, config_path)
    if edit is None:
        return ImplementResult("not_promotable", gap.id, None, (), None, note=hyp.mechanism)

    apply_edit(edit)
    try:
        verdicts = list(gate_runner(edit, gap, hyp))

        if not all(v.passed for v in verdicts):
            revert_edit(edit)
            led.add_negative(NegativeResult(
                approach=hyp.mechanism,
                why_failed="gate(s) failed: " + ",".join(v.name for v in verdicts if not v.passed),
                evidence=f"autoloop Stage 3 implement cycle {cycle}"))
            led.save()
            return ImplementResult("gates_failed", gap.id, edit.device_key, tuple(verdicts), None)

        if apply:
            commit = committer or commit_promotion
            sha = commit(edit, gap, hyp, verdicts, git_cwd=git_cwd)
            led.add_gap(gap.with_status("closed").with_mechanism(hyp.mechanism))
            led.save()
            return ImplementResult("applied", gap.id, edit.device_key, tuple(verdicts), sha)

        revert_edit(edit)
        return ImplementResult("dry_run", gap.id, edit.device_key, tuple(verdicts), None)
    except Exception:
        revert_edit(edit)
        raise


def _boulder_top_open(ledger_root):
    led = Ledger.load(ledger_root)
    open_gaps = [g for g in led.gaps if g.status == "open"]
    return max(open_gaps, key=lambda g: g.gap_mag) if open_gaps else None


def _boulder_proposal(ledger_root, gap, result) -> BoulderProposal:
    led = Ledger.load(ledger_root)
    hyp = next((h for h in led.hypotheses if h.gap_id == gap.id), None)
    status = result.status if result is not None else "no_confirmed"
    return BoulderProposal(
        gap_id=gap.id,
        cause=(hyp.cause if hyp else "uncertain"),
        mechanism=(hyp.mechanism if hyp else ""),
        device_key=(result.device_key if result is not None else None),
        gate_status=status,
        landed=(status == "applied"))


def _boulder_mark_attempted(ledger_root, gap):
    led = Ledger.load(ledger_root)
    led.add_gap(gap.with_status("attempted"))
    led.save()


def run_boulder(*, ledger_root, outputs_root, timestamp, converge: bool = False,
                parity_target: float = 0.90, max_cycles: int = 10, reject_streak: int = 3,
                sense, attribute, implement) -> BoulderResult:
    """Continuous driver. sweep (dry-run) drains current gaps into proposals;
    converge auto-applies + re-senses + loops to a stop condition. The three
    step-functions are injected (CLI binds real ones; tests inject fakes)."""
    ledger_root = Path(ledger_root)
    proposals: list[BoulderProposal] = []

    if not converge:
        sense(0)
        cycle = 0
        while True:
            gap = _boulder_top_open(ledger_root)
            if gap is None:
                break
            attribute(cycle)
            result = implement(cycle, False)
            proposals.append(_boulder_proposal(ledger_root, gap, result))
            _boulder_mark_attempted(ledger_root, gap)
            cycle += 1
        return BoulderResult("sweep", cycle, tuple(proposals), 0, "sweep_complete", None)

    cycle = 0
    landed = 0
    reject = 0
    overall = None
    stop = "cap"
    while cycle < max_cycles:
        overall = sense(cycle)
        if overall is not None and overall >= parity_target:
            stop = "success"
            break
        gap = _boulder_top_open(ledger_root)
        if gap is None:
            stop = "drained"
            break
        attribute(cycle)
        result = implement(cycle, True)
        proposals.append(_boulder_proposal(ledger_root, gap, result))
        if result is not None and result.status == "applied":
            landed += 1
            reject = 0
        else:
            _boulder_mark_attempted(ledger_root, gap)
            if result is not None and result.status == "gates_failed":
                reject += 1
        if reject >= reject_streak:
            stop = "halt"
            break
        cycle += 1
    return BoulderResult("converge", cycle, tuple(proposals), landed, stop, overall)


_DEFAULT_LEVER_PATH = Path(__file__).resolve().parent / "generated" / "lever.py"


def _gap_slug(gap_id: str) -> str:
    """git-branch-safe slug from a gap id."""
    slug = "".join(c if c.isalnum() else "-" for c in gap_id.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:48] or "lever"


def commit_generated_lever(target_path, lever, gap, hypothesis, verdicts, *, git_cwd=None):
    """Commit the (already-written) generated lever body to a FRESH
    feat/autoloop-gen-<slug> branch off the current HEAD, then return to the
    original branch (git restores the identity body on it). Refuses main/master
    and a dirty tree other than the lever file; never pushes. -> (branch, sha)."""
    cwd = str(git_cwd) if git_cwd is not None else None

    def _git(*args, check=True):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, check=check)

    origin = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if origin in ("main", "master"):
        raise RuntimeError(f"refuse to branch/commit from '{origin}'; create an autoloop branch first")

    git_root = str(Path(_git("rev-parse", "--show-toplevel").stdout.strip()).resolve())
    tgt_abs = str(Path(target_path).resolve())
    # Only TRACKED modifications block a commit. Untracked entries ("?? <path>",
    # e.g. an artifact `outputs/` dir) are present in essentially every real
    # working tree and are never swept into the lever-scoped commit, so they must
    # not trip the guard. The commit is path-scoped to the lever file regardless.
    dirty = [ln for ln in _git("status", "--porcelain").stdout.splitlines()
             if ln and not ln.startswith("??")]
    stray = []
    for line in dirty:
        for p in _parse_porcelain_paths([line]):
            if str((Path(git_root) / p).resolve()) != tgt_abs:
                stray.append(line)
                break
    if stray:
        raise RuntimeError(f"refuse to commit: working tree has unrelated changes: {stray[:3]}")

    branch = f"feat/autoloop-gen-{_gap_slug(gap.id)}"
    _git("checkout", "-b", branch)
    try:
        rel = str(Path(tgt_abs).relative_to(git_root))
        _git("add", rel)
        gate_summary = " ".join(f"{v.name}{'✓' if v.passed else '✗'}" for v in verdicts)
        msg = (f"feat(autoloop): codegen lever for gap {gap.id}\n\n"
               f"Auto-generated by autoloop Stage 5.3 from a confirmed, non-promotable hypothesis.\n"
               f"Mechanism: {hypothesis.mechanism}\n"
               f"Rationale: {lever.rationale}\n"
               f"Gates: {gate_summary}\n"
               f"Gap: {gap.id} | Hypothesis-cycle: {hypothesis.cycle}\n"
               f"Flag-gated (autoloop_generated_lever / SOLARLAB_AUTOLOOP_GEN); default OFF. "
               f"Review before merge.")
        _git("commit", "-q", "-m", msg, "--", rel)
        sha = _git("rev-parse", "HEAD").stdout.strip()
    finally:
        _git("checkout", origin)   # restore origin branch (target reverts to identity)
    return branch, sha


def codegen_top_not_promotable(*, ledger_root: Path, outputs_root: Path, config_path,
                               reference_path, cycle: int, timestamp: str, codegen,
                               gate_runner, apply: bool = False, committer=None,
                               git_cwd=None, lever_path=None) -> CodegenResult:
    """One codegen pass: top confirmed gap whose mechanism is NOT promotable ->
    LLM writes a lever body into autoloop/generated/lever.py -> gate -> (restore +
    report | fresh-branch commit). Always restores the identity body on the
    working branch (try/finally)."""
    led = Ledger.load(Path(ledger_root))
    confirmed_ids = {h.gap_id for h in led.hypotheses if h.verdict == "confirmed"}
    candidates = [g for g in led.gaps
                  if g.status == "open" and g.id in confirmed_ids
                  and not g.id.startswith("coverage:")]

    def _hyp(g):
        return next(h for h in led.hypotheses if h.gap_id == g.id and h.verdict == "confirmed")

    # codegen handles ONLY the not-promotable confirmed gaps (promotable -> Stage 3).
    candidates = [g for g in candidates if propose_promotion(_hyp(g), led, config_path) is None]
    if not candidates:
        return CodegenResult("no_target", None, None, (), None, None)
    gap = max(candidates, key=lambda g: g.gap_mag)
    hyp = _hyp(gap)
    if led.is_refuted(hyp.mechanism):
        return CodegenResult("refuted", gap.id, None, (), None, None)

    # A codegen/LLM runtime failure must NEVER crash the dispatch (spec §7):
    # degrade to a no-op result, mirroring LLMAttributor's degrade pattern.
    try:
        lever = codegen.generate(gap, hyp, None)
    except Exception as exc:
        return CodegenResult("no_target", gap.id, None, (), None,
                             f"codegen generate failed: {exc!r}")
    target = Path(lever_path) if lever_path is not None else _DEFAULT_LEVER_PATH
    identity = target.read_text(encoding="utf-8")
    try:
        # Re-validate at the splice boundary: the body may come from any Codegen
        # (e.g. a FakeCodegen that does not self-validate). Reject before it ever
        # lands on disk / gets imported.
        validate_lever_body(lever.body)
        target.write_text(splice_lever_body(LEVER_TEMPLATE, lever.body), encoding="utf-8")
        verdicts = list(gate_runner(gap, hyp, lever))
        if not all(v.passed for v in verdicts):
            led.add_negative(NegativeResult(
                approach=hyp.mechanism,
                why_failed="codegen gate(s) failed: " + ",".join(v.name for v in verdicts if not v.passed),
                evidence=f"autoloop Stage 5.3 codegen cycle {cycle}"))
            led.save()
            return CodegenResult("gates_failed", gap.id, None, tuple(verdicts), None, lever.rationale)
        if apply:
            commit = committer or commit_generated_lever
            # A commit failure (dirty-tree refusal, git error, …) must not crash
            # the dispatch: degrade to a dry-run-equivalent with the error in the
            # rationale. The finally block still restores the identity body.
            try:
                branch, sha = commit(target, lever, gap, hyp, verdicts, git_cwd=git_cwd)
            except Exception as exc:
                return CodegenResult("gates_failed", gap.id, None, tuple(verdicts), None,
                                     f"commit failed: {exc!r}")
            led.add_gap(gap.with_status("closed").with_mechanism(hyp.mechanism))
            led.save()
            return CodegenResult("applied", gap.id, branch, tuple(verdicts), sha, lever.rationale)
        return CodegenResult("dry_run", gap.id, None, tuple(verdicts), None, lever.rationale)
    finally:
        target.write_text(identity, encoding="utf-8")   # ALWAYS restore identity on the working branch

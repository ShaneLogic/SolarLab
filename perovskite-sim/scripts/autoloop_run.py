#!/usr/bin/env python
"""Autoloop CLI (Stage 1 guardian + Stage 2 attribution).

Run one sense-and-record guardian cycle against the SCAPS reference:

    python scripts/autoloop_run.py --once

Exits non-zero if the gate stack fails (CI-friendly).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from perovskite_sim.autoloop import guardian_once

REPO_ROOT = Path(__file__).resolve().parents[1]                      # perovskite-sim/
DEFAULT_REFERENCE = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"
DEFAULT_LEDGER = REPO_ROOT.parent / "docs" / "autoloop" / "ledger"   # SolarLab/docs/...
DEFAULT_OUTPUTS = REPO_ROOT.parent / "outputs" / "autoloop"
DEFAULT_BASELINE = REPO_ROOT / "tests" / "integration" / "autoloop_parity_baseline.json"


def iso_timestamp_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Autoloop (Stage 1 guardian + Stage 2 attribution)")
    ap.add_argument("--once", action="store_true", help="run a single cycle")
    ap.add_argument("--attribute", action="store_true",
                    help="run one attribution pass on the top open gap")
    ap.add_argument("--implement", action="store_true",
                    help="run one implement pass on the top confirmed gap (dry-run unless --apply)")
    ap.add_argument("--apply", action="store_true",
                    help="with --implement: commit the change to the current branch if all gates pass")
    ap.add_argument("--boulder", action="store_true",
                    help="run the continuous boulder (sweep dry-run unless --converge)")
    ap.add_argument("--converge", action="store_true",
                    help="with --boulder/--implement: auto-apply landable fixes and loop")
    ap.add_argument("--llm", action="store_true",
                    help="use the LLM attributor (fallback on gaps the heuristic can't diagnose)")
    ap.add_argument("--llm-model", default="sonnet", help="model for --llm (default sonnet)")
    ap.add_argument("--verify", action="store_true",
                    help="adjudicate LLM novel-cause leads with the G5 multi-skeptic verifier")
    ap.add_argument("--codegen", action="store_true",
                    help="codegen a flag-gated lever for a confirmed, non-promotable cause "
                         "(dry-run unless --apply; commits to a fresh feat/autoloop-gen-* branch)")
    ap.add_argument("--search", action="store_true",
                    help="run a parity-gated, advisory device-design search")
    ap.add_argument("--budget", type=int, default=50, help="design-search eval budget")
    ap.add_argument("--parity-target", type=float, default=0.90)
    ap.add_argument("--max-cycles", type=int, default=10)
    ap.add_argument("--reject-streak", type=int, default=3)
    ap.add_argument("--cycle", type=int, default=0)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--l0-paths", nargs="*", default=["tests/unit/autoloop"])
    return ap.parse_args(argv)


def _load_baseline(path: Path):
    if not path.exists():
        return None
    from perovskite_sim.autoloop.types import ParityScore, SweepScore
    raw = json.loads(path.read_text(encoding="utf-8"))
    per = {k: SweepScore(**v) for k, v in raw.get("per_sweep", {}).items()}
    return ParityScore(overall=raw["overall"], base_deltas=raw.get("base_deltas", {}),
                       per_sweep=per)


def _build_attributor(ns):
    from perovskite_sim.autoloop.attribution import HeuristicAttributor
    if not getattr(ns, "llm", False):
        return HeuristicAttributor()
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.llm_attribution import LLMAttributor
    return LLMAttributor(ClaudeCliRuntime(model=ns.llm_model))


def _build_verifier(ns):
    if not getattr(ns, "verify", False):
        return None
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.verify import MultiSkepticVerifier
    return MultiSkepticVerifier(ClaudeCliRuntime(model=ns.llm_model))


def _build_codegen(ns):
    if not getattr(ns, "codegen", False):
        return None
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.codegen import ClaudeCodegen
    return ClaudeCodegen(ClaudeCliRuntime(model=ns.llm_model))


def _build_codegen_gate_runner(*, config, reference, golden_runner):
    """Build the live --codegen gate stack, threading the gap the orchestrator
    selects into every flag-ON probe.

    The orchestrator calls ``gate_runner(gap, hyp, lever)`` with the gap it just
    picked, so we bind that real ``gap`` into both the flag-ON parity probe (G6)
    and the realized-badness probe (G3). Two correctness requirements that the
    pre-fix inline closures violated:

      * ``SubprocessProbeRunner`` must be built with a REAL ``gap`` — its ``.run``
        dereferences ``self.gap.sweep`` / ``.metric`` / ``.kind``, so ``gap=None``
        raises ``AttributeError`` on every flag-ON probe and G6 never passes.
      * The variant must use ``measure="gap"`` — the only mode ``_probe_worker``
        supports (``"base"`` falls through to the absolute-gap path and KeyErrors
        on the ``None`` metric).
    """
    import math
    from perovskite_sim.autoloop.gates_impl import make_codegen_gate_runner
    from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner

    def gate_runner(gap, hyp, lever):
        def _flag_on():
            # flag-ON parity sweep must complete to a finite badness scalar.
            try:
                val = SubprocessProbeRunner(
                    config_path=config, reference_path=reference, gap=gap).run(
                    {"env_flags": {"SOLARLAB_AUTOLOOP_GEN": "1"},
                     "jv_overrides": {}, "measure": "gap"})
                return (math.isfinite(val), f"badness={val}")
            except Exception as exc:  # noqa: BLE001 — a crashed probe is a fail signal
                return (False, repr(exc))

        def _realized(g):
            return SubprocessProbeRunner(
                config_path=config, reference_path=reference, gap=g).run(
                {"env_flags": {"SOLARLAB_AUTOLOOP_GEN": "1"},
                 "jv_overrides": {}, "measure": "gap"})

        inner = make_codegen_gate_runner(golden_runner=golden_runner,
                                         flag_on_runner=_flag_on,
                                         realized_badness=_realized)
        return inner(gap, hyp, lever)

    return gate_runner


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])

    if ns.search:
        import dataclasses
        from perovskite_sim.autoloop.search import run_design_search, SearchNotTrusted
        try:
            result = run_design_search(
                config_path=ns.config, reference_path=ns.reference,
                outputs_root=ns.outputs_root, timestamp=iso_timestamp_utc(),
                budget=ns.budget, parity_target=ns.parity_target)
        except SearchNotTrusted as exc:
            print(json.dumps({"search": None, "error": str(exc)}))
            return 1
        print(json.dumps({"search": {
            "parity_overall": result.parity_overall, "n_evaluated": result.n_evaluated,
            "best": (dataclasses.asdict(result.best) if result.best else None),
            "top": [dataclasses.asdict(t) for t in result.trials[:5]]}},
            indent=2, sort_keys=True, default=str))
        return 0

    if ns.boulder:
        import dataclasses
        from perovskite_sim.autoloop.orchestrator import (
            run_boulder, guardian_once, attribute_top_gap, implement_top_confirmed)
        from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner
        from perovskite_sim.autoloop.gates_impl import make_implement_gate_runner
        from perovskite_sim.autoloop.provenance import _git

        if ns.converge:
            branch = _git("rev-parse", "--abbrev-ref", "HEAD")
            if branch in ("main", "master"):
                print(json.dumps({"boulder": None,
                                  "error": f"refuse --boulder --converge on '{branch}'; "
                                           "create an autoloop branch first"}))
                return 1

        _attr = _build_attributor(ns)
        _verifier = _build_verifier(ns)

        def sense(cycle):
            rep = guardian_once(ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
                                reference_path=ns.reference, config_path=ns.config,
                                cycle=cycle, timestamp=iso_timestamp_utc(),
                                l0_paths=ns.l0_paths, baseline=None)
            return rep["overall"]

        def attribute(cycle):
            attribute_top_gap(ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
                              config_path=ns.config, reference_path=ns.reference, cycle=cycle,
                              timestamp=iso_timestamp_utc(),
                              probe_runner_factory=lambda g: SubprocessProbeRunner(
                                  config_path=ns.config, reference_path=ns.reference, gap=g),
                              attributor=_attr, verifier=_verifier)

        def implement(cycle, apply):
            def _measure(edit, gap):
                return SubprocessProbeRunner(config_path=edit.config_path,
                                             reference_path=ns.reference, gap=gap).run(
                    {"env_flags": {}, "jv_overrides": {}, "measure": "gap"})
            return implement_top_confirmed(
                ledger_root=ns.ledger_root, outputs_root=ns.outputs_root, config_path=ns.config,
                reference_path=ns.reference, cycle=cycle, timestamp=iso_timestamp_utc(),
                gate_runner=make_implement_gate_runner(measure_badness=_measure), apply=apply)

        result = run_boulder(ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
                             timestamp=iso_timestamp_utc(), converge=ns.converge,
                             parity_target=ns.parity_target, max_cycles=ns.max_cycles,
                             reject_streak=ns.reject_streak,
                             sense=sense, attribute=attribute, implement=implement)
        print(json.dumps({"boulder": dataclasses.asdict(result)}, indent=2,
                         sort_keys=True, default=str))
        return 1 if result.stop_reason == "halt" else 0

    if ns.attribute:
        import dataclasses
        from perovskite_sim.autoloop.orchestrator import attribute_top_gap
        from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner
        hyp = attribute_top_gap(
            ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
            config_path=ns.config, reference_path=ns.reference, cycle=ns.cycle,
            timestamp=iso_timestamp_utc(),
            probe_runner_factory=lambda g: SubprocessProbeRunner(
                config_path=ns.config, reference_path=ns.reference, gap=g),
            attributor=_build_attributor(ns), verifier=_build_verifier(ns))
        if hyp is None:
            print(json.dumps({"attributed": None, "reason": "no open gaps"}))
            return 0
        print(json.dumps({"attributed": dataclasses.asdict(hyp)}, indent=2, sort_keys=True))
        return 0

    if ns.implement:
        import dataclasses
        from perovskite_sim.autoloop.orchestrator import implement_top_confirmed
        from perovskite_sim.autoloop.gates_impl import make_implement_gate_runner
        from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner

        def _measure_realized(edit, gap):
            # G4: re-measure the gap's badness on the flag-on (edited) config.
            runner = SubprocessProbeRunner(config_path=edit.config_path,
                                           reference_path=ns.reference, gap=gap)
            return runner.run({"env_flags": {}, "jv_overrides": {}, "measure": "gap"})

        gate_runner = make_implement_gate_runner(measure_badness=_measure_realized)

        hyp_result = implement_top_confirmed(
            ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
            config_path=ns.config, reference_path=ns.reference, cycle=ns.cycle,
            timestamp=iso_timestamp_utc(), gate_runner=gate_runner, apply=ns.apply)
        print(json.dumps({"implement": dataclasses.asdict(hyp_result)}, indent=2,
                         sort_keys=True, default=str))
        return 1 if hyp_result.status == "gates_failed" and ns.apply else 0

    if ns.codegen:
        import dataclasses
        from perovskite_sim.autoloop.orchestrator import codegen_top_not_promotable
        from perovskite_sim.autoloop.ladder import run_l0

        codegen = _build_codegen(ns)
        if codegen is None:
            print(json.dumps({"codegen": None, "error": "use --codegen together with --llm-capable runtime"}))
            return 1

        def _golden():
            return run_l0(["tests/regression"])

        gate_runner = _build_codegen_gate_runner(
            config=ns.config, reference=ns.reference, golden_runner=_golden)
        result = codegen_top_not_promotable(
            ledger_root=ns.ledger_root, outputs_root=ns.outputs_root, config_path=ns.config,
            reference_path=ns.reference, cycle=ns.cycle, timestamp=iso_timestamp_utc(),
            codegen=codegen, gate_runner=gate_runner, apply=ns.apply)
        print(json.dumps({"codegen": dataclasses.asdict(result)}, indent=2, sort_keys=True, default=str))
        return 1 if result.status == "gates_failed" and ns.apply else 0

    report = guardian_once(
        ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
        reference_path=ns.reference, config_path=ns.config,
        cycle=ns.cycle, timestamp=iso_timestamp_utc(),
        l0_paths=ns.l0_paths, baseline=_load_baseline(ns.baseline),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

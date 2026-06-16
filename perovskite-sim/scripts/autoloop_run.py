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


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])

    if ns.attribute:
        import dataclasses
        from perovskite_sim.autoloop.orchestrator import attribute_top_gap
        from perovskite_sim.autoloop.attribution import HeuristicAttributor
        from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner
        hyp = attribute_top_gap(
            ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
            config_path=ns.config, reference_path=ns.reference, cycle=ns.cycle,
            timestamp=iso_timestamp_utc(),
            probe_runner_factory=lambda g: SubprocessProbeRunner(
                config_path=ns.config, reference_path=ns.reference, gap=g),
            attributor=HeuristicAttributor())
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

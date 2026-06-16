# perovskite_sim/autoloop/orchestrator.py
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable, Optional

from perovskite_sim.autoloop.gates import run_gate_stack, all_passed
from perovskite_sim.autoloop.ladder import run_ladder as _run_ladder
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.provenance import stamp
from perovskite_sim.autoloop.scorecard import gaps_from_score
from perovskite_sim.autoloop.seeds import seed_negative_results
from perovskite_sim.autoloop.types import LadderResult, ParityScore


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
        for g in gaps_from_score(result.score, cycle=cycle):
            if led.is_refuted(g.id):         # never resurface a refuted approach
                continue
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

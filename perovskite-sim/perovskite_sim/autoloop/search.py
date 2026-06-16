# perovskite_sim/autoloop/search.py
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.autoloop.ladder import DEFAULT_JV_KWARGS, build_run_callables
from perovskite_sim.autoloop.scorecard import score_parity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesignKnob:
    axis: str
    low: float
    high: float
    scale: str = "linear"     # "linear" | "log"


@dataclass(frozen=True)
class Trial:
    design: dict
    pce: float
    bracketed: bool


@dataclass(frozen=True)
class SearchResult:
    best: Optional[Trial]
    trials: tuple
    n_evaluated: int
    parity_overall: float
    budget: int


class SearchNotTrusted(RuntimeError):
    """Raised when the model's parity is below the trust threshold."""


DEFAULT_DESIGN_SPACE = [
    DesignKnob("etl_delta_ec_eV", -0.3, 0.3, "linear"),
    DesignKnob("htl_delta_ev_eV", -0.3, 0.3, "linear"),
    DesignKnob("etl_doping_cm3", 1e15, 1e19, "log"),
    DesignKnob("absorber_defect_density_cm3", 1e14, 1e17, "log"),
]


class Optimizer(Protocol):
    def optimize(self, objective: Callable[[dict], tuple], space: list, budget: int) -> tuple: ...


class RandomSearchOptimizer:
    """Seeded uniform random search over the design space (no dependency)."""

    def __init__(self, *, seed: int = 0):
        self.seed = seed

    def _sample(self, rng: random.Random, space: list) -> dict:
        out = {}
        for k in space:
            r = rng.random()
            if k.scale == "log":
                out[k.axis] = 10.0 ** (math.log10(k.low) + r * (math.log10(k.high) - math.log10(k.low)))
            else:
                out[k.axis] = k.low + r * (k.high - k.low)
        return out

    def optimize(self, objective, space, budget) -> tuple:
        if budget <= 0:
            raise ValueError(f"budget must be > 0, got {budget}")
        rng = random.Random(self.seed)
        trials = []
        for _ in range(budget):
            design = self._sample(rng, space)
            pce, bracketed = objective(design)
            trials.append(Trial(design=design, pce=pce, bracketed=bracketed))
        return tuple(sorted(trials, key=lambda t: t.pce, reverse=True))


def make_design_objective(config_path, jv_kwargs: dict):
    """Returns objective(design: dict) -> (pce, bracketed). Applies the design to
    the base stack and runs a J-V; a failed/unbracketed design scores PCE=0."""
    base = load_scaps_yaml(config_path)

    def objective(design: dict) -> tuple:
        try:
            stack = apply_sweep_point(base, SweepPoint("design", "multi", "design", dict(design)))
            m = run_jv_sweep(stack, **jv_kwargs).metrics_fwd
            return (m.PCE if m.voc_bracketed else 0.0, bool(m.voc_bracketed))
        except Exception as exc:                       # logged, not swallowed
            logger.warning("design eval failed %s: %r", design, exc)
            return (0.0, False)

    return objective


import dataclasses
import json


def _default_parity(config_path, reference_path) -> Callable[[], float]:
    def fn() -> float:
        run_point, base_point = build_run_callables(config_path)
        return score_parity(reference_path=reference_path, config_path=config_path,
                            run_point=run_point, base_point=base_point).overall
    return fn


def run_design_search(*, config_path, reference_path, outputs_root, timestamp,
                      space=None, budget: int = 50, parity_target: float = 0.90,
                      optimizer=None, objective=None, parity_fn=None) -> SearchResult:
    """Parity-gated, advisory design search. Refuses unless parity >= target,
    then runs the optimizer and writes an advisory report. Applies nothing."""
    space = space if space is not None else DEFAULT_DESIGN_SPACE
    overall = (parity_fn or _default_parity(config_path, reference_path))()
    if overall < parity_target:
        raise SearchNotTrusted(
            f"model parity {overall:.3f} < target {parity_target} — "
            "refuse to optimize an untrusted model")

    optimizer = optimizer or RandomSearchOptimizer()
    objective = objective or make_design_objective(config_path, DEFAULT_JV_KWARGS)
    trials = optimizer.optimize(objective, space, budget)
    result = SearchResult(best=(trials[0] if trials else None), trials=trials,
                          n_evaluated=len(trials), parity_overall=overall, budget=budget)

    run_dir = Path(outputs_root) / f"search-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "parity_overall": overall, "parity_target": parity_target, "budget": budget,
        "n_evaluated": result.n_evaluated,
        "best": (dataclasses.asdict(result.best) if result.best else None),
        "trials": [dataclasses.asdict(t) for t in trials],
        "note": "ADVISORY — proposed designs, nothing applied to any config",
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True),
                                         encoding="utf-8")
    return result

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

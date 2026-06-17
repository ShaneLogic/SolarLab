# tests/integration/test_autoloop_llm.py
import importlib.util
import os
import shutil
import sys
from pathlib import Path
import pytest

CLI = Path(__file__).resolve().parents[2] / "scripts" / "autoloop_run.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("autoloop_run", CLI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoloop_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_attributor_default_is_heuristic():
    mod = _load_cli()
    ns = mod.parse_args(["--attribute"])
    from perovskite_sim.autoloop.attribution import HeuristicAttributor
    assert isinstance(mod._build_attributor(ns), HeuristicAttributor)


def test_build_attributor_llm_flag():
    mod = _load_cli()
    ns = mod.parse_args(["--attribute", "--llm", "--llm-model", "sonnet"])
    from perovskite_sim.autoloop.llm_attribution import LLMAttributor
    assert isinstance(mod._build_attributor(ns), LLMAttributor)


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("claude") or not os.environ.get("SOLARLAB_LLM_SMOKE"),
                    reason="opt-in real-LLM smoke (set SOLARLAB_LLM_SMOKE=1 + claude installed)")
def test_real_llm_attributor_produces_lead(tmp_path):
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.llm_attribution import LLMAttributor
    from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix
    from perovskite_sim.autoloop.ledger import Ledger
    gap = Gap(id="trend:Et_PVK ETL:V_oc", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
              solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
              status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)
    matrix = AblationMatrix(gap_id=gap.id, baseline_val=40.0, probes=(
        AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 39.9, -0.1, True),
        AblationProbe("grid_n80", "grid", {}, 40.0, 40.1, 0.1, True),
        AblationProbe("dark_jsc", "limiting", {}, 0.0, 0.0, 0.0, True)))
    hyp = LLMAttributor(ClaudeCliRuntime(model="sonnet")).attribute(gap, matrix, Ledger(root=tmp_path))
    assert hyp.verdict == "uncertain"
    assert hyp.mechanism and hyp.cause in {"bug", "numerics", "physics", "data"}

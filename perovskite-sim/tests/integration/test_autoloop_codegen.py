# tests/integration/test_autoloop_codegen.py
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


def test_build_codegen_none_without_flag():
    mod = _load_cli()
    assert mod._build_codegen(mod.parse_args([])) is None


def test_build_codegen_with_flag():
    mod = _load_cli()
    from perovskite_sim.autoloop.codegen import ClaudeCodegen
    assert isinstance(mod._build_codegen(mod.parse_args(["--codegen"])), ClaudeCodegen)


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("claude") or not os.environ.get("SOLARLAB_LLM_SMOKE"),
                    reason="opt-in real-LLM smoke (set SOLARLAB_LLM_SMOKE=1 + claude installed)")
def test_real_codegen_returns_lever():
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.codegen import ClaudeCodegen, GeneratedLever
    from perovskite_sim.autoloop.types import Gap, Hypothesis
    gap = Gap(id="g", metric="V_oc", sweep="x", sweep_point=0.0, solarlab_val=30.0,
              reference_val=70.0, gap_mag=0.4, kind="trend", status="open",
              found_cycle=0, last_attempt_cycle=0, mechanism=None)
    hyp = Hypothesis(gap_id="g", cause="physics",
                     mechanism="missing band-tail Urbach absorption", verdict="confirmed")
    lev = ClaudeCodegen(ClaudeCliRuntime(model="sonnet")).generate(gap, hyp, None)
    assert isinstance(lev, GeneratedLever) and isinstance(lev.body, str) and lev.body

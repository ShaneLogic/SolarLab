# tests/integration/test_autoloop_g5.py
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


def test_build_verifier_none_without_flag():
    mod = _load_cli()
    assert mod._build_verifier(mod.parse_args(["--attribute"])) is None


def test_build_verifier_with_flag():
    mod = _load_cli()
    from perovskite_sim.autoloop.verify import MultiSkepticVerifier
    assert isinstance(mod._build_verifier(mod.parse_args(["--attribute", "--verify"])),
                      MultiSkepticVerifier)


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("claude") or not os.environ.get("SOLARLAB_LLM_SMOKE"),
                    reason="opt-in real-LLM smoke (set SOLARLAB_LLM_SMOKE=1 + claude installed)")
def test_real_g5_verify_returns_verdict(tmp_path):
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.verify import MultiSkepticVerifier
    from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix, Hypothesis
    gap = Gap(id="g", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0, solarlab_val=30.0,
              reference_val=70.0, gap_mag=0.4, kind="trend", status="open",
              found_cycle=0, last_attempt_cycle=0, mechanism=None)
    matrix = AblationMatrix(gap_id="g", baseline_val=40.0,
                            probes=(AblationProbe("f", "flag", {}, 40.0, 39.9, -0.1, True),))
    lead = Hypothesis(gap_id="g", cause="physics",
                      mechanism="missing band-tail Urbach absorption", verdict="uncertain")
    out = MultiSkepticVerifier(ClaudeCliRuntime(model="sonnet")).verify(lead, gap, matrix)
    assert out.verdict in {"confirmed", "refuted", "uncertain"}

# tests/unit/autoloop/test_cli_codegen_probe.py
"""Task 3 (Stage 5.3 remediation): the live --codegen probe wiring must build a
real Gap (not None) and use the only _probe_worker-supported measure mode
("gap"), so the flag-ON G6 check actually runs instead of always returning
(False, AttributeError) from dereferencing None.gap.sweep."""
import importlib.util
import sys
from pathlib import Path

import pytest

from perovskite_sim.autoloop.types import Gap

CLI = Path(__file__).resolve().parents[3] / "scripts" / "autoloop_run.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("autoloop_run", CLI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoloop_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def _gap():
    return Gap(id="trend:Nd_ETL:V_oc", metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_codegen_gate_closures_pass_real_gap_and_measure_gap(monkeypatch, tmp_path):
    """Build the CLI --codegen gate closures and run them against a stubbed probe.

    Asserts: (1) the SubprocessProbeRunner is constructed with a real Gap (never
    None), (2) every variant requests a _probe_worker-supported measure mode
    ("gap" for G6/G3, "dark" for the G2 limiting check — "base" is unrecognised),
    and (3) the flag-ON G6 runner returns (True, ...) for a finite stub instead
    of (False, AttributeError)."""
    mod = _load_cli()

    captured = {"gaps": [], "measures": []}

    class _StubRunner:
        def __init__(self, *, config_path, reference_path, gap, **kw):
            captured["gaps"].append(gap)

        def run(self, variant):
            captured["measures"].append(variant.get("measure"))
            return 5.0   # finite badness

    # Patch the SubprocessProbeRunner symbol the CLI imports inside the --codegen block.
    monkeypatch.setattr("perovskite_sim.autoloop.subprocess_probe.SubprocessProbeRunner",
                        _StubRunner)

    gap = _gap()
    gate_runner = mod._build_codegen_gate_runner(
        config=tmp_path / "c.yaml", reference=tmp_path / "r.json",
        golden_runner=lambda: (True, "golden green"))

    # Drive the full gate stack the way the orchestrator does: gate_runner(gap, hyp, lever).
    verdicts = gate_runner(gap, None, None)

    # G6 ran the flag-ON probe and passed (finite, not an AttributeError).
    g6 = next(v for v in verdicts if v.name == "G6_build")
    assert g6.passed is True, g6.reason

    # Spec §5: the G2 limiting verdict sits on the stack between G6 and G3 and
    # was actually evaluated (its dark-Jsc probe ran).
    assert [v.name for v in verdicts] == ["G6_build", "G2_limiting", "G3_improves"]

    # Every probe was built with the REAL gap (never None).
    assert captured["gaps"], "probe was never constructed"
    assert all(g is not None for g in captured["gaps"])
    assert all(getattr(g, "sweep", None) == "Nd_ETL" for g in captured["gaps"])

    # Every probe used a _probe_worker-supported measure mode (gap for G6/G3,
    # dark for the G2 limiting check) — never the unrecognised "base".
    assert captured["measures"], "probe.run was never called"
    assert set(captured["measures"]) <= {"gap", "dark"}
    assert "dark" in captured["measures"]  # the G2 limiting probe ran

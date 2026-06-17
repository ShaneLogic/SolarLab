# perovskite_sim/autoloop/gates_impl.py
from __future__ import annotations

from typing import Callable

from perovskite_sim.autoloop.types import GateVerdict


def gate_g4_reconciles(predicted_delta: float, realized_delta: float,
                       *, tol: float = 0.5) -> GateVerdict:
    """G4 honest gate for flag-promotion: the promoted flag's measured benefit
    must actually materialize. Deltas are badness changes (negative = improvement).
    Pass iff realized improves (negative) AND reconciles with the predicted
    magnitude within ``tol`` (relative)."""
    if predicted_delta >= 0:
        return GateVerdict("G4_reconcile", False,
                           f"no predicted improvement (Δpred {predicted_delta:.3g} >= 0)")
    improved = realized_delta < 0
    reconciles = abs(realized_delta - predicted_delta) <= tol * abs(predicted_delta)
    passed = improved and reconciles
    return GateVerdict("G4_reconcile", passed,
                       f"Δpred {predicted_delta:.3g}, Δreal {realized_delta:.3g}, "
                       f"tol {tol} (improved={improved}, reconciles={reconciles})")


def gate_g0_bit_identical(golden_runner: Callable[[], tuple[bool, str]]) -> GateVerdict:
    """G0: the legacy/golden regression suite must stay green with the edit
    applied (legacy tier forces the flag off, so it holds by construction; this
    verifies it). ``golden_runner() -> (ok, detail)`` is injected."""
    ok, detail = golden_runner()
    return GateVerdict("G0_legacy_bit_identical", ok, detail)


def gap_baseline_badness(gap) -> float:
    """The gap's pre-fix badness (lower = closer to SCAPS), from its sense-time
    value. Trend gaps store closure% in ``solarlab_val`` (badness = 100 -
    closure); absolute gaps store the signed metric delta (badness = |delta|)."""
    if gap.kind == "trend":
        return 100.0 - gap.solarlab_val
    return abs(gap.solarlab_val)


def make_implement_gate_runner(*, measure_badness: Callable, l0_runner: Callable = None,
                               regression_paths=None):
    """Build the Stage-3 implement gate stack: G1 (numerics) + G0 (legacy
    regression) + G4 (the promoted flag's measured benefit reconciles its
    prediction). ``measure_badness(edit, gap) -> float`` re-measures the gap's
    badness with the flag-on (edited) config; injected so unit tests run without
    the solver. Full-parity G3 is the loop's post-land re-sense, not run here
    (too slow per-implement); 'no regression elsewhere' is covered by G0."""
    from perovskite_sim.autoloop.ladder import run_l0
    l0 = l0_runner or run_l0
    reg = regression_paths or ["tests/regression"]

    def gate_runner(edit, gap, hyp):
        verdicts = []
        ok1, d1 = l0(["tests/unit/autoloop"])
        verdicts.append(GateVerdict("G1_numerics", ok1, d1))
        ok0, d0 = l0(reg)
        verdicts.append(gate_g0_bit_identical(lambda: (ok0, d0)))
        realized = measure_badness(edit, gap)
        realized_delta = realized - gap_baseline_badness(gap)
        verdicts.append(gate_g4_reconciles(hyp.predicted_delta, realized_delta))
        return verdicts

    return gate_runner


def gate_g6_build(*, golden_runner: Callable[[], tuple[bool, str]],
                  flag_on_runner: Callable[[], tuple[bool, str]],
                  lever_module: str = "perovskite_sim.autoloop.generated.lever") -> GateVerdict:
    """G6 (codegen): the generated lever must (1) import/compile, (2) keep the
    legacy suite green with the flag OFF (reuse G0's golden suite), (3) run a
    flag-ON parity sweep to a finite, voc-bracketed result. Cheap fail-fast
    before the G1/G3 physics checks. Runners are injected so unit tests run
    without the solver."""
    import importlib
    try:
        mod = importlib.import_module(lever_module)
        importlib.reload(mod)
    except Exception as exc:                       # SyntaxError/ImportError/etc.
        return GateVerdict("G6_build", False, f"lever import/compile failed: {exc!r}")
    ok_off, d_off = golden_runner()
    if not ok_off:
        return GateVerdict("G6_build", False, f"flag-OFF not bit-identical: {d_off}")
    ok_on, d_on = flag_on_runner()
    if not ok_on:
        return GateVerdict("G6_build", False, f"flag-ON run failed: {d_on}")
    return GateVerdict("G6_build", True, f"import ok; OFF[{d_off}]; ON[{d_on}]")


def make_codegen_gate_runner(*, golden_runner, flag_on_runner, realized_badness,
                             lever_module: str = "perovskite_sim.autoloop.generated.lever"):
    """Codegen gate stack: G6 (build/import + flag-OFF bit-identical + flag-ON
    runs finite) then G3 (flag-ON badness improves vs the gap's sense-time
    baseline). Short-circuits if G6 fails. Returns a callable(gap, hyp, lever).
    `realized_badness(gap) -> float` re-measures the flag-ON gap badness; injected
    so unit tests run without the solver. (G4 reconcile is N/A: a brand-new lever
    has no prior ablation prediction — G3-improves is the honest benefit check.)"""
    def gate_runner(gap, hyp, lever):
        verdicts = [gate_g6_build(golden_runner=golden_runner, flag_on_runner=flag_on_runner,
                                  lever_module=lever_module)]
        if verdicts[0].passed:
            realized = realized_badness(gap)
            base = gap_baseline_badness(gap)
            verdicts.append(GateVerdict("G3_improves", realized < base,
                                        f"badness {realized:.3g} vs baseline {base:.3g}"))
        return verdicts

    return gate_runner

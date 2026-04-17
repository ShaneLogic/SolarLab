"""Default V_max helper for run_jv_sweep.

Phase 1 threads ``DeviceStack.compute_V_bi()`` into the default upper voltage
for J-V sweeps so heterostacks whose V_oc exceeds the manually-configured
``stack.V_bi`` field still produce a forward sweep that crosses J = 0. The
formula lives in the private helper :func:`_default_V_max` so it can be
audited and regressed without running a full sweep:

    V_upper = max(stack.compute_V_bi() * 1.3, 1.4)

These tests pin both branches using the configs that actually exercise them:

- ``nip_MAPbI3.yaml`` (legacy, chi = Eg = 0 so ``compute_V_bi`` falls back to
  the manual ``stack.V_bi`` = 1.1 V): 1.1 × 1.3 = 1.43 V > 1.4 V ⇒
  **V_bi_eff branch wins**.

- ``ionmonger_benchmark.yaml`` (heterostack with chi/Eg set, giving an
  empirical Fermi-level difference V_bi_eff ≈ 0.857 V): 0.857 × 1.3 ≈
  1.114 V < 1.4 V ⇒ **1.4 V floor branch wins**.

It looks counterintuitive that the heterostack is the one hitting the floor,
but that is exactly why the floor exists — the IonMonger benchmark has a
modest band-offset V_bi that, without the backstop, would close the sweep
before V_oc on illuminated runs (V_oc for this stack is around 1.0 V).

Scope note
----------
This file exercises only the helper, not a full sweep. A sweep-level V_oc
regression lives in the integration lane (``test_jv_regression.py``) because
running a Radau integration from V=0 to V_upper is minutes of wall clock,
not the sub-second budget of the unit suite.
"""
from __future__ import annotations

import pytest

from perovskite_sim.experiments.jv_sweep import _default_V_max
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_default_V_max_picks_V_bi_eff_branch_on_legacy_mapbi3():
    """Legacy MAPbI3 (chi=Eg=0 ⇒ V_bi_eff = stack.V_bi = 1.1 V):
    1.1 × 1.3 = 1.43 V clears the 1.4 V floor, so the formula picks V_bi_eff*1.3.
    """
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    V_bi_eff = stack.compute_V_bi()
    V_upper = _default_V_max(stack)
    # Defend the assumption that this config actually selects the V_bi_eff*1.3
    # branch. If a future edit drops stack.V_bi below ~1.08 V the floor would
    # win silently and this test would stop regressing what its name promises.
    assert V_bi_eff * 1.3 > 1.4, (
        f"Test assumption broken: V_bi_eff*1.3 = {V_bi_eff * 1.3:.3f} V "
        f"is at or below the 1.4 V floor, so this test no longer exercises "
        "the V_bi_eff*1.3 branch. Pick a higher-V_bi legacy config or raise "
        "the manual V_bi in nip_MAPbI3.yaml."
    )
    assert V_upper == pytest.approx(V_bi_eff * 1.3, rel=1e-12), (
        f"_default_V_max mispicked branch: got {V_upper:.4f} V, "
        f"expected V_bi_eff*1.3 = {V_bi_eff * 1.3:.4f} V"
    )


def test_default_V_max_honours_floor_on_ionmonger_heterostack():
    """IonMonger heterostack: band-offset V_bi_eff ≈ 0.857 V, so V_bi_eff*1.3 ≈
    1.114 V lies below the 1.4 V floor ⇒ the floor branch wins.
    """
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    V_bi_eff = stack.compute_V_bi()
    V_upper = _default_V_max(stack)
    # Guard the assumption that this config actually needs the floor; without
    # it, the test is a trivial duplicate of the V_bi_eff-branch case.
    assert V_bi_eff * 1.3 < 1.4, (
        f"Test assumption broken: V_bi_eff*1.3 = {V_bi_eff * 1.3:.3f} V "
        f"already exceeds the 1.4 V floor, so the floor branch is not being "
        "exercised by this config. Pick a lower-V_bi heterostack config."
    )
    assert V_upper == pytest.approx(1.4, rel=1e-12), (
        f"_default_V_max ignored the 1.4 V floor: got {V_upper:.4f} V, "
        f"expected 1.4 V (floor > V_bi_eff*1.3 = {V_bi_eff * 1.3:.4f})"
    )


def test_default_V_max_is_monotone_in_V_bi_eff():
    """Sanity: doubling the effective V_bi should non-strictly grow V_upper
    (either the floor still wins, or the V_bi_eff*1.3 branch grows linearly).
    Guards against regressions that, e.g., accidentally clamp the upper
    voltage to ``stack.V_bi``.
    """
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    V1 = _default_V_max(stack)

    # Build a lightweight shim with doubled compute_V_bi to avoid mutating the
    # real dataclass. Duck-typed — _default_V_max only reads compute_V_bi().
    class _DoubleVbi:
        def __init__(self, base):
            self._base = base
        def compute_V_bi(self):
            return 2.0 * self._base.compute_V_bi()

    V2 = _default_V_max(_DoubleVbi(stack))
    assert V2 >= V1, (
        f"_default_V_max is non-monotone in V_bi_eff: "
        f"V1(V_bi_eff)={V1:.4f} V, V2(2*V_bi_eff)={V2:.4f} V"
    )
    # Doubled value must be above the floor (2 × 0.857 × 1.3 ≈ 2.23 V), so the
    # V_bi_eff*1.3 branch is selected; check by construction.
    assert V2 == pytest.approx(2.0 * stack.compute_V_bi() * 1.3, rel=1e-12)

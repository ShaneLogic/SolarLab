"""Phase E1.7 — ``interface_defect_N_t_cm2`` sweep key routing.

The SCAPS partner PDF sweep over the PVK/ETL interface defect density
(p12) needs a sweep handler that drives ``DeviceStack.interface_defects[k]``
SRV via σ·v_th·N_t_areal — NOT the Phase 4a layer-trap-profile knob
(``MaterialParams.trap_N_t_interface``) which the legacy
``interface_trap_density_cm3`` key routes to.

Routing the SCAPS PVK/ETL interface defect density sweep to the wrong
code path is why ``outputs/scaps_validation_e1_5/report.md`` shows the
PVK/ETL interface defect density sweep as ``0 mV / mismatch`` — the
defect doesn't move because the wrong knob is being twiddled. E1.7
adds the right handler.

Contract:
1. ``interface_defect_N_t_cm2`` sweep key modulates
   ``DeviceStack.interfaces[k]`` SRV and the stored electrostatic
   ``InterfaceDefect.N_t_cm2``. Recorded asymmetric base SRVs are scaled by
   the same density ratio; legacy defects fall back to σ=1e-15 cm² and
   v_th=1e7 cm/s.
2. The existing ``InterfaceDefect.E_t_eV`` is preserved and nonpositive or
   nonfinite densities fail closed.
3. The legacy Phase 4a ``MaterialParams.trap_N_t_interface`` is NOT
   touched (independent code path).
4. Target alias: ``pvk/etl``, ``htl/pvk``, ``left``, ``right`` —
   resolved against ``DeviceStack.layers`` adjacent role pair.
5. Sweeping a target with no pre-existing ``InterfaceDefect`` raises
   ValueError — the YAML must declare the defect first.
"""
from __future__ import annotations

import pytest

from perovskite_sim.models.device import InterfaceDefect
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)


def _baseline_scaps_stack():
    """scaps_mirror.yaml on main already declares a PVK/ETL InterfaceDefect
    via the Phase E1.5 ``interfaces:`` block, so the loader returns a
    DeviceStack with interface_defects[1] populated."""
    return load_scaps_yaml("configs/scaps_mirror.yaml")


def test_interface_defect_N_t_cm2_modulates_devicestack_interfaces_srv():
    """N_t_cm2 sweep value translates to SCAPS-direct SRV via
    σ·v_th·N_t_areal, written onto DeviceStack.interfaces[k]."""
    stack = _baseline_scaps_stack()
    # PVK/ETL is the last electrical interface (k = 1 in the 3-layer stack).
    pt = SweepPoint("p", "n", "1e10", {"interface_defect_N_t_cm2": 1.0e10})
    swept = apply_sweep_point(stack, pt)
    # σ=1e-15 cm² · v_th=1e7 cm/s · 1e10 cm⁻² · 1e-2 → SRV = 1 m/s
    assert swept.interfaces[-1] == pytest.approx((1.0, 1.0), rel=1.0e-9)


def test_interface_defect_N_t_cm2_preserves_E_t():
    """Sweeping N_t does NOT touch the E_t on InterfaceDefect."""
    stack = _baseline_scaps_stack()
    baseline_E_t = stack.interface_defects[-1].E_t_eV
    pt = SweepPoint("p", "n", "1e9", {"interface_defect_N_t_cm2": 1.0e9})
    swept = apply_sweep_point(stack, pt)
    assert isinstance(swept.interface_defects[-1], InterfaceDefect)
    assert swept.interface_defects[-1].E_t_eV == pytest.approx(baseline_E_t)


def test_interface_defect_N_t_cm2_updates_charge_capacity_and_preserves_srv_ratio():
    stack = _baseline_scaps_stack()
    base_pair = stack.interfaces[-1]
    base_density = stack.interface_defects[-1].N_t_cm2
    target_density = base_density * 2.5
    pt = SweepPoint(
        "p",
        "n",
        "scaled",
        {"interface_defect_N_t_cm2": target_density},
    )

    swept = apply_sweep_point(stack, pt)

    assert swept.interface_defects[-1].N_t_cm2 == pytest.approx(target_density)
    assert swept.interfaces[-1][0] == pytest.approx(base_pair[0] * 2.5)
    assert swept.interfaces[-1][1] == pytest.approx(base_pair[1] * 2.5)
    assert swept.interfaces[-1][1] / swept.interfaces[-1][0] == pytest.approx(
        base_pair[1] / base_pair[0]
    )


def test_interface_defect_N_t_cm2_does_not_touch_phase_4a_trap_profile():
    """The Phase 4a ``MaterialParams.trap_N_t_interface`` (per-layer
    edge-tapered trap profile) is on a separate code path. Sweeping the
    new key must leave it identically equal to the YAML baseline."""
    stack = _baseline_scaps_stack()
    baseline_etl_trap_N_t = stack.layers[-1].params.trap_N_t_interface
    pt = SweepPoint("p", "n", "1e12", {"interface_defect_N_t_cm2": 1.0e12})
    swept = apply_sweep_point(stack, pt)
    assert swept.layers[-1].params.trap_N_t_interface == baseline_etl_trap_N_t


def test_interface_defect_N_t_cm2_raises_when_no_baseline_defect(tmp_path):
    """Sweeping a target with no pre-existing InterfaceDefect in the
    DeviceStack raises ValueError — the YAML must declare the defect
    first, so the sweep cannot silently turn a defect-free interface
    into a defected one."""
    import dataclasses

    base = _baseline_scaps_stack()
    # Strip both interfaces and interface_defects to simulate a YAML
    # with no top-level ``interfaces:`` block.
    stripped = dataclasses.replace(base, interfaces=(), interface_defects=())
    pt = SweepPoint("p", "n", "1e10", {"interface_defect_N_t_cm2": 1.0e10})
    with pytest.raises(ValueError, match="no InterfaceDefect"):
        apply_sweep_point(stripped, pt)


@pytest.mark.parametrize("density", [0.0, -1.0, float("nan"), float("inf")])
def test_interface_defect_N_t_cm2_rejects_nonpositive_or_nonfinite(density):
    stack = _baseline_scaps_stack()
    point = SweepPoint(
        "invalid",
        "n",
        "invalid",
        {"interface_defect_N_t_cm2": density},
    )

    with pytest.raises(ValueError, match="finite and positive"):
        apply_sweep_point(stack, point)


def test_legacy_interface_trap_density_cm3_still_routes_to_phase_4a():
    """Backward compat: the existing ``interface_trap_density_cm3`` key
    must keep modulating the Phase 4a ``MaterialParams.trap_N_t_interface``
    (volumetric per-layer edge-tapered profile). E1.7 only ADDS a new
    handler; it does not retire the old one."""
    stack = _baseline_scaps_stack()
    pt = SweepPoint(
        "p", "d", "1e13",
        {"interface_trap_density_cm3": 1.0e13, "interface_trap_decay_nm": 10.0},
    )
    swept = apply_sweep_point(stack, pt)
    # Phase 4a path: PVK trap_N_t_interface (the absorber layer's edge
    # trap density) should now be non-None and equal to 1e13 cm⁻³ in SI.
    pvk_params = swept.layers[1].params
    assert pvk_params.trap_N_t_interface is not None
    assert pvk_params.trap_N_t_interface == pytest.approx(1.0e19)  # 1e13 cm⁻³ → 1e19 m⁻³

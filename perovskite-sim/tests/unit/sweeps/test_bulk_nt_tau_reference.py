"""Bulk-N_t sweep: tau must ratio-scale off the config's DECLARED base density.

Regression for the Figures 6-7 flatness artifact (2026-06): the
``absorber_defect_density_cm3`` sweep scaled tau by
``trap_tau_reference_density_m3 / N_t`` with a hardcoded absolute reference of
1e22 m^-3 (1e16 cm^-3), while the loader's base tau corresponds to the
config's declared bulk defects (2 x 1e12 cm^-3 = 2e18 m^-3 on
scaps_mirror_v2). Every swept point in the partner range (1e9-1e15 cm^-3)
therefore ran with a bulk lifetime 10-1e7x LONGER than the baseline — the
sweep never increased the recombination it claimed to sweep, and V_oc was
flat to 0.0 mV by construction (SCAPS shows -39/-11 mV).

Fix: the loader stores the combined declared density on
``MaterialParams.trap_N_t_bulk``, and the sweep uses it as the tau-scaling
reference when present (the 1e22 absolute remains the fallback for configs
that declare no bulk defects, preserving legacy behaviour).
"""
from __future__ import annotations

import pytest

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)

_V2 = "configs/scaps_mirror_v2.yaml"


def _absorber(stack):
    return next(L.params for L in stack.layers if L.role == "absorber")


def test_loader_records_declared_bulk_density():
    """v2 PVK declares two 1e12 cm^-3 defects -> combined 2e18 m^-3."""
    p = _absorber(load_scaps_yaml(_V2))
    assert p.trap_N_t_bulk == pytest.approx(2.0e18)


def test_sweep_scales_tau_off_declared_base():
    """tau_swept = tau_base * (N_t_base / N_t_swept) with the DECLARED base:
    at N_t = 1e15 cm^-3 (1e21 m^-3) the scale is 2e18/1e21 = 1/500."""
    base = load_scaps_yaml(_V2)
    tau0 = _absorber(base).tau_n
    swept = apply_sweep_point(
        base,
        SweepPoint("p", "nt", "1e15", {"absorber_defect_density_cm3": 1e15}),
    )
    assert _absorber(swept).tau_n == pytest.approx(tau0 * 2.0e18 / 1.0e21)
    # base-density point: scale 2e18/1e18 = 2 (the swept sheet varies one of
    # the two declared defects; the combined-tau model carries a bounded 2x
    # softness at/below the base, where the response is sub-mV anyway)
    at_base = apply_sweep_point(
        base,
        SweepPoint("p", "nt", "1e12", {"absorber_defect_density_cm3": 1e12}),
    )
    assert _absorber(at_base).tau_n == pytest.approx(tau0 * 2.0)


def test_fallback_reference_for_configs_without_declared_defects():
    """Configs with trap_N_t_bulk=None keep the legacy 1e22 m^-3 reference."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    assert _absorber(base).trap_N_t_bulk is None
    tau0 = _absorber(base).tau_n
    swept = apply_sweep_point(
        base,
        SweepPoint("p", "nt", "1e16", {"absorber_defect_density_cm3": 1e16}),
    )
    assert _absorber(swept).tau_n == pytest.approx(tau0 * 1.0e22 / 1.0e22)

"""Stage 4 — end-to-end FULL-tier J-V on the flagship migrated preset.

Stages 1-3 pinned the *construction* contract of the migrated TMM presets
(chi/Eg populated, manual V_bi matches compute_V_bi, FULL tier activates
interface_faces). This module pins the *runtime* contract: an actual J-V
sweep on nip_MAPbI3_tmm in FULL tier produces all four figures of merit
inside a physically-reasonable window.

Flagship choice: ``nip_MAPbI3_tmm.yaml`` is the Stage 1a preset, first and
most battle-tested of the four migrated configs. It is also the only one
currently exercised by the slow regression suite (test_tmm_baseline), so
the two checks defend different axes (regression pins a single J_sc value;
this one pins the full FoM window under FULL tier).

Guarded under ``pytest.mark.slow`` because the J-V sweep drives Radau on
a ~160 node grid and takes ~20 s wall — well above the 0.5 s budget the
default unit/integration suite aims for. Run via ``pytest -m slow``.
"""
from __future__ import annotations

import dataclasses

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml

FLAGSHIP_PRESET = "configs/nip_MAPbI3_tmm.yaml"

# Physically-reasonable FoM window for MAPbI3 in FULL tier. Bounds are set
# wider than the current measured values (V_oc ~1.08, J_sc ~210, FF ~0.81,
# PCE ~18.3%) so that ordinary solver/physics drift does not cause churn,
# but tight enough to catch a real regression (e.g. TE cap misconfigured,
# V_bi drift, contact BC sign flip — all of which collapse at least one
# FoM by >20%).
V_OC_BAND = (0.95, 1.20)  # volts
J_SC_BAND = (180.0, 240.0)  # A/m^2 — same band as test_tmm_integration
FF_BAND = (0.65, 0.90)
PCE_BAND = (0.14, 0.22)  # fractional, i.e. 14%-22%

# Forward-vs-reverse hysteresis cap: V_oc split should stay under 50 mV
# on this ion-enabled preset. A larger split is usually the ion sweep
# integrator diverging rather than real physics.
HYSTERESIS_V_OC_TOL = 0.05  # volts


@pytest.mark.slow
def test_flagship_full_tier_jv_figures_of_merit():
    stack = load_device_from_yaml(FLAGSHIP_PRESET)
    stack_full = dataclasses.replace(stack, mode="full")
    result = run_jv_sweep(stack_full, n_points=21)

    for label, metrics in (("fwd", result.metrics_fwd), ("rev", result.metrics_rev)):
        assert V_OC_BAND[0] <= metrics.V_oc <= V_OC_BAND[1], (
            f"{label} V_oc={metrics.V_oc:.3f} V outside {V_OC_BAND}"
        )
        assert J_SC_BAND[0] <= metrics.J_sc <= J_SC_BAND[1], (
            f"{label} J_sc={metrics.J_sc:.2f} A/m^2 outside {J_SC_BAND}"
        )
        assert FF_BAND[0] <= metrics.FF <= FF_BAND[1], (
            f"{label} FF={metrics.FF:.3f} outside {FF_BAND}"
        )
        assert PCE_BAND[0] <= metrics.PCE <= PCE_BAND[1], (
            f"{label} PCE={metrics.PCE*100:.2f}% outside "
            f"{(PCE_BAND[0]*100, PCE_BAND[1]*100)}"
        )

    voc_split = abs(result.metrics_fwd.V_oc - result.metrics_rev.V_oc)
    assert voc_split < HYSTERESIS_V_OC_TOL, (
        f"V_oc hysteresis {voc_split*1000:.1f} mV exceeds "
        f"{HYSTERESIS_V_OC_TOL*1000:.0f} mV — ion sweep may be diverging"
    )

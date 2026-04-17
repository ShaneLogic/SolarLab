"""Photon recycling regression: radiative-limit V_oc boost.

With SRH and Auger killed, V_oc in MAPbI3-like stacks is radiative-
limited and the only handle on it is B_rad. Turning photon recycling on
(``B_rad_eff = B_rad × P_esc`` with ``P_esc ≈ 1/(4n²αd) ≈ 0.05`` for a
400 nm MAPbI3) should lift V_oc by roughly ``V_T·ln(1/P_esc) ≈ 75 mV``
at 300 K. The literature window for MAPbI3 is ~40–100 mV (Pazos-Outón
2016, Richter 2017); we gate on [40, 100] mV here.

Comparing PR on vs PR off is apples-to-apples only when TMM is on in
both runs, so we construct two ``SimulationMode`` instances identical
except for ``use_photon_recycling`` and inject them via monkeypatched
``resolve_mode``. This keeps G(x) (TMM-computed) bit-identical across
the two V_oc measurements.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import K_B, Q
from perovskite_sim.models import mode as _mode_mod
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.mode import FULL
from perovskite_sim.experiments.suns_voc import run_suns_voc


CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "radiative_limit.yaml"
)


@pytest.mark.slow
def test_photon_recycling_voc_boost_in_window(monkeypatch):
    """ΔV_oc(PR on vs PR off) in MAPbI3 radiative-limit preset ∈ [40, 100] mV."""
    stack = load_device_from_yaml(CONFIG)

    # Two SimulationMode instances, identical apart from PR flag. FULL
    # already has TMM, TE, etc. all on.
    mode_pr_on = FULL
    mode_pr_off = replace(FULL, use_photon_recycling=False)

    def _voc_under_mode(active_mode):
        """Monkeypatch resolve_mode to return active_mode, then measure V_oc."""
        monkeypatch.setattr(
            _mode_mod, "resolve_mode", lambda _arg: active_mode,
        )
        # Module-level re-import paths also resolve via this attribute,
        # so patch the solver re-export too.
        import perovskite_sim.solver.mol as mol_mod
        monkeypatch.setattr(
            mol_mod, "resolve_mode", lambda _arg: active_mode,
        )
        # One illumination level = 1 sun, coarse grid for speed.
        res = run_suns_voc(
            stack, suns_levels=(1.0,), N_grid=40, t_settle=1e-3,
        )
        return float(res.V_oc[0])

    voc_on = _voc_under_mode(mode_pr_on)
    voc_off = _voc_under_mode(mode_pr_off)
    delta_mV = (voc_on - voc_off) * 1000.0

    # Photon recycling should raise V_oc (B_rad_eff = B_rad·P_esc < B_rad).
    assert voc_on > voc_off, (
        f"photon recycling lowered V_oc: on={voc_on:.4f} V, off={voc_off:.4f} V"
    )
    # Literature window for MAPbI3 radiative-limit PR boost.
    assert 40.0 <= delta_mV <= 100.0, (
        f"ΔV_oc = {delta_mV:.1f} mV outside literature window [40, 100]. "
        f"voc_off={voc_off:.4f} V, voc_on={voc_on:.4f} V"
    )


@pytest.mark.slow
def test_photon_recycling_voc_below_sq_ceiling(monkeypatch):
    """V_oc with PR on is between 1.0 V and the Shockley-Queisser bound.

    A rough SQ ceiling for MAPbI3 at 1 sun and T=300 K is ~1.33 V (Eg=1.6,
    full black-body reciprocity). The radiative-limit preset with PR on
    should sit above 1.0 V (the measured MAPbI3 V_oc) but strictly below
    the SQ bound. With B_rad at detailed-balance scale and Auger/SRH off
    the diode can exceed ``V_bi = 0.79 V`` — carriers accumulate in the
    degenerate regime — which is physically correct for the rad-limit.
    """
    stack = load_device_from_yaml(CONFIG)
    monkeypatch.setattr(_mode_mod, "resolve_mode", lambda _arg: FULL)
    import perovskite_sim.solver.mol as mol_mod
    monkeypatch.setattr(mol_mod, "resolve_mode", lambda _arg: FULL)

    res = run_suns_voc(stack, suns_levels=(1.0,), N_grid=40, t_settle=1e-3)
    voc = float(res.V_oc[0])

    # Lower sanity gate: comfortably above the measured MAPbI3 V_oc of
    # ~1.1 V. (If this drops below 1.0 V the PR scaling has likely
    # regressed to ``1 - P_esc``.)
    assert voc >= 1.0, f"radiative-limit V_oc too low: {voc:.4f} V"
    # Upper gate: strict SQ ceiling for Eg = 1.6 eV is ~1.33 V; allow a
    # few kT slack for numerical drift.
    V_T = K_B * 300.0 / Q
    sq_ceiling = 1.33
    assert voc <= sq_ceiling + 5 * V_T, (
        f"radiative-limit V_oc exceeds SQ ceiling: {voc:.4f} V > "
        f"{sq_ceiling + 5 * V_T:.4f} V"
    )

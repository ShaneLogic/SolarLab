"""Phase 3.1b regression — self-consistent radiative reabsorption.

These tests pin the physics contract for the per-RHS reabsorption source
G_rad(x) = (1 - P_esc) · <B·n·p>_absorber / thickness applied uniformly
onto absorber nodes.

Guarantees:

1. **Activation flag propagates.**
   When both ``use_photon_recycling`` and ``use_radiative_reabsorption``
   are on and TMM supplies an absorber with Eg > 0, the MaterialArrays
   cache must report ``has_radiative_reabsorption=True`` and carry a
   non-empty absorber-mask tuple. When either flag is off the cache must
   fall back to the Phase 3.1 "B_rad *= P_esc" path and report
   ``has_radiative_reabsorption=False``.

2. **V_oc equivalence with Phase 3.1 at open circuit.**
   At open circuit the absorber n·p is quasi-uniform (flat quasi-Fermi
   levels across the bulk), so the time-averaged reabsorption source
   exactly cancels the extra radiative loss that Phase 3.1 collapsed into
   the B_rad scaling. The two implementations therefore produce the same
   V_oc to within a few mV. If this drifts it means the per-RHS source
   or the build-time scaling is mis-normalised.

3. **V_oc monotonicity versus PR-off.**
   Phase 3.1b must preserve the same V_oc boost as Phase 3.1 relative to
   the no-PR baseline — i.e. photon recycling is still doing the work of
   lifting V_oc into the literature window [40, 100] mV.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models import mode as _mode_mod
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.models.mode import FULL
from perovskite_sim.solver.mol import build_material_arrays
import perovskite_sim.solver.mol as mol_mod
from perovskite_sim.experiments.suns_voc import run_suns_voc


CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "radiative_limit.yaml"
)


def _grid(stack, n_per_layer: int = 20) -> np.ndarray:
    return multilayer_grid(
        [Layer(thickness=l.thickness, N=n_per_layer) for l in electrical_layers(stack)],
        alpha=3.0,
    )


def test_radiative_reabsorption_activates_when_flag_on(monkeypatch):
    """FULL + TMM preset + Eg > 0 absorber ⇒ has_radiative_reabsorption True,
    absorber mask cache populated, and B_rad kept at its bulk value."""
    stack = load_device_from_yaml(CONFIG)
    monkeypatch.setattr(_mode_mod, "resolve_mode", lambda _arg: FULL)
    monkeypatch.setattr(mol_mod, "resolve_mode", lambda _arg: FULL)

    x = _grid(stack)
    mat = build_material_arrays(x, stack)

    assert mat.has_radiative_reabsorption is True
    assert len(mat.absorber_masks) >= 1
    assert len(mat.absorber_p_esc) == len(mat.absorber_masks)
    assert len(mat.absorber_thicknesses) == len(mat.absorber_masks)
    # At least one absorber node must be captured by the mask.
    total_abs_nodes = sum(int(np.count_nonzero(m)) for m in mat.absorber_masks)
    assert total_abs_nodes > 0
    # P_esc in [0, 1] and strictly < 1 for a non-trivial absorber.
    for p_esc in mat.absorber_p_esc:
        assert 0.0 < p_esc < 1.0


def test_radiative_reabsorption_disabled_falls_back_to_phase_3_1(monkeypatch):
    """With use_radiative_reabsorption=False (FAST tier or custom mode),
    the cache must fall back to the Phase 3.1 build-time B_rad scaling —
    ``has_radiative_reabsorption`` is False, no absorber masks are stored.
    """
    stack = load_device_from_yaml(CONFIG)
    mode_31_only = replace(
        FULL,
        use_photon_recycling=True,
        use_radiative_reabsorption=False,
    )
    monkeypatch.setattr(_mode_mod, "resolve_mode", lambda _arg: mode_31_only)
    monkeypatch.setattr(mol_mod, "resolve_mode", lambda _arg: mode_31_only)

    x = _grid(stack)
    mat = build_material_arrays(x, stack)

    assert mat.has_radiative_reabsorption is False
    assert mat.absorber_masks == ()


@pytest.mark.slow
def test_radiative_reabsorption_matches_phase_3_1_at_voc(monkeypatch):
    """At open circuit, Phase 3.1b and Phase 3.1 produce the same V_oc
    within ~5 mV because the absorber n·p is quasi-uniform and the two
    PR formulations are equivalent in that limit.
    """
    stack = load_device_from_yaml(CONFIG)

    mode_31 = replace(
        FULL,
        use_photon_recycling=True,
        use_radiative_reabsorption=False,
    )
    mode_31b = replace(
        FULL,
        use_photon_recycling=True,
        use_radiative_reabsorption=True,
    )

    def _voc_under_mode(active_mode):
        monkeypatch.setattr(
            _mode_mod, "resolve_mode", lambda _arg: active_mode,
        )
        monkeypatch.setattr(
            mol_mod, "resolve_mode", lambda _arg: active_mode,
        )
        res = run_suns_voc(
            stack, suns_levels=(1.0,), N_grid=40, t_settle=1e-3,
        )
        return float(res.V_oc[0])

    voc_31 = _voc_under_mode(mode_31)
    voc_31b = _voc_under_mode(mode_31b)
    delta_mV = (voc_31b - voc_31) * 1000.0

    # Equivalence window: reabsorption and build-time scaling should
    # agree at V_oc to within a few kT/q ≈ 25 mV. A 5 mV band captures
    # the uniform-n·p limit cleanly while tolerating Radau tolerance
    # noise.
    assert abs(delta_mV) <= 5.0, (
        f"Phase 3.1b V_oc drift vs Phase 3.1: |Δ| = {abs(delta_mV):.2f} mV "
        f"(31b={voc_31b:.4f} V, 31={voc_31:.4f} V)"
    )


@pytest.mark.slow
def test_radiative_reabsorption_preserves_voc_boost(monkeypatch):
    """Phase 3.1b still delivers the literature V_oc boost (40–100 mV)
    versus a no-PR baseline. This guarantees the per-RHS source really
    is suppressing net radiative loss, not silently doing nothing.
    """
    stack = load_device_from_yaml(CONFIG)

    mode_off = replace(
        FULL,
        use_photon_recycling=False,
        use_radiative_reabsorption=False,
    )
    mode_31b = replace(
        FULL,
        use_photon_recycling=True,
        use_radiative_reabsorption=True,
    )

    def _voc_under_mode(active_mode):
        monkeypatch.setattr(
            _mode_mod, "resolve_mode", lambda _arg: active_mode,
        )
        monkeypatch.setattr(
            mol_mod, "resolve_mode", lambda _arg: active_mode,
        )
        res = run_suns_voc(
            stack, suns_levels=(1.0,), N_grid=40, t_settle=1e-3,
        )
        return float(res.V_oc[0])

    voc_off = _voc_under_mode(mode_off)
    voc_31b = _voc_under_mode(mode_31b)
    delta_mV = (voc_31b - voc_off) * 1000.0

    assert voc_31b > voc_off, (
        f"Phase 3.1b did not raise V_oc: on={voc_31b:.4f} V, off={voc_off:.4f} V"
    )
    assert 40.0 <= delta_mV <= 100.0, (
        f"Phase 3.1b V_oc boost {delta_mV:.1f} mV outside literature window "
        f"[40, 100]. voc_off={voc_off:.4f} V, voc_on={voc_31b:.4f} V"
    )

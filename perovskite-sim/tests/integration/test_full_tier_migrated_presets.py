"""FULL-tier consistency tests for the Stage 1/2 migrated presets.

Stage 1 (nip_MAPbI3_tmm, pin_MAPbI3_tmm) and Stage 2 (ionmonger_benchmark_tmm,
driftfusion_benchmark_tmm) added chi/Eg on every electrical layer and aligned
the manual V_bi field to compute_V_bi(). This module pins that alignment so
a future edit that silently drifts the two apart fails loudly instead of
breaking the diode at runtime.

Specifically checks, for each migrated preset:
  * compute_V_bi() and stack.V_bi agree within ~30 mV (the consistency gap
    that Stage 1a empirically identified as the threshold above which the
    Poisson BC disagrees with Fermi-level matching at V=0 and the diode
    fails to turn on).
  * FULL tier populates MaterialArrays.interface_faces — i.e. the TE cap
    path actually runs on these stacks (they were migrated *so that* it
    runs).
  * Legacy tier restores interface_faces == () on the same stacks.

The legacy fall-back rule (chi=Eg=0 in all electrical layers => compute_V_bi
returns stack.V_bi verbatim) is exercised separately on the un-migrated
nip_MAPbI3.yaml to make sure the back-compat path stayed intact through
the migration.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.solver.mol import build_material_arrays


def _electrical_grid(stack, n_per_layer: int = 60) -> np.ndarray:
    """Build a drift-diffusion grid that spans only the electrical layers.

    Uses the same tanh-clustered multilayer builder the real solver uses
    (``perovskite_sim.discretization.grid.multilayer_grid``) so that nodes
    land *on* every internal interface. mol.py's interface detection snaps
    to the nearest grid node via ``np.argmin(np.abs(x - offset))``; a naive
    ``np.linspace`` over the electrical thickness can round to the wrong
    side of an interface in nip-architecture stacks with a 200 nm first
    layer (argmin picks a node still inside layer 0, so delta_Ec = 0 and
    TE never activates). The multilayer builder places a node exactly at
    ``sum(thicknesses[:k])`` for every k, which is what the detector needs.

    Substrate layers (role='substrate') are filtered out by
    electrical_layers() before the grid is laid down.
    """
    elec = electrical_layers(stack)
    layers = [Layer(thickness=l.thickness, N=n_per_layer) for l in elec]
    return multilayer_grid(layers)


# Tolerance for the manual-vs-computed V_bi alignment. 30 mV is the empirical
# ceiling from Stage 1a — above that the Poisson BC disagreement with Fermi
# matching is large enough to kill the diode. See commit c93d854 for the
# diagnosis (and the TE CB sign fix that fell out of it).
V_BI_ALIGNMENT_TOL = 0.03  # volts

MIGRATED_PRESETS = [
    "configs/nip_MAPbI3_tmm.yaml",
    "configs/pin_MAPbI3_tmm.yaml",
    "configs/ionmonger_benchmark_tmm.yaml",
    "configs/driftfusion_benchmark_tmm.yaml",
]


@pytest.mark.parametrize("preset", MIGRATED_PRESETS)
def test_migrated_preset_vbi_alignment(preset):
    """Manual V_bi and compute_V_bi() must agree on every migrated preset."""
    stack = load_device_from_yaml(preset)
    manual = stack.V_bi
    computed = stack.compute_V_bi()
    delta = abs(manual - computed)
    assert delta < V_BI_ALIGNMENT_TOL, (
        f"{preset}: manual V_bi={manual:.4f} V disagrees with "
        f"compute_V_bi={computed:.4f} V by {delta*1000:.1f} mV "
        f"(tolerance {V_BI_ALIGNMENT_TOL*1000:.0f} mV). "
        "The Poisson BC will disagree with Fermi-level matching at V=0 "
        "under FULL tier and the diode will not turn on."
    )


@pytest.mark.parametrize("preset", MIGRATED_PRESETS)
def test_migrated_preset_full_mode_activates_te(preset):
    """FULL tier must populate interface_faces on every migrated preset."""
    stack = load_device_from_yaml(preset)
    stack_full = dataclasses.replace(stack, mode="full")
    x = _electrical_grid(stack_full)
    mat = build_material_arrays(x, stack_full)
    assert len(mat.interface_faces) >= 1, (
        f"{preset}: FULL tier produced interface_faces=() — TE cap path "
        "will never run. All four migrated presets have at least one band "
        "offset > 0.05 eV on each internal heterojunction (that's why they "
        "were migrated), so at minimum two interface faces should activate."
    )


@pytest.mark.parametrize("preset", MIGRATED_PRESETS)
def test_migrated_preset_legacy_mode_disables_te(preset):
    """Legacy tier must drop all TE interface faces on the same stacks."""
    stack = load_device_from_yaml(preset)
    stack_legacy = dataclasses.replace(stack, mode="legacy")
    x = _electrical_grid(stack_legacy)
    mat = build_material_arrays(x, stack_legacy)
    assert mat.interface_faces == (), (
        f"{preset}: legacy tier left interface_faces non-empty "
        f"({mat.interface_faces!r}) — TE cap should be disabled entirely."
    )


def test_legacy_preset_vbi_fallback_still_works():
    """nip_MAPbI3.yaml has chi=Eg=0 everywhere, so compute_V_bi must return
    the manual V_bi verbatim. This pins the fall-back rule that Stage 1/2
    migration relied on for backward compatibility.
    """
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    assert stack.compute_V_bi() == pytest.approx(stack.V_bi, abs=1e-12)

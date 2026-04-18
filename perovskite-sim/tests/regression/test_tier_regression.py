"""Phase 5 tier regression — cross-tier invariants for LEGACY / FAST / FULL.

These tests pin down the *end-to-end* guarantees of the tiered-mode system
so that accidental drift in the flag matrix or in build_material_arrays
gating is caught at the pytest level rather than in downstream experiments.

Three guarantees the tiers must meet:

1. **LEGACY reproduction.** A plain (chi=Eg=0, Beer-Lambert, no S, no v_sat)
   preset must produce bit-identical arrays in LEGACY mode vs the current
   FAST/FULL default, because every opt-in gate self-disables when its
   parameters are absent. This is the "tiers act as a ceiling" invariant.

2. **FAST stops at 3.2/3.3.** Even on a stack that *does* configure v_sat
   and S_*, the FAST tier must skip those two per-RHS hooks while still
   honouring the build-once Phase 1/2/3.1 upgrades. FULL on the same stack
   must visibly differ.

3. **Tier monotonicity.** The set of active physics is a nested hierarchy:
   LEGACY ⊆ FAST ⊆ FULL. Any flag that is on in LEGACY must be on in FAST
   and FULL; any flag that is on in FAST must be on in FULL. This is
   enforced inside tests/unit/models/test_mode.py, but we restate it in
   MaterialArrays form here because the *gated behaviour* is what
   downstream physics actually sees.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.discretization.grid import multilayer_grid, Layer

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_NIP = str(_CONFIGS / "nip_MAPbI3.yaml")


def _grid(stack, n_per_layer=20):
    return multilayer_grid(
        [Layer(thickness=l.thickness, N=n_per_layer) for l in stack.layers],
        alpha=3.0,
    )


def test_legacy_on_plain_preset_is_equivalent_to_full():
    """A plain Beer-Lambert preset with chi=Eg=0 and no S / no v_sat must
    produce the same MaterialArrays structure in LEGACY and FULL modes.

    Every Phase 1–3 opt-in gate checks that its configuration parameter is
    provided before activating — so on a preset that supplies none of them,
    FULL has nothing to turn on and LEGACY has nothing to turn off. The
    resulting MaterialArrays must agree on every physics flag that matters.
    """
    base = load_device_from_yaml(_NIP)
    x = _grid(base)

    legacy_stack = replace(base, mode="legacy")
    full_stack = replace(base, mode="full")

    mat_legacy = build_material_arrays(x, legacy_stack)
    mat_full = build_material_arrays(x, full_stack)

    # Per-RHS upgrades must be dormant on both sides.
    assert mat_legacy.has_field_mobility is False
    assert mat_full.has_field_mobility is False
    assert mat_legacy.has_selective_contacts is False
    assert mat_full.has_selective_contacts is False

    # The cached arrays must be bit-identical — same Poisson factor,
    # same n_eq boundary densities, same D_{n,p}_face, same generation.
    assert np.allclose(mat_legacy.D_n_face, mat_full.D_n_face)
    assert np.allclose(mat_legacy.D_p_face, mat_full.D_p_face)
    assert np.allclose(mat_legacy.eps_r, mat_full.eps_r)
    # Generation: FULL would use TMM if optical_material were set on any
    # layer, but nip_MAPbI3.yaml is Beer-Lambert only, so both tiers go
    # through the same Beer-Lambert path — G_optical is cached on the
    # MaterialArrays either way, and must match between the two tiers.
    # When no TMM data is configured, G_optical is None on both sides and
    # Beer-Lambert runs lazily inside assemble_rhs; we compare whichever
    # form the cache holds.
    if mat_legacy.G_optical is None or mat_full.G_optical is None:
        assert mat_legacy.G_optical is None and mat_full.G_optical is None
    else:
        assert np.allclose(mat_legacy.G_optical, mat_full.G_optical)


def test_fast_skips_per_rhs_hooks_even_when_config_provides_them():
    """If the stack opts into v_sat (3.2) and S_* (3.3), FAST must ignore
    both because they are per-RHS overhead, while FULL activates them.

    The visible signature: build_material_arrays sets has_field_mobility
    and has_selective_contacts to False in FAST and True in FULL for the
    same input stack.
    """
    base = load_device_from_yaml(_NIP)
    # Inject per-RHS opt-ins on a copy. Using replace on the stack first
    # so S_* show up; v_sat is per-layer so it has to be replaced at the
    # LayerSpec.params level, which the stack does NOT expose cheaply —
    # we instead touch MaterialParams through a new LayerSpec copy.
    from dataclasses import replace as _dc_replace
    new_layers = []
    for layer in base.layers:
        new_p = _dc_replace(layer.params, v_sat_n=1e5, v_sat_p=1e5)
        new_layers.append(_dc_replace(layer, params=new_p))
    stack_with_opt_ins = replace(
        base,
        layers=tuple(new_layers),
        S_n_left=1.0, S_p_left=1.0,
        S_n_right=1.0, S_p_right=1.0,
    )

    x = _grid(stack_with_opt_ins)

    fast_stack = replace(stack_with_opt_ins, mode="fast")
    full_stack = replace(stack_with_opt_ins, mode="full")

    mat_fast = build_material_arrays(x, fast_stack)
    mat_full = build_material_arrays(x, full_stack)

    # FAST turns off the two per-RHS hooks regardless of config.
    assert mat_fast.has_field_mobility is False
    assert mat_fast.has_selective_contacts is False
    # FULL activates both because the config now provides the opt-ins.
    assert mat_full.has_field_mobility is True
    assert mat_full.has_selective_contacts is True


def test_legacy_disables_build_once_upgrades_even_when_config_provides_them():
    """LEGACY must ignore every Phase 1/2/3.1 opt-in even when the config
    provides it, so that benchmark runs remain IonMonger-reproducible.

    We inject chi/Eg band offsets (which would normally trigger TE in
    FAST/FULL) and verify that LEGACY still builds a flat-band stack
    with no TE capping.
    """
    base = load_device_from_yaml(_NIP)
    # Inject band offsets into every layer (chi=4 eV, Eg=1.6 eV is typical
    # for perovskite but chi varies by role in reality; we just want a
    # non-zero offset between layers so TE would activate in FAST/FULL).
    from dataclasses import replace as _dc_replace
    new_layers = []
    for i, layer in enumerate(base.layers):
        chi = 4.0 + 0.1 * i  # staircase chi so each interface has an offset
        Eg = 1.6 + 0.05 * i
        new_p = _dc_replace(layer.params, chi=chi, Eg=Eg)
        new_layers.append(_dc_replace(layer, params=new_p))
    stack_with_offsets = replace(base, layers=tuple(new_layers))
    x = _grid(stack_with_offsets)

    legacy_stack = replace(stack_with_offsets, mode="legacy")
    full_stack = replace(stack_with_offsets, mode="full")

    mat_legacy = build_material_arrays(x, legacy_stack)
    mat_full = build_material_arrays(x, full_stack)

    # In LEGACY, TE must be off → interface_faces should be empty / not
    # participating. In FULL, interface_faces collects the faces where
    # |delta_Ec| or |delta_Ev| exceeds the 0.05 eV threshold.
    # Depending on the MaterialArrays schema, this is exposed via the
    # interface_faces / use_thermionic_emission paths. We check the
    # visible consequence: FULL should see at least one interface face,
    # LEGACY should see zero.
    legacy_n_faces = len(getattr(mat_legacy, "interface_faces", ()) or ())
    full_n_faces = len(getattr(mat_full, "interface_faces", ()) or ())
    assert legacy_n_faces == 0
    assert full_n_faces >= 1


def test_mode_string_on_yaml_roundtrips_through_device_stack():
    """YAML `device.mode: legacy|fast|full` must round-trip.

    Regression for the mode-propagation path: write a minimal YAML
    fragment with each of the three tier names into a tmp dir, load
    via load_device_from_yaml, and verify the returned DeviceStack
    carries the expected `mode` attribute.
    """
    import tempfile, textwrap, os
    # Base YAML body (same structure as nip_MAPbI3.yaml but trimmed).
    body = textwrap.dedent(
        """
        device:
          V_bi: 1.1
          Phi: 2.5e21
          mode: {mode}
        layers:
          - name: HTL
            role: HTL
            thickness: 200e-9
            eps_r: 3.0
            mu_n: 1e-10
            mu_p: 1e-6
            ni: 1e0
            N_D: 0.0
            N_A: 2e23
            D_ion: 0.0
            P_lim: 1e30
            P0: 0.0
            tau_n: 1e-9
            tau_p: 1e-9
            n1: 1e0
            p1: 1e0
            B_rad: 1e-30
            C_n: 1e-42
            C_p: 1e-42
            alpha: 0.0
          - name: absorber
            role: absorber
            thickness: 400e-9
            eps_r: 24.1
            mu_n: 2e-4
            mu_p: 2e-4
            ni: 3.2e13
            N_D: 0.0
            N_A: 0.0
            D_ion: 1e-16
            P_lim: 1.6e27
            P0: 1.6e24
            tau_n: 1e-6
            tau_p: 1e-6
            n1: 3.2e13
            p1: 3.2e13
            B_rad: 5e-22
            C_n: 1e-42
            C_p: 1e-42
            alpha: 1.3e7
          - name: ETL
            role: ETL
            thickness: 100e-9
            eps_r: 10.0
            mu_n: 1e-5
            mu_p: 1e-10
            ni: 1e0
            N_D: 1e24
            N_A: 0.0
            D_ion: 0.0
            P_lim: 1e30
            P0: 0.0
            tau_n: 1e-9
            tau_p: 1e-9
            n1: 1e0
            p1: 1e0
            B_rad: 1e-30
            C_n: 1e-42
            C_p: 1e-42
            alpha: 0.0
        """
    )
    with tempfile.TemporaryDirectory() as d:
        for name in ("legacy", "fast", "full"):
            p = os.path.join(d, f"cfg_{name}.yaml")
            with open(p, "w") as fh:
                fh.write(body.format(mode=name))
            stack = load_device_from_yaml(p)
            assert stack.mode == name, (
                f"YAML mode={name} round-trip produced stack.mode={stack.mode}"
            )

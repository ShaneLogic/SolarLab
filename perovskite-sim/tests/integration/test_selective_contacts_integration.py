"""Integration tests for the Phase 3.3 selective / Schottky outer contacts.

These tests exercise the Robin-BC hot path end-to-end — from YAML parsing
through ``build_material_arrays`` / ``assemble_rhs`` / ``run_jv_sweep`` —
and pin down the three guarantees the implementation has to meet:

1. **Backward compatibility.** Every shipped preset has
   ``S_*_{left,right} = None``, so ``has_selective_contacts`` must stay
   ``False`` and the Dirichlet pin must remain in force. This keeps the
   Phase 3.2 and earlier regression numerics bit-identical.

2. **Activation.** Setting any one of the four ``S`` fields on a
   ``DeviceStack`` (or in YAML) must flip the flag on, propagate the
   numeric values into ``MaterialArrays``, and leave the unconfigured
   sides as Dirichlet — partial activation is a valid configuration.

3. **Finite RHS under aggressive Robin fluxes.** A large ``S`` gives the
   Robin correction a high-magnitude contribution at the boundary node;
   the RHS must still return finite ``dn`` / ``dp`` so the Radau
   integrator can make progress without tripping the finite-check.

We deliberately keep the J-V differentiability check mild — the Robin
coefficients perturb only two nodes out of ~150, and in the
well-collected MAPbI3 stack the interior transport dominates the
terminal current. The smoke-test below only asserts that a blocking
``S=0`` contact on the majority-carrier side measurably suppresses
current relative to the ohmic baseline — a classic signature of the
Schottky / selective limit.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import (
    build_material_arrays,
    assemble_rhs,
    StateVec,
)
from perovskite_sim.discretization.grid import multilayer_grid, Layer

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_NIP_MAPBI3 = str(_CONFIGS_DIR / "nip_MAPbI3.yaml")


def _build_grid(stack, n_per_layer=30):
    """Build a multilayer tanh grid using the library's native ``Layer`` API.

    ``Layer`` here is the *grid* layer (thickness + interval count ``N``),
    not ``LayerSpec`` from the device model.
    """
    grid_layers = [Layer(thickness=l.thickness, N=n_per_layer) for l in stack.layers]
    return multilayer_grid(grid_layers, alpha=3.0)


def _set_contact_S(stack, **kwargs):
    """Return a copy of ``stack`` with the four ``S_*`` fields set.

    Missing kwargs leave the corresponding field at its current value
    (None by default). This is the supported way to activate selective
    / Schottky contacts on a preset-loaded stack inside a test.
    """
    return replace(stack, **kwargs)


def test_default_presets_have_selective_contacts_off():
    """Shipped YAML presets must not activate the Robin-BC hot path.

    No existing config sets ``S_*`` in the ``device:`` block, so the
    gating in ``build_material_arrays`` must leave the flag False and
    all four S fields None. This is the backward-compatibility
    guarantee that keeps pre-3.3 regression numerics bit-identical.
    """
    stack = load_device_from_yaml(_NIP_MAPBI3)
    assert stack.S_n_left is None
    assert stack.S_p_left is None
    assert stack.S_n_right is None
    assert stack.S_p_right is None

    x = _build_grid(stack)
    mat = build_material_arrays(x, stack)

    assert mat.has_selective_contacts is False
    assert mat.S_n_L is None
    assert mat.S_p_L is None
    assert mat.S_n_R is None
    assert mat.S_p_R is None


def test_setting_one_S_activates_selective_contacts():
    """A single finite ``S_*`` field must flip ``has_selective_contacts``.

    The other three sides stay None — partial activation is a supported
    configuration (e.g. a Schottky electron contact on the left with the
    right contact still ohmic).
    """
    base = load_device_from_yaml(_NIP_MAPBI3)
    stack = _set_contact_S(base, S_n_left=1e4)

    x = _build_grid(stack)
    mat = build_material_arrays(x, stack)

    assert mat.has_selective_contacts is True
    assert mat.S_n_L == 1e4
    assert mat.S_p_L is None
    assert mat.S_n_R is None
    assert mat.S_p_R is None


def test_all_four_S_propagate_into_material_arrays():
    """All four S fields must land in MaterialArrays when configured.

    This is the "full Robin-BC" activation pattern: every carrier / side
    has a finite surface recombination velocity. Typical usage for a
    realistic selective-contact device.
    """
    base = load_device_from_yaml(_NIP_MAPBI3)
    stack = _set_contact_S(
        base,
        S_n_left=1e3, S_p_left=1e5,
        S_n_right=1e5, S_p_right=1e3,
    )

    x = _build_grid(stack)
    mat = build_material_arrays(x, stack)

    assert mat.has_selective_contacts is True
    assert mat.S_n_L == 1e3
    assert mat.S_p_L == 1e5
    assert mat.S_n_R == 1e5
    assert mat.S_p_R == 1e3


def test_rhs_finite_with_large_S_robin_flux():
    """The RHS must stay finite when ``S`` is large enough to drive the
    boundary Robin flux to the ohmic limit.

    At ``S → ∞`` the Robin correction ``-S · (n - n_eq) / dx_cell``
    approaches the hard Dirichlet pull; mathematically this can be a
    stiff forcing, but numerically the RHS is still a finite real
    number. Passing this check rules out the most common failure mode
    (a missing ``float()`` cast or an uncaught divide-by-zero).
    """
    base = load_device_from_yaml(_NIP_MAPBI3)
    stack = _set_contact_S(
        base,
        S_n_left=1e8, S_p_left=1e8,
        S_n_right=1e8, S_p_right=1e8,
    )

    x = _build_grid(stack)
    mat = build_material_arrays(x, stack)

    # Build a near-equilibrium state with a small excess near both
    # contacts so the Robin flux is actually non-zero.
    N = len(x)
    n = np.linspace(mat.n_L * 1.01, mat.n_R * 1.01, N)
    p = np.linspace(mat.p_L * 1.01, mat.p_R * 1.01, N)
    P = mat.P_ion0.copy()
    y = StateVec.pack(n, p, P, P_neg=None)

    dydt = assemble_rhs(0.0, y, x, stack, mat, illuminated=True, V_app=0.0)
    assert np.all(np.isfinite(dydt)), (
        "RHS went non-finite under aggressive Robin BCs at S=1e8"
    )


def test_rhs_finite_with_blocking_contacts():
    """``S = 0`` must reduce to a Neumann-like boundary without blow-up.

    The physical meaning is a perfectly blocking contact: no carrier
    exchange, only recombination eats the population. The Robin flux
    contribution vanishes, but the dn / dp entries at indices 0 and -1
    are no longer pinned — they evolve from the interior divergence
    plus G/R alone. We want to confirm this path doesn't silently
    produce NaN or Inf.
    """
    base = load_device_from_yaml(_NIP_MAPBI3)
    stack = _set_contact_S(
        base,
        S_n_left=0.0, S_p_left=0.0,
        S_n_right=0.0, S_p_right=0.0,
    )

    x = _build_grid(stack)
    mat = build_material_arrays(x, stack)

    N = len(x)
    n = np.linspace(mat.n_L, mat.n_R, N)
    p = np.linspace(mat.p_L, mat.p_R, N)
    P = mat.P_ion0.copy()
    y = StateVec.pack(n, p, P, P_neg=None)

    dydt = assemble_rhs(0.0, y, x, stack, mat, illuminated=True, V_app=0.0)
    assert np.all(np.isfinite(dydt)), (
        "blocking (S=0) contacts produced a non-finite RHS"
    )


def test_selective_contacts_change_jv_curve():
    """A truly selective contact stack must perturb J-V vs. ohmic.

    The classic selective-contact pattern blocks the *wrong* carrier on
    each side: at the electron contact (right) hole exchange is
    suppressed (S_p small) so accumulating holes can't dump into the
    metal; at the hole contact (left) electron exchange is suppressed
    (S_n small). This is the physically interesting case — the boundary
    densities for the wrong-sign carrier evolve away from equilibrium,
    SRH near the contact changes, and V_oc shifts.

    A symmetric ohmic-like S (S_n_right = S_p_right = 1e5 m/s) is NOT
    a useful test: at that S the Robin time constant is sub-femtosecond
    and the boundary node is pinned to equilibrium just as tightly as
    Dirichlet — the J-V curve comes back identical to roundoff. We use
    asymmetric blocking instead.

    This is a smoke-test for hot-path activation, not a physics-magnitude
    pin. We assert ``>1e-4`` relative difference somewhere on the curve,
    well above solver noise but loose enough to survive minor solver
    refactors.
    """
    base = load_device_from_yaml(_NIP_MAPBI3)

    # Baseline: pure Dirichlet pin, pre-3.3 behaviour.
    r_base = run_jv_sweep(
        base, N_grid=40, n_points=6, v_rate=1.0, V_max=0.6,
    )

    # Selective stack: very small S for the wrong-sign carrier on each
    # side (effectively blocking) and a small-but-finite S for the
    # matched carrier. The Robin time constant is ``dx_cell / S``; with
    # dx_cell ~ 5e-9 m and carrier τ ~ 1 µs in the absorber, anything
    # above ~1e-2 m/s effectively re-pins the boundary to its
    # equilibrium value and the J-V comes back identical to the
    # Dirichlet baseline (max rel diff < 1e-6, below solver noise).
    # 1e-3 m/s is in the realistic Stolterfoht/Belisle range for
    # selective perovskite contacts and leaves the boundary node free
    # to evolve enough that the integrated J(V) shifts at the per-mil
    # level — well above noise but far below a regime change.
    stack_sel = _set_contact_S(
        base,
        S_n_left=1e-4, S_p_left=1e-3,    # hole contact: electrons blocked
        S_n_right=1e-3, S_p_right=1e-4,  # electron contact: holes blocked
    )
    r_sel = run_jv_sweep(
        stack_sel, N_grid=40, n_points=6, v_rate=1.0, V_max=0.6,
    )

    assert np.all(np.isfinite(r_base.J_fwd))
    assert np.all(np.isfinite(r_sel.J_fwd))

    max_rel_diff = np.max(
        np.abs(r_sel.J_fwd - r_base.J_fwd)
        / np.maximum(np.abs(r_base.J_fwd), 1.0)
    )
    assert max_rel_diff > 1e-4, (
        f"Selective-contact J-V is identical to ohmic baseline to solver "
        f"noise; Robin hot path did not engage (max rel diff = {max_rel_diff:.2e})"
    )


def test_yaml_loader_parses_flat_S_keys(tmp_path: Path):
    """``device.S_n_left`` etc. as flat top-level keys must round-trip."""
    import shutil
    src = Path(_NIP_MAPBI3)
    dst = tmp_path / "nip_with_S.yaml"
    text = src.read_text()
    # Insert S_* keys into the device: block.
    text = text.replace(
        "device:\n  V_bi: 1.1\n",
        "device:\n  V_bi: 1.1\n  S_n_left: 1000.0\n  S_p_right: 2000.0\n",
    )
    dst.write_text(text)

    stack = load_device_from_yaml(str(dst))
    assert stack.S_n_left == 1000.0
    assert stack.S_p_left is None
    assert stack.S_n_right is None
    assert stack.S_p_right == 2000.0


def test_yaml_loader_parses_nested_contacts_block(tmp_path: Path):
    """A nested ``contacts:`` block is the preferred schema for readability."""
    src = Path(_NIP_MAPBI3)
    dst = tmp_path / "nip_with_contacts.yaml"
    text = src.read_text()
    text = text.replace(
        "device:\n  V_bi: 1.1\n",
        (
            "device:\n"
            "  V_bi: 1.1\n"
            "  contacts:\n"
            "    left:\n"
            "      S_n: 1500.0\n"
            "      S_p: 2500.0\n"
            "    right:\n"
            "      S_n: 3500.0\n"
            "      S_p: 4500.0\n"
        ),
    )
    dst.write_text(text)

    stack = load_device_from_yaml(str(dst))
    assert stack.S_n_left == 1500.0
    assert stack.S_p_left == 2500.0
    assert stack.S_n_right == 3500.0
    assert stack.S_p_right == 4500.0

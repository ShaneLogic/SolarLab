"""Analytical sensitivity tests for DeviceStack.compute_V_bi.

These tests guard against the original Phase-1 failure mode: the legacy
V_bi was a constant config knob, so V_oc was mechanically insensitive to
n_i, doping, and band-edge energies even when the underlying material
changed. compute_V_bi derives V_bi from the Fermi-level difference across
the heterostack, so physically-meaningful inputs must propagate.

All tests here mutate a single contact parameter via dataclasses.replace
and assert (a) the delta direction matches the analytical expectation and
(b) its magnitude is close to the closed-form ideal-contact value. They
do NOT run a J–V sweep — that's covered by the slow benchmark.
"""
import math
from dataclasses import replace

import pytest

from perovskite_sim.constants import V_T
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams


def _make_reference_stack() -> DeviceStack:
    """Minimal n-i-p heterostack matching the ionmonger benchmark contacts.

    Kept standalone from tests/unit/models/test_device_vbi.py so the two
    files stay independently runnable; they share physics, not fixtures.
    """
    htl = LayerSpec(
        name="spiro_HTL", thickness=200e-9, role="HTL",
        params=MaterialParams(
            eps_r=3.0, mu_n=1e-10, mu_p=3.89e-5,
            D_ion=0.0, P_lim=1e30, P0=0.0,
            ni=1e0, tau_n=1e-9, tau_p=1e-9,
            n1=1e0, p1=1e0, B_rad=0.0, C_n=0.0, C_p=0.0,
            alpha=0.0, N_A=1e24, N_D=0.0, chi=2.1, Eg=3.0,
        ),
    )
    absorber = LayerSpec(
        name="MAPbI3", thickness=400e-9, role="absorber",
        params=MaterialParams(
            eps_r=24.1, mu_n=6.62e-3, mu_p=6.62e-3,
            D_ion=1.01e-17, P_lim=1.6e27, P0=1.6e25,
            ni=2.89e10, tau_n=3e-9, tau_p=3e-7,
            n1=2.89e10, p1=2.89e10, B_rad=0.0, C_n=0.0, C_p=0.0,
            alpha=1.3e7, N_A=0.0, N_D=0.0, chi=3.7, Eg=1.6,
        ),
    )
    etl = LayerSpec(
        name="TiO2_ETL", thickness=100e-9, role="ETL",
        params=MaterialParams(
            eps_r=10.0, mu_n=3.89e-4, mu_p=1e-10,
            D_ion=0.0, P_lim=1e30, P0=0.0,
            ni=1e0, tau_n=1e-9, tau_p=1e-9,
            n1=1e0, p1=1e0, B_rad=0.0, C_n=0.0, C_p=0.0,
            alpha=0.0, N_A=0.0, N_D=1e24, chi=4.0, Eg=3.2,
        ),
    )
    return DeviceStack(
        layers=(htl, absorber, etl),
        V_bi=1.1, Phi=1.4e21,
        interfaces=((3e5, 0.1), (0.1, 3e5)),
    )


def _replace_contact(
    stack: DeviceStack, idx: int, **param_updates: float
) -> DeviceStack:
    """Return a new stack with one contact layer's MaterialParams updated."""
    old_layer = stack.layers[idx]
    assert old_layer.params is not None
    new_params = replace(old_layer.params, **param_updates)
    new_layer = replace(old_layer, params=new_params)
    new_layers = tuple(
        new_layer if i == idx else l for i, l in enumerate(stack.layers)
    )
    return replace(stack, layers=new_layers)


def test_vbi_scales_with_etl_doping():
    """Doubling N_D in the ETL raises V_bi by ~V_T·ln(2) (≈17.9 mV).

    For an n-type contact with N_D >> ni: n ≈ N_D, so
        E_F = E_i - V_T·ln(N_D / ni).
    Doubling N_D shifts E_F downward by V_T·ln(2), which raises the
    left-right Fermi level difference (= V_bi) by the same amount.
    """
    stack = _make_reference_stack()
    v_bi_0 = stack.compute_V_bi()

    stack_2x = _replace_contact(stack, idx=2, N_D=2e24)
    v_bi_1 = stack_2x.compute_V_bi()

    expected_delta = V_T * math.log(2.0)
    actual_delta = v_bi_1 - v_bi_0
    assert actual_delta == pytest.approx(expected_delta, abs=1e-4), (
        f"ΔV_bi = {actual_delta * 1e3:.3f} mV, "
        f"expected {expected_delta * 1e3:.3f} mV (V_T·ln 2)"
    )


def test_vbi_scales_with_htl_doping():
    """Doubling N_A in the HTL raises V_bi by ~V_T·ln(2) (≈17.9 mV).

    Symmetric to the ETL test: the p-type Fermi level moves toward E_v
    by V_T·ln(2), widening the contact Fermi split.
    """
    stack = _make_reference_stack()
    v_bi_0 = stack.compute_V_bi()

    stack_2x = _replace_contact(stack, idx=0, N_A=2e24)
    v_bi_1 = stack_2x.compute_V_bi()

    expected_delta = V_T * math.log(2.0)
    actual_delta = v_bi_1 - v_bi_0
    assert actual_delta == pytest.approx(expected_delta, abs=1e-4), (
        f"ΔV_bi = {actual_delta * 1e3:.3f} mV, "
        f"expected {expected_delta * 1e3:.3f} mV (V_T·ln 2)"
    )


def test_vbi_scales_with_contact_ni():
    """Halving ni in both contacts raises V_bi by ~2·V_T·ln(2) (≈35.8 mV).

    For degenerate contacts (n ≈ N_D, p ≈ N_A), ln(n/ni) and ln(p/ni)
    each grow by ln(2) when ni halves. Both contacts contribute, so the
    V_bi delta is twice the single-contact shift.

    This is the canary that originally failed before Phase 1: the legacy
    constant V_bi was numerically blind to ni.
    """
    stack = _make_reference_stack()
    v_bi_0 = stack.compute_V_bi()

    stack_half = _replace_contact(stack, idx=0, ni=0.5)
    stack_half = _replace_contact(stack_half, idx=2, ni=0.5)
    v_bi_1 = stack_half.compute_V_bi()

    expected_delta = 2.0 * V_T * math.log(2.0)
    actual_delta = v_bi_1 - v_bi_0
    assert actual_delta == pytest.approx(expected_delta, abs=2e-4), (
        f"ΔV_bi = {actual_delta * 1e3:.3f} mV, "
        f"expected {expected_delta * 1e3:.3f} mV (2·V_T·ln 2)"
    )


def test_vbi_tracks_etl_electron_affinity():
    """Raising ETL chi by 0.1 eV shifts V_bi by −0.1 V.

    Convention (per _fermi_level): E_F is a positive number measured
    *below* the vacuum level, so a deeper Fermi level is numerically
    larger. Raising chi by 0.1 eV drops the ETL intrinsic level deeper
    by the same amount, pushing E_F,right up by 0.1 (numerically). The
    V_bi = E_F,left − E_F,right drop therefore decreases by 0.1 V.
    """
    stack = _make_reference_stack()
    v_bi_0 = stack.compute_V_bi()

    stack_chi = _replace_contact(stack, idx=2, chi=4.1)
    v_bi_1 = stack_chi.compute_V_bi()

    actual_delta = v_bi_1 - v_bi_0
    assert actual_delta == pytest.approx(-0.1, abs=1e-6), (
        f"ΔV_bi = {actual_delta * 1e3:.3f} mV, expected −100.000 mV"
    )


def test_vbi_monotonic_under_doping_sweep():
    """V_bi must be strictly monotonic in ETL N_D across a decade sweep.

    Non-monotonicity would indicate the two-branch majority-carrier
    formula picked the wrong branch somewhere, which would silently bias
    every downstream V_oc prediction. This is a structural check, not a
    magnitude check.
    """
    stack = _make_reference_stack()
    doping_values = [1e22, 1e23, 1e24, 1e25, 1e26]
    v_bi_values = [
        _replace_contact(stack, idx=2, N_D=N_D).compute_V_bi()
        for N_D in doping_values
    ]
    diffs = [b - a for a, b in zip(v_bi_values, v_bi_values[1:])]
    assert all(d > 0 for d in diffs), (
        f"V_bi non-monotonic in N_D sweep: {v_bi_values}"
    )

"""Stage-A 2D validation gate — six-check parity test.

Runs the 1D `run_jv_sweep` and the 2D `run_jv_sweep_2d` on the same
TMM-enabled MAPbI3 stack with no microstructural features (lateral-uniform
device). Asserts the 2D solver matches the 1D reference within tight
tolerances on V_oc, J_sc, and FF, and that the 2D state remains laterally
uniform (the lateral-uniformity invariant of an extruded 1D problem).

Spec section 7 of `docs/superpowers/specs/2026-04-27-2d-microstructural-extension-design.md`.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep, compute_metrics
from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack


PRESET = "configs/twod/nip_MAPbI3_uniform.yaml"


def _freeze_ions(stack: DeviceStack) -> DeviceStack:
    """Return a copy of the stack with D_ion = 0 in every layer.

    Stage A 2D holds ions as a static background, so 1D-vs-2D parity is only
    meaningful when 1D's ions are also frozen. Otherwise the 1D sweep lets
    P(y) drift with V_app while the 2D Poisson rho stays pinned at the V=0
    profile, producing systematically different V_oc / FF.
    """
    return replace(stack, layers=tuple(
        replace(layer, params=replace(layer.params, D_ion=0.0))
        for layer in stack.layers
    ))


def _maybe_flip_sign(V: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Return J with the same sign convention as the 1D ``compute_metrics``
    expects (J > 0 at V=0 under illumination, J = 0 at V_oc, J < 0 beyond).

    Detection: if the slope of J vs V near V=0 is positive (i.e. J starts
    negative and grows), flip the sign.
    """
    if len(V) < 2:
        return J
    return -J if J[0] < 0 else J


@pytest.mark.regression
@pytest.mark.slow
def test_twod_uniform_matches_1d_within_tolerance():
    """Stage-A validation gate. Six checks per spec §7."""
    stack = _freeze_ions(load_device_from_yaml(PRESET))

    # 1D reference run. Forward+reverse is mandatory in the 1D API; we
    # use only the forward leg. N_grid matches 2D's effective y-node count
    # (Ny_per_layer=10 × 3 electrical layers + 1 boundary = 31) so SG
    # discretisation error is identical between 1D and 2D.
    r1 = run_jv_sweep(stack, N_grid=31, V_max=1.2, n_points=13, illuminated=True)
    V1 = np.asarray(r1.V_fwd)
    J1 = _maybe_flip_sign(V1, np.asarray(r1.J_fwd))
    m1 = compute_metrics(V1, J1)

    # 2D run on same stack, lateral-uniform device. Coarse mesh (Nx=4,
    # Ny_per_layer=10) keeps the Radau LU on ~250x250 system tractable
    # while still resolving each layer; lateral parity does not improve
    # with finer Nx for an extruded 1D state.
    r2 = run_jv_sweep_2d(
        stack=stack,
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        # 1ms ≈ 10⁴ × carrier lifetime — long enough for full quasi-steady
        # state at each voltage step. The default 1e-7s is fine for an
        # ion-driven sweep where ionic memory dominates, but for the Stage A
        # frozen-ion validation we need full carrier relaxation per step
        # to match the slow 1D sweep.
        settle_t=1e-3,
    )
    V2 = np.asarray(r2.V)
    J2 = _maybe_flip_sign(V2, np.asarray(r2.J))
    m2 = compute_metrics(V2, J2)

    print(f"\n1D: V_oc={m1.V_oc*1e3:.3f} mV, J_sc={m1.J_sc:.3f} A/m², FF={m1.FF:.4f}")
    print(f"2D: V_oc={m2.V_oc*1e3:.3f} mV, J_sc={m2.J_sc:.3f} A/m², FF={m2.FF:.4f}")

    # Check 1: V_oc agreement within 0.1 mV (1e-4 V)
    assert abs(m2.V_oc - m1.V_oc) <= 1e-4, (
        f"V_oc(2D)={m2.V_oc:.6f} V vs V_oc(1D)={m1.V_oc:.6f} V "
        f"(diff {(m2.V_oc - m1.V_oc) * 1e3:.3f} mV)"
    )

    # Check 2: J_sc agreement within 0.05 %
    rel_jsc = abs(m2.J_sc - m1.J_sc) / abs(m1.J_sc)
    assert rel_jsc <= 5e-4, (
        f"J_sc relative diff {rel_jsc:.2e} > 5e-4 "
        f"(2D={m2.J_sc:.4f}, 1D={m1.J_sc:.4f} A/m²)"
    )

    # Check 3: FF agreement within 0.001
    assert abs(m2.FF - m1.FF) <= 1e-3, (
        f"FF(2D)={m2.FF:.6f} vs FF(1D)={m1.FF:.6f}"
    )

    # Check 4: lateral invariance of n at V_oc
    i_voc = int(np.argmin(np.abs(V2 - m2.V_oc)))
    snap = r2.snapshots[i_voc]
    n_col0 = snap.n[:, [0]]
    n_lat_var = np.max(np.abs(snap.n - n_col0)) / np.max(np.abs(n_col0))
    assert n_lat_var <= 1e-9, (
        f"n lateral variation {n_lat_var:.2e} exceeds 1e-9"
    )

    # Check 5: divergence of total electron+hole current is small at interior
    # nodes (steady-state continuity in the bulk away from generation hot
    # spots). Tolerance scaled for J_sc magnitude.
    Jx = snap.Jx_n + snap.Jx_p   # (Ny, Nx-1)
    Jy = snap.Jy_n + snap.Jy_p   # (Ny-1, Nx)
    dx = np.diff(snap.x)
    dy = np.diff(snap.y)
    # Build div_J on interior (j=1..Ny-2, i=1..Nx-2)
    Ny, Nx = snap.n.shape
    div_J = np.zeros((Ny - 2, Nx - 2))
    for j in range(div_J.shape[0]):
        for i in range(div_J.shape[1]):
            jj = j + 1
            ii = i + 1
            div_J[j, i] = (
                (Jx[jj, ii] - Jx[jj, ii - 1]) / dx[ii]
                + (Jy[jj, ii] - Jy[jj - 1, ii]) / dy[jj]
            )
    # Compare to recombination scale: at V_oc, R · q · L ~ J_sc, so per-cell
    # |∇·J| can reach ~|J_sc|/L. Use a generous tolerance: 100 × |J_sc| /
    # min(dy) (the recombination/generation balance dominates the bulk).
    tol_div = 100.0 * abs(m2.J_sc) / float(np.min(dy))
    assert np.max(np.abs(div_J)) <= tol_div, (
        f"max|∇·J|={np.max(np.abs(div_J)):.3e} exceeds tolerance {tol_div:.3e}"
    )

    # Check 6: Poisson BCs match — bottom contact φ=0, top contact
    # φ = V_bi − V_app (where V_app is the snapshot's actual voltage,
    # which is the V grid point closest to V_oc, not V_oc itself).
    assert np.allclose(snap.phi[0, :], 0.0, atol=1e-10), (
        "Bottom-contact phi != 0 (Dirichlet BC violated)"
    )
    V_app_snap = float(V2[i_voc])
    expected_top = stack.V_bi - V_app_snap
    assert np.allclose(snap.phi[-1, :], expected_top, atol=1e-6), (
        f"Top-contact phi[{snap.phi[-1,0]:.4f}] != V_bi - V_app["
        f"{expected_top:.4f}] at V_app={V_app_snap:.3f} V"
    )


ROBIN_PRESET = "configs/selective_contacts_demo.yaml"


@pytest.mark.regression
@pytest.mark.slow
def test_twod_robin_parity_vs_1d():
    """Stage B(c.1) primary gate: laterally-uniform 2D with Robin contacts matches
    1D Phase 3.3 within sub-mV V_oc / 5×10⁻⁴ J_sc / 10⁻³ FF.

    Uses configs/selective_contacts_demo.yaml (Beer-Lambert nip MAPbI3 with all
    four S values set). Ions frozen on both sides for a clean comparison.
    """
    stack = _freeze_ions(load_device_from_yaml(ROBIN_PRESET))

    # 1D reference with Phase 3.3 Robin contacts active (has_selective_contacts=True)
    r1 = run_jv_sweep(stack, N_grid=31, V_max=1.2, n_points=13, illuminated=True)
    V1 = np.asarray(r1.V_fwd)
    J1 = _maybe_flip_sign(V1, np.asarray(r1.J_fwd))
    m1 = compute_metrics(V1, J1)

    # 2D with Stage B(c.1) Robin contacts, same grid resolution as Stage-A gate
    r2 = run_jv_sweep_2d(
        stack=stack,
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    V2 = np.asarray(r2.V)
    J2 = _maybe_flip_sign(V2, np.asarray(r2.J))
    m2 = compute_metrics(V2, J2)

    print(
        f"\nRobin 1D: V_oc={m1.V_oc*1e3:.3f} mV  J_sc={m1.J_sc:.3f} A/m²  FF={m1.FF:.4f}"
        f"\nRobin 2D: V_oc={m2.V_oc*1e3:.3f} mV  J_sc={m2.J_sc:.3f} A/m²  FF={m2.FF:.4f}"
    )

    assert abs(m2.V_oc - m1.V_oc) <= 1e-3, (
        f"Robin V_oc(2D)={m2.V_oc:.6f} V vs V_oc(1D)={m1.V_oc:.6f} V "
        f"(diff {(m2.V_oc - m1.V_oc)*1e3:.3f} mV, limit 1 mV)"
    )
    rel_jsc = abs(m2.J_sc - m1.J_sc) / abs(m1.J_sc)
    assert rel_jsc <= 5e-4, (
        f"Robin J_sc rel diff {rel_jsc:.2e} > 5e-4 "
        f"(2D={m2.J_sc:.4f}, 1D={m1.J_sc:.4f} A/m²)"
    )
    assert abs(m2.FF - m1.FF) <= 1e-3, (
        f"Robin FF(2D)={m2.FF:.6f} vs FF(1D)={m1.FF:.6f} "
        f"(diff {abs(m2.FF - m1.FF):.4f}, limit 1e-3)"
    )


@pytest.mark.regression
@pytest.mark.slow
def test_twod_robin_bounded_shift_vs_dirichlet():
    """Robin contacts with aggressive blocking S must shift V_oc by 1–150 mV
    vs Dirichlet baseline.

    Compares: (a) nip_MAPbI3_uniform.yaml (Dirichlet) vs
              (b) the same stack with strongly-blocking S values applied
                  (S_n_left=1e-4, S_p_left=1e-3, S_n_right=1e-3, S_p_right=1e-4 —
                  same envelope as 1D test_selective_contacts_integration).
    The shipped selective_contacts_demo.yaml uses moderate matched-carrier
    S=1e3 which is effectively ohmic at this device thickness and produces no
    measurable shift; aggressive blocking is needed to drive the contact
    physics into the regime where V_oc moves.

    This is a secondary sanity test — the parity gate
    (test_twod_robin_parity_vs_1d) is the primary correctness criterion.
    """
    stack_dirichlet = _freeze_ions(load_device_from_yaml(PRESET))        # no S values
    stack_robin     = replace(stack_dirichlet,
        S_n_left=1e-4, S_p_left=1e-3,
        S_n_right=1e-3, S_p_right=1e-4,
    )

    common_kw = dict(
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    r_d = run_jv_sweep_2d(stack=stack_dirichlet, **common_kw)
    r_r = run_jv_sweep_2d(stack=stack_robin,     **common_kw)

    V_d = np.asarray(r_d.V); J_d = _maybe_flip_sign(V_d, np.asarray(r_d.J))
    V_r = np.asarray(r_r.V); J_r = _maybe_flip_sign(V_r, np.asarray(r_r.J))
    m_d = compute_metrics(V_d, J_d)
    m_r = compute_metrics(V_r, J_r)

    shift_mV = abs(m_d.V_oc - m_r.V_oc) * 1e3
    print(
        f"\nDirichlet: V_oc={m_d.V_oc*1e3:.1f} mV"
        f"\nRobin:     V_oc={m_r.V_oc*1e3:.1f} mV"
        f"\n|ΔV_oc| = {shift_mV:.1f} mV"
    )
    assert 1.0 <= shift_mV <= 150.0, (
        f"|ΔV_oc| = {shift_mV:.1f} mV is outside [1, 150] mV "
        f"(Robin hook inactive or unphysical)"
    )


@pytest.mark.regression
def test_twod_robin_microstructure_coexistence_smoke():
    """Robin contacts + grain boundary produce finite, ordered J-V (no NaN/Inf).

    Uses a coarse fast mesh — correctness is covered by the parity gate.
    """
    from perovskite_sim.twod.microstructure import GrainBoundary
    stack = _freeze_ions(load_device_from_yaml(ROBIN_PRESET))
    ms = Microstructure(grain_boundaries=(
        GrainBoundary(
            x_position=150e-9, width=5e-9,
            tau_n=5e-8, tau_p=5e-8,
            layer_role="absorber",
        ),
    ))
    r = run_jv_sweep_2d(
        stack=stack,
        microstructure=ms,
        lateral_length=300e-9,
        Nx=6,
        V_max=1.0,
        V_step=0.25,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=5,
        settle_t=1e-4,
    )
    V = np.asarray(r.V)
    J = np.asarray(r.J)
    assert np.all(np.isfinite(V)), "Non-finite V in Robin+GB sweep"
    assert np.all(np.isfinite(J)), "Non-finite J in Robin+GB sweep"
    J_sc_sign = _maybe_flip_sign(V, J)[0]
    assert J_sc_sign > 0, "J_sc should be positive under illumination (sign/convergence issue)"


@pytest.mark.regression
@pytest.mark.slow
def test_twod_robin_parity_vs_1d_aggressive_blocking():
    """Stage B(c.1) follow-up gate: 1D vs 2D Robin parity in the genuinely
    non-ohmic regime (S << ohmic limit on both contacts).

    The original parity gate (test_twod_robin_parity_vs_1d) uses
    selective_contacts_demo.yaml whose matched-carrier S=1e3 m/s is effectively
    ohmic at this device thickness — the bit-identical 1D/2D agreement there
    technically validates only the Dirichlet limit of Robin.  This test runs
    the same parity check at the aggressive-blocking S envelope used by the
    bounded-shift sanity test, where the Robin term materially shifts V_oc
    (~36 mV vs Dirichlet) so any sign-table or half-cell-weighting error in
    the 2D port would surface as a 1D/2D divergence here.

    The math at the boundary is identical between 1D and 2D — same
    selective_contact_flux primitive, same equilibrium densities, same
    half-cell weighting — so we expect agreement at or very near the
    displayed-precision level even in the non-ohmic regime.
    """
    base = _freeze_ions(load_device_from_yaml(PRESET))
    stack = replace(base,
        S_n_left=1e-4, S_p_left=1e-3,
        S_n_right=1e-3, S_p_right=1e-4,
    )

    r1 = run_jv_sweep(stack, N_grid=31, V_max=1.2, n_points=13, illuminated=True)
    V1 = np.asarray(r1.V_fwd)
    J1 = _maybe_flip_sign(V1, np.asarray(r1.J_fwd))
    m1 = compute_metrics(V1, J1)

    r2 = run_jv_sweep_2d(
        stack=stack,
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    V2 = np.asarray(r2.V)
    J2 = _maybe_flip_sign(V2, np.asarray(r2.J))
    m2 = compute_metrics(V2, J2)

    print(
        f"\nAggressive Robin 1D: V_oc={m1.V_oc*1e3:.4f} mV  J_sc={m1.J_sc:.4f} A/m²  FF={m1.FF:.6f}"
        f"\nAggressive Robin 2D: V_oc={m2.V_oc*1e3:.4f} mV  J_sc={m2.J_sc:.4f} A/m²  FF={m2.FF:.6f}"
        f"\nΔV_oc = {(m2.V_oc - m1.V_oc)*1e3:+.4f} mV"
        f"  ΔJ_sc/J_sc = {(m2.J_sc - m1.J_sc)/m1.J_sc:+.2e}"
        f"  ΔFF = {m2.FF - m1.FF:+.4e}"
    )

    assert abs(m2.V_oc - m1.V_oc) <= 1e-3, (
        f"Aggressive-Robin V_oc(2D)={m2.V_oc:.6f} V vs V_oc(1D)={m1.V_oc:.6f} V "
        f"(diff {(m2.V_oc - m1.V_oc)*1e3:.3f} mV, limit 1 mV)"
    )
    rel_jsc = abs(m2.J_sc - m1.J_sc) / abs(m1.J_sc)
    assert rel_jsc <= 5e-4, (
        f"Aggressive-Robin J_sc rel diff {rel_jsc:.2e} > 5e-4 "
        f"(2D={m2.J_sc:.4f}, 1D={m1.J_sc:.4f} A/m²)"
    )
    assert abs(m2.FF - m1.FF) <= 1e-3, (
        f"Aggressive-Robin FF(2D)={m2.FF:.6f} vs FF(1D)={m1.FF:.6f} "
        f"(diff {abs(m2.FF - m1.FF):.4f}, limit 1e-3)"
    )


def _stack_with_layer_params(stack, **layer_kwargs):
    """Push v_sat / ct_beta / pf_gamma overrides into per-layer MaterialParams."""
    new_layers = tuple(
        replace(layer, params=replace(layer.params, **layer_kwargs))
        for layer in stack.layers
    )
    return replace(stack, layers=new_layers)


@pytest.mark.regression
@pytest.mark.slow
def test_twod_field_mobility_disabled_path_bit_identical():
    """Stage B(c.2) bit-identical disabled-path regression.

    Compare two 2D J-V sweeps:
      A. Default preset, mode='full', no v_sat / pf_gamma
         → has_field_mobility=False (no params set).
      B. Same preset, mode='legacy', WITH aggressive v_sat=1e2
         → has_field_mobility=False (tier gate disables despite params).

    Both must produce IDENTICAL J-V to floating-point precision because
    the disabled path bypasses the recompute and the constant-D code path
    is unchanged.
    """
    base = _freeze_ions(load_device_from_yaml(PRESET))
    stack_off    = base                                                                    # default mode=full, no v_sat
    stack_legacy = replace(_stack_with_layer_params(base, v_sat_n=1e2, v_sat_p=1e2),
                           mode="legacy")                                                  # tier-disabled
    common_kw = dict(
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    r_off    = run_jv_sweep_2d(stack=stack_off,    **common_kw)
    r_legacy = run_jv_sweep_2d(stack=stack_legacy, **common_kw)
    V_off    = np.asarray(r_off.V);    J_off    = np.asarray(r_off.J)
    V_legacy = np.asarray(r_legacy.V); J_legacy = np.asarray(r_legacy.J)
    np.testing.assert_array_equal(V_off, V_legacy)
    np.testing.assert_allclose(J_off, J_legacy, rtol=1e-12, atol=0.0,
        err_msg="Disabled field-mobility path is not bit-identical")


@pytest.mark.regression
@pytest.mark.slow
def test_twod_field_mobility_parity_vs_1d():
    """Stage B(c.2) primary correctness gate: laterally-uniform 2D with face-normal
    μ(E) at v_sat=1e2 matches 1D Phase 3.2 within (1 mV / 5e-4 / 1e-3).

    In a lateral-uniform device E_x ≈ 0, so face-normal μ(E) reduces to
    "y-face μ(E) only" — exactly what 1D does. Expected: bit-identical or
    sub-microvolt deltas, mirroring the B(c.1) Robin parity gates.
    """
    base = _freeze_ions(load_device_from_yaml(PRESET))
    stack = _stack_with_layer_params(base, v_sat_n=1e2, v_sat_p=1e2)
    # 1D reference
    r1 = run_jv_sweep(stack, N_grid=31, V_max=1.2, n_points=13, illuminated=True)
    V1 = np.asarray(r1.V_fwd)
    J1 = _maybe_flip_sign(V1, np.asarray(r1.J_fwd))
    m1 = compute_metrics(V1, J1)
    # 2D Stage B(c.2)
    r2 = run_jv_sweep_2d(
        stack=stack,
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    V2 = np.asarray(r2.V)
    J2 = _maybe_flip_sign(V2, np.asarray(r2.J))
    m2 = compute_metrics(V2, J2)
    print(
        f"\nμ(E) 1D: V_oc={m1.V_oc*1e3:.4f} mV  J_sc={m1.J_sc:.4f} A/m²  FF={m1.FF:.6f}"
        f"\nμ(E) 2D: V_oc={m2.V_oc*1e3:.4f} mV  J_sc={m2.J_sc:.4f} A/m²  FF={m2.FF:.6f}"
        f"\nΔV_oc = {(m2.V_oc - m1.V_oc)*1e3:+.4f} mV"
        f"  ΔJ_sc/J_sc = {(m2.J_sc - m1.J_sc)/m1.J_sc:+.2e}"
        f"  ΔFF = {m2.FF - m1.FF:+.4e}"
    )
    assert abs(m2.V_oc - m1.V_oc) <= 1e-3, (
        f"μ(E) V_oc(2D)={m2.V_oc:.6f} V vs V_oc(1D)={m1.V_oc:.6f} V "
        f"(diff {(m2.V_oc - m1.V_oc)*1e3:.3f} mV, limit 1 mV)"
    )
    rel_jsc = abs(m2.J_sc - m1.J_sc) / abs(m1.J_sc)
    assert rel_jsc <= 5e-4, (
        f"μ(E) J_sc rel diff {rel_jsc:.2e} > 5e-4 "
        f"(2D={m2.J_sc:.4f}, 1D={m1.J_sc:.4f} A/m²)"
    )
    assert abs(m2.FF - m1.FF) <= 1e-3, (
        f"μ(E) FF(2D)={m2.FF:.6f} vs FF(1D)={m1.FF:.6f} "
        f"(diff {abs(m2.FF - m1.FF):.4f}, limit 1e-3)"
    )


@pytest.mark.regression
@pytest.mark.slow
def test_twod_field_mobility_bounded_shift():
    """v_sat=1e2 in 2D shifts J(V) measurably vs v_sat=0 baseline.

    Confirms the μ(E) hook is materially active rather than silently bypassed.
    Asserts max(|J_on - J_off| / |J_off|) > 1e-6 — same threshold as the 1D
    analog `test_field_mobility_changes_jv_curve`.

    A larger threshold (e.g. 1e-3) is not appropriate here because at v_sat=1e2
    on the Beer-Lambert MAPbI3 nip stack the field-mobility correction is
    physical but modest: empirically the relative J shift peaks at ~7e-4 across
    the sweep (CT-driven μ reduction in the absorber where E_y is largest, but
    flat-band regions near contacts contribute very little). The 1e-6 floor
    catches the "hook completely bypassed" case (shift = 0 to floating-point
    precision) without false-positiving on the genuine ~mid-1e-4 physical
    signal at this preset.
    """
    base = _freeze_ions(load_device_from_yaml(PRESET))
    stack_off = base
    stack_on  = _stack_with_layer_params(base, v_sat_n=1e2, v_sat_p=1e2)
    common_kw = dict(
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    r_off = run_jv_sweep_2d(stack=stack_off, **common_kw)
    r_on  = run_jv_sweep_2d(stack=stack_on,  **common_kw)
    J_off = np.asarray(r_off.J)
    J_on  = np.asarray(r_on.J)
    rel = np.max(np.abs(J_on - J_off) / (np.abs(J_off) + 1e-12))
    print(f"\nμ(E) bounded-shift: max(|ΔJ|/|J|) = {rel:.3e}")
    assert rel > 1e-6, (
        f"μ(E) shift max(|ΔJ|/|J|) = {rel:.3e} below 1e-6 — hook may be inactive"
    )


@pytest.mark.regression
def test_twod_field_mobility_robin_microstructure_coexistence_smoke():
    """μ(E) + Robin contacts + grain boundary on a coarse mesh produce a finite,
    well-ordered J-V (no NaN/Inf, J_sc>0). Cheap test — proves the three per-RHS
    hooks compose without solver hang."""
    from perovskite_sim.twod.microstructure import GrainBoundary
    base = _freeze_ions(load_device_from_yaml(PRESET))
    stack = replace(_stack_with_layer_params(base, v_sat_n=1e3, v_sat_p=1e3),
        S_n_left=1e-4, S_p_left=1e-3,
        S_n_right=1e-3, S_p_right=1e-4,
    )
    ms = Microstructure(grain_boundaries=(
        GrainBoundary(
            x_position=150e-9, width=5e-9,
            tau_n=5e-8, tau_p=5e-8,
            layer_role="absorber",
        ),
    ))
    r = run_jv_sweep_2d(
        stack=stack,
        microstructure=ms,
        lateral_length=300e-9,
        Nx=6,
        V_max=1.0,
        V_step=0.25,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=5,
        settle_t=1e-4,
    )
    V = np.asarray(r.V)
    J = np.asarray(r.J)
    assert np.all(np.isfinite(V)), "Non-finite V in μ(E)+Robin+GB sweep"
    assert np.all(np.isfinite(J)), "Non-finite J in μ(E)+Robin+GB sweep"
    J_sc_sign = _maybe_flip_sign(V, J)[0]
    assert J_sc_sign > 0, "J_sc should be positive under illumination"


@pytest.mark.regression
@pytest.mark.slow
def test_twod_radiative_reabsorption_disabled_path_bit_identical():
    """Stage B(c.3) bit-identical disabled-path regression.

    Compare two 2D J-V sweeps on the BL preset:
      A. mode='full'  → has_radiative_reabsorption_2d=False (no TMM, no P_esc)
      B. mode='legacy' → has_radiative_reabsorption_2d=False (tier disables)

    Both must produce IDENTICAL J-V because Stage B(c.3) keeps the constant-G
    code path (G_to_use is mat.G_optical when the flag is False).
    The chi=Eg=0 BL preset means the other tier-gated flags (TE, band offsets,
    etc.) don't change physics either, so this test isolates the B(c.3) gate.
    """
    base = _freeze_ions(load_device_from_yaml(PRESET))    # PRESET is BL
    stack_full   = base                                   # mode=full default
    stack_legacy = replace(base, mode="legacy")
    common_kw = dict(
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    r_full   = run_jv_sweep_2d(stack=stack_full,   **common_kw)
    r_legacy = run_jv_sweep_2d(stack=stack_legacy, **common_kw)
    np.testing.assert_array_equal(r_full.V, r_legacy.V)
    np.testing.assert_allclose(
        r_full.J, r_legacy.J, rtol=1e-12, atol=0.0,
        err_msg="Stage B(c.3) disabled path is not bit-identical between mode=full and mode=legacy on BL",
    )


RR_TMM_PRESET = "configs/nip_MAPbI3_tmm.yaml"


def _voc_2d_via_warm_start(
    stack: DeviceStack,
    microstructure: Microstructure,
    *,
    voc_seed: float,
    lateral_length: float = 500e-9,
    Nx: int = 4,
    Ny_per_layer: int = 10,
    settle_t: float = 1e-3,
    delta_V: float = 5e-3,
) -> float:
    """Compute V_oc on 2D by bootstrapping from the 1D V_oc steady-state and
    bracketing V_oc with two short 2D settles at ``voc_seed ± delta_V``.

    Why a warm-start bracket rather than a full J-V sweep: Stage A 2D + TMM
    cannot Newton-contract through the V≈0.2 V diode-injection knee — see
    project memory ``project_stage_a_2d_tmm_newton.md``. Walking V from 0
    to V_oc would hang. Bootstrapping 2D directly from the 1D V_oc state
    and bracketing V_oc with two short settles within ±5 mV of voc_seed
    dodges the knee entirely. The diode I-V is approximately linear over a
    10 mV window near V_oc (verified empirically: ΔJ/ΔV ≈ const), so a
    linear interpolation of J vs V across the bracket gives a sub-mV V_oc
    estimate.

    Returns
    -------
    voc_2d : float, the V_oc on the 2D solver, V.
    """
    from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
    from perovskite_sim.solver.mol import StateVec
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.twod.grid_2d import build_grid_2d
    from perovskite_sim.twod.solver_2d import (
        build_material_arrays_2d, run_transient_2d,
        extract_snapshot_2d, compute_terminal_current_2d,
    )

    # Use the same per-layer node count for 1D bootstrap and 2D — the 1D
    # state vector is broadcast to the 2D grid, so Ny must match exactly.
    elec = electrical_layers(stack)
    layers_compat = [Layer(L.thickness, Ny_per_layer) for L in elec]
    x_1d = multilayer_grid(layers_compat)

    y_1d = solve_illuminated_ss(x_1d, stack, V_app=voc_seed, t_settle=settle_t)
    sv = StateVec.unpack(y_1d, len(x_1d))

    grid_2d = build_grid_2d(layers_compat, lateral_length=lateral_length, Nx=Nx)
    mat = build_material_arrays_2d(
        grid_2d, stack, microstructure,
        lateral_bc="periodic", P_ion_static_1d=sv.P,
    )
    Ny, Nx_nodes = grid_2d.Ny, grid_2d.Nx
    n_2d = np.broadcast_to(sv.n[:, None], (Ny, Nx_nodes)).copy()
    p_2d = np.broadcast_to(sv.p[:, None], (Ny, Nx_nodes)).copy()
    y_2d = np.concatenate([n_2d.flatten(), p_2d.flatten()])

    def _settle_J(V: float) -> float:
        y_settled = run_transient_2d(
            y_2d, mat, V_app=V, t_end=settle_t,
            max_step=settle_t / 50.0, max_nfev=100_000,
        )
        snap = extract_snapshot_2d(y_settled, mat, V_app=V)
        return float(compute_terminal_current_2d(snap))

    V_a = voc_seed - delta_V
    V_b = voc_seed + delta_V
    J_a = _settle_J(V_a)
    J_b = _settle_J(V_b)
    if abs(J_b - J_a) < 1e-9:
        raise RuntimeError(
            f"_voc_2d_via_warm_start: degenerate bracket — J_a={J_a}, J_b={J_b} "
            f"at V_a={V_a}, V_b={V_b}"
        )
    voc_2d = V_a - J_a * (V_b - V_a) / (J_b - J_a)
    return voc_2d


def _voc_1d_via_run_suns_voc(stack: DeviceStack, *, t_settle: float = 1e-3) -> float:
    from perovskite_sim.experiments.suns_voc import run_suns_voc
    res = run_suns_voc(stack, suns_levels=(1.0,), N_grid=31, t_settle=t_settle)
    return float(res.V_oc[0])


@pytest.mark.regression
@pytest.mark.slow
def test_twod_radiative_reabsorption_parity_vs_1d():
    """Stage B(c.3) fast V_oc-only smoke: lateral-uniform 2D V_oc matches
    1D Phase 3.1b V_oc within 5 mV on TMM preset with PR + reabsorption ON.

    Originally written as a workaround for the Stage A + TMM Newton stall
    (project_stage_a_2d_tmm_newton.md), now retained as a quick (~12 s)
    feedback test: bootstrap 2D from a 1D V_oc steady-state, bracket V_oc
    with two short ±5 mV 2D settles, linearly interpolate. The companion
    test_twod_radiative_reabsorption_full_sweep_parity_vs_1d does the
    full J_sc/FF/V_oc parity on the same preset using the bisection-in-
    time path, but is much slower (~15 min) — keep both so a developer
    iterating on B(c.3) can get fast feedback while CI gates on the full
    parity.
    """
    base = _freeze_ions(load_device_from_yaml(RR_TMM_PRESET))
    voc_1d = _voc_1d_via_run_suns_voc(base)
    voc_2d = _voc_2d_via_warm_start(base, Microstructure(), voc_seed=voc_1d)
    delta_mV = (voc_2d - voc_1d) * 1e3
    print(
        f"\nrr 1D V_oc = {voc_1d*1e3:.4f} mV"
        f"\nrr 2D V_oc = {voc_2d*1e3:.4f} mV"
        f"\nΔV_oc      = {delta_mV:+.4f} mV  (limit ±5 mV)"
    )
    assert abs(delta_mV) <= 5.0, (
        f"rr V_oc(2D)={voc_2d:.6f} V vs V_oc(1D)={voc_1d:.6f} V "
        f"(diff {delta_mV:.3f} mV, limit 5 mV)"
    )


@pytest.mark.regression
@pytest.mark.slow
def test_twod_radiative_reabsorption_full_sweep_parity_vs_1d():
    """Stage B(c.3) primary correctness gate: full V_max=1.2 J-V sweep on
    nip_MAPbI3_tmm.yaml with PR + reabsorption ON, lateral-uniform 2D
    matches 1D within (5 mV V_oc / 5e-4 J_sc / 1e-3 FF). Replaces the
    V_oc-only workaround that was needed before bisection-in-time landed
    in run_jv_sweep_2d (project_stage_a_2d_tmm_newton.md → RESOLVED).

    Uses settle_t=1e-5 — at the original 1D-mirrored settle_t=1e-3 the
    Stage A 2D solver burns through the bisection budget at high forward
    bias even on legacy mode (per the resolved-memory diagnostic notes).
    settle_t=1e-5 is well inside the carrier-relaxation window for Stage
    A (which has no ion drift), so it changes nothing physically; it just
    gives Newton a smaller per-step state change to contract on.

    Slow (~15 min on a single-thread BLAS box) because each voltage step
    triggers bisection a few levels deep. The companion fast V_oc-only
    smoke test_twod_radiative_reabsorption_parity_vs_1d (~12 s) is the
    right inner-loop check while iterating on B(c.3).
    """
    base = _freeze_ions(load_device_from_yaml(RR_TMM_PRESET))
    # 1D reference (same parameters the 1D regression suite uses)
    r1 = run_jv_sweep(base, N_grid=31, V_max=1.2, n_points=13, illuminated=True)
    V1 = np.asarray(r1.V_fwd)
    J1 = _maybe_flip_sign(V1, np.asarray(r1.J_fwd))
    m1 = compute_metrics(V1, J1)
    # 2D Stage B(c.3) with bisection — settle_t=1e-5 to keep wall time bounded.
    r2 = run_jv_sweep_2d(
        stack=base,
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-5,
    )
    V2 = np.asarray(r2.V)
    J2 = _maybe_flip_sign(V2, np.asarray(r2.J))
    m2 = compute_metrics(V2, J2)
    print(
        f"\nrr full 1D: V_oc={m1.V_oc*1e3:.4f} mV  J_sc={m1.J_sc:.4f} A/m²  FF={m1.FF:.6f}"
        f"\nrr full 2D: V_oc={m2.V_oc*1e3:.4f} mV  J_sc={m2.J_sc:.4f} A/m²  FF={m2.FF:.6f}"
        f"\nΔV_oc = {(m2.V_oc - m1.V_oc)*1e3:+.4f} mV"
        f"  ΔJ_sc/J_sc = {(m2.J_sc - m1.J_sc)/m1.J_sc:+.2e}"
        f"  ΔFF = {m2.FF - m1.FF:+.4e}"
    )
    assert abs(m2.V_oc - m1.V_oc) <= 5e-3, (
        f"rr full V_oc(2D)={m2.V_oc:.6f} V vs V_oc(1D)={m1.V_oc:.6f} V "
        f"(diff {(m2.V_oc - m1.V_oc)*1e3:.3f} mV, limit 5 mV)"
    )
    rel_jsc = abs(m2.J_sc - m1.J_sc) / abs(m1.J_sc)
    assert rel_jsc <= 5e-4, (
        f"rr full J_sc rel diff {rel_jsc:.2e} > 5e-4 "
        f"(2D={m2.J_sc:.4f}, 1D={m1.J_sc:.4f} A/m²)"
    )
    assert abs(m2.FF - m1.FF) <= 1e-3, (
        f"rr full FF(2D)={m2.FF:.6f} vs FF(1D)={m1.FF:.6f} "
        f"(diff {abs(m2.FF - m1.FF):.4e}, limit 1e-3)"
    )


@pytest.mark.regression
@pytest.mark.slow
def test_twod_radiative_reabsorption_voc_boost_in_literature_window(monkeypatch):
    """Stage B(c.3) V_oc boost vs PR-off lies in [40, 100] mV on the 2D
    solver — verifies the per-RHS reabsorption source actually delivers
    the literature signature on 2D, not just on 1D.

    V_oc-only path (see test_twod_radiative_reabsorption_parity_vs_1d
    docstring for why). Computes 2D V_oc twice on the same TMM stack:
    once with FULL tier (PR + reabsorption ON), once with FULL tier but
    use_photon_recycling=False AND use_radiative_reabsorption=False (and
    everything else still on). Mirrors 1D
    test_radiative_reabsorption_preserves_voc_boost via monkeypatch on
    resolve_mode in both ``perovskite_sim.models.mode`` (where 2D
    solver_2d picks it up via local imports) and
    ``perovskite_sim.solver.mol`` (where 1D run_suns_voc binds it at
    module load).

    Comparing tier-vs-tier (FULL vs LEGACY) would change far more than
    just the rr+PR flags — LEGACY also disables TMM, TE, dual ions,
    trap profile, T-scaling — so the V_oc shift would mix several
    physics effects and not be the literature reabsorption signature.
    """
    import perovskite_sim.models.mode as _mode_mod
    import perovskite_sim.solver.mol as _mol_mod
    from perovskite_sim.models.mode import FULL

    # Use the radiative-limit preset (configs/radiative_limit.yaml): bulk
    # radiative recombination dominates because non-radiative channels are
    # killed (τ → 1e3 s, Auger → 0, interface SRV → 0). On this stack the
    # PR-on V_oc boost lands cleanly in the literature [40, 100] mV
    # window; on a regular MAPbI3 stack (RR_TMM_PRESET) non-radiative
    # losses swamp the boost down to sub-mV.
    rr_limit = _freeze_ions(load_device_from_yaml("configs/radiative_limit.yaml"))
    mode_on = FULL                                       # PR + rr ON, all else FULL
    mode_off = replace(
        FULL,
        use_photon_recycling=False,
        use_radiative_reabsorption=False,
    )                                                    # only PR + rr flipped off

    def _voc_2d_under_mode(active_mode):
        monkeypatch.setattr(_mode_mod, "resolve_mode", lambda _arg: active_mode)
        monkeypatch.setattr(_mol_mod, "resolve_mode",  lambda _arg: active_mode)
        voc_1d = _voc_1d_via_run_suns_voc(rr_limit)
        return _voc_2d_via_warm_start(rr_limit, Microstructure(), voc_seed=voc_1d), voc_1d

    voc_2d_off, voc_1d_off = _voc_2d_under_mode(mode_off)
    voc_2d_on,  voc_1d_on  = _voc_2d_under_mode(mode_on)
    boost_mV = (voc_2d_on - voc_2d_off) * 1e3
    print(
        f"\nrr OFF 2D V_oc = {voc_2d_off*1e3:.4f} mV  (1D seed {voc_1d_off*1e3:.4f})"
        f"\nrr ON  2D V_oc = {voc_2d_on*1e3:.4f} mV  (1D seed {voc_1d_on*1e3:.4f})"
        f"\nΔV_oc(boost) = {boost_mV:.2f} mV  (literature window [40, 100] mV)"
    )
    assert 40.0 <= boost_mV <= 100.0, (
        f"V_oc boost = {boost_mV:.2f} mV outside literature window [40, 100] mV "
        f"(2D ON: {voc_2d_on*1e3:.2f} mV, 2D OFF: {voc_2d_off*1e3:.2f} mV)"
    )


@pytest.mark.regression
@pytest.mark.slow
def test_twod_radiative_reabsorption_robin_field_mobility_coexistence_smoke():
    """All four per-RHS / per-stack hooks (radiative reabsorption + Robin
    contacts + μ(E) + grain boundary) compose without NaN/Inf during a
    short V=0 settle. V_oc-only path can't be used here because aggressive
    Robin S values and the GB shift V_oc significantly off the 1D seed,
    and the V≈0.2 V Stage A knee (project_stage_a_2d_tmm_newton.md) blocks
    a sweep approach. V=0 settle always converges (the equilibrium state
    is well-defined) and exercises all four hooks during integration.

    Does NOT assert physical correctness of the composite — that would
    require an independent reference. Asserts only finiteness of the
    settled state and the terminal current.
    """
    from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
    from perovskite_sim.solver.mol import StateVec
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.twod.grid_2d import build_grid_2d
    from perovskite_sim.twod.solver_2d import (
        build_material_arrays_2d, run_transient_2d,
        extract_snapshot_2d, compute_terminal_current_2d,
    )
    from perovskite_sim.twod.microstructure import GrainBoundary

    base = _freeze_ions(load_device_from_yaml(RR_TMM_PRESET))
    stack = replace(_stack_with_layer_params(base, v_sat_n=1e3, v_sat_p=1e3),
        S_n_left=1e-4, S_p_left=1e-3,
        S_n_right=1e-3, S_p_right=1e-4,
    )
    ms = Microstructure(grain_boundaries=(
        GrainBoundary(
            x_position=150e-9, width=5e-9,
            tau_n=5e-8, tau_p=5e-8,
            layer_role="absorber",
        ),
    ))

    # Coarse 2D grid — Ny_per_layer=5, Nx=6, lateral 300 nm.
    Ny_per_layer = 5
    elec = electrical_layers(stack)
    layers_compat = [Layer(L.thickness, Ny_per_layer) for L in elec]
    x_1d = multilayer_grid(layers_compat)

    # Bootstrap from 1D illuminated steady-state at V=0.
    y_1d = solve_illuminated_ss(x_1d, stack, V_app=0.0, t_settle=1e-3)
    sv = StateVec.unpack(y_1d, len(x_1d))

    grid_2d = build_grid_2d(layers_compat, lateral_length=300e-9, Nx=6)
    mat = build_material_arrays_2d(
        grid_2d, stack, ms, lateral_bc="periodic", P_ion_static_1d=sv.P,
    )
    Ny, Nx_nodes = grid_2d.Ny, grid_2d.Nx
    n_2d = np.broadcast_to(sv.n[:, None], (Ny, Nx_nodes)).copy()
    p_2d = np.broadcast_to(sv.p[:, None], (Ny, Nx_nodes)).copy()
    y_2d = np.concatenate([n_2d.flatten(), p_2d.flatten()])

    settle_t = 1e-4
    y_settled = run_transient_2d(
        y_2d, mat, V_app=0.0, t_end=settle_t,
        max_step=settle_t / 50.0, max_nfev=100_000,
    )
    snap = extract_snapshot_2d(y_settled, mat, V_app=0.0)
    J_at_zero = compute_terminal_current_2d(snap)
    print(
        f"\ncoexistence smoke (V=0 settle, all four hooks ON): "
        f"J(V=0) = {J_at_zero:.4e} A/m²"
    )
    assert np.all(np.isfinite(y_settled)), (
        "Non-finite state after rr+Robin+μ(E)+GB V=0 settle"
    )
    assert np.isfinite(J_at_zero), (
        "Non-finite J after rr+Robin+μ(E)+GB V=0 settle"
    )

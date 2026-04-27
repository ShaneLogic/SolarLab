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

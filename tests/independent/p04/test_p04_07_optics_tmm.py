"""P04-07 oracle: TMM energy conservation, closed forms, control-volume photons, budget."""

from __future__ import annotations

import math

import numpy as np
import pytest

import p04_adapter as adapter
import p04_legacy as legacy
from p04_gates import gate
from p04_oracle_optics import (
    Q_E,
    airy_slab,
    fresnel_interface,
    tmm_normal_incidence,
    trapezoid_weights,
)

N_WL = 100
WL_NM = np.linspace(350.0, 750.0, N_WL)
WL = WL_NM * 1e-9
K = (np.where(WL_NM < 380, 0.05, 0.001), 0.4 * np.exp(-(WL_NM - 400) / 150), np.where(WL_NM < 420, 0.02, 0.001))
N_REAL = (2.4, 2.5, 1.8)
D = (50e-9, 400e-9, 150e-9)
STACK = [(n + 1j * k, d) for n, k, d in zip(N_REAL, K, D)]


def _legacy_layers():
    opt = legacy.optics()
    return [opt.TMMLayer(d=d, n=np.full(N_WL, n), k=k) for n, k, d in zip(N_REAL, K, D)]


@pytest.mark.parametrize("n_sub", (1.0, 1.5, 2.0 + 0.1j))
def test_oracle_rta_by_two_independent_routes(n_sub):
    g = gate("P0407-TMM-RTA")
    res = tmm_normal_incidence(WL, STACK, 1.0, n_sub)
    a_p = res.layer_absorptance_poynting().sum(axis=0)
    a_i = res.layer_absorptance_integral().sum(axis=0)
    assert np.max(np.abs(res.R + res.T + a_p - 1.0)) <= g.abs_limit
    assert np.max(np.abs(res.R + res.T + a_i - 1.0)) <= g.abs_limit
    assert np.max(np.abs(a_p - a_i)) <= g.abs_limit


def test_oracle_closed_forms():
    g = gate("P0407-TMM-CLOSED")
    R0, T0 = fresnel_interface(1.0, 2.5 + 0.3j)
    bare = tmm_normal_incidence(np.array([500e-9]), [], 1.0, 2.5 + 0.3j)
    assert abs(bare.R[0] - R0) <= g.abs_limit and abs(bare.T[0] - T0) <= g.abs_limit
    Ra, Ta = airy_slab(WL, 1.0, 2.5 + 1j * K[1], 400e-9, 1.5)
    slab = tmm_normal_incidence(WL, [(2.5 + 1j * K[1], 400e-9)], 1.0, 1.5)
    assert np.max(np.abs(Ra - slab.R)) <= g.abs_limit and np.max(np.abs(Ta - slab.T)) <= g.abs_limit
    n1 = math.sqrt(1.0 * 2.25)
    lam = np.array([600e-9])
    qw = tmm_normal_incidence(lam, [(np.array([n1 + 0j]), lam[0] / (4 * n1))], 1.0, 2.25)
    assert qw.R[0] <= 1e-28
    hw = tmm_normal_incidence(lam, [(np.array([2.0 + 0j]), lam[0] / 4.0)], 1.0, 1.5)
    assert abs(hw.R[0] - fresnel_interface(1.0, 1.5)[0]) <= g.abs_limit
    lossless = tmm_normal_incidence(WL, [(2.0 + 0j, 100e-9), (2.5 + 0j, 300e-9)], 1.0, 1.0)
    assert np.max(np.abs(lossless.layer_absorptance_integral())) <= 1e-15
    thick = tmm_normal_incidence(WL, [(2.5 + 3.0j, 50e-6)], 1.0, 1.0)
    assert np.all(np.isfinite(thick.layer_absorptance_integral()))
    assert np.max(np.abs(thick.R + thick.T + thick.layer_absorptance_integral().sum(0) - 1.0)) <= gate("P0407-TMM-RTA").abs_limit


@pytest.mark.parametrize("n_sub", (1.0, 1.5))
def test_legacy_tmm_matches_oracle(n_sub):
    g = gate("P0407-TMM-RTA")
    opt = legacy.optics()
    layers = _legacy_layers()
    res = tmm_normal_incidence(WL, STACK, 1.0, n_sub)
    R = opt.tmm_reflectance(layers, WL, 1.0, n_sub)
    S, _ = opt._transfer_matrix_stack(layers, WL, 1.0, n_sub)
    T = n_sub * np.abs(1.0 / S[:, 0, 0]) ** 2
    assert np.max(np.abs(R - res.R)) <= g.abs_limit and np.max(np.abs(T - res.T)) <= g.abs_limit
    bounds = np.concatenate(([0.0], np.cumsum(D)))
    C = opt.tmm_cumulative_absorptance(layers, WL, bounds, bounds, 1.0, n_sub)
    assert np.max(np.abs(np.diff(C, axis=0) - res.layer_absorptance_integral())) <= g.abs_limit
    assert np.max(np.abs(C[-1] - (1.0 - res.R - res.T))) <= g.abs_limit


def _oracle_cell_absorbed(res, faces, flux):
    bounds = np.concatenate(([0.0], np.cumsum(res.d)))
    C = np.zeros((faces.size, res.wavelengths_m.size))
    for i, f in enumerate(faces):
        j = min(int(np.searchsorted(bounds, f, side="right")) - 1, len(res.d) - 1)
        prior = res.layer_absorptance_integral()[:j].sum(axis=0) if j else 0.0
        C[i] = prior + res.cumulative_absorbed(j, np.array([f - bounds[j]]))[0]
    w = trapezoid_weights(res.wavelengths_m)
    return (np.diff(C, axis=0) * flux[None, :]) @ w


def test_legacy_control_volume_photons_match_oracle():
    g = gate("P0407-TMM-CELL")
    opt = legacy.optics()
    faces = np.linspace(0.0, sum(D), 31)
    flux = np.full(N_WL, 3e24)
    bounds = np.concatenate(([0.0], np.cumsum(D)))
    out = opt.tmm_absorbed_photon_flux_per_cell(_legacy_layers(), WL, flux, faces, bounds, 1.0, 1.5)
    ref = _oracle_cell_absorbed(tmm_normal_incidence(WL, STACK, 1.0, 1.5), faces, flux)
    assert np.max(np.abs(out - ref)) <= g.rel_limit * np.max(ref)


def test_oracle_device_photon_budget_with_parasitic_layers():
    """incident = R + T + A_electrical + A_parasitic; parasitic never becomes G."""

    g = gate("P0407-TMM-BUDGET")
    ito = (1.9 + 0.01j, 150e-9)
    stack = [ito, *STACK, (0.2 + 3.5j, 80e-9)]  # transparent conductor ... metal-like back layer
    res = tmm_normal_incidence(WL, stack, 1.0, 1.0)
    A = res.layer_absorptance_integral()
    flux = np.full(N_WL, 3e24)
    w = trapezoid_weights(WL)
    incident = flux @ w
    electrical = (A[1:4] * flux).sum(axis=0) @ w
    parasitic = ((A[0] + A[4]) * flux) @ w
    lost = ((res.R + res.T) * flux) @ w
    assert abs((electrical + parasitic + lost) / incident - 1.0) <= g.rel_limit
    assert parasitic > 0.0 and electrical < incident


def test_legacy_device_tmm_budget_matches_quoted_fixture_value():
    """Existing evidence: perovskite-sim/CLAUDE.md and solver/mol.py quote 225.001 A/m^2.

    ``ionmonger_benchmark_tmm`` exact absorbed budget at N_grid = 30 is quoted
    as "225.00" (docstring) / "225.001" (CLAUDE.md, ``tmm_generation``
    docstring); the cell-exact path must reproduce it on a coarse mesh.
    """

    from perovskite_sim.solver.mol import _compute_tmm_generation

    stack = legacy.device_fixture("ionmonger_benchmark_tmm.yaml")
    for N in (10, 30):
        x = legacy.electrical_grid(stack, N)
        G = _compute_tmm_generation(x, stack)
        jsum = Q_E * math.fsum(G * legacy.generation().dual_cell_widths(x))
        assert abs(jsum - 225.001) <= 0.0005, f"N={N}: {jsum:.6f} A/m^2"


# ---------------------------------------------------------------- candidate


@pytest.mark.parametrize("n_sub", (1.0, 1.5))
def test_candidate_tmm_rta(n_sub):
    g = gate("P0407-TMM-RTA")
    R, T, A = (np.asarray(a) for a in adapter.tmm_reflect_transmit_absorb(WL, STACK, 1.0, n_sub))
    res = tmm_normal_incidence(WL, STACK, 1.0, n_sub)
    assert np.max(np.abs(R - res.R)) <= g.abs_limit and np.max(np.abs(T - res.T)) <= g.abs_limit
    assert np.max(np.abs(A - res.layer_absorptance_integral())) <= g.abs_limit
    assert np.max(np.abs(R + T + A.sum(axis=0) - 1.0)) <= g.abs_limit


def test_candidate_control_volume_photons():
    g = gate("P0407-TMM-CELL")
    faces = np.linspace(0.0, sum(D), 31)
    flux = np.full(N_WL, 3e24)
    out = np.asarray(adapter.tmm_cell_absorbed_photons(WL, flux, STACK, faces))
    ref = _oracle_cell_absorbed(tmm_normal_incidence(WL, STACK, 1.0, 1.0), faces, flux)
    assert np.max(np.abs(out - ref)) <= g.rel_limit * np.max(ref)

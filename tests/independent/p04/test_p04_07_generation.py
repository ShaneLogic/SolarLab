"""P04-07 oracle: monochromatic Beer-Lambert control-volume generation, units, direction."""

from __future__ import annotations

import math

import numpy as np
import pytest

import p04_adapter as adapter
import p04_legacy as legacy
from p04_gates import EPS, gate
from p04_oracle_electrostatics import shifted_stack_for_legacy_rule, LayeredStack
from p04_oracle_geometry import (
    GEOMETRY_LEGACY_1D,
    GEOMETRY_LEGACY_2D,
    GEOMETRY_R1_TWO_SIDED,
    integer_endpoint_widths,
    right_owner_layer_index,
    two_sided_faces,
)
from p04_oracle_optics import (
    C_LIGHT,
    H_PLANCK,
    absorbed_between,
    bl_layer_absorption,
    exact_optical_depth,
    nodal_trapezoid_optical_depth,
    photon_flux_per_m_from_irradiance_per_nm,
)
from test_p04_01_poisson import TH, interface_node_grid, two_sided_grid

ALPHA = (2.0e5, 1.3e7, 4.0e6)  # absorbing HTL and ETL: exercises boundary cells
PHI = 1.4e21


def _legacy_bl(x, alpha_node, Phi):
    return legacy.generation().beer_lambert_generation(x, alpha_node, Phi)


@pytest.mark.parametrize("alpha", (ALPHA, (0.0, 1.3e7, 0.0)))
@pytest.mark.parametrize("Ns", ((20, 40, 15), (2, 3, 1), (200, 400, 150)))
def test_legacy_telescoping_and_photon_ceiling(alpha, Ns):
    gt, gb = gate("P0407-BL-TELE"), gate("P0407-BL-BOUND")
    x = interface_node_grid(Ns)
    a_node = np.array(alpha)[right_owner_layer_index(x, TH)]
    G = _legacy_bl(x, a_node, PHI)
    w = integer_endpoint_widths(x)
    total = math.fsum(G * w)
    tau_d = nodal_trapezoid_optical_depth(x, a_node)[-1]
    expected = PHI * -math.expm1(-tau_d)
    assert abs(total / expected - 1.0) <= gt.rel_limit
    assert total <= PHI
    assert np.all(G >= 0.0)
    # Geometry identity: the discrete depth is the true depth of the shifted planes.
    shifted = shifted_stack_for_legacy_rule(x, LayeredStack(TH, (1.0, 1.0, 1.0)))
    assert abs(tau_d - math.fsum(a * d for a, d in zip(alpha, shifted.thicknesses_m))) <= 1e-13 * tau_d
    del gb


def test_legacy_per_layer_split_converges_first_order_to_the_analytic_integral():
    """History: the right-owner nodal channel splits photons with an O(h) bias."""

    exact = bl_layer_absorption(ALPHA, TH, PHI)
    errs, hs = [], []
    for f in (1, 2, 4, 8, 16):
        x = interface_node_grid((20 * f, 40 * f, 15 * f))
        owner = right_owner_layer_index(x, TH)
        absorbed = _legacy_bl(x, np.array(ALPHA)[owner], PHI) * integer_endpoint_widths(x)
        per = np.array([math.fsum(absorbed[owner == k]) for k in range(3)])
        errs.append(np.max(np.abs(per - exact) / exact))
        hs.append(np.max(np.diff(x)))
    p = np.log(np.array(errs[:-1]) / errs[1:]) / np.log(np.array(hs[:-1]) / hs[1:])
    assert 0.8 <= p[-1] <= 1.3, f"orders {np.round(p, 3)}"


def test_oracle_cell_exact_generation_reproduces_layer_integrals():
    """Control volumes aligned with the planes reproduce (BL) per layer exactly."""

    g = gate("P0407-BL-LAYER")
    x = two_sided_grid((9, 17, 5))
    faces = two_sided_faces(x, tuple(np.cumsum(TH)[:-1]))
    tau = exact_optical_depth(faces, ALPHA, TH)
    absorbed = absorbed_between(tau[:-1], tau[1:], PHI)
    bounds = np.cumsum(TH)
    owner = np.searchsorted(bounds, x, side="left").clip(0, 2)
    exact = bl_layer_absorption(ALPHA, TH, PHI)
    for k in range(3):
        assert abs(math.fsum(absorbed[owner == k]) / exact[k] - 1.0) <= g.rel_limit
    assert abs(math.fsum(absorbed) / (PHI * -math.expm1(-sum(a * d for a, d in zip(ALPHA, TH)))) - 1.0) <= g.rel_limit


@pytest.mark.parametrize("geometry", (GEOMETRY_LEGACY_1D, GEOMETRY_R1_TWO_SIDED))
def test_candidate_bl_layer_integrals_and_bounds(geometry):
    gt, gl = gate("P0407-BL-TELE"), gate("P0407-BL-LAYER")
    x = interface_node_grid() if geometry == GEOMETRY_LEGACY_1D else two_sided_grid((9, 17, 5))
    owner = right_owner_layer_index(x, TH) if geometry == GEOMETRY_LEGACY_1D else \
        np.searchsorted(np.cumsum(TH), x, side="left").clip(0, 2)
    planes = () if geometry == GEOMETRY_LEGACY_1D else tuple(np.cumsum(TH)[:-1])  # two-sided needs the planes
    G, w = (np.asarray(a, dtype=float) for a in
            adapter.beer_lambert_generation(geometry, x, np.array(ALPHA)[owner], PHI, interfaces=planes))
    total = math.fsum(G * w)
    assert total <= PHI and np.all(G >= 0.0)
    if geometry == GEOMETRY_LEGACY_1D:
        tau_d = nodal_trapezoid_optical_depth(x, np.array(ALPHA)[owner])[-1]
        assert abs(total / (PHI * -math.expm1(-tau_d)) - 1.0) <= gt.rel_limit
        ref = _legacy_bl(x, np.array(ALPHA)[owner], PHI)
        if adapter.change_class("beer_lambert") == "M":
            assert np.array_equal(G, ref)
    else:
        exact = bl_layer_absorption(ALPHA, TH, PHI)
        for k in range(3):
            assert abs(math.fsum((G * w)[owner == k]) / exact[k] - 1.0) <= gl.rel_limit


@pytest.mark.parametrize("subject", ("legacy", "candidate"))
def test_zero_channels_are_exact(subject):
    x = interface_node_grid()
    owner = right_owner_layer_index(x, TH)
    a_node = np.array((0.0, 1.3e7, 0.0))[owner]

    def run(alpha, Phi):
        if subject == "legacy":
            return _legacy_bl(x, alpha, Phi)
        return adapter.beer_lambert_generation(GEOMETRY_LEGACY_1D, x, alpha, Phi)[0]

    G = np.asarray(run(a_node, PHI))
    first_abs = int(np.flatnonzero(owner == 1)[0])
    dark = np.zeros(x.size, dtype=bool)
    dark[: first_abs - 1] = True  # both dual faces inside the transparent HTL
    assert np.all(G[dark] == 0.0)
    assert np.all(np.asarray(run(a_node, 0.0)) == 0.0)
    assert np.all(np.asarray(run(np.zeros_like(a_node), PHI)) == 0.0)


def test_candidate_contact_cells_keep_the_channel_convention():
    """Integer endpoint: a contact cell weighs the full dx but absorbs only over dx/2.

    The absorbed photons of each cell are compared per cell with the oracle's
    discrete depth at the midpoint faces (bound 8 eps Phi: two rounded
    exponentials <= 1 and one subtraction); the factor-2 alternative (absorbing
    over the full dx) is excluded by orders of magnitude.  The 2D half-endpoint
    channel keeps the same photons over dx/2: exactly twice the source.
    """

    gt = gate("P0407-BL-TELE")
    x = interface_node_grid()
    a_node = np.array(ALPHA)[right_owner_layer_index(x, TH)]
    G, w = (np.asarray(v) for v in adapter.beer_lambert_generation(GEOMETRY_LEGACY_1D, x, a_node, PHI))
    h = np.diff(x)
    assert np.array_equal(w, integer_endpoint_widths(x)) and w[0] == h[0] and w[-1] == h[-1]
    tau = nodal_trapezoid_optical_depth(x, a_node)
    faces = np.concatenate(([tau[0]], 0.5 * (tau[:-1] + tau[1:]), [tau[-1]]))
    absorbed = PHI * (np.exp(-faces[:-1]) - np.exp(-faces[1:]))
    assert np.max(np.abs(G * w - absorbed)) <= 8 * EPS * PHI
    full_cell = PHI * -math.expm1(-tau[1])
    assert abs(G[0] * w[0] - full_cell) > 0.25 * full_cell  # discriminates the dx/2 extent
    assert abs(math.fsum(G * w) / (PHI * -math.expm1(-tau[-1])) - 1.0) <= gt.rel_limit
    G2, w2 = (np.asarray(v) for v in adapter.beer_lambert_generation(GEOMETRY_LEGACY_2D, x, a_node, PHI))
    assert w2[0] == 0.5 * h[0] and w2[-1] == 0.5 * h[-1]
    assert np.array_equal(G2[1:-1], G[1:-1]) and G2[0] == 2.0 * G[0] and G2[-1] == 2.0 * G[-1]


def test_candidate_incidence_direction_is_mirror_symmetric():
    g = gate("P0407-BL-DIR")
    x = interface_node_grid()
    owner = right_owner_layer_index(x, TH)
    a_node = np.array(ALPHA)[owner]
    G_right, _ = adapter.beer_lambert_generation(GEOMETRY_LEGACY_1D, x, a_node, PHI, incidence="right")
    xm = (x[-1] - x)[::-1]
    G_mirror, _ = adapter.beer_lambert_generation(GEOMETRY_LEGACY_1D, xm, a_node[::-1], PHI, incidence="left")
    G_right, G_mirror = np.asarray(G_right), np.asarray(G_mirror)[::-1]
    assert np.max(np.abs(G_right - G_mirror)) <= g.rel_limit * np.max(G_mirror)


# ---------------------------------------------------------------- spectral units


def test_photon_flux_conversion_is_exact_si():
    g = gate("P0407-UNITS")
    wl_nm = np.array([300.0, 500.0, 800.0])
    phi = photon_flux_per_m_from_irradiance_per_nm(np.ones(3), wl_nm)
    expected = 1e9 * wl_nm * 1e-9 / (H_PLANCK * C_LIGHT)
    assert np.all(np.abs(phi / expected - 1.0) <= g.rel_limit)


def test_legacy_am15g_resource_is_per_metre_photon_flux():
    """The shipped AM1.5G table is photons m^-2 s^-1 per METRE of wavelength.

    Converting back at 500 nm must give the ASTM G-173 global-tilt spectral
    irradiance (about 1.53 W m^-2 nm^-1); a per-nm table would be 1e9 off.
    """

    from perovskite_sim.data import load_am15g

    wl, flux = load_am15g(np.array([500.0]))
    irradiance = flux[0] * H_PLANCK * C_LIGHT / (500e-9) * 1e-9
    assert 1.45 <= irradiance <= 1.61

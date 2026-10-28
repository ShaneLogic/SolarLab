"""V1 P04-03 acceptance on a test-local DC scaffold (P04-A, DC only).

* same-temperature thermal equilibrium has zero net flux (exactly with the
  QFL step, within an a-priori rounding bound from densities, and the M-class
  two-term currents within theirs);
* a DOS-ratio change leaves a spurious ``kT ln(N ratio)`` quasi-Fermi step only
  when the DOS terms are left out of the transport potentials;
* illumination at zero bias gives a nonzero, conserved short-circuit current
  bounded by the generation, and the dark equilibrium carries none;
* the contact channels reach their limits in the assembled device: fixed
  reservoirs pin the boundary, Robin approaches them as ``1/S``, a blocked
  channel exchanges exactly nothing and blocking only the minority carrier
  cannot lose current (the FD non-degenerate limit and the Schottky/metal
  reservoirs are M-class tests elsewhere).

The Newton in ``p0403_dc_scaffold`` is test scaffolding, not the P04-04 solver.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import p0403_dc_scaffold as scaffold
from solarlab.physics.contacts import ContactEndpoint, assess_contact_thermodynamics
from solarlab.physics.transport import legacy_carrier_face_currents
from solarlab.physics.transport.electrochemical_flux import (
    carrier_particle_fluxes, electrochemical_step_from_densities,
)

EPS = float(np.finfo(float).eps)
Q = scaffold.Q


@pytest.fixture(scope="module")
def device():
    return scaffold.build_device()


@pytest.fixture(scope="module")
def equilibrium(device):
    return scaffold.equilibrium_state(device)


@pytest.fixture(scope="module")
def illuminated(device):
    return scaffold.illuminated_state(device)


def one_sided_scale(flux, density):
    """``k (B(D) c_L + B(-D) c_R)`` per face, from the flux derivatives."""
    return (np.abs(flux.left_density_derivative) * density[:-1]
            + np.abs(flux.right_density_derivative) * density[1:])


def drive_rounding_bound(device, state, flux, density):
    """A-priori absolute rounding of the electrochemical step on each face.

    The equilibrium densities are ``N exp((E_F + phi + chi [+ Eg])/V_T)``, so ``ln c``
    carries ``eps`` times those reduced energies; the transport step adds the
    same for ``(phi + chi_t [+ Eg_t])/V_T``.
    """
    energies = (abs(device.fermi_level_eV) + np.abs(state.phi)
                + np.abs(device.bands.physical_electron_affinity_eV)
                + np.abs(device.bands.physical_band_gap_eV)
                + np.abs(device.bands.transport_band_gap_eV)) / device.V_T
    per_node = energies + np.abs(np.log(density))
    return 16.0 * EPS * (1.0 + per_node[:-1] + per_node[1:] + np.abs(flux.energy_step))


def test_thermal_equilibrium_has_zero_net_flux(device, equilibrium):
    n, p, phi = equilibrium.n, equilibrium.p, equilibrium.phi
    # The equilibrium reservoirs and the semiconductor contact potential agree.
    assert n[0] == pytest.approx(device.n_eq[0], rel=1e-13)
    assert p[-1] == pytest.approx(device.p_eq[1], rel=1e-13)
    fluxes = carrier_particle_fluxes(device.x, phi, n, p, device.D_n_face, device.D_p_face,
                                     device.V_T, device.bands)
    faces = device.size - 1
    # 1. A common Fermi level is an exactly zero electrochemical step: zero flux.
    exact = carrier_particle_fluxes(device.x, phi, n, p, device.D_n_face, device.D_p_face,
                                    device.V_T, device.bands,
                                    electron_electrochemical_step=np.zeros(faces),
                                    hole_electrochemical_step=np.zeros(faces))
    assert np.all(exact.electron.flux == 0.0) and np.all(exact.hole.flux == 0.0)
    # 2. From the rounded densities: |F| <= |m| k B c_upwind, with |m| bounded a priori.
    for flux, density in ((fluxes.electron, n), (fluxes.hole, p)):
        scale = one_sided_scale(flux, density)
        assert np.all(np.abs(flux.flux) <= drive_rounding_bound(device, equilibrium, flux,
                                                                density) * scale)
        assert np.max(np.abs(flux.flux) / scale) < 1e-12
    # 3. The M-class two-term currents within the same bound (two terms, one more ulp each).
    J_n, J_p = legacy_carrier_face_currents(device.x, phi, n, p, device.D_n_face,
                                            device.D_p_face, device.V_T, device.bands)
    for current, flux, density in ((J_n, fluxes.electron, n), (J_p, fluxes.hole, p)):
        scale = Q * one_sided_scale(flux, density)
        bound = (drive_rounding_bound(device, equilibrium, flux, density) + 4.0 * EPS) * scale
        assert np.all(np.abs(current) <= bound)
    # 4. The dark DC residual vanishes and the contacts certify (5 meV gate).
    residual, _ = scaffold.residual_and_jacobian(device, equilibrium, V_app=0.0,
                                                 generation_scale=0.0)
    assessment = assess_contact_thermodynamics(
        built_in_potential_mode="semiconductor_work_function",
        left=ContactEndpoint(n[0], p[0], device.Nc[0], device.Nv[0],
                             device.bands.physical_electron_affinity_eV[0],
                             device.bands.physical_band_gap_eV[0]),
        right=ContactEndpoint(n[-1], p[-1], device.Nc[-1], device.Nv[-1],
                              device.bands.physical_electron_affinity_eV[-1],
                              device.bands.physical_band_gap_eV[-1]),
        V_bi_bc=device.V_bi_bc, thermal_voltage_V=device.V_T, has_band_reference=True,
        semiconductor_potential_V=device.V_bi_bc)
    assert assessment.certified and assessment.fermi_level_span_eV < 1e-12
    current_scale = np.max(Q * one_sided_scale(fluxes.electron, n)
                           + Q * one_sided_scale(fluxes.hole, p))
    assert np.max(np.abs(scaffold.terminal_current_density(device, equilibrium))) < (
        1e-12 * current_scale)
    assert np.max(np.abs(residual[device.size:])) < 1e-12 * current_scale / Q


@pytest.mark.parametrize("ratio", [1.0, 3.0, 10.0, 100.0])
def test_dos_ratio_fold_removes_the_spurious_quasi_fermi_step(ratio):
    layers = list(scaffold.PIN_LAYERS)
    absorber = layers[1]
    layers[0] = scaffold.ScaffoldLayer(**{**layers[0].__dict__,
                                          "Nc_m3": absorber.Nc_m3 * ratio,
                                          "Nv_m3": absorber.Nv_m3 * ratio ** 0.5})
    folded = scaffold.build_device(layers, fold_dos=True)
    unfolded = scaffold.build_device(layers, fold_dos=False)
    state = scaffold.equilibrium_state(folded)
    n, p, phi = state.n, state.p, state.phi
    shared = int(np.argmin(np.abs(folded.x - layers[0].thickness_m)))
    hetero = shared - 1          # face (shared-1, shared): HTL node -> absorber-owned node
    for device, folded_view in ((folded, True), (unfolded, False)):
        fluxes = carrier_particle_fluxes(device.x, phi, n, p, device.D_n_face,
                                         device.D_p_face, device.V_T, device.bands)
        m_n = electrochemical_step_from_densities(n[:-1], n[1:], fluxes.electron.energy_step)
        m_p = electrochemical_step_from_densities(p[:-1], p[1:], fluxes.hole.energy_step)
        bound_n = drive_rounding_bound(device, state, fluxes.electron, n)
        bound_p = drive_rounding_bound(device, state, fluxes.hole, p)
        if folded_view:
            assert np.all(np.abs(m_n) <= bound_n) and np.all(np.abs(m_p) <= bound_p)
            continue
        # Bare band edges: a spurious step m = ln(N_R/N_L) on every DOS-contrast face,
        # i.e. a quasi-Fermi step V_T ln(ratio), and a flux down that step.
        np.testing.assert_allclose(m_n, np.log(device.Nc[1:] / device.Nc[:-1]), rtol=0,
                                   atol=float(np.max(bound_n)))
        np.testing.assert_allclose(m_p, np.log(device.Nv[1:] / device.Nv[:-1]), rtol=0,
                                   atol=float(np.max(bound_p)))
        assert m_n[hetero] == pytest.approx(-math.log(ratio), abs=1e-11)
        assert m_p[hetero] == pytest.approx(-0.5 * math.log(ratio), abs=1e-11)
        if ratio != 1.0:
            scale = one_sided_scale(fluxes.electron, n)[hetero]
            assert fluxes.electron.flux[hetero] > 1e6 * bound_n[hetero] * scale
    # The physical edges carry a flat electron QFL in both views (band diagrams).
    E_C, _ = folded.bands.physical_band_edges(phi)
    E_Fn = E_C + folded.V_T * np.log(n / folded.Nc)
    assert np.ptp(E_Fn) < 1e-12


def test_illuminated_zero_bias_gives_a_nonzero_short_circuit_current(device, equilibrium,
                                                                     illuminated):
    J = scaffold.terminal_current_density(device, illuminated)
    limit = device.generation_limit_A_m2
    # p contact on the left: electrons leave right, holes left -> J < 0 along +x.
    assert np.all(J < 0.0)
    J_sc = float(np.median(J))
    assert 0.5 * limit < -J_sc <= limit * (1.0 + 1e-12)
    # DC conservation: the total current is the same on every face.
    fluxes = carrier_particle_fluxes(device.x, illuminated.phi, illuminated.n, illuminated.p,
                                     device.D_n_face, device.D_p_face, device.V_T, device.bands)
    scale = Q * np.max(np.abs(fluxes.electron.left_density_derivative) * illuminated.n[:-1]
                       + np.abs(fluxes.hole.left_density_derivative) * illuminated.p[:-1])
    assert np.ptp(J) <= 1e-12 * scale
    # The M-class currents agree on the converged state through the channel signs.
    J_n, J_p = legacy_carrier_face_currents(device.x, illuminated.phi, illuminated.n,
                                            illuminated.p, device.D_n_face, device.D_p_face,
                                            device.V_T, device.bands)
    assert np.max(np.abs((J_n + J_p) - J)) <= 1e-12 * scale
    # Dark at the same bias: no current beyond rounding of the same scale.
    dark, iterations = scaffold.solve_dc(device, equilibrium, generation_scale=0.0)
    assert iterations <= 3
    assert np.max(np.abs(scaffold.terminal_current_density(device, dark))) <= 1e-12 * scale
    assert np.max(np.abs(scaffold.terminal_current_density(device, dark))) < 1e-6 * abs(J_sc)


def solve_with(velocities):
    device = scaffold.build_device(velocities=velocities)
    state = scaffold.illuminated_state(device)
    return device, state, float(np.median(scaffold.terminal_current_density(device, state)))


def current_scale(device, state):
    """``q k (B c_L + B c_R)`` summed over both carriers: the rounding scale of J."""
    fluxes = carrier_particle_fluxes(device.x, state.phi, state.n, state.p, device.D_n_face,
                                     device.D_p_face, device.V_T, device.bands)
    return float(np.max(Q * one_sided_scale(fluxes.electron, state.n)
                        + Q * one_sided_scale(fluxes.hole, state.p)))


def test_contact_channel_limits_in_the_assembled_dc_device(illuminated, device):
    J_fixed = float(np.median(scaffold.terminal_current_density(device, illuminated)))
    channels = ("n_left", "p_left", "n_right", "p_right")
    # Robin -> fixed reservoir: every boundary density approaches its reservoir as 1/S
    # (the contact face carries F = S (c - c_eq) at a fixed supply), and the current
    # approaches the fixed-reservoir current.
    excess = []
    for S in (1e3, 1e4, 1e5, 1e6):
        robin_device, state, J = solve_with({key: S for key in channels})
        left, right = scaffold.contact_currents(robin_device, state)
        tolerance = 1e-12 * current_scale(robin_device, state)
        assert abs(left - J) <= tolerance and abs(right - J) <= tolerance
        assert abs(J - J_fixed) <= 1e-9 * abs(J_fixed)
        excess.append([abs(state.n[0] - robin_device.n_eq[0]),
                       abs(state.p[0] - robin_device.p_eq[0]),
                       abs(state.n[-1] - robin_device.n_eq[1]),
                       abs(state.p[-1] - robin_device.p_eq[1])])
    excess = np.asarray(excess)
    np.testing.assert_allclose(excess[:-1] / excess[1:], 10.0, rtol=0.05)
    # Fixed reservoirs pin the boundary densities exactly (Dirichlet constraint rows).
    assert illuminated.n[0] == pytest.approx(device.n_eq[0], rel=4 * EPS)
    assert illuminated.p[-1] == pytest.approx(device.p_eq[1], rel=4 * EPS)
    # S = 0: the blocked carrier exchanges nothing through its face, exactly, while the
    # other channels stay Robin. Blocking only the minority carrier at each contact
    # (electrons at the hole contact, holes at the electron contact) removes the
    # surface loss, so the collected current cannot drop. (Blocking all four channels
    # leaves no reservoir to fix the DC carrier inventory: that system is singular.)
    selective = {"n_left": 0.0, "p_left": 1e5, "n_right": 1e5, "p_right": 0.0}
    selective_device, state, J_selective = solve_with(selective)
    from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
    blocked_n = boundary_particle_flux(selective_device.channels["n_left"], state.n[0])
    blocked_p = boundary_particle_flux(selective_device.channels["p_right"], state.p[-1])
    assert blocked_n.flux == 0.0 and blocked_p.flux == 0.0
    left, right = scaffold.contact_currents(selective_device, state)
    tolerance = 1e-12 * current_scale(selective_device, state)
    assert abs(left - J_selective) <= tolerance and abs(right - J_selective) <= tolerance
    assert abs(J_selective) >= abs(J_fixed) * (1.0 - 1e-12)

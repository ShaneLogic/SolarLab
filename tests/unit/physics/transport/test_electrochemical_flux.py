"""Unified directed particle flux in stable electrochemical form (N class).

Not bitwise with the historical two-term currents (new evaluation order), so
each property is checked against an independent 60-digit Decimal oracle with
an a-priori rounding bound, and against the M-class currents through the
explicit output-channel adapters ``J_n = -q F_n`` and ``J_p = +q F_p``.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
import math

import numpy as np
import pytest

from solarlab.contracts import PARTICLE, VOLT, VOLUME, FloatArray, Layout, StateView, Support
from solarlab.contracts import VariableSpec
from solarlab.discretization import legacy_sg_current_n, legacy_sg_current_p
from solarlab.discretization.particle_flux import sg_particle_flux
from solarlab.physics.statistics import BandStructure
from solarlab.physics.transport import legacy_carrier_face_currents
from solarlab.physics.transport.electrochemical_flux import (
    carrier_particle_fluxes, electrochemical_step_from_densities,
    electron_current_from_particle_flux, hole_current_from_particle_flux,
    stable_sg_particle_flux,
)

getcontext().prec = 60
EPS = float(np.finfo(float).eps)
Q = 1.602176634e-19


def dec(value):
    return Decimal(float(value))


def bernoulli_dec(x):
    x = dec(x)
    if x == 0:
        return Decimal(1)
    return x / (x.exp() - 1)


def bernoulli_derivative_dec(x):
    x = dec(x)
    if abs(x) < Decimal("1e-20"):
        return Decimal(-1) / 2
    e = x.exp()
    return (e - 1 - x * e) / (e - 1) ** 2


def flux_dec(c_left, c_right, step, k):
    return dec(k) * (bernoulli_dec(step) * dec(c_left) - bernoulli_dec(-step) * dec(c_right))


def one_sided(c_left, c_right, step, k):
    return float(dec(k) * (bernoulli_dec(step) * dec(c_left) + bernoulli_dec(-step) * dec(c_right)))


def random_faces(rng, count, step_scale=40.0):
    c_left = 10.0 ** rng.uniform(-10.0, 27.0, size=count)
    c_right = 10.0 ** rng.uniform(-10.0, 27.0, size=count)
    step = rng.normal(scale=step_scale, size=count)
    k = 10.0 ** rng.uniform(-3.0, 5.0, size=count)
    return c_left, c_right, step, k


def test_stable_flux_matches_the_decimal_oracle_within_its_rounding_bound():
    rng = np.random.default_rng(31)
    c_left, c_right, step, k = random_faces(rng, 600)
    step[:20] = rng.uniform(-700.0, 700.0, size=20)
    result = stable_sg_particle_flux(c_left, c_right, step, k)
    for i in range(c_left.size):
        reference = flux_dec(c_left[i], c_right[i], step[i], k[i])
        # The step rebuilt from binary64 densities carries an absolute rounding of
        # at most ~eps (|ln c_L| + |ln c_R| + |D|); the rest is a few product roundings.
        drive_error = abs(math.log(c_left[i])) + abs(math.log(c_right[i])) + abs(step[i])
        bound = 16.0 * EPS * (1.0 + drive_error) * one_sided(
            c_left[i], c_right[i], step[i], k[i])
        assert abs(Decimal(float(result.flux[i])) - reference) <= Decimal(bound) + Decimal(1e-300)


def exact_step(c_left, c_right, step):
    return float((dec(c_right) / dec(c_left)).ln() + dec(step))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("coefficient", [1.0, 1e200])
def test_extreme_barrier_retains_representable_flux_and_derivatives(reverse, coefficient):
    # N03-F01: B(800) itself underflows, while B(800)*1e50 does not.
    left, right, step = (1e-300, 1e50, -800.0)
    if reverse:
        left, right, step = right, left, -step
    result = stable_sg_particle_flux(np.array([left]), np.array([right]), np.array([step]),
                                     coefficient, electrochemical_step=np.array([exact_step(left, right, step)]))
    reference = flux_dec(left, right, step, coefficient)
    derivative = dec(coefficient) * (bernoulli_derivative_dec(step) * dec(left)
                                    + bernoulli_derivative_dec(-step) * dec(right))
    assert result.flux[0] != 0.0
    assert result.flux[0] == pytest.approx(float(reference), rel=4e-13, abs=0.0)
    assert result.energy_step_derivative[0] == pytest.approx(float(derivative), rel=4e-13, abs=0.0)


def test_extreme_scalar_flux_and_exact_equilibrium():
    m = exact_step(1e-300, 1e50, -800.0)
    result = stable_sg_particle_flux(1e-300, 1e50, -800.0, 1.0, electrochemical_step=m)
    assert result.flux == pytest.approx(float(flux_dec(1e-300, 1e50, -800.0, 1.0)), rel=4e-13, abs=0.0)
    zero = stable_sg_particle_flux(1e300, 1e300, -800.0, 1e300, electrochemical_step=0.0)
    assert zero.flux == 0.0 and zero.coefficient_derivative == 0.0


def test_supplied_electrochemical_step_gives_relative_accuracy_even_for_weak_drives():
    rng = np.random.default_rng(37)
    stable_errors, two_term_errors = [], []
    for drive in (1e-2, 1e-6, 1e-10, 1e-14):
        for _ in range(10):
            c_left = float(10.0 ** rng.uniform(10.0, 26.0))
            step = float(rng.normal(scale=10.0))
            c_right = float(dec(c_left) * (dec(drive) - dec(step)).exp())
            m = exact_step(c_left, c_right, step)
            reference = flux_dec(c_left, c_right, step, 1.0)
            stable = stable_sg_particle_flux(np.array([c_left]), np.array([c_right]),
                                             np.array([step]), 1.0,
                                             electrochemical_step=np.array([m])).flux[0]
            two_term = sg_particle_flux(np.array([c_left]), np.array([c_right]),
                                        np.array([step]), 1.0).flux[0]
            stable_errors.append(abs(float((dec(stable) - reference) / reference)))
            two_term_errors.append(abs(float((dec(two_term) - reference) / reference)))
    # Rounding m to binary64 costs one relative ulp of m; the products a few more.
    assert max(stable_errors) <= 8.0 * EPS
    # The two-term difference cancels: its relative error grows like eps/|m|.
    assert max(two_term_errors) > 1e-6


def test_zero_electrochemical_step_is_exactly_zero_flux():
    rng = np.random.default_rng(41)
    c_left, c_right, step, k = random_faces(rng, 200)
    zero = stable_sg_particle_flux(c_left, c_right, step, k,
                                   electrochemical_step=np.zeros_like(step))
    assert np.all(zero.flux == 0.0) and not np.any(np.signbit(zero.flux))
    assert np.all(stable_sg_particle_flux(c_left, c_left, np.zeros_like(step), k).flux == 0.0)
    negative_zero = stable_sg_particle_flux(c_left, c_right, step, k,
                                            electrochemical_step=np.full_like(step, -0.0))
    assert np.all(negative_zero.flux == 0.0) and not np.any(np.signbit(negative_zero.flux))


def test_reversing_a_face_negates_the_flux_exactly():
    rng = np.random.default_rng(43)
    c_left, c_right, step, k = random_faces(rng, 300)
    m = electrochemical_step_from_densities(c_left, c_right, step)
    forward = stable_sg_particle_flux(c_left, c_right, step, k, electrochemical_step=m)
    backward = stable_sg_particle_flux(c_right, c_left, -step, k, electrochemical_step=-m)
    assert np.array_equal(forward.flux, -backward.flux)


def test_analytic_derivatives_match_the_decimal_oracle():
    rng = np.random.default_rng(47)
    c_left, c_right, step, k = random_faces(rng, 300, step_scale=8.0)
    result = stable_sg_particle_flux(c_left, c_right, step, k)
    for i in range(c_left.size):
        d_left = dec(k[i]) * bernoulli_dec(step[i])
        d_right = -dec(k[i]) * bernoulli_dec(-step[i])
        d_step = dec(k[i]) * (bernoulli_derivative_dec(step[i]) * dec(c_left[i])
                              + bernoulli_derivative_dec(-step[i]) * dec(c_right[i]))
        for value, reference in ((result.left_density_derivative[i], d_left),
                                 (result.right_density_derivative[i], d_right),
                                 (result.energy_step_derivative[i], d_step)):
            assert abs(dec(value) - reference) <= abs(reference) * Decimal(16.0 * EPS)
        assert result.coefficient_derivative[i] * k[i] == pytest.approx(result.flux[i],
                                                                        rel=4 * EPS, abs=0.0)


def grid_state(rng, size=24):
    x = np.cumsum(np.concatenate(([0.0], rng.uniform(1e-9, 2e-8, size=size - 1))))
    phi = np.linspace(0.0, 1.1, size) + rng.normal(scale=0.03, size=size)
    n = 10.0 ** rng.uniform(5.0, 25.0, size=size)
    p = 10.0 ** rng.uniform(5.0, 25.0, size=size)
    chi = np.where(np.arange(size) < size // 2, 3.9, 4.1)
    gap = np.where(np.arange(size) < size // 2, 1.6, 2.0)
    bands = BandStructure(chi, gap, chi + 0.02, gap - 0.05, dos_folded=True, reference_layer=0)
    D_n = rng.uniform(1e-6, 1e-4, size=size - 1)
    D_p = rng.uniform(1e-6, 1e-4, size=size - 1)
    return x, phi, n, p, bands, D_n, D_p


def test_potential_derivatives_follow_each_carriers_energy_sign():
    rng = np.random.default_rng(53)
    x, phi, n, p, bands, D_n, D_p = grid_state(rng)
    V_T = 0.025852
    fluxes = carrier_particle_fluxes(x, phi, n, p, D_n, D_p, V_T, bands)
    for carrier, flux in (("electron", fluxes.electron), ("hole", fluxes.hole)):
        sign = -1.0 if carrier == "electron" else 1.0
        np.testing.assert_array_equal(flux.right_potential_derivative,
                                      sign / V_T * stable_sg_particle_flux(
                                          (n if carrier == "electron" else p)[:-1],
                                          (n if carrier == "electron" else p)[1:],
                                          flux.energy_step, flux.coefficient,
                                      ).energy_step_derivative)
        np.testing.assert_array_equal(flux.left_potential_derivative,
                                      -flux.right_potential_derivative)
    # Central differences in phi at one node confirm the assembled sign.
    node, h = 7, 1e-7
    for carrier in ("electron", "hole"):
        bumped = []
        for delta in (h, -h):
            shifted = phi.copy()
            shifted[node] += delta
            result = carrier_particle_fluxes(x, shifted, n, p, D_n, D_p, V_T, bands)
            bumped.append(getattr(result, carrier).flux)
        numeric = (bumped[0] - bumped[1]) / (2 * h)
        flux = getattr(fluxes, carrier)
        assert numeric[node] == pytest.approx(flux.left_potential_derivative[node], rel=1e-6)
        assert numeric[node - 1] == pytest.approx(flux.right_potential_derivative[node - 1],
                                                  rel=1e-6)


def test_default_step_equals_the_state_view_electrochemical_difference():
    rng = np.random.default_rng(59)
    x, phi, n, p, bands, D_n, D_p = grid_state(rng)
    V_T = 0.025852
    psi_n, psi_p = bands.transport_potentials(phi)
    size = x.size
    layout = Layout((Support("nodes", "cell", (size,)),), (
        VariableSpec("n", "carrier", "nodes", (size,), PARTICLE / VOLUME, lower=0),
        VariableSpec("p", "carrier", "nodes", (size,), PARTICLE / VOLUME, lower=0),
        VariableSpec("psi_n", "transport", "nodes", (size,), VOLT),
        VariableSpec("psi_p", "transport", "nodes", (size,), VOLT)), ())
    state = StateView(layout, [("n", FloatArray(n)), ("p", FloatArray(p)),
                               ("psi_n", FloatArray(psi_n)), ("psi_p", FloatArray(psi_p))])
    pairs = np.column_stack((np.arange(size - 1), np.arange(1, size)))
    fluxes = carrier_particle_fluxes(x, phi, n, p, D_n, D_p, V_T, bands)
    m_n = state.electrochemical_difference("n", "psi_n", pairs, V_T, potential_sign=-1)
    m_p = state.electrochemical_difference("p", "psi_p", pairs, V_T, potential_sign=1)
    assert m_n.tobytes() == electrochemical_step_from_densities(
        n[:-1], n[1:], fluxes.electron.energy_step).tobytes()
    assert m_p.tobytes() == electrochemical_step_from_densities(
        p[:-1], p[1:], fluxes.hole.energy_step).tobytes()
    supplied = carrier_particle_fluxes(x, phi, n, p, D_n, D_p, V_T, bands,
                                       electron_electrochemical_step=m_n,
                                       hole_electrochemical_step=m_p)
    assert supplied.electron.flux.tobytes() == fluxes.electron.flux.tobytes()
    assert supplied.hole.flux.tobytes() == fluxes.hole.flux.tobytes()


def test_output_channels_keep_their_own_signs_against_the_historical_currents():
    rng = np.random.default_rng(61)
    V_T = 0.025852
    for _ in range(30):
        x, phi, n, p, bands, D_n, D_p = grid_state(rng)
        fluxes = carrier_particle_fluxes(x, phi, n, p, D_n, D_p, V_T, bands)
        J_n, J_p = legacy_carrier_face_currents(x, phi, n, p, D_n, D_p, V_T, bands)
        dx = np.diff(x)
        psi_n, psi_p = bands.transport_potentials(phi)
        for current, legacy, step, density, k in (
            (fluxes.electron_current(), J_n, fluxes.electron.energy_step, n, D_n / dx),
            (fluxes.hole_current(), J_p, fluxes.hole.energy_step, p, D_p / dx),
        ):
            scale = np.array([Q * one_sided(density[i], density[i + 1], step[i], k[i])
                              for i in range(step.size)])
            log_ratio = np.abs(np.log(density[1:]) - np.log(density[:-1]))
            bound = 64.0 * EPS * (1.0 + log_ratio + np.abs(step)) * scale
            assert np.all(np.abs(current - legacy) <= bound)
        assert np.array_equal(electron_current_from_particle_flux(fluxes.electron.flux),
                              -Q * fluxes.electron.flux)
        assert np.array_equal(hole_current_from_particle_flux(fluxes.hole.flux),
                              Q * fluxes.hole.flux)
        assert np.array_equal(legacy_sg_current_n(psi_n, n, dx, D_n, V_T), J_n)
        assert np.array_equal(legacy_sg_current_p(psi_p, p, dx, D_p, V_T), J_p)


@pytest.mark.parametrize("bad", [
    dict(c_left=[0.0]), dict(c_right=[-1.0]), dict(step=[math.nan]), dict(k=[-1.0]),
    dict(k=[math.nan]), dict(m=[math.inf]),
])
def test_invalid_inputs_fail_closed(bad):
    args = dict(c_left=[1e20], c_right=[1e21], step=[0.5], k=[10.0], m=None)
    args.update(bad)
    with pytest.raises(ValueError):
        stable_sg_particle_flux(np.array(args["c_left"]), np.array(args["c_right"]),
                                np.array(args["step"]), np.array(args["k"]),
                                electrochemical_step=None if args["m"] is None
                                else np.array(args["m"]))

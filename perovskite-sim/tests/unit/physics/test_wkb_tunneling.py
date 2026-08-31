"""D8-E0 WKB transmission primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.physics.wkb_tunneling import (
    MINIMUM_MEANINGFUL_ACTION,
    WKBTunnellingError,
    decay_constant_per_m,
    reciprocal_net_flux,
    triangular_barrier_action,
    wkb_action,
    wkb_transmission,
    wkb_validity,
)


HEIGHT_EV = 0.30
WIDTH_M = 5.0e-9
MASS_REL = 0.2


def _triangular(
    width_m: float = WIDTH_M, height_eV: float = HEIGHT_EV, points: int = 801
):
    x = np.linspace(0.0, width_m, points)
    return x, height_eV * (1.0 - x / width_m)


def test_action_converges_to_the_analytic_triangular_barrier():
    """The quadrature is checked against a closed form before any physics."""
    exact = triangular_barrier_action(HEIGHT_EV, WIDTH_M, MASS_REL)
    errors = []
    for points in (201, 401, 801, 1601):
        x, barrier = _triangular(points=points)
        errors.append(abs(wkb_action(x, barrier, 0.0, MASS_REL) - exact) / exact)

    assert errors[-1] < 1.0e-5
    orders = [math.log2(coarse / fine) for coarse, fine in zip(errors[:-1], errors[1:])]
    # A square-root turning point gives order 3/2, not 2. Asserting the
    # measured order rather than a tolerance is what would catch an
    # integrator that silently changed behaviour near the turning point.
    assert all(1.35 < order < 1.65 for order in orders), orders


def test_transmission_decays_monotonically_with_width_height_and_mass():
    widths = [2.0e-9, 4.0e-9, 6.0e-9, 8.0e-9]
    by_width = [
        wkb_transmission(*_triangular(width_m=w), 0.0, MASS_REL) for w in widths
    ]
    heights = [0.10, 0.20, 0.30, 0.40]
    by_height = [
        wkb_transmission(*_triangular(height_eV=h), 0.0, MASS_REL) for h in heights
    ]
    x, barrier = _triangular()
    by_mass = [wkb_transmission(x, barrier, 0.0, m) for m in (0.02, 0.1, 0.5, 2.0)]

    for label, values in (
        ("width", by_width),
        ("height", by_height),
        ("mass", by_mass),
    ):
        assert all(a > b for a, b in zip(values, values[1:])), (label, values)
        assert all(0.0 < value <= 1.0 for value in values)


def test_effective_mass_limit_matches_the_closed_form_exponent():
    """Halving the mass must scale the action by exactly sqrt(2)."""
    x, barrier = _triangular()
    heavy = wkb_action(x, barrier, 0.0, 0.4)
    light = wkb_action(x, barrier, 0.0, 0.2)

    assert heavy / light == pytest.approx(math.sqrt(2.0), rel=1.0e-12)


def test_energy_dependence_follows_the_three_halves_power():
    """S(E) ∝ (U0 - E)^{3/2} / U0 for a linear barrier."""
    for energy in (0.0, 0.05, 0.1, 0.2):
        analytic = triangular_barrier_action(HEIGHT_EV, WIDTH_M, MASS_REL, energy)
        x, barrier = _triangular(points=3201)
        assert wkb_action(x, barrier, energy, MASS_REL) == pytest.approx(
            analytic, rel=5.0e-5
        )


def test_shallow_and_thin_barriers_are_reported_as_not_meaningful():
    """WKB says nothing useful about a barrier the carrier barely notices."""
    x = np.linspace(0.0, 1.0e-11, 201)
    barrier = 0.001 * (1.0 - x / x[-1])
    validity = wkb_validity(x, barrier, 0.0, MASS_REL)

    assert validity.action < MINIMUM_MEANINGFUL_ACTION
    assert validity.meaningful_barrier is False
    assert validity.valid is False

    real = wkb_validity(*_triangular(), 0.0, MASS_REL)
    assert real.action > 1.0
    assert real.valid is True


def test_validity_does_not_gate_on_the_turning_point_breakdown():
    """A smooth barrier always fails the local-wavelength test at its turning
    point; that is the textbook Airy region, not a reason to reject it."""
    validity = wkb_validity(*_triangular(), 0.0, MASS_REL)

    assert validity.slowly_varying is False
    assert validity.valid is True  # gated on the action alone


def test_decay_constant_is_zero_outside_the_forbidden_region():
    x, barrier = _triangular(points=101)
    kappa = decay_constant_per_m(barrier, 0.5 * HEIGHT_EV, MASS_REL)

    allowed = barrier <= 0.5 * HEIGHT_EV
    assert np.all(kappa[allowed] == 0.0)
    assert np.all(kappa[~allowed] > 0.0)


def test_equal_occupations_give_an_exactly_zero_net_flux():
    """Reciprocity is structural: one transmission drives both directions."""
    x, barrier = _triangular(points=401)
    energies = np.linspace(0.0, HEIGHT_EV, 48)
    transmission = np.array(
        [wkb_transmission(x, barrier, e, MASS_REL) for e in energies]
    )
    occupation = 1.0 / (1.0 + np.exp((energies - 0.15) / 0.025852))

    flux = reciprocal_net_flux(energies, transmission, occupation, occupation, 1.0e20)

    assert flux.net_flux_m2_s == 0.0
    assert flux.forward_flux_m2_s == flux.reverse_flux_m2_s
    assert flux.forward_flux_m2_s > 0.0


def test_net_flux_reverses_sign_with_the_occupation_difference():
    x, barrier = _triangular(points=401)
    energies = np.linspace(0.0, HEIGHT_EV, 32)
    transmission = np.array(
        [wkb_transmission(x, barrier, e, MASS_REL) for e in energies]
    )
    hot = 1.0 / (1.0 + np.exp((energies - 0.20) / 0.025852))
    cold = 1.0 / (1.0 + np.exp((energies - 0.10) / 0.025852))

    forward = reciprocal_net_flux(energies, transmission, hot, cold, 1.0e20)
    reverse = reciprocal_net_flux(energies, transmission, cold, hot, 1.0e20)

    assert forward.net_flux_m2_s > 0.0
    assert reverse.net_flux_m2_s == pytest.approx(-forward.net_flux_m2_s, rel=1e-15)


def test_invalid_inputs_fail_closed():
    x, barrier = _triangular(points=51)

    with pytest.raises(WKBTunnellingError, match="strictly increase"):
        wkb_action(x[::-1], barrier, 0.0, MASS_REL)
    with pytest.raises(WKBTunnellingError, match="finite and positive"):
        wkb_action(x, barrier, 0.0, 0.0)
    with pytest.raises(WKBTunnellingError, match="match positions_m"):
        wkb_action(x, barrier[:-1], 0.0, MASS_REL)
    with pytest.raises(WKBTunnellingError, match="below the barrier"):
        triangular_barrier_action(HEIGHT_EV, WIDTH_M, MASS_REL, HEIGHT_EV)
    energies = np.linspace(0.0, 0.3, 8)
    with pytest.raises(WKBTunnellingError, match=r"transmission must lie"):
        reciprocal_net_flux(
            energies,
            np.full(8, 1.5),
            np.zeros(8),
            np.zeros(8),
            1.0,
        )

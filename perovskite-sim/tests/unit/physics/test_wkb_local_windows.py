"""D8-E1 local barrier extraction and the two-band (Kane) exponent.

A device grid holds several barriers at once — a heterojunction spike, the
band bending at each contact — so a channel bound to one interface must
integrate only its own. These tests pin that separation, and pin the Zener
exponent against its closed form rather than against a tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.physics.tunneling_channels import local_barrier_window
from perovskite_sim.physics.wkb_tunneling import (
    WKBTunnellingError,
    forbidden_run,
    kane_uniform_field_action,
    two_band_action,
    two_band_decay_constant_per_m,
    two_band_transmission,
    two_band_validity,
    wkb_action,
    windowed_wkb_action,
    windowed_wkb_transmission,
)


MASS_REL = 0.2


def _two_barriers(points: int = 401):
    """Two separated Gaussian barriers on one grid, plus their peak faces."""
    x = np.linspace(0.0, 40.0e-9, points)
    first = 0.30 * np.exp(-(((x - 10.0e-9) / 1.5e-9) ** 2))
    second = 0.45 * np.exp(-(((x - 30.0e-9) / 1.5e-9) ** 2))
    barrier = first + second
    return x, barrier, int(np.argmax(first)), int(np.argmax(second))


def test_forbidden_run_isolates_the_barrier_containing_the_face():
    """Two barriers on one grid must never be merged into one path."""
    x, barrier, first_face, second_face = _two_barriers()
    energy = 0.20

    low_a, high_a = forbidden_run(barrier, energy, first_face)
    low_b, high_b = forbidden_run(barrier, energy, second_face)

    assert high_a < low_b, "the two runs must not overlap"
    assert low_a <= first_face <= high_a
    assert low_b <= second_face <= high_b
    # The whole-grid action sums both barriers; each windowed one must carry
    # only its own. The sum is a little SMALLER than the whole rather than
    # equal to it: the full integral also picks up the trapezoid segments that
    # bridge each run's last forbidden node to the first allowed node, where
    # kappa is already zero. That difference is the turning-point tail, not a
    # third barrier, so it is bounded rather than pinned.
    whole = wkb_action(x, barrier, energy, MASS_REL)
    part_a = windowed_wkb_action(x, barrier, energy, MASS_REL, first_face)
    part_b = windowed_wkb_action(x, barrier, energy, MASS_REL, second_face)
    assert part_a < whole and part_b < whole
    assert part_a + part_b <= whole
    assert part_a + part_b == pytest.approx(whole, rel=0.05)


def test_a_face_outside_every_forbidden_region_carries_no_action():
    """Not blocked is a real answer, distinct from an infinitely thin barrier."""
    x, barrier, first_face, _ = _two_barriers()
    energy = float(np.max(barrier)) + 0.1

    assert forbidden_run(barrier, energy, first_face) is None
    assert windowed_wkb_action(x, barrier, energy, MASS_REL, first_face) == 0.0
    assert windowed_wkb_transmission(x, barrier, energy, MASS_REL, first_face) == 1.0


def test_transmission_differs_between_two_barriers_on_the_same_grid():
    """The taller barrier must be the less transparent one at the same energy."""
    x, barrier, first_face, second_face = _two_barriers()

    low = windowed_wkb_transmission(x, barrier, 0.20, MASS_REL, first_face)
    high = windowed_wkb_transmission(x, barrier, 0.20, MASS_REL, second_face)

    assert 0.0 < high < low < 1.0


def test_local_window_ignores_the_device_wide_tilt():
    """A junction spike is not the band bending between the contacts."""
    x = np.linspace(0.0, 40.0e-9, 401)
    tilt = -4.0 - 0.5 * x / x[-1]
    spike_face = 200
    barrier = tilt + 0.36 * np.exp(-(((x - x[spike_face]) / 1.5e-9) ** 2))

    peak, base = local_barrier_window(barrier, spike_face)

    # The device ends are -4.0 and -4.5; neither may set the window.
    assert peak == pytest.approx(barrier[spike_face], rel=1.0e-9)
    assert -4.5 < base < peak
    assert peak - base < 0.36


def test_a_contact_barrier_needs_the_one_sided_window():
    """A Schottky barrier peaks AT the metal, so it has one minimum, not two."""
    x = np.linspace(0.0, 20.0e-9, 201)
    barrier = 0.5 * np.exp(-x / 4.0e-9)

    peak, base = local_barrier_window(barrier, 0, one_sided=True)
    assert peak == pytest.approx(0.5, rel=1.0e-12)
    assert base < peak

    # The two-sided rule takes the HIGHER of the two bounding minima, which
    # for an endpoint peak is the peak itself — a degenerate, empty window.
    # That is why the caller declares the shape instead of it being inferred.
    degenerate_peak, degenerate_base = local_barrier_window(barrier, 0)
    assert degenerate_peak == degenerate_base


def test_anchor_face_outside_the_transport_faces_fails_closed():
    x, barrier, _, _ = _two_barriers(points=51)

    with pytest.raises(WKBTunnellingError, match="anchor_face"):
        forbidden_run(barrier, 0.2, barrier.size - 1)
    with pytest.raises(WKBTunnellingError, match="anchor_face"):
        local_barrier_window(barrier, -1)


def _uniform_field(gap_eV: float, field_V_m: float, points: int = 40001):
    """Linearly tilted bands: the case the Kane exponent has a closed form for."""
    span = 3.0 * gap_eV / field_V_m
    x = np.linspace(0.0, span, points)
    conduction = -field_V_m * x
    return x, conduction, conduction - gap_eV, points // 2


def test_two_band_action_matches_the_closed_form_zener_exponent():
    """The Kane prefactor is pinned by the closed form, not by a tolerance.

    Under a uniform field the two-band action integrates exactly to
    ``pi sqrt(2 m_r) Eg^{3/2} / (8 hbar q F)``. A single-band exponent cannot
    reproduce this at any prefactor because its turning points are wrong.
    """
    gap, field, mass = 1.0, 1.0e8, 0.1
    x, conduction, valence, face = _uniform_field(gap, field)
    energy = float(conduction[face] - 0.5 * gap)

    action = two_band_action(x, conduction, valence, energy, mass, face)
    exact = kane_uniform_field_action(gap, field, mass)

    assert action == pytest.approx(exact, rel=1.0e-5)


def test_the_zener_exponent_scales_as_the_three_halves_power_of_the_gap():
    """S ∝ Eg^{3/2} / F is the content of the Kane result."""
    field, mass = 1.0e8, 0.1
    actions = []
    for gap in (0.8, 1.0, 1.2):
        x, conduction, valence, face = _uniform_field(gap, field)
        energy = float(conduction[face] - 0.5 * gap)
        actions.append(two_band_action(x, conduction, valence, energy, mass, face))

    for gap, action in zip((0.8, 1.0, 1.2), actions):
        assert action / gap**1.5 == pytest.approx(actions[1], rel=2.0e-5)


def test_the_two_band_decay_constant_vanishes_at_both_turning_points():
    """This is the property the single-band form cannot have.

    An electron tunnelling across the gap leaves the valence band at one
    turning point and enters the conduction band at the other, so kappa must
    go to zero on *both* bands — not once, on one edge.
    """
    gap, field, mass = 1.0, 1.0e8, 0.1
    x, conduction, valence, face = _uniform_field(gap, field, points=2001)
    energy = float(conduction[face] - 0.5 * gap)

    kappa = two_band_decay_constant_per_m(conduction, valence, energy, mass)

    inside = (conduction > energy) & (valence < energy)
    assert np.all(kappa[~inside] == 0.0)
    assert np.all(kappa[inside] > 0.0)
    # The interior maximum is strictly inside the run, not at either edge.
    interior = np.flatnonzero(inside)
    peak = int(interior[int(np.argmax(kappa[inside]))])
    assert interior[0] < peak < interior[-1]
    assert kappa[interior[0]] < kappa[peak]
    assert kappa[interior[-1]] < kappa[peak]
    del x, face


def test_two_band_transmission_is_bounded_and_falls_with_the_gap():
    field, mass = 1.0e8, 0.1
    values = []
    for gap in (0.6, 0.9, 1.2):
        x, conduction, valence, face = _uniform_field(gap, field, points=8001)
        energy = float(conduction[face] - 0.5 * gap)
        values.append(two_band_transmission(x, conduction, valence, energy, mass, face))

    assert all(0.0 < value < 1.0 for value in values)
    assert values[0] > values[1] > values[2]


def test_two_band_validity_reports_the_two_band_action_not_the_single_band_one():
    """Reporting the single-band diagnostics here would describe a barrier the
    Zener channel never crosses."""
    gap, field, mass = 1.0, 1.0e8, 0.1
    x, conduction, valence, face = _uniform_field(gap, field, points=8001)
    energy = float(conduction[face] - 0.5 * gap)

    validity = two_band_validity(x, conduction, valence, energy, mass, face)

    assert validity.action == pytest.approx(
        two_band_action(x, conduction, valence, energy, mass, face), rel=1.0e-12
    )
    assert validity.transmission == pytest.approx(
        math.exp(-2.0 * validity.action), rel=1.0e-12
    )
    assert validity.meaningful_barrier is True
    # The forbidden width is the gap divided by the field, not the whole grid.
    assert validity.forbidden_width_m == pytest.approx(gap / field, rel=1.0e-3)
    assert validity.forbidden_width_m < float(x[-1] - x[0])


def test_two_band_helpers_fail_closed_on_a_non_positive_gap():
    x = np.linspace(0.0, 1.0e-8, 51)
    conduction = np.zeros_like(x)
    valence = np.zeros_like(x)

    with pytest.raises(WKBTunnellingError, match="gap must be positive"):
        two_band_decay_constant_per_m(conduction, valence, -0.5, 0.1)
    with pytest.raises(WKBTunnellingError, match="match positions_m"):
        two_band_action(x, conduction[:-1], valence[:-1], -0.5, 0.1, 0)

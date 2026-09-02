"""D8-P1: what `anchor_face` actually selects, measured.

D8-E1's first correction is documented as "a channel integrates its own
barrier, not the forbidden set — anchored to its own interface face". The
first half is real and still holds: two separated barriers on one grid are no
longer merged. The second half — that the anchor identifies *a barrier* as an
object — is not what the code does, and this file measures the difference.

D8 defines no barrier identity. What `anchor_face` selects is "whichever
connected forbidden run happens to contain this face at this energy", and the
resulting flux is a smooth function of where the anchor is dropped inside a
basin rather than a property of any interface.

This file deliberately does NOT fix any of it. The gates in the registered
lane `wkb-tunnelling-channel-qf-dc-v1` are anchored to the current behaviour,
and correcting the drive fallback below would move magnitudes in the same way
the level-convention fix does — both belong to the same deferred v2 lane
(see `docs/wkb-tunnelling-family-contract.md`, "Retraction (D8-E2R)"). The
purpose here is to make the limitation undeniable and executable.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from perovskite_sim.models.tunneling_channels import IntrabandTunnellingChannel
from perovskite_sim.physics.tunneling_channels import (
    _turning_point_levels,
    intraband_flux,
    local_barrier_window,
)
from perovskite_sim.physics.wkb_tunneling import (
    forbidden_run,
    windowed_wkb_transmission,
)


THERMAL_VOLTAGE_V = 0.025852
_MASS_REL = 0.2

# Node layout mirrors the registered lane config at its coarsest grid:
# 24 intervals per electrical layer over absorber / spike / ETL, so faces
# 0..23 are the absorber, 24..47 the raised interlayer, 48..71 the ETL, and
# the two heterointerfaces sit at faces 23 and 47.
_INTERFACE_FACE = 23
_SECOND_INTERFACE_FACE = 47


def _synthetic_profile():
    """A tilted p-n band with one raised, itself-tilted interlayer.

    Built rather than solved so the mechanism is pinned in the fast lane. The
    quasi-Fermi level is a real LEVEL (below `E_C`), not the solver's
    potential-convention variable — otherwise both Fermi factors saturate and
    every flux here would be exactly zero for the reason D8-E2R documents.
    """

    count = 73
    positions = np.linspace(0.0, 2.62e-7, count)
    conduction = np.empty(count)
    conduction[:24] = -4.50 - 0.09 * np.linspace(0.0, 1.0, 24)
    conduction[24:48] = -4.29 - 0.30 * np.linspace(0.0, 1.0, 24)
    conduction[48:] = -4.62 - 0.05 * np.linspace(0.0, 1.0, count - 48)
    quasi_fermi = -5.19 - 2.2e-4 * np.linspace(0.0, 1.0, count)
    return positions, conduction, quasi_fermi


def _channel(order: int = 96) -> IntrabandTunnellingChannel:
    return IntrabandTunnellingChannel(
        enabled=True, carrier="electron", energy_quadrature_order=order
    )


def _flux(positions, conduction, quasi_fermi, face, order=96):
    return intraband_flux(
        positions,
        conduction,
        _channel(order),
        anchor_face=face,
        carrier="electron",
        quasi_fermi_eV=quasi_fermi,
        thermal_voltage_V=THERMAL_VOLTAGE_V,
    ).net_flux_m2_s


# --------------------------------------------------------------------------
# The window cannot serve as a barrier identity
# --------------------------------------------------------------------------


def test_one_energy_window_is_shared_by_most_of_the_grid():
    """`local_barrier_window` is non-local, so it is not a barrier key.

    It climbs to a local maximum and then walks down to the bounding minima.
    On a device profile that walk converges to the same feature from a large
    basin of starting faces, so the window it returns identifies the basin,
    not the face.
    """
    _, conduction, _ = _synthetic_profile()
    windows = {}
    for face in range(conduction.size - 1):
        windows[face] = tuple(
            round(value, 10) for value in local_barrier_window(conduction, face)
        )

    distinct = Counter(windows.values())
    assert len(distinct) == 2
    # 49 of 72 faces — two thirds of the device — report one window.
    assert distinct.most_common(1)[0][1] == 49
    assert windows[_INTERFACE_FACE] == windows[30]
    assert windows[_INTERFACE_FACE] == windows[_SECOND_INTERFACE_FACE]


def test_the_barrier_is_identified_correctly_from_inside_the_run_only():
    """What the anchoring DOES get right, stated so the limitation is bounded.

    Among anchors that lie INSIDE the forbidden set at a given energy, every
    one returns the same connected run and therefore the same transmission —
    the barrier itself is identified correctly, and that is what makes the
    two-separated-barriers fix real. The limitation is the qualifier: whether
    a given anchor is inside at a given energy is a property of the anchor,
    and outside it the same call reports "no barrier" rather than "this
    barrier, seen from further away".
    """
    positions, conduction, _ = _synthetic_profile()
    peak, base = local_barrier_window(conduction, _INTERFACE_FACE)
    mid = 0.5 * (peak + base)

    runs = {
        face: forbidden_run(conduction, mid, face)
        for face in range(conduction.size - 1)
    }
    inside = {face: run for face, run in runs.items() if run is not None}
    outside = [face for face, run in runs.items() if run is None]

    # Every anchor inside the run agrees, exactly.
    assert len(set(inside.values())) == 1
    assert _INTERFACE_FACE in inside
    transmissions = {
        face: windowed_wkb_transmission(positions, conduction, mid, _MASS_REL, face)
        for face in inside
    }
    assert len(set(transmissions.values())) == 1

    # And the anchors outside report a transparent barrier at the SAME energy
    # the anchors inside report an opaque one — the two cannot be reconciled
    # by any property of the barrier.
    assert outside
    assert set(inside) != set(runs)
    opaque = next(iter(transmissions.values()))
    assert opaque < 1.0
    assert (
        windowed_wkb_transmission(positions, conduction, mid, _MASS_REL, outside[0])
        == 1.0
    )


# --------------------------------------------------------------------------
# The flux is a function of the anchor, not of the interface
# --------------------------------------------------------------------------


def test_the_flux_spans_orders_of_magnitude_with_anchor_position():
    """Same barrier, same window, five orders of magnitude of answer."""
    positions, conduction, quasi_fermi = _synthetic_profile()
    fluxes = {
        face: _flux(positions, conduction, quasi_fermi, face)
        for face in (23, 24, 30, 40, 47)
    }

    reference = fluxes[_INTERFACE_FACE]
    assert reference != 0.0
    # An adjacent face is indistinguishable from the interface...
    assert fluxes[24] / reference == pytest.approx(1.0, rel=1.0e-4)
    # ...and a face deeper into the same interlayer is not.
    assert fluxes[47] / reference > 1.0e4


def test_a_plain_interior_face_is_not_refused():
    """Nothing marks the interface out; any face in the basin answers.

    This is the sharpest statement of the limitation: `anchor_face` is not
    validated against the stack's interfaces, so a face with no physical
    significance produces a comparable, confidently-reported flux.
    """
    positions, conduction, quasi_fermi = _synthetic_profile()

    interior = _flux(positions, conduction, quasi_fermi, 30)

    assert np.isfinite(interior)
    assert interior != 0.0


# --------------------------------------------------------------------------
# The mechanism
# --------------------------------------------------------------------------


def test_the_unblocked_energy_count_grows_with_anchor_distance():
    """Which energies see a barrier at all is an anchor property.

    Near the top of the window the forbidden region narrows around the peak.
    An anchor away from the peak then falls outside it and that energy reports
    `T = 1`. The count is what moves the flux.
    """
    _, conduction, _ = _synthetic_profile()
    peak, base = local_barrier_window(conduction, _INTERFACE_FACE)
    energies = np.linspace(base, peak, 96)

    counts = [
        sum(forbidden_run(conduction, float(e), face) is None for e in energies)
        for face in (23, 30, 40, 47)
    ]

    assert counts == sorted(counts)
    assert counts[0] <= 2
    assert counts[-1] == len(energies)


def test_the_unblocked_branch_restores_the_one_cell_drive_d8_e2r_removed():
    """The `bounds is None` fallback is the superseded adjacent-node drive.

    `turning_point_levels` returns the anchor face's two NODES when the anchor
    is not inside a forbidden run. That is exactly the one-cell difference
    D8-E2R measured as mesh-divergent (the flux halved on every grid
    doubling) and replaced everywhere else. It survives on this branch, so
    those energies lose the barrier AND the driving force at once.
    """
    _, conduction, quasi_fermi = _synthetic_profile()
    peak, base = local_barrier_window(conduction, _INTERFACE_FACE)
    energies = np.linspace(base, peak, 96)
    # A face deep inside the interlayer, chosen because it straddles both
    # branches on this profile: the second interface is entirely outside the
    # run here and would exercise only one of them.
    face = 40

    left, right = _turning_point_levels(conduction, energies, face, quasi_fermi)
    blocked = np.array(
        [forbidden_run(conduction, float(e), face) is not None for e in energies]
    )
    drive = np.abs(left - right)

    # This profile must actually exercise both branches, or the test is vacuous.
    assert blocked.any() and (~blocked).any()

    one_cell = abs(quasi_fermi[face + 1] - quasi_fermi[face])
    # Outside the run: exactly the anchor face's two nodes, i.e. one cell.
    assert drive[~blocked] == pytest.approx(one_cell, rel=1.0e-12)
    # Inside it: the across-barrier drive, larger by roughly the ratio of the
    # barrier width to one cell.
    assert drive[blocked].max() > 5.0 * one_cell


# --------------------------------------------------------------------------
# The same measurement on the registered lane config
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_registered_lane_config_shows_the_same_anchor_dependence():
    """The real numbers, so the limitation is not only a synthetic artifact.

    Measured on `configs/wkb_tunnelling_intraband_spike.yaml` at the lane's
    coarsest grid and its own bias: a plain interior face seven cells from the
    interface reproduces 99.5 % of the interface-anchored flux, while the
    OTHER real heterointerface reports 0.19 % of it. The anchoring therefore
    does not pick out interfaces.

    Note the sign of the effect is opposite to the synthetic case above: here
    the solver hands over the potential-convention level (D8-E2R), so the
    unblocked energies have saturated occupations and contribute ~nothing,
    and the flux FALLS with anchor distance instead of rising. Same defect,
    two faces of it.
    """
    from pathlib import Path

    import perovskite_sim.experiments.quasi_fermi_steady_state as qf
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.device import electrical_layers

    root = Path(__file__).resolve().parents[3]
    stack = load_device_from_yaml(root / "configs/wkb_tunnelling_intraband_spike.yaml")
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, 24) for layer in electrical_layers(stack)),
        alpha=2.0,
    )
    captured: dict = {}
    original = qf.evaluate_tunnelling_channels

    def spy(compiled, **kwargs):
        captured["interface_faces"] = compiled.interface_faces
        captured.update(kwargs)
        return original(compiled, **kwargs)

    qf.evaluate_tunnelling_channels = spy
    try:
        qf.solve_quasi_fermi_steady_state(grid, stack, V_app=0.2, illuminated=False)
    finally:
        qf.evaluate_tunnelling_channels = original

    positions = np.asarray(captured["positions_m"], dtype=float)
    conduction = -(
        np.asarray(captured["potential_V"], dtype=float)
        + np.asarray(captured["affinity_eV"], dtype=float)
    )
    quasi_fermi = np.asarray(captured["electron_quasi_fermi_eV"], dtype=float)
    assert captured["interface_faces"] == (23, 47)

    reference = _flux(positions, conduction, quasi_fermi, 23)
    interior = _flux(positions, conduction, quasi_fermi, 30)
    second_interface = _flux(positions, conduction, quasi_fermi, 47)

    assert interior / reference == pytest.approx(0.995, abs=0.01)
    assert abs(second_interface / reference) < 0.01

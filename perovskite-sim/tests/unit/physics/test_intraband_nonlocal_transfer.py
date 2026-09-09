"""One physical barrier, reciprocal occupations and conservative nonlocal transfer."""

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.tunneling_channels import IntrabandTunnellingChannel
from perovskite_sim.physics.tunneling_channel_device import evaluate_tunnelling_channels
from perovskite_sim.solver.mol import build_material_arrays


def test_device_transfer_crosses_the_barrier_instead_of_one_anchor_face():
    stack = load_device_from_yaml("tests/fixtures/configs/wkb_tunnelling_intraband_spike.yaml")
    x = multilayer_grid([Layer(layer.thickness, 12) for layer in stack.layers])
    material = build_material_arrays(x, stack)
    potential = 0.02 * x / x[-1]
    result = evaluate_tunnelling_channels(
        material.tunnelling_channels, positions_m=x, potential_V=potential,
        affinity_eV=material.chi_phys, band_gap_eV=material.Eg_phys,
        electron_quasi_fermi_eV=-4.15 - 0.02 * x / x[-1],
        hole_quasi_fermi_eV=np.full(x.size, -5.0), thermal_voltage_V=material.V_T_device,
    )
    current = result.electron_face_current_A_m2
    assert np.max(np.abs(current)) > 1e-12
    assert np.count_nonzero(current) > 3


def test_refined_nodes_keep_their_physical_material_on_both_sides_of_a_boundary():
    stack = load_device_from_yaml("tests/fixtures/configs/wkb_resolved_electron_barrier.yaml")
    stack = replace(stack, tunnelling_channels=None)
    left = stack.layers[0].thickness
    right = left + stack.layers[1].thickness
    end = right + stack.layers[2].thickness
    x = np.array([0.0, left - 5e-13, left, left + 5e-13,
                  right - 5e-13, right, right + 5e-13, end])
    material = build_material_arrays(x, stack)
    np.testing.assert_array_equal(material.chi_phys, [4.0, 4.0, 3.7, 3.7, 3.7, 4.0, 4.0, 4.0])


def _profile(points=121):
    from perovskite_sim.physics.tunneling_channels import IntrabandBarrierRegion

    x = np.linspace(0.0, 12e-9, points)
    barrier = 0.3 * np.maximum(1 - np.abs(x - 6e-9) / 2e-9, 0.0)
    start, stop = (points - 1) // 3, 2 * (points - 1) // 3
    region = IntrabandBarrierRegion("triangle", 0, start, stop, points, x[start], x[stop])
    levels = -0.2 - 0.02 * x / x[-1]
    return x, barrier, levels, region


def _evaluate(x, barrier, levels, region, order=96):
    from perovskite_sim.physics.tunneling_channels import intraband_path_flux

    return intraband_path_flux(
        x, barrier, IntrabandTunnellingChannel(enabled=True, carrier="electron",
                                               energy_quadrature_order=order),
        region=region, quasi_fermi_eV=levels, thermal_voltage_V=0.02585,
    )


def test_transfer_conserves_particles_and_dissipates_the_chemical_drive():
    x, barrier, levels, region = _profile()
    path = _evaluate(x, barrier, levels, region)
    transfer = -np.diff(np.r_[0.0, path.particle_face_flux_m2_s, 0.0])
    assert np.any(transfer < 0.0) and np.any(transfer > 0.0)
    assert abs(np.sum(transfer)) < 1e-13 * np.sum(np.abs(transfer))
    assert np.dot(levels, transfer) < 0.0
    np.testing.assert_array_equal(path.particle_face_flux_m2_s[:region.core_start_node - 1], 0.0)
    np.testing.assert_array_equal(path.particle_face_flux_m2_s[region.core_stop_node:], 0.0)
    assert np.all(path.turning_points_m[:, 0] < path.turning_points_m[:, 1])
    assert np.all(path.actions > 0.0)


def test_equal_nonsaturated_occupations_give_exactly_zero_transfer():
    x, barrier, levels, region = _profile()
    path = _evaluate(x, barrier, np.full_like(levels, -0.2), region)
    assert np.all((path.flux.left_occupation > 0.0) & (path.flux.left_occupation < 0.001))
    assert path.flux.forward_flux_m2_s > 0.0
    assert path.flux.forward_flux_m2_s == path.flux.reverse_flux_m2_s
    assert path.flux.net_flux_m2_s == 0.0
    np.testing.assert_array_equal(path.particle_face_flux_m2_s, 0.0)


def test_exchanging_the_two_reservoirs_reverses_the_current():
    x, barrier, levels, region = _profile()
    forward = _evaluate(x, barrier, levels, region)
    reverse = _evaluate(x, barrier, levels[::-1], region)
    assert forward.flux.net_flux_m2_s > 0.0
    assert reverse.flux.net_flux_m2_s == pytest.approx(-forward.flux.net_flux_m2_s, rel=1e-12)
    np.testing.assert_allclose(reverse.particle_face_flux_m2_s,
                               -forward.particle_face_flux_m2_s[::-1], rtol=1e-11,
                               atol=1e-12 * np.max(np.abs(forward.particle_face_flux_m2_s)))


def test_energy_reference_shift_does_not_change_action_or_particle_transfer():
    x, barrier, levels, region = _profile()
    original = _evaluate(x, barrier, levels, region)
    shifted = _evaluate(x, barrier - 4.3, levels - 4.3, region)
    np.testing.assert_allclose(shifted.actions, original.actions, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(shifted.particle_face_flux_m2_s,
                               original.particle_face_flux_m2_s, rtol=1e-10,
                               atol=1e-12 * np.max(np.abs(original.particle_face_flux_m2_s)))


def test_geometry_cannot_silently_move_to_an_unrelated_barrier():
    from perovskite_sim.physics.tunneling_channels import TunnellingChannelError

    x, barrier, levels, region = _profile()
    with pytest.raises(TunnellingChannelError, match="physical boundaries"):
        _evaluate(x, barrier, levels, replace(region, core_left_m=region.core_left_m + 1e-10))


def test_legacy_anchor_changes_do_not_redefine_the_compiled_material_barrier():
    stack = load_device_from_yaml("tests/fixtures/configs/wkb_tunnelling_intraband_spike.yaml")
    x = multilayer_grid([Layer(layer.thickness, 12) for layer in stack.layers])
    material = build_material_arrays(x, stack)
    compiled = material.tunnelling_channels
    kwargs = dict(positions_m=x, potential_V=0.02 * x / x[-1], affinity_eV=material.chi_phys,
                  band_gap_eV=material.Eg_phys, electron_quasi_fermi_eV=-4.15 - 0.02 * x / x[-1],
                  hole_quasi_fermi_eV=np.full(x.size, -5.0), thermal_voltage_V=material.V_T_device)
    original = evaluate_tunnelling_channels(compiled, **kwargs)
    for face in range(compiled.intraband_barrier.core_start_node, compiled.intraband_barrier.core_stop_node - 1):
        moved = evaluate_tunnelling_channels(replace(compiled, interface_faces=(face,)), **kwargs)
        np.testing.assert_array_equal(moved.electron_face_current_A_m2, original.electron_face_current_A_m2)

"""Tunnelling occupations use physical band energies and local thermal DOS."""

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import K_B, Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments import quasi_fermi_steady_state as qf
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.mode import FULL
from perovskite_sim.physics.tunneling_channel_device import (
    TunnellingChannelCapabilityError,
    physical_quasi_fermi_levels_eV,
)
from perovskite_sim.physics.tunneling_channels import _fermi
from perovskite_sim.solver.mol import build_material_arrays


@pytest.mark.parametrize("temperature", [250.0, 300.0, 350.0])
def test_device_supplies_physical_levels_with_dos_contrast(temperature, monkeypatch):
    base = load_device_from_yaml("tests/fixtures/configs/wkb_tunnelling_intraband_spike.yaml")
    layers = []
    for layer, nc_factor, nv_factor in zip(base.layers, (1.0, 3.0, 0.5), (1.0, 0.7, 2.0)):
        nc, nv = layer.params.Nc300 * nc_factor, layer.params.Nv300 * nv_factor
        ni = np.sqrt(nc * nv) * np.exp(-layer.params.Eg / (2 * K_B * 300.0 / Q))
        layers.append(replace(layer, params=replace(
            layer.params, Nc300=nc, Nv300=nv, ni=float(ni), n1=float(ni), p1=float(ni),
        )))
    stack = replace(base, layers=tuple(layers), T=temperature,
                    mode=replace(FULL, use_thermionic_emission=False))
    grid = multilayer_grid([Layer(layer.thickness, 12) for layer in layers], alpha=2.0)
    material = build_material_arrays(grid, stack)
    assert not np.array_equal(material.chi, material.chi_phys)
    assert material.V_T_device == pytest.approx(K_B * temperature / Q)
    captured = {}
    original = qf.evaluate_tunnelling_channels

    def capture(compiled, **kwargs):
        captured.update({key: value.copy() if isinstance(value, np.ndarray) else value
                         for key, value in kwargs.items()})
        return original(compiled, **kwargs)

    monkeypatch.setattr(qf, "evaluate_tunnelling_channels", capture)
    result = qf.solve_quasi_fermi_steady_state(grid, stack, V_app=0.2, illuminated=False, mat=material)
    assert result.certified
    conduction = -captured["potential_V"] - captured["affinity_eV"]
    valence = conduction - captured["band_gap_eV"]
    thermal = K_B * temperature / Q
    expected_n = conduction + thermal * (
        np.log(captured["electron_density_m3"]) - np.log(material.N_C_physical)
    )
    expected_p = valence - thermal * (
        np.log(captured["hole_density_m3"]) - np.log(material.N_V_physical)
    )
    np.testing.assert_allclose(captured["electron_quasi_fermi_eV"], expected_n, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(captured["hole_quasi_fermi_eV"], expected_p, rtol=0.0, atol=1e-12)


def _prescribed_levels(temperature):
    thermal = K_B * temperature / Q
    potential = np.array([0.0, 0.15, 0.25])
    affinity = np.array([4.0, 3.7, 4.1])
    gap = np.array([1.5, 2.1, 1.4])
    nc = np.array([1e24, 3e24, 2e23]) * (temperature / 300.0)**1.5
    nv = np.array([8e23, 6e23, 3e24]) * (temperature / 300.0)**1.5
    fn = np.array([-4.6, -4.7, -4.8])
    fp = np.array([-5.0, -5.2, -5.15])
    conduction = -potential - affinity
    valence = conduction - gap
    return dict(
        potential_V=potential, affinity_eV=affinity, band_gap_eV=gap,
        electron_density_m3=nc * np.exp((fn - conduction) / thermal),
        hole_density_m3=nv * np.exp((valence - fp) / thermal),
        conduction_dos_m3=nc, valence_dos_m3=nv, thermal_voltage_V=thermal,
    ), fn, fp


@pytest.mark.parametrize("temperature", [250.0, 300.0, 350.0])
def test_density_law_recovers_prescribed_physical_levels(temperature):
    inputs, expected_n, expected_p = _prescribed_levels(temperature)
    actual_n, actual_p = physical_quasi_fermi_levels_eV(**inputs)
    np.testing.assert_allclose(actual_n, expected_n, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(actual_p, expected_p, rtol=0.0, atol=2e-14)


def test_global_energy_reference_shift_preserves_occupations_and_drive():
    inputs, _, _ = _prescribed_levels(300.0)
    electron, hole = physical_quasi_fermi_levels_eV(**inputs)
    offset = 0.73
    moved = dict(inputs, affinity_eV=inputs["affinity_eV"] + offset)
    shifted_n, shifted_p = physical_quasi_fermi_levels_eV(**moved)
    np.testing.assert_allclose(shifted_n, electron - offset, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(shifted_p, hole - offset, rtol=0.0, atol=2e-14)
    energies = np.linspace(-4.35, -4.1, 25)
    thermal = inputs["thermal_voltage_V"]
    occupation = _fermi(energies, electron[0], thermal)
    shifted = _fermi(energies - offset, shifted_n[0], thermal)
    assert np.all((occupation > 1e-12) & (occupation < 1e-3))
    np.testing.assert_allclose(shifted, occupation, rtol=1e-12, atol=0.0)
    higher = _fermi(energies, electron[0] + 1e-5, thermal)
    assert np.all(higher > occupation)
    np.testing.assert_array_equal(higher - occupation, -(occupation - higher))


@pytest.mark.parametrize("field", ["conduction_dos_m3", "valence_dos_m3"])
@pytest.mark.parametrize("bad", [None, 0.0, np.nan])
def test_missing_or_invalid_dos_is_never_replaced_by_a_transport_potential(field, bad):
    inputs, _, _ = _prescribed_levels(300.0)
    inputs[field] = None if bad is None else np.full(3, bad)
    with pytest.raises(TunnellingChannelCapabilityError, match="Nc/Nv"):
        physical_quasi_fermi_levels_eV(**inputs)

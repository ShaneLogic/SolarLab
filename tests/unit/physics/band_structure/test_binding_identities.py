"""Binding behaviour (M class): identities, cold compile / rebind and rejections."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import band_structure_test_support as support
from solarlab.physics import band_structure as new
from solarlab.physics.band_structure import (
    DopingProfileSpec,
    GradingSpec,
    LayerParameterInput,
    StaleParameterBindingError,
    TemperatureCapabilityError,
    TemperatureLawSpec,
    compile_layer_parameters,
)


def _layer(**overrides):
    values = dict(
        name="absorber", role="absorber", thickness_m=400e-9, electron_affinity_eV=3.9, band_gap_eV=1.6,
        intrinsic_density_m3=8.05e10, srh_n1_m3=8.05e10, srh_p1_m3=8.05e10, electron_mobility_m2_V_s=2e-4,
        hole_mobility_m2_V_s=2e-4, radiative_coefficient_m3_s=4e-17, doping=DopingProfileSpec(0.0, 0.0),
        conduction_dos_300_m3=2.2e24, valence_dos_300_m3=1.8e25, ion_diffusivity_m2_s=1e-16,
        temperature_laws=TemperatureLawSpec(-1.5, 0.58, -1.2, -3e-4, 200.0),
    )
    values.update(overrides)
    return LayerParameterInput(**values)


def _stack():
    return (
        _layer(name="htl", role="htl", thickness_m=100e-9, band_gap_eV=3.0, electron_affinity_eV=2.1,
               conduction_dos_300_m3=1e26, valence_dos_300_m3=1e26, doping=DopingProfileSpec(1e24, 0.0)),
        _layer(),
        _layer(name="etl", role="etl", thickness_m=50e-9, band_gap_eV=3.2, electron_affinity_eV=4.0,
               conduction_dos_300_m3=None, valence_dos_300_m3=None, doping=DopingProfileSpec(0.0, 1e24)),
    )


def _grid():
    return np.concatenate([np.linspace(0.0, 100e-9, 11), np.linspace(100e-9, 500e-9, 41)[1:],
                           np.linspace(500e-9, 550e-9, 6)[1:]])


_SWITCHES = dict(temperature_scaling=True, band_grading=True, dos_band_potentials=True)
_ARRAYS = ("intrinsic_product_m6", "srh_n1_m3", "srh_p1_m3", "radiative_coefficient_m3_s", "electron_diffusivity_m2_s",
           "hole_diffusivity_m2_s", "ion_diffusivity_m2_s", "conduction_dos_m3", "valence_dos_m3", "thermal_velocity_m_s",
           "acceptor_density_m3", "donor_density_m3")


def _arrays(binding):
    band = binding.band_structure
    yield from (getattr(binding, name) for name in _ARRAYS)
    yield from (band.physical_electron_affinity_eV, band.physical_band_gap_eV, band.transport_electron_affinity_eV,
                band.transport_band_gap_eV)


def _bitwise_equal(a, b) -> bool:
    return all(support.same(x, y) for x, y in zip(_arrays(a), _arrays(b)))


def test_reference_temperature_returns_the_declared_values():
    binding = compile_layer_parameters(_grid(), _stack(), temperature_K=300.0, **_SWITCHES)
    absorber = binding.layer_masks[1]
    assert np.all(binding.intrinsic_product_m6[absorber] == 8.05e10**2)
    assert np.all(binding.srh_n1_m3[absorber] == 8.05e10)
    assert np.all(binding.radiative_coefficient_m3_s[absorber] == 4e-17)
    assert np.all(binding.electron_mobility_m2_V_s[absorber] == 2e-4)
    assert np.all(binding.band_structure.physical_band_gap_eV[absorber] == 1.6)
    assert np.all(binding.conduction_dos_m3[absorber] == 2.2e24)
    assert binding.thermal_voltage_V == new.device_temperature(300.0, temperature_scaling=False)[1]


def test_disabled_temperature_scaling_freezes_every_law_at_300_K():
    hot = compile_layer_parameters(_grid(), _stack(), temperature_K=350.0, temperature_scaling=False,
                                   band_grading=True, dos_band_potentials=True)
    reference = compile_layer_parameters(_grid(), _stack(), temperature_K=300.0, **_SWITCHES)
    assert hot.temperature_K == 300.0 and hot.requested_temperature_K == 350.0
    assert _bitwise_equal(hot, reference)
    assert hot.sha256 != reference.sha256  # the request differs, so the identity does


def test_flat_grade_is_bitwise_the_ungraded_layer():
    layers = list(_stack())
    flat = list(layers)
    flat[1] = dataclasses.replace(layers[1], grading=GradingSpec(band_gap_back_eV=1.6, electron_affinity_back_eV=3.9,
                                                                    profile="parabolic"))
    for T in (300.0, 335.0):
        ungraded = compile_layer_parameters(_grid(), layers, temperature_K=T, **_SWITCHES)
        graded = compile_layer_parameters(_grid(), flat, temperature_K=T, **_SWITCHES)
        assert graded.layer_states[1].graded
        assert _bitwise_equal(graded, ungraded)


def test_grading_switch_off_ignores_the_back_endpoints():
    layers = list(_stack())
    layers[1] = dataclasses.replace(layers[1], grading=GradingSpec(band_gap_back_eV=1.3))
    off = compile_layer_parameters(_grid(), layers, temperature_K=320.0, temperature_scaling=True,
                                   band_grading=False, dos_band_potentials=True)
    plain = compile_layer_parameters(_grid(), _stack(), temperature_K=320.0, temperature_scaling=True,
                                     band_grading=False, dos_band_potentials=True)
    assert _bitwise_equal(off, plain) and not off.layer_states[1].graded


def test_rebind_is_a_cold_compile_and_never_reuses_coefficients():
    cold_320 = compile_layer_parameters(_grid(), _stack(), temperature_K=320.0, **_SWITCHES)
    cold_350 = compile_layer_parameters(_grid(), _stack(), temperature_K=350.0, **_SWITCHES)
    rebound = cold_320.rebind(temperature_K=350.0)
    assert _bitwise_equal(rebound, cold_350) and rebound.sha256 == cold_350.sha256
    back = rebound.rebind(temperature_K=320.0)
    assert _bitwise_equal(back, cold_320) and back.sha256 == cold_320.sha256
    assert cold_320.sha256 != cold_350.sha256
    # Compounding the 320 K values as if they were 300 K inputs would differ:
    compounded = new.mobility_at_temperature(float(cold_320.electron_mobility_m2_V_s[30]), 350.0 * 350.0 / 320.0)
    assert compounded != float(cold_350.electron_mobility_m2_V_s[30])


def test_composition_rebind_recompiles_the_layer():
    graded = list(_stack())
    graded[1] = dataclasses.replace(graded[1], grading=GradingSpec(band_gap_back_eV=1.3, electron_affinity_back_eV=4.0))
    base = compile_layer_parameters(_grid(), _stack(), temperature_K=320.0, **_SWITCHES)
    rebound = base.rebind(layers=graded)
    cold = compile_layer_parameters(_grid(), graded, temperature_K=320.0, **_SWITCHES)
    assert _bitwise_equal(rebound, cold) and rebound.inputs_sha256 != base.inputs_sha256
    with pytest.raises(StaleParameterBindingError):
        base.require(temperature_K=320.0, inputs_sha256=cold.inputs_sha256)


def test_a_binding_refuses_a_different_temperature_request():
    binding = compile_layer_parameters(_grid(), _stack(), temperature_K=320.0, **_SWITCHES)
    binding.require(temperature_K=320.0, inputs_sha256=binding.inputs_sha256)
    with pytest.raises(StaleParameterBindingError, match="rebind"):
        binding.require(temperature_K=321.0)


def test_binding_arrays_are_read_only_and_state_free():
    binding = compile_layer_parameters(_grid(), _stack(), temperature_K=320.0, **_SWITCHES)
    with pytest.raises(ValueError):
        binding.intrinsic_product_m6[0] = 1.0
    with pytest.raises(ValueError):
        binding.layer_masks[0][0] = False
    fields = {field.name for field in dataclasses.fields(binding)}
    assert not {"phi", "electron_density_m3", "field_V_m", "face_mobility"} & fields


def test_time_domain_temperature_is_declared_but_refused():
    assert new.TIME_DOMAIN_TEMPERATURE_QUALIFIED is False
    with pytest.raises(TemperatureCapabilityError, match="pending_p04b"):
        new.require_time_domain_temperature()


@pytest.mark.parametrize(
    "build",
    [
        lambda: GradingSpec(band_gap_back_eV=1.3, profile="Linear"),  # never silently read as linear
        lambda: GradingSpec(band_gap_back_eV=1.3, direction="sideways"),
        lambda: DopingProfileSpec(1e22, 0.0, 1e20, None, "erfc", 1e-8),
        lambda: _layer(thickness_m=0.0),
        lambda: _layer(conduction_dos_300_m3=-1.0),
    ],
)
def test_rejections_at_binding_time(build):
    with pytest.raises(ValueError):
        build()


def test_grid_must_be_covered_by_the_layers():
    with pytest.raises(ValueError, match="cover"):
        compile_layer_parameters(np.linspace(0.0, 700e-9, 30), _stack(), temperature_K=300.0, **_SWITCHES)
    with pytest.raises(ValueError):
        compile_layer_parameters(_grid(), _stack(), temperature_K=0.0, **_SWITCHES)


def test_reference_calibrated_terms_are_declared_not_scaled():
    owners = {term.term: term.owner for term in new.REFERENCE_CALIBRATED_TERMS}
    assert "interface_plane_thermal_velocity" in owners and "interface_defect_reference_densities" in owners
    assert len(new.TEMPERATURE_LAWS) == len({law_id for _, law_id in new.TEMPERATURE_LAWS})

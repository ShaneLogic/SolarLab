"""N class: temperature and composition tangents of the P04-11 parameter laws.

Oracles are independent of the analytic forms: complex-step or central
differences of the M-class value laws, and differences of cold-compiled
bindings at neighbouring temperatures.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from solarlab.physics import band_structure as laws
from solarlab.physics.band_structure import (
    DopingProfileSpec,
    GradingSpec,
    LayerParameterInput,
    TemperatureCapabilityError,
    TemperatureLawSpec,
    compile_layer_parameters,
)
from solarlab.physics.band_structure import sensitivity as tangents

K_B = 1.380649e-23
Q = 1.602176634e-19
NON_REFERENCE = (250.0, 320.0, 350.0, 420.0)


def _central(function, T, h=1e-3):
    return (function(T + h) - function(T - h)) / (2.0 * h)


@pytest.mark.parametrize("T", NON_REFERENCE)
def test_scalar_law_tangents_match_central_differences(T):
    cases = [
        (lambda t: laws.mobility_at_temperature(2e-4, t, -1.5),
         tangents.mobility_temperature_derivative(2e-4, T, -1.5)),
        (lambda t: laws.ion_diffusivity_at_temperature(1e-16, t, 0.58),
         tangents.ion_diffusivity_temperature_derivative(1e-16, T, 0.58)),
        (lambda t: laws.radiative_coefficient_at_temperature(4e-17, t, -1.2),
         tangents.radiative_coefficient_temperature_derivative(4e-17, T, -1.2)),
        (lambda t: laws.band_gap_at_temperature(1.12, t, 4.73e-4, 636.0),
         tangents.band_gap_temperature_derivative(T, 4.73e-4, 636.0)),
        (lambda t: laws.band_gap_at_temperature(1.6, t, -3e-4, 200.0),
         tangents.band_gap_temperature_derivative(T, -3e-4, 200.0)),
        (lambda t: laws.band_edge_dos_at_temperature(2.8e25, t),
         tangents.band_edge_dos_temperature_derivative(2.8e25, T)),
        (lambda t: laws.thermal_velocity_at_temperature(1e5, t),
         tangents.thermal_velocity_temperature_derivative(1e5, T)),
    ]
    for function, analytic in cases:
        assert analytic == pytest.approx(_central(function, T), rel=2e-6)


@pytest.mark.parametrize("T", NON_REFERENCE)
@pytest.mark.parametrize("dos", [(2.8e25, 1.04e25), (None, None)])
def test_intrinsic_density_partials(T, dos):
    ni, d_t, d_eg = tangents.intrinsic_density_temperature_partials(1e16, 1.12, T, *dos)
    assert ni == laws.intrinsic_density_at_temperature(1e16, 1.12, T, *dos)
    assert d_t == pytest.approx(_central(lambda t: laws.intrinsic_density_at_temperature(1e16, 1.12, t, *dos), T), rel=2e-6)
    h = 1e-6
    fd_eg = (laws.intrinsic_density_at_temperature(1e16, 1.12 + h, T, *dos)
             - laws.intrinsic_density_at_temperature(1e16, 1.12 - h, T, *dos)) / (2 * h)
    assert d_eg == pytest.approx(fd_eg, rel=2e-6)


def test_disabled_laws_have_exactly_zero_tangents():
    assert tangents.radiative_coefficient_temperature_derivative(4e-17, 330.0, 0.0) == 0.0
    assert tangents.radiative_coefficient_temperature_derivative(0.0, 330.0, -1.5) == 0.0
    assert tangents.ion_diffusivity_temperature_derivative(0.0, 330.0, 0.58) == 0.0
    assert tangents.band_gap_temperature_derivative(330.0, 0.0, 636.0) == 0.0
    assert tangents.band_gap_temperature_derivative(330.0, 1e-4, -300.0) == 0.0  # degenerate reference: constant law
    with pytest.raises(TemperatureCapabilityError):
        tangents.band_gap_temperature_derivative(250.0, 1e-4, -250.0)  # Varshni pole


def test_thermal_equilibrium_relation_holds_away_from_the_reference():
    Nc, Nv = 2.8e25, 1.04e25
    for T in NON_REFERENCE:
        gap = laws.band_gap_at_temperature(1.12, T, 4.73e-4, 636.0)
        ni = laws.intrinsic_density_at_temperature(1e16, gap, T, Nc, Nv)
        expected = laws.band_edge_dos_at_temperature(Nc, T) * laws.band_edge_dos_at_temperature(Nv, T) * math.exp(
            -gap * Q / (K_B * T)
        )
        assert ni * ni == pytest.approx(expected, rel=1e-14)


def test_reference_jump_of_an_inconsistent_ni_declaration_is_measured_and_refused():
    # silicon-like declaration: ni_300 = 1e16 m^-3 is not the DOS formula's value
    jump = tangents.intrinsic_density_reference_jump(1e16, 1.12, 2.8e25, 1.04e25)
    formula = tangents.intrinsic_density_temperature_partials(1e16, 1.12, 300.0, 2.8e25, 1.04e25)[0]
    assert jump == formula - 1e16 and abs(jump) > 1e-3 * 1e16
    with pytest.raises(TemperatureCapabilityError, match="discontinuous at 300 K"):
        tangents.require_continuous_reference(1e16, 1.12, 2.8e25, 1.04e25)
    consistent = tangents.intrinsic_density_temperature_partials(1e16, 1.12, 300.0, 2.8e25, 1.04e25)[0]
    tangents.require_continuous_reference(consistent, 1.12, 2.8e25, 1.04e25)
    tangents.require_continuous_reference(8.05e10, 1.6)  # scaled branch is continuous by construction


def test_inverse_maps_round_trip():
    for T in NON_REFERENCE + (300.0,):
        for alpha, beta in ((4.73e-4, 636.0), (-3e-4, 200.0), (0.0, 0.0)):
            shifted = laws.band_gap_at_temperature(1.12, T, alpha, beta)
            assert tangents.band_gap_at_reference(shifted, T, alpha, beta) == pytest.approx(1.12, abs=4e-16)
    x = np.linspace(0.0, 500e-9, 101)
    for profile, length in (("linear", None), ("parabolic", None), ("exponential", 80e-9), ("exponential", None)):
        for direction in ("front_to_back", "back_to_front"):
            y = laws.grading_coordinate(x, 500e-9, profile, length, direction)
            back = tangents.grading_position(y, 500e-9, profile, length, direction)
            np.testing.assert_allclose(back, x, rtol=0.0, atol=2e-21)


@pytest.mark.parametrize("direction", ["front_to_back", "back_to_front"])
def test_exponential_inverse_keeps_steep_profile_endpoints(direction):
    from decimal import Decimal, localcontext

    y = np.array([0.0, 0.5, np.nextafter(1.0, 0.0), 1.0])
    thickness, length = 300e-9, 5e-9
    position = tangents.grading_position(y, thickness, "exponential", length, direction)
    with localcontext() as context:
        context.prec = 100
        t, ell = Decimal(thickness), Decimal(length)
        u = 1.0 - y if direction == "back_to_front" else y
        expected = [float(-ell * (1 - Decimal(float(v)) + Decimal(float(v)) * (-t / ell).exp()).ln()) for v in u]
    assert np.all(np.isfinite(position))
    np.testing.assert_allclose(position, expected, rtol=3e-15, atol=0.0)
    assert position[-1 if direction == "front_to_back" else 0] == thickness


def test_reverse_exponential_inverse_preserves_small_supplied_composition():
    from decimal import Decimal, localcontext

    y = np.array([1e-20, 1e-16, 1e-8])
    thickness, length = 300e-9, 5e-9
    actual = tangents.grading_position(y, thickness, "exponential", length, "back_to_front")
    with localcontext() as context:
        context.prec = 120
        t, ell = Decimal(thickness), Decimal(length)
        expected = [float(-ell * (Decimal(float(v)) + (1 - Decimal(float(v))) * (-t / ell).exp()).ln()) for v in y]
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0.0)
    with pytest.raises(ValueError, match="composition"):
        tangents.grading_position(np.array([-1e-20]), thickness, "exponential", length, "back_to_front")


@pytest.mark.parametrize("profile, length", [("linear", None), ("parabolic", None), ("exponential", 80e-9)])
@pytest.mark.parametrize("direction", ["front_to_back", "back_to_front"])
def test_composition_coordinate_slope(profile, length, direction):
    x = np.linspace(10e-9, 490e-9, 25)
    h = 1e-12
    fd = (laws.grading_coordinate(x + h, 500e-9, profile, length, direction)
          - laws.grading_coordinate(x - h, 500e-9, profile, length, direction)) / (2 * h)
    np.testing.assert_allclose(tangents.grading_coordinate_slope(x, 500e-9, profile, length, direction), fd, rtol=1e-6)


def test_composition_law_tangents():
    y = np.linspace(0.0, 1.0, 21)
    h = 1e-7
    for front, back, bowing in ((1.04, 1.68, 0.2), (1.6, 1.1, -0.15)):
        fd = (laws.band_gap_profile(y + h, front, back, bowing) - laws.band_gap_profile(y - h, front, back, bowing)) / (2 * h)
        np.testing.assert_allclose(tangents.band_gap_composition_derivative(y, front, back, bowing), fd, rtol=1e-7, atol=1e-9)
    assert np.all(tangents.band_gap_composition_derivative(y, 1.2, 1.2, 0.0) == 0.0)
    assert np.all(tangents.affinity_composition_derivative(y, 4.1, 4.4) == pytest.approx(0.3))


def _layers(*, graded: bool, narrowed: bool):
    from solarlab.physics.band_structure import BandGapNarrowingSpec

    absorber = LayerParameterInput(
        name="absorber", role="absorber", thickness_m=400e-9, electron_affinity_eV=3.9, band_gap_eV=1.6,
        intrinsic_density_m3=8.05e10, srh_n1_m3=3e9, srh_p1_m3=2.1e12, electron_mobility_m2_V_s=2e-4,
        hole_mobility_m2_V_s=1e-4, radiative_coefficient_m3_s=4e-17, doping=DopingProfileSpec(1e22, 0.0),
        ion_diffusivity_m2_s=1e-16, negative_ion_diffusivity_m2_s=3e-17,
        temperature_laws=TemperatureLawSpec(-1.5, 0.58, -1.2, -3e-4, 200.0),
        grading=GradingSpec(band_gap_back_eV=1.3, electron_affinity_back_eV=4.0, bowing_eV=0.1) if graded else GradingSpec(),
        narrowing=BandGapNarrowingSpec("slotboom") if narrowed else BandGapNarrowingSpec(),
    )
    contact = LayerParameterInput(
        name="htl", role="htl", thickness_m=100e-9, electron_affinity_eV=2.1, band_gap_eV=3.0,
        intrinsic_density_m3=1e5, srh_n1_m3=1e5, srh_p1_m3=1e5, electron_mobility_m2_V_s=1e-6,
        hole_mobility_m2_V_s=1e-6, radiative_coefficient_m3_s=0.0, doping=DopingProfileSpec(1e24, 0.0),
        conduction_dos_300_m3=1e26, valence_dos_300_m3=1e26,
    )
    return (contact, dataclasses.replace(absorber, conduction_dos_300_m3=2.2e24, valence_dos_300_m3=1.8e25))


_FIELDS = (
    ("physical_band_gap_eV_K", lambda b: b.band_structure.physical_band_gap_eV),
    ("transport_band_gap_eV_K", lambda b: b.band_structure.transport_band_gap_eV),
    ("transport_electron_affinity_eV_K", lambda b: b.band_structure.transport_electron_affinity_eV),
    ("intrinsic_product_m6_K", lambda b: b.intrinsic_product_m6),
    ("srh_n1_m3_K", lambda b: b.srh_n1_m3),
    ("srh_p1_m3_K", lambda b: b.srh_p1_m3),
    ("radiative_coefficient_m3_s_K", lambda b: b.radiative_coefficient_m3_s),
    ("electron_diffusivity_K", lambda b: b.electron_diffusivity_m2_s),
    ("hole_mobility_K", lambda b: b.hole_mobility_m2_V_s),
    ("ion_diffusivity_K", lambda b: b.ion_diffusivity_m2_s),
    ("negative_ion_diffusivity_K", lambda b: b.negative_ion_diffusivity_m2_s),
    ("conduction_dos_m3_K", lambda b: b.conduction_dos_m3),
    ("thermal_velocity_K", lambda b: b.thermal_velocity_m_s),
)


@pytest.mark.parametrize("graded, narrowed", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("T", [320.0, 350.0])
def test_binding_sensitivity_matches_differences_of_cold_compiles(graded, narrowed, T):
    x = np.linspace(0.0, 500e-9, 61)
    layers = _layers(graded=graded, narrowed=narrowed)
    switches = dict(temperature_scaling=True, band_grading=True, dos_band_potentials=True)
    binding = compile_layer_parameters(x, layers, temperature_K=T, **switches)
    sensitivity = tangents.binding_temperature_sensitivity(binding)
    h = 1e-3
    plus = binding.rebind(temperature_K=T + h)
    minus = binding.rebind(temperature_K=T - h)
    assert sensitivity.temperature_dependent and not sensitivity.time_domain_qualified
    for name, read in _FIELDS:
        analytic = getattr(sensitivity, name)
        fd = (np.asarray(read(plus)) - np.asarray(read(minus))) / (2 * h)
        finite = np.isfinite(fd)
        assert np.array_equal(finite, np.isfinite(analytic)), name
        scale = np.max(np.abs(fd[finite])) if np.any(finite) else 0.0
        np.testing.assert_allclose(analytic[finite], fd[finite], rtol=1e-5, atol=1e-6 * scale + 1e-300, err_msg=name)
    assert np.all(sensitivity.physical_electron_affinity_eV_K == 0.0)
    assert sensitivity.thermal_voltage_V_K == K_B / Q


def test_binding_sensitivity_without_temperature_scaling_is_zero():
    binding = compile_layer_parameters(np.linspace(0.0, 500e-9, 31), _layers(graded=False, narrowed=False),
                                       temperature_K=350.0, temperature_scaling=False, band_grading=True,
                                       dos_band_potentials=True)
    sensitivity = tangents.binding_temperature_sensitivity(binding)
    assert not sensitivity.temperature_dependent
    assert all(np.all(getattr(sensitivity, name) == 0.0) for name, _ in _FIELDS)


def test_binding_sensitivity_at_300_K_refuses_a_discontinuous_declaration():
    layers = list(_layers(graded=False, narrowed=False))
    binding = compile_layer_parameters(np.linspace(0.0, 500e-9, 31), layers, temperature_K=300.0,
                                       temperature_scaling=True, band_grading=True, dos_band_potentials=True)
    with pytest.raises(TemperatureCapabilityError, match="discontinuous"):
        tangents.binding_temperature_sensitivity(binding)

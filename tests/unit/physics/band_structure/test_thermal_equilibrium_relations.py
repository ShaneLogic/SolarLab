"""M class: thermal-equilibrium relations and independent closed forms of the value laws.

These tests read the migrated (M-class) laws and binding only. The closed
forms are evaluated in 50-digit decimal arithmetic, independently of NumPy.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest

from solarlab.physics import band_structure as laws
from solarlab.physics.band_structure import (
    BandGapNarrowingSpec,
    DopingProfileSpec,
    GradingSpec,
    LayerParameterInput,
    TemperatureLawSpec,
    compile_layer_parameters,
)

getcontext().prec = 50
K_B = Decimal("1.380649e-23")
Q = Decimal("1.602176634e-19")
T_REF = Decimal(300)
NON_REFERENCE = (320.0, 350.0)
EPS = np.finfo(float).eps


def _close(value: float, exact: Decimal, ulps: int) -> bool:
    return abs(Decimal(value) - exact) <= ulps * Decimal(EPS) * abs(exact)


@pytest.mark.parametrize("T", NON_REFERENCE)
def test_two_non_reference_temperatures_match_decimal_closed_forms(T):
    t = Decimal(T)
    r = t / T_REF
    Nc, Nv, gap = Decimal("2.8e25"), Decimal("1.04e25"), Decimal("1.12")
    exact_dos_ni = (Nc * Nv * r**3).sqrt() * (-(gap * Q) / (2 * K_B * t)).exp()
    assert _close(laws.intrinsic_density_at_temperature(1e16, 1.12, T, 2.8e25, 1.04e25), exact_dos_ni, 16)
    exact_scaled_ni = Decimal("8.05e10") * r ** Decimal("1.5") * (-(Decimal("1.6") * Q) / (2 * K_B) * (1 / t - 1 / T_REF)).exp()
    assert _close(laws.intrinsic_density_at_temperature(8.05e10, 1.6, T), exact_scaled_ni, 16)
    assert _close(laws.mobility_at_temperature(2e-4, T, -1.5), Decimal("2e-4") * r ** Decimal("-1.5"), 8)
    exact_D = Decimal("1e-16") * (-(Decimal("0.58") * Q) / K_B * (1 / t - 1 / T_REF)).exp()
    assert _close(laws.ion_diffusivity_at_temperature(1e-16, T, 0.58), exact_D, 64)
    assert _close(laws.radiative_coefficient_at_temperature(4e-17, T, -1.2), Decimal("4e-17") * r ** Decimal("-1.2"), 8)
    alpha, beta = Decimal("4.73e-4"), Decimal(636)
    exact_gap = Decimal("1.12") - alpha * (t * t / (t + beta) - T_REF * T_REF / (T_REF + beta))
    assert _close(laws.band_gap_at_temperature(1.12, T, 4.73e-4, 636.0), exact_gap, 8)
    assert _close(laws.band_edge_dos_at_temperature(2.8e25, T), Nc * r ** Decimal("1.5"), 8)
    assert _close(laws.thermal_velocity_at_temperature(1e5, T), Decimal("1e5") * r.sqrt(), 8)


@pytest.mark.parametrize("T", NON_REFERENCE + (250.0,))
def test_dos_branch_satisfies_mass_action_away_from_the_reference(T):
    gap = laws.band_gap_at_temperature(1.12, T, 4.73e-4, 636.0)
    ni = laws.intrinsic_density_at_temperature(1e16, gap, T, 2.8e25, 1.04e25)
    product = laws.band_edge_dos_at_temperature(2.8e25, T) * laws.band_edge_dos_at_temperature(1.04e25, T)
    assert ni * ni == pytest.approx(product * np.exp(-gap * 1.602176634e-19 / (1.380649e-23 * T)), rel=32 * EPS)


def test_declared_ni_is_kept_at_the_reference_even_when_the_dos_formula_differs():
    """The historical 300 K semantics: the law jumps when the declaration is inconsistent.

    The silicon research presets declare ni_300 = 1e16 m^-3 with Nc/Nv/Eg whose
    DOS formula gives 6.18e15 m^-3; just above 300 K the law switches to the
    formula, a -38.2 % jump. Kept on purpose (changing it is a physics revision).
    """

    # csi_*_pn_research.yaml: ni = 1.0e16, Eg = 1.124, Nc300 = 2.8e25, Nv300 = 1.04e25
    at_reference = laws.intrinsic_density_at_temperature(1e16, 1.124, 300.0, 2.8e25, 1.04e25)
    just_above = laws.intrinsic_density_at_temperature(1e16, 1.124, 300.0 + 1e-9, 2.8e25, 1.04e25)
    assert at_reference == 1e16
    assert just_above / 1e16 - 1.0 == pytest.approx(-0.3821, abs=5e-5)
    consistent = laws.intrinsic_density_at_temperature(8.05e10, 1.6, 300.0 + 1e-9)
    assert consistent / 8.05e10 - 1.0 == pytest.approx(0.0, abs=1e-6)


def _layer(**overrides):
    values = dict(
        name="absorber", role="absorber", thickness_m=400e-9, electron_affinity_eV=4.05, band_gap_eV=1.12,
        intrinsic_density_m3=6.0e15, srh_n1_m3=6.0e15 * 3.0, srh_p1_m3=6.0e15 / 3.0, electron_mobility_m2_V_s=0.14,
        hole_mobility_m2_V_s=0.045, radiative_coefficient_m3_s=1.1e-20, doping=DopingProfileSpec(3e24, 0.0),
        conduction_dos_300_m3=2.8e25, valence_dos_300_m3=1.04e25,
        temperature_laws=TemperatureLawSpec(-1.5, 0.58, -1.5, 4.73e-4, 636.0),
    )
    values.update(overrides)
    return LayerParameterInput(**values)


@pytest.mark.parametrize(
    "layer",
    [
        _layer(),
        _layer(narrowing=BandGapNarrowingSpec("slotboom")),
        _layer(grading=GradingSpec(band_gap_back_eV=1.3, electron_affinity_back_eV=3.95, profile="parabolic", bowing_eV=0.05)),
        _layer(grading=GradingSpec(band_gap_back_eV=1.3), narrowing=BandGapNarrowingSpec("slotboom")),
    ],
)
@pytest.mark.parametrize("T", (300.0,) + NON_REFERENCE)
def test_srh_references_keep_mass_action_and_einstein_holds(layer, T):
    x = np.linspace(0.0, 400e-9, 41)
    binding = compile_layer_parameters(x, (layer,), temperature_K=T, temperature_scaling=True, band_grading=True,
                                       dos_band_potentials=True)
    np.testing.assert_allclose(binding.srh_n1_m3 * binding.srh_p1_m3, binding.intrinsic_product_m6, rtol=16 * EPS)
    assert np.array_equal(binding.electron_diffusivity_m2_s, binding.electron_mobility_m2_V_s * binding.thermal_voltage_V)
    assert np.array_equal(binding.hole_diffusivity_m2_s, binding.hole_mobility_m2_V_s * binding.thermal_voltage_V)


def test_graded_intrinsic_product_follows_the_local_gap():
    layer = _layer(grading=GradingSpec(band_gap_back_eV=1.3, profile="linear"))
    x = np.linspace(0.0, 400e-9, 41)
    for T in NON_REFERENCE:
        binding = compile_layer_parameters(x, (layer,), temperature_K=T, temperature_scaling=True, band_grading=True,
                                           dos_band_potentials=True)
        state = binding.layer_states[0]
        gap = binding.band_structure.physical_band_gap_eV
        expected = state.intrinsic_density_m3**2 * np.exp(-(gap - state.band_gap_front_eV) / binding.thermal_voltage_V)
        np.testing.assert_allclose(binding.intrinsic_product_m6, expected, rtol=4 * EPS)
        # Both endpoints carry the same Varshni shift, so the grade itself is temperature independent.
        reference = compile_layer_parameters(x, (layer,), temperature_K=300.0, temperature_scaling=True,
                                             band_grading=True, dos_band_potentials=True)
        shift = laws.band_gap_at_temperature(1.12, T, 4.73e-4, 636.0) - 1.12
        np.testing.assert_allclose(gap - reference.band_structure.physical_band_gap_eV, shift, atol=8 * EPS)


def test_band_gap_narrowing_lowers_the_gap_and_raises_the_affinity():
    plain = compile_layer_parameters(np.linspace(0.0, 400e-9, 21), (_layer(),), temperature_K=320.0,
                                     temperature_scaling=True, band_grading=True, dos_band_potentials=True)
    narrowed = compile_layer_parameters(np.linspace(0.0, 400e-9, 21), (_layer(narrowing=BandGapNarrowingSpec("slotboom")),),
                                        temperature_K=320.0, temperature_scaling=True, band_grading=True,
                                        dos_band_potentials=True)
    dEg = narrowed.band_gap_narrowing_eV
    assert np.all(dEg > 0.0)
    np.testing.assert_allclose(plain.band_structure.physical_band_gap_eV - narrowed.band_structure.physical_band_gap_eV,
                               dEg, atol=4 * EPS)
    np.testing.assert_allclose(narrowed.band_structure.physical_electron_affinity_eV
                               - plain.band_structure.physical_electron_affinity_eV, 0.5 * dEg, atol=8 * EPS)
    np.testing.assert_allclose(narrowed.intrinsic_product_m6 / plain.intrinsic_product_m6,
                               np.exp(dEg / narrowed.thermal_voltage_V), rtol=8 * EPS)


def test_doping_profile_forward_and_edge_values():
    spec = DopingProfileSpec(1e25, 0.0, 1e21, None, "gaussian", 50e-9, "front")
    assert laws.doping_at_position(spec, 0.0, 400e-9)[0] == 1e25
    deep = laws.doping_at_position(spec, 400e-9, 400e-9)[0]
    assert deep == pytest.approx(1e21, rel=1e-12)
    one_e = laws.doping_at_position(spec, 50e-9, 400e-9)[0]
    assert one_e == pytest.approx(1e21 + (1e25 - 1e21) * np.exp(-1.0), rel=1e-15)
    back = DopingProfileSpec(1e25, 0.0, 1e21, None, "gaussian", 50e-9, "back")
    assert laws.doping_at_position(back, 400e-9, 400e-9)[0] == 1e25

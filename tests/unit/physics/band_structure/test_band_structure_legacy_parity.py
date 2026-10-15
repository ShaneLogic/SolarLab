"""M-class parity: P04-11 band-structure laws and binding against the legacy engine.

Bitwise comparisons only (``struct`` bits for scalars, ``tobytes`` for arrays);
the legacy package is imported here as the frozen reference.
"""

from __future__ import annotations

import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest

import band_structure_test_support as support
from perovskite_sim.physics import band_gap_narrowing as legacy_bgn
from perovskite_sim.physics import doping as legacy_doping
from perovskite_sim.physics import grading as legacy_grading
from perovskite_sim.physics import temperature as legacy_temperature
from solarlab.physics import band_structure as new
from solarlab.physics.band_structure import temperature as new_temperature

TEMPERATURES = (150.0, 250.0, 299.99999, 300.0, 300.00001, 320.0, 350.0, 450.0)


def _bits(value: float) -> bytes:
    return struct.pack("<d", float(value))


# --- temperature laws ---------------------------------------------------------


@pytest.mark.parametrize("T", TEMPERATURES)
@pytest.mark.parametrize(
    "ni, gap, Nc, Nv",
    [(1.0e16, 1.12, 2.8e25, 1.04e25), (8.05e10, 1.6, None, None), (3.0e12, 1.55, 2.2e24, None), (1e16, 1.12, 0.0, 1e25)],
)
def test_intrinsic_density_law_is_bitwise_legacy(T, ni, gap, Nc, Nv):
    assert _bits(new.intrinsic_density_at_temperature(ni, gap, T, Nc, Nv)) == _bits(
        legacy_temperature.ni_at_T(ni, gap, T, Nc, Nv)
    )


@pytest.mark.parametrize("T", TEMPERATURES)
def test_power_and_arrhenius_laws_are_bitwise_legacy(T):
    for mu, gamma in ((2e-4, -1.5), (1e-2, 0.0), (3.3e-5, 2.1), (0.0, -1.5)):
        assert _bits(new.mobility_at_temperature(mu, T, gamma)) == _bits(legacy_temperature.mu_at_T(mu, T, gamma))
    for D, Ea in ((1e-16, 0.58), (0.0, 0.58), (3e-17, 0.0), (2e-15, -0.2)):
        assert _bits(new.ion_diffusivity_at_temperature(D, T, Ea)) == _bits(legacy_temperature.D_ion_at_T(D, T, Ea))
    for B, gamma in ((4e-17, -1.5), (4e-17, 0.0), (0.0, -1.5), (1e-16, 0.7)):
        assert _bits(new.radiative_coefficient_at_temperature(B, T, gamma)) == _bits(
            legacy_temperature.B_rad_at_T(B, T, gamma)
        )


@pytest.mark.parametrize("T", TEMPERATURES + (250.0, 636.0))
@pytest.mark.parametrize("alpha, beta", [(4.73e-4, 636.0), (-3e-4, 200.0), (0.0, 0.0), (1e-4, -300.0), (1e-4, -250.0)])
def test_varshni_law_is_bitwise_legacy(T, alpha, beta):
    for gap in (1.12, 1.6):
        assert _bits(new.band_gap_at_temperature(gap, T, alpha, beta)) == _bits(
            legacy_temperature.eg_at_T(gap, T, alpha, beta)
        )


@pytest.mark.parametrize("T", TEMPERATURES)
def test_inline_material_build_laws_are_the_historical_expressions(T):
    """DOS, thermal velocity and SRH scale: the expressions inlined in the legacy build."""

    T_REF = legacy_temperature.T_REF
    for dos in (2.8e25, 1.04e25, 3.0e23):
        scale = (T / T_REF) ** 1.5 if T != T_REF else 1.0
        assert _bits(new.band_edge_dos_at_temperature(dos, T)) == _bits(float(dos) * scale)
    for v in (1e5, 2.3e5):
        scale = (T / T_REF) ** 0.5 if T != T_REF else 1.0
        assert _bits(new.thermal_velocity_at_temperature(v, T)) == _bits(float(v) * scale)
    ni_T = legacy_temperature.ni_at_T(1e16, 1.12, T, 2.8e25, 1.04e25)
    expected = ni_T / 1e16 if T != T_REF else 1.0
    assert _bits(new.srh_reference_scale(ni_T, 1e16, T)) == _bits(expected)
    assert new.srh_reference_scale(ni_T, 0.0, T) == 1.0


def test_device_temperature_selects_the_historical_tier_values():
    from perovskite_sim.constants import V_T as legacy_reference_thermal_voltage

    assert new.device_temperature(350.0, temperature_scaling=False) == (300.0, legacy_reference_thermal_voltage)
    T, V_T = new.device_temperature(350.0, temperature_scaling=True)
    assert T == 350.0 and _bits(V_T) == _bits(legacy_temperature.thermal_voltage(350.0))
    assert _bits(new.device_temperature(300.0, temperature_scaling=True)[1]) == _bits(legacy_reference_thermal_voltage)
    assert new_temperature.T_REF == legacy_temperature.T_REF


# --- band-gap narrowing -------------------------------------------------------


@pytest.mark.parametrize("density", [0.0, 1e-300, 1.0, 1e10, 9.99e22, 1e23, 1.0001e23, 1e26])
@pytest.mark.parametrize("params", [{}, {"reference_energy_eV": 0.0045, "reference_density_m3": 3e22, "log_shape": 0.3}])
def test_slotboom_law_is_bitwise_legacy(density, params):
    assert _bits(new.slotboom_band_gap_narrowing(density, **params)) == _bits(
        legacy_bgn.slotboom_band_gap_narrowing(density, **params)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(electron_affinity_eV=4.05, band_gap_eV=1.12, acceptor_density_m3=1e25, donor_density_m3=0.0, model="slotboom"),
        dict(electron_affinity_eV=4.05, band_gap_eV=1.12, acceptor_density_m3=3e23, donor_density_m3=2e24,
             model="Slotboom", conduction_band_fraction=0.3),
        dict(electron_affinity_eV=3.9, band_gap_eV=1.6, acceptor_density_m3=1e20, donor_density_m3=1e20, model="off"),
    ],
)
def test_band_gap_narrowing_state_is_bitwise_legacy(kwargs):
    mine = new.apply_band_gap_narrowing(**kwargs)
    old = legacy_bgn.apply_band_gap_narrowing(**kwargs)
    for name in old.__dataclass_fields__:
        mine_value, old_value = getattr(mine, name), getattr(old, name)
        assert (mine_value == old_value) and (not isinstance(old_value, float) or _bits(mine_value) == _bits(old_value))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(electron_affinity_eV=4.0, band_gap_eV=1.1, acceptor_density_m3=1e20, donor_density_m3=0.0, model="off",
             log_shape=0.4),
        dict(electron_affinity_eV=4.0, band_gap_eV=0.05, acceptor_density_m3=1e27, donor_density_m3=0.0, model="slotboom"),
        dict(electron_affinity_eV=4.0, band_gap_eV=1.1, acceptor_density_m3=1e20, donor_density_m3=0.0, model="jain"),
        dict(electron_affinity_eV=4.0, band_gap_eV=1.1, acceptor_density_m3=-1.0, donor_density_m3=0.0, model="slotboom"),
        dict(electron_affinity_eV=4.0, band_gap_eV=1.1, acceptor_density_m3=1e20, donor_density_m3=0.0, model="slotboom",
             conduction_band_fraction=1.5),
    ],
)
def test_band_gap_narrowing_rejections_match_legacy(kwargs):
    with pytest.raises(ValueError) as old_error:
        legacy_bgn.apply_band_gap_narrowing(**kwargs)
    with pytest.raises(ValueError) as new_error:
        new.apply_band_gap_narrowing(**kwargs)
    assert str(new_error.value) == str(old_error.value)


# --- grading --------------------------------------------------------------------


_X = np.concatenate([np.linspace(0.0, 500e-9, 41), [-1e-9, 501e-9, 1e-15, 499.999e-9]])


@pytest.mark.parametrize("profile", ["linear", "parabolic", "exponential", "unknown"])
@pytest.mark.parametrize("length", [None, 0.0, -5e-9, 80e-9, 1e-3, 1e-300])
@pytest.mark.parametrize("direction", ["front_to_back", "back_to_front"])
def test_grading_coordinate_is_bitwise_legacy(profile, length, direction):
    assert support.same(
        new.grading_coordinate(_X, 500e-9, profile, length, direction),
        legacy_grading.grading_coordinate(_X, 500e-9, profile, length, direction),
    )


@pytest.mark.parametrize("front, back, bowing", [(1.2, 1.2, 0.0), (1.04, 1.68, 0.2), (1.6, 1.1, -0.15), (1.3, 1.3, 0.4)])
def test_grading_material_laws_are_bitwise_legacy(front, back, bowing):
    y = legacy_grading.grading_coordinate(_X, 500e-9, "parabolic")
    assert support.same(new.band_gap_profile(y, front, back, bowing), legacy_grading.band_gap_profile(y, front, back, bowing))
    assert support.same(new.affinity_profile(y, 4.1 * front, 4.1 * back), legacy_grading.affinity_profile(y, 4.1 * front, 4.1 * back))
    assert _bits(new.minimum_band_gap_eV(front, back, bowing)) == _bits(legacy_grading.minimum_band_gap_eV(front, back, bowing))
    Eg = legacy_grading.band_gap_profile(y, front, back, bowing)
    for V_T in (0.025851999786435535, 0.03016):
        assert support.same(new.grade_ni_sq(1e32, Eg, front, V_T), legacy_grading.grade_ni_sq(1e32, Eg, front, V_T))
        for mine, old in zip(new.grade_n1_p1(3e15, 7e16, Eg, front, V_T), legacy_grading.grade_n1_p1(3e15, 7e16, Eg, front, V_T)):
            assert support.same(mine, old)
    assert _bits(new.minimum_band_gap_eV(front)) == _bits(legacy_grading.minimum_band_gap_eV(front))


def test_grading_presence_rule_is_the_legacy_one():
    for back_gap, back_chi in ((None, None), (1.5, None), (None, 3.9), (1.2, 4.0)):
        legacy_params = SimpleNamespace(Eg_back=back_gap, chi_back=back_chi)
        assert new.GradingSpec(back_gap, back_chi).graded is legacy_grading.has_grading_params(legacy_params)


# --- doping profiles ------------------------------------------------------------


def _legacy_params(**kwargs):
    defaults = dict(N_A=0.0, N_D=0.0, N_A_bulk=None, N_D_bulk=None, doping_profile_shape=None,
                    doping_decay_length=None, doping_edge="front")
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _spec(params):
    return new.DopingProfileSpec(params.N_A, params.N_D, params.N_A_bulk, params.N_D_bulk,
                                 params.doping_profile_shape, params.doping_decay_length, params.doping_edge)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(N_A=1e22, N_D=3e20),
        dict(N_A=1e25, N_A_bulk=1e21, doping_profile_shape="gaussian", doping_decay_length=3.3e-7),
        dict(N_D=4e24, N_D_bulk=0.0, doping_profile_shape="gaussian", doping_decay_length=1e-8, doping_edge="back"),
        dict(N_A=2e23, N_D=5e24, N_A_bulk=0.0, N_D_bulk=1e20, doping_profile_shape="gaussian", doping_decay_length=7e-8),
    ],
)
def test_doping_profiles_are_bitwise_legacy(kwargs):
    params = _legacy_params(**kwargs)
    x = np.linspace(0.0, 4e-7, 57)
    mine = new.layer_doping_profiles(x, 4e-7, _spec(params))
    old = legacy_doping.layer_doping_profiles(x, 4e-7, params)
    assert support.same(mine[0], old[0]) and support.same(mine[1], old[1])
    for position in (0.0, 1.7e-7, 4e-7):
        assert new.doping_at_position(_spec(params), position, 4e-7) == legacy_doping.doping_at_position(params, position, 4e-7)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(doping_profile_shape="gaussian"),
        dict(doping_decay_length=1e-8),
        dict(N_A_bulk=1e20, doping_profile_shape="erfc", doping_decay_length=1e-8),
        dict(N_A_bulk=1e20, doping_profile_shape="gaussian", doping_decay_length=1e-8, doping_edge="middle"),
        dict(N_A_bulk=1e20, doping_profile_shape="gaussian", doping_decay_length=0.0),
        dict(N_A_bulk=1e20, doping_profile_shape="gaussian", doping_decay_length=None),
        dict(N_A=-1.0, N_A_bulk=1e20, doping_profile_shape="gaussian", doping_decay_length=1e-8),
        dict(N_D_bulk=math.inf, doping_profile_shape="gaussian", doping_decay_length=1e-8),
    ],
)
def test_doping_profile_rejections_match_legacy(kwargs):
    params = _legacy_params(**kwargs)
    with pytest.raises(ValueError) as old_error:
        legacy_doping.validate_doping_profile_params(params)
    with pytest.raises(ValueError) as new_error:
        _spec(params)
    assert str(new_error.value) == str(old_error.value)


# --- the whole binding against the legacy material build --------------------------

PRESETS = (
    "nip_MAPbI3.yaml",                       # no DOS: scaled ni branch
    "ionmonger_benchmark.yaml",
    "cigs_graded_notch.yaml",                # band grading
    "csi_incomplete_ionization_bgn_pn_research.yaml",  # BGN + Varshni + DOS (research lane)
    "csi_vannijen2025_pn_cv.yaml",           # Gaussian doping profile, T = 298.15 K
    "driftfusion_calado2016_repro.yaml",     # LEGACY tier: scaling, grading and fold off
    "dual_ion_demo.yaml",                    # negative-ion diffusivity
    "field_mobility_demo.yaml",
    "scaps_mirror_v2.yaml",                  # DOS fold across four DOS-bearing layers
)


def _pairs(binding, mat):
    band = binding.band_structure
    pairs = {
        "chi": (band.transport_electron_affinity_eV, mat.chi),
        "Eg": (band.transport_band_gap_eV, mat.Eg),
        "chi_phys": (band.physical_electron_affinity_eV, mat.chi_phys),
        "Eg_phys": (band.physical_band_gap_eV, mat.Eg_phys),
        "ni_sq": (binding.intrinsic_product_m6, mat.ni_sq),
        "n1": (binding.srh_n1_m3, mat.n1),
        "p1": (binding.srh_p1_m3, mat.p1),
        "B_rad": (binding.radiative_coefficient_m3_s, mat.B_rad),
        "D_n_node": (binding.electron_diffusivity_m2_s, mat.D_n_node),
        "D_p_node": (binding.hole_diffusivity_m2_s, mat.D_p_node),
        "D_ion_node": (binding.ion_diffusivity_m2_s, mat.D_ion_node),
        "N_A": (binding.acceptor_density_m3, mat.N_A),
        "N_D": (binding.donor_density_m3, mat.N_D),
        "N_C_physical": (binding.conduction_dos_m3, mat.N_C_physical),
        "N_V_physical": (binding.valence_dos_m3, mat.N_V_physical),
        "v_th_physical": (binding.thermal_velocity_m_s, mat.v_th_physical),
    }
    if mat.band_gap_narrowing_eV is not None:
        pairs["band_gap_narrowing_eV"] = (binding.band_gap_narrowing_eV, mat.band_gap_narrowing_eV)
    if mat.D_ion_neg_node is not None:
        pairs["D_ion_neg_node"] = (binding.negative_ion_diffusivity_m2_s, mat.D_ion_neg_node)
    return pairs


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("T", [300.0, 320.0, 350.0])
def test_binding_is_bitwise_the_legacy_material_build(name, T):
    stack = support.load_stack(name, temperature_K=T)
    for n_grid in (31, 60):
        x = support.electrical_grid(stack, n_grid)
        mat = support.legacy_material_arrays(x, stack)
        binding = support.binding_for(stack, x)
        mismatched = [key for key, (mine, old) in _pairs(binding, mat).items() if not support.same(mine, old)]
        assert not mismatched, (name, T, n_grid, mismatched)
        assert _bits(binding.temperature_K) == _bits(mat.T_device)
        assert _bits(binding.thermal_voltage_V) == _bits(mat.V_T_device)


def test_preset_declared_temperature_is_honoured():
    stack = support.load_stack("csi_vannijen2025_pn_cv.yaml")
    assert stack.T == 298.15
    x = support.electrical_grid(stack, 60)
    mat = support.legacy_material_arrays(x, stack)
    binding = support.binding_for(stack, x)
    assert binding.temperature_K == 298.15
    assert all(support.same(mine, old) for mine, old in _pairs(binding, mat).values())

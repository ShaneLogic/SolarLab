"""D7-E3 local metastable configuration closure."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.models.defects import BulkDefectKinetics
from perovskite_sim.models.multivalent_defects import (
    DOUBLE_ELECTRON_CAPTURE,
    DOUBLE_HOLE_CAPTURE,
    ELECTRON_CAPTURE_HOLE_EMISSION,
    HOLE_CAPTURE_ELECTRON_EMISSION,
    MetastableConversionKinetics,
    MetastableDefectDefinition,
    MultivalentDefectConfiguration,
    MultivalentEnergyLevels,
)
from perovskite_sim.physics.metastable_defect_closure import (
    MetastableConfigurationClosureError,
    evaluate_metastable_configuration_closure,
)
from perovskite_sim.physics.temperature import thermal_voltage


TEMPERATURE_K = 300.0
GAP_EV = 0.80
NC_M3 = 1.0e24
NV_M3 = 8.0e23
TRANSITION_EV = 0.35
THERMAL_V = thermal_voltage(TEMPERATURE_K)


def _kinetics() -> BulkDefectKinetics:
    return BulkDefectKinetics(
        sigma_n_m2=2.0e-19,
        sigma_p_m2=7.0e-20,
        thermal_velocity_n_m_s=1.0e5,
        thermal_velocity_p_m_s=8.0e4,
    )


def _configuration(family: str, charges: tuple[int, ...]):
    return MultivalentDefectConfiguration(
        family=family,
        charge_states_e=charges,
        degeneracy_convention="unity",
        state_degeneracies=(1.0,) * len(charges),
        energy_levels=MultivalentEnergyLevels(
            first_transition_eV_above_vb=0.30,
            correlation_energies_eV=(0.15,),
        ),
        transition_kinetics=(_kinetics(), _kinetics()),
    )


def _definition(
    *,
    electron_path: str = DOUBLE_ELECTRON_CAPTURE,
    hole_path: str = DOUBLE_HOLE_CAPTURE,
    electron_capture_eV: float = 0.20,
    hole_capture_eV: float = 0.25,
    phonon_frequency_Hz: float = 1.0e15,
    capture_n_m3_s: float = 1.0e-15,
    capture_p_m3_s: float = 1.0e-15,
) -> MetastableDefectDefinition:
    electron_emission = electron_capture_eV + (
        2.0 * (GAP_EV - TRANSITION_EV)
        if electron_path == DOUBLE_ELECTRON_CAPTURE
        else GAP_EV - 2.0 * TRANSITION_EV
    )
    hole_emission = hole_capture_eV + (
        2.0 * TRANSITION_EV
        if hole_path == DOUBLE_HOLE_CAPTURE
        else 2.0 * TRANSITION_EV - GAP_EV
    )
    return MetastableDefectDefinition(
        name="metastable_center",
        total_density_m3=2.0e21,
        donor_configuration=_configuration("double_donor", (2, 1, 0)),
        acceptor_configuration=_configuration("double_acceptor", (0, -1, -2)),
        donor_conversion_state_index=1,
        acceptor_conversion_state_index=1,
        conversion_kinetics=MetastableConversionKinetics(
            transition_energy_eV_above_vb=TRANSITION_EV,
            electron_capture_activation_eV=electron_capture_eV,
            electron_emission_activation_eV=electron_emission,
            hole_capture_activation_eV=hole_capture_eV,
            hole_emission_activation_eV=hole_emission,
            electron_capture_path=electron_path,
            hole_capture_path=hole_path,
            capture_n_m3_s=capture_n_m3_s,
            capture_p_m3_s=capture_p_m3_s,
            phonon_frequency_Hz=phonon_frequency_Hz,
        ),
    )


def _evaluate(n, p, definition):
    return evaluate_metastable_configuration_closure(
        n,
        p,
        definition,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )


@pytest.mark.parametrize(
    "electron_path", [DOUBLE_ELECTRON_CAPTURE, ELECTRON_CAPTURE_HOLE_EMISSION]
)
@pytest.mark.parametrize(
    "hole_path", [DOUBLE_HOLE_CAPTURE, HOLE_CAPTURE_ELECTRON_EMISSION]
)
def test_every_pathway_reproduces_the_two_electron_boltzmann_law(
    electron_path,
    hole_path,
):
    """Detailed balance, measured rather than assumed.

    The configuration change moves two electrons, so at thermal equilibrium
    the donor fraction must be 1 / (1 + exp(2 (F - E_t) / V_T)) whatever
    pathway pair carries it. This is the check that ties the schema's frozen
    barrier relations to the carrier activities this closure uses; a wrong
    activity factor breaks it by orders of magnitude.
    """
    definition = _definition(electron_path=electron_path, hole_path=hole_path)
    fermi_level = np.linspace(0.15, 0.65, 41)
    n = NC_M3 * np.exp(-(GAP_EV - fermi_level) / THERMAL_V)
    p = NV_M3 * np.exp(-fermi_level / THERMAL_V)

    result = _evaluate(n, p, definition)

    expected = 1.0 / (1.0 + np.exp(2.0 * (fermi_level - TRANSITION_EV) / THERMAL_V))
    np.testing.assert_allclose(
        result.donor_fraction,
        expected,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.donor_fraction + result.acceptor_fraction,
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_configuration_densities_share_one_total_density():
    definition = _definition()
    n = np.geomspace(1.0e14, 1.0e20, 7)
    p = np.geomspace(1.0e19, 1.0e13, 7)

    result = _evaluate(n, p, definition)

    np.testing.assert_allclose(
        result.donor_density_m3 + result.acceptor_density_m3,
        definition.total_density_m3,
        rtol=1.0e-15,
        atol=0.0,
    )
    assert result.maximum_stationary_residual_s1 < 1.0e-6 * result.maximum_rate_s1


def test_analytic_configuration_tangent_matches_a_resolvable_difference():
    definition = _definition()
    # Stay off the saturated tails: where the fraction is pinned at 0 or 1 the
    # derivative is ~1e-23 and a finite difference is pure cancellation noise.
    n = np.geomspace(1.0e16, 1.0e18, 5)
    p = 2.9e34 / n

    result = _evaluate(n, p, definition)
    step = 1.0e-5
    forward_n = _evaluate(n * (1.0 + step), p, definition).donor_fraction
    backward_n = _evaluate(n * (1.0 - step), p, definition).donor_fraction
    forward_p = _evaluate(n, p * (1.0 + step), definition).donor_fraction
    backward_p = _evaluate(n, p * (1.0 - step), definition).donor_fraction

    np.testing.assert_allclose(
        result.donor_fraction_derivative_n_m3,
        (forward_n - backward_n) / (2.0 * step * n),
        rtol=1.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.donor_fraction_derivative_p_m3,
        (forward_p - backward_p) / (2.0 * step * p),
        rtol=1.0e-6,
        atol=0.0,
    )


def test_electron_rich_and_hole_rich_limits_select_opposite_configurations():
    definition = _definition()
    intrinsic = math.sqrt(NC_M3 * NV_M3 * math.exp(-GAP_EV / THERMAL_V))

    electron_rich = _evaluate(
        np.array([1.0e3 * intrinsic]),
        np.array([1.0e-3 * intrinsic]),
        definition,
    )
    hole_rich = _evaluate(
        np.array([1.0e-3 * intrinsic]),
        np.array([1.0e3 * intrinsic]),
        definition,
    )

    # Capturing electrons converts donor -> acceptor, so an electron-rich
    # working point must empty the donor configuration and vice versa.
    assert electron_rich.donor_fraction[0] < hole_rich.donor_fraction[0]
    assert electron_rich.acceptor_fraction[0] > 0.5
    assert hole_rich.donor_fraction[0] > 0.5


def test_rate_above_the_phonon_attempt_frequency_fails_closed():
    """A lattice reconfiguration cannot outrun its own attempt frequency.

    The declared attempt frequency here is deliberately far below the
    capture-driven rate the same document implies, which is exactly the
    inconsistent input the guard exists to reject.
    """
    definition = _definition(phonon_frequency_Hz=1.0e-4)
    n = np.array([1.0e20])
    p = np.array([1.0e14])

    with pytest.raises(
        MetastableConfigurationClosureError,
        match="phonon attempt frequency",
    ):
        _evaluate(n, p, definition)


def test_non_positive_carriers_and_wrong_types_fail_closed():
    definition = _definition()

    with pytest.raises(ValueError, match="finite and positive"):
        _evaluate(np.array([0.0]), np.array([1.0e16]), definition)
    with pytest.raises(ValueError, match="finite and positive"):
        _evaluate(np.array([1.0e16]), np.array([-1.0e16]), definition)
    with pytest.raises(TypeError, match="MetastableDefectDefinition"):
        _evaluate(np.array([1.0e16]), np.array([1.0e16]), object())


def test_results_are_immutable_and_identity_binds_the_definition():
    definition = _definition()
    denser = _definition(electron_capture_eV=0.22)
    n = np.array([1.0e16, 1.0e17])
    p = np.array([1.0e17, 1.0e16])

    result = _evaluate(n, p, definition)
    other = _evaluate(n, p, denser)

    assert not result.donor_fraction.flags.writeable
    assert not result.donor_to_acceptor_rate_s1.flags.writeable
    assert result.closure_identity_sha256 != other.closure_identity_sha256
    assert len(result.closure_identity_sha256) == 64

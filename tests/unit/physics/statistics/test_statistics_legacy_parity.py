"""Carrier statistics: one MB/restricted-FD definition (M class) and the FD MB limit.

M class: every value is compared with the legacy engine (read-only import) by
``tobytes()`` on the same inputs; error types and messages are compared too.
Acceptance (V1 P04-03): Fermi-Dirac reduces to Maxwell-Boltzmann in the
non-degenerate limit -- bitwise for ``eta <= -40`` and within the table's own
interpolation/rounding error above.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from perovskite_sim import constants as legacy_constants
from perovskite_sim.physics import fermi_dirac as legacy_fd
from perovskite_sim.physics import statistics as legacy_statistics
from perovskite_sim.physics import temperature as legacy_temperature

from solarlab.physics import statistics as statistics
from solarlab.physics.statistics import (
    FERMI_DIRAC, MAXWELL_BOLTZMANN, fermi_dirac_half, fermi_dirac_half_log_derivative,
    fermi_dirac_half_table, generalized_einstein_factor, inverse_fermi_dirac_half,
    mass_action_carrier_densities, reduced_fermi_level_from_density, thermal_voltage,
)

ETAS = np.concatenate((
    np.linspace(-60.0, 40.0, 4001),
    np.linspace(-40.0, 20.0, 241),            # table nodes (step 0.25 = 50 table steps)
    [-40.0, 20.0, np.nextafter(-40.0, 0.0), np.nextafter(20.0, 0.0), 0.0, -0.0, 1e-300,
     -1e-300, -40.000000001, 19.999999999, -39.995, 19.995, 5e-3, -5e-3],
))
RATIOS = np.concatenate((np.logspace(-30.0, 12.0, 3001), [0.0, 1.0, math.inf, 4.25e-18,
                                                        4.25e-18 * (1 + 1e-15)]))


def same(a, b):
    return np.asarray(a, dtype=float).tobytes() == np.asarray(b, dtype=float).tobytes()


def same_outcome(new, old, *args, **kwargs):
    """Equal value bytes, or the same exception type and message."""
    try:
        expected = old(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - compare the legacy failure exactly
        with pytest.raises(type(error)) as caught:
            new(*args, **kwargs)
        return str(caught.value) == str(error)
    return repr(new(*args, **kwargs)) == repr(expected)


def test_thermal_voltage_and_reference_constants_match_the_legacy_engine():
    assert statistics.REFERENCE_THERMAL_VOLTAGE_V == legacy_constants.V_T
    assert statistics.BOLTZMANN_CONSTANT_J_K == legacy_constants.K_B
    for temperature in (300.0, 250.0, 333.3, 77, 1e-3, 1.0e4):
        assert same(thermal_voltage(temperature), legacy_temperature.thermal_voltage(temperature))
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            thermal_voltage(bad)


def test_fermi_dirac_table_is_the_legacy_table_bitwise():
    for new, old in zip(fermi_dirac_half_table(), legacy_fd._half_table()):
        assert same(new, old)
    eta, values, log_values = fermi_dirac_half_table()
    assert not values.flags.writeable and not log_values.flags.writeable


def test_fermi_dirac_half_family_matches_the_legacy_engine_bitwise():
    for eta in ETAS:
        assert same(fermi_dirac_half(eta), legacy_fd.fermi_dirac_half(eta))
        assert same(fermi_dirac_half_log_derivative(eta),
                    legacy_fd.fermi_dirac_half_log_derivative(eta))
    for special in (math.nan, math.inf, -math.inf):
        assert same(fermi_dirac_half(special), legacy_fd.fermi_dirac_half(special))
        assert same(fermi_dirac_half_log_derivative(special),
                    legacy_fd.fermi_dirac_half_log_derivative(special))
    for ratio in RATIOS:
        assert same(inverse_fermi_dirac_half(ratio), legacy_fd.inverse_fermi_dirac_half(ratio))
    for bad in (-1.0, math.nan):
        assert same_outcome(inverse_fermi_dirac_half, legacy_fd.inverse_fermi_dirac_half, bad)


@pytest.mark.parametrize("model", [MAXWELL_BOLTZMANN, FERMI_DIRAC, " Fermi_Dirac "])
def test_carrier_maps_share_one_definition_and_match_the_legacy_engine(model):
    for eta in np.concatenate((np.linspace(-50.0, 30.0, 801), [-745.5, 709.0, 710.0])):
        for name in ("carrier_occupation", "carrier_logarithmic_compressibility",
                     "generalized_einstein_factor"):
            assert same_outcome(getattr(statistics, name), getattr(legacy_statistics, name),
                                eta, statistics=model)
        for dos in (1.0e24, 2.5e26):
            for name in ("carrier_density_from_reduced_fermi_level",
                         "carrier_density_derivative_reduced_fermi_level"):
                assert same_outcome(getattr(statistics, name),
                                    getattr(legacy_statistics, name), eta, dos,
                                    statistics=model)
    for density in np.concatenate((np.logspace(-10.0, 28.0, 400), [0.0])):
        assert same_outcome(statistics.reduced_fermi_level_from_density,
                            legacy_statistics.reduced_fermi_level_from_density,
                            density, 1.0e25, statistics=model)


@pytest.mark.parametrize("call", [
    ("carrier_occupation", (math.nan,), {}),
    ("carrier_occupation", (0.0,), {"statistics": "blakemore"}),
    ("carrier_occupation", (0.0,), {"statistics": 3}),
    ("carrier_logarithmic_compressibility", (math.inf,), {}),
    ("carrier_density_from_reduced_fermi_level", (0.0, 0.0), {}),
    ("reduced_fermi_level_from_density", (-1.0, 1e25), {}),
    ("reduced_fermi_level_from_density", (1.0, math.inf), {}),
])
def test_carrier_map_rejections_match_the_legacy_messages(call):
    name, args, kwargs = call
    assert same_outcome(getattr(statistics, name), getattr(legacy_statistics, name),
                        *args, **kwargs)


def test_dopant_ionization_matches_the_legacy_engine_bitwise():
    rng = np.random.default_rng(3)
    for _ in range(400):
        eta_n, eta_p = rng.uniform(-60.0, 40.0, size=2)
        binding = float(rng.choice([0.0, 0.005, 0.05, 0.3]))
        thermal = float(rng.choice([0.0258, 0.01, 0.04]))
        for name, degeneracy in (("ionized_donor_fraction", 2.0),
                                 ("ionized_acceptor_fraction", 4.0)):
            assert same_outcome(getattr(statistics, name), getattr(legacy_statistics, name),
                                eta_n if "donor" in name else eta_p,
                                binding_energy_eV=binding, thermal_voltage_V=thermal,
                                degeneracy=degeneracy)
        for model in ("fully_ionized", "discrete_level"):
            kwargs = dict(reduced_electron_fermi_level=eta_n, reduced_hole_fermi_level=eta_p,
                          donor_density_m3=float(rng.choice([0.0, 1e22, 1e26])),
                          acceptor_density_m3=float(rng.choice([0.0, 3e21, 2e25])),
                          thermal_voltage_V=thermal, model=model)
            if model == "discrete_level":
                kwargs.update(donor_binding_energy_eV=0.045, acceptor_binding_energy_eV=0.12)
            assert same_outcome(statistics.dopant_charge_state,
                                legacy_statistics.dopant_charge_state, **kwargs)
    for bad in ({"model": "fully_ionized", "donor_binding_energy_eV": 0.1},
                {"model": "discrete_level", "donor_density_m3": 1e20},
                {"model": "partial"}):
        kwargs = dict(reduced_electron_fermi_level=0.0, reduced_hole_fermi_level=0.0,
                      donor_density_m3=0.0, acceptor_density_m3=0.0, thermal_voltage_V=0.0258)
        kwargs.update(bad)
        assert same_outcome(statistics.dopant_charge_state,
                            legacy_statistics.dopant_charge_state, **kwargs)


NEUTRALITY_CASES = list(itertools.product(
    (MAXWELL_BOLTZMANN, FERMI_DIRAC), (300.0, 250.0, 400.0), (0.0, 0.5, 1.12, 1.55, 3.4),
    (0.0, 1e15, 1e22, 1e26), (0.0, 1e16, 1e23, 3e26),
))


@pytest.mark.parametrize("ionization", ["fully_ionized", "discrete_level"])
def test_charge_neutrality_matches_the_legacy_engine_bitwise(ionization):
    for model, temperature, gap, acceptors, donors in NEUTRALITY_CASES:
        kwargs = dict(temperature_K=temperature, band_gap_eV=gap,
                      effective_conduction_dos_m3=2.2e24, effective_valence_dos_m3=1.8e25,
                      acceptor_density_m3=acceptors, donor_density_m3=donors,
                      statistics=model, dopant_ionization_model=ionization)
        if ionization == "discrete_level":
            kwargs.update(donor_binding_energy_eV=0.05 if donors else None,
                          acceptor_binding_energy_eV=0.12 if acceptors else None)
        assert same_outcome(statistics.solve_charge_neutrality,
                            legacy_statistics.solve_charge_neutrality, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"dopant_ionization_model": "fully_ionized", "donor_degeneracy": 3.0},
    {"dopant_ionization_model": "fully_ionized", "acceptor_binding_energy_eV": 0.1},
    {"dopant_ionization_model": "nope"},
    {"temperature_K": -1.0},
    {"band_gap_eV": -0.1},
    {"effective_conduction_dos_m3": 0.0},
])
def test_charge_neutrality_rejections_match_the_legacy_messages(kwargs):
    call = dict(temperature_K=300.0, band_gap_eV=1.1, effective_conduction_dos_m3=1e25,
                effective_valence_dos_m3=1e25, donor_density_m3=1e22)
    call.update(kwargs)
    assert same_outcome(statistics.solve_charge_neutrality,
                        legacy_statistics.solve_charge_neutrality, **call)


def legacy_equilibrium_np(N_D_val, N_A_val, ni_val):
    """Verbatim copy of the closure inside ``solver.mol.build_material_arrays``."""
    net = 0.5 * (N_D_val - N_A_val)
    disc = np.sqrt(net ** 2 + ni_val ** 2)
    if net >= 0:
        n_val = net + disc
        p_val = ni_val ** 2 / n_val
    else:
        p_val = -net + disc
        n_val = ni_val ** 2 / p_val
    return float(n_val), float(p_val)


def test_mass_action_closure_matches_the_historical_contact_closure_bitwise():
    rng = np.random.default_rng(11)
    for _ in range(3000):
        donors = float(10.0 ** rng.uniform(-5.0, 27.0)) * float(rng.integers(0, 2))
        acceptors = float(10.0 ** rng.uniform(-5.0, 27.0)) * float(rng.integers(0, 2))
        ni_sq = float(10.0 ** rng.uniform(-60.0, 40.0))
        ni = np.sqrt(ni_sq)
        expected = legacy_equilibrium_np(np.float64(donors), np.float64(acceptors), ni)
        assert same(mass_action_carrier_densities(donors, acceptors, float(ni)), expected)
    assert same(mass_action_carrier_densities(1e22, 0.0, 0.0), legacy_equilibrium_np(
        np.float64(1e22), np.float64(0.0), np.float64(0.0)))
    with pytest.raises(ValueError, match="compensated"):
        mass_action_carrier_densities(1e20, 1e20, 0.0)


def test_mass_action_closure_is_the_mb_neutrality_solution():
    """Same statistics definition: the closed form and the bisection agree."""
    for donors, acceptors in ((1e23, 0.0), (0.0, 4e22), (2e21, 1e21), (0.0, 0.0)):
        state = statistics.solve_charge_neutrality(
            temperature_K=300.0, band_gap_eV=1.2, effective_conduction_dos_m3=2e24,
            effective_valence_dos_m3=3e25, donor_density_m3=donors, acceptor_density_m3=acceptors)
        ni = math.sqrt(2e24 * 3e25 * math.exp(-1.2 / thermal_voltage(300.0)))
        n, p = mass_action_carrier_densities(donors, acceptors, ni)
        assert n == pytest.approx(state.electron_density_m3, rel=1e-11)
        assert p == pytest.approx(state.hole_density_m3, rel=1e-11)


def test_fermi_dirac_reduces_to_maxwell_boltzmann_in_the_non_degenerate_limit():
    # Bitwise identical at and below the table edge.
    for eta in (-40.0, -45.0, -60.0, -200.0):
        assert same(fermi_dirac_half(eta), math.exp(eta))
        assert same(generalized_einstein_factor(eta, statistics=FERMI_DIRAC), 1.0)
        density = 1.0e25 * math.exp(eta)
        assert same(reduced_fermi_level_from_density(density, 1.0e25, statistics=FERMI_DIRAC),
                    reduced_fermi_level_from_density(density, 1.0e25))
    # Above it: F_1/2 = e^eta (1 - e^eta / 2^1.5 + ...), up to the table's
    # linear-in-log interpolation error h^2/8 |(log F)''| <= 1e-8 e^eta-scale.
    for eta in np.linspace(-39.99, -5.0, 2001):
        series = sum((-1) ** (k + 1) * math.exp(k * eta) / k ** 1.5 for k in range(1, 40))
        assert abs(fermi_dirac_half(eta) / series - 1.0) <= 1.0e-8
        assert abs(inverse_fermi_dirac_half(fermi_dirac_half(eta)) - eta) <= 4.0 * np.spacing(40.0)
        # The Einstein factor is a table secant: |g - 1| is e^eta / 2^1.5 plus the
        # secant's own rounding eps*|log F|/h.
        g = generalized_einstein_factor(eta, statistics=FERMI_DIRAC)
        assert abs(g - 1.0) <= math.exp(eta) + 64.0 * np.finfo(float).eps * 40.0 / 5.0e-3

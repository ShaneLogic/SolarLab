"""The new barrier certificate has explicit physics scope and independent axes."""

from pathlib import Path

import pytest

from perovskite_sim.validation.numerical_certificate import MatrixPoint, load_refinement_registry
from perovskite_sim.validation.resolved_tunnelling_refinement import (
    resolved_tunnelling_protocol,
    run_resolved_intraband_refinement,
)


ROOT = Path(__file__).resolve().parents[3]
LANE = "wkb-resolved-intraband-qf-dc-v2"


def _registry():
    return load_refinement_registry(ROOT / "reproducibility/NumericalRefinementRegistry.yaml", project_root=ROOT)


def test_resolved_reference_declares_independent_refinements_and_a_visible_effect():
    lane = _registry().lane(LANE)
    assert lane.grid_values == (24, 48, 96)
    assert lane.tolerance_factors == (1.0, 0.1, 0.01)
    assert lane.options["energy_quadrature_orders"] == [96, 192, 384]
    gates = {gate.metric: gate for gate in lane.quality_gates}
    assert gates["terminal_effect_fraction"].limit == 0.01
    assert gates["injected_face_count"].operator == "ge"
    assert gates["equilibrium_tunnelling_current_A_m2"].limit > 0.0
    observables = {gate.metric for gate in lane.observables}
    assert {"terminal_tunnelling_effect_A_m2", "intraband_event_flux_m2_s",
            "potential_profile_V", "electron_log_density_profile"} <= observables


def test_historical_definition_keeps_its_original_thresholds_and_retraction():
    old = _registry().lane("wkb-tunnelling-channel-qf-dc-v1")
    gates = {gate.metric: gate for gate in old.quality_gates}
    assert gates["channel_flux_fraction_of_terminal_current"].limit == 0.01
    assert gates["equilibrium_net_flux_m2_s"].limit == 0.0
    assert gates["injected_face_count"].limit == 1.0
    assert any("RETRACTED" in statement for statement in old.limitations)


def test_protocol_records_the_physical_barrier_and_transfer_definition():
    protocol = resolved_tunnelling_protocol(_registry().lane(LANE))
    assert protocol["device_implementation"] == "wkb-tunnelling-channel-device-v2"
    assert protocol["particle_transfer"] == "turning_point_nodal_weights"
    assert protocol["energy_rule"] == "open_gauss_legendre"
    assert protocol["supply_prefactor_m2_s_eV"] == 1e24
    assert "tolerance_factor" not in protocol


@pytest.mark.slow
def test_real_resolved_cell_has_a_conservative_observable_electron_channel():
    measurement = run_resolved_intraband_refinement(_registry().lane(LANE), MatrixPoint(24, 1.0), ROOT)
    quality = {item.name: item.values[0] for item in measurement.quality}
    assert quality["all_states_certified"] == 1.0
    assert quality["contact_thermodynamics_consistent"] == 1.0
    assert quality["terminal_effect_fraction"] > 0.01
    assert quality["particle_balance_relative_error"] < 1e-12
    assert quality["path_current_relative_error"] < 1e-12
    assert quality["injected_face_count"] > 3
    assert quality["maximum_reservoir_occupation"] < 0.05
    assert quality["minimum_reservoir_occupation"] > 1e-12
    assert quality["disabled_bit_identical"] == 1.0

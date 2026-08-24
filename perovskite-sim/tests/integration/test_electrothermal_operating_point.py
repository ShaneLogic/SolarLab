from __future__ import annotations

from perovskite_sim.experiments.electrothermal import (
    ElectrothermalJVProtocol,
    ElectrothermalOperatingPointProtocol,
    solve_electrothermal_operating_point,
)
from perovskite_sim.experiments.external_circuit import ExternalCircuitProtocol
from perovskite_sim.experiments.thermal_balance import LumpedThermalProtocol
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_real_mobile_ion_jv_closes_lumped_electrothermal_root():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    thermal = LumpedThermalProtocol(
        absorbed_optical_power_W_m2=800.0,
        ambient_temperature_K=300.0,
        areal_heat_capacity_J_m2_K=2000.0,
        heat_transfer_coefficient_W_m2_K=20.0,
        emissivity=0.85,
        maximum_temperature_K=380.0,
        steady_power_residual_tolerance_W_m2=0.2,
    )
    circuit = ExternalCircuitProtocol(
        series_resistance_ohm_m2=2.0e-4,
        shunt_resistance_ohm_m2=0.2,
    )
    electrical = ElectrothermalJVProtocol(
        grid_points_per_electrical_layer=10,
        voltage_points_per_branch=6,
        scan_rate_V_s=20.0,
        voltage_max_V=1.2,
        relative_tolerance=1.0e-4,
    )
    operating = ElectrothermalOperatingPointProtocol(
        temperature_absolute_tolerance_K=0.005,
    )

    result = solve_electrothermal_operating_point(
        stack,
        thermal,
        circuit,
        electrical,
        operating,
    )

    assert result.certified
    assert 315.0 < result.operating_temperature_K < 335.0
    assert 150.0 < result.terminal_power_W_m2 < thermal.absorbed_optical_power_W_m2
    assert abs(result.power_balance_residual_W_m2) <= (
        thermal.steady_power_residual_tolerance_W_m2
    )
    assert result.electrical_evaluations >= 3
    assert all(
        evaluation.source_certified
        and evaluation.external_circuit_certified
        for evaluation in result.temperature_evaluations
    )
    assert result.final_external_result.source_experiment_protocol_sha256 == (
        next(
            evaluation.source_experiment_protocol_sha256
            for evaluation in result.temperature_evaluations
            if evaluation.temperature_K == result.operating_temperature_K
        )
    )

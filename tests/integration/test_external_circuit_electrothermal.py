"""Integration and legacy-parity tests for P04-13 external circuit & electrothermal coupling."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.research_1d import optical_generation as optical_example  # noqa: E402
from examples.research_1d.ionic_multilayer import (  # noqa: E402
    prepare_confined_ion_device,
)
from solarlab.config.resolve_device import resolve_device  # noqa: E402
from solarlab.device.compile_1d import compile_device_1d  # noqa: E402
from solarlab.device.inputs import DeviceInput  # noqa: E402
from solarlab.device.resolved import PreparedDevice  # noqa: E402
from solarlab.experiments.electrothermal import (  # noqa: E402
    ElectrothermalCapabilityError,
    ElectrothermalJVProtocol,
    ElectrothermalOperatingPointProtocol,
    solve_electrothermal_operating_point,
)
from solarlab.experiments.external_circuit import (  # noqa: E402
    ExternalACCircuitProtocol,
    ExternalCircuitProtocol,
    apply_external_circuit_steady,
    compose_external_ac_impedance,
    map_external_circuit_branch,
)
from solarlab.experiments.impedance import solve_impedance  # noqa: E402
from solarlab.experiments.jv.steady import run_steady_jv  # noqa: E402
from solarlab.experiments.thermal_balance import (  # noqa: E402
    LumpedThermalProtocol,
    ThermalIntegrationProtocol,
    run_lumped_thermal_transient,
    solve_lumped_thermal_steady_state,
    thermal_power_ledger,
)
from solarlab.solvers.nonlinear.steady_1d import (  # noqa: E402
    SteadyOptions,
    solve_steady_state,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas() -> object:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def illuminated_prepared() -> PreparedDevice:
    return optical_example.prepare_case()


@pytest.fixture(scope="module")
def illuminated_ionic_prepared(illuminated_prepared: PreparedDevice) -> PreparedDevice:
    base_ion = prepare_confined_ion_device()
    doc = illuminated_prepared.to_input().editing_data()
    doc["id"] = "illuminated_confined_ion_electrothermal"
    doc["settings"]["ion_steric_diffusion_only"] = True
    absorber = next(lyr for lyr in doc["layers"] if lyr["role"] == "absorber")
    absorber["parameters"].update(P0=1e24, P_lim=1e27, D_ion=1e-16, E_a_ion=0.58)
    _ = base_ion
    return resolve_device(
        DeviceInput.model_validate(doc),
        illuminated_prepared.defaults,
        illuminated_prepared.resources,
        sources=illuminated_prepared.sources,
    )


def test_external_circuit_and_thermal_balance_identities(
    illuminated_prepared: PreparedDevice,
) -> None:
    compiled = compile_device_1d(illuminated_prepared, intervals_per_layer=(8, 8, 8))
    voltages = np.linspace(0.0, 1.25, 31)
    steady_jv = run_steady_jv(
        compiled, voltages, generation_scale=1.0, incident_power_W_m2=1000.0
    )

    sl_dc = ExternalCircuitProtocol(
        series_resistance_ohm_m2=7.5e-4,
        shunt_resistance_ohm_m2=0.25,
    )
    sl_branch = map_external_circuit_branch(
        steady_jv.voltage_V,
        steady_jv.output_current_A_m2,
        sl_dc,
        incident_power_W_m2=1000.0,
    )
    expected_shunt = steady_jv.voltage_V / 0.25
    expected_j_term = steady_jv.output_current_A_m2 - expected_shunt
    expected_v_drop = expected_j_term * 7.5e-4
    np.testing.assert_allclose(sl_branch.J_shunt, expected_shunt, rtol=1e-15)
    np.testing.assert_allclose(sl_branch.J_terminal, expected_j_term, rtol=1e-15)
    np.testing.assert_allclose(sl_branch.series_voltage_drop_V, expected_v_drop, rtol=1e-15)
    np.testing.assert_allclose(sl_branch.V_terminal, steady_jv.voltage_V - expected_v_drop, rtol=1e-15)
    np.testing.assert_allclose(
        sl_branch.power_terminal_W_m2, sl_branch.V_terminal * sl_branch.J_terminal, rtol=1e-15
    )

    sl_th = LumpedThermalProtocol(
        absorbed_optical_power_W_m2=820.0,
        ambient_temperature_K=298.15,
        areal_heat_capacity_J_m2_K=950.0,
        heat_transfer_coefficient_W_m2_K=18.5,
        emissivity=0.82,
        constant_internal_heat_W_m2=12.0,
        maximum_temperature_K=400.0,
    )
    p_exp = float(np.max(sl_branch.power_terminal_W_m2))

    sl_ss = solve_lumped_thermal_steady_state(
        sl_th, exported_electrical_power_fn=lambda _t: p_exp
    )
    ss_ledger = thermal_power_ledger(
        sl_ss.temperature_K, sl_th, exported_electrical_power_W_m2=p_exp
    )
    assert abs(ss_ledger.net_storage_rate_W_m2) <= 1e-8

    t_eval = np.linspace(0.0, 180.0, 37)
    sl_int = ThermalIntegrationProtocol(method="DOP853")
    sl_tr = run_lumped_thermal_transient(
        sl_th,
        t_eval,
        initial_temperature_K=298.15,
        exported_electrical_power_fn=lambda _t, _temp: p_exp,
        integration=sl_int,
    )
    assert sl_tr.certified
    assert sl_tr.temperature_K[-1] > 298.15
    assert sl_tr.temperature_K[-1] == pytest.approx(sl_ss.temperature_K, rel=5e-2)


def test_external_dc_and_small_signal_ac_on_compiled_1d_device(
    illuminated_prepared: PreparedDevice,
) -> None:
    compiled = compile_device_1d(illuminated_prepared, intervals_per_layer=(8, 8, 8))
    voltages = np.linspace(0.0, 1.25, 26)
    steady_jv = run_steady_jv(
        compiled, voltages, generation_scale=1.0, incident_power_W_m2=1000.0
    )

    dc_prot = ExternalCircuitProtocol(
        series_resistance_ohm_m2=8e-4,
        shunt_resistance_ohm_m2=0.2,
    )
    mapped = apply_external_circuit_steady(steady_jv, dc_prot)
    assert mapped.source_identity == compiled.identity
    assert mapped.branch.metrics.open_circuit_voltage_V is not None
    assert steady_jv.metrics.open_circuit_voltage_V is not None
    assert (
        mapped.branch.metrics.open_circuit_voltage_V
        < steady_jv.metrics.open_circuit_voltage_V
    )
    assert (
        mapped.branch.metrics.maximum_power_W_m2 is not None
        and steady_jv.metrics.maximum_power_W_m2 is not None
        and mapped.branch.metrics.maximum_power_W_m2
        < steady_jv.metrics.maximum_power_W_m2
    )

    dc_point = solve_steady_state(
        compiled, voltage_V=0.6, generation_scale=1.0, options=SteadyOptions()
    )
    freqs = np.array([10.0, 1e3, 1e5, 1e7], dtype=float)
    ac_dev = solve_impedance(compiled, dc_point, freqs)
    y_before = ac_dev.admittance_density_S_m2.copy()

    ac_prot = ExternalACCircuitProtocol(
        series_resistance_ohm_m2=5e-4,
        shunt_resistance_ohm_m2=0.25,
        external_capacitance_F_m2=2e-5,
    )
    ac_ext = compose_external_ac_impedance(ac_dev, ac_prot)

    # Original SmallSignalResult1D and CompiledDevice1D remain completely unmutated
    np.testing.assert_array_equal(ac_dev.admittance_density_S_m2, y_before)
    assert ac_ext.source_identity == compiled.identity

    omega = 2.0 * np.pi * freqs
    expected_z = 5e-4 + 1.0 / (y_before + (1.0 / 0.25) + 1j * omega * 2e-5)
    np.testing.assert_allclose(
        ac_ext.impedance_area_ohm_m2, expected_z, rtol=1e-13, atol=1e-14
    )


def test_electrothermal_operating_point_modes_fresh_rebinding_and_mutual_exclusion(
    illuminated_prepared: PreparedDevice,
) -> None:
    thermal_prot = LumpedThermalProtocol(
        absorbed_optical_power_W_m2=800.0,
        ambient_temperature_K=300.0,
        areal_heat_capacity_J_m2_K=800.0,
        heat_transfer_coefficient_W_m2_K=22.0,
        emissivity=0.8,
        maximum_temperature_K=345.0,
    )
    circuit_prot = ExternalCircuitProtocol(
        series_resistance_ohm_m2=4e-4,
        shunt_resistance_ohm_m2=0.5,
    )
    elec_prot = ElectrothermalJVProtocol(
        intervals_per_layer=10,
        voltage_points_per_branch=25,
        scan_rate_V_s=2.0,
        voltage_max_V=1.22,
        incident_power_W_m2=1000.0,
        driver="steady_jv",
    )

    with pytest.raises(ElectrothermalCapabilityError):
        solve_electrothermal_operating_point(
            illuminated_prepared,
            thermal_prot,
            circuit_prot,
            elec_prot,
            external_temperature_K=320.0,
        )

    res_mpp = solve_electrothermal_operating_point(
        illuminated_prepared,
        thermal_prot,
        circuit_prot,
        elec_prot,
        ElectrothermalOperatingPointProtocol(
            operating_point="sampled_terminal_maximum_power"
        ),
    )
    assert res_mpp.certified
    assert res_mpp.time_domain_temperature_qualified is False
    assert 300.0 < res_mpp.operating_temperature_K < 345.0
    assert res_mpp.terminal_power_W_m2 > 0.0
    assert abs(res_mpp.power_balance_residual_W_m2) <= 1e-6

    rebound_hashes = {ev.rebound_prepared_sha256 for ev in res_mpp.temperature_evaluations}
    compiled_ids = {ev.compiled_identity for ev in res_mpp.temperature_evaluations}
    assert len(rebound_hashes) == res_mpp.electrical_evaluations
    assert len(compiled_ids) == res_mpp.electrical_evaluations
    assert all(ev.fresh_state_initialized for ev in res_mpp.temperature_evaluations)

    res_oc = solve_electrothermal_operating_point(
        illuminated_prepared,
        thermal_prot,
        circuit_prot,
        elec_prot,
        ElectrothermalOperatingPointProtocol(operating_point="open_circuit"),
    )
    assert res_oc.certified
    assert res_oc.terminal_power_W_m2 == 0.0
    assert res_oc.terminal_current_A_m2 == 0.0
    assert res_oc.operating_temperature_K > res_mpp.operating_temperature_K

    res_load = solve_electrothermal_operating_point(
        illuminated_prepared,
        thermal_prot,
        circuit_prot,
        elec_prot,
        ElectrothermalOperatingPointProtocol(
            operating_point="fixed_load",
            load_resistance_ohm_m2=0.005,
        ),
    )
    assert res_load.certified
    assert res_load.terminal_voltage_V == pytest.approx(
        0.005 * res_load.terminal_current_A_m2, rel=1e-9, abs=1e-9
    )
    assert (
        res_mpp.operating_temperature_K
        < res_load.operating_temperature_K
        < res_oc.operating_temperature_K
    )


def test_electrothermal_transient_strict_with_mobile_ions_never_carries_ionic_state(
    illuminated_ionic_prepared: PreparedDevice,
) -> None:
    thermal_prot = LumpedThermalProtocol(
        absorbed_optical_power_W_m2=750.0,
        ambient_temperature_K=300.0,
        heat_transfer_coefficient_W_m2_K=28.0,
        emissivity=0.6,
        maximum_temperature_K=340.0,
    )
    elec_prot = ElectrothermalJVProtocol(
        intervals_per_layer=8,
        voltage_points_per_branch=15,
        scan_rate_V_s=1.0,
        voltage_max_V=1.28,
        substeps_per_point=2,
        incident_power_W_m2=1000.0,
        driver="transient_jv_strict",
    )
    res = solve_electrothermal_operating_point(
        illuminated_ionic_prepared,
        thermal_prot,
        ExternalCircuitProtocol(series_resistance_ohm_m2=2e-4),
        elec_prot,
        ElectrothermalOperatingPointProtocol(
            temperature_absolute_tolerance_K=1e-2,
            power_residual_tolerance_W_m2=0.5,
        ),
    )
    assert res.certified
    assert 300.0 < res.operating_temperature_K < 340.0
    assert len({ev.compiled_identity for ev in res.temperature_evaluations}) == len(
        res.temperature_evaluations
    )

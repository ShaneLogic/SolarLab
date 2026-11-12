"""T2/T4 integration and perovskite_sim parity tests for dark J-V and waveform J-V."""

from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments.dark_jv import run_dark_jv as legacy_run_dark_jv
from perovskite_sim.experiments.waveform_jv import (
    JVWaveform as LegacyJVWaveform,
    run_waveform_jv as legacy_run_waveform_jv,
)
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.dark_jv import DarkJVOptions, run_dark_jv
from solarlab.experiments.waveform_jv import (
    JVWaveform,
    WaveformJVOptions,
    WaveformSegment,
    build_piecewise_waveform,
    run_waveform_jv,
)
from solarlab.reproducibility.sources import SourceDocument
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

ROOT = Path(__file__).resolve().parents[2]


def _import_example(mod_name: str) -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(mod_name)


@pytest.fixture(scope="module", autouse=True)
def blas1() -> object:
    with threadpool_limits(limits=1):
        yield


def _legacy_stack_from_prepared(prepared: PreparedDevice) -> DeviceStack:
    mapping = prepared.to_mapping()
    settings = mapping["settings"]
    layers: list[LayerSpec] = []
    for layer in prepared.layers:
        p = dict(layer.parameters)
        mp = MaterialParams(
            eps_r=float(p["eps_r"]),
            mu_n=float(p["mu_n"]),
            mu_p=float(p["mu_p"]),
            D_ion=float(p["D_ion"]),
            P_lim=float(p["P_lim"]),
            P0=float(p["P0"]),
            ni=float(p["ni"]),
            tau_n=float(p["tau_n"]),
            tau_p=float(p["tau_p"]),
            n1=float(p["n1"]),
            p1=float(p["p1"]),
            B_rad=float(p["B_rad"]),
            C_n=float(p["C_n"]),
            C_p=float(p["C_p"]),
            alpha=float(p["alpha"]),
            N_A=float(p["N_A"]),
            N_D=float(p["N_D"]),
            chi=float(p["chi"]),
            Eg=float(p["Eg"]),
            A_star_n=float(p["A_star_n"]),
            A_star_p=float(p["A_star_p"]),
            v_th=float(p["v_th"]),
            D_ion_neg=float(p["D_ion_neg"]),
            P0_neg=float(p["P0_neg"]),
            P_lim_neg=float(p["P_lim_neg"]),
            Nc300=float(p["Nc300"]),
            Nv300=float(p["Nv300"]),
            mu_T_gamma=float(p["mu_T_gamma"]),
            B_rad_T_gamma=float(p["B_rad_T_gamma"]),
            varshni_alpha=float(p["varshni_alpha"]),
            varshni_beta=float(p["varshni_beta"]),
            incoherent=bool(p["incoherent"]),
        )
        layers.append(
            LayerSpec(
                name=layer.id,
                thickness=float(layer.thickness),
                params=mp,
                role=layer.role,
            )
        )
    return DeviceStack(
        layers=tuple(layers),
        V_bi=float(settings["V_bi"]),
        Phi=float(settings["Phi"]),
        T=float(settings["T"]),
        mode=str(settings["mode"]),
        built_in_potential_mode=str(settings["built_in_potential_mode"]),
        dos_band_potentials=bool(settings["dos_band_potentials"]),
    )


def _prepare_homojunction_pin(base_prepared: PreparedDevice) -> PreparedDevice:
    import json

    editing = base_prepared.to_input().editing_data()
    ni_abs = 363572901223.5858
    for layer in editing["layers"]:
        params = dict(layer["parameters"])
        params.update(
            {
                "chi": 3.9,
                "Eg": 1.6,
                "Nc300": 1e25,
                "Nv300": 1e25,
                "ni": ni_abs,
                "n1": ni_abs,
                "p1": ni_abs,
            }
        )
        layer["parameters"] = params
    editing["electrical_grid"] = [
        {"layer": layer["id"], "interval_weight": 1.0, "alpha": 3.0}
        for layer in editing["layers"]
    ]
    src = SourceDocument("test.homojunction_pin", json.dumps(editing).encode("utf-8"))
    return resolve_device(
        DeviceInput.model_validate(editing),
        base_prepared.defaults,
        base_prepared.resources,
        sources=(*base_prepared.sources, src),
    )


def test_dark_jv_parity_sign_conventions_and_diode_extraction() -> None:
    steady_ex = _import_example("examples.research_1d.steady")
    prepared = _prepare_homojunction_pin(steady_ex.load_device())
    compiled = compile_device_1d(prepared, intervals_per_layer=8)
    legacy_stack = _legacy_stack_from_prepared(prepared)

    res = run_dark_jv(
        compiled,
        V_min=0.0,
        V_max=1.05,
        n_points=28,
        options=DarkJVOptions(
            steady=SteadyOptions(
                continuity_tolerance_A_m2=1e-7,
                current_spread_tolerance_A_m2=1e-7,
            )
        ),
    )
    leg = legacy_run_dark_jv(
        legacy_stack,
        V_max=1.05,
        n_points=28,
        N_grid=24,
        v_rate=0.5,
        rtol=1e-4,
        atol=1e-6,
    )

    assert res.completed
    assert res.fit_failure_reason is None
    assert np.allclose(
        res.pv_output_current_A_m2, -res.voltage_conjugate_current_A_m2, atol=1e-12
    )
    # Forward dark injection has negative PV output current and positive conjugate current
    fwd = res.voltage_V > 0.2
    assert np.all(res.pv_output_current_A_m2[fwd] < 0.0)
    assert np.all(res.voltage_conjugate_current_A_m2[fwd] > 0.0)

    assert 1.0 <= res.n_ideality <= 2.5
    assert res.saturation_current_J0_A_m2 > 0.0
    assert res.r_squared > 0.999

    # T4 parity against quasi-static legacy dark J-V in diode injection regime
    high_inj = res.voltage_V >= 0.75
    np.testing.assert_allclose(
        res.pv_output_current_A_m2[high_inj],
        leg.J[high_inj],
        rtol=5e-3,
        atol=1e-3,
    )
    assert res.n_ideality == pytest.approx(leg.n_ideality, rel=3.5e-2)


def test_coupled_ionic_dark_jv_sweep() -> None:
    ion_multi = _import_example("examples.research_1d.ionic_multilayer")
    ion_steady = _import_example("examples.research_1d.ionic_steady")
    prepared_ion = ion_multi.prepare_confined_ion_device()
    compiled_ion = compile_ionic_device_1d(
        prepared_ion,
        intervals_per_layer=6,
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    controls = ion_steady.controls(compiled_ion.ion_node_count)
    res = run_dark_jv(
        compiled_ion,
        voltage_V=np.linspace(0.0, 0.85, 18),
        options=DarkJVOptions(ionic_steady=controls, min_diode_window_points=5),
    )
    assert res.completed
    assert np.allclose(
        res.pv_output_current_A_m2, -res.voltage_conjugate_current_A_m2, atol=1e-12
    )
    assert res.fit_failure_reason is None
    assert 1.0 <= res.n_ideality <= 2.5


def test_waveform_jv_triangular_corner_displacement_and_legacy_parity() -> None:
    steady_ex = _import_example("examples.research_1d.steady")
    prepared = _prepare_homojunction_pin(steady_ex.load_device())
    compiled = compile_device_1d(prepared, intervals_per_layer=8)
    legacy_stack = _legacy_stack_from_prepared(prepared)

    # 1) Fast ramp in the dark verifies exact capacitive displacement jump (+C*dV/dt -> -C*dV/dt)
    fast_pw = build_piecewise_waveform(
        [
            WaveformSegment("up", 1e-4, 0.0, 0.2, generation_scale=0.0, n_steps=5, role="forward_ramp"),
            WaveformSegment("hold_top", 5e-5, 0.2, 0.2, generation_scale=0.0, n_steps=2, role="hold"),
            WaveformSegment("down", 1e-4, 0.2, 0.0, generation_scale=0.0, n_steps=5, role="reverse_ramp"),
        ]
    )
    fast_res = run_waveform_jv(compiled, fast_pw)
    assert fast_res.completed
    assert fast_res.accepted_steps == 12
    assert fast_res.waveform.corner_indices == (0, 5, 7, 12)
    assert np.allclose(
        fast_res.pv_output_current_A_m2,
        -fast_res.voltage_conjugate_current_A_m2,
        atol=1e-12,
    )
    # On ascending ramp dV/dt > 0, voltage-conjugate displacement current is positive (+C dV/dt);
    # on descending ramp dV/dt < 0, voltage-conjugate displacement current is negative (-C |dV/dt|).
    assert fast_res.voltage_conjugate_displacement_A_m2[4] > 0.0
    assert fast_res.voltage_conjugate_displacement_A_m2[7] < 0.0

    # 2) Illuminated slow triangular J-V parity against legacy run_waveform_jv
    wf = JVWaveform(
        start_voltage_V=0.0,
        dark_seed_s=0.0,
        dark_prep_s=0.0,
        branch_dwell_s=1e-3,
        turnaround_dark=False,
    )
    leg_wf = LegacyJVWaveform(
        start_voltage_V=0.0,
        dark_seed_s=0.0,
        dark_prep_s=0.0,
        branch_dwell_s=1e-3,
        turnaround_dark=False,
    )
    steady_opts = SteadyOptions(
        continuity_tolerance_A_m2=1e-7,
        current_spread_tolerance_A_m2=1e-7,
    )
    res_jv = run_waveform_jv(
        compiled,
        wf,
        V_max=1.25,
        v_rate=0.5,
        n_points=26,
        generation_scale=1.0,
        options=WaveformJVOptions(steady=steady_opts),
    )
    leg_jv = legacy_run_waveform_jv(
        legacy_stack,
        leg_wf,
        N_grid=24,
        n_points=26,
        v_rate=0.5,
        V_max=1.25,
        illuminated=True,
        rtol=1e-4,
        atol=1e-6,
    )
    assert res_jv.completed
    assert res_jv.metrics_fwd is not None and res_jv.metrics_rev is not None
    assert res_jv.metrics_fwd.open_circuit_voltage_V is not None
    assert res_jv.metrics_rev.open_circuit_voltage_V is not None
    assert abs(res_jv.metrics_fwd.open_circuit_voltage_V - leg_jv.metrics_fwd.V_oc) <= 2e-3
    assert abs(res_jv.metrics_rev.open_circuit_voltage_V - leg_jv.metrics_rev.V_oc) <= 2e-3
    assert res_jv.hysteresis_index_rev_norm is not None
    assert abs(res_jv.hysteresis_index_rev_norm) < 2e-3  # ion-free quasi-static has ~0 hysteresis


def test_coupled_ionic_waveform_jv_conserves_domain_inventory() -> None:
    ion_multi = _import_example("examples.research_1d.ionic_multilayer")
    ion_steady = _import_example("examples.research_1d.ionic_steady")
    prepared_ion = ion_multi.prepare_confined_ion_device()
    compiled_ion = compile_ionic_device_1d(
        prepared_ion,
        intervals_per_layer=6,
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    base_controls = ion_steady.controls(compiled_ion.ion_node_count)
    controls = replace(
        base_controls,
        electronic=replace(
            base_controls.electronic,
            continuity_tolerance_A_m2=1e-7,
            current_spread_tolerance_A_m2=1e-7,
        ),
    )
    pw_ion = build_piecewise_waveform(
        [
            WaveformSegment("fwd", 0.05, 0.0, 0.4, generation_scale=0.0, n_steps=8, role="forward_ramp"),
            WaveformSegment("hold", 0.02, 0.4, 0.4, generation_scale=0.0, n_steps=3, role="hold"),
            WaveformSegment("rev", 0.05, 0.4, 0.0, generation_scale=0.0, n_steps=8, role="reverse_ramp"),
        ]
    )
    res_ion = run_waveform_jv(
        compiled_ion,
        pw_ion,
        options=WaveformJVOptions(ionic_steady=controls),
    )
    assert res_ion.completed
    assert len(res_ion.inventory_relative_drift) == len(compiled_ion.ion_species_names)
    assert all(drift <= 1e-12 for drift in res_ion.inventory_relative_drift)

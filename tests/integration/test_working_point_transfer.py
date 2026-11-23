"""Integration tests for P05-01b multi-stage working-point transfer and paired perturbations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import CompiledDevice1D, compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.ionic_1d import IonicState1D
from solarlab.experiments.ionic_transient import IonicTransientOptions
from solarlab.experiments.protocol import Protocol
from solarlab.experiments.solarscale import (
    BOLTZMANN_CONSTANT_J_K,
    prepare_solarscale_device,
)
from solarlab.experiments.transient import TransientOptions, TransientProtocol
from solarlab.experiments.working_point_transfer import (
    WorkingPointBundle,
    capture_working_point,
    integrate_stage_from_working_point,
    physical_state_equivalent,
    run_paired_perturbation_stages,
    transfer_working_point,
)
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, solve_ionic_steady_state
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state


def _consistent_ni(eg_ev: float, nc: float = 1.0e25, nv: float = 1.0e25) -> float:
    vt = BOLTZMANN_CONSTANT_J_K * 300.0 / ELEMENTARY_CHARGE_C
    return math.sqrt(nc * nv) * math.exp(-eg_ev / (2.0 * vt))


def _device_doc(
    *,
    absorber_eg: float = 1.6,
    tau_s: float = 2.0e-7,
    p0: float = 0.0,
    d_ion: float = 0.0,
) -> dict[str, Any]:
    ni_htl = _consistent_ni(2.2)
    ni_abs = _consistent_ni(absorber_eg)
    ni_etl = _consistent_ni(2.4)
    return {
        "device": {
            "T": 300.0,
            "Phi": 2.5e21,
            "built_in_potential_mode": "semiconductor_work_function",
        },
        "layers": [
            {
                "name": "HTL",
                "role": "HTL",
                "thickness": 80.0e-9,
                "eps_r": 3.5,
                "mu_n": 1.0e-6,
                "mu_p": 1.0e-4,
                "chi": 3.5,
                "Eg": 2.2,
                "Nc300": 1.0e25,
                "Nv300": 1.0e25,
                "ni": ni_htl,
                "n1": ni_htl,
                "p1": ni_htl,
                "N_A": 1.0e24,
                "N_D": 0.0,
                "tau_n": 1.0e-7,
                "tau_p": 1.0e-7,
                "B_rad": 1.0e-18,
                "alpha": 1.0e5,
            },
            {
                "name": "absorber",
                "role": "absorber",
                "thickness": 320.0e-9,
                "eps_r": 24.0,
                "mu_n": 2.0e-4,
                "mu_p": 2.0e-4,
                "chi": 3.9,
                "Eg": absorber_eg,
                "Nc300": 1.0e25,
                "Nv300": 1.0e25,
                "ni": ni_abs,
                "n1": ni_abs,
                "p1": ni_abs,
                "N_A": 0.0,
                "N_D": 0.0,
                "tau_n": tau_s,
                "tau_p": tau_s,
                "B_rad": 1.0e-18,
                "alpha": 2.0e6,
                "P0": p0,
                "D_ion": d_ion,
                "P_lim": 1.0e27,
            },
            {
                "name": "ETL",
                "role": "ETL",
                "thickness": 80.0e-9,
                "eps_r": 4.0,
                "mu_n": 1.0e-4,
                "mu_p": 1.0e-6,
                "chi": 4.0,
                "Eg": 2.4,
                "Nc300": 1.0e25,
                "Nv300": 1.0e25,
                "ni": ni_etl,
                "n1": ni_etl,
                "p1": ni_etl,
                "N_A": 0.0,
                "N_D": 1.0e24,
                "tau_n": 1.0e-7,
                "tau_p": 1.0e-7,
                "B_rad": 1.0e-18,
                "alpha": 1.0e5,
            },
        ],
    }


def _transient_opts() -> TransientOptions:
    return TransientOptions(
        1.0e-11,
        1.0e-8,
        1.0e-11,
        0.1,
        2.0,
        SolveBudget(50, 1000, 50, 24),
        32,
        30.0,
        "previous",
        POISSON_CONTACT_BALANCE_V1,
    )


def _stage_protocol_doc(
    phases: list[str],
    transfers: list[dict[str, Any]],
) -> Protocol:
    return Protocol.from_document(
        {
            "schema": "solarlab.protocol.v1",
            "id": "multistage_1d_protocol",
            "version": 1,
            "experiment_id": "multistage_transient_1d",
            "preparation": {"schema": "solarlab.prep.v1", "mode": "finite_time"},
            "waveform": {"schema": "solarlab.waveform.v1", "kind": "piecewise_step"},
            "switches": {"schema": "solarlab.switches.v1", "light": "switched"},
            "sampling": {"schema": "solarlab.sampling.v1", "grid": "explicit"},
            "control_mode": "voltage_source",
            "phases": phases,
            "transfers": transfers,
        }
    )


def test_electronic_multistage_transfer_continuity_and_displacement_current() -> None:
    prepared, _ = prepare_solarscale_device(_device_doc())
    device = compile_device_1d(prepared, intervals_per_layer=6)
    opts = _transient_opts()

    dc0 = solve_steady_state(device, voltage_V=0.0, generation_scale=1.0)
    wp0 = capture_working_point(
        device,
        dc0.state,
        point_id="wp_t0",
        owner_epoch="ep1",
        stage_id="stage1",
        time_s=0.0,
        generation_scale=1.0,
        is_equilibrium=False,
        prepared=prepared,
    )

    stage1_drive = TransientProtocol(
        times_s=np.linspace(0.0, 2.0e-7, 4),
        voltage_V=np.linspace(0.0, 0.30, 4),
        generation_scale=np.ones(4),
    )
    trace1 = integrate_stage_from_working_point(
        device,
        wp0,
        stage1_drive,
        options=opts,
        stage_id="stage1",
        final_point_id="wp_stage1_left",
        owner_epoch="ep1",
    )
    assert trace1.outcome == "completed"
    assert trace1.final_bundle is not None

    left_json = trace1.final_bundle.to_json()
    left_reloaded = WorkingPointBundle.from_json(left_json)
    proto = _stage_protocol_doc(
        ["stage1", "stage2"],
        [
            {
                "from": "stage1",
                "to": "stage2",
                "working_point": {
                    "id": left_reloaded.id,
                    "sha256": left_reloaded.sha256,
                },
                "preserve": ["state", "inventories", "materials", "damage", "grid"],
            }
        ],
    )

    # Step voltage from 0.30 -> 0.45 V and illumination 1.0 -> 1.2 suns at t = 2e-7 s.
    xfer = transfer_working_point(
        proto,
        left_reloaded,
        device,
        from_stage="stage1",
        to_stage="stage2",
        boundary_time_s=2.0e-7,
        right_voltage_V=0.45,
        right_generation_scale=1.2,
        right_point_id="wp_stage2_right",
        owner_epoch="ep1",
        right_encoding="physical_fields",
    )
    assert xfer.rebound_materials is False
    assert physical_state_equivalent(trace1.states[-1], xfer.right_state, rtol=1.0e-12)

    stage2_drive = TransientProtocol(
        times_s=np.linspace(2.0e-7, 4.0e-7, 4),
        voltage_V=np.full(4, 0.45),
        generation_scale=np.full(4, 1.2),
    )
    trace2 = integrate_stage_from_working_point(
        device,
        xfer.boundary.right_bundle,
        stage2_drive,
        options=opts,
        stage_id="stage2",
        final_point_id="wp_stage2_end",
        owner_epoch="ep1",
    )
    assert trace2.outcome == "completed"
    # Step-to-step Backward Euler displacement current across the voltage jump is non-zero!
    assert abs(float(trace2.terminal_displacement_current_A_m2[0])) > 1.0e-2
    assert float(np.max(trace2.maxwell_spread_A_m2)) < 1.0e-7


def test_coupled_ionic_multistage_transfer_and_inventory_conservation() -> None:
    doc_ion = _device_doc(p0=2.0e23, d_ion=1.0e-12)
    doc_ion["device"]["ion_steric_diffusion_only"] = True
    doc_ion["layers"][0]["Eg"] = 1.8
    doc_ion["layers"][0]["chi"] = 3.6
    doc_ion["layers"][0]["ni"] = _consistent_ni(1.8)
    doc_ion["layers"][0]["n1"] = _consistent_ni(1.8)
    doc_ion["layers"][0]["p1"] = _consistent_ni(1.8)
    doc_ion["layers"][2]["Eg"] = 2.0
    doc_ion["layers"][2]["ni"] = _consistent_ni(2.0)
    doc_ion["layers"][2]["n1"] = _consistent_ni(2.0)
    doc_ion["layers"][2]["p1"] = _consistent_ni(2.0)
    prepared, _ = prepare_solarscale_device(doc_ion)
    ion_layer_id = next(layer.id for layer in prepared.layers if layer.role == "absorber")
    ion_dev = compile_ionic_device_1d(
        prepared,
        intervals_per_layer=6,
        ion_domain_layer=ion_layer_id,
        ion_domain_endpoints=("blocking", "blocking"),
    )
    steady_opts = IonicSteadyOptions(
        SteadyOptions(),
        1.0e-11,
        5.0e6,
        5.0e6,
        1.0,
        ion_dev.ion_node_count - 1,
        SolveBudget(60, 600, 60, 40),
        30.0,
    )
    steady0 = solve_ionic_steady_state(
        ion_dev, voltage_V=0.002, generation_scale=0.0, options=steady_opts
    )
    assert steady0.accepted and steady0.state is not None
    inv0 = steady0.state.domain_inventory_m2

    ion_opts = IonicTransientOptions(
        electronic=_transient_opts(),
        ion_balance_tolerance_m2_s=5.0e6,
        ion_logit_step=1.0,
    )
    wp0 = capture_working_point(
        ion_dev,
        steady0.state,
        point_id="wp_ion_0",
        owner_epoch="ep_ion",
        stage_id="prebias",
        time_s=0.0,
        generation_scale=0.0,
        is_equilibrium=False,
        prepared=prepared,
    )

    stage1_drive = TransientProtocol(
        times_s=np.array([0.0, 1.0e-4, 2.0e-4]),
        voltage_V=np.array([0.002, 0.15, 0.30]),
        generation_scale=np.zeros(3),
    )
    trace1 = integrate_stage_from_working_point(
        ion_dev,
        wp0,
        stage1_drive,
        options=ion_opts,
        stage_id="prebias",
        final_point_id="wp_ion_1_left",
        owner_epoch="ep_ion",
        final_encoding="physical_fields",
    )
    assert trace1.outcome == "completed" and trace1.final_bundle is not None

    proto = _stage_protocol_doc(
        ["prebias", "relax"],
        [
            {
                "from": "prebias",
                "to": "relax",
                "working_point": {
                    "id": trace1.final_bundle.id,
                    "sha256": trace1.final_bundle.sha256,
                },
                "preserve": ["state", "inventories", "materials", "damage", "grid"],
            }
        ],
    )
    xfer = transfer_working_point(
        proto,
        trace1.final_bundle,
        ion_dev,
        from_stage="prebias",
        to_stage="relax",
        boundary_time_s=2.0e-4,
        right_voltage_V=0.10,
        right_generation_scale=0.0,
        right_point_id="wp_ion_2_right",
        owner_epoch="ep_ion",
        right_encoding="coordinates",
    )
    assert physical_state_equivalent(trace1.states[-1], xfer.right_state, rtol=1.0e-12)

    stage2_drive = TransientProtocol(
        times_s=np.array([2.0e-4, 3.0e-4, 4.0e-4]),
        voltage_V=np.full(3, 0.10),
        generation_scale=np.zeros(3),
    )
    trace2 = integrate_stage_from_working_point(
        ion_dev,
        xfer.boundary.right_bundle,
        stage2_drive,
        options=ion_opts,
        stage_id="relax",
        final_point_id="wp_ion_2_end",
        owner_epoch="ep_ion",
    )
    assert trace2.outcome == "completed"
    final_ion = trace2.states[-1]
    assert isinstance(final_ion, IonicState1D)
    rel_inv_drift = abs(final_ion.domain_inventory_m2 - inv0) / inv0
    assert rel_inv_drift <= 1.0e-12


def test_damage_rebinding_across_three_stages_and_paired_perturbation_runner() -> None:
    prep_base, _ = prepare_solarscale_device(_device_doc(tau_s=2.0e-7))
    prep_dmg, _ = prepare_solarscale_device(_device_doc(tau_s=2.0e-8))
    dev_base = compile_device_1d(prep_base, intervals_per_layer=5)
    dev_dmg = compile_device_1d(prep_dmg, intervals_per_layer=5)
    opts = _transient_opts()

    dc0 = solve_steady_state(dev_base, voltage_V=0.0, generation_scale=1.0)
    wp0 = capture_working_point(
        dev_base,
        dc0.state,
        point_id="wp0",
        owner_epoch="ep_dmg",
        stage_id="s1",
        time_s=0.0,
        generation_scale=1.0,
        is_equilibrium=False,
        prepared=prep_base,
    )
    # Finite-time non-equilibrium preparation in stage s1.
    tr1 = integrate_stage_from_working_point(
        dev_base,
        wp0,
        TransientProtocol(
            times_s=np.array([0.0, 5.0e-8, 1.0e-7]),
            voltage_V=np.array([0.0, 0.25, 0.40]),
            generation_scale=np.ones(3),
        ),
        options=opts,
        stage_id="s1",
        final_point_id="wp1_left",
        owner_epoch="ep_dmg",
    )
    assert tr1.outcome == "completed" and tr1.final_bundle is not None
    assert tr1.final_bundle.sections["history"]["is_equilibrium"] is False

    # Paired perturbation + unpulsed control from the non-equilibrium WorkingPointBundle!
    paired_proto = _stage_protocol_doc(["s1", "pulse"], [])
    pulsed_drive = TransientProtocol(
        times_s=np.array([1.0e-7, 1.5e-7, 2.0e-7, 2.5e-7]),
        voltage_V=np.array([0.40, 0.45, 0.40, 0.40]),
        generation_scale=np.array([1.0, 1.15, 1.0, 1.0]),
    )
    paired = run_paired_perturbation_stages(
        dev_base,
        tr1.final_bundle,
        pulsed_drive,
        options=opts,
        protocol=paired_proto,
        run_id="paired_run_01",
        experiment_id="transient_perturbation_1d",
        owner_epoch="ep_dmg",
    )
    p_doc = paired.pulsed_result.export()
    c_doc = paired.control_result.export()
    assert p_doc["header"]["bindings"]["initial"] == c_doc["header"]["bindings"]["initial"]
    assert p_doc["header"]["bindings"]["initial"]["sha256"] == tr1.final_bundle.sha256
    assert p_doc["header"]["evidence"][0]["label"] == "FIRST_SLICE"
    assert float(np.max(np.abs(paired.differential_total_current_A_m2))) > 1.0e-1

    # Stage 1 -> Stage 2 without damage change keeps dev_base; Stage 2 -> Stage 3 with damage rebinds!
    proto_s12 = _stage_protocol_doc(
        ["s1", "s2", "s3"],
        [
            {
                "from": "s1",
                "to": "s2",
                "working_point": {
                    "id": tr1.final_bundle.id,
                    "sha256": tr1.final_bundle.sha256,
                },
                "preserve": ["state", "inventories", "grid"],
            }
        ],
    )
    rebind_log: list[str] = []

    def _rebind(_dmg: Any) -> CompiledDevice1D:
        rebind_log.append("rebound")
        return dev_dmg

    xfer12 = transfer_working_point(
        proto_s12,
        tr1.final_bundle,
        dev_base,
        from_stage="s1",
        to_stage="s2",
        boundary_time_s=1.0e-7,
        right_voltage_V=0.40,
        right_generation_scale=1.0,
        right_point_id="wp2_right",
        owner_epoch="ep_dmg",
        rebind_damaged_device=_rebind,
    )
    assert xfer12.rebound_materials is False and rebind_log == []

    proto_s23 = _stage_protocol_doc(
        ["s1", "s2", "s3"],
        [
            {
                "from": "s2",
                "to": "s3",
                "working_point": {
                    "id": xfer12.boundary.right_bundle.id,
                    "sha256": xfer12.boundary.right_bundle.sha256,
                },
                "preserve": ["state", "inventories", "grid"],
            }
        ],
    )
    xfer23 = transfer_working_point(
        proto_s23,
        xfer12.boundary.right_bundle,
        dev_base,
        from_stage="s2",
        to_stage="s3",
        boundary_time_s=1.0e-7,
        right_voltage_V=0.40,
        right_generation_scale=1.0,
        right_point_id="wp3_right",
        owner_epoch="ep_dmg",
        next_damage_state={"status": "stress_trap_growth", "tau_s": 2.0e-8},
        rebind_damaged_device=_rebind,
    )
    assert xfer23.rebound_materials is True and rebind_log == ["rebound"]
    assert xfer23.compiled_device.identity == dev_dmg.identity
    tr3 = integrate_stage_from_working_point(
        dev_dmg,
        xfer23.boundary.right_bundle,
        TransientProtocol(
            times_s=np.array([1.0e-7, 1.5e-7, 2.0e-7]),
            voltage_V=np.full(3, 0.40),
            generation_scale=np.ones(3),
        ),
        options=opts,
        stage_id="s3",
        final_point_id="wp3_end",
        owner_epoch="ep_dmg",
    )
    assert tr3.outcome == "completed"

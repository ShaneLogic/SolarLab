"""Unit tests for P05-01b working-point transfer and P05-05 identifiability / SolarScale."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import CompiledDevice1D, compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.identifiability import (
    FailedSampleRecord,
    GenericIdentifiabilityParameter,
    GenericIdentifiabilityProtocol,
    collinearity_index,
    confounded_parameter_groups,
    identifiability_to_experiment_result,
    run_protocol_identifiability,
    subset_collinearity_indices,
)
from solarlab.experiments.protocol import Protocol
from solarlab.experiments.results import EVIDENCE_LABELS, decode_array
from solarlab.experiments.solarscale import (
    BOLTZMANN_CONSTANT_J_K,
    generate_solarlab_inputs,
    manifest_sha256,
    plan_solarlab_import,
    prepare_solarscale_device,
)
from solarlab.experiments.transient import TransientOptions, TransientProtocol
from solarlab.experiments.working_point_transfer import (
    COORDINATE_ENCODING_1D,
    PHYSICAL_FIELD_ENCODING_1D,
    WorkingPointBundle,
    WorkingPointTransferError,
    build_stage_experiment_result,
    capture_working_point,
    integrate_stage_from_working_point,
    physical_state_equivalent,
    reconstruct_ionic_state_1d,
    reconstruct_state_1d,
    transfer_working_point,
)
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import solve_steady_state


def _consistent_ni(eg_ev: float, nc: float = 1.0e25, nv: float = 1.0e25) -> float:
    vt = BOLTZMANN_CONSTANT_J_K * 300.0 / ELEMENTARY_CHARGE_C
    return math.sqrt(nc * nv) * math.exp(-eg_ev / (2.0 * vt))


def _inline_device_doc(*, absorber_eg: float = 1.6, p0: float = 0.0, d_ion: float = 0.0) -> dict[str, Any]:
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
                "tau_n": 2.0e-7,
                "tau_p": 2.0e-7,
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


def _compile_electronic(absorber_eg: float = 1.6, intervals: int = 4) -> tuple[PreparedDevice, CompiledDevice1D]:
    prepared, _ = prepare_solarscale_device(_inline_device_doc(absorber_eg=absorber_eg))
    return prepared, compile_device_1d(prepared, intervals_per_layer=intervals)


def _make_protocol(
    *,
    wp_id: str,
    wp_sha: str,
    preserve: list[str],
    phases: list[str] | None = None,
) -> Protocol:
    declared_phases = phases or ["prep", "probe"]
    return Protocol.from_document(
        {
            "schema": "solarlab.protocol.v1",
            "id": "proto_transfer_unit",
            "version": 1,
            "experiment_id": "transient_step_1d",
            "preparation": {"schema": "solarlab.prep.v1", "mode": "transient"},
            "waveform": {"schema": "solarlab.waveform.v1", "kind": "step"},
            "switches": {"schema": "solarlab.switches.v1", "light": "step"},
            "sampling": {"schema": "solarlab.sampling.v1", "points": 3},
            "control_mode": "voltage_source",
            "phases": declared_phases,
            "transfers": [
                {
                    "from": declared_phases[0],
                    "to": declared_phases[1],
                    "working_point": {"id": wp_id, "sha256": wp_sha},
                    "preserve": preserve,
                }
            ],
        }
    )


def test_working_point_bundle_roundtrip_physical_identity_and_shifted_voltage() -> None:
    prepared, device = _compile_electronic()
    sol = solve_steady_state(device, voltage_V=0.35, generation_scale=1.0)

    bundle_coord = capture_working_point(
        device,
        sol.state,
        point_id="wp_coord",
        owner_epoch="epoch_1",
        stage_id="prep",
        time_s=1.0e-4,
        generation_scale=1.0,
        is_equilibrium=False,
        encoding="coordinates",
        prepared=prepared,
        evidence_labels=("FIRST_SLICE",),
    )
    bundle_shifted = capture_working_point(
        device,
        sol.state,
        point_id="wp_shifted",
        owner_epoch="epoch_1",
        stage_id="prep",
        time_s=1.0e-4,
        generation_scale=1.0,
        is_equilibrium=False,
        encoding="coordinates",
        reference_voltage_V=0.75,
        prepared=prepared,
        evidence_labels=("FIRST_SLICE",),
    )
    bundle_phys = capture_working_point(
        device,
        sol.state,
        point_id="wp_phys",
        owner_epoch="epoch_1",
        stage_id="prep",
        time_s=1.0e-4,
        generation_scale=1.0,
        is_equilibrium=False,
        encoding="physical_fields",
        prepared=prepared,
        evidence_labels=("FIRST_SLICE",),
    )

    assert bundle_coord.state_encoding == COORDINATE_ENCODING_1D
    assert bundle_phys.state_encoding == PHYSICAL_FIELD_ENCODING_1D
    # Physical identity depends strictly on physical fields + geometry, invariant to encoding / ref V.
    assert bundle_coord.physical_identity == bundle_phys.physical_identity
    assert bundle_coord.physical_identity == bundle_shifted.physical_identity

    restored = WorkingPointBundle.from_json(bundle_coord.to_json())
    assert restored.sha256 == bundle_coord.sha256

    rec_coord = reconstruct_state_1d(restored, device)
    rec_shifted = reconstruct_state_1d(bundle_shifted, device)
    rec_phys = reconstruct_state_1d(bundle_phys, device)
    assert physical_state_equivalent(sol.state, rec_coord)
    assert physical_state_equivalent(sol.state, rec_shifted)
    assert physical_state_equivalent(sol.state, rec_phys)

    # Tampering with any section invalidates SHA-256 verification.
    bad_doc = bundle_phys.to_document()
    bad_doc["sections"]["history"]["time_s"] = 9.99
    with pytest.raises(WorkingPointTransferError, match="SHA-256 mismatch"):
        WorkingPointBundle.from_document(bad_doc)

    # Ionic bundle round-trip on finite-site positive ions.
    prep_ion, _ = prepare_solarscale_device(
        _inline_device_doc(p0=1.0e24, d_ion=1.0e-15)
    )
    ion_layer_id = next(lyr.id for lyr in prep_ion.layers if lyr.role == "absorber")
    ion_dev = compile_ionic_device_1d(
        prep_ion,
        intervals_per_layer=4,
        ion_domain_layer=ion_layer_id,
        ion_domain_endpoints=("blocking", "blocking"),
    )
    ion_state0 = ion_dev.initial_state(voltage_V=0.2)
    ion_coord = capture_working_point(
        ion_dev,
        ion_state0,
        point_id="wp_ion_c",
        owner_epoch="ep_i",
        stage_id="prep",
        time_s=0.0,
        generation_scale=0.0,
        is_equilibrium=False,
        encoding="coordinates",
        prepared=prep_ion,
    )
    ion_phys = capture_working_point(
        ion_dev,
        ion_state0,
        point_id="wp_ion_p",
        owner_epoch="ep_i",
        stage_id="prep",
        time_s=0.0,
        generation_scale=0.0,
        is_equilibrium=False,
        encoding="physical_fields",
        prepared=prep_ion,
    )
    assert ion_coord.physical_identity == ion_phys.physical_identity
    assert physical_state_equivalent(
        ion_state0, reconstruct_ionic_state_1d(ion_phys, ion_dev)
    )


def test_transfer_preserve_refusals_and_damage_gated_material_rebinding() -> None:
    prepared, device = _compile_electronic(absorber_eg=1.6, intervals=4)
    _, finer_device = _compile_electronic(absorber_eg=1.6, intervals=6)
    _, damaged_device = _compile_electronic(absorber_eg=1.55, intervals=4)

    sol = solve_steady_state(device, voltage_V=0.2, generation_scale=1.0)
    bundle = capture_working_point(
        device,
        sol.state,
        point_id="wp_stage1_end",
        owner_epoch="epoch_1",
        stage_id="prep",
        time_s=2.0e-5,
        generation_scale=1.0,
        is_equilibrium=False,
        prepared=prepared,
    )

    # 1. Grid mismatch refused when 'grid' is preserved.
    proto_strict = _make_protocol(
        wp_id=bundle.id,
        wp_sha=bundle.sha256,
        preserve=["state", "inventories", "materials", "damage", "grid"],
    )
    with pytest.raises(WorkingPointTransferError):
        transfer_working_point(
            proto_strict,
            bundle,
            finer_device,
            from_stage="prep",
            to_stage="probe",
            boundary_time_s=2.0e-5,
            right_voltage_V=0.4,
            right_generation_scale=1.0,
            right_point_id="wp_stage2_start",
            owner_epoch="epoch_1",
        )

    # 2. Preserving 'coordinate_reference' forbids reference-voltage jumps.
    proto_coord_lock = _make_protocol(
        wp_id=bundle.id,
        wp_sha=bundle.sha256,
        preserve=["state", "coordinate_reference", "materials", "grid"],
    )
    with pytest.raises(WorkingPointTransferError, match="coordinate_reference"):
        transfer_working_point(
            proto_coord_lock,
            bundle,
            device,
            from_stage="prep",
            to_stage="probe",
            boundary_time_s=2.0e-5,
            right_voltage_V=0.5,
            right_generation_scale=1.0,
            right_point_id="wp_stage2_start",
            owner_epoch="epoch_1",
        )

    # 3. Omitting 'state' from preserve NEVER silently resets to steady state.
    proto_no_state = _make_protocol(
        wp_id=bundle.id,
        wp_sha=bundle.sha256,
        preserve=["materials", "damage", "grid"],
    )
    with pytest.raises(WorkingPointTransferError):
        transfer_working_point(
            proto_no_state,
            bundle,
            device,
            from_stage="prep",
            to_stage="probe",
            boundary_time_s=2.0e-5,
            right_voltage_V=0.2,
            right_generation_scale=1.0,
            right_point_id="wp_stage2_start",
            owner_epoch="epoch_1",
        )

    # 4. Damage-gated rebinding: unchanged damage MUST NOT invoke rebind callback;
    # changed damage invokes rebind callback ONLY when 'damage'/'materials' are not locked.
    rebind_calls: list[dict[str, Any]] = []

    def _rebind_cb(dmg: Any) -> CompiledDevice1D:
        rebind_calls.append(dict(dmg))
        return damaged_device

    proto_damage_allowed = _make_protocol(
        wp_id=bundle.id,
        wp_sha=bundle.sha256,
        preserve=["state", "inventories", "grid"],
    )
    unchanged_res = transfer_working_point(
        proto_damage_allowed,
        bundle,
        device,
        from_stage="prep",
        to_stage="probe",
        boundary_time_s=2.0e-5,
        right_voltage_V=0.4,
        right_generation_scale=1.0,
        right_point_id="wp_unchanged",
        owner_epoch="epoch_1",
        rebind_damaged_device=_rebind_cb,
    )
    assert unchanged_res.rebound_materials is False
    assert unchanged_res.compiled_device.identity == device.identity
    assert rebind_calls == []

    # Damage change refused if 'materials' is preserved.
    with pytest.raises(WorkingPointTransferError, match="damage state changed"):
        transfer_working_point(
            proto_strict,
            bundle,
            device,
            from_stage="prep",
            to_stage="probe",
            boundary_time_s=2.0e-5,
            right_voltage_V=0.4,
            right_generation_scale=1.0,
            right_point_id="wp_fail_dmg",
            owner_epoch="epoch_1",
            next_damage_state={"status": "photo_degraded", "delta_eg_ev": -0.05},
            rebind_damaged_device=_rebind_cb,
        )

    changed_res = transfer_working_point(
        proto_damage_allowed,
        bundle,
        device,
        from_stage="prep",
        to_stage="probe",
        boundary_time_s=2.0e-5,
        right_voltage_V=0.4,
        right_generation_scale=1.0,
        right_point_id="wp_damaged",
        owner_epoch="epoch_1",
        next_damage_state={"status": "photo_degraded", "delta_eg_ev": -0.05},
        rebind_damaged_device=_rebind_cb,
    )
    assert changed_res.rebound_materials is True
    assert len(rebind_calls) == 1
    assert changed_res.compiled_device.identity == damaged_device.identity
    assert physical_state_equivalent(sol.state, changed_res.right_state)


def test_partial_stage_trace_nan_masking_and_evidence_label_preservation() -> None:
    prepared, device = _compile_electronic()
    sol = solve_steady_state(device, voltage_V=0.0, generation_scale=1.0)
    bundle = capture_working_point(
        device,
        sol.state,
        point_id="wp_init",
        owner_epoch="epoch_1",
        stage_id="prep",
        time_s=0.0,
        generation_scale=1.0,
        is_equilibrium=False,
        prepared=prepared,
        evidence_labels=("FIRST_SLICE",),
        limitations=("synthetic_unit_test",),
    )
    proto = _make_protocol(
        wp_id=bundle.id,
        wp_sha=bundle.sha256,
        preserve=["state", "inventories", "materials", "damage", "grid"],
    )
    # Force an incomplete trajectory on step 2 via 1-iteration Newton budget on a huge jump.
    stage_proto = TransientProtocol(
        times_s=np.array([0.0, 1.0e-7, 2.0e-7]),
        voltage_V=np.array([0.0, 0.0, 4.5]),
        generation_scale=np.array([1.0, 1.0, 1.0]),
    )
    tight_opts = TransientOptions(
        1.0e-11,
        1.0e-8,
        1.0e-11,
        0.05,
        0.5,
        SolveBudget(1, 2, 1, 0),
        10,
        5.0,
        "previous",
        POISSON_CONTACT_BALANCE_V1,
    )
    trace = integrate_stage_from_working_point(
        device,
        bundle,
        stage_proto,
        options=tight_opts,
        stage_id="probe",
        final_point_id="wp_end",
        owner_epoch="epoch_1",
    )
    assert trace.outcome != "completed"
    assert math.isfinite(float(trace.terminal_total_current_A_m2[0]))
    assert math.isnan(float(trace.terminal_total_current_A_m2[1]))

    exp_res = build_stage_experiment_result(
        trace,
        protocol=proto,
        run_id="run_partial",
        attempt_id="att_1",
        experiment_id="transient_step_1d",
    )
    exported = exp_res.export()
    assert exported["header"]["outcome"] == trace.outcome
    assert exported["header"]["evidence"][0]["label"] == "FIRST_SLICE"
    assert exported["header"]["evidence"][0]["label"] in EVIDENCE_LABELS
    assert exported["metrics"]["final_terminal_total_current_A_m2"]["status"] == "unidentified"
    decoded_tot = decode_array(exported["payload"]["terminal_total_current_A_m2"])
    assert math.isfinite(float(decoded_tot[0]))
    assert math.isnan(float(decoded_tot[1]))


def test_collinearity_confounding_failed_samples_and_solarscale_provenance(
    tmp_path: Path,
) -> None:
    # Orthogonal columns -> collinearity index == 1.0, no confounded groups.
    j_orth = np.array([[2.0, 0.0], [0.0, 5.0], [0.0, 0.0]])
    gamma_orth = collinearity_index(j_orth)
    assert gamma_orth is not None and math.isclose(gamma_orth, 1.0, rel_tol=1e-12)
    assert confounded_parameter_groups(j_orth, ("p1", "p2"), collinearity_limit=20.0) == ()

    # Near-collinear columns -> large collinearity index and grouped confounding.
    j_coll = np.array([[1.0, 1.0, 0.0], [2.0, 2.0 + 1e-5, 0.0], [0.0, 0.0, 3.0]])
    subsets = subset_collinearity_indices(j_coll, ("nt", "sigma", "offset"))
    assert subsets[("nt", "sigma")] is not None and subsets[("nt", "sigma")] > 1.0e4
    assert subsets[("nt", "offset")] is not None and math.isclose(
        subsets[("nt", "offset")] or 0.0, 1.0, rel_tol=1e-10
    )
    groups = confounded_parameter_groups(
        j_coll, ("nt", "sigma", "offset"), collinearity_limit=100.0
    )
    assert groups == (("nt", "sigma"),)

    # FailedSampleRecord retention in run_protocol_identifiability + ExperimentResult export.
    gen_proto = GenericIdentifiabilityProtocol(
        protocol_id="synthetic_fail_probe",
        parameters=(
            GenericIdentifiabilityParameter("a", -1.0, 1.0, 0.0),
            GenericIdentifiabilityParameter("b", -1.0, 1.0, 0.0),
        ),
        observable_labels=("obs0", "obs1", "obs2"),
        observable_units=("A/m^2", "A/m^2", "A/m^2"),
        standard_deviations=(0.05, 0.05, 0.05),
        profile_grid_count=5,
    )

    def _forward_with_domain_hole(coords: np.ndarray) -> np.ndarray:
        if float(coords[0]) > 0.65:
            raise RuntimeError("out_of_domain_probe")
        a_val = 10.0 ** float(coords[0])
        b_val = 10.0 ** float(coords[1])
        return np.array([a_val + b_val, 2.0 * a_val - b_val, a_val * b_val], dtype=float)

    id_res = run_protocol_identifiability(
        gen_proto,
        _forward_with_domain_hole,
        multistart_fractions=((0.5, 0.5), (0.9, 0.5)),
    )
    assert len(id_res.failed_samples) >= 1
    assert isinstance(id_res.failed_samples[0], FailedSampleRecord)
    assert id_res.analysis_certified is False

    exp_env = identifiability_to_experiment_result(
        id_res,
        run_id="id_run_1",
        attempt_id="id_att_1",
        evidence_label="FIRST_SLICE",
    )
    env_doc = exp_env.export()
    assert env_doc["header"]["outcome"] == "partial"
    assert len(env_doc["payload"]["failed_samples"]) == len(id_res.failed_samples)

    # SolarScale provenance filtering: reject 'missing' and 'swept' as fixed inputs,
    # and convert cm^2/(V s) -> m^2/(V s) (* 1e-4).
    records_file = tmp_path / "records.json"
    template_file = tmp_path / "template.yaml"
    records_payload = {
        "records": [
            {
                "material_id": "mat_valid",
                "formula": "CsPbI3",
                "screening": {
                    "production_solarlab_ready": True,
                    "readiness": "promising",
                    "final_fom_score": 0.91,
                },
                "properties": {
                    "band_gap_hse_ev": {
                        "value": 1.68,
                        "unit": "eV",
                        "provenance": {"kind": "computed", "source": "vasp"},
                    },
                    "dielectric_static_avg": {
                        "value": 21.5,
                        "unit": "1",
                        "provenance": {"kind": "computed", "source": "dfpt"},
                    },
                    "electron_mobility_cm2_v_s": {
                        "value": 25.0,
                        "unit": "cm^2/(V s)",
                        "provenance": {"kind": "computed", "source": "amset"},
                    },
                    "hole_mobility_cm2_v_s": {
                        "value": 999.0,
                        "unit": "cm^2/(V s)",
                        "provenance": {"kind": "swept", "source": "grid"},
                    },
                    "absorber_thickness_m": {
                        "value": 500.0e-9,
                        "unit": "m",
                        "provenance": {"kind": "swept", "source": "grid"},
                    },
                },
            },
            {
                "material_id": "mat_missing_eps",
                "formula": "CsSnI3",
                "screening": {
                    "production_solarlab_ready": True,
                    "readiness": "promising",
                    "final_fom_score": 0.85,
                },
                "properties": {
                    "band_gap_hse_ev": {
                        "value": 1.35,
                        "unit": "eV",
                        "provenance": {"kind": "computed", "source": "vasp"},
                    },
                    "dielectric_static_avg": {
                        "value": 30.0,
                        "unit": "1",
                        "provenance": {"kind": "missing", "source": "none"},
                    },
                },
            },
        ]
    }
    records_file.write_text(json.dumps(records_payload), encoding="utf-8")
    import yaml

    template_file.write_text(
        yaml.safe_dump(_inline_device_doc(absorber_eg=1.6), sort_keys=False),
        encoding="utf-8",
    )

    plan = plan_solarlab_import(
        records_file,
        template_path=template_file,
        import_policy="production",
        activate_bandgap=True,
    )
    selected_by_id = {c["material_id"]: c for c in plan["selected"]}
    skipped_by_id = {c["material_id"]: c for c in plan["skipped"]}
    assert "mat_valid" in selected_by_id
    assert math.isclose(
        selected_by_id["mat_valid"]["mapped_parameters"]["absorber.mu_n"], 25.0e-4
    )
    assert "absorber.mu_p" not in selected_by_id["mat_valid"]["mapped_parameters"]
    assert "mat_missing_eps" in skipped_by_id
    assert "dielectric_static_avg" in skipped_by_id["mat_missing_eps"]["missing_required"]

    out_a = tmp_path / "gen_a"
    out_b = tmp_path / "gen_b"
    man_a = generate_solarlab_inputs(
        records_file,
        template_path=template_file,
        out_dir=out_a,
        import_policy="production",
        activate_bandgap=True,
    )
    _ = generate_solarlab_inputs(
        records_file,
        template_path=template_file,
        out_dir=out_b,
        import_policy="production",
        activate_bandgap=True,
    )
    # Volatile fields ignored; config contents match; manifest digest deterministic when paths match.
    assert len(man_a["generated"]) == 1
    assert manifest_sha256(man_a) == manifest_sha256(json.loads(json.dumps(man_a)))
    assert (out_a / "mat_valid.yaml").read_text(encoding="utf-8") == (
        out_b / "mat_valid.yaml"
    ).read_text(encoding="utf-8")

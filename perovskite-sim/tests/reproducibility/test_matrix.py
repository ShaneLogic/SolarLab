from __future__ import annotations

from dataclasses import fields, replace

import pytest
import yaml

from backend.main import _stack_to_config_dict, stack_from_dict
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.reproducibility import (
    project_root,
    semantic_sha256,
    validate_matrix,
    verify_baseline,
)
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.scaps_compat.loader import load_scaps_yaml


ROOT = project_root()


def _matrix():
    return yaml.safe_load(
        (ROOT / "reproducibility/config_benchmark_matrix.yaml").read_text()
    )


def test_matrix_covers_and_loads_every_shipped_config():
    report = validate_matrix(ROOT)
    assert report["configs"] == 49
    assert report["resources"] == 21
    assert report["schemas"] == {
        "standard-device-v1": 43,
        "scaps-device-v1": 5,
        "tandem-v1": 1,
    }


def test_default_thermal_velocity_preserves_frozen_semantics():
    baseline = MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-4,
        mu_p=1.0e-4,
        D_ion=0.0,
        P_lim=1.0e30,
        P0=0.0,
        ni=1.0e16,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=1.0e16,
        p1=1.0e16,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
    )
    historical_optional_fields = {
        "N_A_bulk",
        "N_D_bulk",
        "doping_profile_shape",
        "doping_decay_length",
        "doping_edge",
        "v_th",
        "carrier_statistics",
        "dopant_ionization_model",
        "donor_binding_energy_eV",
        "acceptor_binding_energy_eV",
        "donor_degeneracy",
        "acceptor_degeneracy",
        "band_gap_narrowing_model",
        "bgn_reference_energy_eV",
        "bgn_reference_density_m3",
        "bgn_log_shape",
        "bgn_conduction_band_fraction",
        "bulk_trap_distribution",
        "cigs_graded_optics",
        "defect_schema_version",
        "defect_model",
        "bulk_defects",
    }
    historical_payload = {
        field.name: getattr(baseline, field.name)
        for field in fields(baseline)
        if field.name not in historical_optional_fields
    }

    assert semantic_sha256(baseline) == semantic_sha256(historical_payload)
    assert semantic_sha256(replace(baseline, v_th=2.0e5)) != (
        semantic_sha256(baseline)
    )
    mb_with_dos = replace(
        baseline,
        Eg=1.124,
        Nc300=2.8e25,
        Nv300=1.04e25,
    )
    fd_with_dos = replace(
        mb_with_dos,
        carrier_statistics="fermi_dirac",
    )
    assert semantic_sha256(fd_with_dos) != semantic_sha256(mb_with_dos)


def test_built_in_potential_modes_preserve_only_inert_frozen_fields():
    compatibility = load_device_from_yaml(
        str(ROOT / "configs/ionmonger_benchmark.yaml")
    )
    historical_payload = {
        field.name: getattr(compatibility, field.name)
        for field in fields(compatibility)
        if field.name not in {
            "built_in_potential_mode",
            "work_function_left_eV",
            "work_function_right_eV",
            "grid_interval_weights",
            "grid_alphas",
            "jv_solver_policy",
            "interface_charge_closure",
            "interface_charge_rebaseline_acknowledged",
            "graded_optics",
        }
    }

    assert semantic_sha256(compatibility) == semantic_sha256(historical_payload)

    physical = replace(
        compatibility,
        built_in_potential_mode="metal_work_function",
        work_function_left_eV=5.2,
        work_function_right_eV=4.1,
    )
    assert semantic_sha256(physical) != semantic_sha256(compatibility)
    assert semantic_sha256(replace(physical, V_bi=9.9)) == semantic_sha256(
        physical
    )


def test_parked_interface_charge_defaults_preserve_historical_hash():
    compatibility = load_device_from_yaml(
        str(ROOT / "configs/ionmonger_benchmark.yaml")
    )
    historical_payload = {
        field.name: getattr(compatibility, field.name)
        for field in fields(compatibility)
        if field.name not in {
            "built_in_potential_mode",
            "work_function_left_eV",
            "work_function_right_eV",
            "grid_interval_weights",
            "grid_alphas",
            "jv_solver_policy",
            "interface_charge_closure",
            "interface_charge_rebaseline_acknowledged",
            "graded_optics",
        }
    }
    research_intent = replace(
        compatibility,
        interface_charge_closure="equilibrium_referenced",
        interface_charge_rebaseline_acknowledged=True,
    )

    assert semantic_sha256(compatibility) == semantic_sha256(historical_payload)
    assert semantic_sha256(research_intent) != semantic_sha256(compatibility)


def test_standard_schema_registers_spatial_doping_profile_contract():
    registry = yaml.safe_load(
        (ROOT / "reproducibility/schema_registry.yaml").read_text()
    )
    profile = registry["schemas"]["standard-device-v1"][
        "optional_layer_groups"
    ]["spatial_doping_profile"]
    assert set(profile["activation_keys"]) == {"N_A_bulk", "N_D_bulk"}
    assert set(profile["companion_keys"]) == {
        "doping_profile_shape",
        "doping_decay_length",
        "doping_edge",
    }
    assert profile["supported_shapes"] == ["gaussian"]
    assert set(profile["supported_edges"]) == {"front", "back"}
    assert profile["density_units"] == "m-3"
    assert profile["length_units"] == "m"


def test_standard_schema_registers_bulk_carrier_statistics_contract():
    registry = yaml.safe_load(
        (ROOT / "reproducibility/schema_registry.yaml").read_text()
    )
    statistics = registry["schemas"]["standard-device-v1"][
        "optional_layer_groups"
    ]["bulk_carrier_statistics"]
    assert statistics["default"] == "maxwell_boltzmann"
    assert set(statistics["supported_modes"]) == {
        "maxwell_boltzmann",
        "fermi_dirac",
    }
    assert set(statistics["fermi_dirac_required_keys"]) == {
        "Eg",
        "Nc300",
        "Nv300",
    }

    ionization = registry["schemas"]["standard-device-v1"][
        "optional_layer_groups"
    ]["dopant_ionization"]
    assert ionization["default"] == "fully_ionized"
    assert set(ionization["supported_modes"]) == {
        "fully_ionized",
        "discrete_level",
    }
    assert set(ionization["discrete_level_required_material_keys"]) == {
        "Eg",
        "Nc300",
        "Nv300",
    }

    narrowing = registry["schemas"]["standard-device-v1"][
        "optional_layer_groups"
    ]["band_gap_narrowing"]
    assert narrowing["default"] == "off"
    assert narrowing["supported_modes"] == ["off", "slotboom"]
    assert set(narrowing["slotboom_required_material_keys"]) == {
        "Eg",
        "Nc300",
        "Nv300",
    }
    assert set(narrowing["companion_keys"]) == {
        "bgn_reference_energy_eV",
        "bgn_reference_density_m3",
        "bgn_log_shape",
        "bgn_conduction_band_fraction",
    }


def test_p0_patch_and_frozen_files_match_manifest():
    report = verify_baseline("p0-certified-2026-08-01", ROOT)
    assert report["base_commit"] == "c23e5b9beb3c356250ea32dcb09c78dc45ba28ec"
    assert report["checked_files"] == 15
    assert report["worktree_checked_files"] == 0
    assert report["reconstruction_verified"] is True
    assert report["git_tag_created"] is False


def test_benchmark_config_links_are_explicit_and_bidirectional():
    matrix = _matrix()
    reverse = {
        entry["path"]: set(entry["benchmarks"])
        for entry in matrix["configs"]
    }
    for benchmark_id, benchmark in matrix["benchmarks"].items():
        assert benchmark["configs"]
        assert benchmark["node_ids"]
        assert all("::" in node_id for node_id in benchmark["node_ids"])
        assert set(benchmark["configs"]) == {
            path for path, benchmark_ids in reverse.items()
            if benchmark_id in benchmark_ids
        }


def test_partial_status_always_has_physical_benchmark_evidence():
    matrix = _matrix()
    physical_kinds = {"external", "internal", "numerical"}
    for entry in matrix["configs"]:
        if entry["status"] != "partial":
            continue
        kinds = {
            matrix["benchmarks"][benchmark_id]["kind"]
            for benchmark_id in entry["benchmarks"]
        }
        assert kinds.intersection(physical_kinds), entry["path"]


def test_dynamic_defect_transient_production_evidence_is_source_bound():
    matrix = _matrix()
    benchmark = matrix["benchmarks"][
        "dynamic-defect-ion-transient-production-closure"
    ]
    config = next(
        entry
        for entry in matrix["configs"]
        if entry["path"]
        == "configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    )
    evidence = " ".join(benchmark["limitations"])

    assert benchmark["status"] == "pass"
    assert benchmark["claim_level"] == "internal-numerical-candidate"
    assert config["status"] == "certified"
    assert "4f13a4bebfc71275bb83394e184144965d1359a6" in evidence
    assert "464da3ec6e0bb94fbd40a82bdc9325b29eabbb6622dc8fcad699b505a4434f5f" in evidence
    assert "52c63f74e5e139487aebce1e3ebe576d4861fb566788261e40d594e8f76f703b" in evidence
    assert "366/366" in evidence
    assert "not SCAPS transient parity" in evidence


def test_nk_manifest_covers_every_csv_stem_exactly():
    manifest = yaml.safe_load(
        (ROOT / "perovskite_sim/data/nk/manifest.yaml").read_text()
    )
    csv_stems = {
        path.stem for path in (ROOT / "perovskite_sim/data/nk").glob("*.csv")
    }
    assert set(manifest) == csv_stems
    assert manifest["NiOx"]["source"] == "repository-parameterized-fit"
    assert "b3235885ca67befb7540fb19f01b7102d0a12378" in manifest["NiOx"]["reference"]


@pytest.mark.parametrize(
    "relpath",
    [
        entry["path"]
        for entry in _matrix()["configs"]
        if entry["schema"] == "standard-device-v1"
    ],
)
def test_standard_yaml_and_inline_backend_have_identical_semantics(relpath):
    with (ROOT / relpath).open() as stream:
        raw = yaml.safe_load(stream)
    yaml_stack = load_device_from_yaml(str(ROOT / relpath))
    inline_stack = stack_from_dict(raw)
    assert semantic_sha256(inline_stack) == semantic_sha256(yaml_stack)


@pytest.mark.parametrize(
    "relpath",
    [
        entry["path"]
        for entry in _matrix()["configs"]
        if entry["schema"] == "scaps-device-v1"
    ],
)
def test_scaps_to_inline_roundtrip_preserves_solver_semantics(relpath):
    scaps_stack = load_scaps_yaml(str(ROOT / relpath))
    inline_stack = stack_from_dict(_stack_to_config_dict(scaps_stack))
    assert semantic_sha256(inline_stack) == semantic_sha256(scaps_stack)


def test_matrix_does_not_promote_open_gaps_to_certified():
    matrix = _matrix()
    gaps = {
        key for key, value in matrix["benchmarks"].items()
        if value["kind"] == "gap"
    }
    assert gaps == {
        "csi-grid-envelope-gap",
    }
    for entry in matrix["configs"]:
        if gaps.intersection(entry["benchmarks"]):
            assert entry["status"] != "certified"


def test_calibrated_external_reproductions_are_not_certified():
    matrix = _matrix()
    calibrated = {
        benchmark_id: benchmark
        for benchmark_id, benchmark in matrix["benchmarks"].items()
        if benchmark.get("claim_level") == "calibrated_reproduction"
    }
    assert set(calibrated) == {
        "courtier2019-ionmonger",
        "calado2016-driftfusion",
    }
    for benchmark in calibrated.values():
        assert benchmark["evidence_tier"] == "calibration_only"
        assert set(benchmark["protocol"]) == {"local", "source"}
        assert set(benchmark["calibration"]["parameters"]) == {"device.V_bi"}
        assert benchmark["calibration"]["targets"]
        assert benchmark["non_calibration_checks"]
        assert not set(benchmark["calibration"]["targets"]).intersection(
            benchmark["non_calibration_checks"]
        )

    config_by_path = {entry["path"]: entry for entry in matrix["configs"]}
    for benchmark_id, benchmark in calibrated.items():
        for relpath in benchmark["configs"]:
            entry = config_by_path[relpath]
            assert benchmark_id in entry["benchmarks"]
            assert entry["status"] == "partial"


def test_every_p1_gap_has_a_reproduction_and_acceptance_contract():
    data = yaml.safe_load((ROOT / "reproducibility/p1_gaps.yaml").read_text())
    gaps = {gap["id"]: gap for gap in data["gaps"]}
    assert set(gaps) == {
        "cigs-2um-graded-notch",
        "external-solver-curve-crosscheck",
        "ionmonger-residual-ss-96",
        "scaps-low-doping-etl",
        "lin2019-tandem-jsc-pce",
        "csi-qf-jv-grid-convergence",
        "csi-transient-jv-grid-envelope",
        "csi-mott-schottky-convergence",
    }
    assert gaps["scaps-low-doping-etl"]["status"] == "closed"
    assert gaps["scaps-low-doping-etl"]["resolution"]
    assert gaps["ionmonger-residual-ss-96"]["status"] == "closed"
    assert gaps["ionmonger-residual-ss-96"]["resolution"]
    assert gaps["cigs-2um-graded-notch"]["status"] == "closed"
    assert gaps["cigs-2um-graded-notch"]["resolution"]
    assert gaps["csi-qf-jv-grid-convergence"]["status"] == "closed"
    assert gaps["csi-qf-jv-grid-convergence"]["resolution"]
    assert gaps["lin2019-tandem-jsc-pce"]["status"] == "closed"
    assert gaps["lin2019-tandem-jsc-pce"]["resolution"]
    assert data["phase"] == "P1"
    assert data["phase_status"] == "closed_with_p2_deferrals"
    assert set(data["closure_summary"]["closed"]) == {
        gap_id for gap_id, gap in gaps.items() if gap["status"] == "closed"
    }
    assert set(data["closure_summary"]["deferred_to_p2"]) == {
        gap_id for gap_id, gap in gaps.items()
        if gap["status"] == "deferred_to_p2"
    }
    assert set(data["closure_summary"]["deferred_to_p2"]) == {
        "csi-transient-jv-grid-envelope",
        "csi-mott-schottky-convergence",
        "external-solver-curve-crosscheck",
    }
    assert not {gap_id for gap_id, gap in gaps.items() if gap["status"] == "open"}
    for gap in gaps.values():
        assert gap["status"] in {"closed", "deferred_to_p2"}
        assert gap["reproduction"]
        assert len(gap["acceptance"]) >= 3
        if gap["status"] == "deferred_to_p2":
            assert gap["next_experiment"]
            assert gap["deferral_reason"]

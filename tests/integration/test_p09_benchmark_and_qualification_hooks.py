"""Integration tests for P09-01 benchmark pre-flight hooks and standalone driver.

Covers:
1. ``run_steady_jv(..., stop_after_voc=True)`` early stopping at the first
   post-``V_oc`` sign change and 0-ULP invariance when ``stop_after_voc=False``.
2. ``FrequencyOptions.max_frequencies == 128`` default and 60-frequency ``AC60``
   acceptance without raising ``ValueError``.
3. ``allow_legacy_benchmark_deck`` / ``legacy_benchmark_deck_scope`` opt-in on
   ``compile_device_1d`` and ``compile_device_2d``, keeping all default ``v1``
   ``CapabilityError`` refusals intact when ``False``.
4. Standalone ``python -I -B scripts/benchmarks/p09_candidate_validation.py``
   execution with zero ``perovskite_sim``, ``backend``, or ``legacy_engine``
   imports, ``RuntimeTiming`` / ``reconcile_outer_timing`` receipts, and
   ``scientific_qualification=False``.
"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from scripts.benchmarks import p09_candidate_validation as p09_driver
from solarlab.config.device_import import import_standard_device
from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.device.compile_1d import compile_device_1d, legacy_benchmark_deck_scope
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.defaults import DefaultCatalog
from solarlab.equations.dc_1d import CapabilityError
from solarlab.experiments.ionic_impedance import IonicImpedanceOptions
from solarlab.experiments.jv.steady import run_steady_jv
from solarlab.experiments.two_dimensional.inputs import JV2DInput, MicrostructureInput
from solarlab.experiments.two_dimensional.jv_sweep_2d import (
    project_prepared_for_electronic_2d,
)
from solarlab.materials import (
    load_packaged_default_catalog_mapping,
    load_packaged_preset_sources,
    load_packaged_resource_library,
)
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.mesh import PHYSICAL_BOUNDARY_TWO_SIDED_V1
from solarlab.physics.interface_transport import (
    SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
)
from solarlab.solvers.linear.frequency_domain import FrequencyOptions
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "p09_candidate_validation.py"


def _catalog_and_resources() -> tuple[DefaultCatalog, ResourceLibrary]:
    raw_cat = load_packaged_default_catalog_mapping()
    defaults = DefaultCatalog.from_mapping(
        {
            **raw_cat,
            "material": {
                "D_ion": 0.0,
                "P0": 0.0,
                "P_lim": 1.0e30,
                **raw_cat["material"],
            },
        }
    )
    return defaults, load_packaged_resource_library()


def test_frequency_options_default_supports_ac60_spectrum() -> None:
    opts = FrequencyOptions()
    assert opts.max_frequencies == 128
    imp_opts = IonicImpedanceOptions()
    assert imp_opts.linear.max_frequencies == 128


def test_default_compilers_still_refuse_legacy_benchmark_decks() -> None:
    defaults, resources = _catalog_and_resources()

    # 1. SCAPS mirror preset refused when allow_legacy_benchmark_deck=False
    scaps_src = next(
        doc for doc in load_packaged_preset_sources() if doc.id == "scaps_mirror_v2.yaml"
    )
    scaps_in = import_scaps_device(
        scaps_src, id="scaps_mirror_v2", defaults=defaults
    ).validated_update({"spectrum": "am15g"})
    scaps_prep = resolve_device(scaps_in, defaults, resources, sources=(scaps_src,))
    with pytest.raises(CapabilityError):
        compile_device_1d(
            scaps_prep,
            intervals_per_layer=10,
            interface_topology="two_sided_trace",
            geometry_name=PHYSICAL_BOUNDARY_TWO_SIDED_V1,
            thermionic_mode=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
            allow_legacy_benchmark_deck=False,
        )

    # 2. Front-incoherent TMM substrate deck refused when allow_legacy_benchmark_deck=False
    tmm_path = REPO_ROOT / "tests/fixtures/configs/nip_MAPbI3_tmm.yaml"
    tmm_src = SourceDocument(id="nip_MAPbI3_tmm", content=tmm_path.read_bytes())
    tmm_in = import_standard_device(tmm_src, id="nip_tmm", defaults=defaults)
    tmm_no_ion_layers = [
        {**lyr, "parameters": {**lyr["parameters"], "D_ion": 0.0, "P0": 0.0}}
        for lyr in tmm_in.model_dump(mode="json", exclude_unset=True)["layers"]
    ]
    tmm_clean = resolve_device(
        tmm_in.validated_update({"spectrum": "am15g", "layers": tmm_no_ion_layers}),
        defaults,
        resources,
        sources=(tmm_src,),
    )
    with pytest.raises(CapabilityError):
        compile_device_1d(
            tmm_clean,
            intervals_per_layer=10,
            optical_model="tmm",
            allow_legacy_benchmark_deck=False,
        )
    with legacy_benchmark_deck_scope():
        compiled_tmm = compile_device_1d(
            tmm_clean, intervals_per_layer=10, optical_model="tmm"
        )
    assert "legacy_benchmark_deck" in compiled_tmm.capabilities

    # 3. 2D uniform legacy deck refused when allow_legacy_benchmark_deck=False
    twod_path = (
        REPO_ROOT / "tests/fixtures/configs/twod/nip_MAPbI3_uniform.yaml"
    )
    twod_src = SourceDocument(id="twod_uniform", content=twod_path.read_bytes())
    twod_in = import_standard_device(
        twod_src, id="twod_uniform", defaults=defaults
    ).validated_update(
        {
            "settings": {
                "V_bi": 1.1,
                "Phi": 2.5e21,
                "dos_band_potentials": False,
                "flat_band_metal_contacts": True,
                "ion_steric_shared_site": False,
            }
        }
    )
    twod_prep = project_prepared_for_electronic_2d(
        resolve_device(twod_in, defaults, resources, sources=(twod_src,)),
        strip_ions=True,
        strip_interfaces=False,
    )
    spatial = JV2DInput(
        kind="jv_2d",
        lateral_length=5.0e-7,
        Nx=4,
        Ny_per_layer=10,
        lateral_bc="periodic",
        microstructure=MicrostructureInput(grain_boundaries=()),
    )
    with pytest.raises(CapabilityError):
        compile_device_2d(twod_prep, spatial, allow_legacy_benchmark_deck=False)
    compiled_2d = compile_device_2d(
        twod_prep, spatial, allow_legacy_benchmark_deck=True
    )
    assert "legacy_benchmark_deck" in compiled_2d.column.capabilities


def test_run_steady_jv_stop_after_voc_matches_full_sweep_and_preserves_default() -> None:
    defaults, resources = _catalog_and_resources()
    scaps_src = next(
        doc for doc in load_packaged_preset_sources() if doc.id == "scaps_mirror_v2.yaml"
    )
    scaps_in = import_scaps_device(
        scaps_src, id="scaps_mirror_v2", defaults=defaults
    ).validated_update({"spectrum": "am15g"})
    scaps_prep = resolve_device(scaps_in, defaults, resources, sources=(scaps_src,))
    compiled = compile_device_1d(
        scaps_prep,
        intervals_per_layer=10,
        interface_topology="two_sided_trace",
        geometry_name=PHYSICAL_BOUNDARY_TWO_SIDED_V1,
        thermionic_mode=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
        allow_legacy_benchmark_deck=True,
    )
    steady_opts = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-5, current_spread_tolerance_A_m2=1.0e-5
    )
    v_grid_16 = np.linspace(0.0, 1.6, 24)
    v_grid_135 = np.linspace(0.0, 1.35, 20)

    with pytest.raises(TypeError, match="stop_after_voc"):
        run_steady_jv(compiled, v_grid_16, stop_after_voc=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_bias_bisect"):
        run_steady_jv(compiled, v_grid_16, max_bias_bisect=-1)

    res_scaps_16 = run_steady_jv(
        compiled,
        v_grid_16,
        generation_scale=1.0,
        incident_power_W_m2=1000.0,
        options=steady_opts,
        stop_after_voc=True,
        max_bias_bisect=4,
    )
    assert 2 <= res_scaps_16.voltage_V.size < v_grid_16.size
    assert res_scaps_16.output_current_A_m2[-2] > 0.0
    assert res_scaps_16.output_current_A_m2[-1] <= 0.0

    res_stopped = run_steady_jv(
        compiled,
        v_grid_135,
        generation_scale=1.0,
        incident_power_W_m2=1000.0,
        options=steady_opts,
        stop_after_voc=True,
        max_bias_bisect=4,
    )
    res_full = run_steady_jv(
        compiled,
        v_grid_135,
        generation_scale=1.0,
        incident_power_W_m2=1000.0,
        options=steady_opts,
        stop_after_voc=False,
        max_bias_bisect=4,
    )
    n_stop = int(res_stopped.voltage_V.size)
    assert 2 <= n_stop < v_grid_135.size
    assert res_stopped.output_current_A_m2[-2] > 0.0
    assert res_stopped.output_current_A_m2[-1] <= 0.0
    # Prefix through the stopping bracket is 0-ULP bit-identical!
    np.testing.assert_array_equal(res_stopped.voltage_V, res_full.voltage_V[:n_stop])
    np.testing.assert_array_equal(
        res_stopped.output_current_A_m2,
        res_full.output_current_A_m2[:n_stop],
    )
    assert res_stopped.metrics == res_full.metrics
    assert "stop_after_voc" not in res_full.solver_controls


def test_p09_candidate_driver_ast_has_zero_legacy_imports() -> None:
    tree = ast.parse(DRIVER_SCRIPT.read_text(encoding="utf-8"), filename=str(DRIVER_SCRIPT))
    forbidden = ("perovskite_sim", "backend", "solarlab_server.legacy_engine")
    offenders: list[str] = []
    private_solarlab_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == p or alias.name.startswith(p + ".") for p in forbidden
                ):
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if any(
                node.module == p or node.module.startswith(p + ".") for p in forbidden
            ):
                offenders.append(node.module)
            if node.module == "solarlab" or node.module.startswith("solarlab."):
                for alias in node.names:
                    if alias.name.startswith("_"):
                        private_solarlab_imports.append(f"{node.module}:{alias.name}")
    assert offenders == []
    assert private_solarlab_imports == []


def test_p09_candidate_driver_in_process_and_isolated_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. In-process tolerance for pre-existing sys.modules entries from sibling tests,
    #    while default assert_zero_legacy_modules=True still rejects them.
    sentinel_name = "perovskite_sim._pytest_collection_sentinel"
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, sentinel_name, types.ModuleType(sentinel_name))
        with pytest.raises(RuntimeError, match="forbidden legacy modules loaded"):
            p09_driver.run_perf_candidate(
                repo_root=REPO_ROOT,
                selected_cases=("AC60",),
                repeats=1,
            )
        perf_inproc = p09_driver.run_perf_candidate(
            repo_root=REPO_ROOT,
            selected_cases=("AC60",),
            repeats=1,
            assert_zero_legacy_modules=False,
        )
    assert perf_inproc["scientific_qualification"] is False
    assert perf_inproc["cases"]["AC60"]["all_repeats_completed"] is True
    assert perf_inproc["cases"]["AC60"]["repeats"][0]["accepted_frequency_count"] == 60
    assert perf_inproc["cases"]["AC60"]["repeats"][0]["outer_timing"]["passed"] is True

    # 2. Isolated subprocess CLI (-I -B) covering all 5 PerformanceCasesV1 workloads
    #    (using --repo-data-root alias), scaps-sweep, and hysteresis-matrix deduplication
    perf_json = tmp_path / "perf_all5_smoke.json"
    scaps_json = tmp_path / "scaps_smoke.json"
    hi_json = tmp_path / "hi_smoke.json"
    env = {
        key: val
        for key, val in os.environ.items()
        if not key.startswith("PYTHON")
    }
    for key in p09_driver.THREAD_ENV_KEYS:
        env[key] = "1"

    proc_perf = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(DRIVER_SCRIPT),
            "--repo-data-root",
            str(REPO_ROOT),
            "--require-single-thread-blas",
            "perf-candidate",
            "--cases",
            "all",
            "--repeats",
            "1",
            "--spatial-nx",
            "2,4",
            "--transient-variants",
            "ionmonger_benchmark",
            "--output",
            str(perf_json),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    perf_summary = json.loads(proc_perf.stdout.strip())
    assert perf_summary["scientific_qualification"] is False
    perf_doc = json.loads(perf_json.read_text(encoding="utf-8"))
    assert set(perf_doc["cases"]) == {
        "TransientJvN80",
        "TransientJvN200",
        "ScapsSteady",
        "AC60",
        "Spatial2DStageA",
    }
    assert perf_doc["cases"]["TransientJvN80"]["variants"][0]["all_repeats_completed"] is True
    assert (
        perf_doc["cases"]["TransientJvN80"]["variants"][0]["repeats"][0]["outer_timing"]["passed"]
        is True
    )
    assert perf_doc["cases"]["TransientJvN200"]["variants"][0]["all_repeats_completed"] is True
    assert (
        perf_doc["cases"]["TransientJvN200"]["variants"][0]["repeats"][0]["outer_timing"]["passed"]
        is True
    )
    assert perf_doc["cases"]["ScapsSteady"]["all_repeats_completed"] is True
    assert perf_doc["cases"]["ScapsSteady"]["repeats"][0]["outer_timing"]["passed"] is True
    assert perf_doc["cases"]["AC60"]["all_repeats_completed"] is True
    assert perf_doc["cases"]["AC60"]["repeats"][0]["outer_timing"]["passed"] is True
    growth = perf_doc["cases"]["Spatial2DStageA"]["growth"]
    assert len(growth["pairs"]) == 1
    assert math.isfinite(growth["worst_pair_beta_median"])
    assert all(
        scale["all_repeats_completed"]
        and scale["repeats"][0]["outer_timing"]["passed"]
        for scale in perf_doc["cases"]["Spatial2DStageA"]["scales"]
    )

    proc_scaps = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(DRIVER_SCRIPT),
            "--repo-root",
            str(REPO_ROOT),
            "--require-single-thread-blas",
            "scaps-sweep",
            "--case-ids",
            "scaps_base,scaps_cbo:row4,scaps_cbo:row7,scaps_cbo:row12,scaps_cbo:row17",
            "--output",
            str(scaps_json),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    scaps_summary = json.loads(proc_scaps.stdout.strip())
    assert scaps_summary["scientific_qualification"] is False
    scaps_doc = json.loads(scaps_json.read_text(encoding="utf-8"))
    assert scaps_doc["summary_counts"]["total_executed"] == 5
    assert scaps_doc["summary_counts"]["reference_missing"] == 1
    assert scaps_doc["summary_counts"]["solver_failed"] == 0

    proc_hi = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(DRIVER_SCRIPT),
            "--repo-root",
            str(REPO_ROOT),
            "--require-single-thread-blas",
            "hysteresis-matrix",
            "--case-ids",
            "hi_density_00,hi_diffusivity_00,hi_diffusivity_10,hi_rate_40",
            "--output",
            str(hi_json),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    hi_summary = json.loads(proc_hi.stdout.strip())
    assert hi_summary["scientific_qualification"] is False
    hi_doc = json.loads(hi_json.read_text(encoding="utf-8"))
    assert hi_doc["requested_case_count"] == 4
    # hi_diffusivity_10 and hi_rate_40 share the exact nominal request_sha256!
    assert hi_doc["distinct_executed_request_count"] == 3
    assert hi_doc["completed_case_count"] == 4
    by_id = {c["id"]: c for c in hi_doc["cases"]}
    assert by_id["hi_rate_40"]["reused_from_case_id"] == "hi_diffusivity_10"
    assert by_id["hi_diffusivity_10"]["completed"] is True
    assert by_id["hi_rate_40"]["completed"] is True
    assert by_id["hi_diffusivity_10"]["comparison_deltas"]["HI_normalized_abs_diff"] < 0.005
    for zero_id in ("hi_density_00", "hi_diffusivity_00"):
        zero_case = by_id[zero_id]
        assert zero_case["completed"] is True
        assert zero_case["failure_reason"] is None
        assert abs(zero_case["candidate_metrics"]["HI_normalized"]) < 1.0e-3

    # Verify SCAPS CBO alignment trend (cliff < base == row7 < plateau, no V_bi inversion)
    scaps_by_id = {c["id"]: c for c in scaps_doc["cases"]}
    assert (
        scaps_by_id["scaps_cbo:row7"]["candidate_metrics"]["Voc_V"]
        == pytest.approx(scaps_by_id["scaps_base"]["candidate_metrics"]["Voc_V"], abs=1.0e-12)
    )
    assert (
        scaps_by_id["scaps_cbo:row4"]["candidate_metrics"]["Voc_V"]
        < scaps_by_id["scaps_base"]["candidate_metrics"]["Voc_V"]
        < scaps_by_id["scaps_cbo:row12"]["candidate_metrics"]["Voc_V"]
        < 1.25
    )
    assert abs(
        scaps_by_id["scaps_base"]["candidate_metrics"]["Voc_V"]
        - scaps_by_id["scaps_base"]["reference_metrics"]["Voc_V"]
    ) < 0.01
    assert abs(
        scaps_by_id["scaps_base"]["candidate_metrics"]["PCE_percent"]
        - scaps_by_id["scaps_base"]["reference_metrics"]["PCE_percent"]
    ) < 0.5


def test_compile_device_1d_scaps_gaussian_quadrature_default_invariance() -> None:
    from solarlab.physics.recombination import (
        SCAPS_GAUSSIAN_INTERFACE_QUADRATURE,
        SCAPS_INTERFACE_EXTREMA_ANCHOR,
        SINGLE_LEVEL_INTERFACE_DEFECT,
    )

    defaults, resources = _catalog_and_resources()
    scaps_src = next(
        doc for doc in load_packaged_preset_sources() if doc.id == "scaps_mirror_v2.yaml"
    )
    scaps_in = import_scaps_device(
        scaps_src, id="scaps_mirror_v2", defaults=defaults
    ).validated_update({"spectrum": "am15g"})
    scaps_prep = resolve_device(scaps_in, defaults, resources, sources=(scaps_src,))

    comp_default = compile_device_1d(
        scaps_prep,
        intervals_per_layer=10,
        interface_topology="two_sided_trace",
        geometry_name=PHYSICAL_BOUNDARY_TWO_SIDED_V1,
        thermionic_mode=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
        interface_band_reference_mode=SCAPS_INTERFACE_EXTREMA_ANCHOR,
        allow_legacy_benchmark_deck=True,
    )
    comp_explicit_single = compile_device_1d(
        scaps_prep,
        intervals_per_layer=10,
        interface_topology="two_sided_trace",
        geometry_name=PHYSICAL_BOUNDARY_TWO_SIDED_V1,
        thermionic_mode=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
        interface_band_reference_mode=SCAPS_INTERFACE_EXTREMA_ANCHOR,
        interface_defect_distribution_mode=SINGLE_LEVEL_INTERFACE_DEFECT,
        allow_legacy_benchmark_deck=True,
    )
    assert comp_default.identity == comp_explicit_single.identity
    assert comp_default.interface_model is not None
    assert comp_explicit_single.interface_model is not None
    assert (
        comp_default.interface_model.identity_sha256
        == comp_explicit_single.interface_model.identity_sha256
    )
    assert all(
        spec.physics.trap_quadrature is None
        for spec in comp_default.interface_model.two_sided
    )

    comp_gauss = compile_device_1d(
        scaps_prep,
        intervals_per_layer=10,
        interface_topology="two_sided_trace",
        geometry_name=PHYSICAL_BOUNDARY_TWO_SIDED_V1,
        thermionic_mode=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
        interface_band_reference_mode=SCAPS_INTERFACE_EXTREMA_ANCHOR,
        interface_defect_distribution_mode=SCAPS_GAUSSIAN_INTERFACE_QUADRATURE,
        allow_legacy_benchmark_deck=True,
    )
    assert comp_gauss.identity != comp_default.identity
    assert comp_gauss.interface_model is not None
    assert (
        comp_gauss.interface_model.identity_sha256
        != comp_default.interface_model.identity_sha256
    )
    assert len(comp_gauss.interface_model.two_sided) == 2
    # HTL/PVK is single-level in scaps_mirror_v2.yaml; PVK/ETL is 7-level Gaussian!
    assert comp_gauss.interface_model.two_sided[0].physics.trap_quadrature is None
    assert comp_gauss.interface_model.two_sided[1].physics.trap_quadrature is not None
    assert (
        comp_gauss.interface_model.two_sided[1].physics.trap_quadrature.energies_eV.shape
        == (7,)
    )

    with pytest.raises(ValueError, match="requires two_sided_trace"):
        compile_device_1d(
            scaps_prep,
            intervals_per_layer=10,
            interface_topology="deduplicated",
            interface_defect_distribution_mode=SCAPS_GAUSSIAN_INTERFACE_QUADRATURE,
            allow_legacy_benchmark_deck=True,
        )

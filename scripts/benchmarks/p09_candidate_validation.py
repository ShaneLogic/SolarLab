"""Standalone legacy-free P09-01 candidate validation and timing driver.

Executes prospective candidate measurements on the native ``solarlab`` core
with zero imports from ``perovskite_sim``, ``backend``, or retired server
shims (runnable under ``python -I -B``):

- ``perf-candidate``: all five ``reproducibility/PerformanceCasesV1.json``
  workloads (``TransientJvN80``, ``TransientJvN200``, ``ScapsSteady``,
  ``AC60``, ``Spatial2DStageA``) with actual assembled unknown counts,
  ``RuntimeTiming`` phase partitions, ``reconcile_outer_timing`` receipts, and
  empirical 2D sparse scaling exponents ``beta``.
- ``scaps-sweep``: the 262-point ``ScapsComparisonContractV1.json`` /
  ``RefactorReferenceManifestV1.json["scaps"]`` sweep (base + 13 coordinate
  groups, retaining all 14 ``reference_missing`` rows and honest gate status).
- ``hysteresis-matrix``: the 45-case (43-distinct-request)
  ``HysteresisReferenceContractV2.json`` /
  ``RefactorReferenceManifestV1.json["hi"]`` 7-phase protocol matrix,
  deduplicating identical ``request_sha256`` runs with explicit reuse
  provenance.
- ``smoke-all``: fast end-to-end smoke across all three candidate lanes.

All outputs carry ``scientific_qualification=False`` and
``physical_certificate_status="not_evaluated"``; failed or out-of-tolerance
points are retained with their original classification and never relabeled.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from typing import Any, cast

import numpy as np
from scipy.sparse import bmat, diags, eye
from scipy.sparse.linalg import spsolve
import yaml

# Allow standalone ``python -I -B scripts/benchmarks/p09_candidate_validation.py``
# execution without shadowing an installed wheel in site-packages: only fall back
# to repository ``src`` / root when ``solarlab`` or ``scripts`` is not importable.
_THIS_FILE = Path(__file__).resolve()
_BENCHMARKS_DIR = _THIS_FILE.parent
_DEFAULT_REPO_ROOT = _BENCHMARKS_DIR.parents[1]
if importlib.util.find_spec("solarlab") is None:
    _src_dir = _DEFAULT_REPO_ROOT / "src"
    if _src_dir.is_dir() and str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
if importlib.util.find_spec("scripts") is None:
    if _DEFAULT_REPO_ROOT.is_dir() and str(_DEFAULT_REPO_ROOT) not in sys.path:
        sys.path.append(str(_DEFAULT_REPO_ROOT))

from scripts.benchmarks.runtime_timing import (  # noqa: E402
    RuntimeTiming,
    reconcile_outer_timing,
)
from scripts.benchmarks.scaps_reference import (  # noqa: E402
    CONTRACT as SCAPS_CONTRACT_REL,
    CONTRACT_SHA256 as SCAPS_CONTRACT_SHA256,
    MANIFEST as MANIFEST_REL,
    build_cases as build_scaps_cases,
    case_document as build_scaps_case_document,
)
from solarlab.config.device_import import import_standard_device  # noqa: E402
from solarlab.config.resolve_device import resolve_device  # noqa: E402
from solarlab.config.scaps_input import import_scaps_device  # noqa: E402
from solarlab.contracts import frozen_array  # noqa: E402
from solarlab.contracts.units import ELEMENTARY_CHARGE_C  # noqa: E402
from solarlab.device.compile_1d import (  # noqa: E402
    compile_device_1d,
    legacy_benchmark_deck_scope,
)
from solarlab.device.compile_2d import compile_device_2d  # noqa: E402
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d  # noqa: E402
from solarlab.device.compile_ionic_2d import (  # noqa: E402
    build_radiative_reabsorption_2d,
)
from solarlab.device.defaults import DefaultCatalog  # noqa: E402
from solarlab.device.resolved import PreparedDevice  # noqa: E402
from solarlab.equations.assembly.classical_1d import (  # noqa: E402
    AssembledDC1DProblem,
    classical_dc_1d,
)
from solarlab.equations.assembly.ionic_1d import (  # noqa: E402
    AssembledConstrainedIonic1DProblem,
    AssembledIonic1DProblem,
)
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem  # noqa: E402
from solarlab.equations.dc_2d import CompiledDevice2D, State2D  # noqa: E402
from solarlab.equations.ionic_2d import (  # noqa: E402
    RadiativeReabsorptionSpec2D,
    recompute_g_with_rad_2d,
)
from solarlab.experiments.ionic_impedance import (  # noqa: E402
    IonicImpedanceOptions,
    solve_ionic_impedance,
)
from solarlab.experiments.ionic_transient import IonicTransientOptions  # noqa: E402
from solarlab.experiments.jv.steady import (  # noqa: E402
    run_steady_jv,
    steady_jv_metrics,
)
from solarlab.experiments.suns_voc import default_ionic_steady_options  # noqa: E402
from solarlab.experiments.transient import TransientOptions  # noqa: E402
from solarlab.experiments.two_dimensional.inputs import (  # noqa: E402
    JV2DInput,
    MicrostructureInput,
)
from solarlab.experiments.two_dimensional.jv_sweep_2d import (  # noqa: E402
    embed_frozen_ion_excess_2d,
    extract_snapshot_2d,
    project_prepared_for_electronic_2d,
)
from solarlab.experiments.waveform_jv import (  # noqa: E402
    WaveformJVOptions,
    WaveformSegment,
    build_piecewise_waveform,
    package_ionic_steady_initial,
    run_waveform_jv,
)
from solarlab.materials import (  # noqa: E402
    load_packaged_default_catalog_mapping,
    load_packaged_preset_sources,
    load_packaged_resource_library,
)
from solarlab.materials.resources import ResourceLibrary, ResourceTable  # noqa: E402
from solarlab.materials.source import SourceDocument  # noqa: E402
from solarlab.mesh import PHYSICAL_BOUNDARY_TWO_SIDED_V1  # noqa: E402
from solarlab.physics.interface_transport import (  # noqa: E402
    SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
)
from solarlab.solvers.linear.frequency_domain import FrequencyOptions  # noqa: E402
from solarlab.solvers.nonlinear.ionic_1d import (  # noqa: E402
    IonicSteadyOptions,
    solve_ionic_steady_state,
)
from solarlab.solvers.nonlinear.newton import NewtonSettings, solve_newton  # noqa: E402
from solarlab.solvers.nonlinear.problem import (  # noqa: E402
    CallableProblem,
    ResidualScaling,
    SolveBudget,
    StepControl,
)
from solarlab.solvers.nonlinear.production import (  # noqa: E402
    POISSON_CONTACT_BALANCE_V1,
    poisson_row_weights,
    production_convergence_norm,
    production_coordinates,
    production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import (  # noqa: E402
    SteadyOptions,
    SteadyStateError,
    physical_gates,
)
from solarlab.solvers.time import (  # noqa: E402
    BackwardEuler1DStep,
    step_accounting,
)
from solarlab.solvers.time.assembled_ionic_backward_euler import (  # noqa: E402
    AssembledIonicBackwardEuler1DStep,
)
from solarlab.solvers.time.assembled_storage import AssembledStorage1D  # noqa: E402
from solarlab.solvers.time.backward_euler_2d import (  # noqa: E402
    BackwardEuler2DOptions,
    compute_carrier_only_terminal_current_2d,
    run_backward_euler_2d,
)

SCHEMA_PERF = "solarlab.p09_perf_candidate.v1"
SCHEMA_SCAPS = "solarlab.p09_scaps_candidate.v1"
SCHEMA_HI = "solarlab.p09_hysteresis_candidate.v1"
SCHEMA_SMOKE = "solarlab.p09_smoke_all.v1"

PERF_CONTRACT_REL = "reproducibility/PerformanceCasesV1.json"
HI_CONTRACT_REL = "reproducibility/HysteresisReferenceContractV2.json"
HI_CONTRACT_SHA256 = "9a62e36920dd558e19c196cefed6cc191c74253db05ea54a0e1644d6a22eb96a"

THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
FORBIDDEN_MODULE_PREFIXES: tuple[str, ...] = (
    "perovskite_sim",
    "backend",
    "solarlab_server.legacy_engine",
)


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    if sys.platform == "darwin":
        return raw
    return raw * 1024


def _loaded_forbidden_legacy_modules() -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in FORBIDDEN_MODULE_PREFIXES
            )
        )
    )


def _check_legacy_modules_entry(*, assert_zero_legacy_modules: bool) -> tuple[str, ...]:
    loaded = _loaded_forbidden_legacy_modules()
    if assert_zero_legacy_modules and loaded:
        raise RuntimeError(
            f"forbidden legacy modules loaded in standalone candidate runner: {list(loaded)}"
        )
    return loaded


def _check_legacy_modules_exit(
    pre_loaded: tuple[str, ...], *, assert_zero_legacy_modules: bool
) -> None:
    loaded = _loaded_forbidden_legacy_modules()
    newly_loaded = sorted(set(loaded) - set(pre_loaded))
    if newly_loaded:
        raise RuntimeError(
            f"candidate runner imported forbidden legacy modules during execution: {newly_loaded}"
        )
    if assert_zero_legacy_modules and loaded:
        raise RuntimeError(
            f"forbidden legacy modules loaded in standalone candidate runner: {list(loaded)}"
        )


def _thread_environment_record(require_single_thread: bool = False) -> dict[str, Any]:
    values = {key: os.environ.get(key) for key in THREAD_ENV_KEYS}
    deterministic = all(values[key] == "1" for key in THREAD_ENV_KEYS)
    if require_single_thread and not deterministic:
        raise RuntimeError(
            f"single-thread BLAS environment required ({THREAD_ENV_KEYS} must all equal '1'); got {values}"
        )
    return {
        "variables": values,
        "single_thread_pinned": deterministic,
    }


def _host_environment_record(require_single_thread: bool = False) -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "max_rss_bytes": _max_rss_bytes(),
        "blas_threads": _thread_environment_record(require_single_thread),
    }


def _load_packaged_catalog_and_resources() -> tuple[DefaultCatalog, ResourceLibrary]:
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
    resources = load_packaged_resource_library()
    return defaults, resources


def _prepare_standard_yaml_device(
    yaml_bytes: bytes,
    *,
    source_id: str,
    device_id: str,
    defaults: DefaultCatalog,
    resources: ResourceLibrary,
) -> tuple[PreparedDevice, bool, str | None, bool]:
    src = SourceDocument(id=source_id, content=yaml_bytes)
    dev_in = import_standard_device(src, id=device_id, defaults=defaults)
    has_neg = any(
        float(lyr.parameters.P0_neg or 0.0) > 0.0
        or float(lyr.parameters.D_ion_neg or 0.0) > 0.0
        for lyr in dev_in.layers
    )
    has_opt = any(lyr.parameters.optical_material is not None for lyr in dev_in.layers)
    ion_layers = [
        lyr.id
        for lyr in dev_in.layers
        if lyr.role != "substrate" and float(lyr.parameters.P0 or 0.0) > 0.0
    ]
    settings_patch: dict[str, Any] = {
        "dos_band_potentials": False,
        "flat_band_metal_contacts": True,
    }
    if not has_neg:
        settings_patch["ion_steric_shared_site"] = False
    patch: dict[str, Any] = {
        "settings": {
            **dev_in.settings.model_dump(exclude_unset=True),
            **settings_patch,
        }
    }
    if has_opt and dev_in.spectrum is None:
        patch["spectrum"] = "am15g"
    dev_in = dev_in.validated_update(patch)
    prepared = resolve_device(dev_in, defaults, resources, sources=(src,))
    ion_layer_id = ion_layers[0] if ion_layers else None
    return prepared, bool(ion_layers), ion_layer_id, has_opt


def _prepare_scaps_yaml_device(
    yaml_bytes: bytes,
    *,
    source_id: str,
    device_id: str,
    defaults: DefaultCatalog,
    resources: ResourceLibrary,
) -> PreparedDevice:
    src = SourceDocument(id=source_id, content=yaml_bytes)
    dev_in = import_scaps_device(src, id=device_id, defaults=defaults)
    has_opt = any(lyr.parameters.optical_material is not None for lyr in dev_in.layers)
    if has_opt and dev_in.spectrum is None:
        dev_in = dev_in.validated_update({"spectrum": "am15g"})
    return resolve_device(dev_in, defaults, resources, sources=(src,))


def _prepare_dark_quasineutral_ionic_state(dev: Any, v0: float = 0.0) -> Any:
    n_nodes = dev.node_count
    orig = AssembledIonic1DProblem(dev, v0, 0.0)
    y = dev.initial_state(voltage_V=v0).u.copy()
    r0 = orig.residual(y)
    j0_diag = orig.jacobian(y).diagonal()
    for idx in (n_nodes, 2 * n_nodes - 1, 2 * n_nodes, 3 * n_nodes - 1):
        if j0_diag[idx] != 0.0:
            y[idx] -= r0[idx] / j0_diag[idx]
    xi = (dev.nodes_m - dev.nodes_m[0]) / (dev.nodes_m[-1] - dev.nodes_m[0])
    y[n_nodes : 2 * n_nodes] = y[n_nodes] * (1.0 - xi) + y[2 * n_nodes - 1] * xi
    y[2 * n_nodes : 3 * n_nodes] = (
        y[2 * n_nodes] * (1.0 - xi) + y[3 * n_nodes - 1] * xi
    )
    for _ in range(30):
        r_p = orig.residual(y)[:n_nodes]
        if float(np.max(np.abs(r_p))) < 1.0e-12:
            break
        dphi = spsolve(orig.jacobian(y)[:n_nodes, :n_nodes], -r_p)
        step = min(1.0, 0.2 / max(float(np.max(np.abs(dphi))), 1.0e-30))
        y[:n_nodes] += step * dphi
    return dev.state(y, voltage_V=v0)


def _warm_density_newton_on_be_step(
    dev: Any,
    step: AssembledIonicBackwardEuler1DStep,
    y0: np.ndarray,
    vk: float,
    settings: NewtonSettings,
    conv_norm: Any,
    eye_n: Any,
    eye_ion: Any,
) -> np.ndarray:
    """Physical-density tangent preconditioner on the exact BE residual."""
    n_nodes = dev.node_count
    vt = dev.electronic_component.thermal_voltage_V
    y = y0.copy()
    r0 = step.residual(y)
    j0 = step.jacobian(y).diagonal()
    for idx in (0, n_nodes - 1, n_nodes, 2 * n_nodes - 1, 2 * n_nodes, 3 * n_nodes - 1):
        if j0[idx] != 0.0:
            y[idx] -= r0[idx] / j0[idx]
    for _ in range(25):
        res = step.residual(y)
        if float(conv_norm(settings.scaling.scaled(res))) <= 0.8:
            return y
        phi, n_arr, p_arr, _ = dev.fields_by_species(y, voltage_V=vk)
        n_safe = np.maximum(n_arr, 1.0e-250)
        p_safe = np.maximum(p_arr, 1.0e-250)
        transform = bmat(
            [
                [eye_n, None, None, None],
                [-eye_n / vt, diags(1.0 / n_safe), None, None],
                [-eye_n / vt, None, diags(-1.0 / p_safe), None],
                [None, None, None, eye_ion],
            ],
            format="csc",
        )
        dx = spsolve(step.jacobian(y) @ transform, -res)
        dphi = dx[:n_nodes]
        dn = dx[n_nodes : 2 * n_nodes]
        dp = dx[2 * n_nodes : 3 * n_nodes]
        du_ion = dx[3 * n_nodes :]
        alpha = 1.0
        sig_n = (dn < 0.0) & (n_arr > 1.0e-6)
        if np.any(sig_n):
            alpha = min(alpha, float(np.min(-0.95 * n_arr[sig_n] / dn[sig_n])))
        sig_p = (dp < 0.0) & (p_arr > 1.0e-6)
        if np.any(sig_p):
            alpha = min(alpha, float(np.min(-0.95 * p_arr[sig_p] / dp[sig_p])))
        max_dphi = float(np.max(np.abs(dphi)))
        if max_dphi * alpha > 0.25:
            alpha = 0.25 / max_dphi
        max_du_ion = float(np.max(np.abs(du_ion)))
        if max_du_ion * alpha > 1.0:
            alpha = 1.0 / max_du_ion
        phi_new = phi + alpha * dphi
        n_new = np.maximum(n_arr + alpha * dn, 1.0e-250)
        p_new = np.maximum(p_arr + alpha * dp, 1.0e-250)
        u_n_new = y[n_nodes : 2 * n_nodes] + np.log(n_new / n_safe) - (phi_new - phi) / vt
        u_p_new = (
            y[2 * n_nodes : 3 * n_nodes] - np.log(p_new / p_safe) - (phi_new - phi) / vt
        )
        phi_new[0], phi_new[-1] = y[0], y[n_nodes - 1]
        u_n_new[0], u_n_new[-1] = y[n_nodes], y[2 * n_nodes - 1]
        u_p_new[0], u_p_new[-1] = y[2 * n_nodes], y[3 * n_nodes - 1]
        y = np.r_[phi_new, u_n_new, u_p_new, y[3 * n_nodes :] + alpha * du_ion]
    return y


def _advance_ionic_be_interval(
    dev: Any,
    prev_state: Any,
    v_target: float,
    dt_total: float,
    settings: NewtonSettings,
    conv_norm: Any,
    eye_n: Any,
    eye_ion: Any,
    *,
    max_bisect: int = 10,
) -> tuple[Any, dict[str, Any], int]:
    v0 = float(prev_state.voltage_V)
    last_scaled = math.inf
    for level in range(max_bisect + 1):
        n_sub = 1 << level
        dt_sub = dt_total / n_sub
        cur = prev_state
        ok = True
        last_acc: dict[str, Any] | None = None
        for idx_sub in range(1, n_sub + 1):
            vk = v0 + (v_target - v0) * (idx_sub / n_sub)
            step = AssembledIonicBackwardEuler1DStep(
                AssembledIonic1DProblem(dev, vk, 1.0), cur, dt_sub
            )
            u0 = step.initial_coordinates("previous", voltage_lift=(vk != cur.voltage_V))
            nw = solve_newton(step, u0, settings, convergence_norm=conv_norm)
            if not nw.accepted:
                u_warm = _warm_density_newton_on_be_step(
                    dev, step, u0, vk, settings, conv_norm, eye_n, eye_ion
                )
                nw = solve_newton(step, u_warm, settings, convergence_norm=conv_norm)
            last_scaled = float(nw.scaled_residual)
            if not nw.accepted:
                ok = False
                break
            cur = dev.state(nw.state, voltage_V=vk)
            if idx_sub == n_sub:
                last_acc = step.accounting(nw.state)
        if ok and last_acc is not None:
            return cur, last_acc, level
    raise RuntimeError(
        f"BE step failed advancing {v0:.4f} -> {v_target:.4f} V over {dt_total:.3e} s "
        f"(last scaled_residual={last_scaled:.3e})"
    )


def _build_ionic_be_controls(dev: Any) -> tuple[NewtonSettings, Any, Any, Any]:
    n_nodes = dev.node_count
    m_ions = dev.ion_node_count
    n_ion_total = m_ions * dev.ion_species_count
    eye_n = eye(n_nodes, format="csc")
    eye_ion = eye(n_ion_total, format="csc")
    weights = np.r_[
        production_residual_scaling(
            dev.electronic_component,
            poisson_tolerance=1.0e-9,
            continuity_tolerance_A_m2=1.0e-3,
            contact_log_tolerance=1.0e-9,
        ).row_weights,
        np.full(n_ion_total, m_ions / 1.0e6),
    ]
    steps = np.r_[
        production_coordinates(
            dev.electronic_component, potential_step_V=0.2, quasi_fermi_step=3.0
        ).max_step,
        np.full(n_ion_total, 1.0),
    ]
    settings = NewtonSettings(
        ResidualScaling(weights),
        1.0,
        StepControl(steps),
        SolveBudget(50, 800, 50, 20),
        (),
        False,
    )
    elec_norm = production_convergence_norm(
        dev.electronic_component, convergence_policy=POISSON_CONTACT_BALANCE_V1
    )
    assert elec_norm is not None

    def conv_norm(sc: np.ndarray) -> float:
        return max(
            float(elec_norm(sc[: 3 * n_nodes])),
            *(
                math.fsum(abs(sc[slc]).tolist()) / m_ions
                for slc in dev.ion_species_slices
            ),
        )

    return settings, conv_norm, eye_n, eye_ion


def _run_transient_jv_variant(
    repo_root: Path,
    variant_spec: dict[str, Any],
    defaults: DefaultCatalog,
    resources: ResourceLibrary,
) -> dict[str, Any]:
    cfg_path = repo_root / variant_spec["config_path"]
    yaml_bytes = cfg_path.read_bytes()
    actual_sha = hashlib.sha256(yaml_bytes).hexdigest()
    if actual_sha != variant_spec["config_sha256"]:
        raise ValueError(
            f"config SHA-256 mismatch for {cfg_path}: {actual_sha} != {variant_spec['config_sha256']}"
        )
    args = variant_spec["arguments"]
    n_grid = int(args["N_grid"])
    n_points = int(args["n_points"])
    v_rate = float(args["v_rate"])
    v_max = float(args["V_max"])

    timing = RuntimeTiming()
    outer_t0 = time.perf_counter()
    inner_t0 = time.perf_counter()
    with timing.phase("recorded_run"):
        with timing.phase("controller"):
            prepared, has_ions, ion_layer_id, has_opt = _prepare_standard_yaml_device(
                yaml_bytes,
                source_id=variant_spec["config_path"],
                device_id=variant_spec["id"],
                defaults=defaults,
                resources=resources,
            )
            if not has_ions or ion_layer_id is None:
                raise ValueError(f"{variant_spec['id']} expected mobile-ion absorber")
            elec_layers = [
                lyr for lyr in prepared.layers if lyr.role != "substrate"
            ]
            intervals = max(1, round((n_grid - 1) / len(elec_layers)))
            with legacy_benchmark_deck_scope():
                dev = compile_ionic_device_1d(
                    prepared,
                    intervals_per_layer=intervals,
                    ion_domain_layer=ion_layer_id,
                    ion_domain_endpoints=("blocking", "blocking"),
                    optical_model="tmm" if has_opt else None,
                )
            settings, conv_norm, eye_n, eye_ion = _build_ionic_be_controls(dev)
            state = _prepare_dark_quasineutral_ionic_state(dev, 0.0)
            bisection_total = 0
            # 1 ms initial illuminated settle from uniform P = P_0
            for dt_settle in (1.0e-8, 1.0e-6, 9.899e-5, 9.0e-4):
                state, acc0, b_lvl = _advance_ionic_be_interval(
                    dev, state, 0.0, dt_settle, settings, conv_norm, eye_n, eye_ion
                )
                bisection_total += b_lvl
            init_inv = float(
                np.dot(dev.ion_geometry.cell_measures_m, state.positive_ion_density_m3)
            )
            max_inv_drift = 0.0
            polarity = float(dev.electronic_component.voltage_polarity)
            j0 = float(
                -polarity
                * np.asarray(acc0["terminal_total_current_density_A_m2"], dtype=float)[0]
            )
            v_fwd = np.linspace(0.0, v_max, n_points)
            j_fwd: np.ndarray = np.empty(n_points, dtype=float)
            j_fwd[0] = j0
            dt_pt = (v_max / (n_points - 1)) / v_rate
            for idx in range(1, n_points):
                state, acc, b_lvl = _advance_ionic_be_interval(
                    dev,
                    state,
                    float(v_fwd[idx]),
                    dt_pt,
                    settings,
                    conv_norm,
                    eye_n,
                    eye_ion,
                )
                bisection_total += b_lvl
                j_fwd[idx] = float(
                    -polarity
                    * np.asarray(
                        acc["terminal_total_current_density_A_m2"], dtype=float
                    )[0]
                )
                cur_inv = float(
                    np.dot(
                        dev.ion_geometry.cell_measures_m, state.positive_ion_density_m3
                    )
                )
                max_inv_drift = max(
                    max_inv_drift, abs(cur_inv - init_inv) / max(init_inv, 1.0e-30)
                )

            v_rev = v_fwd[::-1].copy()
            j_rev: np.ndarray = np.empty(n_points, dtype=float)
            j_rev[0] = j_fwd[-1]
            for idx in range(1, n_points):
                state, acc, b_lvl = _advance_ionic_be_interval(
                    dev,
                    state,
                    float(v_rev[idx]),
                    dt_pt,
                    settings,
                    conv_norm,
                    eye_n,
                    eye_ion,
                )
                bisection_total += b_lvl
                j_rev[idx] = float(
                    -polarity
                    * np.asarray(
                        acc["terminal_total_current_density_A_m2"], dtype=float
                    )[0]
                )
                cur_inv = float(
                    np.dot(
                        dev.ion_geometry.cell_measures_m, state.positive_ion_density_m3
                    )
                )
                max_inv_drift = max(
                    max_inv_drift, abs(cur_inv - init_inv) / max(init_inv, 1.0e-30)
                )

        with timing.phase("observation"):
            mf = steady_jv_metrics(v_fwd, j_fwd, incident_power_W_m2=1000.0)
            mr = steady_jv_metrics(v_rev, j_rev, incident_power_W_m2=1000.0)
            hi_norm = (
                float(
                    (mr.maximum_power_W_m2 - mf.maximum_power_W_m2)
                    / mr.maximum_power_W_m2
                )
                if (
                    mf.maximum_power_W_m2 is not None
                    and mr.maximum_power_W_m2 is not None
                    and mr.maximum_power_W_m2 > 0.0
                )
                else None
            )
    inner_elapsed = time.perf_counter() - inner_t0
    timing_snap = timing.snapshot("returned_completed")
    obs_now = time.perf_counter()
    ident = {"case": variant_spec["id"], "N_grid": str(n_grid)}
    outer_rec = reconcile_outer_timing(
        expected_identity=ident,
        actual_identity=ident,
        clock_names=("time.perf_counter",) * 3,
        outer_started_s=outer_t0,
        inner_started_s=inner_t0,
        inner_elapsed_s=inner_elapsed,
        observed_interval_s=(obs_now, obs_now),
        os_elapsed_s=obs_now - outer_t0,
        whole_limit_s=900.0,
        inner_passed=True,
        cleanup_positive=True,
        returned_code=0,
    )
    voc_bracketed = bool(
        mf.open_circuit_voltage_V is not None
        and mr.open_circuit_voltage_V is not None
        and np.any(j_fwd < 0.0)
        and np.any(j_rev < 0.0)
    )
    return {
        "variant_id": variant_spec["id"],
        "completed": voc_bracketed,
        "configured_N_grid": n_grid,
        "electrical_node_count": int(dev.node_count),
        "ion_node_count": int(dev.ion_node_count),
        "assembled_unknown_count": int(dev.state_size),
        "compiled_identity": dev.identity,
        "elapsed_capture_s": inner_elapsed,
        "bisection_subdivisions": bisection_total,
        "max_inventory_relative_drift": max_inv_drift,
        "voc_bracketed_both_branches": voc_bracketed,
        "metrics_forward": {
            "Voc_V": mf.open_circuit_voltage_V,
            "Jsc_A_m2": mf.short_circuit_current_A_m2,
            "FF": mf.fill_factor,
            "PCE": mf.efficiency,
            "Pmax_W_m2": mf.maximum_power_W_m2,
        },
        "metrics_reverse": {
            "Voc_V": mr.open_circuit_voltage_V,
            "Jsc_A_m2": mr.short_circuit_current_A_m2,
            "FF": mr.fill_factor,
            "PCE": mr.efficiency,
            "Pmax_W_m2": mr.maximum_power_W_m2,
        },
        "HI_normalized": hi_norm,
        "runtime_timing": timing_snap,
        "outer_timing": outer_rec,
    }


def _run_scaps_steady_perf(
    case_spec: dict[str, Any],
    defaults: DefaultCatalog,
    resources: ResourceLibrary,
) -> dict[str, Any]:
    preset_src = next(
        doc for doc in load_packaged_preset_sources() if doc.id == "scaps_mirror_v2.yaml"
    )
    args = case_spec["arguments"]
    n_grid = int(args["N_grid"])
    v_max = float(args["V_max"])
    n_points = int(args["n_points"])
    stop_after_voc = bool(args["stop_after_voc"])

    timing = RuntimeTiming()
    outer_t0 = time.perf_counter()
    inner_t0 = time.perf_counter()
    with timing.phase("recorded_run"):
        with timing.phase("controller"):
            prepared = _prepare_scaps_yaml_device(
                preset_src.content,
                source_id=preset_src.id,
                device_id="scaps_mirror_v2",
                defaults=defaults,
                resources=resources,
            )
            elec_layers = [
                lyr for lyr in prepared.layers if lyr.role != "substrate"
            ]
            intervals = max(1, round((n_grid - 1) / len(elec_layers)))
            compiled = compile_device_1d(
                prepared,
                intervals_per_layer=intervals,
                interface_topology="two_sided_trace",
                geometry_name=PHYSICAL_BOUNDARY_TWO_SIDED_V1,
                thermionic_mode=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
                allow_legacy_benchmark_deck=True,
            )
            v_targets = np.linspace(0.0, v_max, n_points)
            res = run_steady_jv(
                compiled,
                v_targets,
                generation_scale=1.0,
                incident_power_W_m2=1000.0,
                options=SteadyOptions(
                    continuity_tolerance_A_m2=1.0e-4,
                    current_spread_tolerance_A_m2=1.0e-4,
                ),
                stop_after_voc=stop_after_voc,
                max_bias_bisect=4,
            )
        with timing.phase("observation"):
            m = res.metrics
            stopped_early = int(res.voltage_V.size) < n_points
    inner_elapsed = time.perf_counter() - inner_t0
    timing_snap = timing.snapshot("returned_completed")
    obs_now = time.perf_counter()
    ident = {"case": "ScapsSteady", "N_grid": str(n_grid)}
    outer_rec = reconcile_outer_timing(
        expected_identity=ident,
        actual_identity=ident,
        clock_names=("time.perf_counter",) * 3,
        outer_started_s=outer_t0,
        inner_started_s=inner_t0,
        inner_elapsed_s=inner_elapsed,
        observed_interval_s=(obs_now, obs_now),
        os_elapsed_s=obs_now - outer_t0,
        whole_limit_s=600.0,
        inner_passed=True,
        cleanup_positive=True,
        returned_code=0,
    )
    return {
        "completed": bool(m.open_circuit_voltage_V is not None),
        "configured_N_grid": n_grid,
        "electrical_node_count": int(compiled.node_count),
        "assembled_unknown_count": int(3 * compiled.node_count),
        "compiled_identity": compiled.identity,
        "target_bias_count": n_points,
        "evaluated_bias_count": int(res.voltage_V.size),
        "stopped_after_voc": stopped_early,
        "last_evaluated_voltage_V": float(res.voltage_V[-1]),
        "elapsed_capture_s": inner_elapsed,
        "metrics": {
            "Voc_V": m.open_circuit_voltage_V,
            "Jsc_mA_cm2": (
                None
                if m.short_circuit_current_A_m2 is None
                else float(m.short_circuit_current_A_m2) * 0.1
            ),
            "FF_percent": None if m.fill_factor is None else float(m.fill_factor) * 100.0,
            "PCE_percent": (
                None
                if m.efficiency is None
                else float(m.efficiency) * 100.0
            ),
        },
        "runtime_timing": timing_snap,
        "outer_timing": outer_rec,
    }


def _solve_certified_ionic_dc_point(
    dev: Any, voltage_V: float, generation_scale: float
) -> tuple[Any, IonicSteadyOptions]:
    n_nodes = dev.node_count
    m_ions = dev.ion_node_count
    n_ion_total = m_ions * dev.ion_species_count
    orig = AssembledIonic1DProblem(dev, voltage_V, generation_scale)
    cprob = AssembledConstrainedIonic1DProblem(orig, m_ions - 1)
    y = dev.initial_state(voltage_V=voltage_V).u.copy()
    for _ in range(25):
        r_p = orig.residual(y)[:n_nodes]
        if float(np.max(np.abs(r_p))) < 1.0e-12:
            break
        dphi = spsolve(orig.jacobian(y)[:n_nodes, :n_nodes], -r_p)
        y[:n_nodes] += min(1.0, 0.2 / max(float(np.max(np.abs(dphi))), 1.0e-30)) * dphi
    vt = dev.electronic_component.thermal_voltage_V
    eye_n = eye(n_nodes, format="csc")
    eye_ion = eye(n_ion_total, format="csc")
    dt = 1.0e-9
    for _ in range(60):
        phi, n_arr, p_arr, _ = dev.fields_by_species(y, voltage_V=voltage_V)
        res = cprob.residual(y)
        j_u = cprob.jacobian(y)
        transform = bmat(
            [
                [eye_n, None, None, None],
                [-eye_n / vt, diags(1.0 / n_arr), None, None],
                [-eye_n / vt, None, diags(-1.0 / p_arr), None],
                [None, None, None, eye_ion],
            ],
            format="csc",
        )
        s_lil = (orig.storage_jacobian(y) @ transform).tolil()
        for r_idx in cprob.constraint_rows.values():
            s_lil.rows[r_idx] = []
            s_lil.data[r_idx] = []
        dx = spsolve(j_u @ transform - (1.0 / dt) * s_lil.tocsc(), -res)
        dphi = dx[:n_nodes]
        dn = dx[n_nodes : 2 * n_nodes]
        dp = dx[2 * n_nodes : 3 * n_nodes]
        alpha = 1.0
        if np.any(dn < 0.0):
            alpha = min(alpha, float(np.min(-0.9 * n_arr[dn < 0.0] / dn[dn < 0.0])))
        if np.any(dp < 0.0):
            alpha = min(alpha, float(np.min(-0.9 * p_arr[dp < 0.0] / dp[dp < 0.0])))
        max_dphi = float(np.max(np.abs(dphi)))
        if max_dphi * alpha > 0.2:
            alpha = 0.2 / max_dphi
        phi_new = phi + alpha * dphi
        n_new = n_arr + alpha * dn
        p_new = p_arr + alpha * dp
        u_n_new = y[n_nodes : 2 * n_nodes] + np.log(n_new / n_arr) - (phi_new - phi) / vt
        u_p_new = (
            y[2 * n_nodes : 3 * n_nodes] - np.log(p_new / p_arr) - (phi_new - phi) / vt
        )
        y = np.r_[phi_new, u_n_new, u_p_new, y[3 * n_nodes :]]
        if alpha > 0.2:
            dt = min(dt * 3.0, 1.0e12)

    opts_tight = IonicSteadyOptions(
        electronic=SteadyOptions(
            continuity_tolerance_A_m2=1.0e-4, current_spread_tolerance_A_m2=1.0e-4
        ),
        inventory_relative_tolerance=1.0e-12,
        ion_balance_tolerance_m2_s=1.0e6,
        ion_flux_tolerance_m2_s=1.0e6,
        ion_logit_step=1.0,
        inventory_row_node=max(m_ions - 1, 0),
        budget=SolveBudget(80, 800, 80, 24),
        wall_s=30.0,
    )
    r1 = solve_ionic_steady_state(
        dev,
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        options=opts_tight,
        initial_state=dev.state(y, voltage_V=voltage_V),
    )
    if r1.accepted:
        return r1, opts_tight
    opts_cert = IonicSteadyOptions(
        electronic=SteadyOptions(
            continuity_tolerance_A_m2=1.0e-3, current_spread_tolerance_A_m2=1.0e-3
        ),
        inventory_relative_tolerance=1.0e-12,
        ion_balance_tolerance_m2_s=1.0e4,
        ion_flux_tolerance_m2_s=1.0e4,
        ion_logit_step=1.0,
        inventory_row_node=max(m_ions - 1, 0),
        budget=SolveBudget(80, 800, 80, 24),
        wall_s=30.0,
    )
    r2 = solve_ionic_steady_state(
        dev,
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        options=opts_cert,
        initial_state=dev.state(r1.candidate_coordinates, voltage_V=voltage_V),
    )
    return r2, opts_cert


def _run_ac60_perf(
    repo_root: Path,
    case_spec: dict[str, Any],
    defaults: DefaultCatalog,
    resources: ResourceLibrary,
) -> dict[str, Any]:
    cfg_path = repo_root / case_spec["config_path"]
    yaml_bytes = cfg_path.read_bytes()
    n_grid = int(case_spec["configured_N_grid"])
    op_spec = case_spec["operating_point"]
    v_dc = float(op_spec["V_dc_V"])
    delta_v = float(op_spec["delta_V_V"])
    frequencies = [float(f) for f in case_spec["frequencies_Hz"]]

    # One-time DC preparation (reported separately per PerformanceCasesV1 timing_boundary)
    prep_t0 = time.perf_counter()
    prepared, _, ion_layer_id, _ = _prepare_standard_yaml_device(
        yaml_bytes,
        source_id=case_spec["config_path"],
        device_id="ionmonger_ac60",
        defaults=defaults,
        resources=resources,
    )
    assert ion_layer_id is not None
    elec_layers = [lyr for lyr in prepared.layers if lyr.role != "substrate"]
    intervals = max(1, round((n_grid - 1) / len(elec_layers)))
    with legacy_benchmark_deck_scope():
        dev_prep = compile_ionic_device_1d(
            prepared,
            intervals_per_layer=intervals,
            ion_domain_layer=ion_layer_id,
            ion_domain_endpoints=("blocking", "blocking"),
        )
    dc_res, dc_opts = _solve_certified_ionic_dc_point(dev_prep, v_dc, 1.0)
    init_packet = package_ionic_steady_initial(
        dev_prep, dc_res, voltage_V=v_dc, generation_scale=1.0, controls=dc_opts
    )
    dc_preparation_s = time.perf_counter() - prep_t0

    # Primary timed AC60 spectrum including fresh device/operator compilation
    timing = RuntimeTiming()
    outer_t0 = time.perf_counter()
    inner_t0 = time.perf_counter()
    with timing.phase("recorded_run"):
        with timing.phase("controller"):
            with legacy_benchmark_deck_scope():
                dev_ac = compile_ionic_device_1d(
                    prepared,
                    intervals_per_layer=intervals,
                    ion_domain_layer=ion_layer_id,
                    ion_domain_endpoints=("blocking", "blocking"),
                )
            ac_opts = IonicImpedanceOptions(
                linear=FrequencyOptions(relative_tolerance=1.0e-5),
                current_absolute_tolerance_A_m2_V=5.0e-4,
                inventory_relative_tolerance_per_V=1.0e-8,
            )
            ac_res = solve_ionic_impedance(
                dev_ac,
                init_packet,
                frequencies,
                voltage_phasor_V=delta_v,
                options=ac_opts,
            )
        with timing.phase("observation"):
            max_lin_be = max(
                float(att.linear_solution.componentwise_backward_error)
                for att in ac_res.attempts
                if att.linear_solution is not None
            )
            max_orig_be = max(
                float(att.observation["original_componentwise_backward_error"])
                for att in ac_res.attempts
                if att.observation is not None
            )
            max_spread = max(
                float(att.observation["consumer_Maxwell_spread_A_m2_V"])
                for att in ac_res.attempts
                if att.observation is not None
            )
            max_inv = max(
                float(att.observation["consumer_inventory_relative_per_V"]["positive"])
                for att in ac_res.attempts
                if att.observation is not None
            )
    inner_elapsed = time.perf_counter() - inner_t0
    timing_snap = timing.snapshot(
        "returned_completed" if ac_res.completed else "returned_other_status"
    )
    obs_now = time.perf_counter()
    ident = {"case": "AC60", "N_grid": str(n_grid)}
    outer_rec = reconcile_outer_timing(
        expected_identity=ident,
        actual_identity=ident,
        clock_names=("time.perf_counter",) * 3,
        outer_started_s=outer_t0,
        inner_started_s=inner_t0,
        inner_elapsed_s=inner_elapsed,
        observed_interval_s=(obs_now, obs_now),
        os_elapsed_s=obs_now - outer_t0,
        whole_limit_s=600.0,
        inner_passed=bool(ac_res.completed),
        cleanup_positive=True,
        returned_code=0 if ac_res.completed else 1,
    )
    return {
        "completed": bool(ac_res.completed),
        "configured_N_grid": n_grid,
        "electrical_node_count": int(dev_ac.node_count),
        "ion_node_count": int(dev_ac.ion_node_count),
        "assembled_unknown_count": int(dev_ac.state_size),
        "compiled_identity": dev_ac.identity,
        "requested_frequency_count": len(frequencies),
        "accepted_frequency_count": sum(1 for att in ac_res.attempts if att.accepted),
        "dc_preparation_s_excluded_from_primary": dc_preparation_s,
        "elapsed_capture_s": inner_elapsed,
        "whole_workflow_s": dc_preparation_s + inner_elapsed,
        "max_linear_backward_error": max_lin_be,
        "max_original_rows_backward_error": max_orig_be,
        "max_all_face_maxwell_spread_A_m2_V": max_spread,
        "max_inventory_relative_per_V": max_inv,
        "runtime_timing": timing_snap,
        "outer_timing": outer_rec,
    }


def _dwell_time_grid(duration_s: float, divisor: int) -> np.ndarray:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    n_unif = max(int(divisor), 4)
    dt_unif = duration_s / n_unif
    t_early = min(1.0e-10, dt_unif * 0.125)
    if t_early < dt_unif * 0.5:
        early: np.ndarray = np.geomspace(t_early, dt_unif, 6, dtype=np.float64)
        if n_unif == 1:
            return np.r_[0.0, early]
        tail: np.ndarray = np.linspace(dt_unif, duration_s, n_unif, dtype=np.float64)[1:]
        return np.r_[0.0, early, tail]
    return np.linspace(0.0, duration_s, n_unif + 1, dtype=np.float64)


def _bake_electronic_reabsorption_2d(
    device: CompiledDevice2D,
    state: State2D,
    rr_specs: tuple[RadiativeReabsorptionSpec2D, ...],
) -> CompiledDevice2D:
    g = device.geometry
    shape = g.shape
    _, n_arr, p_arr = device.fields(state.u, voltage_V=state.voltage_V)
    b_rad = device.recombination.coefficients.radiative_m3_s.reshape(shape)
    ni_sq = device.recombination.coefficients.ni_sq_m6.reshape(shape)
    base_g = device.generation_source_m3_s.reshape(shape)
    aug = recompute_g_with_rad_2d(
        G_optical=base_g,
        n=n_arr.reshape(shape),
        p=p_arr.reshape(shape),
        B_rad=b_rad,
        ni_sq=ni_sq,
        x=g.x_m,
        y=g.y_geometry.nodes_m,
        absorber_y_ranges=tuple((s.y_start, s.y_stop) for s in rr_specs),
        absorber_p_esc=tuple(s.escape_probability for s in rr_specs),
        absorber_areas=tuple(s.area_m2 for s in rr_specs),
    )
    gen_y = np.mean(aug, axis=1)
    new_gen = replace(device.generation_y, source_m3_s=frozen_array(gen_y))
    baked_sha = hashlib.sha256(
        device.identity.encode()
        + b":baked_rr_2d:"
        + np.ascontiguousarray(gen_y, dtype="<f8").tobytes()
    ).hexdigest()
    return replace(device, sha256=baked_sha, generation_y=new_gen)


def _run_spatial_2d_stage_a_variant(
    repo_root: Path,
    nx: int,
    ny_per_layer: int,
    defaults: DefaultCatalog,
    resources: ResourceLibrary,
) -> dict[str, Any]:
    cfg_rel = "perovskite-sim/tests/fixtures/configs/twod/nip_MAPbI3_uniform.yaml"
    yaml_bytes = (repo_root / cfg_rel).read_bytes()
    timing = RuntimeTiming()
    outer_t0 = time.perf_counter()
    inner_t0 = time.perf_counter()
    with timing.phase("recorded_run"):
        with timing.phase("controller"):
            prepared, _, ion_layer_id, _ = _prepare_standard_yaml_device(
                yaml_bytes,
                source_id=cfg_rel,
                device_id=f"twod_stage_a_nx{nx}",
                defaults=defaults,
                resources=resources,
            )
            assert ion_layer_id is not None
            geom_spatial = JV2DInput(
                kind="jv_2d",
                lateral_length=5.0e-7,
                Nx=nx,
                Ny_per_layer=ny_per_layer,
                lateral_bc="periodic",
                microstructure=MicrostructureInput(grain_boundaries=()),
            )
            base_clean = project_prepared_for_electronic_2d(
                prepared, strip_ions=True, strip_interfaces=False
            )
            elec_layers = [lyr for lyr in prepared.layers if lyr.role != "substrate"]
            with legacy_benchmark_deck_scope():
                dev_2d = compile_device_2d(
                    base_clean, geom_spatial, allow_legacy_benchmark_deck=True
                )
                rr_specs = build_radiative_reabsorption_2d(prepared, dev_2d)
                ion_1d = compile_ionic_device_1d(
                    prepared,
                    intervals_per_layer=(ny_per_layer,) * len(elec_layers),
                    ion_domain_layer=ion_layer_id,
                    ion_domain_endpoints=("blocking", "blocking"),
                )
            settings_1d, conv_norm_1d, eye_n, eye_ion = _build_ionic_be_controls(ion_1d)
            st_1d = _prepare_dark_quasineutral_ionic_state(ion_1d, 0.0)
            for dt_settle in (1.0e-8, 1.0e-6, 9.899e-5, 9.0e-4):
                st_1d, _, _ = _advance_ionic_be_interval(
                    ion_1d, st_1d, 0.0, dt_settle, settings_1d, conv_norm_1d, eye_n, eye_ion
                )
            n_1d = ion_1d.node_count
            excess_1d: np.ndarray = np.zeros(n_1d, dtype=float)
            excess_1d[ion_1d.ion_node_indices] = (
                st_1d.positive_ion_density_m3 - ion_1d.positive_ion_reference_m3
            ) * ion_1d.ion_charge_weight_ratio
            dev_2d = embed_frozen_ion_excess_2d(dev_2d, excess_1d)
            elec_state = dev_2d.state(
                dev_2d.extrude_coordinates(st_1d.u[: 3 * n_1d]), voltage_V=0.0
            )
            d_grid = _dwell_time_grid(1.0e-3, 50)
            be_opts = BackwardEuler2DOptions(
                electronic=SteadyOptions(
                    continuity_tolerance_A_m2=1.0e-3,
                    current_spread_tolerance_A_m2=1.0e-3,
                ),
                budget=SolveBudget(50, 800, 50, 20),
                max_bisect=6,
                radiative_reabsorption=rr_specs,
            )
            voltages = np.round(np.arange(0.0, 1.2 + 1.0e-9, 0.1), 10)
            j_vals: list[float] = []
            max_lat_var = 0.0
            for fv in voltages:
                fv_float = float(fv)
                v_fn = lambda _t, v=fv_float: v  # noqa: E731
                g_fn = lambda _t: 1.0  # noqa: E731
                run_pt = run_backward_euler_2d(
                    dev_2d,
                    elec_state,
                    d_grid,
                    voltage_fn=v_fn,
                    generation_fn=g_fn,
                    options=be_opts,
                )
                if not run_pt.accepted and rr_specs:
                    baked_elec = _bake_electronic_reabsorption_2d(
                        dev_2d, elec_state, rr_specs
                    )
                    baked_s0 = baked_elec.state(
                        elec_state.u, voltage_V=elec_state.voltage_V
                    )
                    run_baked = run_backward_euler_2d(
                        baked_elec,
                        baked_s0,
                        d_grid,
                        voltage_fn=v_fn,
                        generation_fn=g_fn,
                        options=replace(be_opts, radiative_reabsorption=()),
                    )
                    if run_baked.accepted:
                        elec_state = dev_2d.state(
                            run_baked.states[-1].u, voltage_V=fv_float
                        )
                        run_pt = replace(
                            run_baked, states=(*run_baked.states[:-1], elec_state)
                        )
                if not run_pt.accepted:
                    raise RuntimeError(
                        f"Spatial2DStageA Nx={nx} failed at V={fv_float}: {run_pt.termination}"
                    )
                elec_state = run_pt.states[-1]
                snap = extract_snapshot_2d(
                    dev_2d,
                    elec_state,
                    generation_scale=1.0,
                    radiative_reabsorption=rr_specs,
                )
                max_lat_var = max(
                    max_lat_var, float(np.max(np.ptp(snap.phi, axis=1)))
                )
                ports_j = compute_carrier_only_terminal_current_2d(
                    dev_2d,
                    elec_state,
                    generation_scale=1.0,
                    radiative_reabsorption=rr_specs,
                )
                j_vals.append(float(ports_j[1]))
        with timing.phase("observation"):
            j_out = -float(dev_2d.column.voltage_polarity) * np.asarray(
                j_vals, dtype=float
            )
            m = steady_jv_metrics(voltages, j_out, incident_power_W_m2=1000.0)
    inner_elapsed = time.perf_counter() - inner_t0
    timing_snap = timing.snapshot("returned_completed")
    obs_now = time.perf_counter()
    ident = {"case": "Spatial2DStageA", "Nx": str(nx)}
    outer_rec = reconcile_outer_timing(
        expected_identity=ident,
        actual_identity=ident,
        clock_names=("time.perf_counter",) * 3,
        outer_started_s=outer_t0,
        inner_started_s=inner_t0,
        inner_elapsed_s=inner_elapsed,
        observed_interval_s=(obs_now, obs_now),
        os_elapsed_s=obs_now - outer_t0,
        whole_limit_s=3600.0,
        inner_passed=bool(max_lat_var < 1.0e-9),
        cleanup_positive=True,
        returned_code=0 if max_lat_var < 1.0e-9 else 1,
    )
    return {
        "variant_id": f"Nx{nx}",
        "completed": bool(max_lat_var < 1.0e-9 and m.open_circuit_voltage_V is not None),
        "configured_Nx": nx,
        "configured_Ny_per_layer": ny_per_layer,
        "actual_Nx": int(dev_2d.geometry.nx),
        "actual_Ny": int(dev_2d.geometry.ny),
        "physical_node_count": int(dev_2d.node_count),
        "assembled_unknown_count": int(3 * dev_2d.node_count),
        "compiled_identity": dev_2d.identity,
        "bias_point_count": int(voltages.size),
        "max_lateral_potential_variation_V": max_lat_var,
        "lateral_variation_passed": bool(max_lat_var < 1.0e-9),
        "elapsed_capture_s": inner_elapsed,
        "metrics": {
            "Voc_V": m.open_circuit_voltage_V,
            "Jsc_A_m2": m.short_circuit_current_A_m2,
            "FF": m.fill_factor,
            "PCE": m.efficiency,
        },
        "runtime_timing": timing_snap,
        "outer_timing": outer_rec,
    }


def _compute_spatial_growth_summary(
    scale_records: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(scale_records) < 2:
        return {"status": "insufficient_scales_for_pair_exponent", "pairs": []}
    unknowns = [int(rec["assembled_unknown_count"]) for rec in scale_records]
    if any(unknowns[idx + 1] <= unknowns[idx] for idx in range(len(unknowns) - 1)):
        raise ValueError(
            f"Spatial2DStageA actual unknown counts must be strictly increasing, got {unknowns}"
        )
    pairs: list[dict[str, Any]] = []
    for i in range(len(scale_records)):
        for j in range(i + 1, len(scale_records)):
            small, large = scale_records[i], scale_records[j]
            u_s = float(small["assembled_unknown_count"])
            u_l = float(large["assembled_unknown_count"])
            denom = math.log(u_l / u_s)
            t_s_med = float(small["median_elapsed_capture_s"])
            t_l_med = float(large["median_elapsed_capture_s"])
            t_s_all = [float(x) for x in small["repeat_elapsed_capture_s"]]
            t_l_all = [float(x) for x in large["repeat_elapsed_capture_s"]]
            beta_med = math.log(t_l_med / t_s_med) / denom
            beta_low = math.log(min(t_l_all) / max(t_s_all)) / denom
            beta_high = math.log(max(t_l_all) / min(t_s_all)) / denom
            pairs.append(
                {
                    "pair": f"{small['variant_id']}->{large['variant_id']}",
                    "unknowns_small": int(u_s),
                    "unknowns_large": int(u_l),
                    "beta_median": beta_med,
                    "beta_low": beta_low,
                    "beta_high": beta_high,
                }
            )
    worst_beta = max(p["beta_median"] for p in pairs)
    worst_interval = [
        max(p["beta_low"] for p in pairs),
        max(p["beta_high"] for p in pairs),
    ]
    return {
        "status": "measured_candidate_only_baseline_pending",
        "pairs": pairs,
        "worst_pair_beta_median": worst_beta,
        "worst_pair_beta_interval": worst_interval,
        "candidate_upper_below_2_5": bool(worst_interval[1] < 2.5),
    }


def run_perf_candidate(
    *,
    repo_root: Path,
    selected_cases: tuple[str, ...] | None = None,
    repeats: int = 1,
    spatial_nx: tuple[int, ...] = (4, 8, 16),
    transient_variants: tuple[str, ...] | None = None,
    require_single_thread: bool = False,
    assert_zero_legacy_modules: bool = True,
) -> dict[str, Any]:
    pre_legacy = _check_legacy_modules_entry(
        assert_zero_legacy_modules=assert_zero_legacy_modules
    )
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    perf_path = repo_root / PERF_CONTRACT_REL
    perf_sha = _file_sha256(perf_path)
    perf_doc = json.loads(perf_path.read_text(encoding="utf-8"))
    cases_by_id = {c["id"]: c for c in perf_doc["cases"]}
    target_ids = tuple(cases_by_id) if selected_cases is None else selected_cases
    for cid in target_ids:
        if cid not in cases_by_id:
            raise ValueError(f"unknown PerformanceCasesV1 case id: {cid!r}")

    defaults, resources = _load_packaged_catalog_and_resources()
    results: dict[str, Any] = {}

    for cid in target_ids:
        spec = cases_by_id[cid]
        if cid in {"TransientJvN80", "TransientJvN200"}:
            variants_out = []
            for v_spec in spec["variants"]:
                if (
                    transient_variants is not None
                    and v_spec["id"] not in transient_variants
                ):
                    continue
                runs = [
                    _run_transient_jv_variant(repo_root, v_spec, defaults, resources)
                    for _ in range(repeats)
                ]
                times = [float(r["elapsed_capture_s"]) for r in runs]
                variants_out.append(
                    {
                        "variant_id": v_spec["id"],
                        "assembled_unknown_count": runs[0]["assembled_unknown_count"],
                        "electrical_node_count": runs[0]["electrical_node_count"],
                        "ion_node_count": runs[0]["ion_node_count"],
                        "repeat_elapsed_capture_s": times,
                        "median_elapsed_capture_s": statistics.median(times),
                        "worst_elapsed_capture_s": max(times),
                        "all_repeats_completed": all(r["completed"] for r in runs),
                        "repeats": runs,
                    }
                )
            results[cid] = {
                "case_id": cid,
                "minimum_speedup_target": spec["minimum_speedup"],
                "variants": variants_out,
            }
        elif cid == "ScapsSteady":
            runs = [
                _run_scaps_steady_perf(spec, defaults, resources)
                for _ in range(repeats)
            ]
            times = [float(r["elapsed_capture_s"]) for r in runs]
            results[cid] = {
                "case_id": cid,
                "minimum_speedup_target": spec["minimum_speedup"],
                "assembled_unknown_count": runs[0]["assembled_unknown_count"],
                "electrical_node_count": runs[0]["electrical_node_count"],
                "repeat_elapsed_capture_s": times,
                "median_elapsed_capture_s": statistics.median(times),
                "worst_elapsed_capture_s": max(times),
                "all_repeats_completed": all(r["completed"] for r in runs),
                "repeats": runs,
            }
        elif cid == "AC60":
            runs = [
                _run_ac60_perf(repo_root, spec, defaults, resources)
                for _ in range(repeats)
            ]
            times = [float(r["elapsed_capture_s"]) for r in runs]
            results[cid] = {
                "case_id": cid,
                "minimum_speedup_target": spec["minimum_speedup"],
                "assembled_unknown_count": runs[0]["assembled_unknown_count"],
                "electrical_node_count": runs[0]["electrical_node_count"],
                "ion_node_count": runs[0]["ion_node_count"],
                "repeat_elapsed_capture_s": times,
                "median_elapsed_capture_s": statistics.median(times),
                "worst_elapsed_capture_s": max(times),
                "all_repeats_completed": all(r["completed"] for r in runs),
                "repeats": runs,
            }
        elif cid == "Spatial2DStageA":
            scale_records = []
            for nx_val in spatial_nx:
                runs = [
                    _run_spatial_2d_stage_a_variant(
                        repo_root, int(nx_val), 10, defaults, resources
                    )
                    for _ in range(repeats)
                ]
                times = [float(r["elapsed_capture_s"]) for r in runs]
                scale_records.append(
                    {
                        "variant_id": f"Nx{nx_val}",
                        "configured_Nx": int(nx_val),
                        "assembled_unknown_count": runs[0]["assembled_unknown_count"],
                        "physical_node_count": runs[0]["physical_node_count"],
                        "repeat_elapsed_capture_s": times,
                        "median_elapsed_capture_s": statistics.median(times),
                        "worst_elapsed_capture_s": max(times),
                        "all_repeats_completed": all(r["completed"] for r in runs),
                        "repeats": runs,
                    }
                )
            results[cid] = {
                "case_id": cid,
                "minimum_speedup_target": spec["minimum_speedup"],
                "scales": scale_records,
                "growth": _compute_spatial_growth_summary(scale_records),
            }

    _check_legacy_modules_exit(
        pre_legacy, assert_zero_legacy_modules=assert_zero_legacy_modules
    )
    return {
        "schema": SCHEMA_PERF,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "performance_contract_sha256": perf_sha,
        "repeats_per_case": repeats,
        "environment": _host_environment_record(require_single_thread),
        "cases": results,
        "scientific_qualification": False,
        "physical_certificate_status": "not_evaluated",
    }


def _decimal_ulp(value: float, floor_val: float) -> float:
    dec = Decimal(str(value)).normalize()
    exp = dec.as_tuple().exponent
    ulp = float(Decimal(10) ** exp) if isinstance(exp, int) and exp < 0 else floor_val
    return max(ulp, floor_val)


def _evaluate_scaps_point_worker(task: dict[str, Any]) -> dict[str, Any]:
    defaults, resources = _load_packaged_catalog_and_resources()
    case_id = str(task["id"])
    group_id = str(task["group_id"])
    coords = list(task["coordinates"])
    ref_status = str(task["reference_status"])
    ref_values: dict[str, float | None] | None = task["reference_values"]
    template_doc = task["template_doc"]
    gates_spec = task["pointwise_gates"]
    res_floors = task["resolution_floor"]

    t0 = time.perf_counter()
    doc_dict, updates = build_scaps_case_document(template_doc, group_id, coords)
    yaml_bytes = yaml.safe_dump(doc_dict, sort_keys=False).encode("utf-8")
    input_sha = hashlib.sha256(yaml_bytes).hexdigest()

    cand_metrics: dict[str, float | None] | None = None
    solver_error: str | None = None
    evaluated_points = 0
    compiled_id: str | None = None
    try:
        prepared = _prepare_scaps_yaml_device(
            yaml_bytes,
            source_id=f"scaps://{case_id}",
            device_id=case_id.replace(":", "_"),
            defaults=defaults,
            resources=resources,
        )
        compiled = compile_device_1d(
            prepared,
            intervals_per_layer=10,
            interface_topology="two_sided_trace",
            geometry_name=PHYSICAL_BOUNDARY_TWO_SIDED_V1,
            thermionic_mode=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
            allow_legacy_benchmark_deck=True,
        )
        compiled_id = compiled.identity
        jv_res = run_steady_jv(
            compiled,
            np.linspace(0.0, 1.6, 24),
            generation_scale=1.0,
            incident_power_W_m2=1000.0,
            options=SteadyOptions(
                continuity_tolerance_A_m2=1.0e-4,
                current_spread_tolerance_A_m2=1.0e-4,
            ),
            stop_after_voc=True,
            max_bias_bisect=4,
        )
        evaluated_points = int(jv_res.voltage_V.size)
        m = jv_res.metrics
        cand_metrics = {
            "Voc_V": m.open_circuit_voltage_V,
            "Jsc_mA_cm2": (
                None
                if m.short_circuit_current_A_m2 is None
                else float(m.short_circuit_current_A_m2) * 0.1
            ),
            "FF_percent": None if m.fill_factor is None else float(m.fill_factor) * 100.0,
            "PCE_percent": (
                None
                if m.efficiency is None
                else float(m.efficiency) * 100.0
            ),
        }
        if any(v is None for v in cand_metrics.values()):
            solver_error = "metrics_undefined_or_voc_not_bracketed"
    except (ValueError, SteadyStateError, RuntimeError) as exc:
        solver_error = f"{type(exc).__name__}: {exc}"

    elapsed_s = time.perf_counter() - t0
    metric_comparisons: dict[str, Any] = {}
    if ref_status == "reference_missing" or ref_values is None:
        disposition = "reference_missing"
    elif solver_error is not None or cand_metrics is None:
        disposition = "solver_failed"
    else:
        all_passed = True
        for q_name, g_rule in gates_spec.items():
            ref_v = ref_values.get(q_name)
            cand_v = cand_metrics.get(q_name)
            if ref_v is None or cand_v is None:
                all_passed = False
                metric_comparisons[q_name] = {"passed": False, "reason": "null_metric"}
                continue
            u_rep = _decimal_ulp(float(ref_v), float(res_floors.get(q_name, 1.0e-6)))
            allowance = (
                float(g_rule["atol"]) + float(g_rule["rtol"]) * abs(float(ref_v)) + u_rep
            )
            diff = abs(float(cand_v) - float(ref_v))
            passed = bool(diff <= allowance)
            all_passed = all_passed and passed
            metric_comparisons[q_name] = {
                "candidate": float(cand_v),
                "reference": float(ref_v),
                "abs_diff": diff,
                "u_report": u_rep,
                "allowance": allowance,
                "passed": passed,
            }
        disposition = (
            "passed_external_gate" if all_passed else "exceeded_external_tolerance"
        )

    return {
        "id": case_id,
        "group_id": group_id,
        "coordinates": coords,
        "reference_status": ref_status,
        "disposition": disposition,
        "input_yaml_sha256": input_sha,
        "compiled_identity": compiled_id,
        "evaluated_bias_points": evaluated_points,
        "elapsed_s": elapsed_s,
        "updates": updates,
        "candidate_metrics": cand_metrics,
        "reference_metrics": ref_values,
        "metric_comparisons": metric_comparisons,
        "error": solver_error,
    }


def run_scaps_sweep(
    *,
    repo_root: Path,
    selected_groups: tuple[str, ...] | None = None,
    selected_case_ids: tuple[str, ...] | None = None,
    max_cases: int | None = None,
    parallel_workers: int = 1,
    require_single_thread: bool = False,
    assert_zero_legacy_modules: bool = True,
) -> dict[str, Any]:
    pre_legacy = _check_legacy_modules_entry(
        assert_zero_legacy_modules=assert_zero_legacy_modules
    )
    if not 1 <= parallel_workers <= 4:
        raise ValueError("parallel_workers must be between 1 and 4")
    contract_path = repo_root / SCAPS_CONTRACT_REL
    if _file_sha256(contract_path) != SCAPS_CONTRACT_SHA256:
        raise ValueError("ScapsComparisonContractV1.json SHA-256 mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest_path = repo_root / MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    preset_src = next(
        doc for doc in load_packaged_preset_sources() if doc.id == "scaps_mirror_v2.yaml"
    )
    template_doc = dict(yaml.safe_load(preset_src.content.decode("utf-8")))
    all_built = build_scaps_cases(template_doc, manifest["scaps"]["groups"])

    # Index reference values and resolution floors
    ref_lookup: dict[str, dict[str, float | None] | None] = {
        "scaps_base": dict(manifest["scaps"]["base"]["values"])
    }
    for grp in manifest["scaps"]["groups"]:
        gid = grp["id"]
        for pt in grp["points"]:
            cid = f"{gid}:row{pt['worksheet_row']}"
            ref_lookup[cid] = pt.get("values")
    floors_by_group: dict[str, dict[str, float]] = {
        row["group_id"]: dict(row["reported_column_resolution_floor"])
        for row in contract["reported_precision"]["per_group"]
    }
    floors_by_group["scaps_base"] = {
        "Voc_V": 1.0e-4,
        "Jsc_mA_cm2": 1.0e-6,
        "FF_percent": 1.0e-2,
        "PCE_percent": 1.0e-2,
    }

    tasks: list[dict[str, Any]] = []
    for item in all_built:
        cid = item["id"]
        gid = item["group_id"]
        if selected_groups is not None and gid not in selected_groups:
            continue
        if selected_case_ids is not None and cid not in selected_case_ids:
            continue
        tasks.append(
            {
                "id": cid,
                "group_id": gid,
                "coordinates": item["coordinates"],
                "reference_status": item["reference_status"],
                "reference_values": ref_lookup.get(cid),
                "template_doc": template_doc,
                "pointwise_gates": contract["pointwise_gates"],
                "resolution_floor": floors_by_group[gid],
            }
        )
        if max_cases is not None and len(tasks) >= max_cases:
            break
    if selected_case_ids is not None:
        found_ids = {t["id"] for t in tasks}
        missing_ids = sorted(set(selected_case_ids) - found_ids)
        if missing_ids:
            raise ValueError(f"unknown SCAPS case IDs: {missing_ids}")

    t0 = time.perf_counter()
    if parallel_workers == 1 or len(tasks) <= 1:
        rows = [_evaluate_scaps_point_worker(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=ctx) as pool:
            rows = list(pool.map(_evaluate_scaps_point_worker, tasks))
    wall_s = time.perf_counter() - t0

    counts = {
        "total_executed": len(rows),
        "passed_external_gate": sum(
            1 for r in rows if r["disposition"] == "passed_external_gate"
        ),
        "exceeded_external_tolerance": sum(
            1 for r in rows if r["disposition"] == "exceeded_external_tolerance"
        ),
        "reference_missing": sum(
            1 for r in rows if r["disposition"] == "reference_missing"
        ),
        "solver_failed": sum(1 for r in rows if r["disposition"] == "solver_failed"),
    }
    _check_legacy_modules_exit(
        pre_legacy, assert_zero_legacy_modules=assert_zero_legacy_modules
    )
    return {
        "schema": SCHEMA_SCAPS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scaps_contract_sha256": SCAPS_CONTRACT_SHA256,
        "manifest_sha256": _file_sha256(manifest_path),
        "preset_sha256": preset_src.sha256,
        "parallel_workers": parallel_workers,
        "wall_elapsed_s": wall_s,
        "environment": _host_environment_record(require_single_thread),
        "summary_counts": counts,
        "cases": rows,
        "scientific_qualification": False,
        "physical_certificate_status": "not_evaluated",
    }


def _build_hi_7phase_waveform(v_rate: float) -> Any:
    v_start = -1.0
    v_max = 1.2
    ramp_dur = (v_max - v_start) / v_rate
    segments = (
        WaveformSegment("dark_seed", 120.0, 0.0, 0.0, 0.0, n_steps=12, role="preconditioning"),
        WaveformSegment("dark_prebias", 30.0, v_start, v_start, 0.0, n_steps=10, role="preconditioning"),
        WaveformSegment("forward_light_dwell", 0.5, v_start, v_start, 1.0, n_steps=8, role="hold"),
        WaveformSegment("forward_ramp", ramp_dur, v_start, v_max, 1.0, n_steps=110, role="forward_ramp"),
        WaveformSegment("dark_turnaround", 3.0, v_max, v_max, 0.0, n_steps=8, role="hold"),
        WaveformSegment("reverse_light_dwell", 0.5, v_max, v_max, 1.0, n_steps=8, role="hold"),
        WaveformSegment("reverse_ramp", ramp_dur, v_max, v_start, 1.0, n_steps=110, role="reverse_ramp"),
    )
    return build_piecewise_waveform(
        segments,
        initial_voltage_V=0.0,
        initial_generation_scale=0.0,
        continuity_tol_V=2.0,  # step from 0 V dark_seed to -1 V dark_prebias per frozen 7-phase protocol
    )


def _extract_hi_ramp_branch(
    seg: WaveformSegment,
    slc: slice,
    v_arr: np.ndarray,
    gen_arr: np.ndarray,
    pv_tot: np.ndarray,
    *,
    init_voltage_V: float,
    init_generation_scale: float,
    init_pv_A_m2: float,
) -> tuple[np.ndarray, np.ndarray]:
    v_seg: np.ndarray = v_arr[slc]
    j_seg: np.ndarray = pv_tot[slc]
    if slc.start > 0:
        prev_idx = slc.start - 1
        if abs(float(gen_arr[prev_idx]) - seg.generation_scale) <= 1.0e-12 and (
            v_seg.size == 0 or abs(float(v_arr[prev_idx]) - float(v_seg[0])) > 1.0e-14
        ):
            v_seg = np.r_[float(v_arr[prev_idx]), v_seg]
            j_seg = np.r_[float(pv_tot[prev_idx]), j_seg]
    elif abs(init_generation_scale - seg.generation_scale) <= 1.0e-12 and (
        v_seg.size == 0 or abs(init_voltage_V - float(v_seg[0])) > 1.0e-14
    ):
        v_seg = np.r_[init_voltage_V, v_seg]
        j_seg = np.r_[init_pv_A_m2, j_seg]
    return v_seg, j_seg


def _warm_electronic_density_newton_on_be_step(
    compiled: CompiledDevice1D,
    step: BackwardEuler1DStep,
    u0: np.ndarray,
    vk: float,
    settings: NewtonSettings,
    conv_norm: Any,
    eye_n: Any,
) -> np.ndarray:
    """3-block physical-density tangent preconditioner on the exact electronic BE residual."""
    n_nodes = compiled.node_count
    vt = compiled.thermal_voltage_V
    u = u0.copy()
    r0 = step.residual(u)
    j0 = step.jacobian(u).diagonal()
    for idx in (0, n_nodes - 1, n_nodes, 2 * n_nodes - 1, 2 * n_nodes, 3 * n_nodes - 1):
        if j0[idx] != 0.0:
            u[idx] -= r0[idx] / j0[idx]
    for _ in range(25):
        res = step.residual(u)
        scaled = settings.scaling.scaled(res)
        norm_val = (
            float(np.max(np.abs(scaled)))
            if conv_norm is None
            else float(conv_norm(scaled))
        )
        if norm_val <= 0.8:
            return u
        phi, n_arr, p_arr = compiled.fields(u, voltage_V=vk)
        n_safe = np.maximum(n_arr, 1.0e-250)
        p_safe = np.maximum(p_arr, 1.0e-250)
        transform = bmat(
            [
                [eye_n, None, None],
                [-eye_n / vt, diags(1.0 / n_safe), None],
                [-eye_n / vt, None, diags(-1.0 / p_safe)],
            ],
            format="csc",
        )
        dx = spsolve(step.jacobian(u) @ transform, -res)
        dphi = dx[:n_nodes]
        dn = dx[n_nodes : 2 * n_nodes]
        dp = dx[2 * n_nodes : 3 * n_nodes]
        alpha = 1.0
        sig_n = (dn < 0.0) & (n_arr > 1.0e-6)
        if np.any(sig_n):
            alpha = min(alpha, float(np.min(-0.95 * n_arr[sig_n] / dn[sig_n])))
        sig_p = (dp < 0.0) & (p_arr > 1.0e-6)
        if np.any(sig_p):
            alpha = min(alpha, float(np.min(-0.95 * p_arr[sig_p] / dp[sig_p])))
        max_dphi = float(np.max(np.abs(dphi)))
        if max_dphi * alpha > 0.25:
            alpha = 0.25 / max_dphi
        phi_new = phi + alpha * dphi
        n_new = np.maximum(n_arr + alpha * dn, 1.0e-250)
        p_new = np.maximum(p_arr + alpha * dp, 1.0e-250)
        u_n_new = (
            u[n_nodes : 2 * n_nodes] + np.log(n_new / n_safe) - (phi_new - phi) / vt
        )
        u_p_new = (
            u[2 * n_nodes : 3 * n_nodes] - np.log(p_new / p_safe) - (phi_new - phi) / vt
        )
        phi_new[0], phi_new[-1] = u[0], u[n_nodes - 1]
        u_n_new[0], u_n_new[-1] = u[n_nodes], u[2 * n_nodes - 1]
        u_p_new[0], u_p_new[-1] = u[2 * n_nodes], u[3 * n_nodes - 1]
        u = np.r_[phi_new, u_n_new, u_p_new]
    return u


def _run_ion_free_hi_waveform(
    compiled: CompiledDevice1D,
    pw: Any,
    *,
    steady_opts: SteadyOptions,
    transient_opts: TransientOptions,
    incident_power_W_m2: float,
) -> dict[str, Any]:
    """Run the 7-phase HI waveform on an ion-free legacy deck (c0_m3 == 0.0)."""
    count = compiled.node_count
    eye_n = eye(count, format="csc")
    polarity = float(compiled.voltage_polarity)
    v0 = float(pw.initial_voltage_V)
    g0 = float(pw.initial_generation_scale)

    def pb_res(phi: np.ndarray) -> np.ndarray:
        _, n_d, p_d = compiled.fields(np.r_[phi, np.zeros(2 * count)], voltage_V=v0)
        if not np.all(np.isfinite(n_d)) or not np.all(np.isfinite(p_d)):
            return np.full(count, np.nan)
        rho = compiled.charge_ledger(n_d, p_d).space_charge_C_m3()
        return compiled.poisson.residual(phi, rho, *compiled.contact_potentials(v0))

    def pb_jac(phi: np.ndarray) -> Any:
        _, n_d, p_d = compiled.fields(np.r_[phi, np.zeros(2 * count)], voltage_V=v0)
        slope = np.zeros(count)
        slope[1:-1] = (
            -ELEMENTARY_CHARGE_C
            * compiled.poisson.interior_widths_m
            * (n_d[1:-1] + p_d[1:-1])
            / compiled.thermal_voltage_V
        )
        return compiled.poisson.jacobian() + diags(slope)

    pb = solve_newton(
        CallableProblem(pb_res, pb_jac),
        compiled.reference_potential_V,
        NewtonSettings(
            ResidualScaling(poisson_row_weights(compiled) / steady_opts.poisson_tolerance),
            1.0,
            StepControl(np.full(count, steady_opts.potential_step_V)),
            steady_opts.budget(),
            (),
            False,
        ),
    )
    if not pb.accepted:
        raise SteadyStateError("ion_free_hi_pb_seed", pb)

    dark_prob = AssembledDC1DProblem(compiled, v0, g0)
    dark_newton = solve_newton(
        dark_prob,
        np.r_[pb.state, np.zeros(2 * count)],
        NewtonSettings(
            production_residual_scaling(
                compiled,
                poisson_tolerance=steady_opts.poisson_tolerance,
                continuity_tolerance_A_m2=steady_opts.continuity_tolerance_A_m2,
                contact_log_tolerance=steady_opts.contact_log_tolerance,
            ),
            1.0,
            production_coordinates(
                compiled,
                potential_step_V=steady_opts.potential_step_V,
                quasi_fermi_step=steady_opts.quasi_fermi_step,
            ),
            steady_opts.budget(),
            physical_gates(dark_prob, steady_opts, equilibrium=False),
            False,
        ),
        convergence_norm=production_convergence_norm(
            compiled, convergence_policy=steady_opts.convergence_policy
        ),
    )
    if not dark_newton.accepted:
        raise SteadyStateError("ion_free_hi_dark_dc", dark_newton)

    init_pv_A_m2 = float(
        -polarity * np.asarray(dark_prob.current_density(dark_newton.state), dtype=float)[0]
    )
    state_curr = compiled.state(dark_newton.state, voltage_V=v0)
    binding = classical_dc_1d(compiled)
    tr_settings = NewtonSettings(
        production_residual_scaling(
            compiled,
            poisson_tolerance=transient_opts.poisson_tolerance,
            continuity_tolerance_A_m2=transient_opts.continuity_tolerance_A_m2,
            contact_log_tolerance=transient_opts.contact_log_tolerance,
        ),
        1.0,
        production_coordinates(
            compiled,
            potential_step_V=transient_opts.potential_step_V,
            quasi_fermi_step=transient_opts.quasi_fermi_step,
        ),
        transient_opts.newton_budget,
        (),
        False,
    )
    conv_norm = production_convergence_norm(
        compiled, convergence_policy=transient_opts.convergence_policy
    )
    protocol = pw.protocol
    n_req = int(protocol.times_s.size - 1)
    v_list: list[float] = []
    gen_list: list[float] = []
    pv_list: list[float] = []
    fail_reason: str | None = None
    deadline = time.monotonic() + transient_opts.wall_s

    for idx in range(1, protocol.times_s.size):
        if time.monotonic() >= deadline:
            fail_reason = f"step_failed_at_{float(protocol.times_s[idx]):.6g}s:wall_budget_exhausted"
            break
        t_start = float(protocol.times_s[idx - 1])
        t_end = float(protocol.times_s[idx])
        dt_total = t_end - t_start
        v_prev = float(state_curr.voltage_V)
        v_step = float(protocol.voltage_V[idx])
        g_step = float(protocol.generation_scale[idx])
        step_ok = False
        last_term = "stagnated"
        final_state = state_curr
        final_acc: dict[str, Any] | None = None
        for level in range(11):
            n_sub = 1 << level
            dt_sub = dt_total / n_sub
            sub_state = state_curr
            sub_ok = True
            for sub_idx in range(1, n_sub + 1):
                if time.monotonic() >= deadline:
                    last_term = "wall_budget_exhausted"
                    sub_ok = False
                    break
                vk = v_prev + (v_step - v_prev) * (sub_idx / n_sub)
                endpoint = cast(
                    DC1DProblem,
                    AssembledStorage1D(
                        AssembledDC1DProblem(compiled, vk, g_step, binding)
                    ),
                )
                step = BackwardEuler1DStep(endpoint, sub_state, dt_sub)
                seed = step.initial_coordinates(
                    transient_opts.predictor, voltage_lift=(vk != sub_state.voltage_V)
                )
                newton = solve_newton(step, seed, tr_settings, convergence_norm=conv_norm)
                if not newton.accepted:
                    u_warm = _warm_electronic_density_newton_on_be_step(
                        compiled, step, seed, vk, tr_settings, conv_norm, eye_n
                    )
                    newton = solve_newton(
                        step, u_warm, tr_settings, convergence_norm=conv_norm
                    )
                last_term = newton.termination.value
                if not newton.accepted:
                    sub_ok = False
                    break
                sub_state = compiled.state(newton.state, voltage_V=vk)
                if sub_idx == n_sub:
                    final_acc = step_accounting(step, newton.state)
            if sub_ok and final_acc is not None:
                step_ok = True
                final_state = sub_state
                break
            if last_term == "wall_budget_exhausted":
                break
        if not step_ok or final_acc is None:
            fail_reason = f"step_failed_at_{t_end:.6g}s:{last_term}"
            break
        term_tot = np.asarray(
            final_acc["terminal_total_current_density_A_m2"], dtype=float
        )
        v_list.append(v_step)
        gen_list.append(g_step)
        pv_list.append(float(-polarity * term_tot[0]))
        state_curr = final_state

    n_acc = len(v_list)
    completed = n_acc == n_req and fail_reason is None
    v_arr: np.ndarray = np.asarray(v_list, dtype=float)
    gen_arr: np.ndarray = np.asarray(gen_list, dtype=float)
    pv_tot: np.ndarray = np.asarray(pv_list, dtype=float)
    mf = None
    mr = None
    for seg, slc in zip(pw.segments, pw.segment_step_slices, strict=True):
        if slc.stop > n_acc:
            break
        if seg.role == "forward_ramp" and mf is None:
            v_f, j_f = _extract_hi_ramp_branch(
                seg,
                slc,
                v_arr,
                gen_arr,
                pv_tot,
                init_voltage_V=v0,
                init_generation_scale=g0,
                init_pv_A_m2=init_pv_A_m2,
            )
            if v_f.size >= 2 and np.unique(v_f).size == v_f.size:
                mf = steady_jv_metrics(v_f, j_f, incident_power_W_m2=incident_power_W_m2)
        elif seg.role == "reverse_ramp" and mr is None:
            v_r, j_r = _extract_hi_ramp_branch(
                seg,
                slc,
                v_arr,
                gen_arr,
                pv_tot,
                init_voltage_V=v0,
                init_generation_scale=g0,
                init_pv_A_m2=init_pv_A_m2,
            )
            if v_r.size >= 2 and np.unique(v_r).size == v_r.size:
                mr = steady_jv_metrics(v_r, j_r, incident_power_W_m2=incident_power_W_m2)

    return {
        "completed": completed,
        "compiled_identity": compiled.identity,
        "accepted_steps": n_acc,
        "requested_steps": n_req,
        "inventory_relative_drift": [],
        "metrics_fwd": mf,
        "metrics_rev": mr,
        "failure_reason": fail_reason,
    }


def _evaluate_hi_request_worker(task: dict[str, Any]) -> dict[str, Any]:
    defaults, base_resources = _load_packaged_catalog_and_resources()
    nom_req = json.loads(json.dumps(task["nominal_request"]))
    d_ion = float(task["D_ion_m2_s"])
    c0_m3 = float(task["c0_m3"])
    v_rate = float(task["scan_rate_V_s"])
    gen_rate = float(nom_req["params"]["waveform"]["uniform_generation_rate_m3_s"])

    dev_doc = nom_req["device"]
    for lyr in dev_doc["layers"]:
        if lyr["role"] == "absorber":
            lyr["D_ion"] = 0.0 if c0_m3 == 0.0 else d_ion
            lyr["P0"] = c0_m3
        lyr["alpha"] = 0.0
    yaml_bytes = yaml.safe_dump(dev_doc, sort_keys=False).encode("utf-8")
    src = SourceDocument(id=f"hi://{task['primary_case_id']}", content=yaml_bytes)

    t0 = time.perf_counter()
    error_msg: str | None = None
    try:
        dev_in = import_standard_device(
            src, id=task["primary_case_id"], defaults=defaults
        )
        settings_patch: dict[str, Any] = {
            **dev_in.settings.model_dump(exclude_unset=True),
            "dos_band_potentials": False,
            "flat_band_metal_contacts": True,
            "ion_steric_shared_site": False,
        }
        if c0_m3 == 0.0:
            settings_patch["ion_steric_diffusion_only"] = False
        dev_in = dev_in.validated_update({"settings": settings_patch})
        # First probe nodes on dark/zero-alpha device to build exact FixedGenerationProfile JSON
        with legacy_benchmark_deck_scope():
            probe_1d = compile_device_1d(
                resolve_device(
                    dev_in.validated_update(
                        {
                            "settings": {
                                **settings_patch,
                                "ion_steric_diffusion_only": False,
                            },
                            "layers": [
                                {
                                    **l_d,
                                    "parameters": {**l_d["parameters"], "D_ion": 0.0, "P0": 0.0},
                                }
                                for l_d in dev_in.model_dump(mode="json", exclude_unset=True)["layers"]
                            ],
                        }
                    ),
                    defaults,
                    base_resources,
                    sources=(src,),
                ),
                intervals_per_layer=20,
            )
        gen_json = json.dumps(
            {
                "coordinate_m": [float(x) for x in probe_1d.nodes_m],
                "generation_m3_s": [gen_rate] * int(probe_1d.nodes_m.size),
            },
            sort_keys=True,
        ).encode("utf-8")
        gen_table = ResourceTable(
            "hi_uniform_gen",
            "fixed_generation",
            SourceDocument("hi://uniform_gen.json", gen_json),
        )
        hi_resources = ResourceLibrary((*base_resources.tables, gen_table))
        dev_in_gen = dev_in.validated_update({"fixed_generation": "hi_uniform_gen"})
        prepared = resolve_device(dev_in_gen, defaults, hi_resources, sources=(src,))

        pw = _build_hi_7phase_waveform(v_rate)
        elec_steady = SteadyOptions(
            continuity_tolerance_A_m2=1.0e-3, current_spread_tolerance_A_m2=1.0e-3
        )
        tr_elec = TransientOptions(
            poisson_tolerance=1.0e-9,
            continuity_tolerance_A_m2=1.0e-2,
            contact_log_tolerance=1.0e-9,
            potential_step_V=0.25,
            quasi_fermi_step=3.0,
            newton_budget=SolveBudget(60, 1200, 60, 24),
            max_steps=400,
            wall_s=180.0,
            predictor="previous",
            convergence_policy=POISSON_CONTACT_BALANCE_V1,
        )

        if c0_m3 == 0.0:
            with legacy_benchmark_deck_scope():
                compiled_1d = compile_device_1d(prepared, intervals_per_layer=20)
            ion_free_res = _run_ion_free_hi_waveform(
                compiled_1d,
                pw,
                steady_opts=elec_steady,
                transient_opts=tr_elec,
                incident_power_W_m2=1000.0,
            )
            wf_completed = bool(ion_free_res["completed"])
            wf_identity = str(ion_free_res["compiled_identity"])
            wf_acc_steps = int(ion_free_res["accepted_steps"])
            wf_req_steps = int(ion_free_res["requested_steps"])
            wf_inv_drift = list(ion_free_res["inventory_relative_drift"])
            mf = ion_free_res["metrics_fwd"]
            mr = ion_free_res["metrics_rev"]
            wf_fail_reason = ion_free_res["failure_reason"]
        else:
            with legacy_benchmark_deck_scope():
                absorber_id = next(
                    lyr.id for lyr in prepared.layers if lyr.role == "absorber"
                )
                compiled_ion = compile_ionic_device_1d(
                    prepared,
                    intervals_per_layer=20,
                    ion_domain_layer=absorber_id,
                    ion_domain_endpoints=("blocking", "blocking"),
                )
            ion_steady = default_ionic_steady_options(
                compiled_ion.ion_node_count, electronic=elec_steady
            )
            wf_opts = WaveformJVOptions(
                steady=elec_steady,
                ionic_steady=ion_steady,
                transient=tr_elec,
                ionic_transient=IonicTransientOptions(
                    electronic=tr_elec,
                    ion_balance_tolerance_m2_s=1.0e6,
                    ion_logit_step=1.0,
                ),
                incident_power_W_m2=1000.0,
                require_completed=False,
            )
            wf_res = run_waveform_jv(compiled_ion, pw, options=wf_opts)
            wf_completed = bool(wf_res.completed)
            wf_identity = wf_res.compiled_identity
            wf_acc_steps = int(wf_res.accepted_steps)
            wf_req_steps = int(wf_res.requested_steps)
            wf_inv_drift = list(wf_res.inventory_relative_drift)
            mf = wf_res.metrics_fwd
            mr = wf_res.metrics_rev
            wf_fail_reason = wf_res.failure_reason

        elapsed_s = time.perf_counter() - t0
        p_fwd = None if mf is None else mf.maximum_power_W_m2
        p_rev = None if mr is None else mr.maximum_power_W_m2
        hi_p = (
            float(p_rev / p_fwd - 1.0)
            if (p_fwd is not None and p_rev is not None and p_fwd > 0.0)
            else None
        )
        hi_norm = (
            float((p_rev - p_fwd) / p_rev)
            if (p_fwd is not None and p_rev is not None and p_rev > 0.0)
            else None
        )
        return {
            "request_sha256": task["request_sha256"],
            "primary_case_id": task["primary_case_id"],
            "completed": wf_completed,
            "compiled_identity": wf_identity,
            "accepted_steps": wf_acc_steps,
            "requested_steps": wf_req_steps,
            "elapsed_s": elapsed_s,
            "D_ion_m2_s": d_ion,
            "c0_m3": c0_m3,
            "scan_rate_V_s": v_rate,
            "inventory_relative_drift": wf_inv_drift,
            "candidate_metrics": {
                "HI_P": hi_p,
                "HI_normalized": hi_norm,
                "Pmax_forward_W_m2": p_fwd,
                "Pmax_reverse_W_m2": p_rev,
                "Voc_forward_V": None if mf is None else mf.open_circuit_voltage_V,
                "Voc_reverse_V": None if mr is None else mr.open_circuit_voltage_V,
                "Jsc_forward_A_m2": None if mf is None else mf.short_circuit_current_A_m2,
                "Jsc_reverse_A_m2": None if mr is None else mr.short_circuit_current_A_m2,
            },
            "failure_reason": wf_fail_reason,
        }
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        return {
            "request_sha256": task["request_sha256"],
            "primary_case_id": task["primary_case_id"],
            "completed": False,
            "compiled_identity": None,
            "accepted_steps": 0,
            "requested_steps": 0,
            "elapsed_s": time.perf_counter() - t0,
            "D_ion_m2_s": d_ion,
            "c0_m3": c0_m3,
            "scan_rate_V_s": v_rate,
            "inventory_relative_drift": [],
            "candidate_metrics": None,
            "failure_reason": error_msg,
        }


def run_hysteresis_matrix(
    *,
    repo_root: Path,
    selected_axes: tuple[str, ...] | None = None,
    selected_case_ids: tuple[str, ...] | None = None,
    max_cases: int | None = None,
    parallel_workers: int = 1,
    require_single_thread: bool = False,
    assert_zero_legacy_modules: bool = True,
) -> dict[str, Any]:
    pre_legacy = _check_legacy_modules_entry(
        assert_zero_legacy_modules=assert_zero_legacy_modules
    )
    if not 1 <= parallel_workers <= 4:
        raise ValueError("parallel_workers must be between 1 and 4")
    contract_path = repo_root / HI_CONTRACT_REL
    if _file_sha256(contract_path) != HI_CONTRACT_SHA256:
        raise ValueError("HysteresisReferenceContractV2.json SHA-256 mismatch")
    manifest_path = repo_root / MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nominal_request = manifest["hi"]["nominal_request"]

    requested_cases: list[dict[str, Any]] = []
    for row in manifest["hi"]["cases"]:
        if selected_axes is not None and row["axis"] not in selected_axes:
            continue
        if selected_case_ids is not None and row["id"] not in selected_case_ids:
            continue
        requested_cases.append(row)
        if max_cases is not None and len(requested_cases) >= max_cases:
            break
    if selected_case_ids is not None:
        found_hi_ids = {r["id"] for r in requested_cases}
        missing_hi_ids = sorted(set(selected_case_ids) - found_hi_ids)
        if missing_hi_ids:
            raise ValueError(f"unknown hysteresis case IDs: {missing_hi_ids}")

    unique_tasks: dict[str, dict[str, Any]] = {}
    for row in requested_cases:
        r_sha = str(row["request_sha256"])
        if r_sha not in unique_tasks:
            unique_tasks[r_sha] = {
                "request_sha256": r_sha,
                "primary_case_id": row["id"],
                "D_ion_m2_s": row["D_ion_m2_s"],
                "c0_m3": row["c0_m3"],
                "scan_rate_V_s": row["scan_rate_V_s"],
                "nominal_request": nominal_request,
            }

    task_list = list(unique_tasks.values())
    t0 = time.perf_counter()
    if parallel_workers == 1 or len(task_list) <= 1:
        solved_list = [_evaluate_hi_request_worker(t) for t in task_list]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=ctx) as pool:
            solved_list = list(pool.map(_evaluate_hi_request_worker, task_list))
    wall_s = time.perf_counter() - t0

    solved_by_sha = {item["request_sha256"]: item for item in solved_list}
    case_records: list[dict[str, Any]] = []
    for row in requested_cases:
        r_sha = str(row["request_sha256"])
        solved = solved_by_sha[r_sha]
        reused_from = (
            None if solved["primary_case_id"] == row["id"] else solved["primary_case_id"]
        )
        ref_m = row["metrics"]
        cand_m = solved["candidate_metrics"]
        deltas = None
        if cand_m is not None and cand_m["HI_normalized"] is not None:
            deltas = {
                "HI_normalized_abs_diff": abs(
                    float(cand_m["HI_normalized"])
                    - float(ref_m["HI_normalized_recorded"])
                ),
                "HI_P_abs_diff": (
                    None
                    if cand_m["HI_P"] is None
                    else abs(float(cand_m["HI_P"]) - float(ref_m["HI_P_recorded"]))
                ),
                "Pmax_forward_abs_diff_W_m2": abs(
                    float(cand_m["Pmax_forward_W_m2"])
                    - float(ref_m["Pmax_forward_W_m2"])
                ),
                "Pmax_reverse_abs_diff_W_m2": abs(
                    float(cand_m["Pmax_reverse_W_m2"])
                    - float(ref_m["Pmax_reverse_W_m2"])
                ),
            }
        case_records.append(
            {
                "id": row["id"],
                "axis": row["axis"],
                "parameter_value": row["parameter_value"],
                "request_sha256": r_sha,
                "reused_from_case_id": reused_from,
                "completed": solved["completed"],
                "compiled_identity": solved["compiled_identity"],
                "elapsed_s": 0.0 if reused_from is not None else solved["elapsed_s"],
                "inventory_relative_drift": solved["inventory_relative_drift"],
                "candidate_metrics": cand_m,
                "historical_reference_metrics": ref_m,
                "comparison_deltas": deltas,
                "failure_reason": solved["failure_reason"],
            }
        )

    _check_legacy_modules_exit(
        pre_legacy, assert_zero_legacy_modules=assert_zero_legacy_modules
    )
    return {
        "schema": SCHEMA_HI,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hysteresis_contract_sha256": HI_CONTRACT_SHA256,
        "manifest_sha256": _file_sha256(manifest_path),
        "parallel_workers": parallel_workers,
        "requested_case_count": len(case_records),
        "distinct_executed_request_count": len(task_list),
        "completed_case_count": sum(1 for c in case_records if c["completed"]),
        "wall_elapsed_s": wall_s,
        "environment": _host_environment_record(require_single_thread),
        "cases": case_records,
        "scientific_qualification": False,
        "physical_certificate_status": "not_evaluated",
    }


def run_smoke_all(
    *,
    repo_root: Path,
    require_single_thread: bool = False,
    assert_zero_legacy_modules: bool = True,
) -> dict[str, Any]:
    perf = run_perf_candidate(
        repo_root=repo_root,
        selected_cases=(
            "TransientJvN80",
            "TransientJvN200",
            "ScapsSteady",
            "AC60",
            "Spatial2DStageA",
        ),
        repeats=1,
        spatial_nx=(4, 8),
        transient_variants=("ionmonger_benchmark",),
        require_single_thread=require_single_thread,
        assert_zero_legacy_modules=assert_zero_legacy_modules,
    )
    scaps = run_scaps_sweep(
        repo_root=repo_root,
        selected_case_ids=("scaps_base", "scaps_cbo:row7", "scaps_cbo:row17"),
        parallel_workers=1,
        require_single_thread=require_single_thread,
        assert_zero_legacy_modules=assert_zero_legacy_modules,
    )
    hi = run_hysteresis_matrix(
        repo_root=repo_root,
        selected_case_ids=("hi_density_00", "hi_diffusivity_10", "hi_rate_40"),
        parallel_workers=1,
        require_single_thread=require_single_thread,
        assert_zero_legacy_modules=assert_zero_legacy_modules,
    )
    return {
        "schema": SCHEMA_SMOKE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "perf_candidate": perf,
        "scaps_sweep": scaps,
        "hysteresis_matrix": hi,
        "scientific_qualification": False,
        "physical_certificate_status": "not_evaluated",
    }


def _write_json_receipt(payload: dict[str, Any], output_path: Path | None) -> str:
    payload_with_digest = {
        **payload,
        "payload_sha256": _json_digest(payload),
    }
    text = json.dumps(payload_with_digest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        "--repo-data-root",
        dest="repo_root",
        type=Path,
        default=_DEFAULT_REPO_ROOT,
        help="Repository root containing reproducibility contracts and fixture configs.",
    )
    parser.add_argument(
        "--require-single-thread-blas",
        action="store_true",
        help="Fail closed unless all six BLAS thread environment variables equal '1'.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_perf = sub.add_parser("perf-candidate", help="Run PerformanceCasesV1 candidate benchmarks.")
    p_perf.add_argument("--cases", type=str, default="all")
    p_perf.add_argument("--repeats", type=int, default=1)
    p_perf.add_argument("--spatial-nx", type=str, default="4,8,16")
    p_perf.add_argument("--transient-variants", type=str, default="all")
    p_perf.add_argument("--output", type=Path, default=None)

    p_scaps = sub.add_parser("scaps-sweep", help="Run SCAPS 262-case comparison sweep.")
    p_scaps.add_argument("--groups", type=str, default="all")
    p_scaps.add_argument("--case-ids", type=str, default=None)
    p_scaps.add_argument("--max-cases", type=int, default=None)
    p_scaps.add_argument("--parallel-workers", type=int, default=1)
    p_scaps.add_argument("--output", type=Path, default=None)

    p_hi = sub.add_parser("hysteresis-matrix", help="Run Hysteresis 45-case protocol matrix.")
    p_hi.add_argument("--axes", type=str, default="all")
    p_hi.add_argument("--case-ids", type=str, default=None)
    p_hi.add_argument("--max-cases", type=int, default=None)
    p_hi.add_argument("--parallel-workers", type=int, default=1)
    p_hi.add_argument("--output", type=Path, default=None)

    p_smoke = sub.add_parser("smoke-all", help="Run fast smoke across all three P09-01 lanes.")
    p_smoke.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    req_st = bool(args.require_single_thread_blas)

    if args.command == "perf-candidate":
        sel_cases = (
            None
            if args.cases == "all"
            else tuple(part.strip() for part in args.cases.split(",") if part.strip())
        )
        nx_tuple = tuple(int(x.strip()) for x in args.spatial_nx.split(",") if x.strip())
        tr_vars = (
            None
            if args.transient_variants == "all"
            else tuple(
                part.strip()
                for part in args.transient_variants.split(",")
                if part.strip()
            )
        )
        payload = run_perf_candidate(
            repo_root=repo_root,
            selected_cases=sel_cases,
            repeats=int(args.repeats),
            spatial_nx=nx_tuple,
            transient_variants=tr_vars,
            require_single_thread=req_st,
        )
    elif args.command == "scaps-sweep":
        sel_groups = (
            None
            if args.groups == "all"
            else tuple(part.strip() for part in args.groups.split(",") if part.strip())
        )
        sel_ids = (
            None
            if not args.case_ids
            else tuple(part.strip() for part in args.case_ids.split(",") if part.strip())
        )
        payload = run_scaps_sweep(
            repo_root=repo_root,
            selected_groups=sel_groups,
            selected_case_ids=sel_ids,
            max_cases=args.max_cases,
            parallel_workers=int(args.parallel_workers),
            require_single_thread=req_st,
        )
    elif args.command == "hysteresis-matrix":
        sel_axes = (
            None
            if args.axes == "all"
            else tuple(part.strip() for part in args.axes.split(",") if part.strip())
        )
        sel_ids = (
            None
            if not args.case_ids
            else tuple(part.strip() for part in args.case_ids.split(",") if part.strip())
        )
        payload = run_hysteresis_matrix(
            repo_root=repo_root,
            selected_axes=sel_axes,
            selected_case_ids=sel_ids,
            max_cases=args.max_cases,
            parallel_workers=int(args.parallel_workers),
            require_single_thread=req_st,
        )
    else:
        payload = run_smoke_all(repo_root=repo_root, require_single_thread=req_st)

    file_sha = _write_json_receipt(payload, args.output)
    summary = {
        "schema": payload["schema"],
        "output_path": None if args.output is None else str(args.output.resolve()),
        "receipt_sha256": file_sha,
        "scientific_qualification": payload["scientific_qualification"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

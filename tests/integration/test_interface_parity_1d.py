"""Live numerical parity tests (T4) between new-core solarlab and legacy perovskite_sim (P04-09).

Verifies end-to-end 1D numerical parity across:
1. Illuminated DC J-V figures of merit (|dVoc| <= 1e-4 V, relative Jsc/FF/PCE <= 1e-4) on
   multilayer HTL/perovskite/ETL devices with:
   - Active bare v_n/v_p heterointerface SRH + finite 1D thermionic emission (TE);
   - Explicit single-level interface defects (cross-carrier, plane projection, plane closure);
   - Equilibrium-referenced charged two-sided heterointerfaces (f_eq, Delta_sigma(V), f(V), J-V).
2. Implicit Backward Euler transient step (BackwardEuler1DStep on AssembledDC1DProblem vs
   perovskite_sim assemble_rhs implicit Euler step) with active interface SRH, defect, and TE.
3. Small-signal AC admittance/impedance across 1 Hz .. 1 MHz (solve_frequency_systems on
   AssembledDC1DProblem vs perovskite_sim run_quasi_fermi_impedance & solve_frequency_domain).
"""

from __future__ import annotations

from collections.abc import Iterator
import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.optimize import root
from threadpoolctl import threadpool_limits

from perovskite_sim.constants import EPS_0, K_B, Q
from perovskite_sim.experiments import steady_state as legacy_ss
from perovskite_sim.experiments.interface_charge_jv import (
    build_interface_charge_jv_protocol,
    solve_interface_charge_jv,
)
from perovskite_sim.experiments.jv_sweep import (
    JVMetrics,
    _compute_current_ss,
    build_electrical_grid,
    compute_metrics,
)
from perovskite_sim.experiments.quasi_fermi_impedance import run_quasi_fermi_impedance
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_equilibrium_referenced_interface_charge_dark_reference,
    build_two_sided_trace_grid,
    solve_equilibrium_referenced_interface_charge_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.continuity import carrier_face_currents
from perovskite_sim.physics.generation import beer_lambert_generation
from perovskite_sim.solver.mol import (
    MaterialArrays,
    _harmonic_face_average,
    assemble_rhs,
    build_material_arrays,
)
from perovskite_sim.solver.small_signal import (
    SmallSignalEvaluation,
    solve_frequency_domain,
)

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.terms import FaceValue
from solarlab.equations.dc_1d import CompiledDevice1D, State1D
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.physics.generation.profile import GenerationProfile
from solarlab.solvers.linear.frequency_domain import solve_frequency_systems
from solarlab.solvers.nonlinear.newton import NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.production import (
    production_coordinates,
    production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import (
    POISSON_CONTACT_BALANCE_V1,
    SteadyOptions,
    SteadyStateError,
    solve_steady_state,
)
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep

ROOT = Path(__file__).resolve().parents[2]
LEGACY_PKG = ROOT / "perovskite-sim" / "perovskite_sim"
LEGACY_DEFAULTS = (
    "models/parameters.py",
    "models/device.py",
    "models/defects.py",
    "physics/statistics.py",
    "physics/band_gap_narrowing.py",
    "scaps_compat/loader.py",
    "constants.py",
    "twod/microstructure.py",
    "models/tandem_config.py",
)

Vector = NDArray[np.float64]


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def legacy_catalog() -> DefaultCatalog:
    return read_legacy_default_catalog(
        tuple(SourceDocument(name, (LEGACY_PKG / name).read_bytes()) for name in LEGACY_DEFAULTS)
    )


def _thermodynamic_ni(nc_m3: float, nv_m3: float, eg_eV: float, temp_K: float = 300.0) -> float:
    vt = K_B * temp_K / Q
    return float(math.sqrt(nc_m3 * nv_m3) * math.exp(-eg_eV / (2.0 * vt)))


def _base_pin_yaml_dict() -> dict[str, Any]:
    """Explicit 3-layer HTL / perovskite absorber / ETL stack with exact ni^2 == Nc*Nv*exp(-Eg/Vt)."""
    ni_htl = _thermodynamic_ni(2.0e25, 3.0e25, 1.80)
    ni_abs = _thermodynamic_ni(1.0e25, 1.0e25, 1.60)
    ni_etl = _thermodynamic_ni(4.0e25, 2.0e25, 2.00)
    return {
        "device": {
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "ion_steric_diffusion_only": False,
            "ion_steric_shared_site": False,
            "T": 300.0,
            "Phi": 2.0e21,
            "mode": "full",
        },
        "electrical_grid": {
            "interval_weights": {"htl": 1.0, "absorber": 2.0, "etl": 1.0},
            "alphas": {"htl": 3.0, "absorber": 3.0, "etl": 3.0},
        },
        "layers": [
            {
                "name": "htl",
                "role": "HTL",
                "thickness": 8.0e-8,
                "eps_r": 3.0,
                "mu_n": 3.8681727071833605e-5,
                "mu_p": 3.8681727071833605e-5,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_htl,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_htl,
                "p1": ni_htl,
                "B_rad": 1.0e-18,
                "C_n": 1.0e-42,
                "C_p": 1.0e-42,
                "alpha": 0.0,
                "N_A": 1.0e24,
                "N_D": 0.0,
                "chi": 3.6,
                "Eg": 1.80,
                "Nc300": 2.0e25,
                "Nv300": 3.0e25,
                "A_star_n": 0.0,
                "A_star_p": 0.0,
            },
            {
                "name": "absorber",
                "role": "absorber",
                "thickness": 3.0e-7,
                "eps_r": 25.0,
                "mu_n": 0.0019340863535916802,
                "mu_p": 0.0019340863535916802,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_abs,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_abs,
                "p1": ni_abs,
                "B_rad": 1.0e-18,
                "C_n": 1.0e-42,
                "C_p": 1.0e-42,
                "alpha": 2.0e6,
                "N_A": 0.0,
                "N_D": 0.0,
                "chi": 3.9,
                "Eg": 1.60,
                "Nc300": 1.0e25,
                "Nv300": 1.0e25,
                "A_star_n": 0.0,
                "A_star_p": 0.0,
            },
            {
                "name": "etl",
                "role": "ETL",
                "thickness": 8.0e-8,
                "eps_r": 9.0,
                "mu_n": 3.8681727071833605e-5,
                "mu_p": 3.8681727071833605e-5,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_etl,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_etl,
                "p1": ni_etl,
                "B_rad": 1.0e-18,
                "C_n": 1.0e-42,
                "C_p": 1.0e-42,
                "alpha": 0.0,
                "N_A": 0.0,
                "N_D": 1.0e24,
                "chi": 4.0,
                "Eg": 2.00,
                "Nc300": 4.0e25,
                "Nv300": 2.0e25,
                "A_star_n": 0.0,
                "A_star_p": 0.0,
            },
        ],
    }


def _moderate_pin_yaml_dict() -> dict[str, Any]:
    """Moderate-doping 3-layer HTL / absorber / ETL stack keeping minority carriers well above floor."""
    doc = _base_pin_yaml_dict()
    ni_htl = _thermodynamic_ni(2.0e25, 3.0e25, 1.70)
    ni_etl = _thermodynamic_ni(4.0e25, 2.0e25, 1.80)
    doc["layers"][0].update(
        {"N_A": 5.0e21, "Eg": 1.70, "ni": ni_htl, "n1": ni_htl, "p1": ni_htl}
    )
    doc["layers"][2].update(
        {"N_D": 5.0e21, "Eg": 1.80, "ni": ni_etl, "n1": ni_etl, "p1": ni_etl}
    )
    return doc


def _load_paired_models(
    raw_doc: dict[str, Any],
    catalog: DefaultCatalog,
    tmp_path: Path,
    name: str,
) -> tuple[PreparedDevice, DeviceStack]:
    """Load the exact same JSON/YAML payload into both PreparedDevice and legacy DeviceStack."""
    payload = json.dumps(raw_doc, allow_nan=False).encode()
    yaml_file = tmp_path / f"{name}.yaml"
    yaml_file.write_bytes(payload)
    source = SourceDocument(str(yaml_file), payload)
    dev_input = import_standard_device(source, id=name, defaults=catalog)
    prepared = resolve_device(dev_input, catalog, ResourceLibrary(), sources=(source,))
    stack = load_device_from_yaml(str(yaml_file))
    return prepared, stack


def _step_solarlab_to_voltage(
    dev: CompiledDevice1D,
    prev_state: State1D | None,
    v_prev: float | None,
    v_target: float,
    opts: SteadyOptions,
) -> State1D:
    """Advance steady-state solution to ``v_target`` using bounded bias substeps + bisection."""
    if prev_state is None or v_prev is None:
        return solve_steady_state(
            dev,
            voltage_V=v_target,
            generation_scale=1.0,
            options=opts,
        ).state

    state_cur = prev_state
    v_cur = v_prev
    max_step = max(float(opts.max_bias_step_V), 1.0e-3)
    while abs(v_target - v_cur) > 1.0e-12:
        step = min(max_step, abs(v_target - v_cur)) * (1.0 if v_target >= v_cur else -1.0)
        v_try = v_target if abs(v_target - v_cur) <= abs(step) + 1.0e-12 else v_cur + step
        while True:
            try:
                ss = solve_steady_state(
                    dev,
                    initial_state=state_cur,
                    voltage_V=v_try,
                    generation_scale=1.0,
                    options=opts,
                )
                state_cur = ss.state
                v_cur = v_try
                break
            except SteadyStateError:
                if abs(v_try - v_cur) <= 1.0e-3:
                    raise
                v_try = 0.5 * (v_cur + v_try)
    return state_cur


def _sweep_solarlab_jv(
    dev: CompiledDevice1D,
    voltages_V: Vector,
    *,
    options: SteadyOptions | None = None,
    stop_after_voc: bool = True,
) -> tuple[Vector, Vector, JVMetrics, list[State1D]]:
    """Sweep illuminated J-V on AssembledDC1DProblem and extract solar J-V metrics."""
    opts = (
        SteadyOptions(max_bias_step_V=0.025, convergence_policy=POISSON_CONTACT_BALANCE_V1)
        if options is None
        else options
    )
    v_done: list[float] = []
    j_done: list[float] = []
    states: list[State1D] = []
    prev_state: State1D | None = None
    v_prev: float | None = None
    for v_raw in voltages_V:
        v_val = float(v_raw)
        st = _step_solarlab_to_voltage(dev, prev_state, v_prev, v_val, opts)
        assm = AssembledDC1DProblem(dev, voltage_V=v_val, generation_scale=1.0)
        j_faces = assm.current_density(st.u)
        j_val = float(-dev.voltage_polarity * np.median(j_faces))
        v_done.append(v_val)
        j_done.append(j_val)
        states.append(st)
        prev_state = st
        v_prev = v_val
        if stop_after_voc and len(j_done) >= 2 and j_done[0] > 0.0 and j_val <= 0.0:
            break
    v_arr = np.asarray(v_done, dtype=float)
    j_arr = np.asarray(j_done, dtype=float)
    metrics = compute_metrics(v_arr, j_arr, P_in=1000.0)
    assert metrics.voc_bracketed, f"Voc not bracketed: V={v_arr[-1]:.3f}, J={j_arr[-1]:.3e}"
    return v_arr, j_arr, metrics, states


def _sweep_legacy_ss_jv(
    stack: DeviceStack,
    x: Vector,
    voltages_V: Vector,
    *,
    te_softness: float = 0.02,
    stop_after_voc: bool = True,
    max_step_V: float = 0.025,
) -> tuple[Vector, Vector, JVMetrics, list[Vector], MaterialArrays]:
    """Sweep illuminated J-V on perovskite_sim steady_state.solve_steady_state on identical grid x."""
    mat = dataclasses.replace(build_material_arrays(x, stack), te_softness=te_softness)
    v_done: list[float] = []
    j_done: list[float] = []
    ys: list[Vector] = []
    y_prev: Vector | None = None
    v_prev: float | None = None
    for v_raw in voltages_V:
        v_target = float(v_raw)
        if y_prev is None or v_prev is None:
            res = legacy_ss.solve_steady_state(x, stack, v_target, illuminated=True, mat=mat)
            y_cur = res.y
        else:
            y_cur = y_prev
            v_cur = v_prev
            while abs(v_target - v_cur) > 1.0e-12:
                step = min(max_step_V, abs(v_target - v_cur)) * (
                    1.0 if v_target >= v_cur else -1.0
                )
                v_try = v_target if abs(v_target - v_cur) <= abs(step) + 1.0e-12 else v_cur + step
                while True:
                    try:
                        res = legacy_ss.solve_steady_state(
                            x,
                            stack,
                            v_try,
                            illuminated=True,
                            mat=mat,
                            y0=y_cur,
                        )
                        y_cur = res.y
                        v_cur = v_try
                        break
                    except legacy_ss.SteadyStateError:
                        if abs(v_try - v_cur) <= 1.0e-3:
                            raise
                        v_try = 0.5 * (v_cur + v_try)
        j_val = float(_compute_current_ss(x, y_cur, stack, v_target, mat=mat))
        v_done.append(v_target)
        j_done.append(j_val)
        ys.append(y_cur)
        y_prev = y_cur
        v_prev = v_target
        if stop_after_voc and len(j_done) >= 2 and j_done[0] > 0.0 and j_val <= 0.0:
            break
    v_arr = np.asarray(v_done, dtype=float)
    j_arr = np.asarray(j_done, dtype=float)
    metrics = compute_metrics(v_arr, j_arr, P_in=1000.0)
    assert metrics.voc_bracketed, f"Legacy Voc not bracketed: V={v_arr[-1]:.3f}, J={j_arr[-1]:.3e}"
    return v_arr, j_arr, metrics, ys, mat


def _assert_jv_metrics_parity(m_new: JVMetrics, m_old: JVMetrics) -> dict[str, float]:
    d_voc = abs(m_new.V_oc - m_old.V_oc)
    rel_jsc = abs(m_new.J_sc - m_old.J_sc) / abs(m_old.J_sc)
    rel_ff = abs(m_new.FF - m_old.FF) / abs(m_old.FF)
    rel_pce = abs(m_new.PCE - m_old.PCE) / abs(m_old.PCE)
    assert d_voc <= 1.0e-4, f"|dVoc| = {d_voc:.3e} V > 1e-4 V"
    assert rel_jsc <= 1.0e-4, f"rel_dJsc = {rel_jsc:.3e} > 1e-4"
    assert rel_ff <= 1.0e-4, f"rel_dFF = {rel_ff:.3e} > 1e-4"
    assert rel_pce <= 1.0e-4, f"rel_dPCE = {rel_pce:.3e} > 1e-4"
    return {
        "abs_dVoc_V": d_voc,
        "rel_dJsc": rel_jsc,
        "rel_dFF": rel_ff,
        "rel_dPCE": rel_pce,
    }


def _clean_json(val: Any) -> Any:
    if isinstance(val, dict):
        return {k: _clean_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_clean_json(v) for v in val]
    if hasattr(val, "as_tuple"):
        return float(val)
    return val


def test_live_jv_parity_bare_srh_and_finite_thermionic_emission(
    legacy_catalog: DefaultCatalog,
    tmp_path: Path,
) -> None:
    """T4 DC J-V parity: active bare v_n/v_p SRH + finite 1D TE on 3-layer HTL/perovskite/ETL."""
    doc_zero = _base_pin_yaml_dict()
    doc_zero["device"]["interfaces"] = [[0.0, 0.0], [0.0, 0.0]]
    prep_zero, _ = _load_paired_models(doc_zero, legacy_catalog, tmp_path, "pin_zero_srh")
    dev_zero = compile_device_1d(prep_zero, intervals_per_layer=(8, 16, 8))

    doc_active = _base_pin_yaml_dict()
    doc_active["device"]["interfaces"] = [[35.0, 20.0], [18.0, 40.0]]
    for layer in doc_active["layers"]:
        layer["A_star_n"] = 1.2017e6
        layer["A_star_p"] = 1.2017e6
    prep_active, stack_active = _load_paired_models(
        doc_active, legacy_catalog, tmp_path, "pin_bare_srh_te"
    )
    dev_active = compile_device_1d(
        prep_active,
        intervals_per_layer=(8, 16, 8),
        te_softness=0.02,
        te_regularization_mode="legacy_logistic",
    )
    x_legacy = build_electrical_grid(stack_active, 32)
    np.testing.assert_allclose(dev_active.nodes_m, x_legacy, rtol=0.0, atol=1.0e-15)

    voltages = np.linspace(0.0, 1.32, 34)
    _, _, m_zero, _ = _sweep_solarlab_jv(dev_zero, voltages)
    v_old, j_old, m_old, _, _ = _sweep_legacy_ss_jv(
        stack_active, x_legacy, voltages, te_softness=0.02
    )
    v_new, j_new, m_new, _ = _sweep_solarlab_jv(dev_active, v_old, stop_after_voc=False)
    np.testing.assert_array_equal(v_new, v_old)

    assert m_zero.V_oc - m_new.V_oc > 0.03
    _assert_jv_metrics_parity(m_new, m_old)
    np.testing.assert_allclose(j_new, j_old, rtol=5.0e-5, atol=5.0e-4)

    # Flux-limiting small Richardson constant where the 1D TE cap is visibly active!
    doc_te_cap = copy.deepcopy(doc_active)
    for layer in doc_te_cap["layers"]:
        layer["A_star_n"] = 1.0e-22
        layer["A_star_p"] = 1.0e-22
    prep_cap, stack_cap = _load_paired_models(
        doc_te_cap, legacy_catalog, tmp_path, "pin_strong_te_cap"
    )
    dev_cap = compile_device_1d(
        prep_cap,
        intervals_per_layer=(8, 16, 8),
        te_softness=0.02,
        te_regularization_mode="legacy_logistic",
    )
    v_cap_old, j_cap_old, m_cap_old, _, _ = _sweep_legacy_ss_jv(
        stack_cap, x_legacy, voltages, te_softness=0.02
    )
    _, j_cap_new, m_cap_new, _ = _sweep_solarlab_jv(dev_cap, v_cap_old, stop_after_voc=False)
    assert abs(m_cap_new.J_sc - m_new.J_sc) > 0.03
    _assert_jv_metrics_parity(m_cap_new, m_cap_old)
    np.testing.assert_allclose(j_cap_new, j_cap_old, rtol=5.0e-5, atol=5.0e-4)


@pytest.mark.parametrize(
    "closure_mode",
    ["cross_carrier", "plane_projection", "plane_closure"],
)
def test_live_jv_parity_interface_defect_modes(
    closure_mode: str,
    legacy_catalog: DefaultCatalog,
    tmp_path: Path,
) -> None:
    """T4 DC J-V parity: explicit single-level interface defects across deduplicated modes."""
    doc_ref = _moderate_pin_yaml_dict()
    prep_ref, _ = _load_paired_models(
        doc_ref, legacy_catalog, tmp_path, f"pin_defect_zero_{closure_mode}"
    )
    dev_ref = compile_device_1d(prep_ref, intervals_per_layer=(8, 16, 8))

    # Mix bare SRV on HTL/absorber with an explicit microscopic defect on absorber/ETL!
    doc = _moderate_pin_yaml_dict()
    for layer in doc["layers"]:
        layer["A_star_n"] = 1.2017e6
        layer["A_star_p"] = 1.2017e6
    doc["device"]["interfaces"] = [[20.0, 15.0], [0.0, 0.0]]
    doc["device"]["interface_defects"] = [
        None,
        {
            "sigma_n_cm2": 8.0e-17,
            "sigma_p_cm2": 6.0e-17,
            "v_th_cm_s": 1.0e7,
            "N_t_cm2": 5.0e10,
            "E_t_eV_below_cb": 0.60,
        },
    ]
    if closure_mode == "plane_projection":
        doc["device"]["interface_plane_projection"] = True
    elif closure_mode == "plane_closure":
        doc["device"]["interface_plane_closure"] = True
        doc["device"]["interface_plane_generation"] = True

    prep, stack = _load_paired_models(doc, legacy_catalog, tmp_path, f"pin_defect_{closure_mode}")
    dev = compile_device_1d(
        prep,
        intervals_per_layer=(8, 16, 8),
        te_softness=0.02,
        te_regularization_mode="legacy_logistic",
    )
    x_legacy = build_electrical_grid(stack, 32)
    np.testing.assert_allclose(dev.nodes_m, x_legacy, rtol=0.0, atol=1.0e-15)

    voltages = np.linspace(0.0, 1.25, 26)
    _, _, m_ref, _ = _sweep_solarlab_jv(dev_ref, voltages)
    v_old, j_old, m_old, _, _ = _sweep_legacy_ss_jv(stack, x_legacy, voltages, te_softness=0.02)
    _, j_new, m_new, _ = _sweep_solarlab_jv(dev, v_old, stop_after_voc=False)

    assert m_ref.V_oc - m_new.V_oc > 0.04
    _assert_jv_metrics_parity(m_new, m_old)
    np.testing.assert_allclose(j_new, j_old, rtol=5.0e-5, atol=5.0e-4)


def test_live_jv_parity_charged_two_sided_interface_closure(
    legacy_catalog: DefaultCatalog,
    tmp_path: Path,
) -> None:
    """T4 DC J-V + sheet-charge parity: equilibrium-referenced two-sided heterointerfaces."""
    # Part 1: Public D4-E3 photovoltaic reference fixture (interface_charge_jv_research.yaml)
    ref_yaml = ROOT / "perovskite-sim/tests/fixtures/configs/interface_charge_jv_research.yaml"
    raw_ref_doc = _clean_json(dict(load_yaml_mapping(ref_yaml.read_bytes())))
    raw_ref_doc["device"] = {
        **dict(raw_ref_doc["device"]),
        "ion_steric_diffusion_only": False,
        "ion_steric_shared_site": False,
    }
    prep_ref, stack_ref = _load_paired_models(
        raw_ref_doc,
        legacy_catalog,
        tmp_path,
        "iface_charge_ref",
    )
    x_dedup = build_electrical_grid(stack_ref, 16)
    l0 = stack_ref.layers[0].thickness
    n_l0 = int(np.sum(x_dedup <= l0 + 1.0e-15)) - 1
    n_l1 = (x_dedup.size - 1) - n_l0
    x_ref = build_two_sided_trace_grid(x_dedup, stack_ref)
    dev_ref_native = compile_device_1d(prep_ref, intervals_per_layer=(n_l0, n_l1))
    np.testing.assert_allclose(dev_ref_native.nodes_m, x_ref, rtol=0.0, atol=1.0e-15)

    # Bind exact nodal Beer-Lambert source used by legacy QF solver so discrete PDEs match 1-to-1
    stack_ref_off = dataclasses.replace(
        stack_ref,
        interface_charge_closure="off",
        interface_charge_rebaseline_acknowledged=False,
    )
    mat_ref_off = build_material_arrays(x_ref, stack_ref_off)
    g_ref_leg = beer_lambert_generation(x_ref, mat_ref_off.alpha, stack_ref.Phi)
    dev_ref = dataclasses.replace(
        dev_ref_native,
        generation=GenerationProfile(
            method="legacy_beer_lambert_midpoint_oracle",
            geometry_name=dev_ref_native.geometry.name,
            source_m3_s=g_ref_leg,
            absorbed_m2_s=g_ref_leg * dev_ref_native.geometry.cell_measures_m,
            binding_sha256=dev_ref_native.generation.binding_sha256,
            control_volume_sha256=dev_ref_native.generation.control_volume_sha256,
        ),
    )

    voltages_ref = np.linspace(0.0, 0.12, 7)
    proto_ref = build_interface_charge_jv_protocol(stack_ref, voltages_ref)
    legacy_ref = solve_interface_charge_jv(x_ref, stack_ref, proto_ref)
    v_ref_used = np.asarray(legacy_ref.sweep.voltages_V, dtype=float)
    ref_opts = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-11,
        max_bias_step_V=0.02,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    _, j_ref_new, m_ref_new, states_ref = _sweep_solarlab_jv(
        dev_ref,
        v_ref_used,
        options=ref_opts,
        stop_after_voc=False,
    )
    _assert_jv_metrics_parity(m_ref_new, legacy_ref.sweep.metrics)
    np.testing.assert_allclose(j_ref_new, legacy_ref.sweep.currents_A_m2, rtol=1.0e-5, atol=1.0e-6)

    # Also verify native BeerLambertBinding end-to-end compilation agrees within O(dx^2) optical quadrature on N=16
    _, _, m_ref_native, _ = _sweep_solarlab_jv(
        dev_ref_native,
        voltages_ref,
        options=ref_opts,
        stop_after_voc=True,
    )
    assert abs(m_ref_native.V_oc - legacy_ref.sweep.metrics.V_oc) <= 5.0e-4

    # Dark equilibrium reference occupancy + biased sheet charge & occupancy parity!
    assert dev_ref.interface_model is not None
    f_eq_new = np.array(
        [
            s.charge.equilibrium_occupancy
            for s in dev_ref.interface_model.two_sided
            if s.charge is not None
        ]
    )
    np.testing.assert_allclose(
        f_eq_new,
        legacy_ref.dark_reference.equilibrium_occupancy,
        rtol=1.0e-7,
        atol=1.0e-9,
    )
    for k, (st, pt_old) in enumerate(zip(states_ref, legacy_ref.sweep.points)):
        sigma_new = dev_ref.interface_sheet_charges_C_m2(st.u, voltage_V=float(v_ref_used[k]))
        occ_new = dev_ref.interface_occupancies(st.u, voltage_V=float(v_ref_used[k]))
        np.testing.assert_allclose(
            sigma_new,
            np.asarray(pt_old.interface_incremental_sheet_charge_C_m2, dtype=float),
            rtol=1.0e-6,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            occ_new,
            np.asarray(pt_old.interface_occupancy, dtype=float),
            rtol=1.0e-6,
            atol=1.0e-8,
        )

    # Part 2: 3-layer HTL / perovskite / ETL stack with simultaneous charged defects at BOTH junctions!
    htl_3l = copy.deepcopy(raw_ref_doc["layers"][1])
    htl_3l.update({"name": "charge_htl", "role": "HTL", "N_A": 1.0e21, "N_D": 0.0, "chi": 3.95})
    etl_3l = copy.deepcopy(raw_ref_doc["layers"][1])
    etl_3l["N_D"] = 1.0e21
    doc_3l = copy.deepcopy(raw_ref_doc)
    doc_3l["device"]["Phi"] = 0.0
    doc_3l["layers"] = [htl_3l, raw_ref_doc["layers"][0], etl_3l]
    doc_3l["electrical_grid"] = {
        "interval_weights": {"charge_htl": 1.0, "charge_absorber": 1.0, "charge_etl": 1.0},
        "alphas": {"charge_htl": 2.0, "charge_absorber": 2.0, "charge_etl": 2.0},
    }
    doc_3l["device"]["interface_defects"] = [
        {
            "sigma_n_cm2": 6.0e-18,
            "sigma_p_cm2": 1.0e-17,
            "v_th_cm_s": 1.0e7,
            "N_t_cm2": 1.0e11,
            "E_t_eV_below_cb": 0.60,
        },
        {
            "sigma_n_cm2": 8.0e-18,
            "sigma_p_cm2": 6.0e-18,
            "v_th_cm_s": 1.0e7,
            "N_t_cm2": 1.5e11,
            "E_t_eV_below_cb": 0.55,
        },
    ]
    prep_3l, stack_3l = _load_paired_models(doc_3l, legacy_catalog, tmp_path, "pin_3l_charged")
    x_3l = build_two_sided_trace_grid(build_electrical_grid(stack_3l, 12), stack_3l)
    dev_3l = compile_device_1d(prep_3l, intervals_per_layer=(4, 4, 4))
    np.testing.assert_allclose(dev_3l.nodes_m, x_3l, rtol=0.0, atol=1.0e-15)

    dark_ref_3l = build_equilibrium_referenced_interface_charge_dark_reference(x_3l, stack_3l)
    assert dev_3l.interface_model is not None
    f_eq_3l = np.array(
        [
            s.charge.equilibrium_occupancy
            for s in dev_3l.interface_model.two_sided
            if s.charge is not None
        ]
    )
    np.testing.assert_allclose(
        f_eq_3l, dark_ref_3l.equilibrium_occupancy, rtol=1.0e-9, atol=1.0e-11
    )

    seed_old = None
    prev_new: State1D | None = None
    opts_3l = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-11,
        max_bias_step_V=0.04,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    for v_val in (0.0, 0.08, 0.16, 0.24):
        res_old = solve_equilibrium_referenced_interface_charge_steady_state(
            x_3l,
            stack_3l,
            v_val,
            dark_reference=dark_ref_3l,
            illuminated=False,
            initial_state=seed_old,
        )
        seed_old = res_old
        st_new = solve_steady_state(
            dev_3l,
            initial_state=prev_new,
            voltage_V=v_val,
            generation_scale=0.0,
            options=opts_3l,
        ).state
        prev_new = st_new
        assm_3l = AssembledDC1DProblem(dev_3l, voltage_V=v_val, generation_scale=0.0)
        j_3l_new = float(-dev_3l.voltage_polarity * np.median(assm_3l.current_density(st_new.u)))
        sigma_3l_new = dev_3l.interface_sheet_charges_C_m2(st_new.u, voltage_V=v_val)
        occ_3l_new = dev_3l.interface_occupancies(st_new.u, voltage_V=v_val)
        np.testing.assert_allclose(
            sigma_3l_new,
            np.asarray(res_old.interface_incremental_sheet_charge_C_m2, dtype=float),
            rtol=1.0e-7,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            occ_3l_new,
            np.asarray(res_old.interface_occupancy, dtype=float),
            rtol=1.0e-7,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            j_3l_new,
            float(res_old.current_A_m2),
            rtol=1.0e-4,
            atol=2.5e-8,
        )
    assert np.max(np.abs(sigma_3l_new)) > 1.0e-5


def test_live_backward_euler_step_parity_vs_perovskite_sim(
    legacy_catalog: DefaultCatalog,
    tmp_path: Path,
) -> None:
    """T4 Backward Euler transient step parity on 3-layer device with active interface defect + TE."""
    doc = _moderate_pin_yaml_dict()
    for layer in doc["layers"]:
        layer["A_star_n"] = 1.2017e6
        layer["A_star_p"] = 1.2017e6
    doc["device"]["interfaces"] = [[20.0, 15.0], [0.0, 0.0]]
    doc["device"]["interface_plane_projection"] = True
    doc["device"]["interface_defects"] = [
        None,
        {
            "sigma_n_cm2": 2.0e-16,
            "sigma_p_cm2": 1.5e-16,
            "v_th_cm_s": 1.0e7,
            "N_t_cm2": 5.0e10,
            "E_t_eV_below_cb": 0.60,
        },
    ]
    prep, stack = _load_paired_models(doc, legacy_catalog, tmp_path, "pin_be_parity")
    dev = compile_device_1d(
        prep,
        intervals_per_layer=(8, 16, 8),
        te_softness=0.02,
        te_regularization_mode="legacy_logistic",
    )
    x = build_electrical_grid(stack, 32)
    np.testing.assert_allclose(dev.nodes_m, x, rtol=0.0, atol=1.0e-15)
    mat = dataclasses.replace(build_material_arrays(x, stack), te_softness=0.02)
    n_nodes = dev.node_count

    v0, v1, dt_s = 0.50, 0.65, 1.0e-7

    # 1. Solve initial steady state at V = 0.50 V in both engines and confirm steady parity
    ss0_new = solve_steady_state(
        dev,
        voltage_V=v0,
        generation_scale=1.0,
        options=SteadyOptions(max_bias_step_V=0.05),
    )
    ss0_old = legacy_ss.solve_steady_state(x, stack, v0, illuminated=True, mat=mat)
    np.testing.assert_allclose(ss0_new.state.electron_density_m3, ss0_old.y[:n_nodes], rtol=1.0e-6)
    np.testing.assert_allclose(
        ss0_new.state.hole_density_m3, ss0_old.y[n_nodes : 2 * n_nodes], rtol=1.0e-6
    )

    # 2. Solve implicit Backward Euler step V0 -> V1 over dt_s on AssembledDC1DProblem in solarlab
    endpoint_new = AssembledDC1DProblem(dev, voltage_V=v1, generation_scale=1.0)
    be_step = BackwardEuler1DStep(endpoint=endpoint_new, previous=ss0_new.state, dt_s=dt_s)  # type: ignore[arg-type]
    be_settings = NewtonSettings(
        production_residual_scaling(dev),
        1.0,
        production_coordinates(dev),
        SteadyOptions().budget(),
        (),
        False,
    )
    be_res_new = solve_newton(be_step, be_step.initial_coordinates(voltage_lift=True), be_settings)
    assert be_res_new.accepted
    st1_new = dev.state(be_res_new.state, voltage_V=v1)

    # 3. Solve the identical implicit Backward Euler step on perovskite_sim assemble_rhs
    y0_old = ss0_old.y.copy()
    n_int = n_nodes - 2
    z_init = np.r_[np.log(y0_old[1 : n_nodes - 1]), np.log(y0_old[n_nodes + 1 : 2 * n_nodes - 1])]

    def legacy_be_residual(z: Vector) -> Vector:
        y_trial = y0_old.copy()
        y_trial[1 : n_nodes - 1] = np.exp(z[:n_int])
        y_trial[n_nodes + 1 : 2 * n_nodes - 1] = np.exp(z[n_int:])
        y_trial[0] = mat.n_L
        y_trial[n_nodes - 1] = mat.n_R
        y_trial[n_nodes] = mat.p_L
        y_trial[2 * n_nodes - 1] = mat.p_R
        dydt = assemble_rhs(0.0, y_trial, x, stack, mat=mat, illuminated=True, V_app=v1)
        dn = (y_trial[1 : n_nodes - 1] - y0_old[1 : n_nodes - 1]) / dt_s - dydt[1 : n_nodes - 1]
        dp = (
            (y_trial[n_nodes + 1 : 2 * n_nodes - 1] - y0_old[n_nodes + 1 : 2 * n_nodes - 1]) / dt_s
            - dydt[n_nodes + 1 : 2 * n_nodes - 1]
        )
        return np.r_[
            dn * dt_s / y_trial[1 : n_nodes - 1],
            dp * dt_s / y_trial[n_nodes + 1 : 2 * n_nodes - 1],
        ]

    sol_old = root(legacy_be_residual, z_init, method="hybr", tol=1.0e-12)
    assert sol_old.success, sol_old.message
    y1_old = y0_old.copy()
    y1_old[1 : n_nodes - 1] = np.exp(sol_old.x[:n_int])
    y1_old[n_nodes + 1 : 2 * n_nodes - 1] = np.exp(sol_old.x[n_int:])
    phi0_old = legacy_ss._phi_from_y(x, mat, y0_old, v0)
    phi1_old = legacy_ss._phi_from_y(x, mat, y1_old, v1)

    np.testing.assert_allclose(st1_new.potential_V, phi1_old, rtol=1.0e-8, atol=1.0e-9)
    np.testing.assert_allclose(
        st1_new.electron_density_m3, y1_old[:n_nodes], rtol=1.0e-6, atol=1.0e-3
    )
    np.testing.assert_allclose(
        st1_new.hole_density_m3, y1_old[n_nodes : 2 * n_nodes], rtol=1.0e-6, atol=1.0e-3
    )

    dx = np.diff(x)
    eps_face = EPS_0 * _harmonic_face_average(mat.eps_r)
    j_cond_new = endpoint_new.current_density(st1_new.u)
    j_disp_new = (
        eps_face * (-(np.diff(st1_new.potential_V) - np.diff(ss0_new.state.potential_V)) / dx) / dt_s
    )
    j_tot_new = j_cond_new + j_disp_new

    jn_old, jp_old = carrier_face_currents(
        x,
        phi1_old,
        y1_old[:n_nodes],
        y1_old[n_nodes : 2 * n_nodes],
        dict(mat.carrier_params),
    )
    j_disp_old = eps_face * (-(np.diff(phi1_old) - np.diff(phi0_old)) / dx) / dt_s
    j_tot_old = (jn_old + jp_old) + j_disp_old
    np.testing.assert_allclose(j_tot_new, j_tot_old, rtol=1.0e-6, atol=1.0e-5)


def _solarlab_ac_admittance_spectrum(
    dev: CompiledDevice1D,
    steady_state: State1D,
    v_dc: float,
    frequencies_Hz: Vector,
    *,
    generation_scale: float = 1.0,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    """Compute passive-convention small-signal AC admittance Y(omega) on AssembledDC1DProblem."""
    op = AssembledDC1DProblem(dev, voltage_V=v_dc, generation_scale=generation_scale)
    u0 = steady_state.u
    n_nodes = dev.node_count
    j0 = op.jacobian(u0)
    k0 = op.storage_jacobian(u0)

    phi_right_per_V = complex(-dev.voltage_polarity, 0.0)
    rhs = np.zeros(3 * n_nodes, dtype=np.complex128)
    rhs[n_nodes - 1] = phi_right_per_V
    fixed: dict[int, complex] = {0: 0.0j, n_nodes - 1: phi_right_per_V}
    for offset in (n_nodes, 2 * n_nodes):
        fixed[offset] = 0.0j
        fixed[offset + n_nodes - 1] = -phi_right_per_V / dev.thermal_voltage_V

    ac_sols = solve_frequency_systems(j0, k0, rhs, frequencies_Hz, fixed_coordinates=fixed)

    # Exact cancellation-free analytical face conduction tangents from CarrierTransport!
    eval_state = op.binding.assembler.state(u0, op.inputs)
    assert eval_state is not None
    t_eval = op.binding.transport.evaluate(eval_state, derivatives=True)
    e_fv = t_eval["electron"]
    h_fv = t_eval["hole"]
    assert isinstance(e_fv, FaceValue) and isinstance(h_fv, FaceValue)

    n_faces = n_nodes - 1
    d_jcond_du = np.zeros((n_faces, 3 * n_nodes), dtype=float)
    zero_vec = np.zeros(n_faces, dtype=float)
    for charge, fv in ((-Q, e_fv), (Q, h_fv)):
        l_phi = np.asarray(fv.left.get("potential", zero_vec), dtype=float)
        r_phi = np.asarray(fv.right.get("potential", zero_vec), dtype=float)
        l_zn = np.asarray(fv.left.get("electron", zero_vec), dtype=float)
        r_zn = np.asarray(fv.right.get("electron", zero_vec), dtype=float)
        l_zp = np.asarray(fv.left.get("hole", zero_vec), dtype=float)
        r_zp = np.asarray(fv.right.get("hole", zero_vec), dtype=float)
        for f in range(n_faces):
            d_jcond_du[f, f] += charge * l_phi[f]
            d_jcond_du[f, f + 1] += charge * r_phi[f]
            d_jcond_du[f, n_nodes + f] += charge * l_zn[f]
            d_jcond_du[f, n_nodes + f + 1] += charge * r_zn[f]
            d_jcond_du[f, 2 * n_nodes + f] += charge * l_zp[f]
            d_jcond_du[f, 2 * n_nodes + f + 1] += charge * r_zp[f]

    cap_face = dev.poisson.face_capacitance_F_m2
    dx = np.diff(dev.nodes_m)
    stencil_faces: set[int] = set()
    if dev.interface_model is not None:
        for dspec in dev.interface_model.defect_srh:
            if dspec.plane_params is not None or dspec.projection_weight > 0.0:
                stencil_faces.update((dspec.node - 1, dspec.node))
        for tspec in dev.interface_model.two_sided:
            stencil_faces.add(tspec.face)
    bulk_mask = np.array([f not in stencil_faces for f in range(n_faces)], dtype=bool)
    dx_bulk = dx[bulk_mask]
    weights = dx_bulk / float(np.sum(dx_bulk))
    polarity = dev.voltage_polarity

    y_spectrum = np.empty(frequencies_Hz.size, dtype=np.complex128)
    y_faces_all = np.empty((frequencies_Hz.size, n_faces), dtype=np.complex128)
    max_rel_spread = 0.0
    for idx_f, (freq, sol) in enumerate(zip(frequencies_Hz, ac_sols)):
        assert sol.componentwise_backward_error <= 1.0e-11
        du = sol.solution
        dphi = du[:n_nodes]
        omega = 2.0 * np.pi * float(freq)
        j_cond_ac = polarity * (d_jcond_du @ du)
        j_disp_ac = polarity * 1j * omega * (-cap_face * np.diff(dphi))
        y_faces = j_cond_ac + j_disp_ac
        y_faces_all[idx_f] = y_faces
        y_bulk = y_faces[bulk_mask]
        y_avg = complex(np.sum(weights * y_bulk))
        y_spectrum[idx_f] = y_avg
        spread = float(np.max(np.abs(y_bulk - y_avg)) / max(abs(y_avg), 1.0e-30))
        max_rel_spread = max(max_rel_spread, spread)

    return y_spectrum, y_faces_all, max_rel_spread


def test_live_small_signal_ac_admittance_parity_vs_perovskite_sim(
    legacy_catalog: DefaultCatalog,
    tmp_path: Path,
) -> None:
    """T4 small-signal AC admittance/impedance parity across 1 Hz .. 1 MHz vs perovskite_sim."""
    freqs = np.logspace(0.0, 6.0, 7)
    v_dc = 0.65

    # Case 1: Active bare v_n/v_p heterointerface SRH compared directly against run_quasi_fermi_impedance
    # (run_quasi_fermi_impedance builds a matched-band stack without heterojunction offsets; match chi/Eg/DOS)
    doc_srh = _moderate_pin_yaml_dict()
    ni_match = _thermodynamic_ni(1.0e25, 1.0e25, 1.60)
    for layer in doc_srh["layers"]:
        layer["chi"] = 3.9
        layer["Eg"] = 1.60
        layer["Nc300"] = 1.0e25
        layer["Nv300"] = 1.0e25
        layer["ni"] = ni_match
        layer["n1"] = ni_match
        layer["p1"] = ni_match
    doc_srh["device"]["interfaces"] = [[30.0, 20.0], [15.0, 35.0]]
    prep_srh, stack_srh = _load_paired_models(doc_srh, legacy_catalog, tmp_path, "pin_ac_srh")
    dev_srh = compile_device_1d(prep_srh, intervals_per_layer=(8, 16, 8))
    x_srh = build_electrical_grid(stack_srh, 32)

    ss_srh = solve_steady_state(
        dev_srh,
        voltage_V=v_dc,
        generation_scale=1.0,
        options=SteadyOptions(max_bias_step_V=0.05),
    )
    y_new_srh, _, spread_srh = _solarlab_ac_admittance_spectrum(dev_srh, ss_srh.state, v_dc, freqs)
    qf_imp = run_quasi_fermi_impedance(
        x_srh,
        stack_srh,
        freqs,
        V_dc=v_dc,
        delta_V=0.005,
        illuminated=True,
    )
    assert spread_srh <= 1.0e-8
    np.testing.assert_allclose(y_new_srh, qf_imp.Y, rtol=1.0e-4, atol=1.0e-6)
    np.testing.assert_allclose(1.0 / y_new_srh, qf_imp.Z, rtol=1.0e-4, atol=1.0e-8)

    # Case 2: Full heterojunction explicit interface defects (plane_closure) + finite 1D TE vs solve_frequency_domain
    doc_def = _moderate_pin_yaml_dict()
    for layer in doc_def["layers"]:
        layer["A_star_n"] = 1.2017e6
        layer["A_star_p"] = 1.2017e6
    doc_def["device"]["interfaces"] = [[20.0, 15.0], [0.0, 0.0]]
    doc_def["device"]["interface_plane_closure"] = True
    doc_def["device"]["interface_plane_generation"] = True
    doc_def["device"]["interface_defects"] = [
        None,
        {
            "sigma_n_cm2": 2.0e-16,
            "sigma_p_cm2": 1.5e-16,
            "v_th_cm_s": 1.0e7,
            "N_t_cm2": 5.0e10,
            "E_t_eV_below_cb": 0.60,
        },
    ]
    prep_def, stack_def = _load_paired_models(doc_def, legacy_catalog, tmp_path, "pin_ac_defect_te")
    dev_def = compile_device_1d(
        prep_def,
        intervals_per_layer=(8, 16, 8),
        te_softness=0.02,
        te_regularization_mode="legacy_logistic",
    )
    x_def = build_electrical_grid(stack_def, 32)
    mat_def = dataclasses.replace(build_material_arrays(x_def, stack_def), te_softness=0.02)
    ss_def_new = solve_steady_state(
        dev_def,
        voltage_V=v_dc,
        generation_scale=1.0,
        options=SteadyOptions(max_bias_step_V=0.05),
    )
    y_new_def, _, spread_def = _solarlab_ac_admittance_spectrum(
        dev_def, ss_def_new.state, v_dc, freqs
    )
    assert spread_def <= 1.0e-7

    ss_def_old = legacy_ss.solve_steady_state(x_def, stack_def, v_dc, illuminated=True, mat=mat_def)
    n_nodes = x_def.size
    n_int = n_nodes - 2
    y0_old = ss_def_old.y.copy()
    z0_old = np.zeros(2 * n_int, dtype=float)
    dx_def = np.diff(x_def)
    eps_face_def = EPS_0 * _harmonic_face_average(mat_def.eps_r)
    weights_def = dx_def / float(x_def[-1] - x_def[0])
    pol_def = float(mat_def.junction_polarity)

    def legacy_eval(coord: Vector, voltage: float) -> SmallSignalEvaluation:
        y_vec = y0_old.copy()
        y_vec[1 : n_nodes - 1] = y0_old[1 : n_nodes - 1] * np.exp(coord[:n_int])
        y_vec[n_nodes + 1 : 2 * n_nodes - 1] = y0_old[n_nodes + 1 : 2 * n_nodes - 1] * np.exp(
            coord[n_int:]
        )
        y_vec[0] = mat_def.n_L
        y_vec[n_nodes - 1] = mat_def.n_R
        y_vec[n_nodes] = mat_def.p_L
        y_vec[2 * n_nodes - 1] = mat_def.p_R
        phi_vec = legacy_ss._phi_from_y(x_def, mat_def, y_vec, voltage)
        dydt = assemble_rhs(
            0.0, y_vec, x_def, stack_def, mat=mat_def, illuminated=True, V_app=voltage
        )
        jn, jp = carrier_face_currents(
            x_def,
            phi_vec,
            y_vec[:n_nodes],
            y_vec[n_nodes : 2 * n_nodes],
            dict(mat_def.carrier_params),
        )
        storage = np.r_[y_vec[1 : n_nodes - 1], y_vec[n_nodes + 1 : 2 * n_nodes - 1]]
        rate = np.r_[dydt[1 : n_nodes - 1], dydt[n_nodes + 1 : 2 * n_nodes - 1]]
        cond_passive = pol_def * (jn + jp)
        disp_charge_passive = pol_def * eps_face_def * (-np.diff(phi_vec) / dx_def)
        return SmallSignalEvaluation(
            storage=storage,
            rate=rate,
            conduction_current_faces=cond_passive,
            displacement_charge_faces=disp_charge_passive,
        )

    fd_old = solve_frequency_domain(
        legacy_eval,
        z0_old,
        v_dc,
        freqs,
        state_step=1.0e-5,
        voltage_step=1.0e-5,
        face_weights=weights_def,
    )
    np.testing.assert_allclose(y_new_def, fd_old.admittance, rtol=1.0e-4, atol=1.0e-6)
    np.testing.assert_allclose(1.0 / y_new_def, fd_old.impedance, rtol=1.0e-4, atol=1.0e-8)

    # Case 3: Equilibrium-referenced charged two-sided trace small-signal AC Maxwell conservation
    # + quasi-static DC conductance parity against perovskite_sim charged QF steady-state slope!
    ref_yaml = ROOT / "perovskite-sim/tests/fixtures/configs/interface_charge_jv_research.yaml"
    raw_ch = _clean_json(dict(load_yaml_mapping(ref_yaml.read_bytes())))
    raw_ch["device"] = {
        **dict(raw_ch["device"]),
        "ion_steric_diffusion_only": False,
        "ion_steric_shared_site": False,
    }
    prep_ch, stack_ch = _load_paired_models(raw_ch, legacy_catalog, tmp_path, "pin_ac_charged")
    x_ch_dedup = build_electrical_grid(stack_ch, 16)
    l0_ch = stack_ch.layers[0].thickness
    n0_ch = int(np.sum(x_ch_dedup <= l0_ch + 1.0e-15)) - 1
    n1_ch = (x_ch_dedup.size - 1) - n0_ch
    x_ch = build_two_sided_trace_grid(x_ch_dedup, stack_ch)
    dev_ch_nat = compile_device_1d(prep_ch, intervals_per_layer=(n0_ch, n1_ch))
    stack_ch_off = dataclasses.replace(
        stack_ch,
        interface_charge_closure="off",
        interface_charge_rebaseline_acknowledged=False,
    )
    mat_ch_off = build_material_arrays(x_ch, stack_ch_off)
    g_ch_leg = beer_lambert_generation(x_ch, mat_ch_off.alpha, stack_ch.Phi)
    dev_ch = dataclasses.replace(
        dev_ch_nat,
        generation=GenerationProfile(
            method="legacy_beer_lambert_midpoint_oracle",
            geometry_name=dev_ch_nat.geometry.name,
            source_m3_s=g_ch_leg,
            absorbed_m2_s=g_ch_leg * dev_ch_nat.geometry.cell_measures_m,
            binding_sha256=dev_ch_nat.generation.binding_sha256,
            control_volume_sha256=dev_ch_nat.generation.control_volume_sha256,
        ),
    )

    v_ch_dc = 0.05
    ss_ch = solve_steady_state(
        dev_ch,
        voltage_V=v_ch_dc,
        generation_scale=1.0,
        options=SteadyOptions(
            continuity_tolerance_A_m2=1.0e-11,
            max_bias_step_V=0.02,
            convergence_policy=POISSON_CONTACT_BALANCE_V1,
        ),
    )
    freqs_ch = np.r_[1.0e-4, freqs]
    _, y_faces_ch, _ = _solarlab_ac_admittance_spectrum(dev_ch, ss_ch.state, v_ch_dc, freqs_ch)
    assert dev_ch.interface_model is not None
    assert len(dev_ch.interface_model.two_sided) == 1
    f_iface = dev_ch.interface_model.two_sided[0].face
    # Within each bulk layer, Maxwell total current J_cond + j*omega*D is spatially uniform;
    # across the QSS static sheet-charge plane, D jumps by Delta_sigma(u) so J_tot jumps by j*omega*d(Delta_sigma).
    for k_f in range(freqs_ch.size):
        left_y = y_faces_ch[k_f, :f_iface]
        right_y = y_faces_ch[k_f, f_iface + 1 :]
        for seg in (left_y, right_y):
            rel_seg_spread = float(
                np.max(np.abs(seg - seg[0])) / max(abs(seg[0]), 1.0e-30)
            )
            assert rel_seg_spread <= 1.0e-6

    dv = 5.0e-4
    dark_ref_ch = build_equilibrium_referenced_interface_charge_dark_reference(x_ch, stack_ch)
    s_minus = solve_equilibrium_referenced_interface_charge_steady_state(
        x_ch, stack_ch, v_ch_dc - dv, dark_reference=dark_ref_ch, illuminated=True
    )
    s_plus = solve_equilibrium_referenced_interface_charge_steady_state(
        x_ch,
        stack_ch,
        v_ch_dc + dv,
        dark_reference=dark_ref_ch,
        illuminated=True,
        initial_state=s_minus,
    )
    # Passive current is -J_solar, so dI_passive/dV == -(J_solar(V+dv) - J_solar(V-dv)) / (2*dv)
    g_dc_legacy = -(float(s_plus.current_A_m2 - s_minus.current_A_m2) / (2.0 * dv))
    for y_dc_seg in (complex(y_faces_ch[0, 0]), complex(y_faces_ch[0, -1])):
        np.testing.assert_allclose(y_dc_seg.real, g_dc_legacy, rtol=1.0e-4, atol=1.0e-6)
        assert abs(y_dc_seg.imag) <= 1.0e-5 * abs(y_dc_seg.real)

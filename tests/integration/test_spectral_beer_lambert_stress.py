"""Integration stress tests for spectral Beer-Lambert on scaps_mirror_v2 (P08-04).

Re-certifies the historical archive/wip-beer-lambert (658baa84) N=60 stress
case on the refactored solarlab 1D kernel:
- Exact spectral Beer-Lambert photon budget (~260.14 A/m^2 on 300..1000 nm AM1.5G)
  with incoherent front glass substrate and mesh-invariant control-volume source.
- DC equilibrium, illuminated J-V operating points bracketing V_oc, and spatial
  refinement (N = 30 -> 60 -> 120 -> 240).
- Conservative Backward-Euler transient illumination turn-on and 5 V/s forward
  bias scan across absorber-front node 20 (where legacy density-variable Radau
  drove p < 0 and collapsed step size), proving min(n) > 0, min(p) > 0 without
  clipping and exact step-accounting conservation closure.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, cast

import numpy as np
import pytest

from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.device.compile_1d import CompiledDevice1D, compile_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError
from solarlab.experiments.transient import (
    TransientOptions,
    TransientProtocol,
    solve_transient,
)
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.physics.generation import sample_tabulated
from solarlab.physics.statistics.thermal import thermal_voltage
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    solve_equilibrium,
    solve_steady_state,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "solarlab" / "materials" / "data"


def _am15g_resource_table(n_wavelengths: int = 200) -> ResourceTable:
    raw = np.loadtxt(DATA_DIR / "am15g.csv", delimiter=",", comments="#")
    wl_nm = np.linspace(300.0, 1000.0, n_wavelengths)
    flux = sample_tabulated(raw[:, 0], raw[:, 1], wl_nm, name="am15g")
    lines = ["wavelength_nm,spectral_flux_m-2_s-1_m-1\n"]
    lines.extend(f"{repr(float(w))},{repr(float(f))}\n" for w, f in zip(wl_nm, flux))
    name = f"am15g_{n_wavelengths}"
    return ResourceTable(
        name,
        "spectrum",
        SourceDocument(f"{name}.csv", "".join(lines).encode("utf-8")),
    )


def _prepare_scaps_mirror_v2_new_core(
    defaults: DefaultCatalog,
    *,
    n_wavelengths: int = 200,
) -> PreparedDevice:
    """Import scaps_mirror_v2.yaml and bind its physical stack on the 1D kernel."""

    scaps_src = SourceDocument(
        "configs/scaps_mirror_v2.yaml",
        (ROOT / "src/solarlab/materials/data/presets/scaps_mirror_v2.yaml").read_bytes(),
    )
    nk_tables = tuple(
        ResourceTable(
            mat,
            "nk",
            SourceDocument(f"nk:{mat}.csv", (DATA_DIR / "nk" / f"{mat}.csv").read_bytes()),
        )
        for mat in ("glass", "spiro_OMeTAD", "MAPbI3", "TiO2")
    )
    am15g = _am15g_resource_table(n_wavelengths)
    resources = ResourceLibrary((am15g, *nk_tables))

    imported = import_scaps_device(scaps_src, id="scaps_mirror_v2", defaults=defaults)
    data: dict[str, Any] = imported.editing_data()
    data.pop("legacy_fields", None)
    data["spectrum"] = am15g.name

    temp_k = 300.0
    vt = float(thermal_voltage(temp_k))
    data["settings"].update(
        T=temp_k,
        mode="full",
        built_in_potential_mode="semiconductor_work_function",
        dos_band_potentials=True,
        flat_band_metal_contacts=False,
        ion_steric_diffusion_only=False,
        ion_steric_shared_site=False,
        het_recomb_despike=0.0,
        jv_solver_policy="general",
        Phi=0.0,
    )

    scaps_fallbacks = dict(defaults.scaps)
    for layer in data["layers"]:
        layer["parameterization"] = "standard"
        layer.pop("scaps_defect_metadata", None)
        params = layer.setdefault("parameters", {})
        for key, fallback in scaps_fallbacks.items():
            params.setdefault(key, fallback)
        if layer["role"] == "substrate":
            # Optical-only glass front substrate retains incoherent=True.
            assert params.get("incoherent") is True
            params.update(ni=1.0, n1=1.0, p1=1.0)
            continue
        nc = float(params["Nc300"])
        nv = float(params["Nv300"])
        eg = float(params["Eg"])
        ni = float(np.sqrt(nc * nv) * np.exp(-eg / (2.0 * vt)))
        params.update(
            ni=ni,
            n1=ni,
            p1=ni,
            tau_n=1.0e-3,
            tau_p=1.0e-3,
            P0=0.0,
            P0_neg=0.0,
            D_ion=0.0,
            D_ion_neg=0.0,
        )
        if layer.get("bulk_defects"):
            # Bind both Perovskite-CB and Perovskite-VB as explicit neutral DEF-1 SRH.
            layer["defect_schema_version"] = "solarlab-explicit-bulk-defects-v1"
            layer["defect_model"] = "explicit_quasi_steady"
            for defect in layer["bulk_defects"]:
                defect["charge_transition"] = "neutral"
                defect["neutral_reference"] = "all_occupancies"
                defect["degeneracy"] = 1.0

    # Both HTL/PVK and PVK/ETL interface defects from scaps_mirror_v2.yaml are
    # already imported as InterfaceDefectInput (N_t=1e12 cm^-2, sigma=1e-19 cm^2,
    # E_t=0.6 eV below CB, with preserved partner_metadata).
    assert sum(1 for iface in data.get("interfaces", []) if iface.get("defect")) == 2

    return resolve_device(DeviceInput.model_validate(data), defaults, resources)


@pytest.fixture(scope="module")
def defaults() -> DefaultCatalog:
    from solarlab.materials.packaged import load_packaged_default_catalog
    return load_packaged_default_catalog("device_only")


@pytest.fixture(scope="module")
def scaps_prepared(defaults: DefaultCatalog) -> PreparedDevice:
    return _prepare_scaps_mirror_v2_new_core(defaults, n_wavelengths=200)


@pytest.fixture(scope="module")
def scaps_device_n60(scaps_prepared: PreparedDevice) -> CompiledDevice1D:
    # N=60 total electrical intervals (20 per layer -> 61 nodes; HTL/PVK shared
    # absorber-front node is exact index 20, matching the 658baa84 witness).
    return compile_device_1d(
        scaps_prepared,
        intervals_per_layer=(20, 20, 20),
        optical_model="beer_lambert_spectral",
    )


def test_scaps_mirror_v2_n60_dc_jv_positivity_and_spatial_refinement(
    scaps_prepared: PreparedDevice,
    scaps_device_n60: CompiledDevice1D,
) -> None:
    # Incoherent glass substrate fails closed on coherent TMM, while compiling
    # cleanly on single-pass beer_lambert_spectral.
    with pytest.raises(CapabilityError, match="incoherent"):
        compile_device_1d(
            scaps_prepared, intervals_per_layer=(20, 20, 20), optical_model="tmm"
        )

    assert scaps_device_n60.node_count == 61
    assert scaps_device_n60.geometry.interfaces[0].shared_node == 20
    assert scaps_device_n60.generation.method == "beer_lambert_spectral"
    assert scaps_device_n60.generation_limit_A_m2 == pytest.approx(260.14, abs=0.15)

    options = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-6,
        current_spread_tolerance_A_m2=1.0e-6,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    eq = solve_equilibrium(scaps_device_n60, options=options)
    assert eq.converged
    assert np.min(eq.state.electron_density_m3) > 0.0
    assert np.min(eq.state.hole_density_m3) > 0.0
    assert eq.state.hole_density_m3[20] > 0.0

    # Short-circuit solve on deduplicated N=60 (exact node 20 shared HTL/PVK node).
    sc_dedup = solve_steady_state(
        scaps_device_n60,
        voltage_V=0.0,
        generation_scale=1.0,
        options=options,
    )
    assert sc_dedup.converged
    assert all(gate.passed for gate in sc_dedup.physical_gates)
    assert np.min(sc_dedup.state.electron_density_m3) > 0.0
    assert np.min(sc_dedup.state.hole_density_m3) > 0.0
    assert sc_dedup.state.hole_density_m3[20] > 0.0

    # Full illuminated J-V sweep across V_oc (0.0 -> 1.19 V) on two-sided trace
    # topology (59 nodes; HTL/PVK absorber-front right trace node is index 20),
    # resolving the 1.54 eV HTL/PVK conduction-band offset through V_oc (~1.178 V).
    scaps_trace_n60 = compile_device_1d(
        scaps_prepared,
        intervals_per_layer=(20, 20, 20),
        optical_model="beer_lambert_spectral",
        interface_topology="two_sided_trace",
    )
    assert scaps_trace_n60.node_count == 59
    assert scaps_trace_n60.geometry.interfaces[0].right_node == 20
    assert scaps_trace_n60.generation_limit_A_m2 == pytest.approx(
        scaps_device_n60.generation_limit_A_m2, rel=1.0e-13
    )

    sc = solve_steady_state(
        scaps_trace_n60,
        voltage_V=0.0,
        generation_scale=1.0,
        options=options,
    )
    assert sc.converged
    assert all(gate.passed for gate in sc.physical_gates)

    voltages = (0.0, 0.2, 0.5, 0.8, 1.0, 1.10, 1.14, 1.16, 1.17, 1.18, 1.19)
    currents: list[float] = [float(np.mean(sc.contact_current_density_A_m2))]
    seed = sc
    for v_bias in voltages[1:]:
        pt = solve_steady_state(
            scaps_trace_n60,
            initial_state=seed.state,
            voltage_V=v_bias,
            generation_scale=1.0,
            options=options,
        )
        assert pt.converged
        assert all(gate.passed for gate in pt.physical_gates)
        assert np.min(pt.state.electron_density_m3) > 0.0
        assert np.min(pt.state.hole_density_m3) > 0.0
        assert pt.state.hole_density_m3[20] > 0.0
        currents.append(float(np.mean(pt.contact_current_density_A_m2)))
        seed = pt

    # Photogenerated collection at V=0 has |J_sc| near ~258.95 A/m^2 (< 260.14 A/m^2
    # optical budget) and crosses zero cleanly between 1.17 V and 1.18 V (V_oc ~ 1.178 V).
    assert abs(currents[0]) == pytest.approx(258.95, abs=0.25)
    assert abs(currents[0]) <= scaps_trace_n60.generation_limit_A_m2
    assert currents[-3] < 0.0 < currents[-2]

    # Spatial refinement N = 30 -> 60 -> 120 -> 240: photon budget is mesh-exact
    # and short-circuit current converges smoothly.
    jsc_by_mesh: list[float] = []
    gen_limits: list[float] = []
    for n_layer in (10, 20, 40, 80):
        dev_n = compile_device_1d(
            scaps_prepared,
            intervals_per_layer=(n_layer, n_layer, n_layer),
            optical_model="beer_lambert_spectral",
        )
        gen_limits.append(dev_n.generation_limit_A_m2)
        sc_n = solve_steady_state(
            dev_n, voltage_V=0.0, generation_scale=1.0, options=options
        )
        assert np.min(sc_n.state.hole_density_m3) > 0.0
        jsc_by_mesh.append(abs(float(np.mean(sc_n.contact_current_density_A_m2))))

    np.testing.assert_allclose(gen_limits, gen_limits[0], rtol=1.0e-13)
    mesh_diffs = [
        abs(jsc_by_mesh[i + 1] - jsc_by_mesh[i]) for i in range(len(jsc_by_mesh) - 1)
    ]
    assert mesh_diffs[-1] < mesh_diffs[0]
    assert mesh_diffs[-1] < 0.1


def test_scaps_mirror_v2_n60_conservative_be_transient_positivity_and_closure(
    scaps_device_n60: CompiledDevice1D,
) -> None:
    started = time.perf_counter()
    steady_opts = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-6,
        current_spread_tolerance_A_m2=1.0e-6,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    dark_eq = solve_equilibrium(scaps_device_n60, options=steady_opts)
    sc_eq = solve_steady_state(
        scaps_device_n60,
        voltage_V=0.0,
        generation_scale=1.0,
        options=steady_opts,
    )

    # Stage 1: Dark-to-1-sun step turn-on across geometric time steps (0 -> 1 sun
    # over 1e-11 .. 1e-5 s), directly exercising the absorber-front node-20 regime
    # where legacy 658baa84 density-variable Radau drove p < 0 and collapsed dt.
    turn_on_times = np.concatenate(([0.0], np.geomspace(1.0e-11, 1.0e-5, 13)))
    turn_on_protocol = TransientProtocol(
        times_s=turn_on_times,
        voltage_V=np.zeros_like(turn_on_times),
        generation_scale=np.concatenate(([0.0], np.ones(13))),
    )
    turn_on_opts = TransientOptions(
        poisson_tolerance=1.0e-11,
        continuity_tolerance_A_m2=1.0e-6,
        contact_log_tolerance=1.0e-11,
        potential_step_V=0.1,
        quasi_fermi_step=2.0,
        newton_budget=SolveBudget(
            max_iterations=80,
            max_residual_evaluations=5000,
            max_jacobian_evaluations=1000,
            max_backtracks=30,
        ),
        max_steps=64,
        wall_s=15.0,
        predictor="generation_inventory",
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    turn_on_result = solve_transient(
        scaps_device_n60,
        dark_eq,
        turn_on_protocol,
        options=turn_on_opts,
        initial_options=steady_opts,
        method="backward_euler",
    )

    # Stage 2: Illuminated 5 V/s forward scan (0.0 -> 0.80 V over 16 steps) with
    # 10% sinusoidal illumination modulation from accepted short-circuit state.
    scan_steps = 16
    scan_voltages = np.linspace(0.0, 0.80, scan_steps + 1)
    scan_times = scan_voltages / 5.0
    scan_gen = 1.0 + 0.1 * np.sin(np.pi * np.linspace(0.0, 1.0, scan_steps + 1))
    scan_gen[0] = 1.0
    scan_protocol = TransientProtocol(
        times_s=scan_times,
        voltage_V=scan_voltages,
        generation_scale=scan_gen,
    )
    scan_opts = TransientOptions(
        poisson_tolerance=1.0e-11,
        continuity_tolerance_A_m2=1.0e-6,
        contact_log_tolerance=1.0e-11,
        potential_step_V=0.1,
        quasi_fermi_step=2.0,
        newton_budget=SolveBudget(
            max_iterations=80,
            max_residual_evaluations=5000,
            max_jacobian_evaluations=1000,
            max_backtracks=30,
        ),
        max_steps=64,
        wall_s=15.0,
        predictor="previous",
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    scan_result = solve_transient(
        scaps_device_n60,
        sc_eq,
        scan_protocol,
        options=scan_opts,
        initial_options=steady_opts,
        method="backward_euler",
    )
    elapsed_s = time.perf_counter() - started
    assert elapsed_s < 10.0

    for res in (turn_on_result, scan_result):
        assert res.completed
        assert not res.failed_attempts

        # Strictly positive n and p across ALL nodes (specifically node 20) and ALL steps.
        min_n = min(float(np.min(st.electron_density_m3)) for st in res.states)
        min_p = min(float(np.min(st.hole_density_m3)) for st in res.states)
        min_p_node20 = min(float(st.hole_density_m3[20]) for st in res.states)
        assert min_n > 0.0
        assert min_p > 0.0
        assert min_p_node20 > 0.0

        # Verify conservative Backward-Euler step accounting at every accepted step.
        for attempt in res.attempts:
            assert attempt.accepted
            acc = attempt.accounting
            assert isinstance(acc, dict)
            assert float(cast(float, acc["poisson_normalized_max"])) <= 1.0e-11
            assert float(cast(float, acc["continuity_defect_bound_A_m2"])) <= 1.0e-6
            assert float(
                np.max(np.abs(np.asarray(acc["maxwell_identity_error_A_m2"])))
            ) <= 1.0e-9
            assert abs(float(cast(float, acc["global_charge_balance_A_m2"]))) <= 1.0e-6

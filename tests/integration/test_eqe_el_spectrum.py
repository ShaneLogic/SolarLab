"""T2/T4 integration and perovskite_sim parity tests for EQE(lambda) and EL spectra."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments.el_spectrum import (
    run_el_spectrum as legacy_run_el_spectrum,
)
from perovskite_sim.experiments.eqe import compute_eqe as legacy_compute_eqe
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from solarlab.config.resolve_device import resolve_device
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.el_spectrum import (
    BOLTZMANN_J_K,
    ELOptions,
    blackbody_photon_flux_m2_s_m,
    run_el_spectrum,
)
from solarlab.experiments.eqe import EQEOptions, compute_eqe
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument as MaterialSourceDocument
from solarlab.reproducibility.sources import SourceDocument
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state

ROOT = Path(__file__).resolve().parents[2]


def _import_example(mod_name: str) -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(mod_name)


@pytest.fixture(scope="module", autouse=True)
def blas1() -> object:
    with threadpool_limits(limits=1):
        yield


def _spectral_resource_library() -> ResourceLibrary:
    nk_path = ROOT / "perovskite-sim" / "perovskite_sim" / "data" / "nk" / "MAPbI3.csv"
    nk_bytes = nk_path.read_bytes()

    am15g_path = ROOT / "perovskite-sim" / "perovskite_sim" / "data" / "am15g.csv"
    data_lines: list[str] = []
    for line in am15g_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        wl_val = float(stripped.split(",", 1)[0])
        if 300.0 <= wl_val <= 1100.0:
            data_lines.append(stripped)
    spec_bytes = (
        "wavelength_nm,spectral_flux_m-2_s-1_m-1\n" + "\n".join(data_lines) + "\n"
    ).encode("utf-8")

    return ResourceLibrary(
        (
            ResourceTable(
                "MAPbI3",
                "nk",
                MaterialSourceDocument("MAPbI3.csv", nk_bytes),
            ),
            ResourceTable(
                "am15g",
                "spectrum",
                MaterialSourceDocument("am15g.csv", spec_bytes),
            ),
        )
    )


def _prepare_spectral_device(
    base_prepared: PreparedDevice,
    *,
    absorber_overrides: dict[str, Any] | None = None,
) -> PreparedDevice:
    editing = base_prepared.to_input().editing_data()
    editing["spectrum"] = "am15g"
    editing["fixed_generation"] = None
    settings = dict(editing.get("settings") or {})
    settings["Phi"] = 2.54e21
    editing["settings"] = settings
    editing["electrical_grid"] = [
        {"layer": layer["id"], "interval_weight": 1.0, "alpha": 3.0}
        for layer in editing["layers"]
        if layer["role"] != "substrate"
    ]

    ni_abs = 363572901223.5858
    for layer in editing["layers"]:
        params = dict(layer.get("parameters") or {})
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
        if layer["role"] == "absorber":
            params["optical_material"] = "MAPbI3"
            params.pop("n_optical", None)
            params["alpha"] = 0.0
            if absorber_overrides:
                params.update(absorber_overrides)
        else:
            params["n_optical"] = 1.8
            params["alpha"] = 0.0
        layer["parameters"] = params

    src = SourceDocument(
        "test.spectral_mapbi3_binding",
        json.dumps(editing, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return resolve_device(
        DeviceInput.model_validate(editing),
        base_prepared.defaults,
        _spectral_resource_library(),
        sources=(*base_prepared.sources, src),
    )


def _legacy_stack_from_prepared(prepared: PreparedDevice) -> DeviceStack:
    mapping = prepared.to_mapping()
    settings = mapping["settings"]
    layers: list[LayerSpec] = []
    for layer in prepared.layers:
        p = dict(layer.parameters)
        opt_mat = layer.optical.name if layer.optical is not None else None
        n_opt = None if p["n_optical"] is None else float(p["n_optical"])
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
            optical_material=opt_mat,
            n_optical=n_opt,
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


def test_eqe_finite_flux_linear_response_photon_budget_and_jsc_closure() -> None:
    steady_ex = _import_example("examples.research_1d.steady")
    prepared = _prepare_spectral_device(steady_ex.load_device())

    wl_coarse = np.array([350.0, 450.0, 550.0, 650.0, 750.0, 950.0])
    steady_opts = SteadyOptions(
        continuity_tolerance_A_m2=1e-7,
        current_spread_tolerance_A_m2=1e-7,
    )
    eqe_ff = compute_eqe(
        prepared,
        wl_coarse,
        Phi_incident=1e18,
        options=EQEOptions(
            intervals_per_layer=12,
            perturbation_mode="finite_flux",
            generation_discretization="cell_flux",
            steady=steady_opts,
        ),
    )
    eqe_lr = compute_eqe(
        prepared,
        wl_coarse,
        Phi_incident=1e18,
        options=EQEOptions(
            intervals_per_layer=12,
            perturbation_mode="linear_response",
            generation_discretization="cell_flux",
            steady=steady_opts,
        ),
    )

    assert np.all(eqe_ff.converged_mask)
    assert np.all(eqe_lr.converged_mask)
    assert np.max(np.abs(eqe_ff.photon_budget.spectral_closure_residual())) <= 1e-12
    assert np.all(eqe_ff.EQE >= -1e-9)
    assert np.all(eqe_ff.EQE <= eqe_ff.electrical_absorptance + 1e-6)
    assert eqe_ff.EQE[-1] < 1e-4  # sub-bandgap 850 nm MAPbI3 k -> 0

    # Small-flux finite perturbation agrees with exact linearized DC Jacobian tangent!
    np.testing.assert_allclose(eqe_ff.EQE, eqe_lr.EQE, rtol=5e-4, atol=5e-4)

    # Fine-grid J_sc_eqe closure against full-spectrum TMM 1-sun DC J_sc
    wl_fine = np.linspace(300.0, 900.0, 61)
    eqe_fine = compute_eqe(
        prepared,
        wl_fine,
        Phi_incident=1e18,
        options=EQEOptions(
            intervals_per_layer=12,
            perturbation_mode="linear_response",
            generation_discretization="cell_flux",
            steady=steady_opts,
            require_reference_spectrum=True,
        ),
    )
    compiled_full = compile_device_1d(prepared, intervals_per_layer=12)
    sol_full = solve_steady_state(
        compiled_full,
        voltage_V=0.0,
        generation_scale=1.0,
        options=steady_opts,
    )
    jsc_full = float(
        -compiled_full.voltage_polarity * sol_full.current_density_A_m2[0]
    )
    assert math.isfinite(eqe_fine.J_sc_eqe_A_m2)
    assert eqe_fine.integration_failure_reason is None
    assert eqe_fine.J_sc_eqe_A_m2 == pytest.approx(jsc_full, rel=2e-2)


def test_eqe_legacy_engine_parity_and_coupled_ionic_support() -> None:
    steady_ex = _import_example("examples.research_1d.steady")
    prepared = _prepare_spectral_device(steady_ex.load_device())
    legacy_stack = _legacy_stack_from_prepared(prepared)

    wl_parity = np.array([450.0, 650.0, 950.0])
    steady_opts = SteadyOptions(
        continuity_tolerance_A_m2=1e-7,
        current_spread_tolerance_A_m2=1e-7,
    )
    res_node = compute_eqe(
        prepared,
        wl_parity,
        Phi_incident=4e21,
        options=EQEOptions(
            intervals_per_layer=8,
            perturbation_mode="finite_flux",
            generation_discretization="node_profile",
            steady=steady_opts,
        ),
    )
    leg_eqe = legacy_compute_eqe(
        legacy_stack,
        wavelengths_nm=wl_parity,
        Phi_incident=4e21,
        N_grid=24,
        t_settle=1e-3,
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(res_node.EQE, leg_eqe.EQE, atol=2e-3, rtol=2e-3)
    assert res_node.J_sc_eqe_A_m2 == pytest.approx(leg_eqe.J_sc_integrated, rel=5e-3)

    # Coupled confined ionic EQE around genuine ionic DC operating point
    ion_multi = _import_example("examples.research_1d.ionic_multilayer")
    ion_steady = _import_example("examples.research_1d.ionic_steady")
    prepared_ion_spec = _prepare_spectral_device(
        ion_multi.prepare_confined_ion_device()
    )
    base_controls = ion_steady.controls(7)
    controls = replace(
        base_controls,
        electronic=replace(
            base_controls.electronic,
            continuity_tolerance_A_m2=1e-7,
            current_spread_tolerance_A_m2=1e-7,
        ),
    )
    res_ion = compute_eqe(
        prepared_ion_spec,
        [500.0, 650.0, 950.0],
        Phi_incident=1e20,
        options=EQEOptions(
            intervals_per_layer=6,
            ion_domain_layer="absorber",
            ionic_steady=controls,
        ),
    )
    assert np.all(res_ion.converged_mask)
    assert res_ion.EQE[0] > 0.2
    assert res_ion.EQE[2] < 1e-4


def test_el_spectrum_reciprocity_boltzmann_scaling_and_legacy_parity() -> None:
    steady_ex = _import_example("examples.research_1d.steady")
    base_dev = steady_ex.load_device()
    prepared_srh = _prepare_spectral_device(
        base_dev, absorber_overrides={"tau_n": 1e-8, "tau_p": 1e-8}
    )
    prepared_rad = _prepare_spectral_device(
        base_dev,
        absorber_overrides={"tau_n": 1.0, "tau_p": 1.0, "C_n": 0.0, "C_p": 0.0},
    )
    legacy_stack = _legacy_stack_from_prepared(prepared_srh)

    wl_el = np.linspace(650.0, 900.0, 26)
    steady_opts = SteadyOptions(
        continuity_tolerance_A_m2=1e-7,
        current_spread_tolerance_A_m2=1e-7,
    )
    el_low = run_el_spectrum(
        prepared_srh,
        V_inj_V=0.95,
        wavelengths_nm=wl_el,
        options=ELOptions(
            intervals_per_layer=16,
            absorptance_mode="exact_layer",
            steady=steady_opts,
        ),
    )
    el_high = run_el_spectrum(
        prepared_srh,
        V_inj_V=1.05,
        wavelengths_nm=wl_el,
        options=ELOptions(
            intervals_per_layer=16,
            absorptance_mode="exact_layer",
            steady=steady_opts,
        ),
    )

    # Exact Rau reciprocity identity check: Phi_EL = A_abs * phi_bb * exp(q V / kT)
    v_t = BOLTZMANN_J_K * el_high.temperature_K / ELEMENTARY_CHARGE_C
    expected_ratio = math.exp((1.05 - 0.95) / v_t)
    pos = el_low.Phi_EL_m2_s_m > 0.0
    np.testing.assert_allclose(
        el_high.Phi_EL_m2_s_m[pos] / el_low.Phi_EL_m2_s_m[pos],
        expected_ratio,
        rtol=1e-12,
    )
    phi_bb = blackbody_photon_flux_m2_s_m(wl_el * 1e-9, el_high.temperature_K)
    np.testing.assert_allclose(
        el_high.Phi_EL_m2_s_m,
        el_high.absorber_absorptance * phi_bb * math.exp(1.05 / v_t),
        rtol=1e-12,
    )

    # Suppressing SRH & Auger boosts luminescent efficiency and cuts non-radiative V_oc loss
    el_rad = run_el_spectrum(
        prepared_rad,
        V_inj_V=1.0,
        wavelengths_nm=wl_el,
        options=ELOptions(intervals_per_layer=16, steady=steady_opts),
    )
    el_srh_1v = run_el_spectrum(
        prepared_srh,
        V_inj_V=1.0,
        wavelengths_nm=wl_el,
        options=ELOptions(intervals_per_layer=16, steady=steady_opts),
    )
    assert el_rad.EQE_EL > 50.0 * el_srh_1v.EQE_EL
    assert el_rad.delta_V_nr_mV < el_srh_1v.delta_V_nr_mV - 100.0

    # Live parity against perovskite_sim.experiments.el_spectrum (node_trapezoid quadrature)
    el_node = run_el_spectrum(
        prepared_srh,
        V_inj_V=1.0,
        wavelengths_nm=wl_el,
        options=ELOptions(
            intervals_per_layer=8,
            absorptance_mode="node_trapezoid",
            steady=steady_opts,
        ),
    )
    leg_el = legacy_run_el_spectrum(
        legacy_stack,
        V_inj=1.0,
        wavelengths_nm=wl_el,
        N_grid=24,
        n_points_dark=15,
        v_rate=1.0,
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        el_node.absorber_absorptance, leg_el.absorber_absorptance, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        el_node.EL_spectrum_m2_s_nm, leg_el.EL_spectrum, rtol=1e-10, atol=1e-20
    )
    assert el_node.J_inj_pv_A_m2 == pytest.approx(leg_el.J_inj, rel=2e-3)
    assert el_node.EQE_EL == pytest.approx(leg_el.EQE_EL, rel=2e-3)
    assert el_node.delta_V_nr_mV == pytest.approx(leg_el.delta_V_nr_mV, abs=0.2)

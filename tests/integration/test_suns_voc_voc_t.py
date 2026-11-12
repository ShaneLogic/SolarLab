"""T2/T4 integration and perovskite_sim parity tests for Suns-V_oc and V_oc(T)."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments.protocol import VocSearchProtocol
from perovskite_sim.experiments.suns_voc import run_suns_voc as legacy_run_suns_voc
from perovskite_sim.experiments.voc_t import run_voc_t as legacy_run_voc_t
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.suns_voc import SunsVocOptions, run_suns_voc
from solarlab.experiments.voc_t import VocTOptions, run_voc_t
from solarlab.reproducibility.sources import SourceDocument

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


def test_suns_voc_parity_with_legacy_engine_and_dc_ideality() -> None:
    steady_ex = _import_example("examples.research_1d.steady")
    prepared = _prepare_homojunction_pin(steady_ex.load_device())
    compiled = compile_device_1d(prepared, intervals_per_layer=8)
    legacy_stack = _legacy_stack_from_prepared(prepared)
    parity_suns = (0.1, 1.0, 5.0)

    res = run_suns_voc(
        compiled,
        parity_suns,
        options=SunsVocOptions(voc_voltage_tol_V=1e-7),
    )
    leg = legacy_run_suns_voc(
        legacy_stack,
        suns_levels=parity_suns,
        N_grid=24,
        t_settle=2e-3,
        rtol=1e-4,
        atol=1e-6,
        voc_search=VocSearchProtocol(
            coarse_dwell_s=2e-3,
            bisection_dwell_s=2e-3,
            final_settle_s=2e-3,
            bisection_tolerance_V=1e-5,
            bisection_max_steps=45,
        ),
    )

    assert res.completed
    assert all(r is None for r in res.point_reasons)
    assert res.pseudo_ff_reason is None
    assert np.all(np.diff(res.V_oc_V) > 0.0)
    assert 1.0 <= res.ideality_factor_fit <= 2.5
    assert res.ideality_r_squared > 0.99
    assert np.all((res.local_ideality_factor >= 0.9) & (res.local_ideality_factor <= 2.5))

    # T4 parity: |dVoc| <= 1e-4 V and relative J_sc <= 2e-4 across 0.1..5.0 suns!
    np.testing.assert_allclose(res.V_oc_V, leg.V_oc, atol=1e-4, rtol=0.0)
    np.testing.assert_allclose(res.J_sc_A_m2, leg.J_sc, rtol=2e-4, atol=1e-6)
    assert res.pseudo_FF == pytest.approx(leg.pseudo_FF, rel=2e-4, abs=2e-4)


def test_suns_voc_unbracketed_cap_records_nan_and_explicit_reason() -> None:
    steady_ex = _import_example("examples.research_1d.steady")
    prepared = steady_ex.load_device()
    compiled = compile_device_1d(prepared, intervals_per_layer=8)

    # Cap voc_bracket_max_V at 0.5 V so V_oc (~1.2 V) cannot be bracketed.
    res = run_suns_voc(
        compiled,
        [0.1, 1.0],
        options=SunsVocOptions(voc_bracket_step_V=0.25, voc_bracket_max_V=0.5),
    )
    assert not res.completed
    assert np.all(res.converged_mask)
    assert not np.any(res.voc_bracketed_mask)
    assert np.all(np.isnan(res.V_oc_V))
    assert np.all(np.isfinite(res.J_sc_A_m2) & (res.J_sc_A_m2 > 0.0))
    assert all(
        r is not None and r.startswith("voc_not_bracketed:") for r in res.point_reasons
    )
    assert math.isnan(res.pseudo_FF)
    assert res.pseudo_ff_reason == "missing_or_nonpositive_reference"

    with pytest.raises(RuntimeError, match="failed to converge and bracket"):
        run_suns_voc(
            compiled,
            [1.0],
            options=SunsVocOptions(
                voc_bracket_step_V=0.25,
                voc_bracket_max_V=0.5,
                require_all_bracketed=True,
            ),
        )


def test_voc_t_cold_rebind_and_legacy_parity() -> None:
    steady_ex = _import_example("examples.research_1d.steady")
    prepared = _prepare_homojunction_pin(steady_ex.load_device())
    legacy_stack = _legacy_stack_from_prepared(prepared)

    res = run_voc_t(
        prepared,
        T_min=280.0,
        T_max=320.0,
        n_points=3,
        options=VocTOptions(
            intervals_per_layer=8,
            suns_voc=SunsVocOptions(voc_voltage_tol_V=1e-7),
        ),
    )
    leg = legacy_run_voc_t(
        legacy_stack,
        T_min=280.0,
        T_max=320.0,
        n_points=3,
        N_grid=24,
        v_rate=0.5,
        V_max=1.3,
        jv_n_points=66,
        rtol=1e-4,
        atol=1e-6,
    )

    assert res.completed
    assert res.fit_failure_reason is None
    assert len(set(res.compiled_identities)) == 3  # distinct cold-compiled device per T
    assert np.all(np.diff(res.V_oc_V) < 0.0)
    assert res.slope_V_K < 0.0
    assert 1.4 <= res.activation_energy_eV <= 1.8
    assert res.r_squared > 0.999

    np.testing.assert_allclose(res.V_oc_V, leg.V_oc_arr, atol=2.5e-3, rtol=0.0)
    assert res.activation_energy_eV == pytest.approx(leg.E_A_eV, abs=5e-3)


def test_coupled_ionic_suns_voc_solves_genuine_ionic_dc_operating_points() -> None:
    from dataclasses import replace

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
    res_ion = run_suns_voc(
        compiled_ion,
        [0.01, 0.05],
        options=SunsVocOptions(
            ionic_steady=controls,
            voc_voltage_tol_V=2e-4,
            voc_bracket_step_V=0.2,
        ),
    )
    assert res_ion.completed
    assert np.all(np.isfinite(res_ion.V_oc_V))
    assert res_ion.V_oc_V[1] > res_ion.V_oc_V[0] > 0.5
    assert res_ion.J_sc_A_m2[1] > res_ion.J_sc_A_m2[0] > 0.0

"""T2/T4 integration and parity tests for CF, CV, and Mott-Schottky recipes."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments.mott_schottky import (
    run_mott_schottky as legacy_run_mott_schottky,
)
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError
from solarlab.experiments.mott_schottky import (
    CFOptions,
    MottSchottkyOptions,
    run_capacitance_frequency,
    run_capacitance_voltage,
    run_mott_schottky,
)
from solarlab.materials.resources import ResourceLibrary
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
    no_thermionic = all(
        float(dict(layer.parameters)["A_star_n"]) == 0.0
        and float(dict(layer.parameters)["A_star_p"]) == 0.0
        for layer in prepared.layers
    )
    return DeviceStack(
        layers=tuple(layers),
        V_bi=float(settings["V_bi"]),
        Phi=float(settings["Phi"]),
        T=float(settings["T"]),
        mode="legacy" if no_thermionic else str(settings["mode"]),
        built_in_potential_mode=str(settings["built_in_potential_mode"]),
        dos_band_potentials=bool(settings["dos_band_potentials"]),
    )


def _prepare_doped_pn_homojunction(base_prepared: PreparedDevice) -> PreparedDevice:
    import json
    from solarlab.contracts.units import ELEMENTARY_CHARGE_C
    from solarlab.experiments.mott_schottky import BOLTZMANN_J_K

    ni_si = math.sqrt(2.8e25 * 1.04e25) * math.exp(
        -1.12 * ELEMENTARY_CHARGE_C / (2.0 * BOLTZMANN_J_K * 300.0)
    )
    editing = base_prepared.to_input().editing_data()
    editing["id"] = "doped_pn_homojunction"
    editing["electrical_grid"] = [
        {"layer": "emitter", "interval_weight": 40.0, "alpha": 3.0},
        {"layer": "absorber", "interval_weight": 80.0, "alpha": 3.0},
    ]
    editing["interfaces"] = []
    editing["layers"] = [
        {
            "id": "emitter",
            "name": "n+ emitter",
            "role": "ETL",
            "thickness": 8e-8,
            "parameters": {
                "chi": 4.05,
                "Eg": 1.12,
                "Nc300": 2.8e25,
                "Nv300": 1.04e25,
                "eps_r": 11.7,
                "N_D": 1e23,
                "N_A": 0.0,
                "ni": ni_si,
                "n1": ni_si,
                "p1": ni_si,
                "mu_n": 0.05,
                "mu_p": 0.02,
                "tau_n": 1e-5,
                "tau_p": 1e-5,
                "alpha": 0.0,
                "D_ion": 0.0,
                "D_ion_neg": 0.0,
                "P0": 0.0,
                "P0_neg": 0.0,
            },
        },
        {
            "id": "absorber",
            "name": "p base",
            "role": "absorber",
            "thickness": 2.0e-6,
            "parameters": {
                "chi": 4.05,
                "Eg": 1.12,
                "Nc300": 2.8e25,
                "Nv300": 1.04e25,
                "eps_r": 11.7,
                "N_D": 0.0,
                "N_A": 2e22,
                "ni": ni_si,
                "n1": ni_si,
                "p1": ni_si,
                "mu_n": 0.05,
                "mu_p": 0.02,
                "tau_n": 1e-5,
                "tau_p": 1e-5,
                "alpha": 0.0,
                "D_ion": 0.0,
                "D_ion_neg": 0.0,
                "P0": 0.0,
                "P0_neg": 0.0,
            },
        },
    ]
    src = SourceDocument(
        "test.doped_pn_homojunction",
        json.dumps(editing, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return resolve_device(
        DeviceInput.model_validate(editing),
        base_prepared.defaults,
        ResourceLibrary(),
        sources=(*base_prepared.sources, src),
    )


def test_electronic_cf_cv_and_mott_schottky_legacy_parity() -> None:
    steady_ex = _import_example("examples.research_1d.steady")
    prepared = steady_ex.load_device()
    compiled = compile_device_1d(prepared, intervals_per_layer=8)

    # 1) CF including 0 Hz (masked as 'zero_frequency', never 0.0!)
    cf = run_capacitance_frequency(
        compiled,
        [0.0, 1e2, 1e4, 1e6],
        bias_voltage_V=0.0,
    )
    assert cf.active_species == ("electrons", "holes")
    assert cf.frozen_species == ()
    assert not cf.valid_capacitive_mask[0]
    assert cf.point_reasons[0] == "zero_frequency"
    assert math.isnan(cf.capacitance_F_m2[0])
    assert np.all(cf.valid_capacitive_mask[1:])
    assert np.all(cf.capacitance_F_m2[1:] > 0.0)

    # 2) Mott-Schottky parity against legacy quasi_fermi_frequency engine on doped p-n diode
    pn_prepared = _prepare_doped_pn_homojunction(prepared)
    pn_compiled = compile_device_1d(pn_prepared, intervals_per_layer=60)
    pn_legacy = _legacy_stack_from_prepared(pn_prepared)

    v_grid = np.linspace(-0.2, 0.3, 6)
    ms = run_mott_schottky(
        pn_compiled,
        voltage_V=v_grid,
        frequency_Hz=1e5,
        prepared=pn_prepared,
        options=MottSchottkyOptions(),
    )
    leg_ms = legacy_run_mott_schottky(
        pn_legacy,
        V_range=v_grid,
        frequency=1e5,
        N_grid=120,
        impedance_method="quasi_fermi_frequency",
    )

    assert np.all(ms.valid_capacitive_mask)
    assert ms.fit_failure_reason is None
    np.testing.assert_allclose(ms.capacitance_F_m2, leg_ms.C, rtol=1e-3, atol=1e-12)
    assert ms.eps_r_used == pytest.approx(leg_ms.eps_r_used)
    assert ms.apparent_built_in_voltage_V == pytest.approx(leg_ms.V_bi_fit, abs=2e-3)
    assert ms.apparent_doping_m3 == pytest.approx(leg_ms.N_eff_fit, rel=5e-3)
    assert ms.r_squared > 0.999


def test_single_domain_ionic_mobile_vs_frozen_species_dispersion() -> None:
    ion_ac_ex = _import_example("examples.research_1d.ionic_impedance")
    ion_steady_ex = _import_example("examples.research_1d.ionic_steady")
    compiled_ion = ion_ac_ex.compile_case("shared")
    controls = ion_steady_ex.controls(compiled_ion.ion_node_count)
    freqs = [0.01, 1.0, 1e3, 1e6]

    cf_mob = run_capacitance_frequency(
        compiled_ion,
        freqs,
        bias_voltage_V=0.002,
        options=CFOptions(ionic_steady=controls, ion_mode="mobile"),
    )
    cf_frz = run_capacitance_frequency(
        compiled_ion,
        freqs,
        bias_voltage_V=0.002,
        options=CFOptions(ionic_steady=controls, ion_mode="frozen"),
    )

    assert "ion:positive" in cf_mob.active_species
    assert cf_mob.frozen_species == ()
    assert cf_frz.active_species == ("electrons", "holes")
    assert "ion:positive" in cf_frz.frozen_species

    assert np.all(cf_mob.valid_capacitive_mask)
    assert np.all(cf_frz.valid_capacitive_mask)
    # At 0.01 Hz mobile ions enhance low-frequency capacitance above frozen ions;
    # at 1 MHz ions freeze out dynamically and mobile -> frozen!
    assert cf_mob.capacitance_F_m2[0] > cf_frz.capacitance_F_m2[0]
    assert cf_mob.capacitance_F_m2[-1] == pytest.approx(
        cf_frz.capacitance_F_m2[-1], rel=1e-3
    )

    # CV on coupled ionic device
    cv_mob = run_capacitance_voltage(
        compiled_ion,
        [-0.05, 0.0, 0.05],
        frequency_Hz=1e5,
        options=CFOptions(ionic_steady=controls, ion_mode="mobile"),
    )
    assert np.all(cv_mob.dc_converged_mask)
    assert np.all(cv_mob.valid_capacitive_mask)


def test_multiple_domain_ionic_cf_and_cv() -> None:
    mult_dom = _import_example("examples.research_1d.multiple_ionic_domains")
    ion_steady_ex = _import_example("examples.research_1d.ionic_steady")
    device = compile_multiple_ionic_device_1d(
        mult_dom.prepare_multiple_ionic_device(),
        intervals_per_layer=mult_dom.INTERVALS,
        ion_domains=mult_dom.DOMAINS,
    )
    controls = ion_steady_ex.controls(4)

    cf_mult = run_capacitance_frequency(
        device,
        [0.1, 100.0, 1e5],
        bias_voltage_V=0.0,
        options=CFOptions(
            ionic_steady=controls,
            ion_mode="mobile",
            voltage_phasor_V=2e-6 + 1e-6j,
        ),
    )
    assert len(cf_mult.active_species) >= 4  # electrons, holes + 2 domains
    assert cf_mult.frozen_species == ()
    assert np.all(cf_mult.ac_accepted_mask)
    assert np.all(cf_mult.valid_capacitive_mask)

    # Two-domain AC explicitly refuses unsupported frozen mode via CapabilityError
    with pytest.raises(CapabilityError, match="mobile positive ions"):
        run_capacitance_frequency(
            device,
            [0.1, 100.0, 1e5],
            bias_voltage_V=0.0,
            options=CFOptions(
                ionic_steady=controls,
                ion_mode="frozen",
                voltage_phasor_V=2e-6 + 1e-6j,
            ),
        )

    cv_mult = run_capacitance_voltage(
        device,
        [0.0, 0.002, 0.005],
        frequency_Hz=1e4,
        options=CFOptions(
            ionic_steady=controls,
            ion_mode="mobile",
            voltage_phasor_V=2e-6 + 1e-6j,
        ),
    )
    assert np.all(cv_mult.dc_converged_mask)
    assert np.all(cv_mult.valid_capacitive_mask)

"""Integration and legacy-parity tests for P04-13 2T monolithic tandem J-V and joint TMM optics."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.research_1d.steady import load_device  # noqa: E402
from perovskite_sim.experiments import tandem_jv as legacy_tandem_jv  # noqa: E402
from solarlab.config.resolve_device import resolve_tandem  # noqa: E402
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q  # noqa: E402

K_B = 1.380649e-23
from solarlab.device.inputs import TandemInput  # noqa: E402
from solarlab.device.resolved import PreparedTandem  # noqa: E402
from solarlab.experiments.tandem_jv import (  # noqa: E402
    PHOTON_BUDGET_TOLERANCE,
    TandemCapabilityError,
    TandemJVProtocol,
    compute_tandem_generation,
    run_tandem_jv,
    series_match_jv,
)
from solarlab.materials.resources import ResourceLibrary, ResourceTable  # noqa: E402
from solarlab.materials.source import SourceDocument  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def single_thread_blas() -> object:
    with threadpool_limits(limits=1):
        yield


def _intrinsic_density_300K(nc300: float, nv300: float, eg_eV: float) -> float:
    return math.sqrt(nc300 * nv300) * math.exp(-eg_eV * Q / (2.0 * K_B * 300.0))


def _set_layer_band_structure(
    layer_doc: dict[str, Any],
    *,
    chi_eV: float,
    eg_eV: float,
    thickness_m: float | None = None,
) -> None:
    if thickness_m is not None:
        layer_doc["thickness"] = thickness_m
    params = layer_doc["parameters"]
    nc300 = float(params["Nc300"])
    nv300 = float(params["Nv300"])
    ni = _intrinsic_density_300K(nc300, nv300, eg_eV)
    params.update(chi=chi_eV, Eg=eg_eV, ni=ni, n1=ni, p1=ni)


@pytest.fixture(scope="module")
def prepared_monolithic_tandem() -> PreparedTandem:
    base = load_device()
    wavelengths_nm = (400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900)

    spec_csv = "wavelength_nm,spectral_flux_m-2_s-1_m-1\n" + "".join(
        f"{wl},4.5e27\n" for wl in wavelengths_nm
    )
    top_nk_csv = "wavelength_nm,n,k\n" + "".join(
        f"{wl},{2.55 if wl <= 650 else 2.40},{0.18 if wl <= 650 else 0.0}\n"
        for wl in wavelengths_nm
    )
    junc_nk_csv = "wavelength_nm,n,k\n" + "".join(
        f"{wl},2.05,0.015\n" for wl in wavelengths_nm
    )
    bot_nk_csv = "wavelength_nm,n,k\n" + "".join(
        f"{wl},2.70,0.16\n" for wl in wavelengths_nm
    )
    back_nk_csv = "wavelength_nm,n,k\n" + "".join(
        f"{wl},0.25,3.20\n" for wl in wavelengths_nm
    )

    tables = (
        ResourceTable(
            "tandem_photons",
            "spectrum",
            SourceDocument("tandem.photons.csv", spec_csv.encode("utf-8")),
        ),
        ResourceTable(
            "top_widegap_nk",
            "nk",
            SourceDocument("tandem.top_nk.csv", top_nk_csv.encode("utf-8")),
        ),
        ResourceTable(
            "ito_junction_nk",
            "nk",
            SourceDocument("tandem.junc_nk.csv", junc_nk_csv.encode("utf-8")),
        ),
        ResourceTable(
            "bot_narrowgap_nk",
            "nk",
            SourceDocument("tandem.bot_nk.csv", bot_nk_csv.encode("utf-8")),
        ),
        ResourceTable(
            "ag_back_nk",
            "nk",
            SourceDocument("tandem.back_nk.csv", back_nk_csv.encode("utf-8")),
        ),
    )

    top_doc = base.to_input().editing_data()
    top_doc["id"] = "tandem_top_widegap_pin"
    top_doc["spectrum"] = "tandem_photons"
    top_doc["settings"]["Phi"] = 0.0
    for lyr in top_doc["layers"]:
        p = lyr["parameters"]
        p.update(alpha=0.0, incoherent=False)
        if lyr["id"] == "htl":
            _set_layer_band_structure(lyr, chi_eV=3.55, eg_eV=1.90)
            p.update(optical_material=None, n_optical=1.95)
        elif lyr["id"] == "absorber":
            _set_layer_band_structure(
                lyr, chi_eV=3.85, eg_eV=1.68, thickness_m=2.5e-7
            )
            p.update(optical_material="top_widegap_nk", n_optical=None)
        else:
            p.update(optical_material=None, n_optical=2.05)

    bot_doc = base.to_input().editing_data()
    bot_doc["id"] = "tandem_bottom_narrowgap_pin"
    bot_doc["spectrum"] = "tandem_photons"
    bot_doc["settings"]["Phi"] = 0.0
    for lyr in bot_doc["layers"]:
        p = lyr["parameters"]
        p.update(alpha=0.0, incoherent=False)
        if lyr["id"] == "htl":
            _set_layer_band_structure(lyr, chi_eV=3.65, eg_eV=1.65)
            p.update(optical_material=None, n_optical=2.00)
        elif lyr["id"] == "absorber":
            _set_layer_band_structure(
                lyr, chi_eV=3.95, eg_eV=1.42, thickness_m=4.5e-7
            )
            p.update(optical_material="bot_narrowgap_nk", n_optical=None)
        else:
            p.update(optical_material=None, n_optical=2.10)

    tandem_doc: dict[str, Any] = {
        "schema_version": "solarlab.tandem-preparation.v1",
        "id": "monolithic_2t_research_tandem",
        "source_schema_version": 1,
        "device_type": "tandem_2T_monolithic",
        "top_cell": top_doc,
        "bottom_cell": bot_doc,
        "top_cell_reference": "inline:top_widegap_pin",
        "bottom_cell_reference": "inline:bottom_narrowgap_pin",
        "junction_model": "ideal_ohmic",
        "light_direction": "top_first",
        "junction_stack": [
            {
                "id": "recomb_ito",
                "name": "ITO recombination layer",
                "thickness": 2.0e-8,
                "optical_material": "ito_junction_nk",
                "incoherent": False,
            }
        ],
        "back_reflector": {
            "id": "ag_reflector",
            "name": "Ag back reflector",
            "thickness": 1.0e-7,
            "optical_material": "ag_back_nk",
            "incoherent": False,
        },
    }
    raw_src = SourceDocument(
        "tandem.declaration.json",
        json.dumps(tandem_doc, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    lib = ResourceLibrary(tables=tables)
    return resolve_tandem(
        TandemInput.model_validate(tandem_doc),
        base.defaults,
        lib,
        sources=(*base.sources, raw_src, *(tbl.source for tbl in tables)),
    )


def test_joint_tmm_tandem_photon_budget_closure_and_parasitic_exclusion(
    prepared_monolithic_tandem: PreparedTandem,
) -> None:
    optics = compute_tandem_generation(
        prepared_monolithic_tandem,
        intervals_top=(8, 8, 8),
        intervals_bottom=(8, 8, 8),
    )

    # 1. Exact spectral photon conservation across [top, junction, bottom, back_reflector]
    closure_res = optics.photon_budget.spectral_closure_residual()
    assert float(np.max(np.abs(closure_res))) <= PHOTON_BUDGET_TOLERANCE

    # 2. Control-volume integrated sub-cell generation matches joint TMM electrical absorption
    total_subcell_gen = optics.top_absorbed_m2_s + optics.bottom_absorbed_m2_s
    assert total_subcell_gen == pytest.approx(
        optics.photon_budget.electrical_absorbed_m2_s, rel=1e-12
    )
    assert optics.top_absorbed_m2_s > 0.0
    assert optics.bottom_absorbed_m2_s > 0.0

    # 3. Parasitic absorption in ITO recombination layer + Ag back reflector is > 0
    #    and strictly excluded from sub-cell carrier generation
    assert optics.photon_budget.parasitic_absorbed_m2_s > 0.0
    assert 0.0 < optics.parasitic_fraction < 1.0
    assert (
        total_subcell_gen + optics.photon_budget.parasitic_absorbed_m2_s
        == pytest.approx(
            float(np.sum(optics.photon_budget.layer_absorbed_m2_s)), rel=1e-12
        )
    )


def test_monolithic_2t_tandem_jv_voc_addition_current_limiting_and_legacy_parity(
    prepared_monolithic_tandem: PreparedTandem,
) -> None:
    prot = TandemJVProtocol(
        intervals_per_layer=8,
        voltage_points_per_subcell=37,
        top_voltage_min_V=-1.05,
        top_voltage_max_V=1.34,
        bottom_voltage_min_V=-0.1,
        bottom_voltage_max_V=1.08,
        matched_points=101,
    )
    res = run_tandem_jv(prepared_monolithic_tandem, prot)

    assert res.top_compiled_identity != res.bottom_compiled_identity
    assert res.top_metrics_fwd.open_circuit_voltage_V is not None
    assert res.bottom_metrics_fwd.open_circuit_voltage_V is not None
    assert res.metrics_fwd.open_circuit_voltage_V is not None
    assert res.metrics_fwd.short_circuit_current_A_m2 is not None
    assert res.top_metrics_fwd.short_circuit_current_A_m2 is not None
    assert res.bottom_metrics_fwd.short_circuit_current_A_m2 is not None

    # 1. Series voltage addition at open circuit: V_oc_tandem == V_oc_top + V_oc_bot
    voc_sum = (
        res.top_metrics_fwd.open_circuit_voltage_V
        + res.bottom_metrics_fwd.open_circuit_voltage_V
    )
    assert res.metrics_fwd.open_circuit_voltage_V == pytest.approx(voc_sum, abs=2e-3)
    assert res.metrics_fwd.open_circuit_voltage_V > 2.0

    # 2. Tandem short-circuit current is bounded by the current-limiting sub-cell
    limiting_jsc = min(
        res.top_metrics_fwd.short_circuit_current_A_m2,
        res.bottom_metrics_fwd.short_circuit_current_A_m2,
    )
    assert res.metrics_fwd.short_circuit_current_A_m2 == pytest.approx(
        limiting_jsc, rel=0.025
    )

    # 3. Live parity against perovskite_sim.experiments.tandem_jv.series_match_jv
    leg_j, _leg_vt, _leg_vb, leg_v = legacy_tandem_jv.series_match_jv(
        res.top_J_fwd,
        res.top_V_fwd,
        res.bottom_J_fwd,
        res.bottom_V_fwd,
        V_junction=-0.035,
    )
    sl_v_off, sl_j_off = series_match_jv(
        res.top_J_fwd,
        res.top_V_fwd,
        res.bottom_J_fwd,
        res.bottom_V_fwd,
        V_junction=-0.035,
        n_points=101,
    )
    leg_voc = float(np.interp(0.0, leg_j, leg_v))
    sl_voc = float(np.interp(0.0, sl_j_off[::-1], sl_v_off[::-1]))
    assert sl_voc == pytest.approx(leg_voc, abs=1e-12)
    assert sl_voc == pytest.approx(
        res.metrics_fwd.open_circuit_voltage_V - 0.035, abs=2e-3
    )

    # 4. Spectrum conflict guard fails closed
    bad_top = prepared_monolithic_tandem.top_cell.to_input().editing_data()
    bad_top["spectrum"] = "nonexistent_other_spectrum"
    bad_tandem_doc = prepared_monolithic_tandem.to_input().editing_data()
    bad_tandem_doc["top_cell"] = bad_top
    extra_tbl = ResourceTable(
        "nonexistent_other_spectrum",
        "spectrum",
        SourceDocument(
            "other.csv",
            b"wavelength_nm,spectral_flux_m-2_s-1_m-1\n400,1e27\n900,1e27\n",
        ),
    )
    merged_lib = ResourceLibrary(
        tables=(*prepared_monolithic_tandem.resources.tables, extra_tbl)
    )
    bad_prepared = resolve_tandem(
        TandemInput.model_validate(bad_tandem_doc),
        prepared_monolithic_tandem.defaults,
        merged_lib,
        sources=(*prepared_monolithic_tandem.sources, extra_tbl.source),
    )
    with pytest.raises(TandemCapabilityError, match="disagree"):
        compute_tandem_generation(bad_prepared, intervals_top=4, intervals_bottom=4)

"""Live numerical parity tests (T4) for coupled dynamic-defect + mobile-ion impedance (P05-03b).

Verifies end-to-end 1D small-signal AC parity between ``solve_defect_ion_impedance`` /
``run_defect_capacitance_frequency`` and ``perovskite_sim.experiments.defect_ion_combined_impedance``
across all three coupled defect-ion capabilities:
1. ``bulk_defect_plus_ions`` (including dual positive + negative mobile ions);
2. ``interface_defect_plus_ions`` (two-sided equilibrium-referenced interface defect + mobile ions);
3. ``bulk_interface_defect_plus_ions`` (simultaneous charged bulk defect + two-sided interface defect + ions).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.constants import K_B, Q
from perovskite_sim.experiments.defect_ion_combined_impedance import (
    run_defect_ion_combined_impedance,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_two_sided_trace_grid,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.models.mode import SimulationMode

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.capacitance_frequency import (
    capacitance_frequency_from_admittance,
    run_defect_capacitance_frequency,
)
from solarlab.experiments.defect_ion_impedance import (
    DefectIonImpedanceOptions,
    solve_defect_ion_impedance,
)
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument

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


@pytest.fixture(scope="module", autouse=True)
def _blas1() -> Iterator[None]:
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


def _load_paired(
    raw_doc: dict[str, Any],
    catalog: DefaultCatalog,
    tmp_path: Path,
    name: str,
) -> tuple[PreparedDevice, DeviceStack]:
    payload = json.dumps(raw_doc, allow_nan=False).encode()
    yaml_file = tmp_path / f"{name}.yaml"
    yaml_file.write_bytes(payload)
    source = SourceDocument(str(yaml_file), payload)
    dev_input = import_standard_device(source, id=name, defaults=catalog)
    prepared = resolve_device(dev_input, catalog, ResourceLibrary(), sources=(source,))
    stack = load_device_from_yaml(str(yaml_file))
    if raw_doc.get("device", {}).get("interface_charge_closure") != "equilibrium_referenced":
        stack = replace(stack, mode=SimulationMode(name="full", use_thermionic_emission=False))
    return prepared, stack


def _coupled_pin_yaml(
    *,
    with_bulk: bool,
    with_iface: bool,
    dual_ions: bool = False,
) -> dict[str, Any]:
    if not with_iface:
        ni_abs = _thermodynamic_ni(1.0e24, 8.0e23, 0.85)
        abs_only: dict[str, Any] = {
            "name": "absorber",
            "role": "absorber",
            "thickness": 3.0e-7,
            "eps_r": 20.0,
            "mu_n": 2.0e-3,
            "mu_p": 2.0e-3,
            "D_ion": 1.0e-14,
            "P0": 1.0e22,
            "P_lim": 2.0e24 if dual_ions else 1.0e24,
            "D_ion_neg": 6.0e-15 if dual_ions else 0.0,
            "P0_neg": 1.0e22 if dual_ions else 0.0,
            "P_lim_neg": 2.0e24 if dual_ions else 1.0e24,
            "ni": ni_abs,
            "tau_n": 1.0e-6,
            "tau_p": 1.0e-6,
            "n1": ni_abs,
            "p1": ni_abs,
            "B_rad": 0.0,
            "C_n": 0.0,
            "C_p": 0.0,
            "alpha": 0.0,
            "N_A": 0.0,
            "N_D": 0.0,
            "chi": 4.00,
            "Eg": 0.85,
            "Nc300": 1.0e24,
            "Nv300": 8.0e23,
            "defect_schema_version": "solarlab-explicit-bulk-defects-v1",
            "defect_model": "explicit_quasi_steady",
            "bulk_defects": [
                {
                    "name": "BulkAcceptor",
                    "charge_transition": "acceptor",
                    "neutral_reference": "empty",
                    "degeneracy": 1.0,
                    "distribution": {
                        "kind": "single_level",
                        "normalization": "integrated_total",
                        "total_density_m3": 2.0e21,
                        "center_eV_above_vb": 0.39,
                    },
                    "kinetics": {
                        "sigma_n_m2": 2.0e-19,
                        "sigma_p_m2": 8.0e-20,
                        "thermal_velocity_n_m_s": 1.0e5,
                        "thermal_velocity_p_m_s": 8.0e4,
                    },
                }
            ],
        }
        return {
            "schema_version": "standard-device-v1",
            "device": {
                "built_in_potential_mode": "semiconductor_work_function",
                "dos_band_potentials": True,
                "ion_steric_diffusion_only": False,
                "ion_steric_shared_site": False,
                "T": 300.0,
                "Phi": 0.0,
                "mode": "full",
            },
            "electrical_grid": {
                "interval_weights": {"absorber": 1.0},
                "alphas": {"absorber": 5.5},
            },
            "layers": [abs_only],
        }

    ni_layer = _thermodynamic_ni(1.0e25, 1.0e25, 1.05)
    left_layer: dict[str, Any] = {
        "name": "left",
        "role": "absorber",
        "thickness": 1.0e-7,
        "eps_r": 16.0,
        "mu_n": 2.0e-3,
        "mu_p": 2.0e-3,
        "D_ion": 1.0e-14,
        "P0": 1.0e22,
        "P_lim": 1.0e24,
        "ni": ni_layer,
        "tau_n": 1.0e-6,
        "tau_p": 1.0e-6,
        "n1": ni_layer,
        "p1": ni_layer,
        "B_rad": 0.0,
        "C_n": 0.0,
        "C_p": 0.0,
        "alpha": 0.0,
        "N_A": 0.0,
        "N_D": 0.0,
        "chi": 4.00,
        "Eg": 1.05,
        "Nc300": 1.0e25,
        "Nv300": 1.0e25,
    }
    if with_bulk:
        left_layer.update(
            {
                "defect_schema_version": "solarlab-explicit-bulk-defects-v1",
                "defect_model": "explicit_quasi_steady",
                "bulk_defects": [
                    {
                        "name": "BulkAcceptor",
                        "charge_transition": "acceptor",
                        "neutral_reference": "empty",
                        "degeneracy": 1.0,
                        "distribution": {
                            "kind": "single_level",
                            "normalization": "integrated_total",
                            "total_density_m3": 1.0e21,
                            "center_eV_above_vb": 0.50,
                        },
                        "kinetics": {
                            "sigma_n_m2": 2.0e-19,
                            "sigma_p_m2": 8.0e-20,
                            "thermal_velocity_n_m_s": 1.0e5,
                            "thermal_velocity_p_m_s": 8.0e4,
                        },
                    }
                ],
            }
        )

    return {
        "schema_version": "standard-device-v1",
        "device": {
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "ion_steric_diffusion_only": False,
            "ion_steric_shared_site": False,
            "interface_charge_closure": "equilibrium_referenced",
            "interface_charge_rebaseline_acknowledged": True,
            "interface_defects": [
                {
                    "sigma_n_cm2": 6.0e-17,
                    "sigma_p_cm2": 4.0e-17,
                    "v_th_cm_s": 1.0e7,
                    "N_t_cm2": 7.0e10,
                    "E_t_eV_below_cb": 0.50,
                }
            ],
            "T": 300.0,
            "Phi": 0.0,
            "mode": "full",
        },
        "electrical_grid": {
            "interval_weights": {"left": 6.0, "right": 4.0},
            "alphas": {"left": 5.0, "right": 5.0},
        },
        "layers": [
            left_layer,
            {
                "name": "right",
                "role": "ETL",
                "thickness": 1.0e-7,
                "eps_r": 16.0,
                "mu_n": 1.0e-3,
                "mu_p": 1.0e-3,
                "D_ion": 0.0,
                "P0": 0.0,
                "P_lim": 1.0e24,
                "ni": ni_layer,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_layer,
                "p1": ni_layer,
                "B_rad": 0.0,
                "C_n": 0.0,
                "C_p": 0.0,
                "alpha": 0.0,
                "N_A": 0.0,
                "N_D": 0.0,
                "chi": 4.08,
                "Eg": 1.05,
                "Nc300": 1.0e25,
                "Nv300": 1.0e25,
            },
        ],
    }


@pytest.mark.parametrize(
    ("expected_cap", "with_bulk", "with_iface", "dual_ions"),
    [
        ("bulk_defect_plus_ions", True, False, True),
        ("interface_defect_plus_ions", False, True, False),
        ("bulk_interface_defect_plus_ions", True, True, False),
    ],
)
def test_live_defect_ion_combined_impedance_parity(
    expected_cap: str,
    with_bulk: bool,
    with_iface: bool,
    dual_ions: bool,
    legacy_catalog: DefaultCatalog,
    tmp_path: Path,
) -> None:
    """T4 parity of coupled defect+ion impedance and C-f against perovskite_sim across all 3 capabilities."""
    doc = _coupled_pin_yaml(with_bulk=with_bulk, with_iface=with_iface, dual_ions=dual_ions)
    prep, stack = _load_paired(doc, legacy_catalog, tmp_path, f"parity_{expected_cap}")

    freqs = np.array([1.0e-2, 1.0e-1, 10.0, 1.0e3, 1.0e7])
    v_dc = 0.0 if with_iface else 0.06
    intervals: tuple[int, ...] = (6, 4) if with_iface else (8,)

    res_new = solve_defect_ion_impedance(
        prep,
        freqs,
        voltage_V=v_dc,
        options=DefectIonImpedanceOptions(intervals_per_layer=intervals),
    )
    assert res_new.capability == expected_cap
    assert res_new.scientific_qualification is False
    assert res_new.maximum_ion_inventory_relative_per_V <= 1.0e-12
    assert (
        res_new.working_state_digest
        == res_new.coupled.working_state_digest
        == res_new.frozen_ions.working_state_digest
        == res_new.frozen_traps.working_state_digest
        == res_new.frozen_ions_and_traps.working_state_digest
    )

    x_dedup = build_electrical_grid(stack, sum(intervals))
    x_grid = build_two_sided_trace_grid(x_dedup, stack) if with_iface else x_dedup
    np.testing.assert_allclose(res_new.device.nodes_m, x_grid, rtol=0.0, atol=1.0e-15)

    res_old = run_defect_ion_combined_impedance(
        x_grid,
        stack,
        freqs,
        V_dc=v_dc,
        refinement_factors=(1.0, 0.5, 0.25),
        require_certificate=False,
    )
    assert res_old.certificate.capability == expected_cap

    np.testing.assert_allclose(
        res_new.coupled.admittance_density_S_m2,
        res_old.admittance_S_m2,
        rtol=1.0e-4,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        res_new.coupled.impedance_area_ohm_m2,
        res_old.impedance_ohm_m2,
        rtol=1.0e-4,
        atol=1.0e-8,
    )

    cf_target = prep if not with_iface else res_new.device
    cf_new = run_defect_capacitance_frequency(
        cf_target,
        freqs,
        voltage_V=v_dc,
        intervals_per_layer=intervals,
    )
    cf_from_coupled = capacitance_frequency_from_admittance(
        freqs, res_new.coupled.admittance_density_S_m2, operating_voltage_V=v_dc
    )
    np.testing.assert_allclose(
        cf_new.parallel_capacitance_F_m2,
        cf_from_coupled.parallel_capacitance_F_m2,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    cf_old = capacitance_frequency_from_admittance(
        freqs, res_old.admittance_S_m2, operating_voltage_V=v_dc
    )
    # Legacy perovskite_sim uses O(1e-7) forward finite differences on Y = G + j*w*C,
    # inducing an O(1e-7 * |Y| / w) floor in Im(Y)/w when G >> w*C; at high frequency
    # (w*C >= G) direct C_p relative parity is strictly < 1e-4:
    np.testing.assert_allclose(
        cf_new.parallel_capacitance_F_m2[-1],
        cf_old.parallel_capacitance_F_m2[-1],
        rtol=1.0e-4,
        atol=1.0e-10,
    )
    valid_cap = ~np.isnan(cf_old.parallel_capacitance_F_m2)
    omega_valid = 2.0 * np.pi * freqs[valid_cap]
    admittance_cap_tol = 1.0e-4 * np.abs(res_old.admittance_S_m2[valid_cap]) / omega_valid
    assert np.all(
        np.abs(cf_new.parallel_capacitance_F_m2[valid_cap] - cf_old.parallel_capacitance_F_m2[valid_cap])
        <= np.maximum(1.0e-4 * np.abs(cf_old.parallel_capacitance_F_m2[valid_cap]), admittance_cap_tol)
        + 1.0e-10
    )

    # Local coarse-grid trap occupancy perturbations match to ~2e-3..3e-3 on N=8..10
    # (legacy ion_migration.py uses dx[0] / 0.5*(dx_l+dx_r) boundary ion cells rather
    # than exact half-cells, while terminal Y, Z, and C_p match within 1e-4 above):
    if with_bulk:
        occ_bt_new: NDArray[np.complex128] = np.asarray(
            [pt.bulk_trap_occupancy_perturbation_per_V for pt in res_new.coupled.points],
            dtype=np.complex128,
        )
        assert res_old.bulk_trap_occupancy_response_per_V is not None
        np.testing.assert_allclose(
            occ_bt_new,
            res_old.bulk_trap_occupancy_response_per_V,
            rtol=4.0e-3,
            atol=1.0e-6,
        )

    if with_iface:
        occ_it_new: NDArray[np.complex128] = np.asarray(
            [pt.interface_trap_occupancy_perturbation_per_V for pt in res_new.coupled.points],
            dtype=np.complex128,
        )
        assert res_old.interface_occupancy_response_per_V is not None
        np.testing.assert_allclose(
            occ_it_new,
            res_old.interface_occupancy_response_per_V,
            rtol=4.0e-3,
            atol=3.0e-5,
        )

    # Electrostatic non-additivity across coupled frequency band:
    y_coupled = res_new.coupled.admittance_density_S_m2
    y_superpos = (
        res_new.frozen_ions.admittance_density_S_m2
        + res_new.frozen_traps.admittance_density_S_m2
        - res_new.frozen_ions_and_traps.admittance_density_S_m2
    )
    assert float(np.max(np.abs(y_coupled - y_superpos) / np.abs(y_coupled))) > 1.0e-4

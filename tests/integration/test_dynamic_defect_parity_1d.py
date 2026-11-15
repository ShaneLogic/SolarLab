"""Live numerical parity tests (T4) for dynamic, distributed, multivalent, and metastable 1D defects (P04-10, P05-03b, P05-04b).

Verifies end-to-end 1D numerical parity against legacy ``perovskite_sim`` on identical grids across:
1. Bulk dynamic-defect impedance (single-level & v2 Gaussian continuous energy distribution) and C-f;
2. Two-sided dynamic interface-defect impedance with equilibrium-referenced sheet charge;
3. Narrow-gate v1 dynamic interface-defect + positive-ion transient and charged bulk-defect transient;
4. Stationary multivalent v4 (amphoteric + double-acceptor) illuminated J-V figures of merit;
5. Metastable two-configuration stationary preparation, frozen 1-sun J-V, and operator-split transient.
"""

from __future__ import annotations

from collections.abc import Iterator
import copy
from dataclasses import replace
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.constants import K_B, Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.bulk_defect_transient import (
    BulkDefectTransientPolicy,
    run_bulk_defect_device_transient,
)
from perovskite_sim.experiments.defect_aware_impedance import (
    run_bulk_defect_device_impedance,
)
from perovskite_sim.experiments.interface_defect_aware_impedance import (
    run_interface_defect_device_impedance,
)
from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientPolicy,
    run_interface_defect_ion_device_transient,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid, compute_metrics
from perovskite_sim.experiments.metastable_preparation import (
    prepare_metastable_configuration,
    solve_frozen_metastable_jv_sweep,
)
from perovskite_sim.experiments.metastable_transient import (
    run_metastable_configuration_transient,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_two_sided_trace_grid,
    solve_quasi_fermi_jv_sweep,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.defects import BulkDefectKinetics
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.mode import SimulationMode
from perovskite_sim.models.multivalent_defects import (
    MetastableConversionKinetics,
    MetastableDefectDefinition,
    MetastablePreparationNumerics,
    MetastablePreparationProtocol,
    MultivalentDefectConfiguration,
    MultivalentEnergyLevels,
)
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.metastable_defect_closure import (
    evaluate_metastable_configuration_closure,
)

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.compile_dynamic_defect_1d import compile_dynamic_defect_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.dynamic_defects_1d import (
    solve_dynamic_defect_steady_1d,
)
from solarlab.experiments.capacitance_frequency import (
    capacitance_frequency_from_admittance,
    run_defect_capacitance_frequency,
)
from solarlab.experiments.dynamic_defect_impedance import (
    DynamicDefectImpedanceOptions,
    solve_bulk_defect_impedance,
    solve_interface_defect_impedance,
)
from solarlab.experiments.dynamic_defect_transient import (
    DynamicDefectTransientOptions,
    solve_bulk_defect_transient,
    solve_dynamic_defect_transient,
)
from solarlab.experiments.metastable_transient import (
    prepare_metastable_state,
    solve_frozen_metastable_jv,
    solve_metastable_transient,
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

Vector: TypeAlias = NDArray[np.float64]


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


def _clean_json(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    return value


def _load_paired_models(
    raw_doc: dict[str, Any],
    catalog: DefaultCatalog,
    tmp_path: Path,
    name: str,
) -> tuple[PreparedDevice, DeviceStack]:
    payload = json.dumps(_clean_json(raw_doc), allow_nan=False).encode()
    yaml_file = tmp_path / f"{name}.yaml"
    yaml_file.write_bytes(payload)
    source = SourceDocument(str(yaml_file), payload)
    dev_input = import_standard_device(source, id=name, defaults=catalog)
    prepared = resolve_device(dev_input, catalog, ResourceLibrary(), sources=(source,))
    stack = load_device_from_yaml(str(yaml_file))
    if raw_doc.get("device", {}).get("interface_charge_closure") != "equilibrium_referenced":
        stack = replace(stack, mode=SimulationMode(name="full", use_thermionic_emission=False))
    return prepared, stack


def _three_layer_pin_yaml(*, phi_flux: float = 0.0) -> dict[str, Any]:
    ni_htl = _thermodynamic_ni(2.0e24, 2.0e24, 1.50)
    ni_abs = _thermodynamic_ni(1.0e24, 8.0e23, 1.20)
    ni_etl = _thermodynamic_ni(2.0e24, 2.0e24, 1.50)
    return {
        "schema_version": "standard-device-v1",
        "device": {
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "ion_steric_diffusion_only": False,
            "ion_steric_shared_site": False,
            "T": 300.0,
            "Phi": phi_flux,
            "mode": "full",
        },
        "electrical_grid": {
            "interval_weights": {"htl": 1.0, "absorber": 1.0, "etl": 1.0},
            "alphas": {"htl": 2.0, "absorber": 2.0, "etl": 2.0},
        },
        "layers": [
            {
                "name": "htl",
                "role": "HTL",
                "thickness": 6.0e-8,
                "eps_r": 12.0,
                "mu_n": 1.0e-3,
                "mu_p": 1.0e-3,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_htl,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_htl,
                "p1": ni_htl,
                "B_rad": 0.0,
                "C_n": 0.0,
                "C_p": 0.0,
                "alpha": 0.0,
                "N_A": 2.0e22,
                "N_D": 0.0,
                "chi": 3.85,
                "Eg": 1.50,
                "Nc300": 2.0e24,
                "Nv300": 2.0e24,
            },
            {
                "name": "absorber",
                "role": "absorber",
                "thickness": 2.4e-7,
                "eps_r": 20.0,
                "mu_n": 2.0e-3,
                "mu_p": 2.0e-3,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_abs,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_abs,
                "p1": ni_abs,
                "B_rad": 0.0,
                "C_n": 0.0,
                "C_p": 0.0,
                "alpha": 3.5e5,
                "N_A": 0.0,
                "N_D": 0.0,
                "chi": 3.95,
                "Eg": 1.20,
                "Nc300": 1.0e24,
                "Nv300": 8.0e23,
            },
            {
                "name": "etl",
                "role": "ETL",
                "thickness": 6.0e-8,
                "eps_r": 12.0,
                "mu_n": 1.0e-3,
                "mu_p": 1.0e-3,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_etl,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_etl,
                "p1": ni_etl,
                "B_rad": 0.0,
                "C_n": 0.0,
                "C_p": 0.0,
                "alpha": 0.0,
                "N_A": 0.0,
                "N_D": 2.0e22,
                "chi": 4.05,
                "Eg": 1.50,
                "Nc300": 2.0e24,
                "Nv300": 2.0e24,
            },
        ],
    }


def test_live_bulk_and_interface_dynamic_defect_impedance_parity(
    legacy_catalog: DefaultCatalog,
    tmp_path: Path,
) -> None:
    """T4 parity on single-level & Gaussian bulk defect AC, C-f, and two-sided interface AC."""
    freqs = np.array([1.0e-3, 1.0, 1.0e3, 1.0e7])

    # 1(a). Distributed v2 Gaussian charged bulk acceptor impedance + C-f parity:
    doc_gauss = _three_layer_pin_yaml()
    doc_gauss["layers"][1].update(
        {
            "defect_schema_version": "solarlab-explicit-bulk-defects-v2",
            "defect_model": "explicit_quasi_steady",
            "bulk_defects": [
                {
                    "name": "GaussianAcceptor",
                    "charge_transition": "acceptor",
                    "neutral_reference": "empty",
                    "degeneracy": 1.0,
                    "distribution": {
                        "kind": "gaussian",
                        "normalization": "integrated_total",
                        "total_density_m3": 3.0e21,
                        "center_eV_above_vb": 0.58,
                        "energy_reference": "above_valence_band",
                        "width_eV": 0.05,
                        "width_convention": "gaussian_standard_deviation",
                        "support_width_multiplier": 6.0,
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
    prep_g, stack_g = _load_paired_models(doc_gauss, legacy_catalog, tmp_path, "bulk_gauss_ac")
    res_new_g = solve_bulk_defect_impedance(
        prep_g,
        freqs,
        voltage_V=0.25,
        options=DynamicDefectImpedanceOptions(intervals_per_layer=(3, 3, 3), energy_quadrature_order=8),
    )
    x_g = build_electrical_grid(stack_g, 9)
    np.testing.assert_allclose(res_new_g.device.nodes_m, x_g, rtol=0.0, atol=1.0e-15)

    res_old_g = run_bulk_defect_device_impedance(
        x_g,
        stack_g,
        freqs,
        V_dc=0.25,
        defect_energy_quadrature_order=8,
        refinement_factors=(1.0, 0.5),
        require_certificate=False,
    )
    np.testing.assert_allclose(
        res_new_g.admittance_density_S_m2, res_old_g.admittance_S_m2, rtol=1.0e-4, atol=1.0e-8
    )
    np.testing.assert_allclose(
        res_new_g.impedance_area_ohm_m2, res_old_g.impedance_ohm_m2, rtol=1.0e-4, atol=1.0e-8
    )
    cf_new = run_defect_capacitance_frequency(
        prep_g,
        freqs,
        voltage_V=0.25,
        intervals_per_layer=(3, 3, 3),
        energy_quadrature_order=8,
    )
    cf_old = capacitance_frequency_from_admittance(freqs, res_old_g.admittance_S_m2, operating_voltage_V=0.25)
    np.testing.assert_allclose(
        cf_new.parallel_capacitance_F_m2, cf_old.parallel_capacitance_F_m2, rtol=1.0e-4, atol=1.0e-10
    )

    # 1(b). Two-sided dynamic interface defect impedance parity:
    ni_if = _thermodynamic_ni(1.0e25, 1.0e25, 1.35)
    doc_if: dict[str, Any] = {
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
                    "sigma_n_cm2": 8.0e-17,
                    "sigma_p_cm2": 5.0e-17,
                    "v_th_cm_s": 1.0e7,
                    "N_t_cm2": 8.0e10,
                    "E_t_eV_below_cb": 0.62,
                },
                {
                    "sigma_n_cm2": 6.0e-17,
                    "sigma_p_cm2": 4.0e-17,
                    "v_th_cm_s": 1.0e7,
                    "N_t_cm2": 6.0e10,
                    "E_t_eV_below_cb": 0.65,
                },
            ],
            "T": 300.0,
            "Phi": 0.0,
            "mode": "full",
        },
        "electrical_grid": {
            "interval_weights": {"htl": 1.0, "absorber": 1.0, "etl": 1.0},
            "alphas": {"htl": 2.0, "absorber": 2.0, "etl": 2.0},
        },
        "layers": [
            {
                "name": lname,
                "role": lrole,
                "thickness": 1.0e-7,
                "eps_r": 16.0,
                "mu_n": 1.0e-3,
                "mu_p": 1.0e-3,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
                "P0": 0.0,
                "ni": ni_if,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_if,
                "p1": ni_if,
                "B_rad": 0.0,
                "C_n": 0.0,
                "C_p": 0.0,
                "alpha": 0.0,
                "N_A": 0.0,
                "N_D": 0.0,
                "chi": lchi,
                "Eg": 1.35,
                "Nc300": 1.0e25,
                "Nv300": 1.0e25,
            }
            for lname, lrole, lchi in (
                ("htl", "HTL", 4.00),
                ("absorber", "absorber", 4.05),
                ("etl", "ETL", 4.10),
            )
        ],
    }
    prep_if, stack_if = _load_paired_models(doc_if, legacy_catalog, tmp_path, "iface_dyn_ac")
    res_new_if = solve_interface_defect_impedance(
        prep_if,
        freqs,
        voltage_V=0.10,
        options=DynamicDefectImpedanceOptions(intervals_per_layer=(3, 3, 3)),
    )
    x_if = build_two_sided_trace_grid(build_electrical_grid(stack_if, 9), stack_if)
    np.testing.assert_allclose(res_new_if.device.nodes_m, x_if, rtol=0.0, atol=1.0e-15)

    res_old_if = run_interface_defect_device_impedance(
        x_if,
        stack_if,
        freqs,
        V_dc=0.10,
        refinement_factors=(1.0, 0.5),
        require_certificate=False,
    )
    np.testing.assert_allclose(
        res_new_if.admittance_density_S_m2, res_old_if.admittance_S_m2, rtol=1.0e-4, atol=1.0e-8
    )
    np.testing.assert_allclose(
        res_new_if.impedance_area_ohm_m2, res_old_if.impedance_ohm_m2, rtol=1.0e-4, atol=1.0e-8
    )
    occ_resp_new: NDArray[np.complex128] = np.asarray(
        [pt.interface_trap_occupancy_perturbation_per_V for pt in res_new_if.ac_sweep.points],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(
        occ_resp_new, res_old_if.interface_occupancy_response_per_V, rtol=1.0e-4, atol=1.0e-7
    )


def test_live_dynamic_defect_and_bulk_defect_transient_parity(
    legacy_catalog: DefaultCatalog,
    tmp_path: Path,
) -> None:
    """T4 parity on narrow-gate v1 interface-defect+ion transient and D6-E1 bulk-defect transient."""
    # 2(a). Narrow-gate v1 dark interface-defect + absorber positive-ion transient:
    fixture_yaml = (
        ROOT / "perovskite-sim/tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    )
    raw_v1 = dict(load_yaml_mapping(fixture_yaml.read_bytes()))
    raw_v1["device"] = {
        **dict(raw_v1["device"]),
        "ion_steric_diffusion_only": False,
        "ion_steric_shared_site": False,
    }
    prep_v1, stack_v1 = _load_paired_models(raw_v1, legacy_catalog, tmp_path, "v1_dyn_tr")
    times_v1 = np.array([0.0, 1.0e-7, 2.0e-6, 5.0e-5])
    volts_v1 = np.array([0.0, 0.08, 0.08, 0.0])

    tr_new_v1 = solve_dynamic_defect_transient(
        prep_v1,
        times_v1,
        volts_v1,
        options=DynamicDefectTransientOptions(
            intervals_per_layer=(4, 4), substeps_per_interval=4, bdf_order=1
        ),
    )
    x_v1 = build_two_sided_trace_grid(build_electrical_grid(stack_v1, 8), stack_v1)
    np.testing.assert_allclose(tr_new_v1.grid_m, x_v1, rtol=0.0, atol=1.0e-15)

    tr_old_v1 = run_interface_defect_ion_device_transient(
        x_v1,
        stack_v1,
        times_v1,
        volts_v1,
        policy=InterfaceDefectIonTransientPolicy(refinement_substeps=(2, 4)),
        require_certificate=False,
    )
    np.testing.assert_allclose(
        tr_new_v1.terminal_current_density_A_m2,
        np.mean(tr_old_v1.interface_total_current_A_m2[:, 0, :], axis=-1),
        rtol=5.0e-4,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        tr_new_v1.interface_sheet_charges_C_m2,
        tr_old_v1.interface_sheet_charge_C_m2,
        rtol=5.0e-4,
        atol=1.0e-10,
    )
    abs_nodes = tr_new_v1.device.ionic_component.ion_node_indices  # type: ignore[union-attr]
    np.testing.assert_allclose(
        tr_new_v1.positive_ion_density_m3[:, abs_nodes],
        tr_old_v1.positive_ion_density_m3[:, abs_nodes],
        rtol=1.0e-4,
        atol=1.0e14,
    )

    # 2(b). Charged single-level bulk-defect transient (D6-E1):
    ni_bt = _thermodynamic_ni(1.0e24, 8.0e23, 0.80)
    doc_bt: dict[str, Any] = {
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
            "alphas": {"absorber": 2.5},
        },
        "layers": [
            {
                "name": "absorber",
                "role": "absorber",
                "thickness": 3.0e-7,
                "eps_r": 20.0,
                "mu_n": 2.0e-3,
                "mu_p": 2.0e-3,
                "D_ion": 0.0,
                "P0": 0.0,
                "P_lim": 1.0e30,
                "ni": ni_bt,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "n1": ni_bt,
                "p1": ni_bt,
                "B_rad": 0.0,
                "C_n": 0.0,
                "C_p": 0.0,
                "alpha": 0.0,
                "N_A": 0.0,
                "N_D": 0.0,
                "chi": 4.0,
                "Eg": 0.80,
                "Nc300": 1.0e24,
                "Nv300": 8.0e23,
                "defect_schema_version": "solarlab-explicit-bulk-defects-v1",
                "defect_model": "explicit_quasi_steady",
                "bulk_defects": [
                    {
                        "name": "SingleLevelAcceptor",
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
                            "sigma_p_m2": 7.0e-20,
                            "thermal_velocity_n_m_s": 1.0e5,
                            "thermal_velocity_p_m_s": 8.0e4,
                        },
                    }
                ],
            }
        ],
    }
    prep_bt, stack_bt = _load_paired_models(doc_bt, legacy_catalog, tmp_path, "bulk_dyn_tr")
    times_bt = np.array([0.0, 1.0e-8, 1.0e-6, 1.0e-4])
    volts_bt = np.array([0.0, 0.02, 0.02, 0.0])

    tr_new_bt = solve_bulk_defect_transient(
        prep_bt,
        times_bt,
        volts_bt,
        options=DynamicDefectTransientOptions(
            intervals_per_layer=(5,), substeps_per_interval=4, bdf_order=1
        ),
    )
    x_bt = build_electrical_grid(stack_bt, 5)
    tr_old_bt = run_bulk_defect_device_transient(
        x_bt,
        stack_bt,
        times_bt,
        volts_bt,
        policy=BulkDefectTransientPolicy(refinement_substeps=(2, 4)),
        require_certificate=False,
    )
    np.testing.assert_allclose(
        tr_new_bt.bulk_trap_occupancy, tr_old_bt.trap_occupancy, rtol=5.0e-4, atol=1.0e-6
    )
    np.testing.assert_allclose(
        tr_new_bt.potential_V, tr_old_bt.electrostatic_potential_V, rtol=5.0e-4, atol=1.0e-5
    )
    j_old_bt = np.median(tr_old_bt.total_current_faces_A_m2, axis=1)
    np.testing.assert_allclose(
        tr_new_bt.terminal_current_density_A_m2, j_old_bt, rtol=5.0e-4, atol=1.0e-5
    )


def test_live_multivalent_v4_illuminated_jv_parity(
    legacy_catalog: DefaultCatalog,
    tmp_path: Path,
) -> None:
    """T4 parity on illuminated 1-sun J-V for multivalent v4 amphoteric + double-acceptor defects."""
    doc_mv = _three_layer_pin_yaml(phi_flux=2.5e21)
    # Moderate absorption coefficient so Beer-Lambert midpoint vs exact CV quadrature matches < 5e-6 on N=14:
    doc_mv["layers"][1]["alpha"] = 1.5e5
    doc_mv["electrical_grid"]["interval_weights"] = {"htl": 4.0, "absorber": 6.0, "etl": 4.0}
    doc_mv["layers"][1].update(
        {
            "defect_schema_version": "solarlab-explicit-bulk-defects-v4",
            "defect_model": "explicit_quasi_steady",
            "bulk_defects": [
                {
                    "name": "AmphotericCenter",
                    "total_density_m3": 1.5e21,
                    "configuration": {
                        "family": "amphoteric",
                        "charge_states_e": [1, 0, -1],
                        "degeneracy_convention": "scaps_binomial",
                        "state_degeneracies": [1.0, 2.0, 1.0],
                        "energy_levels": {
                            "first_transition_eV_above_vb": 0.42,
                            "correlation_energies_eV": [0.28],
                            "energy_reference": "above_valence_band",
                        },
                        "transition_kinetics": [
                            {
                                "sigma_n_m2": 2.0e-19,
                                "sigma_p_m2": 1.0e-19,
                                "thermal_velocity_n_m_s": 1.0e5,
                                "thermal_velocity_p_m_s": 1.0e5,
                            },
                            {
                                "sigma_n_m2": 1.0e-19,
                                "sigma_p_m2": 2.0e-19,
                                "thermal_velocity_n_m_s": 1.0e5,
                                "thermal_velocity_p_m_s": 1.0e5,
                            },
                        ],
                    },
                },
                {
                    "name": "DoubleAcceptor",
                    "total_density_m3": 1.0e21,
                    "configuration": {
                        "family": "double_acceptor",
                        "charge_states_e": [0, -1, -2],
                        "degeneracy_convention": "unity",
                        "state_degeneracies": [1.0, 1.0, 1.0],
                        "energy_levels": {
                            "first_transition_eV_above_vb": 0.48,
                            "correlation_energies_eV": [0.24],
                            "energy_reference": "above_valence_band",
                        },
                        "transition_kinetics": [
                            {
                                "sigma_n_m2": 1.5e-19,
                                "sigma_p_m2": 1.5e-19,
                                "thermal_velocity_n_m_s": 1.0e5,
                                "thermal_velocity_p_m_s": 1.0e5,
                            },
                            {
                                "sigma_n_m2": 8.0e-20,
                                "sigma_p_m2": 1.5e-19,
                                "thermal_velocity_n_m_s": 1.0e5,
                                "thermal_velocity_p_m_s": 1.0e5,
                            },
                        ],
                    },
                },
            ],
        }
    )
    prep_mv, stack_mv = _load_paired_models(doc_mv, legacy_catalog, tmp_path, "multivalent_jv")
    dev_mv = compile_dynamic_defect_device_1d(
        prep_mv, intervals_per_layer=(4, 6, 4), dynamic_bulk_defects=False
    )
    x_mv = build_electrical_grid(stack_mv, 14)
    np.testing.assert_allclose(dev_mv.nodes_m, x_mv, rtol=0.0, atol=1.0e-15)

    voltages = np.linspace(0.0, 0.85, 6)
    sweep_old = solve_quasi_fermi_jv_sweep(
        x_mv,
        stack_mv,
        voltages,
        illumination_steps=(0.0, 1.0e-4, 1.0e-2, 1.0),
        stop_after_voc=True,
        minimum_voltage_step_V=0.01,
        continuity_tolerance_A_m2=5.0e-3,
        current_spread_tolerance_A_m2=5.0e-3,
    )
    v_used = np.asarray(sweep_old.voltages_V, dtype=np.float64)

    j_solar_new = np.empty_like(v_used)
    prev_u: Vector | None = None
    prev_v = 0.0
    for idx, vf in enumerate(v_used):
        seed = None if prev_u is None else dev_mv.rebase_coordinates(prev_u, prev_v, float(vf))
        st = solve_dynamic_defect_steady_1d(
            dev_mv,
            voltage_V=float(vf),
            generation_scale=1.0,
            initial_u=seed,
            voltage_step_max_V=0.05,
        )
        j_solar_new[idx] = -st.terminal_current_density_A_m2
        prev_u = st.state.u
        prev_v = float(vf)

    m_new = compute_metrics(v_used, j_solar_new, P_in=1000.0)
    m_old = sweep_old.metrics
    assert abs(m_new.V_oc - m_old.V_oc) <= 1.0e-4
    assert abs(m_new.J_sc - m_old.J_sc) / abs(m_old.J_sc) <= 1.0e-4
    assert abs(m_new.FF - m_old.FF) / abs(m_old.FF) <= 1.0e-4
    assert abs(m_new.PCE - m_old.PCE) / abs(m_old.PCE) <= 1.0e-4
    np.testing.assert_allclose(j_solar_new, sweep_old.currents_A_m2, rtol=1.0e-4, atol=1.0e-4)


def test_live_metastable_preparation_frozen_jv_and_transient_parity(
    legacy_catalog: DefaultCatalog,
) -> None:
    """T4 parity on metastable preparation (330 K -> 300 K), frozen 1-sun J-V, and operator-split transient."""
    gap_eV = 0.80
    nc_m3 = 1.0e24
    nv_m3 = 8.0e23
    ni_val = _thermodynamic_ni(nc_m3, nv_m3, gap_eV, 300.0)
    trans_eV = 0.35
    ec_act = 0.20
    hc_act = 0.25
    phi_flux = 2.0e21
    alpha_val = 1.0e5

    kin_dict = {
        "sigma_n_m2": 2.0e-19,
        "sigma_p_m2": 7.0e-20,
        "thermal_velocity_n_m_s": 1.0e5,
        "thermal_velocity_p_m_s": 8.0e4,
    }
    prep_dict = {
        "schema_version": "solarlab-metastable-preparation-v1",
        "preparation_limit": "stationary_infinite_time",
        "preparation_temperature_K": 300.0,
        "preparation_voltage_V": 0.04,
        "preparation_illumination_suns": 0.0,
        "voltage_continuation_steps": 0,
        "illumination_continuation_steps": 0,
        "measurement_temperature_K": 300.0,
        "configuration_freeze_stage": "after_stationary_preparation_before_measurement",
        "freeze_configuration_during_measurement": True,
        "measurement_protocol_sha256": "a" * 64,
        "numerics": {
            "initial_donor_fraction_guess": 0.5,
            "max_iterations": 80,
            "relative_tolerance": 1.0e-10,
            "clamping_factor": 0.5,
            "final_unclamped_refinement": True,
        },
    }
    canon_doc = {
        "schema_version": "solarlab.device-preparation.v1",
        "source_format": "canonical",
        "id": "meta_parity",
        "settings": {
            "T": 300.0,
            "Phi": phi_flux,
            "mode": "legacy",
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "ion_steric_diffusion_only": False,
            "ion_steric_shared_site": False,
        },
        "layers": [
            {
                "id": "meta_absorber",
                "name": "meta_absorber",
                "role": "absorber",
                "thickness": 300.0e-9,
                "parameters": {
                    "eps_r": 20.0,
                    "mu_n": 2.0e-3,
                    "mu_p": 2.0e-3,
                    "D_ion": 0.0,
                    "P_lim": 1.0e30,
                    "P0": 0.0,
                    "ni": ni_val,
                    "tau_n": 1.0e-6,
                    "tau_p": 1.0e-6,
                    "n1": ni_val,
                    "p1": ni_val,
                    "B_rad": 0.0,
                    "C_n": 0.0,
                    "C_p": 0.0,
                    "alpha": alpha_val,
                    "N_A": 0.0,
                    "N_D": 0.0,
                    "chi": 4.0,
                    "Eg": gap_eV,
                    "Nc300": nc_m3,
                    "Nv300": nv_m3,
                },
                "metastable_document": {
                    "schema_version": "solarlab-metastable-bulk-defects-v1",
                    "defect_model": "explicit_metastable_frozen",
                    "metastable_defects": [
                        {
                            "id": "meta_center",
                            "name": "metastable_center",
                            "total_density_m3": 2.0e21,
                            "donor_configuration": {
                                "family": "double_donor",
                                "charge_states_e": [2, 1, 0],
                                "degeneracy_convention": "unity",
                                "state_degeneracies": [1.0, 1.0, 1.0],
                                "energy_levels": {
                                    "first_transition_eV_above_vb": 0.30,
                                    "correlation_energies_eV": [0.15],
                                    "energy_reference": "above_valence_band",
                                },
                                "transition_kinetics": [copy.deepcopy(kin_dict), copy.deepcopy(kin_dict)],
                            },
                            "acceptor_configuration": {
                                "family": "double_acceptor",
                                "charge_states_e": [0, -1, -2],
                                "degeneracy_convention": "unity",
                                "state_degeneracies": [1.0, 1.0, 1.0],
                                "energy_levels": {
                                    "first_transition_eV_above_vb": 0.30,
                                    "correlation_energies_eV": [0.15],
                                    "energy_reference": "above_valence_band",
                                },
                                "transition_kinetics": [copy.deepcopy(kin_dict), copy.deepcopy(kin_dict)],
                            },
                            "donor_conversion_state_index": 1,
                            "acceptor_conversion_state_index": 1,
                            "conversion_kinetics": {
                                "transition_energy_eV_above_vb": trans_eV,
                                "electron_capture_activation_eV": ec_act,
                                "electron_emission_activation_eV": ec_act + 2.0 * (gap_eV - trans_eV),
                                "hole_capture_activation_eV": hc_act,
                                "hole_emission_activation_eV": hc_act + 2.0 * trans_eV,
                                "electron_capture_path": "double_electron_capture",
                                "hole_capture_path": "double_hole_capture",
                                "capture_n_m3_s": 1.0e-15,
                                "capture_p_m3_s": 1.0e-15,
                                "phonon_frequency_Hz": 1.0e15,
                            },
                        }
                    ],
                },
                "metastable_preparation": prep_dict,
            }
        ],
    }

    prepared_new = resolve_device(
        DeviceInput.model_validate(canon_doc), legacy_catalog, ResourceLibrary()
    )
    state_new = prepare_metastable_state(prepared_new, intervals_per_layer=4)

    # Identical legacy DeviceStack + MetastableDefectDefinition + MetastablePreparationProtocol:
    grid_old = multilayer_grid([Layer(300.0e-9, 4)])
    np.testing.assert_allclose(
        state_new.grid_m, grid_old, rtol=0.0, atol=1.0e-15
    )

    leg_kin = BulkDefectKinetics(**kin_dict)
    leg_def = MetastableDefectDefinition(
        name="metastable_center",
        total_density_m3=2.0e21,
        donor_configuration=MultivalentDefectConfiguration(
            family="double_donor",
            charge_states_e=(2, 1, 0),
            degeneracy_convention="unity",
            state_degeneracies=(1.0, 1.0, 1.0),
            energy_levels=MultivalentEnergyLevels(
                first_transition_eV_above_vb=0.30, correlation_energies_eV=(0.15,)
            ),
            transition_kinetics=(leg_kin, leg_kin),
        ),
        acceptor_configuration=MultivalentDefectConfiguration(
            family="double_acceptor",
            charge_states_e=(0, -1, -2),
            degeneracy_convention="unity",
            state_degeneracies=(1.0, 1.0, 1.0),
            energy_levels=MultivalentEnergyLevels(
                first_transition_eV_above_vb=0.30, correlation_energies_eV=(0.15,)
            ),
            transition_kinetics=(leg_kin, leg_kin),
        ),
        donor_conversion_state_index=1,
        acceptor_conversion_state_index=1,
        conversion_kinetics=MetastableConversionKinetics(
            transition_energy_eV_above_vb=trans_eV,
            electron_capture_activation_eV=ec_act,
            electron_emission_activation_eV=ec_act + 2.0 * (gap_eV - trans_eV),
            hole_capture_activation_eV=hc_act,
            hole_emission_activation_eV=hc_act + 2.0 * trans_eV,
            electron_capture_path="double_electron_capture",
            hole_capture_path="double_hole_capture",
            capture_n_m3_s=1.0e-15,
            capture_p_m3_s=1.0e-15,
            phonon_frequency_Hz=1.0e15,
        ),
    )
    leg_proto = MetastablePreparationProtocol(
        schema_version="solarlab-metastable-preparation-v1",
        preparation_limit="stationary_infinite_time",
        preparation_temperature_K=300.0,
        preparation_voltage_V=0.04,
        preparation_illumination_suns=0.0,
        voltage_continuation_steps=0,
        illumination_continuation_steps=0,
        measurement_temperature_K=300.0,
        configuration_freeze_stage="after_stationary_preparation_before_measurement",
        freeze_configuration_during_measurement=True,
        measurement_protocol_sha256="a" * 64,
        numerics=MetastablePreparationNumerics(
            initial_donor_fraction_guess=0.5,
            max_iterations=80,
            relative_tolerance=1.0e-10,
            clamping_factor=0.5,
            final_unclamped_refinement=True,
        ),
    )
    leg_stack = DeviceStack(
        layers=(
            LayerSpec(
                "meta_absorber",
                300.0e-9,
                MaterialParams(
                    eps_r=20.0,
                    mu_n=2.0e-3,
                    mu_p=2.0e-3,
                    D_ion=0.0,
                    P_lim=1.0e30,
                    P0=0.0,
                    ni=ni_val,
                    tau_n=1.0e-6,
                    tau_p=1.0e-6,
                    n1=ni_val,
                    p1=ni_val,
                    B_rad=0.0,
                    C_n=0.0,
                    C_p=0.0,
                    alpha=alpha_val,
                    N_A=0.0,
                    N_D=0.0,
                    chi=4.0,
                    Eg=gap_eV,
                    Nc300=nc_m3,
                    Nv300=nv_m3,
                ),
                "absorber",
            ),
        ),
        V_bi=0.0,
        Phi=phi_flux,
        interfaces=(),
        mode="legacy",
        built_in_potential_mode="semiconductor_work_function",
    )
    frozen_old = prepare_metastable_configuration(
        grid_old, leg_stack, leg_def, leg_proto, layer_name="meta_absorber"
    )
    np.testing.assert_allclose(
        state_new.donor_fractions[0], frozen_old.donor_fraction, rtol=1.0e-7, atol=1.0e-9
    )

    # Frozen 1-sun J-V parity:
    v_jv = np.array([0.0, 0.01])
    jv_new = solve_frozen_metastable_jv(state_new, v_jv, generation_scale=1.0)
    jv_old = solve_frozen_metastable_jv_sweep(
        grid_old,
        leg_stack,
        frozen_old,
        v_jv,
        illumination_steps=(0.0, 1.0e-2, 1.0),
        continuity_tolerance_A_m2=2.0e-4,
        current_spread_tolerance_A_m2=2.0e-4,
    )
    np.testing.assert_allclose(
        -jv_new.current_densities_A_m2, jv_old.currents_A_m2, rtol=1.0e-4, atol=1.0e-5
    )

    # Operator-split dark relaxation transient parity at T = 300 K, V = 0.0 V:
    mask_old = frozen_old.active_nodes
    closure_300 = evaluate_metastable_configuration_closure(
        np.asarray(frozen_old.preparation_state.y[: grid_old.size], dtype=float)[mask_old],
        np.asarray(frozen_old.preparation_state.y[grid_old.size : 2 * grid_old.size], dtype=float)[mask_old],
        leg_def,
        band_gap_eV=gap_eV,
        effective_conduction_dos_m3=nc_m3,
        effective_valence_dos_m3=nv_m3,
        temperature_K=300.0,
    )
    tau_min = float(
        np.min(1.0 / (np.asarray(closure_300.donor_to_acceptor_rate_s1) + np.asarray(closure_300.acceptor_to_donor_rate_s1)))
    )
    t_meta = np.arange(3, dtype=np.float64) * (0.05 * tau_min)
    tr_new = solve_metastable_transient(state_new, t_meta, voltage_V=0.0, generation_scale=0.0)
    tr_old = run_metastable_configuration_transient(
        grid_old, leg_stack, frozen_old, t_meta, V_app=0.0, illuminated=False
    )
    assert tr_new.certificate.passed is True
    assert tr_old.certificate.certified is True
    np.testing.assert_allclose(tr_new.donor_fractions[0], tr_old.donor_fraction, rtol=1.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(
        tr_new.stationary_fractions[0], tr_old.stationary_fraction, rtol=1.0e-6, atol=1.0e-8
    )
    np.testing.assert_allclose(
        tr_new.relaxation_times_s[0], tr_old.relaxation_time_s, rtol=1.0e-6, atol=1.0e-12
    )
    np.testing.assert_allclose(
        tr_new.configuration_charge_transfer_C_m2,
        tr_old.configuration_charge_transfer_C_m2,
        rtol=1.0e-5,
        atol=1.0e-12,
    )

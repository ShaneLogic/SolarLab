"""Unified assembly, steady QSS identity, Jacobian/storage tangents, and coupled AC tests (T2, P04-10)."""

from __future__ import annotations

from collections.abc import Iterator
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_dynamic_defect_1d import compile_dynamic_defect_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.dynamic_defects_1d import (
    AssembledDynamicDefect1DProblem,
    DynamicDefectSmallSignal1D,
    DynamicDefectStorage1D,
    dynamic_defect_system_1d,
    solve_dynamic_defect_steady_1d,
)
from solarlab.equations.dc_1d import CapabilityError
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[3]
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
def defaults() -> DefaultCatalog:
    return read_legacy_default_catalog(
        tuple(SourceDocument(n, (LEGACY_PKG / n).read_bytes()) for n in LEGACY_DEFAULTS)
    )


def _mat(eg: float = 1.6, chi: float = 3.9, *, nd: float = 0.0, na: float = 0.0) -> dict[str, Any]:
    vt = 1.380649e-23 * 300.0 / 1.602176634e-19
    nc = 2.0e24
    nv = 2.0e24
    ni = float(np.sqrt(nc * nv * np.exp(-eg / vt)))
    return {
        "Eg": eg,
        "chi": chi,
        "Nc300": nc,
        "Nv300": nv,
        "ni": ni,
        "n1": ni,
        "p1": ni,
        "eps_r": 24.0,
        "mu_n": 2.0e-4,
        "mu_p": 2.0e-4,
        "tau_n": 1.0e-6,
        "tau_p": 1.0e-6,
        "B_rad": 1.0e-16,
        "C_n": 0.0,
        "C_p": 0.0,
        "alpha": 1.0e7,
        "A_star_n": 1.2e5,
        "A_star_p": 1.2e5,
        "N_D": nd,
        "N_A": na,
        "P0": 0.0,
        "P_lim": 1.0e27,
        "D_ion": 0.0,
    }


def _settings(**extra: Any) -> dict[str, Any]:
    return {
        "T": 300.0,
        "Phi": 0.0,
        "built_in_potential_mode": "semiconductor_work_function",
        "dos_band_potentials": True,
        "ion_steric_diffusion_only": False,
        "ion_steric_shared_site": False,
        **extra,
    }


def _bulk_defect_entry(
    defect_id: str = "t1",
    *,
    density: float = 3.0e21,
    kind: str = "single_level",
) -> dict[str, Any]:
    dist: dict[str, Any] = {
        "kind": kind,
        "normalization": "integrated_total",
        "total_density_m3": density,
    }
    if kind == "gaussian":
        dist["energy_reference"] = "above_valence_band"
        dist["width_eV"] = 0.05
        dist["width_convention"] = "gaussian_standard_deviation"
        dist["support_width_multiplier"] = 6.0
    return {
        "id": defect_id,
        "name": defect_id,
        "charge_transition": "acceptor",
        "neutral_reference": "empty",
        "degeneracy": 1.0,
        "energy_level": {"reference": "above_valence_band", "value_eV": 0.72},
        "distribution": dist,
        "kinetics": {
            "sigma_n_m2": 1.5e-19,
            "sigma_p_m2": 1.0e-19,
            "thermal_velocity_n_m_s": 1.0e5,
            "thermal_velocity_p_m_s": 1.0e5,
        },
    }


def _iface_defect_entry(
    defect_id: str = "id1",
    *,
    density_m2: float = 1.5e15,
    calibration_factor: float = 1.0,
) -> dict[str, Any]:
    return {
        "id": defect_id,
        "total_density_m2": density_m2,
        "trap_depth_eV": 0.62,
        "energy_reference": "below_conduction_band",
        "calibration_factor": calibration_factor,
        "iface_state_calibration_factor": 1.0,
        "kinetics": {
            "sigma_n_m2": 1.2e-19,
            "sigma_p_m2": 1.0e-19,
            "thermal_velocity_n_m_s": 1.0e5,
            "thermal_velocity_p_m_s": 1.0e5,
        },
    }


def _resolve(doc: dict[str, Any], defaults: DefaultCatalog) -> PreparedDevice:
    return resolve_device(DeviceInput.model_validate(doc), defaults, ResourceLibrary())


def _three_layer_doc(
    *,
    with_bulk: bool = True,
    bulk_kind: str = "single_level",
    with_iface: bool = True,
    with_ions: bool = True,
    bulk_density: float = 3.0e21,
) -> dict[str, Any]:
    abs_layer: dict[str, Any] = {
        "id": "abs",
        "name": "Abs",
        "role": "absorber",
        "thickness": 320e-9,
        "parameters": {
            **_mat(1.6, 3.9),
            **(
                {"P0": 1.5e23, "P_lim": 1.0e25, "D_ion": 2.0e-15}
                if with_ions
                else {}
            ),
        },
    }
    if with_bulk:
        abs_layer["defect_model"] = "explicit_quasi_steady"
        abs_layer["defect_schema_version"] = (
            "solarlab-explicit-bulk-defects-v1"
            if bulk_kind == "single_level"
            else "solarlab-explicit-bulk-defects-v2"
        )
        abs_layer["bulk_defects"] = [
            _bulk_defect_entry("t1", density=bulk_density, kind=bulk_kind)
        ]

    if0: dict[str, Any] = {"id": "if_htl_abs", "left": "htl", "right": "abs"}
    if with_iface:
        if0["defect"] = _iface_defect_entry("id1")

    return {
        "schema_version": "solarlab.device-preparation.v1",
        "source_format": "canonical",
        "id": f"pin_{with_bulk}_{bulk_kind}_{with_iface}_{with_ions}",
        "layers": [
            {"id": "htl", "name": "HTL", "role": "htl", "thickness": 70e-9, "parameters": _mat(2.1, 3.5, na=2e22)},
            abs_layer,
            {"id": "etl", "name": "ETL", "role": "etl", "thickness": 70e-9, "parameters": _mat(2.2, 3.95, nd=2e22)},
        ],
        "interfaces": [
            if0,
            {"id": "if_abs_etl", "left": "abs", "right": "etl"},
        ],
        "settings": _settings(
            interface_charge_closure="equilibrium_referenced" if with_iface else "off",
            interface_charge_rebaseline_acknowledged=with_iface,
        ),
    }


def _assert_jacobian_and_storage_fd(
    dev: Any, u0: np.ndarray, *, voltage_V: float = 0.35
) -> None:
    rng = np.random.default_rng(20261114)
    # Perturb away from exact steady QSS root so dynamic trap storage rates are nonzero:
    u_test = u0.copy()
    N = dev.node_count
    u_test[1 : N - 1] += 0.005 * rng.standard_normal(N - 2)
    u_test[N + 1 : 2 * N - 1] += 0.02 * rng.standard_normal(N - 2)
    u_test[2 * N + 1 : 3 * N - 1] += 0.02 * rng.standard_normal(N - 2)
    if u_test.size > 3 * N:
        u_test[3 * N :] += 0.15 * rng.standard_normal(u_test.size - 3 * N)

    prob = AssembledDynamicDefect1DProblem(dev, voltage_V=voltage_V, generation_scale=0.0)
    j_ana = prob.jacobian(u_test).toarray()
    eps = 1.0e-7
    eps = 1.0e-5
    j_fd = np.zeros_like(j_ana)
    for col in range(u_test.size):
        step = eps * max(1.0, abs(float(u_test[col])))
        u_p1 = u_test.copy()
        u_p1[col] += step
        u_m1 = u_test.copy()
        u_m1[col] -= step
        u_p2 = u_test.copy()
        u_p2[col] += 2.0 * step
        u_m2 = u_test.copy()
        u_m2[col] -= 2.0 * step
        j_fd[:, col] = (
            -prob.residual(u_p2)
            + 8.0 * prob.residual(u_p1)
            - 8.0 * prob.residual(u_m1)
            + prob.residual(u_m2)
        ) / (12.0 * step)
    rel_j = float(np.linalg.norm(j_ana - j_fd) / np.linalg.norm(j_fd))
    assert rel_j <= 1.0e-10

    sys_tr = dynamic_defect_system_1d(dev, inventory_row_node=None)
    prob_tr = AssembledDynamicDefect1DProblem(
        dev, voltage_V=voltage_V, generation_scale=0.0, system=sys_tr
    )
    adapter = DynamicDefectStorage1D(prob_tr)
    k_ana = adapter.storage_tangent(u_test).toarray()
    k_fd = np.zeros_like(k_ana)
    for col in range(u_test.size):
        step = eps * max(1.0, abs(float(u_test[col])))
        u_p1 = u_test.copy()
        u_p1[col] += step
        u_m1 = u_test.copy()
        u_m1[col] -= step
        u_p2 = u_test.copy()
        u_p2[col] += 2.0 * step
        u_m2 = u_test.copy()
        u_m2[col] -= 2.0 * step
        k_fd[:, col] = (
            -adapter.storage(u_p2)
            + 8.0 * adapter.storage(u_p1)
            - 8.0 * adapter.storage(u_m1)
            + adapter.storage(u_m2)
        ) / (12.0 * step)
    rel_k = float(np.linalg.norm(k_ana - k_fd) / np.linalg.norm(k_fd))
    assert rel_k <= 1.0e-10


def test_assembled_jacobians_and_storage_tangents_across_all_topologies(
    defaults: DefaultCatalog,
) -> None:
    """Verify full FD residual & storage Jacobians across bulk, interface, and coupled ion topologies."""
    cases = [
        (True, "gaussian", False, False),
        (False, "single_level", True, False),
        (True, "single_level", False, True),
        (False, "single_level", True, True),
        (True, "single_level", True, True),
    ]
    for with_bulk, bulk_kind, with_iface, with_ions in cases:
        prep = _resolve(
            _three_layer_doc(
                with_bulk=with_bulk,
                bulk_kind=bulk_kind,
                with_iface=with_iface,
                with_ions=with_ions,
            ),
            defaults,
        )
        dev = compile_dynamic_defect_device_1d(
            prep,
            intervals_per_layer=6,
            dynamic_bulk_defects=with_bulk,
            dynamic_interface_defects=with_iface,
            energy_quadrature_order=8,
            ion_domain_layer="abs" if with_ions else None,
            ion_domain_endpoints=("blocking", "blocking") if with_ions else None,
        )
        st = solve_dynamic_defect_steady_1d(dev, voltage_V=0.35, generation_scale=0.0)
        assert st.converged
        _assert_jacobian_and_storage_fd(dev, st.state.u, voltage_V=0.35)


def test_steady_qss_embedding_and_zero_density_limit(defaults: DefaultCatalog) -> None:
    """Verify dynamic steady state matches stationary QSS and N_t -> 0 approaches baseline."""
    prep = _resolve(
        _three_layer_doc(with_bulk=True, with_iface=True, with_ions=False), defaults
    )
    dev_dyn = compile_dynamic_defect_device_1d(
        prep,
        intervals_per_layer=8,
        dynamic_bulk_defects=True,
        dynamic_interface_defects=True,
    )
    dev_qss = compile_dynamic_defect_device_1d(
        prep,
        intervals_per_layer=8,
        dynamic_bulk_defects=False,
        dynamic_interface_defects=False,
    )
    st_dyn = solve_dynamic_defect_steady_1d(dev_dyn, voltage_V=0.45, generation_scale=0.0)
    st_qss = solve_dynamic_defect_steady_1d(dev_qss, voltage_V=0.45, generation_scale=0.0)
    assert st_dyn.converged and st_qss.converged
    assert st_dyn.terminal_current_density_A_m2 == pytest.approx(
        st_qss.terminal_current_density_A_m2, rel=1e-11
    )
    np.testing.assert_allclose(
        st_dyn.state.electronic_state.potential_V,
        st_qss.state.electronic_state.potential_V,
        rtol=1e-11,
        atol=1e-12,
    )

    # Small-density limit N_t -> 0 reproduces lifetime-SRH-free electronic baseline
    # (explicit_quasi_steady replaces absorber effective-lifetime tau_n/tau_p SRH):
    doc_zero = _three_layer_doc(
        with_bulk=True, with_iface=False, with_ions=False, bulk_density=1.0e3
    )
    doc_clean = _three_layer_doc(with_bulk=False, with_iface=False, with_ions=False)
    doc_clean["layers"][1]["parameters"]["tau_n"] = 1.0e30
    doc_clean["layers"][1]["parameters"]["tau_p"] = 1.0e30
    prep_zero = _resolve(doc_zero, defaults)
    prep_clean = _resolve(doc_clean, defaults)
    dev_zero = compile_dynamic_defect_device_1d(
        prep_zero, intervals_per_layer=8, dynamic_bulk_defects=False
    )
    dev_clean = compile_dynamic_defect_device_1d(
        prep_clean, intervals_per_layer=8, dynamic_bulk_defects=False
    )
    st_zero = solve_dynamic_defect_steady_1d(dev_zero, voltage_V=0.85, generation_scale=0.0)
    st_clean = solve_dynamic_defect_steady_1d(dev_clean, voltage_V=0.85, generation_scale=0.0)
    assert st_zero.terminal_current_density_A_m2 == pytest.approx(
        st_clean.terminal_current_density_A_m2, rel=1e-9
    )


def test_small_signal_ac_limits_digest_invariance_and_non_additivity(
    defaults: DefaultCatalog,
) -> None:
    """Verify 0 Hz FD slope, low/high-f limits, ion conservation, and electrostatic non-additivity."""
    prep = _resolve(
        _three_layer_doc(with_bulk=True, with_iface=True, with_ions=True), defaults
    )
    dev = compile_dynamic_defect_device_1d(
        prep,
        intervals_per_layer=8,
        dynamic_bulk_defects=True,
        dynamic_interface_defects=True,
        ion_domain_layer="abs",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    v0 = 0.40
    st = solve_dynamic_defect_steady_1d(dev, voltage_V=v0, generation_scale=0.0)
    ss = DynamicDefectSmallSignal1D(dev, st.state, generation_scale=0.0)

    freqs = np.array([0.0, 1.0e-7, 1.0, 1.0e3, 1.0e9])
    coupled = ss.solve_sweep(freqs)
    f_ions = ss.solve_sweep(freqs, freeze_ions=True)
    f_traps = ss.solve_sweep(
        freqs, freeze_bulk_traps=True, freeze_interface_traps=True
    )
    f_both = ss.solve_sweep(
        freqs,
        freeze_ions=True,
        freeze_bulk_traps=True,
        freeze_interface_traps=True,
    )

    assert (
        coupled.working_state_digest
        == f_ions.working_state_digest
        == f_traps.working_state_digest
        == f_both.working_state_digest
    )

    # Low-frequency limit approaches 0 Hz QSS and central-FD dJ/dV:
    y0 = complex(coupled.admittance_density_S_m2[0])
    y_lf = complex(coupled.admittance_density_S_m2[1])
    assert abs(y_lf - y0) / abs(y0) <= 1.0e-4

    dv = 1.0e-5
    st_p = solve_dynamic_defect_steady_1d(
        dev,
        voltage_V=v0 + dv,
        generation_scale=0.0,
        initial_u=dev.rebase_coordinates(st.state.u, v0, v0 + dv),
    )
    st_m = solve_dynamic_defect_steady_1d(
        dev,
        voltage_V=v0 - dv,
        generation_scale=0.0,
        initial_u=dev.rebase_coordinates(st.state.u, v0, v0 - dv),
    )
    dj_dv = (st_p.terminal_current_density_A_m2 - st_m.terminal_current_density_A_m2) / (2.0 * dv)
    assert y0.imag == pytest.approx(0.0, abs=1e-16)
    assert y0.real == pytest.approx(dj_dv, rel=1e-5)

    # High-frequency limit approaches frozen-species admittance:
    y_hf = complex(coupled.admittance_density_S_m2[-1])
    y_hf_frozen = complex(f_both.admittance_density_S_m2[-1])
    assert abs(y_hf - y_hf_frozen) / abs(y_hf_frozen) <= 1.0e-4

    # Ion inventory conservation across all frequencies including 0 Hz:
    for pt in coupled.points:
        assert pt.ion_inventory_relative_per_V["positive"] <= 1.0e-12

    # Coupled defect + ion admittance is NOT equal to naive linear superposition at 1 Hz:
    idx_mid = 2
    y_coupled_mid = coupled.admittance_density_S_m2[idx_mid]
    y_naive_sum = (
        f_ions.admittance_density_S_m2[idx_mid]
        + f_traps.admittance_density_S_m2[idx_mid]
        - f_both.admittance_density_S_m2[idx_mid]
    )
    assert abs(y_coupled_mid - y_naive_sum) / abs(y_coupled_mid) > 1.0e-4


def test_compile_capability_refusals(defaults: DefaultCatalog) -> None:
    """Verify pre-solve CapabilityError refusals on unsupported dynamic combinations."""
    # 1. Calibrated interface defect refused:
    bad_cal = _three_layer_doc(with_bulk=False, with_iface=True, with_ions=False)
    bad_cal["interfaces"][0]["defect"] = _iface_defect_entry(calibration_factor=2.5)
    prep_cal = _resolve(bad_cal, defaults)
    with pytest.raises(CapabilityError, match="calibration"):
        compile_dynamic_defect_device_1d(prep_cal, dynamic_interface_defects=True)

    # 2. Two-sided interior ion layer with < 3 trace nodes refused:
    prep_ion = _resolve(
        _three_layer_doc(with_bulk=False, with_iface=True, with_ions=True), defaults
    )
    with pytest.raises(CapabilityError, match="at least 3 trace nodes"):
        compile_dynamic_defect_device_1d(
            prep_ion,
            intervals_per_layer=3,
            dynamic_interface_defects=True,
            ion_domain_layer="abs",
            ion_domain_endpoints=("blocking", "blocking"),
        )

    # 3. Multivalent v4 refused when dynamic_bulk_defects=True:
    mv_doc = copy.deepcopy(_three_layer_doc(with_bulk=False, with_iface=False, with_ions=False))
    mv_doc["layers"][1].update(
        {
            "defect_model": "explicit_quasi_steady",
            "defect_schema_version": "solarlab-explicit-bulk-defects-v4",
            "bulk_defects": [
                {
                    "id": "amp1",
                    "name": "Amphoteric",
                    "total_density_m3": 5.0e21,
                    "configuration": {
                        "family": "amphoteric",
                        "charge_states_e": [1, 0, -1],
                        "degeneracy_convention": "scaps_binomial",
                        "state_degeneracies": [1.0, 2.0, 1.0],
                        "energy_levels": {
                            "first_transition_eV_above_vb": 0.55,
                            "correlation_energies_eV": [0.35],
                            "energy_reference": "above_valence_band",
                        },
                        "transition_kinetics": [
                            {
                                "sigma_n_m2": 1e-19,
                                "sigma_p_m2": 1e-19,
                                "thermal_velocity_n_m_s": 1e5,
                                "thermal_velocity_p_m_s": 1e5,
                            },
                            {
                                "sigma_n_m2": 1e-19,
                                "sigma_p_m2": 1e-19,
                                "thermal_velocity_n_m_s": 1e5,
                                "thermal_velocity_p_m_s": 1e5,
                            },
                        ],
                    },
                }
            ],
        }
    )
    prep_mv = _resolve(mv_doc, defaults)
    with pytest.raises(CapabilityError, match="multivalent"):
        compile_dynamic_defect_device_1d(prep_mv, dynamic_bulk_defects=True)

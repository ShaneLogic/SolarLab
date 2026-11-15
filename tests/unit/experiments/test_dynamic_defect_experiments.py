"""Unit tests for dynamic-defect C-f, impedance, coupled defect-ion impedance, and transients (P05-03b, P05-04b)."""

from __future__ import annotations

from collections.abc import Iterator
import copy
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.defects import MetastablePreparationInput
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.capacitance_frequency import (
    DefectCFError,
    DefectCFOptions,
    capacitance_frequency_from_admittance,
    run_defect_capacitance_frequency,
)
from solarlab.experiments.defect_ion_impedance import (
    DefectIonImpedanceCapabilityError,
    DefectIonImpedanceOptions,
    solve_defect_ion_impedance,
)
from solarlab.experiments.dynamic_defect_impedance import (
    DynamicDefectImpedanceCapabilityError,
    DynamicDefectImpedanceOptions,
    classify_dynamic_defect_capability,
    solve_bulk_defect_impedance,
    solve_interface_defect_impedance,
)
from solarlab.experiments.dynamic_defect_transient import (
    DynamicDefectTransientCapabilityError,
    DynamicDefectTransientOptions,
    classify_dynamic_defect_transient_capability,
    solve_bulk_defect_ion_transient,
    solve_bulk_defect_transient,
    solve_dynamic_defect_transient,
    solve_interface_defect_ion_transient,
    solve_interface_defect_transient,
)
from solarlab.experiments.metastable_transient import (
    MetastablePreparationError,
    prepare_metastable_state,
    solve_frozen_metastable_jv,
    solve_frozen_metastable_steady,
    solve_metastable_transient,
)
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


def _bulk_defect(
    charge_transition: str = "acceptor",
    kind: str = "single_level",
) -> dict[str, Any]:
    dist: dict[str, Any] = {
        "kind": kind,
        "normalization": "integrated_total",
        "total_density_m3": 2.0e21,
    }
    if kind == "gaussian":
        dist["energy_reference"] = "above_valence_band"
        dist["width_eV"] = 0.05
        dist["width_convention"] = "gaussian_standard_deviation"
        dist["support_width_multiplier"] = 6.0
    neutral_ref = (
        "filled"
        if charge_transition == "donor"
        else ("all_occupancies" if charge_transition == "neutral" else "empty")
    )
    return {
        "id": "t1",
        "name": "Trap1",
        "charge_transition": charge_transition,
        "neutral_reference": neutral_ref,
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


def _iface_defect(calibration_factor: float = 1.0) -> dict[str, Any]:
    return {
        "id": "id1",
        "total_density_m2": 1.2e15,
        "trap_depth_eV": 0.60,
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


def _pin_doc(
    *,
    with_bulk: bool = False,
    bulk_transition: str = "acceptor",
    bulk_kind: str = "single_level",
    with_iface: bool = False,
    with_ions: bool = False,
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
        abs_layer["bulk_defects"] = [_bulk_defect(bulk_transition, bulk_kind)]

    if0: dict[str, Any] = {"id": "if_htl_abs", "left": "htl", "right": "abs"}
    if with_iface:
        if0["defect"] = _iface_defect()

    return {
        "schema_version": "solarlab.device-preparation.v1",
        "source_format": "canonical",
        "id": "exp_pin",
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


def _reverse_to_nip(doc: dict[str, Any]) -> dict[str, Any]:
    rev = copy.deepcopy(doc)
    rev["id"] = f"nip_{doc['id']}"
    rev["layers"] = list(reversed(rev["layers"]))
    revs: list[dict[str, Any]] = []
    for iface in reversed(rev["interfaces"]):
        ni = copy.deepcopy(iface)
        ni["left"], ni["right"] = iface["right"], iface["left"]
        revs.append(ni)
    rev["interfaces"] = revs
    return rev


def _two_layer_v1_transient_doc() -> dict[str, Any]:
    return {
        "schema_version": "solarlab.device-preparation.v1",
        "source_format": "canonical",
        "id": "v1_prod_tr",
        "layers": [
            {
                "id": "abs",
                "name": "Abs",
                "role": "absorber",
                "thickness": 300e-9,
                "parameters": {
                    **_mat(1.6, 3.9, na=1e21),
                    "P0": 1.0e23,
                    "P_lim": 1.0e25,
                    "D_ion": 2.0e-15,
                },
            },
            {
                "id": "etl",
                "name": "ETL",
                "role": "etl",
                "thickness": 80e-9,
                "parameters": _mat(2.2, 3.95, nd=2e22),
            },
        ],
        "interfaces": [
            {"id": "if_abs_etl", "left": "abs", "right": "etl", "defect": _iface_defect()},
        ],
        "settings": _settings(
            interface_charge_closure="equilibrium_referenced",
            interface_charge_rebaseline_acknowledged=True,
        ),
    }


def test_capacitance_frequency_debye_analytics_masking_and_passivity(
    defaults: DefaultCatalog,
) -> None:
    """Verify exact Debye C_p/C_s/Walter peak, susceptance masking, and dark passivity."""
    g0 = 0.02
    c_inf = 4.0e-4
    dc = 2.0e-4
    tau = 1.0e-3
    freqs = np.geomspace(1.0, 1.0e6, 241)
    omega = 2.0 * np.pi * freqs
    y_debye = g0 + 1j * omega * (c_inf + dc / (1.0 + 1j * omega * tau))

    cf = capacitance_frequency_from_admittance(freqs, y_debye, operating_voltage_V=0.0)
    assert cf.scientific_qualification is False
    assert cf.physical_certificate_status == "not_evaluated"
    assert cf.dark_equilibrium_passive is True
    assert np.all(cf.capacitance_accepted)

    expected_cp = c_inf + dc / (1.0 + (omega * tau) ** 2)
    expected_gp = g0 + (omega**2 * tau * dc) / (1.0 + (omega * tau) ** 2)
    np.testing.assert_allclose(cf.parallel_capacitance_F_m2, expected_cp, rtol=1e-13)
    np.testing.assert_allclose(cf.parallel_conductance_S_m2, expected_gp, rtol=1e-13)
    np.testing.assert_allclose(
        cf.loss_tangent, expected_gp / (omega * expected_cp), rtol=1e-13
    )

    f_peak_exact = 1.0 / (2.0 * math.pi * tau)
    assert cf.peak_walter_frequency_Hz is not None
    assert cf.peak_walter_value_F_m2 is not None
    assert abs(math.log10(cf.peak_walter_frequency_Hz / f_peak_exact)) < 0.03
    assert cf.peak_walter_value_F_m2 == pytest.approx(0.5 * dc, rel=5e-3)

    # Masking ledger (0 Hz, inductive Im(Y) < 0, zero Im(Y) == 0, unresolved phase):
    f_mix = np.array([0.0, 10.0, 100.0, 1000.0, 10000.0])
    y_mix = np.array([1.0 + 0.0j, 1.0 - 0.5j, 1.0 + 0.0j, 100.0 + 0.01j, 0.1 + 1.0j])
    masked = capacitance_frequency_from_admittance(
        f_mix,
        y_mix,
        operating_voltage_V=0.2,
        options=DefectCFOptions(minimum_phase_angle_rad=0.05),
    )
    assert masked.mask_reasons == (
        "zero_frequency",
        "inductive_susceptance",
        "zero_susceptance",
        "unresolved_phase",
        None,
    )
    assert list(masked.capacitance_accepted) == [False, False, False, False, True]

    with pytest.raises(DefectCFError, match="inductive"):
        capacitance_frequency_from_admittance(
            np.array([10.0]),
            np.array([1.0 - 0.2j]),
            operating_voltage_V=0.2,
            options=DefectCFOptions(strict_capacitive=True),
        )

    with pytest.raises(DefectCFError, match="passivity"):
        capacitance_frequency_from_admittance(
            np.array([10.0, 100.0]),
            np.array([-1.0e-3 + 0.1j, 0.2 + 0.5j]),
            operating_voltage_V=0.0,
            generation_scale=0.0,
        )

    # End-to-end device C-f run (both p-i-n and reversed n-i-p across bulk & interface traps):
    prep = _resolve(_pin_doc(with_bulk=True), defaults)
    dev_cf = run_defect_capacitance_frequency(
        prep, np.array([1.0, 100.0, 1.0e5]), intervals_per_layer=6
    )
    assert dev_cf.dark_equilibrium_passive is True
    assert np.all(dev_cf.capacitance_accepted)

    for nip_doc in (
        _reverse_to_nip(_pin_doc(with_bulk=True)),
        _reverse_to_nip(_pin_doc(with_iface=True)),
    ):
        nip_cf = run_defect_capacitance_frequency(
            _resolve(nip_doc, defaults),
            np.array([1.0e-1, 10.0, 1.0e4]),
            voltage_V=0.0,
            generation_scale=0.0,
            intervals_per_layer=6,
        )
        assert nip_cf.dark_equilibrium_passive is True
        assert np.all(nip_cf.capacitance_accepted)


def test_dynamic_defect_and_combined_ion_impedance_capabilities_and_execution(
    defaults: DefaultCatalog,
) -> None:
    """Verify all 5 capability classifications, refusals, FD 0 Hz check, and 4-mode AC."""
    assert (
        classify_dynamic_defect_capability(_resolve(_pin_doc(with_bulk=True), defaults))
        == "bulk_dynamic_defect"
    )
    assert (
        classify_dynamic_defect_capability(_resolve(_pin_doc(with_iface=True), defaults))
        == "interface_dynamic_defect"
    )
    assert (
        classify_dynamic_defect_capability(
            _resolve(_pin_doc(with_bulk=True, with_ions=True), defaults)
        )
        == "bulk_defect_plus_ions"
    )
    assert (
        classify_dynamic_defect_capability(
            _resolve(_pin_doc(with_iface=True, with_ions=True), defaults)
        )
        == "interface_defect_plus_ions"
    )
    assert (
        classify_dynamic_defect_capability(
            _resolve(_pin_doc(with_bulk=True, with_iface=True, with_ions=True), defaults)
        )
        == "bulk_interface_defect_plus_ions"
    )

    # Refusal: bulk + interface WITHOUT active mobile ions:
    with pytest.raises(DynamicDefectImpedanceCapabilityError, match="without active mobile ions"):
        classify_dynamic_defect_capability(
            _resolve(_pin_doc(with_bulk=True, with_iface=True, with_ions=False), defaults)
        )

    # Refusal: no defects:
    with pytest.raises(DynamicDefectImpedanceCapabilityError, match="requires an explicit"):
        classify_dynamic_defect_capability(_resolve(_pin_doc(), defaults))

    # Refusal: neutral bulk defect has zero electrostatic coupling so cannot carry dynamic AC DOFs:
    prep_neutral_bulk = _resolve(_pin_doc(with_bulk=True, bulk_transition="neutral"), defaults)
    with pytest.raises(DynamicDefectImpedanceCapabilityError, match="neutral"):
        classify_dynamic_defect_capability(prep_neutral_bulk)
    with pytest.raises(DynamicDefectImpedanceCapabilityError, match="neutral"):
        solve_bulk_defect_impedance(prep_neutral_bulk, [1.0, 100.0])

    # Refusal: interface_charge_closure='off' refused across dynamic interface impedance & joint AC:
    off_iface_doc = _pin_doc(with_iface=True)
    off_iface_doc["settings"]["interface_charge_closure"] = "off"
    off_iface_doc["settings"]["interface_charge_rebaseline_acknowledged"] = False
    prep_off_iface = _resolve(off_iface_doc, defaults)
    with pytest.raises(DynamicDefectImpedanceCapabilityError, match="equilibrium_referenced"):
        classify_dynamic_defect_capability(prep_off_iface)
    with pytest.raises(DynamicDefectImpedanceCapabilityError, match="equilibrium_referenced"):
        solve_interface_defect_impedance(prep_off_iface, [1.0, 100.0])

    off_joint_doc = _pin_doc(with_iface=True, with_ions=True)
    off_joint_doc["settings"]["interface_charge_closure"] = "off"
    off_joint_doc["settings"]["interface_charge_rebaseline_acknowledged"] = False
    with pytest.raises(DefectIonImpedanceCapabilityError, match="equilibrium_referenced"):
        solve_defect_ion_impedance(_resolve(off_joint_doc, defaults), [1.0, 100.0])

    # Refusal: solve_defect_ion_impedance on ion-free device:
    prep_bulk = _resolve(_pin_doc(with_bulk=True), defaults)
    with pytest.raises(DefectIonImpedanceCapabilityError, match="active mobile ions"):
        solve_defect_ion_impedance(prep_bulk, [1.0, 100.0, 1.0e4])

    freqs = np.array([1.0e-6, 10.0, 1.0e4, 1.0e9])
    res_bulk = solve_bulk_defect_impedance(
        prep_bulk,
        freqs,
        voltage_V=0.55,
        options=DynamicDefectImpedanceOptions(intervals_per_layer=6, verify_zero_hz_fd=True),
    )
    assert res_bulk.scientific_qualification is False
    assert res_bulk.zero_hz_fd_relative_error is not None
    assert res_bulk.zero_hz_fd_relative_error <= 1.0e-4
    assert res_bulk.low_frequency_qss_relative_error <= 1.0e-3
    assert res_bulk.high_frequency_frozen_relative_error <= 1.0e-3

    prep_iface = _resolve(_pin_doc(with_iface=True), defaults)
    res_iface = solve_interface_defect_impedance(
        prep_iface,
        freqs,
        voltage_V=0.25,
        options=DynamicDefectImpedanceOptions(intervals_per_layer=6),
    )
    assert res_iface.capability == "interface_dynamic_defect"

    prep_joint = _resolve(
        _pin_doc(with_bulk=True, with_iface=True, with_ions=True), defaults
    )
    res_joint = solve_defect_ion_impedance(
        prep_joint,
        freqs,
        voltage_V=0.25,
        options=DefectIonImpedanceOptions(intervals_per_layer=6),
    )
    assert res_joint.capability == "bulk_interface_defect_plus_ions"
    assert res_joint.scientific_qualification is False
    assert (
        res_joint.working_state_digest
        == res_joint.coupled.working_state_digest
        == res_joint.frozen_ions.working_state_digest
        == res_joint.frozen_traps.working_state_digest
        == res_joint.frozen_ions_and_traps.working_state_digest
    )
    assert res_joint.maximum_ion_inventory_relative_per_V <= 1.0e-12
    for sw in (res_joint.coupled, res_joint.frozen_ions):
        assert np.all(sw.maxwell_spread_A_m2_V <= 1.0e-11 * np.abs(sw.admittance_density_S_m2))
        assert np.all(sw.maxwell_spread_A_m2_V[:3] <= 1.0e-11)
    assert res_joint.maximum_maxwell_spread_A_m2_V == max(
        float(np.max(res_joint.coupled.maxwell_spread_A_m2_V)),
        float(np.max(res_joint.frozen_ions.maxwell_spread_A_m2_V)),
    )

    # Dark 0 V reversed n-i-p impedance FD slope & Maxwell spread regression check:
    nip_freqs = np.array([1.0e-2, 1.0, 1.0e3, 1.0e5])
    res_nip_bulk = solve_bulk_defect_impedance(
        _resolve(_reverse_to_nip(_pin_doc(with_bulk=True)), defaults),
        nip_freqs,
        voltage_V=0.0,
        generation_scale=0.0,
        options=DynamicDefectImpedanceOptions(intervals_per_layer=8, verify_zero_hz_fd=True),
    )
    assert res_nip_bulk.zero_hz_fd_relative_error is not None
    assert res_nip_bulk.zero_hz_fd_relative_error <= 1.0e-5
    assert np.all(res_nip_bulk.admittance_density_S_m2.real > 0.0)
    assert float(np.max(res_nip_bulk.ac_sweep.maxwell_spread_A_m2_V)) <= 1.0e-9

    res_nip_iface = solve_interface_defect_impedance(
        _resolve(_reverse_to_nip(_pin_doc(with_iface=True)), defaults),
        nip_freqs,
        voltage_V=0.0,
        generation_scale=0.0,
        options=DynamicDefectImpedanceOptions(intervals_per_layer=8, verify_zero_hz_fd=True),
    )
    assert res_nip_iface.zero_hz_fd_relative_error is not None
    assert res_nip_iface.zero_hz_fd_relative_error <= 1.0e-5
    assert np.all(res_nip_iface.admittance_density_S_m2.real > 0.0)
    assert float(np.max(res_nip_iface.ac_sweep.maxwell_spread_A_m2_V)) <= 1.0e-9


def test_dynamic_defect_transients_and_narrow_gate_refusals(
    defaults: DefaultCatalog,
) -> None:
    """Verify narrow v1 production transient gates and all 5 transient drivers."""
    prep_v1 = _resolve(_two_layer_v1_transient_doc(), defaults)
    assert (
        classify_dynamic_defect_transient_capability(prep_v1)
        == "interface_defect_plus_positive_ions"
    )

    times = np.array([0.0, 1.0e-5, 5.0e-5, 2.0e-4])
    volts = np.array([0.0, 0.30, 0.30, 0.0])

    # Narrow-gate refusals:
    with pytest.raises(DynamicDefectTransientCapabilityError, match="dark-only"):
        solve_dynamic_defect_transient(prep_v1, times, volts, illuminated=True)

    with pytest.raises(
        DynamicDefectTransientCapabilityError, match="right_continuous_step_and_hold"
    ):
        solve_dynamic_defect_transient(
            prep_v1,
            times,
            volts,
            options=DynamicDefectTransientOptions(voltage_interpolation="piecewise_linear"),
        )

    with pytest.raises(DynamicDefectTransientCapabilityError, match="two electrical layers"):
        classify_dynamic_defect_transient_capability(
            _resolve(_pin_doc(with_iface=True, with_ions=True), defaults)
        )

    no_ack_doc = _two_layer_v1_transient_doc()
    no_ack_doc["settings"]["interface_charge_rebaseline_acknowledged"] = False
    with pytest.raises((ValueError, DynamicDefectTransientCapabilityError), match="acknowledgement"):
        classify_dynamic_defect_transient_capability(_resolve(no_ack_doc, defaults))

    # Execute narrow-gate v1 transient:
    opts = DynamicDefectTransientOptions(intervals_per_layer=6, bdf_order=2)
    tr_v1 = solve_dynamic_defect_transient(prep_v1, times, volts, options=opts)
    assert tr_v1.scientific_qualification is False
    assert tr_v1.maximum_ion_inventory_relative_drift <= 1.0e-12
    assert tr_v1.interface_trap_occupancy.shape == (4, 1)

    # D6-E1 bulk transient + neutral/Gaussian refusals (both ion-free and ion-coupled):
    with pytest.raises(DynamicDefectTransientCapabilityError, match="neutral"):
        solve_bulk_defect_transient(
            _resolve(_pin_doc(with_bulk=True, bulk_transition="neutral"), defaults),
            times,
            volts,
        )
    with pytest.raises(DynamicDefectTransientCapabilityError, match="neutral"):
        solve_bulk_defect_ion_transient(
            _resolve(
                _pin_doc(with_bulk=True, bulk_transition="neutral", with_ions=True), defaults
            ),
            times,
            volts,
        )
    off_tr_iface = _pin_doc(with_iface=True)
    off_tr_iface["settings"]["interface_charge_closure"] = "off"
    off_tr_iface["settings"]["interface_charge_rebaseline_acknowledged"] = False
    with pytest.raises(DynamicDefectTransientCapabilityError, match="equilibrium_referenced"):
        solve_interface_defect_transient(_resolve(off_tr_iface, defaults), times, volts)

    off_tr_iface_ion = _pin_doc(with_iface=True, with_ions=True)
    off_tr_iface_ion["settings"]["interface_charge_closure"] = "off"
    off_tr_iface_ion["settings"]["interface_charge_rebaseline_acknowledged"] = False
    with pytest.raises(DynamicDefectTransientCapabilityError, match="equilibrium_referenced"):
        solve_interface_defect_ion_transient(_resolve(off_tr_iface_ion, defaults), times, volts)
    with pytest.raises(DynamicDefectTransientCapabilityError, match="single_level"):
        solve_bulk_defect_transient(
            _resolve(_pin_doc(with_bulk=True, bulk_kind="gaussian"), defaults),
            times,
            volts,
        )

    tr_bulk = solve_bulk_defect_transient(
        _resolve(_pin_doc(with_bulk=True), defaults), times, volts, options=opts
    )
    assert tr_bulk.capability == "bulk_defect_transient"

    tr_iface = solve_interface_defect_transient(
        _resolve(_pin_doc(with_iface=True), defaults), times, volts, options=opts
    )
    assert tr_iface.capability == "interface_defect_transient"

    tr_bi = solve_bulk_defect_ion_transient(
        _resolve(_pin_doc(with_bulk=True, with_ions=True), defaults),
        times,
        volts,
        options=opts,
    )
    assert tr_bi.capability == "bulk_defect_ion_transient"
    assert tr_bi.maximum_ion_inventory_relative_drift <= 1.0e-12

    tr_ii = solve_interface_defect_ion_transient(
        _resolve(_pin_doc(with_iface=True, with_ions=True), defaults),
        times,
        volts,
        options=opts,
    )
    assert tr_ii.capability == "interface_defect_ion_transient"
    assert tr_ii.maximum_ion_inventory_relative_drift <= 1.0e-12


def test_metastable_preparation_frozen_jv_and_operator_split_transient(
    defaults: DefaultCatalog,
) -> None:
    """Verify mandatory unclamped preparation gate, frozen J-V invariance, and dt/2 audit."""
    prep_dict = {
        "schema_version": "solarlab-metastable-preparation-v1",
        "preparation_limit": "stationary_infinite_time",
        "preparation_temperature_K": 320.0,
        "preparation_voltage_V": 0.45,
        "preparation_illumination_suns": 0.0,
        "voltage_continuation_steps": 0,
        "illumination_continuation_steps": 0,
        "measurement_temperature_K": 290.0,
        "configuration_freeze_stage": "after_stationary_preparation_before_measurement",
        "freeze_configuration_during_measurement": True,
        "measurement_protocol_sha256": "c" * 64,
        "numerics": {
            "initial_donor_fraction_guess": 0.50,
            "max_iterations": 25,
            "relative_tolerance": 1.0e-7,
            "clamping_factor": 0.7,
            "final_unclamped_refinement": True,
        },
    }
    meta_doc = {
        "schema_version": "solarlab.device-preparation.v1",
        "source_format": "canonical",
        "id": "meta_exp",
        "settings": _settings(Phi=2.5e21),
        "layers": [
            {
                "id": "abs",
                "name": "Abs",
                "role": "absorber",
                "thickness": 280e-9,
                "metastable_document": {
                    "schema_version": "solarlab-metastable-bulk-defects-v1",
                    "defect_model": "explicit_metastable_frozen",
                    "metastable_defects": [
                        {
                            "id": "vse_vcu",
                            "name": "MetastableComplex",
                            "total_density_m3": 2.0e21,
                            "donor_configuration": {
                                "family": "single_donor",
                                "charge_states_e": [1, 0],
                                "degeneracy_convention": "unity",
                                "state_degeneracies": [1.0, 1.0],
                                "energy_levels": {
                                    "first_transition_eV_above_vb": 1.35,
                                    "correlation_energies_eV": [],
                                    "energy_reference": "above_valence_band",
                                },
                                "transition_kinetics": [
                                    {
                                        "sigma_n_m2": 1e-19,
                                        "sigma_p_m2": 1e-20,
                                        "thermal_velocity_n_m_s": 1e5,
                                        "thermal_velocity_p_m_s": 1e5,
                                    }
                                ],
                            },
                            "acceptor_configuration": {
                                "family": "single_acceptor",
                                "charge_states_e": [0, -1],
                                "degeneracy_convention": "unity",
                                "state_degeneracies": [1.0, 1.0],
                                "energy_levels": {
                                    "first_transition_eV_above_vb": 0.30,
                                    "correlation_energies_eV": [],
                                    "energy_reference": "above_valence_band",
                                },
                                "transition_kinetics": [
                                    {
                                        "sigma_n_m2": 1e-20,
                                        "sigma_p_m2": 1e-19,
                                        "thermal_velocity_n_m_s": 1e5,
                                        "thermal_velocity_p_m_s": 1e5,
                                    }
                                ],
                            },
                            "donor_conversion_state_index": 0,
                            "acceptor_conversion_state_index": 1,
                            "conversion_kinetics": {
                                "transition_energy_eV_above_vb": 0.22,
                                "electron_capture_activation_eV": 0.25,
                                "electron_emission_activation_eV": 0.25 + 2.0 * (1.6 - 0.22),
                                "hole_capture_activation_eV": 0.30,
                                "hole_emission_activation_eV": 0.30 + 2.0 * 0.22,
                                "electron_capture_path": "double_electron_capture",
                                "hole_capture_path": "double_hole_capture",
                                "capture_n_m3_s": 1e-16,
                                "capture_p_m3_s": 1e-16,
                                "phonon_frequency_Hz": 1e12,
                            },
                        }
                    ],
                },
                "metastable_preparation": prep_dict,
                "parameters": _mat(1.6, 3.9, na=1e22),
            }
        ],
    }
    prepared = _resolve(meta_doc, defaults)
    state = prepare_metastable_state(prepared, intervals_per_layer=8)
    assert state.clamped_iterations >= 1
    assert state.unclamped_refinement_change <= 1.0e-7

    # Under-iterated clamped protocol must fail rather than masquerade as converged:
    bad_proto_dict = copy.deepcopy(prep_dict)
    bad_numerics = cast(dict[str, Any], bad_proto_dict["numerics"])
    bad_numerics["max_iterations"] = 1
    bad_numerics["clamping_factor"] = 0.05
    bad_proto = MetastablePreparationInput.model_validate(bad_proto_dict)
    with pytest.raises(MetastablePreparationError):
        prepare_metastable_state(prepared, preparation=bad_proto, intervals_per_layer=8)

    y_before = state.donor_fractions[0].copy()
    st0 = solve_frozen_metastable_steady(state, voltage_V=0.0)
    assert st0.converged
    jv = solve_frozen_metastable_jv(state, np.array([0.0, 0.15, 0.30]))
    assert jv.current_densities_A_m2.shape == (3,)
    np.testing.assert_array_equal(jv.donor_fractions[0], y_before)

    # Operator-split relaxation at V = 0.0 V, T = 290 K:
    tr = solve_metastable_transient(
        state,
        np.linspace(0.0, 1.0e-3, 5),
        voltage_V=0.0,
    )
    assert tr.scientific_qualification is False
    assert tr.certificate.passed is True
    assert tr.certificate.maximum_charge_transfer_relative_error <= 1.0e-10

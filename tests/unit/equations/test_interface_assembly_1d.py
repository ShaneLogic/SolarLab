"""Unit and assembly tests for 1D heterointerface closures on the unified Assembler (P04-09).

Covers:
- Exact binary64 equality between ``DC1DProblem`` and ``AssembledDC1DProblem`` across all
  supported 1D heterointerface modes (bare SRH, single-level defect SRH, plane projection,
  plane closure with/without generation, 1D thermionic emission capping, uncharged and
  equilibrium-referenced charged two-sided traces);
- Exact analytical sparse Jacobians vs directional finite differences across every mode;
- Invariant 1: finite interface transport (1D TE cap and two-sided trace) replaces—never
  acts in parallel with—the internal Scharfetter-Gummel face flux;
- Invariant 2: ``v_n = v_p = 0``, ``A_star = 0``, default flags remain 100% bitwise identical
  to a device without interface declarations;
- Invariant 3: falsified ``interface_shared_occupancy``, 1D ``interface_two_sided``,
  ``te_physical_norm``, ``interface_tunneling``, enabled ``tunnelling_channels``,
  ``het_recomb_despike != 0``, and unqualified combinations raise ``CapabilityError``
  before any nonlinear solve;
- Single constitutive evaluation per trial state for Gauss sheet-charge jump + carrier
  capture/emission, dark thermal equilibrium ``sigma == 0``, Gauss jump ``D_R - D_L == sigma``,
  and left/right stack reversal symmetry;
- Backward Euler storage-difference step, small-signal AC frequency solve, and multilayer
  1D single-domain + two-domain ionic assembly parity with active heterointerface closures.
"""

from __future__ import annotations

from collections.abc import Iterator
import copy
import math
from typing import Any
from unittest import mock

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.sparse import csc_matrix, issparse
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import (
    AssembledIonic1DProblem,
    AssembledMultipleIonic1DProblem,
)
from solarlab.equations.assembly.small_signal import AssembledIonicSmallSignal1D
from solarlab.equations.dc_1d import CapabilityError, CompiledDevice1D, CompiledInterfaceModel1D, DC1DProblem
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.ionic_small_signal import IonicSmallSignal1D
from solarlab.equations.multiple_ionic_1d import MultipleIonic1DProblem
from solarlab.physics.electrostatics.interface_traces import sheet_charge_row_weights
from solarlab.physics.interface_transport import TwoSidedBulkState, solve_charged_two_sided_interface_qss
from solarlab.physics.recombination.interface import interface_srh_rate
from solarlab.solvers.linear.frequency_domain import solve_frequency_systems
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_equilibrium, solve_steady_state
from solarlab.solvers.time.assembled_ionic_backward_euler import AssembledIonicBackwardEuler1DStep
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep

Vector = NDArray[np.float64]


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def base_prepared() -> PreparedDevice:
    return load_device()


def _edit_prepared(prepared: PreparedDevice, mutate_fn: Any) -> PreparedDevice:
    doc = copy.deepcopy(prepared.to_input().editing_data())
    mutate_fn(doc)
    return resolve_device(
        DeviceInput.model_validate(doc),
        prepared.defaults,
        prepared.resources,
        sources=prepared.sources,
    )


def _defect_dict(
    defect_id: str,
    *,
    trap_depth_eV: float = 0.55,
    energy_reference: str = "below_conduction_band",
    total_density_m2: float = 2.5e15,
    sigma_n_m2: float = 1.0e-19,
    sigma_p_m2: float = 2.0e-19,
    v_th_n_m_s: float = 1.0e5,
    v_th_p_m_s: float = 1.0e5,
    calibration_factor: float = 1.0,
) -> dict[str, Any]:
    return {
        "id": defect_id,
        "trap_depth_eV": trap_depth_eV,
        "energy_reference": energy_reference,
        "total_density_m2": total_density_m2,
        "kinetics": {
            "sigma_n_m2": sigma_n_m2,
            "sigma_p_m2": sigma_p_m2,
            "thermal_velocity_n_m_s": v_th_n_m_s,
            "thermal_velocity_p_m_s": v_th_p_m_s,
        },
        "calibration_factor": calibration_factor,
        "iface_state_calibration_factor": 1.0,
    }


def _canonical_csc_bytes(mat: csc_matrix) -> tuple[tuple[int, int], bytes, bytes, bytes]:
    c = csc_matrix(mat, copy=True)
    c.sum_duplicates()
    c.sort_indices()
    return c.shape, c.indptr.tobytes(), c.indices.tobytes(), c.data.tobytes()


def _assert_dc_assembled_bitwise_equal(
    device: CompiledDevice1D, u: Vector, *, voltage_V: float, generation_scale: float
) -> None:
    orig = DC1DProblem(device, voltage_V=voltage_V, generation_scale=generation_scale)
    assm = AssembledDC1DProblem(device, voltage_V=voltage_V, generation_scale=generation_scale)

    f_orig = orig.residual(u)
    f_assm = assm.residual(u)
    assert f_orig.tobytes() == f_assm.tobytes()

    j_orig = orig.jacobian(u)
    j_assm = assm.jacobian(u)
    assert issparse(j_orig) and issparse(j_assm)
    assert _canonical_csc_bytes(j_orig) == _canonical_csc_bytes(j_assm)

    fl_orig = orig.fluxes(u)
    fl_assm = assm.fluxes(u)
    assert fl_orig.electron.flux.tobytes() == fl_assm.electron.flux.tobytes()
    assert fl_orig.hole.flux.tobytes() == fl_assm.hole.flux.tobytes()


def _perturbed_trial_state(device: CompiledDevice1D, *, voltage_V: float, seed: int = 7) -> Vector:
    rng = np.random.default_rng(seed)
    n = device.node_count
    u_ref = np.r_[device.reference_potential_V, np.zeros(2 * n)]
    u0 = device.rebase_coordinates(u_ref, 0.0, voltage_V)
    # Smooth interior perturbation keeping boundary potentials and reservoirs clean.
    interior = np.sin(np.linspace(0.0, math.pi, n))
    dphi = 0.015 * interior * (1.0 + 0.2 * rng.standard_normal(n))
    dphi[0] = 0.0
    dphi[-1] = 0.0
    dzn = 0.25 * interior * (0.6 + 0.3 * rng.standard_normal(n))
    dzp = -0.20 * interior * (0.5 + 0.3 * rng.standard_normal(n))
    dzn[[0, -1]] = 0.0
    dzp[[0, -1]] = 0.0
    u = u0.copy()
    u[:n] += dphi
    u[n : 2 * n] += dzn
    u[2 * n :] += dzp
    return u


def _assert_directional_jacobian_fd(
    problem: AssembledDC1DProblem,
    u: Vector,
    *,
    step: float = 1.0e-5,
    rtol: float = 2.0e-6,
    directions: int = 5,
) -> float:
    """Verify ``J(u) @ v`` against central FD along random unit directions; return max rel err."""
    jac = problem.jacobian(u)
    rng = np.random.default_rng(12345)
    dense = jac.toarray()
    row_scale = np.maximum(np.linalg.norm(dense, axis=1), 1.0)
    worst_rel = 0.0
    for _ in range(directions):
        v = rng.standard_normal(u.size)
        v /= np.linalg.norm(v)
        jv = jac @ v
        fp = problem.residual(u + step * v)
        fm = problem.residual(u - step * v)
        fd = (fp - fm) / (2.0 * step)
        diff = np.abs(jv - fd)
        denom = np.maximum(np.abs(fd), 1.0e-7 * row_scale)
        rel = float(np.max(diff / denom))
        worst_rel = max(worst_rel, rel)
        np.testing.assert_allclose(jv, fd, rtol=rtol, atol=1.0e-7 * float(np.max(row_scale)))
    return worst_rel


def test_invariant2_zero_interface_parameters_are_bitwise_identical(
    base_prepared: PreparedDevice,
) -> None:
    explicit_zero = _edit_prepared(
        base_prepared,
        lambda doc: doc.__setitem__(
            "interfaces",
            [
                {"id": "htl_abs", "left": "htl", "right": "absorber", "v_n": 0.0, "v_p": 0.0},
                {"id": "abs_etl", "left": "absorber", "right": "etl", "v_n": 0.0, "v_p": 0.0},
            ],
        ),
    )
    dev_base = compile_device_1d(base_prepared, intervals_per_layer=8)
    dev_zero = compile_device_1d(explicit_zero, intervals_per_layer=8)
    assert dev_base.interface_model is None
    assert dev_zero.interface_model is None
    assert "heterointerface_1d" not in dev_base.capabilities
    assert "heterointerface_1d" not in dev_zero.capabilities

    u = _perturbed_trial_state(dev_base, voltage_V=0.55)
    p_base = AssembledDC1DProblem(dev_base, voltage_V=0.55, generation_scale=1.0)
    p_zero = AssembledDC1DProblem(dev_zero, voltage_V=0.55, generation_scale=1.0)
    assert p_base.residual(u).tobytes() == p_zero.residual(u).tobytes()
    assert _canonical_csc_bytes(p_base.jacobian(u)) == _canonical_csc_bytes(p_zero.jacobian(u))


@pytest.mark.parametrize(
    ("mode_name", "mutate"),
    [
        (
            "bare_srh",
            lambda doc: doc.__setitem__(
                "interfaces",
                [
                    {"id": "htl_abs", "left": "htl", "right": "absorber", "v_n": 25.0, "v_p": 40.0},
                    {"id": "abs_etl", "left": "absorber", "right": "etl", "v_n": 15.0, "v_p": 35.0},
                ],
            ),
        ),
        (
            "defect_cross_carrier_out_of_order",
            lambda doc: doc.__setitem__(
                "interfaces",
                [
                    {
                        "id": "abs_etl",
                        "left": "absorber",
                        "right": "etl",
                        "defect": _defect_dict("d_etl", trap_depth_eV=0.50, calibration_factor=0.75),
                    },
                    {
                        "id": "htl_abs",
                        "left": "htl",
                        "right": "absorber",
                        "defect": _defect_dict(
                            "d_htl",
                            trap_depth_eV=1.05,
                            energy_reference="above_valence_band",
                        ),
                    },
                ],
            ),
        ),
        (
            "plane_projection",
            lambda doc: (
                doc["settings"].__setitem__("interface_plane_projection", True),
                doc.__setitem__(
                    "interfaces",
                    [
                        {
                            "id": "htl_abs",
                            "left": "htl",
                            "right": "absorber",
                            "defect": _defect_dict("d_htl", trap_depth_eV=0.60),
                        },
                        {
                            "id": "abs_etl",
                            "left": "absorber",
                            "right": "etl",
                            "defect": _defect_dict("d_etl", trap_depth_eV=0.55),
                        },
                    ],
                ),
            ),
        ),
        (
            "plane_closure_clamped",
            lambda doc: (
                doc["settings"].__setitem__("interface_plane_closure", True),
                doc.__setitem__(
                    "interfaces",
                    [
                        {
                            "id": "htl_abs",
                            "left": "htl",
                            "right": "absorber",
                            "defect": _defect_dict("d_htl", trap_depth_eV=0.60),
                        },
                        {
                            "id": "abs_etl",
                            "left": "absorber",
                            "right": "etl",
                            "defect": _defect_dict("d_etl", trap_depth_eV=0.55),
                        },
                    ],
                ),
            ),
        ),
        (
            "plane_closure_with_generation",
            lambda doc: (
                doc["settings"].__setitem__("interface_plane_closure", True),
                doc["settings"].__setitem__("interface_plane_generation", True),
                doc.__setitem__(
                    "interfaces",
                    [
                        {
                            "id": "htl_abs",
                            "left": "htl",
                            "right": "absorber",
                            "defect": _defect_dict("d_htl", trap_depth_eV=0.60),
                        },
                        {
                            "id": "abs_etl",
                            "left": "absorber",
                            "right": "etl",
                            "defect": _defect_dict("d_etl", trap_depth_eV=0.55),
                        },
                    ],
                ),
            ),
        ),
        (
            "thermionic_cap_and_srh",
            lambda doc: (
                doc["layers"][0]["parameters"].update({"A_star_n": 2.5e5, "A_star_p": 3.0e5}),
                doc["layers"][1]["parameters"].update({"A_star_n": 1.2017e6, "A_star_p": 1.2017e6}),
                doc.__setitem__(
                    "interfaces",
                    [
                        {"id": "htl_abs", "left": "htl", "right": "absorber", "v_n": 20.0, "v_p": 30.0},
                        {"id": "abs_etl", "left": "absorber", "right": "etl", "v_n": 10.0, "v_p": 25.0},
                    ],
                ),
            ),
        ),
    ],
)
def test_deduplicated_modes_bitwise_parity_and_directional_fd(
    base_prepared: PreparedDevice, mode_name: str, mutate: Any
) -> None:
    prep = _edit_prepared(base_prepared, mutate)
    reg_mode = "compact_support" if mode_name == "thermionic_cap_and_srh" else "legacy_logistic"
    dev = compile_device_1d(prep, intervals_per_layer=8, te_regularization_mode=reg_mode)
    assert dev.interface_model is not None
    assert "heterointerface_1d" in dev.capabilities

    for voltage_V, gen_scale in ((0.0, 0.0), (0.65, 1.0), (-0.35, 0.0)):
        u = _perturbed_trial_state(dev, voltage_V=voltage_V)
        _assert_dc_assembled_bitwise_equal(dev, u, voltage_V=voltage_V, generation_scale=gen_scale)
        prob = AssembledDC1DProblem(dev, voltage_V=voltage_V, generation_scale=gen_scale)
        _assert_directional_jacobian_fd(prob, u)


def test_bare_srh_areal_rate_deposition_not_confused_with_volumetric(
    base_prepared: PreparedDevice,
) -> None:
    """Verify that doubling SRV changes only the shared interface cell residual by exactly R_areal (m^-2 s^-1)."""
    prep1 = _edit_prepared(
        base_prepared,
        lambda doc: doc.__setitem__(
            "interfaces",
            [
                {"id": "htl_abs", "left": "htl", "right": "absorber", "v_n": 12.0, "v_p": 18.0},
                {"id": "abs_etl", "left": "absorber", "right": "etl", "v_n": 0.0, "v_p": 0.0},
            ],
        ),
    )
    prep2 = _edit_prepared(
        base_prepared,
        lambda doc: doc.__setitem__(
            "interfaces",
            [
                {"id": "htl_abs", "left": "htl", "right": "absorber", "v_n": 24.0, "v_p": 36.0},
                {"id": "abs_etl", "left": "absorber", "right": "etl", "v_n": 0.0, "v_p": 0.0},
            ],
        ),
    )
    d0 = compile_device_1d(base_prepared, intervals_per_layer=8)
    d1 = compile_device_1d(prep1, intervals_per_layer=8)
    d2 = compile_device_1d(prep2, intervals_per_layer=8)

    # Evaluate at the converged zero-SRV steady state where f0[n + idx] ~ 0,
    # avoiding catastrophic cancellation from O(1e20) unrelaxed bulk fluxes.
    u = solve_steady_state(d0, voltage_V=0.6, generation_scale=1.0).state.u
    f0 = AssembledDC1DProblem(d0, voltage_V=0.6, generation_scale=1.0).residual(u)
    f1 = AssembledDC1DProblem(d1, voltage_V=0.6, generation_scale=1.0).residual(u)
    f2 = AssembledDC1DProblem(d2, voltage_V=0.6, generation_scale=1.0).residual(u)

    n = d0.node_count
    idx = int(d0.geometry.interfaces[0].shared_node)  # type: ignore[arg-type]
    diff1 = f1 - f0
    diff2 = f2 - f0
    # Continuity equations on the Assembler carry integrated areal residual [m^-2 s^-1]:
    # CELL_SOURCE multiplies (-R_areal / dx_cell[idx]) by dx_cell[idx] == -R_areal,
    # independent of mesh spacing dx_cell[idx]!
    _, n_u, p_u = d1.fields(u, voltage_V=0.6)
    assert d1.interface_model is not None
    spec1 = d1.interface_model.bare_srh[0]
    expected_srh = interface_srh_rate(
        float(n_u[idx]),
        float(p_u[idx]),
        v_n_m_s=spec1.v_n_m_s,
        v_p_m_s=spec1.v_p_m_s,
        ni_sq_m6=spec1.ni_sq_m6,
        n1_m3=spec1.n1_m3,
        p1_m3=spec1.p1_m3,
    )
    assert abs(diff1[n + idx]) > 1.0e5
    assert diff1[n + idx] == pytest.approx(-expected_srh.rate_m2_s, rel=1e-14)
    np.testing.assert_allclose(diff1[n + idx], diff1[2 * n + idx], rtol=1e-14)
    np.testing.assert_allclose(diff2[n + idx], 2.0 * diff1[n + idx], rtol=1e-14)

    mask = np.ones(3 * n, dtype=bool)
    mask[[n + idx, 2 * n + idx]] = False
    np.testing.assert_array_equal(diff1[mask], 0.0)


def test_invariant1_thermionic_cap_replaces_not_parallels_sg_face_flux(
    base_prepared: PreparedDevice,
) -> None:
    """When 1D TE capping activates on an internal heterointerface face, |J_face| <= |J_SG|."""
    prep_te = _edit_prepared(
        base_prepared,
        lambda doc: (
            # Tiny Richardson constant so legacy unnormalized TE bound (A* T^2 n / q) strongly caps SG.
            doc["layers"][0]["parameters"].update({"A_star_n": 1.0e-45, "A_star_p": 1.0e-45}),
            doc["layers"][1]["parameters"].update({"A_star_n": 1.0e-45, "A_star_p": 1.0e-45}),
        ),
    )
    dev_sg = compile_device_1d(base_prepared, intervals_per_layer=8)
    dev_hard = compile_device_1d(prep_te, intervals_per_layer=8, te_softness=0.0)
    dev_soft = compile_device_1d(prep_te, intervals_per_layer=8, te_softness=0.03)
    assert dev_hard.interface_model is not None
    assert len(dev_hard.interface_model.thermionic_faces) == 2
    assert any(s.cap_electron for s in dev_hard.interface_model.thermionic_faces)
    assert any(s.cap_hole for s in dev_hard.interface_model.thermionic_faces)

    u = _perturbed_trial_state(dev_sg, voltage_V=0.75)
    fl_sg = AssembledDC1DProblem(dev_sg, voltage_V=0.75, generation_scale=1.0).fluxes(u)
    fl_hard = AssembledDC1DProblem(dev_hard, voltage_V=0.75, generation_scale=1.0).fluxes(u)
    fl_soft = AssembledDC1DProblem(dev_soft, voltage_V=0.75, generation_scale=1.0).fluxes(u)

    for te_spec in dev_hard.interface_model.thermionic_faces:
        f = te_spec.face
        # Replacement caps the magnitude strictly below uncapped SG; parallel addition would increase it!
        if te_spec.cap_electron:
            assert abs(fl_hard.electron.flux[f]) < 0.05 * abs(fl_sg.electron.flux[f])
            assert abs(fl_soft.electron.flux[f]) < 0.05 * abs(fl_sg.electron.flux[f])
        if te_spec.cap_hole:
            assert abs(fl_hard.hole.flux[f]) < 0.05 * abs(fl_sg.hole.flux[f])
            assert abs(fl_soft.hole.flux[f]) < 0.05 * abs(fl_sg.hole.flux[f])

    # Non-interface faces remain 100% identical to SG!
    non_iface = np.ones(dev_sg.node_count - 1, dtype=bool)
    for te_spec in dev_hard.interface_model.thermionic_faces:
        non_iface[te_spec.face] = False
    np.testing.assert_array_equal(fl_hard.electron.flux[non_iface], fl_sg.electron.flux[non_iface])
    np.testing.assert_array_equal(fl_hard.hole.flux[non_iface], fl_sg.hole.flux[non_iface])


def _enable_all_layer_richardson(doc: dict[str, Any]) -> None:
    for layer in doc["layers"]:
        layer.setdefault("parameters", {}).update({"A_star_n": 1.2017e6, "A_star_p": 1.2017e6})


def test_two_sided_uncharged_and_charged_closures_equilibrium_gauss_jump_and_single_eval(
    base_prepared: PreparedDevice,
) -> None:
    prep_uncharged = _edit_prepared(
        base_prepared,
        lambda doc: (
            _enable_all_layer_richardson(doc),
            doc.__setitem__(
                "interfaces",
                [
                    {
                        "id": "htl_abs",
                        "left": "htl",
                        "right": "absorber",
                        "defect": _defect_dict("d_htl", trap_depth_eV=0.58, total_density_m2=4.0e15),
                    },
                    {
                        "id": "abs_etl",
                        "left": "absorber",
                        "right": "etl",
                        "defect": _defect_dict("d_etl", trap_depth_eV=0.52, total_density_m2=3.0e15),
                    },
                ],
            ),
        ),
    )
    prep_charged = _edit_prepared(
        prep_uncharged,
        lambda doc: doc["settings"].update(
            {
                "interface_charge_closure": "equilibrium_referenced",
                "interface_charge_rebaseline_acknowledged": True,
            }
        ),
    )

    dev_dedup = compile_device_1d(prep_uncharged, intervals_per_layer=8)
    dev_ts_uncharged = compile_device_1d(
        prep_uncharged, intervals_per_layer=8, interface_topology="two_sided_trace"
    )
    dev_ts_charged = compile_device_1d(prep_charged, intervals_per_layer=8)

    # Two shared interface nodes were removed in two_sided_trace!
    assert dev_ts_uncharged.node_count == dev_dedup.node_count - 2
    assert dev_ts_charged.node_count == dev_dedup.node_count - 2
    assert dev_ts_charged.interface_model is not None
    assert dev_ts_charged.interface_model.has_sheet_charge

    # 1. Dark equilibrium solve on charged two-sided trace: zero net current and zero incremental sheet charge!
    dark_charged = solve_equilibrium(dev_ts_charged)
    assert dark_charged.converged
    np.testing.assert_allclose(dark_charged.current_density_A_m2, 0.0, atol=1e-10)
    phi_eq, n_eq, p_eq = (
        dark_charged.state.potential_V,
        dark_charged.state.electron_density_m3,
        dark_charged.state.hole_density_m3,
    )
    u_eq = dark_charged.state.u
    n_cnt = dev_ts_charged.node_count
    zn_eq = u_eq[n_cnt:2 * n_cnt]
    zp_eq = u_eq[2 * n_cnt:]
    for spec in dev_ts_charged.interface_model.two_sided:
        assert spec.charge is not None
        qss_eq = solve_charged_two_sided_interface_qss(
            spec.trace_electrostatics,
            spec.physics,
            TwoSidedBulkState(
                float(phi_eq[spec.left_node]),
                float(phi_eq[spec.right_node]),
                float(n_eq[spec.left_node]),
                float(p_eq[spec.left_node]),
                float(n_eq[spec.right_node]),
                float(p_eq[spec.right_node]),
                electron_electrochemical_step=float(zn_eq[spec.right_node] - zn_eq[spec.left_node]),
                hole_electrochemical_step=float(-(zp_eq[spec.right_node] - zp_eq[spec.left_node])),
            ),
            spec.charge,
        )
        assert qss_eq.incremental_sheet_charge_C_m2 == pytest.approx(0.0, abs=1e-15)

    # 2. Biased/illuminated state: bitwise parity DC1DProblem == AssembledDC1DProblem, FD Jacobian,
    #    Gauss sheet-charge row deposition, and single evaluate_state call per assemble()!
    for dev in (dev_ts_uncharged, dev_ts_charged):
        u = _perturbed_trial_state(dev, voltage_V=0.55)
        _assert_dc_assembled_bitwise_equal(dev, u, voltage_V=0.55, generation_scale=1.0)
        prob = AssembledDC1DProblem(dev, voltage_V=0.55, generation_scale=1.0)
        _assert_directional_jacobian_fd(prob, u, step=1.0e-5, rtol=5.0e-6)

    u_ch = _perturbed_trial_state(dev_ts_charged, voltage_V=0.62)
    prob_ch = AssembledDC1DProblem(dev_ts_charged, voltage_V=0.62, generation_scale=1.0)
    prob_un = AssembledDC1DProblem(dev_ts_uncharged, voltage_V=0.62, generation_scale=1.0)

    # Verify single constitutive evaluation per trial state in AssembledDC1DProblem!
    with mock.patch.object(
        CompiledInterfaceModel1D,
        "evaluate_state",
        wraps=dev_ts_charged.interface_model.evaluate_state,
    ) as spy:
        _ = prob_ch.residual(u_ch)
        assert spy.call_count == 1
        _ = prob_ch.jacobian(u_ch)
        assert spy.call_count == 2

    # Verify Poisson residual difference between charged and uncharged at the SAME (u_ch)
    # equals the exact sheet-charge row weights w_L * sigma, w_R * sigma!
    f_ch = prob_ch.residual(u_ch)
    f_un = prob_un.residual(u_ch)
    phi_c, n_c, p_c = dev_ts_charged.fields(u_ch, voltage_V=0.62)
    expected_poisson_diff = np.zeros(dev_ts_charged.node_count)
    for spec in dev_ts_charged.interface_model.two_sided:
        assert spec.charge is not None
        qss = solve_charged_two_sided_interface_qss(
            spec.trace_electrostatics,
            spec.physics,
            TwoSidedBulkState(
                float(phi_c[spec.left_node]),
                float(phi_c[spec.right_node]),
                float(n_c[spec.left_node]),
                float(p_c[spec.left_node]),
                float(n_c[spec.right_node]),
                float(p_c[spec.right_node]),
                electron_electrochemical_step=float(
                    u_ch[dev_ts_charged.node_count + spec.right_node]
                    - u_ch[dev_ts_charged.node_count + spec.left_node]
                ),
                hole_electrochemical_step=float(
                    -(
                        u_ch[2 * dev_ts_charged.node_count + spec.right_node]
                        - u_ch[2 * dev_ts_charged.node_count + spec.left_node]
                    )
                ),
            ),
            spec.charge,
        )
        wl, wr = sheet_charge_row_weights(
            spec.trace_electrostatics.eps_r_left,
            spec.trace_electrostatics.left_distance_m,
            spec.trace_electrostatics.eps_r_right,
            spec.trace_electrostatics.right_distance_m,
        )
        # Note: Assembler negates CELL_SOURCE into F = div(flux) - source, so source = -w * sigma -> F += +w * sigma!
        expected_poisson_diff[spec.left_node] += wl * qss.incremental_sheet_charge_C_m2
        expected_poisson_diff[spec.right_node] += wr * qss.incremental_sheet_charge_C_m2

    assert np.max(np.abs(expected_poisson_diff)) > 1.0e-6
    np.testing.assert_allclose(
        f_ch[: dev_ts_charged.node_count] - f_un[: dev_ts_charged.node_count],
        expected_poisson_diff,
        rtol=1e-13,
        atol=1e-18,
    )


def test_left_right_stack_reversal_symmetry_on_charged_two_sided_device(
    base_prepared: PreparedDevice,
) -> None:
    """Reversing HTL/absorber/ETL -> ETL/absorber/HTL (zero optics) gives J_rev(V) == -J_fwd(V)."""
    prep_fwd = _edit_prepared(
        base_prepared,
        lambda doc: (
            _enable_all_layer_richardson(doc),
            doc["settings"].update(
                {
                    "Phi": 0.0,
                    "interface_charge_closure": "equilibrium_referenced",
                    "interface_charge_rebaseline_acknowledged": True,
                }
            ),
            doc.__setitem__(
                "interfaces",
                [
                    {
                        "id": "htl_abs",
                        "left": "htl",
                        "right": "absorber",
                        "defect": _defect_dict("d_htl", trap_depth_eV=0.58, total_density_m2=2.0e15),
                    },
                    {
                        "id": "abs_etl",
                        "left": "absorber",
                        "right": "etl",
                        "defect": _defect_dict("d_etl", trap_depth_eV=0.52, total_density_m2=3.0e15),
                    },
                ],
            ),
        ),
    )
    prep_rev = _edit_prepared(
        prep_fwd,
        lambda doc: (
            doc.__setitem__("layers", list(reversed(doc["layers"]))),
            doc.__setitem__("electrical_grid", list(reversed(doc["electrical_grid"]))),
            doc.__setitem__(
                "interfaces",
                [
                    {
                        "id": "etl_abs",
                        "left": "etl",
                        "right": "absorber",
                        "defect": _defect_dict("d_etl", trap_depth_eV=0.52, total_density_m2=3.0e15),
                    },
                    {
                        "id": "abs_htl",
                        "left": "absorber",
                        "right": "htl",
                        "defect": _defect_dict("d_htl", trap_depth_eV=0.58, total_density_m2=2.0e15),
                    },
                ],
            ),
        ),
    )

    dev_fwd = compile_device_1d(prep_fwd, intervals_per_layer=(8, 12, 8))
    dev_rev = compile_device_1d(prep_rev, intervals_per_layer=(8, 12, 8))
    assert dev_fwd.voltage_polarity == -dev_rev.voltage_polarity

    # Same forward diode bias V_diode = 0.45 V under each stack's polarity convention!
    sym_opts = SteadyOptions(max_bias_step_V=0.025)
    sol_fwd = solve_steady_state(dev_fwd, voltage_V=0.45, generation_scale=0.0, options=sym_opts)
    sol_rev = solve_steady_state(dev_rev, voltage_V=0.45, generation_scale=0.0, options=sym_opts)
    assert sol_fwd.converged and sol_rev.converged

    j_out_fwd = float(-dev_fwd.voltage_polarity * sol_fwd.contact_current_density_A_m2[0])
    j_out_rev = float(-dev_rev.voltage_polarity * sol_rev.contact_current_density_A_m2[0])
    assert abs(j_out_fwd) > 1.0e-7
    assert j_out_rev == pytest.approx(j_out_fwd, rel=1e-9)
    np.testing.assert_allclose(
        sol_rev.state.electron_density_m3,
        sol_fwd.state.electron_density_m3[::-1],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        sol_rev.state.hole_density_m3,
        sol_fwd.state.hole_density_m3[::-1],
        rtol=1e-9,
    )


@pytest.mark.parametrize(
    ("case_id", "mutate", "compile_kwargs", "match_substr"),
    [
        (
            "shared_occupancy",
            lambda doc: doc["settings"].__setitem__("interface_shared_occupancy", True),
            {},
            "interface_shared_occupancy",
        ),
        (
            "two_sided_additive_1d",
            lambda doc: doc["settings"].__setitem__("interface_two_sided", True),
            {},
            "interface_two_sided",
        ),
        (
            "te_physical_norm_p08_03",
            lambda doc: doc["layers"][0]["parameters"].__setitem__("A_star_n", 1.0e5)
            or doc["settings"].__setitem__("te_physical_norm", True),
            {},
            "te_physical_norm",
        ),
        (
            "interface_tunneling",
            lambda doc: doc["settings"].__setitem__("interface_tunneling", True),
            {},
            "interface_tunneling",
        ),
        (
            "het_recomb_despike",
            lambda doc: doc["settings"].__setitem__("het_recomb_despike", 0.05),
            {},
            "het_recomb_despike",
        ),
        (
            "charge_closure_unacknowledged",
            lambda doc: doc["settings"].update(
                {
                    "interface_charge_closure": "equilibrium_referenced",
                    "interface_charge_rebaseline_acknowledged": False,
                }
            ),
            {},
            "rebaseline",
        ),
        (
            "charge_closure_on_deduplicated",
            lambda doc: (
                _enable_all_layer_richardson(doc),
                doc["settings"].update(
                    {
                        "interface_charge_closure": "equilibrium_referenced",
                        "interface_charge_rebaseline_acknowledged": True,
                    }
                ),
                doc.__setitem__(
                    "interfaces",
                    [
                        {
                            "id": "htl_abs",
                            "left": "htl",
                            "right": "absorber",
                            "defect": _defect_dict("d1"),
                        },
                        {
                            "id": "abs_etl",
                            "left": "absorber",
                            "right": "etl",
                            "defect": _defect_dict("d2"),
                        },
                    ],
                ),
            ),
            {"interface_topology": "deduplicated"},
            "two_sided_trace",
        ),
        (
            "charge_closure_with_nonunit_calibration",
            lambda doc: (
                _enable_all_layer_richardson(doc),
                doc["settings"].update(
                    {
                        "interface_charge_closure": "equilibrium_referenced",
                        "interface_charge_rebaseline_acknowledged": True,
                    }
                ),
                doc.__setitem__(
                    "interfaces",
                    [
                        {
                            "id": "htl_abs",
                            "left": "htl",
                            "right": "absorber",
                            "defect": _defect_dict("d1", calibration_factor=0.5),
                        },
                        {
                            "id": "abs_etl",
                            "left": "absorber",
                            "right": "etl",
                            "defect": _defect_dict("d2"),
                        },
                    ],
                ),
            ),
            {},
            "calibration factors",
        ),
        (
            "plane_generation_without_closure",
            lambda doc: doc["settings"].__setitem__("interface_plane_generation", True),
            {},
            "interface_plane_closure",
        ),
        (
            "plane_projection_and_closure_together",
            lambda doc: doc["settings"].update(
                {"interface_plane_projection": True, "interface_plane_closure": True}
            ),
            {},
            "mutually exclusive",
        ),
        (
            "plane_closure_without_defect",
            lambda doc: doc["settings"].__setitem__("interface_plane_closure", True),
            {},
            "requires at least one interface defect",
        ),
    ],
)
def test_invariant3_refused_interface_combinations_raise_capability_error(
    base_prepared: PreparedDevice,
    case_id: str,
    mutate: Any,
    compile_kwargs: dict[str, Any],
    match_substr: str,
) -> None:
    with pytest.raises((CapabilityError, ValueError), match=match_substr):
        prep = _edit_prepared(base_prepared, mutate)
        compile_device_1d(prep, intervals_per_layer=6, **compile_kwargs)


def test_backward_euler_and_small_signal_ac_on_assembled_charged_interface(
    base_prepared: PreparedDevice,
) -> None:
    prep = _edit_prepared(
        base_prepared,
        lambda doc: (
            _enable_all_layer_richardson(doc),
            doc["settings"].update(
                {
                    "interface_charge_closure": "equilibrium_referenced",
                    "interface_charge_rebaseline_acknowledged": True,
                }
            ),
            doc.__setitem__(
                "interfaces",
                [
                    {
                        "id": "htl_abs",
                        "left": "htl",
                        "right": "absorber",
                        "defect": _defect_dict("d_htl", trap_depth_eV=0.58, total_density_m2=3.0e15),
                    },
                    {
                        "id": "abs_etl",
                        "left": "absorber",
                        "right": "etl",
                        "defect": _defect_dict("d_etl", trap_depth_eV=0.52, total_density_m2=3.0e15),
                    },
                ],
            ),
        ),
    )
    dev = compile_device_1d(prep, intervals_per_layer=8)
    steady = solve_steady_state(
        dev,
        voltage_V=0.55,
        generation_scale=1.0,
        options=SteadyOptions(max_bias_step_V=0.05),
    )
    assert steady.converged

    # 1. Backward Euler step on AssembledDC1DProblem: verify BE Jacobian matches directional FD!
    endpoint = AssembledDC1DProblem(dev, voltage_V=0.60, generation_scale=1.0)
    be_step = BackwardEuler1DStep(endpoint=endpoint, previous=steady.state, dt_s=1.0e-7)  # type: ignore[arg-type]
    u_trial = be_step.initial_coordinates(voltage_lift=True)
    j_be = be_step.jacobian(u_trial)
    v = np.random.default_rng(99).standard_normal(u_trial.size)
    v /= np.linalg.norm(v)
    h = 1.0e-5
    fd_be = (be_step.residual(u_trial + h * v) - be_step.residual(u_trial - h * v)) / (2.0 * h)
    jv_be = j_be @ v
    row_scale = float(np.max(np.linalg.norm(j_be.toarray(), axis=1)))
    np.testing.assert_allclose(jv_be, fd_be, rtol=5.0e-6, atol=1.0e-7 * row_scale)

    # 2. Small-signal AC solve (J + j * omega * M) * du = rhs across 1 Hz .. 1 MHz!
    op = AssembledDC1DProblem(dev, voltage_V=0.55, generation_scale=1.0)
    u0 = steady.state.u
    j0 = op.jacobian(u0)
    k0 = op.storage_jacobian(u0)
    n_nodes = dev.node_count
    v_ac = complex(0.01, 0.0)
    phi_right = -dev.voltage_polarity * v_ac
    rhs_ac = np.zeros(3 * n_nodes, dtype=np.complex128)
    rhs_ac[n_nodes - 1] = phi_right
    fixed_ac: dict[int, complex] = {0: 0.0j, n_nodes - 1: phi_right}
    for offset in (n_nodes, 2 * n_nodes):
        fixed_ac[offset] = 0.0j
        fixed_ac[offset + n_nodes - 1] = -phi_right / dev.thermal_voltage_V

    freqs = (1.0, 1.0e2, 1.0e4, 1.0e6)
    ac_solutions = solve_frequency_systems(j0, k0, rhs_ac, freqs, fixed_coordinates=fixed_ac)
    assert len(ac_solutions) == len(freqs)
    for sol in ac_solutions:
        assert np.all(np.isfinite(sol.solution))
        assert np.linalg.norm(sol.solution) > 0.0
        assert sol.componentwise_backward_error <= 1.0e-11


def test_multilayer_ionic_and_multiple_ionic_interface_assembly_parity(
    base_prepared: PreparedDevice,
) -> None:
    """Verify Ionic1DProblem == AssembledIonic1DProblem + BE + AC and 2-domain MultipleIonic parity."""
    prep_ion = _edit_prepared(
        base_prepared,
        lambda doc: (
            doc["layers"][0]["parameters"].update({"A_star_n": 2.5e5, "A_star_p": 3.0e5}),
            doc["layers"][1]["parameters"].update(
                {
                    "A_star_n": 2.0e5,
                    "A_star_p": 2.5e5,
                    "P0": 1.0e23,
                    "P_lim": 1.0e26,
                    "D_ion": 1.0e-15,
                }
            ),
            doc.__setitem__(
                "interfaces",
                [
                    {
                        "id": "htl_abs",
                        "left": "htl",
                        "right": "absorber",
                        "defect": _defect_dict("d_htl", trap_depth_eV=0.60),
                    },
                    {"id": "abs_etl", "left": "absorber", "right": "etl", "v_n": 18.0, "v_p": 32.0},
                ],
            ),
        ),
    )
    ion_dev = compile_ionic_device_1d(
        prep_ion,
        intervals_per_layer=6,
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    assert ion_dev.electronic_component.interface_model is not None

    n_e = ion_dev.node_count
    u_e = _perturbed_trial_state(ion_dev.electronic_component, voltage_V=0.45)
    u_ion = np.r_[u_e, 0.05 * np.cos(np.linspace(0.0, math.pi, ion_dev.ion_node_count))]

    orig_ion = Ionic1DProblem(ion_dev, voltage_V=0.45, generation_scale=1.0)
    assm_ion = AssembledIonic1DProblem(ion_dev, voltage_V=0.45, generation_scale=1.0)
    assert orig_ion.residual(u_ion).tobytes() == assm_ion.residual(u_ion).tobytes()
    assert _canonical_csc_bytes(orig_ion.jacobian(u_ion)) == _canonical_csc_bytes(
        assm_ion.jacobian(u_ion)
    )

    # Directional FD check on AssembledIonic1DProblem Jacobian!
    j_ion = assm_ion.jacobian(u_ion)
    rng = np.random.default_rng(314)
    v = rng.standard_normal(u_ion.size)
    v /= np.linalg.norm(v)
    h = 1.0e-5
    fd_ion = (assm_ion.residual(u_ion + h * v) - assm_ion.residual(u_ion - h * v)) / (2.0 * h)
    row_scale = float(np.max(np.linalg.norm(j_ion.toarray(), axis=1)))
    np.testing.assert_allclose(j_ion @ v, fd_ion, rtol=5.0e-6, atol=1.0e-7 * row_scale)

    # AssembledIonicBackwardEuler1DStep == IonicBackwardEuler1DStep + AssembledIonicSmallSignal1D!
    prev_state = ion_dev.state(u_ion, voltage_V=0.45)
    be_orig = IonicBackwardEuler1DStep(orig_ion, prev_state, dt_s=1.0e-4)
    be_ion = AssembledIonicBackwardEuler1DStep(assm_ion, prev_state, dt_s=1.0e-4)
    assert be_orig.residual(u_ion).tobytes() == be_ion.residual(u_ion).tobytes()
    assert _canonical_csc_bytes(be_orig.jacobian(u_ion)) == _canonical_csc_bytes(be_ion.jacobian(u_ion))
    assert np.all(np.isfinite(be_ion.residual(u_ion)))
    assert np.all(np.isfinite(be_ion.jacobian(u_ion).data))

    ac_orig = IonicSmallSignal1D(orig_ion, prev_state, "fixed_reference_v1")
    ac_ion = AssembledIonicSmallSignal1D(assm_ion, prev_state, "fixed_reference_v1")
    phasor = complex(0.01, 0.0)
    sys_orig = ac_orig.system(1.0e3, voltage_phasor_V=phasor)
    sys_assm = ac_ion.system(1.0e3, voltage_phasor_V=phasor)
    assert _canonical_csc_bytes(sys_orig.jacobian) == _canonical_csc_bytes(sys_assm.jacobian)
    assert sys_orig.rhs.tobytes() == sys_assm.rhs.tobytes()
    ac_res = solve_frequency_systems(
        sys_assm.jacobian,
        sys_assm.storage_jacobian,
        sys_assm.rhs,
        [1.0e3],
        fixed_coordinates=sys_assm.fixed_coordinates,
        rhs_rate=sys_assm.rhs_rate,
    )[0]
    obs_orig = ac_orig.observe(ac_res.solution, 1.0e3, voltage_phasor_V=phasor)
    obs_assm = ac_ion.observe(ac_res.solution, 1.0e3, voltage_phasor_V=phasor)
    assert obs_assm["admittance_density_S_m2"] == obs_orig["admittance_density_S_m2"]
    adm = obs_assm["admittance_density_S_m2"]
    assert isinstance(adm, complex) and np.isfinite(adm.real) and np.isfinite(adm.imag)

    # Two-domain MultipleIonic1DProblem == AssembledMultipleIonic1DProblem with active interface SRH!
    prep_multi = _edit_prepared(
        base_prepared,
        lambda doc: (
            doc["layers"][0]["parameters"].update({"P0": 5.0e22, "P_lim": 1.0e26, "D_ion": 1.0e-16}),
            doc["layers"][2]["parameters"].update({"P0": 8.0e22, "P_lim": 1.0e26, "D_ion": 2.0e-16}),
            doc.__setitem__(
                "interfaces",
                [
                    {"id": "htl_abs", "left": "htl", "right": "absorber", "v_n": 15.0, "v_p": 25.0},
                    {"id": "abs_etl", "left": "absorber", "right": "etl", "v_n": 20.0, "v_p": 30.0},
                ],
            ),
        ),
    )
    multi_dev = compile_multiple_ionic_device_1d(
        prep_multi,
        intervals_per_layer=6,
        ion_domains=(("htl", ("blocking", "blocking")), ("etl", ("blocking", "blocking"))),
    )
    assert multi_dev.electronic_component.interface_model is not None
    u_multi = np.r_[
        _perturbed_trial_state(multi_dev.electronic_component, voltage_V=0.35),
        np.zeros(multi_dev.state_size - 3 * n_e),
    ]
    orig_m = MultipleIonic1DProblem(multi_dev, voltage_V=0.35, generation_scale=1.0)
    assm_m = AssembledMultipleIonic1DProblem(multi_dev, voltage_V=0.35, generation_scale=1.0)
    assert orig_m.residual(u_multi).tobytes() == assm_m.residual(u_multi).tobytes()
    assert _canonical_csc_bytes(orig_m.jacobian(u_multi)) == _canonical_csc_bytes(
        assm_m.jacobian(u_multi)
    )


def test_mixed_interfaces_and_interface_defects_defect_slot_wins(
    base_prepared: PreparedDevice,
) -> None:
    """When an interface slot carries both v_n=v_p=0 and an explicit defect, the defect wins."""
    prep = _edit_prepared(
        base_prepared,
        lambda doc: doc.__setitem__(
            "interfaces",
            [
                {"id": "htl_abs", "left": "htl", "right": "absorber", "v_n": 20.0, "v_p": 15.0},
                {
                    "id": "abs_etl",
                    "left": "absorber",
                    "right": "etl",
                    "v_n": 0.0,
                    "v_p": 0.0,
                    "defect": _defect_dict("d_etl", trap_depth_eV=0.58, total_density_m2=5.0e14),
                },
            ],
        ),
    )
    dev = compile_device_1d(prep, intervals_per_layer=6)
    assert dev.interface_model is not None
    assert len(dev.interface_model.bare_srh) == 1
    assert dev.interface_model.bare_srh[0].interface_index == 0
    assert dev.interface_model.bare_srh[0].v_n_m_s == pytest.approx(20.0)
    assert len(dev.interface_model.defect_srh) == 1
    assert dev.interface_model.defect_srh[0].interface_index == 1
    assert dev.interface_model.defect_srh[0].v_n_m_s > 0.0
    assert dev.interface_model.defect_srh[0].v_p_m_s > 0.0


"""T2 unit tests for 1D nonlocal equation assembly, WKB sparsity support, and AC gates."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_nonlocal_1d import (
    PHOTON_RECYCLING_OFF,
    compile_nonlocal_device_1d,
    solve_nonlocal_steady_state_1d,
)
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.nonlocal_1d import (
    AssembledNonlocalDC1DProblem,
    SparsitySupportChangedError,
    nonlocal_classical_dc_1d,
    solve_nonlocal_small_signal_1d,
)
from solarlab.equations.dc_1d import CapabilityError, DC1DProblem
from solarlab.equations.nonlocal_1d import NonlocalDC1DProblem, evaluate_nonlocal_couplings_1d
from solarlab.materials.resources import ResourceLibrary
from solarlab.physics.generation.radiative_reabsorption import (
    CONTROL_VOLUME_CONSERVATIVE_V1,
    HISTORICAL_TRAPEZOID_UNIFORM_V1,
    LAGGED_BAKED_GENERATION_V1,
    dense_reabsorption_jacobian,
    evaluate_radiative_reabsorption,
    thermionic_cap_directional_derivative,
)
from solarlab.physics.transport.field_mobility_tangents import (
    field_mobility_directional_derivative,
)
from solarlab.physics.transport.tunneling_channels import (
    PREALLOCATED_RESERVOIR_SUPERSET_V1,
    STRICT_WINDOW_REBIND_V1,
    TunnellingChannelDocument,
)
from solarlab.solvers.nonlinear.steady_1d import solve_steady_state


@pytest.fixture(scope="module", autouse=True)
def blas1() -> object:
    with threadpool_limits(limits=1):
        yield


def _csc_dense(mat: csc_matrix) -> np.ndarray:
    out = np.zeros(mat.shape, dtype=mat.dtype)
    for col in range(mat.shape[1]):
        start, stop = int(mat.indptr[col]), int(mat.indptr[col + 1])
        out[mat.indices[start:stop], col] = mat.data[start:stop]
    return out


def _central_fd_jacobian(
    problem: AssembledNonlocalDC1DProblem | NonlocalDC1DProblem,
    u0: np.ndarray,
    *,
    step: float = 1.0e-6,
) -> np.ndarray:
    n_dof = u0.size
    j_fd: NDArray[np.float64] = np.zeros((n_dof, n_dof), dtype=float)
    for col in range(n_dof):
        h = step * max(1.0, abs(float(u0[col])))
        up = u0.copy()
        um = u0.copy()
        up[col] += h
        um[col] -= h
        fp = problem.residual(up)
        fm = problem.residual(um)
        j_fd[:, col] = (fp - fm) / (2.0 * h)
    return j_fd


def _with_tunnelling_defaults(defaults: DefaultCatalog) -> DefaultCatalog:
    doc = TunnellingChannelDocument()
    channel_defaults = (
        (
            "cigs_graded_optics",
            (
                ("model", "minoura_2015"),
                ("slices", 12),
                ("kk_quadrature_order", 160),
            ),
        ),
        ("band_to_band", tuple(doc.band_to_band.to_dict().items())),
        ("intraband", tuple(doc.intraband.to_dict().items())),
        ("interface_defect_assisted", tuple(doc.interface_defect_assisted.to_dict().items())),
        ("contact", tuple(doc.contact.to_dict().items())),
    )
    return replace(defaults, complex_defaults=channel_defaults)


@pytest.fixture(scope="module")
def radiative_prepared() -> PreparedDevice:
    base = load_device()
    defaults = _with_tunnelling_defaults(base.defaults)
    doc: dict[str, Any] = base.to_input().editing_data()
    for layer in doc["layers"]:
        if layer["role"] == "absorber":
            layer["parameters"]["B_rad"] = 2.5e-15
            layer["parameters"]["tau_n"] = 5.0e-6
            layer["parameters"]["tau_p"] = 5.0e-6
    return resolve_device(
        DeviceInput.model_validate(doc),
        defaults,
        base.resources,
        sources=base.sources,
    )


@pytest.fixture(scope="module")
def wkb_prepared() -> PreparedDevice:
    base = load_device()
    defaults = _with_tunnelling_defaults(base.defaults)
    common_params: dict[str, Any] = {
        "eps_r": 20.0,
        "mu_n": 2.0e-3,
        "mu_p": 2.0e-3,
        "D_ion": 0.0,
        "P_lim": 1.0e30,
        "P0": 0.0,
        "ni": 2.2495700957e11,
        "tau_n": 1.0e-6,
        "tau_p": 1.0e-6,
        "n1": 2.2495700957e11,
        "p1": 2.2495700957e11,
        "B_rad": 0.0,
        "C_n": 0.0,
        "C_p": 0.0,
        "alpha": 0.0,
        "N_A": 0.0,
        "N_D": 1.0e22,
        "chi": 4.0,
        "Eg": 1.5,
        "Nc300": 1.0e24,
        "Nv300": 8.0e23,
    }
    barrier_params = {
        **common_params,
        "mu_n": 2.0e-5,
        "mu_p": 2.0e-5,
        "N_D": 0.0,
        "chi": 3.7,
        "Eg": 2.1,
        "ni": 2.0526804442e6,
        "n1": 2.0526804442e6,
        "p1": 2.0526804442e6,
    }
    right_params = {**common_params, "N_D": 2.0e22}
    doc: dict[str, Any] = {
        "schema_version": "solarlab.device-preparation.v1",
        "source_format": "canonical",
        "id": "wkb_resolved_electron_barrier_unit",
        "settings": {
            "Phi": 1.0e21,
            "mode": "legacy",
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
        },
        "tunnelling_channels": {
            "schema_version": "solarlab-wkb-tunnelling-channels-v1",
            "intraband": {
                "enabled": True,
                "electron_effective_mass_rel": 0.2,
                "hole_effective_mass_rel": 0.2,
                "carrier": "electron",
                "energy_quadrature_order": 96,
            },
        },
        "layers": [
            {
                "id": "left_reservoir",
                "name": "left_reservoir",
                "role": "ETL",
                "thickness": 150.0e-9,
                "parameters": common_params,
            },
            {
                "id": "electron_barrier",
                "name": "electron_barrier",
                "role": "ETL",
                "thickness": 3.0e-9,
                "parameters": barrier_params,
            },
            {
                "id": "right_reservoir",
                "name": "right_reservoir",
                "role": "ETL",
                "thickness": 100.0e-9,
                "parameters": right_params,
            },
        ],
    }
    return resolve_device(DeviceInput.model_validate(doc), defaults, ResourceLibrary(()))


# ---------------------------------------------------------------------------
# 1. Zero-coupling & all-disabled bitwise identity
# ---------------------------------------------------------------------------


def test_disabled_nonlocal_and_inert_tunnelling_are_bit_identical_to_classical(
    radiative_prepared: PreparedDevice,
) -> None:
    local_dev = compile_device_1d(radiative_prepared, intervals_per_layer=8)
    nonlocal_off = compile_nonlocal_device_1d(
        radiative_prepared, intervals_per_layer=8, photon_recycling_mode=PHOTON_RECYCLING_OFF
    )
    assert nonlocal_off.sha256 == local_dev.sha256
    assert nonlocal_classical_dc_1d(nonlocal_off).identity == AssembledDC1DProblem(local_dev).binding.identity

    # Also test an explicit all-disabled tunnelling_channels block in configuration!
    doc = radiative_prepared.to_input().editing_data()
    doc["tunnelling_channels"] = {
        "schema_version": "solarlab-wkb-tunnelling-channels-v1",
        "intraband": {"enabled": False, "carrier": "electron", "energy_quadrature_order": 16},
    }
    prep_inert = resolve_device(
        DeviceInput.model_validate(doc),
        radiative_prepared.defaults,
        radiative_prepared.resources,
        sources=radiative_prepared.sources,
    )
    nonlocal_inert = compile_nonlocal_device_1d(prep_inert, intervals_per_layer=8)
    assert nonlocal_inert.tunnelling_channels is None
    assert nonlocal_inert.has_nonlocal_couplings is False

    steady_local = solve_steady_state(local_dev, voltage_V=0.45, generation_scale=1.0)
    u = steady_local.state.u

    p_loc_asm = AssembledDC1DProblem(local_dev, 0.45, 1.0)
    p_nl_asm = AssembledNonlocalDC1DProblem(nonlocal_off, 0.45, 1.0)
    p_inert_asm = AssembledNonlocalDC1DProblem(nonlocal_inert, 0.45, 1.0)
    p_loc_dir = DC1DProblem(local_dev, 0.45, 1.0)
    p_nl_dir = NonlocalDC1DProblem(nonlocal_off, 0.45, 1.0)

    assert np.array_equal(p_nl_asm.residual(u), p_loc_asm.residual(u))
    assert np.array_equal(p_inert_asm.residual(u), p_loc_asm.residual(u))
    assert np.array_equal(p_nl_dir.residual(u), p_loc_dir.residual(u))

    j_loc = p_loc_asm.jacobian(u)
    j_nl = p_nl_asm.jacobian(u)
    j_inert = p_inert_asm.jacobian(u)
    assert np.array_equal(j_nl.data, j_loc.data)
    assert np.array_equal(j_inert.data, j_loc.data)
    assert np.array_equal(j_nl.indices, j_loc.indices)
    assert np.array_equal(j_nl.indptr, j_loc.indptr)

    m_loc = p_loc_asm.storage_jacobian(u)
    m_nl = p_nl_asm.storage_jacobian(u)
    assert np.array_equal(m_nl.data, m_loc.data)
    assert np.array_equal(m_nl.indices, m_loc.indices)
    assert np.array_equal(m_nl.indptr, m_loc.indptr)

    assert np.array_equal(p_nl_asm.current_density(u), p_loc_asm.current_density(u))
    assert np.array_equal(p_nl_asm.contact_current_density(u), p_loc_asm.contact_current_density(u))


# ---------------------------------------------------------------------------
# 2. Sparse rank-1 reabsorption Jacobian vs FD & dense reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,cross_region",
    [
        (CONTROL_VOLUME_CONSERVATIVE_V1, False),
        (CONTROL_VOLUME_CONSERVATIVE_V1, True),
        (HISTORICAL_TRAPEZOID_UNIFORM_V1, False),
    ],
)
def test_nonlocal_reabsorption_sparse_jacobian_matches_fd_and_rank1_reference(
    radiative_prepared: PreparedDevice,
    mode: str,
    cross_region: bool,
) -> None:
    abs_id = next(layer.id for layer in radiative_prepared.layers if layer.role == "absorber")
    # For cross_region=True, route reabsorbed photons into a distinct downstream node subset!
    overrides: dict[str, NDArray[np.intp]] | None = (
        {abs_id: np.arange(10, 20, dtype=np.intp)} if cross_region else None
    )
    dev = compile_nonlocal_device_1d(
        radiative_prepared,
        intervals_per_layer=(6, 10, 6),
        photon_recycling_mode=mode,
        explicit_escape_probabilities={abs_id: 0.12},
        receiver_node_overrides=overrides,
    )
    solved = solve_nonlocal_steady_state_1d(dev, voltage_V=0.55, generation_scale=1.0)
    u = solved.state.u

    prob_asm = AssembledNonlocalDC1DProblem(dev, 0.55, 1.0)
    prob_dir = NonlocalDC1DProblem(dev, 0.55, 1.0)
    np.testing.assert_allclose(prob_asm.residual(u), prob_dir.residual(u), rtol=0.0, atol=1e-22)

    j_asm = _csc_dense(prob_asm.jacobian(u))
    j_dir = _csc_dense(prob_dir.jacobian(u))
    np.testing.assert_allclose(j_asm, j_dir, rtol=1e-14, atol=1e-22)

    j_fd = _central_fd_jacobian(prob_asm, u, step=1.0e-6)
    rel_frob = np.linalg.norm(j_asm - j_fd) / np.linalg.norm(j_fd)
    assert rel_frob <= 1e-8

    # Also check the pure nonlocal Jacobian addition against dense_reabsorption_jacobian!
    ext = evaluate_nonlocal_couplings_1d(dev, u, voltage_V=0.55, derivatives=True)
    assert ext.jacobian_addition is not None
    j_add_sparse = _csc_dense(ext.jacobian_addition)

    count = dev.node_count
    _, n, p = dev.fields(u, voltage_V=0.55)
    ev = evaluate_radiative_reabsorption(dev.radiative_reabsorption, n, p, derivatives=True)
    dn_dense, dp_dense = dense_reabsorption_jacobian(ev, count)
    widths = dev.geometry.cell_measures_m
    j_add_ref: NDArray[np.float64] = np.zeros((3 * count, 3 * count), dtype=float)
    for row_offset in (count, 2 * count):
        # Rows 1..count-2 are unconstrained interior carrier rows under ohmic contacts.
        j_add_ref[row_offset + 1 : row_offset + count - 1, count : 2 * count] = (
            widths[1:-1, None] * dn_dense[1:-1, :] * n[None, :]
        )
        j_add_ref[row_offset + 1 : row_offset + count - 1, 2 * count : 3 * count] = (
            -widths[1:-1, None] * dp_dense[1:-1, :] * p[None, :]
        )
    rel_ref = np.linalg.norm(j_add_sparse - j_add_ref) / np.linalg.norm(j_add_ref)
    assert rel_ref <= 1e-14


# ---------------------------------------------------------------------------
# 3. WKB intraband sparse Jacobian & turning-point support-window shift test
# ---------------------------------------------------------------------------


def test_wkb_sparse_jacobian_and_support_window_shift_policies(
    wkb_prepared: PreparedDevice,
) -> None:
    counts = (8, 8, 8)
    dev_superset = compile_nonlocal_device_1d(
        wkb_prepared,
        intervals_per_layer=counts,
        wkb_sparsity_policy=PREALLOCATED_RESERVOIR_SUPERSET_V1,
    )
    dev_strict = compile_nonlocal_device_1d(
        wkb_prepared,
        intervals_per_layer=counts,
        wkb_sparsity_policy=STRICT_WINDOW_REBIND_V1,
    )

    s_low = solve_nonlocal_steady_state_1d(
        dev_superset, voltage_V=-0.02, generation_scale=0.0
    )
    s_high = solve_nonlocal_steady_state_1d(
        dev_superset, voltage_V=0.22, generation_scale=0.0
    )

    p_low = AssembledNonlocalDC1DProblem(dev_superset, -0.02, 0.0)
    p_high = AssembledNonlocalDC1DProblem(dev_superset, 0.22, 0.0)

    # Confirm the WKB turning-point support window actually shifts across grid cells!
    ext_low = evaluate_nonlocal_couplings_1d(
        dev_superset, s_low.state.u, voltage_V=-0.02, derivatives=True
    )
    ext_high = evaluate_nonlocal_couplings_1d(
        dev_superset, s_high.state.u, voltage_V=0.22, derivatives=True
    )
    assert ext_low.wkb_support is not None and ext_high.wkb_support is not None
    assert ext_low.wkb_support.support_signature != ext_high.wkb_support.support_signature
    assert ext_low.wkb_support.turning_point_face_pairs != ext_high.wkb_support.turning_point_face_pairs
    assert ext_low.wkb_support.contained_in_reservoir_superset is True
    assert ext_high.wkb_support.contained_in_reservoir_superset is True

    # Under PREALLOCATED_RESERVOIR_SUPERSET_V1, 0 nonzeros are dropped at BOTH windows!
    for prob, state_u in ((p_low, s_low.state.u), (p_high, s_high.state.u)):
        j_ana = _csc_dense(prob.jacobian(state_u))
        j_fd = _central_fd_jacobian(prob, state_u, step=1.0e-6)
        rel_err = np.linalg.norm(j_ana - j_fd) / np.linalg.norm(j_fd)
        assert rel_err <= 1e-8

    # Under STRICT_WINDOW_REBIND_V1, a shifted turning-point window raises SparsitySupportChangedError
    # until explicitly rebound via .rebind_for_state(u)!
    strict_prob = AssembledNonlocalDC1DProblem(dev_strict, 0.22, 0.0).rebind_for_state(
        dev_strict.rebase_coordinates(s_low.state.u, -0.02, 0.22)
    )
    with pytest.raises(SparsitySupportChangedError, match="wkb_support_shifted"):
        strict_prob.jacobian(s_high.state.u)

    rebound_prob = strict_prob.rebind_for_state(s_high.state.u)
    j_strict = _csc_dense(rebound_prob.jacobian(s_high.state.u))
    j_superset_high = _csc_dense(p_high.jacobian(s_high.state.u))
    np.testing.assert_allclose(j_strict, j_superset_high, rtol=1e-14, atol=0.0)

    # Full solve under STRICT_WINDOW_REBIND_V1 succeeds by rebinding across continuation stages!
    s_strict_solved = solve_nonlocal_steady_state_1d(
        dev_strict, voltage_V=0.22, generation_scale=0.0
    )
    np.testing.assert_allclose(
        s_strict_solved.current_density_A_m2, s_high.current_density_A_m2, rtol=1e-12, atol=1e-15
    )


# ---------------------------------------------------------------------------
# 4. Small-signal AC capability gates, Maxwell spread & DC conductance limit
# ---------------------------------------------------------------------------


def test_small_signal_ac_refuses_wkb_lagged_and_nonsmooth_surfaces_and_matches_dc_slope(
    radiative_prepared: PreparedDevice,
    wkb_prepared: PreparedDevice,
) -> None:
    # (a) WKB is refused in transient storage, nonzero alpha/udot, non-QF driver, and AC!
    with pytest.raises(CapabilityError, match="quasi_fermi_steady_state"):
        compile_nonlocal_device_1d(wkb_prepared, intervals_per_layer=6, driver="mol_transient")

    wkb_dev = compile_nonlocal_device_1d(wkb_prepared, intervals_per_layer=(8, 8, 8))
    wkb_eq = solve_nonlocal_steady_state_1d(wkb_dev, voltage_V=0.0, generation_scale=0.0)
    wkb_prob = AssembledNonlocalDC1DProblem(wkb_dev, 0.0, 0.0)

    with pytest.raises(CapabilityError, match="wkb_transient_unsupported"):
        wkb_prob.storage(wkb_eq.state.u)
    with pytest.raises(CapabilityError, match="wkb_transient_unsupported"):
        wkb_prob.storage_jacobian(wkb_eq.state.u)
    with pytest.raises(CapabilityError, match="wkb_transient_unsupported"):
        wkb_prob.jacobian(wkb_eq.state.u, alpha=1.0)
    with pytest.raises(CapabilityError, match="wkb_transient_unsupported"):
        wkb_prob.residual(wkb_eq.state.u, np.ones_like(wkb_eq.state.u))
    with pytest.raises(CapabilityError, match="wkb_ac_unsupported"):
        solve_nonlocal_small_signal_1d(wkb_prob, wkb_eq.state.u, [100.0])

    # (b) Radiative reabsorption at dark equilibrium (R_tot == 0 clamp surface) is refused in AC!
    abs_id = next(layer.id for layer in radiative_prepared.layers if layer.role == "absorber")
    reabs_dev = compile_nonlocal_device_1d(
        radiative_prepared,
        intervals_per_layer=8,
        photon_recycling_mode=CONTROL_VOLUME_CONSERVATIVE_V1,
        explicit_escape_probabilities={abs_id: 0.08},
    )
    reabs_dark = solve_nonlocal_steady_state_1d(reabs_dev, voltage_V=0.0, generation_scale=0.0)
    dark_prob = AssembledNonlocalDC1DProblem(reabs_dev, 0.0, 0.0)
    with pytest.raises(CapabilityError, match="non_smooth_ac_switching_surface"):
        solve_nonlocal_small_signal_1d(dark_prob, reabs_dark.state.u, [10.0])

    # (c) Lagged baked reabsorption is refused in AC!
    reabs_light = solve_nonlocal_steady_state_1d(reabs_dev, voltage_V=0.55, generation_scale=1.0)
    _, n_light, p_light = reabs_dev.fields(reabs_light.state.u, voltage_V=0.55)
    lagged_dev = compile_nonlocal_device_1d(
        radiative_prepared,
        intervals_per_layer=8,
        photon_recycling_mode=LAGGED_BAKED_GENERATION_V1,
        explicit_escape_probabilities={abs_id: 0.08},
        lagged_carrier_densities_m3=(n_light, p_light),
    )
    lagged_prob = AssembledNonlocalDC1DProblem(lagged_dev, 0.55, 1.0)
    with pytest.raises(CapabilityError, match="lagged_reabsorption_ac_unsupported"):
        solve_nonlocal_small_signal_1d(lagged_prob, reabs_light.state.u, [10.0])

    # (d) Active TE crossover & PF cusp surfaces passed to AC are refused!
    smooth_prob = AssembledNonlocalDC1DProblem(reabs_dev, 0.55, 1.0)
    te_kink = thermionic_cap_directional_derivative([5.0], [5.0], [1.0], [0.2])
    with pytest.raises(CapabilityError, match="thermionic_cap"):
        solve_nonlocal_small_signal_1d(
            smooth_prob, reabs_light.state.u, [10.0], thermionic_electron=te_kink
        )
    pf_cusp = field_mobility_directional_derivative([1e-4], [0.0], [1e4], [0.0], [2.0], [1e-4])
    with pytest.raises(CapabilityError, match="field_mobility"):
        solve_nonlocal_small_signal_1d(
            smooth_prob, reabs_light.state.u, [10.0], field_mobility_electron=pf_cusp
        )

    # (e) Smooth illuminated point passes AC, conserves total Maxwell current, and matches DC dJ/dV!
    freqs = [0.0, 10.0, 1.0e4, 1.0e7]
    ac = solve_nonlocal_small_signal_1d(smooth_prob, reabs_light.state.u, freqs)
    ac_scale = np.max(np.abs(ac.total_current_density_A_m2), axis=1)
    assert np.all(ac.maxwell_spread_A_m2_V <= 1.0e-8 + 1.0e-10 * ac_scale)

    dv = 1.0e-5
    s_plus = solve_nonlocal_steady_state_1d(
        reabs_dev, initial_state=reabs_light.state, voltage_V=0.55 + dv, generation_scale=1.0
    )
    s_minus = solve_nonlocal_steady_state_1d(
        reabs_dev, initial_state=reabs_light.state, voltage_V=0.55 - dv, generation_scale=1.0
    )
    j_plus = float(np.mean(s_plus.current_density_A_m2))
    j_minus = float(np.mean(s_minus.current_density_A_m2))
    g_dc_fd = reabs_dev.voltage_polarity * (j_plus - j_minus) / (2.0 * dv)
    assert float(np.real(ac.admittance_density_S_m2[0])) == pytest.approx(g_dc_fd, rel=5e-6)
    assert abs(float(np.imag(ac.admittance_density_S_m2[0]))) == 0.0

    # Confirm nonlocal reabsorption visibly alters AC admittance vs P_esc = 1.0 (no trapping)!
    no_trap_dev = compile_nonlocal_device_1d(
        radiative_prepared,
        intervals_per_layer=8,
        photon_recycling_mode=CONTROL_VOLUME_CONSERVATIVE_V1,
        explicit_escape_probabilities={abs_id: 1.0},
    )
    no_trap_light = solve_nonlocal_steady_state_1d(no_trap_dev, voltage_V=0.55, generation_scale=1.0)
    ac_no_trap = solve_nonlocal_small_signal_1d(
        AssembledNonlocalDC1DProblem(no_trap_dev, 0.55, 1.0), no_trap_light.state.u, [0.0]
    )
    assert abs(ac.admittance_density_S_m2[0] - ac_no_trap.admittance_density_S_m2[0]) > 1.0e-3 * abs(
        ac_no_trap.admittance_density_S_m2[0]
    )

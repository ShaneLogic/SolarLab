"""Preparation boundary tests; no DC optimizer or Poisson solve is executed.

The independent solver is replaced by an explicit synthetic result. These
tests check fixed-input ownership, binary64 projection and import routing.
Actual physical acceptance belongs to the separately bounded validation.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_baseline_initial as initial
from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as precision
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from perovskite_sim.physics.compensated import DD


@dataclass(frozen=True)
class _DC:
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    potential_V: np.ndarray
    electron_qf_increment_V: np.ndarray
    hole_qf_increment_V: np.ndarray
    positive_ion_density_m3: np.ndarray
    interface_occupancy: np.ndarray
    negative_ion_density_m3: object = None
    bulk_trap_occupancy: object = None
    certificate: str = "original_accepted_certificate"
    state_sha256: str = "original_fixed_coordinate_identity"


@pytest.fixture
def initial_case(monkeypatch):
    qf = SimpleNamespace(V_app=0.0, phi0=np.array([0.0, .1, .2, .5]),
        log_n0=np.array([.1, .2, .3, .4]), log_p0=np.array([.5, .6, .7, .8]))
    material = SimpleNamespace(has_dual_ions=False, ion_steric_diffusion_only=True,
        iface_qss_left_nodes=(1,), iface_qss_right_nodes=(2,),
        eps_r=np.array([2., 3., 4., 5.]), iface_qss_left_distances_m=(.2,),
        iface_qss_right_distances_m=(.3,), V_T_device=.25,
        poisson_factor=SimpleNamespace(C=np.ones(3), h_cell=np.ones(2)),
        N_D=np.arange(4.), N_A=np.arange(4.) * .5, P_ion0=np.ones(4))
    dc = _DC(np.array([7., 2., 3., 11.]), np.array([13., 5., 6., 17.]),
        np.array([0., .11, .22, .5]), np.array([0., .1, -.05, 0.]),
        np.array([0., -.1, .025, 0.]), np.array([1., 1.1, .9, 1.]), np.array([.3]))
    dynamic = SimpleNamespace(y=np.r_[dc.electron_density_m3, dc.hole_density_m3],
                              phi=dc.potential_V.copy(), interface_charge_dynamic=object())
    geometry = SimpleNamespace(fixed_sheet_charge_C_m2=0., potential_jump_right_minus_left_V=0.)
    monkeypatch.setattr(initial, "_material_two_sided_interface_problem",
                        lambda *args, **kwargs: (geometry, None, None))
    monkeypatch.setattr(initial, "poisson_right_boundary", lambda *args: .5)
    return SimpleNamespace(stack=object(), qf=qf, material=material, dc=dc,
                           dynamic=dynamic, f_ref=np.array([.25]), traps=np.array([3.]))


def _canonicalize(case):
    return initial.canonicalize_baseline_initial_dc(
        case.stack, case.material, case.qf, case.dc, case.dynamic,
        equilibrium_occupancy=case.f_ref, trap_density_m2=case.traps,
    )


def test_only_rounded_potential_reaches_original_binary64_population_map(initial_case, monkeypatch):
    case = initial_case
    root = DD(np.array([0., .125, .25, .5]), np.array([0., 2.**-58, -2.**-57, 0.]))
    calls = []

    def solve(inputs):
        calls.append(inputs)
        # Deliberately unusable fine populations prove that only phi is used.
        return {"phi_V": root, "n_m3": object(), "p_m3": object()}, {"converged": True}

    monkeypatch.setattr(precision, "solve_independent_poisson", solve)
    result = _canonicalize(case)
    expected_phi = root.to_float()
    dphi = expected_phi - case.qf.phi0
    expected_n = np.exp(case.qf.log_n0 + (case.dc.electron_qf_increment_V + dphi) / .25)
    expected_p = np.exp(case.qf.log_p0 + (case.dc.hole_qf_increment_V - dphi) / .25)
    expected_n[[0, -1]] = case.dc.electron_density_m3[[0, -1]]
    expected_p[[0, -1]] = case.dc.hole_density_m3[[0, -1]]
    assert len(calls) == 1
    for actual, expected, old in (
        (result.potential_V, expected_phi, case.dc.potential_V),
        (result.electron_density_m3, expected_n, case.dc.electron_density_m3),
        (result.hole_density_m3, expected_p, case.dc.hole_density_m3),
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64
        assert not np.shares_memory(actual, old)
    for field in ("electron_qf_increment_V", "hole_qf_increment_V",
                  "positive_ion_density_m3", "interface_occupancy", "certificate", "state_sha256"):
        assert getattr(result, field) is getattr(case.dc, field)
    assert not np.shares_memory(result.potential_V, root.hi)
    assert not np.shares_memory(result.potential_V, root.lo)


def test_solver_inputs_own_original_global_anchors_fixed_words_and_contacts(initial_case, monkeypatch):
    case, captured = initial_case, []

    def solve(inputs):
        captured.append(inputs)
        return {"phi_V": DD(case.dc.potential_V)}, {"converged": True}

    monkeypatch.setattr(precision, "solve_independent_poisson", solve)
    _canonicalize(case)
    inputs = captured[0]
    payload = inputs.to_dict()
    assert inputs.seed_provenance == "independent_legacy_fixed_qf_evaluation"
    for field, expected in (("phi0_V", case.qf.phi0), ("log_n0", case.qf.log_n0),
                            ("log_p0", case.qf.log_p0)):
        np.testing.assert_array_equal(payload["preparation"][field], expected)
    for field, expected in (("dqfn_V", case.dc.electron_qf_increment_V),
                            ("dqfp_V", case.dc.hole_qf_increment_V),
                            ("positive_m3", case.dc.positive_ion_density_m3),
                            ("occupancy", case.dc.interface_occupancy)):
        np.testing.assert_array_equal(payload["fixed_inputs"][field]["hi"], expected)
        np.testing.assert_array_equal(payload["fixed_inputs"][field]["lo"], np.zeros_like(expected))
    np.testing.assert_array_equal(payload["preparation"]["contact_n_m3"], [7., 11.])
    np.testing.assert_array_equal(payload["preparation"]["contact_p_m3"], [13., 17.])
    assert not {"n_m3", "p_m3", "positive_flux_m2_s", "positive_rate_m3_s"}.intersection(
        payload["fixed_inputs"])
    case.qf.phi0[:] = 99.
    case.dc.positive_ion_density_m3[:] = 99.
    case.dynamic.phi[:] = 99.
    assert inputs.to_dict() == payload


def test_initial_solver_failure_has_no_second_attempt_or_old_potential_fallback(initial_case, monkeypatch):
    calls = []

    def fail(inputs):
        calls.append(inputs)
        raise RuntimeError("bounded fixed-QF solver failed")

    monkeypatch.setattr(precision, "solve_independent_poisson", fail)
    with pytest.raises(RuntimeError, match="bounded fixed-QF solver failed"):
        _canonicalize(initial_case)
    assert len(calls) == 1


@pytest.mark.parametrize("canonicalize", [False, True])
def test_make_system_preserves_import_path_and_builds_local_state_once(initial_case, monkeypatch, canonicalize):
    case = initial_case
    microscopic = SimpleNamespace(trap_density_m2=case.traps, capture_velocities_m_s=(), document_sha256=())
    monkeypatch.setattr(common, "_research_charge_off_stack", lambda stack: (stack, microscopic))
    case.qf.evaluate_quasi_fermi_increments_defect_ion_combined = lambda *a, **k: case.dynamic
    monkeypatch.setattr(common, "_QuasiFermiSystem", lambda *a, **k: case.qf)
    monkeypatch.setattr(common, "_build_ion_layout", lambda material: object())
    monkeypatch.setattr(common, "InterfaceIonDarkReference", lambda *a: SimpleNamespace(dc=a[-1]))
    corrected = replace(case.dc, electron_density_m3=np.full(4, 23.),
                        hole_density_m3=np.full(4, 29.), potential_V=np.array([0., .2, .3, .5]))
    init_calls, constructor_calls = [], []

    def initialize(*args, **kwargs):
        init_calls.append((args, kwargs))
        return corrected

    def construct(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(initial, "canonicalize_baseline_initial_dc", initialize)
    monkeypatch.setattr(common, "ControlledPhysicalInterfaceIonSystem", construct)
    result = common._make_system(case.stack, np.arange(4.), case.material, case.dc,
        {"f_ref": case.f_ref}, object(), SimpleNamespace(site_occupancy_ceiling=.999),
        canonicalize=canonicalize)
    selected = corrected if canonicalize else case.dc
    assert len(init_calls) == int(canonicalize)
    assert len(constructor_calls) == 1
    args, _ = constructor_calls[0]
    assert args[3] is selected
    assert args[5].dc is selected
    assert args[6] is selected.interface_occupancy
    np.testing.assert_array_equal(args[7].y, np.r_[selected.electron_density_m3, selected.hole_density_m3])
    assert args[7].phi is selected.potential_V
    assert args[7].interface_charge_dynamic is case.dynamic.interface_charge_dynamic
    assert result.common_dc_state is selected

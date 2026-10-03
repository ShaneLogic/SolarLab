"""Stable transport from saved one-word inputs; no preparation or time solve."""
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments.interface_defect_ion_transient import _InterfaceIonTransientSystem
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import PhysicalInterfaceIonSystem
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import (
    ControlledPhysicalInterfaceIonSystem, R1DynamicsControls,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_input_lift import RebasedInputLiftR1System
from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_physics as independent
from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as precision
from perovskite_sim.physics.compensated import DD


@pytest.fixture
def baseline():
    owner = object.__new__(ControlledPhysicalInterfaceIonSystem)
    owner._controls = R1DynamicsControls()
    owner.grid = np.arange(4.)
    owner.widths = np.array([.5, 1., 1., .5])
    owner.node_count, owner.interface_count = 4, 0
    owner.interface_faces, owner.left_nodes, owner.right_nodes = (), (), ()
    owner.positive_nodes = np.arange(4)
    owner.negative_nodes = np.empty(0, dtype=int)
    owner.thermal_voltage, owner.polarity = .25, -1.
    owner.site_occupancy_ceiling = .999
    owner.material = SimpleNamespace(has_dual_ions=False, ion_steric_diffusion_only=True,
        ion_steric_shared_site=False, V_T_device=.25, P_lim_node=np.full(4, 100.),
        P_lim_face=np.full(3, 100.), D_ion_face=np.array([1., 0., 2.]),
        dx_cell=owner.widths, physical_cell_faces_m=np.array([0., .5, 1.5, 2.5, 3.]),
        iface_qss_interface_positions_m=(), chi=np.zeros(4), Eg=np.ones(4),
        D_n_face=np.ones(3), D_p_face=np.ones(3),
        poisson_factor=SimpleNamespace(C=np.ones(3)))
    owner.qfn_reference = owner.qfp_reference = np.zeros(4)
    owner.ion_layout = SimpleNamespace(positive_components=((0, 1, 2, 3),))
    owner.common_dc_state = SimpleNamespace(positive_ion_density_m3=np.ones(4))
    state = SimpleNamespace(phi=np.array([0., 2.**-60, -2.**-60, 0.]),
        positive=np.ones(4), negative=None, n=np.ones(4), p=np.ones(4),
        dqfn=np.zeros(4), dqfp=np.zeros(4), occupancy=np.empty(0), local=(),
        current_n=np.zeros(3), current_p=np.zeros(3), positive_flux=np.zeros(3))
    return owner, state


def _decimal(value):
    return Decimal.from_float(float(value))


def _decimal_transport(owner, state):
    """Independent high-precision laws applied to the exact saved float values."""
    with localcontext() as context:
        context.prec = 90
        population = list(map(_decimal, state.positive))
        phi = list(map(_decimal, state.phi))
        chemical = [-(1-p/_decimal(capacity)).ln() for p, capacity in
                    zip(population, owner.material.P_lim_node)]
        def bernoulli(x):
            return Decimal(1) if not x else x/(x.exp()-1)
        flux = []
        for k, diffusion in enumerate(owner.material.D_ion_face):
            drive = ((phi[k+1]-phi[k])/_decimal(owner.material.V_T_device)
                     + chemical[k+1]-chemical[k])
            flux.append(_decimal(diffusion)/_decimal(owner.grid[k+1]-owner.grid[k])*(
                bernoulli(drive)*population[k]-bernoulli(-drive)*population[k+1]))
        bounded = [Decimal(0), *flux, Decimal(0)]
        rate = [-(bounded[k+1]-bounded[k])/_decimal(width)
                for k, width in enumerate(owner.widths)]
        return np.array(list(map(float, flux))), np.array(list(map(float, rate)))


def test_small_actual_high_word_drive_survives_sg_cancellation(baseline):
    owner, state = baseline
    expected_flux, expected_rate = _decimal_transport(owner, state)
    rate, negative_rate, flux, negative_flux = owner._ion_fields(state.phi, state.positive, None)
    np.testing.assert_array_equal(flux, expected_flux)
    np.testing.assert_array_equal(rate, expected_rate)
    assert negative_rate is negative_flux is None
    assert flux[0] != 0. and flux[1] == 0. and flux[2] != 0.
    # The unmodified ancestor remains an explicit historical binary64 path.
    old = _InterfaceIonTransientSystem._ion_fields(owner, state.phi, state.positive, None)
    np.testing.assert_array_equal(old[2], 0.)
    assert np.any(flux != old[2])


def test_dd_is_temporary_and_rate_rounds_after_fv_difference(baseline, monkeypatch):
    owner, state = baseline
    poison = object()
    for name in ("_fine_work", "_fine_reference", "_input_lift_work", "_input_lift_reference",
                 "_step_reference", "system"):
        setattr(owner, name, poison)
    before = vars(owner).copy()
    input_phi, input_population = state.phi.copy(), state.positive.copy()
    synthetic = DD(np.ones(3), np.array([2.**-60, 0., -2.**-60]))
    def flux_from_actual_words(phi, positive, material, spacing):
        np.testing.assert_array_equal(phi.hi, input_phi)
        np.testing.assert_array_equal(positive.hi, input_population)
        np.testing.assert_array_equal(phi.lo, 0.)
        np.testing.assert_array_equal(positive.lo, 0.)
        assert material is owner.material
        np.testing.assert_array_equal(spacing, np.diff(owner.grid))
        return synthetic
    monkeypatch.setattr(precision, "ion_flux_pair", flux_from_actual_words)
    rate, _, flux, _ = owner._ion_fields(state.phi, state.positive, None)
    np.testing.assert_array_equal(flux, np.ones(3))
    np.testing.assert_array_equal(rate[1:3], [2.**-60, 2.**-60])
    assert all(type(value) is np.ndarray and value.dtype == np.float64 for value in (rate, flux))
    np.testing.assert_array_equal(state.phi, input_phi)
    np.testing.assert_array_equal(state.positive, input_population)
    assert vars(owner).keys() == before.keys()
    assert all(vars(owner)[name] is value for name, value in before.items())
    with pytest.raises(TypeError, match="implicit DD rounding"):
        owner._ion_fields(DD(state.phi), state.positive, None)


def test_disabled_transport_skips_the_stable_constitutive_evaluation(baseline, monkeypatch):
    owner, state = baseline
    owner._controls = R1DynamicsControls(nu_I=0)
    monkeypatch.setattr(precision, "ion_flux_pair", lambda *a, **k: pytest.fail("disabled ion call"))
    rate, negative_rate, flux, negative_flux = owner._ion_fields(state.phi, state.positive, None)
    np.testing.assert_array_equal(rate, 0.)
    np.testing.assert_array_equal(flux, 0.)
    assert negative_rate is negative_flux is None


@pytest.mark.parametrize("kind", [precision.CompensatedR1System, RebasedInputLiftR1System])
def test_subclass_initialization_keeps_the_original_super_path(baseline, monkeypatch, kind):
    baseline_owner, state = baseline
    owner = object.__new__(kind)
    owner.__dict__.update(baseline_owner.__dict__)
    owner._fine_work = {}
    owner._input_lift_work = None
    sentinel = object()
    monkeypatch.setattr(PhysicalInterfaceIonSystem, "_ion_fields", lambda *a: sentinel)
    monkeypatch.setattr(precision, "ion_flux_pair", lambda *a, **k: pytest.fail("baseline intercepted subclass"))
    assert owner._ion_fields(state.phi, state.positive, None) is sentinel


@pytest.mark.parametrize("feature", ["has_dual_ions", "ion_steric_diffusion_only"])
def test_other_constitutive_models_keep_original_super_path(baseline, monkeypatch, feature):
    owner, state = baseline
    setattr(owner.material, feature, not getattr(owner.material, feature))
    sentinel = object()
    monkeypatch.setattr(PhysicalInterfaceIonSystem, "_ion_fields", lambda *a: sentinel)
    assert owner._ion_fields(state.phi, state.positive, None) is sentinel


def test_original_site_ceiling_and_analytic_jacobian_are_retained(baseline, monkeypatch):
    owner, state = baseline
    owner.material.P_lim_node = np.ones(4)
    with pytest.raises(ValueError, match="pre-clipping bound"):
        owner._ion_fields(state.phi, np.full(4, .999), None)
    sentinel = object()
    monkeypatch.setattr(PhysicalInterfaceIonSystem, "_ion_jacobians", lambda *a: sentinel)
    assert owner._ion_jacobians(state.phi, state.positive, None) is sentinel


def test_saved_physical_reader_is_independent_and_detects_old_or_forged_flux(baseline, monkeypatch):
    owner, state = baseline
    _, _, state.positive_flux, _ = owner._ion_fields(state.phi, state.positive, None)
    state.current_n, state.current_p, independent_flux, _ = independent._currents(owner, state)
    np.testing.assert_array_equal(independent_flux, state.positive_flux)
    expected_flux, _ = _decimal_transport(owner, state)
    np.testing.assert_array_equal(independent_flux, expected_flux)
    def forbidden(*args, **kwargs):
        pytest.fail("saved independent reader reached a live or legacy production flux")
    monkeypatch.setattr(precision, "ion_flux_pair", forbidden)
    monkeypatch.setattr(independent, "ion_face_flux", forbidden)
    monkeypatch.setattr(owner, "_ion_fields", forbidden)
    row = independent.independent_physics_row(owner, state, None, 0.)
    assert row["passed"]
    assert row["assessment"]["reference_accuracy_qualified"] is None
    assert row["assessment"]["reference_accuracy_status"] == "not_assessed_shared_constitutive_laws"
    assert independent.STABLE_BASELINE_ION_DEPENDENCY in row["shared_constitutive_dependencies"]
    assert "ion_migration.ion_face_flux" not in row["shared_constitutive_dependencies"]
    state.positive_flux = np.zeros(3)
    reported = {"electron_current_A_m2": state.current_n,
                "hole_current_A_m2": state.current_p,
                "positive_ion_flux_m2_s": state.positive_flux}
    bad = independent.independent_physics_row(owner, state, None, 0., reported=reported)
    assert not bad["checks"]["positive_ion_flux_m2_s_state_content_matches"]["passed"]
    assert not bad["checks"]["positive_ion_flux_m2_s_reported_content_matches"]["passed"]
    assert not bad["passed"]


def test_saved_physical_reader_reassembles_geometry_instead_of_solver_widths(baseline, monkeypatch):
    owner, state = baseline
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_global_reference as reference
    actual = reference.evaluate_independent_saved_high_ion_transport
    physical_widths = owner.widths.copy()
    owner.widths = owner.widths*2
    def captured(**kwargs):
        np.testing.assert_array_equal(kwargs["control_volume_width_m"], physical_widths)
        return actual(**kwargs)
    monkeypatch.setattr(reference, "evaluate_independent_saved_high_ion_transport", captured)
    state.current_n, state.current_p, state.positive_flux, _ = independent._currents(owner, state)
    row = independent.independent_physics_row(owner, state, None, 0.)
    assert not row["checks"]["physical_geometry_matches"]["passed"]
    assert not row["passed"]

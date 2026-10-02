"""Analytic current identities and independently assembled failure witnesses."""
from copy import deepcopy
from types import SimpleNamespace as Namespace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_physics as independent


@pytest.fixture
def fixture():
    material=Namespace(has_dual_ions=False,iface_qss_interface_positions_m=(),
        poisson_factor=Namespace(C=np.ones(3)),chi=np.zeros(4),Eg=np.ones(4),
        D_n_face=np.ones(3),D_p_face=np.ones(3),D_ion_face=np.zeros(3),
        P_lim_face=np.full(3,2.),P_lim_node=np.full(4,2.),ion_steric_diffusion_only=True)
    system=Namespace(grid=np.arange(4.),material=material,thermal_voltage=1.,
        node_count=4,interface_count=0,interface_faces=(),left_nodes=(),right_nodes=(),
        qfn_reference=np.zeros(4),qfp_reference=np.zeros(4),eps_face=np.ones(3),
        controls=Namespace(nu_I=0,nu_t=1),widths=np.array([.5,1.,1.,.5]),polarity=1.,
        electron_slice=slice(0,2),hole_slice=slice(2,4),trap_slice=slice(4,4),
        positive_slice=slice(4,8),potential_slice=slice(8,10),local_slice=slice(10,10),
        positive_nodes=np.arange(4),trap_density=np.array([]),
        ion_layout=Namespace(positive_components=((0,1,2,3),)),
        common_dc_state=Namespace(positive_ion_density_m3=np.ones(4)))
    state=Namespace(coordinate=np.zeros(10),n=np.ones(4),p=np.ones(4),positive=np.ones(4),
        occupancy=np.array([]),phi=np.zeros(4),dqfn=np.zeros(4),dqfp=np.zeros(4),local=(),
        current_n=np.zeros(3),current_p=np.zeros(3),positive_flux=np.zeros(3))
    return system,state,deepcopy(state)


def test_equilibrium_and_forbidden_solver_shortcuts(fixture):
    system,state,previous=fixture
    def forbidden(*args,**kwargs):
        pytest.fail("independent reconstruction called a solver diagnostic")
    for name in ("transient_current_metrics","solver_current_metrics","storage_increment",
                 "potential_increment","charge_balance_metrics","interface_current_sides"):
        setattr(system,name,forbidden)
    report=independent.independent_physics_row(system,state,previous,1.)
    assert report["passed"]
    assert set(report)==independent.ROW_FIELDS
    assert set(report["metrics"])==set(independent.METRIC_LIMITS)
    assert report["metrics"]==dict.fromkeys(independent.METRIC_LIMITS,0.)


@pytest.mark.parametrize("gradient",[-1e-3,1e-3])
def test_constitutive_current_has_analytic_field_sign_and_detects_shared_sign_error(fixture,gradient):
    system,state,previous=fixture
    state.phi=gradient*system.grid
    state.dqfn=-state.phi
    state.dqfp=state.phi
    n,p,_,_=independent._currents(system,state)
    # Equal populations: B(x)-B(-x)=-x gives both currents exactly -q*dphi/dx.
    np.testing.assert_allclose(n,-Q*gradient,rtol=5e-15,atol=0.)
    np.testing.assert_allclose(p,-Q*gradient,rtol=5e-15,atol=0.)
    state.current_n,state.current_p=-n,-p
    reported={"electron_current_A_m2":-n,"hole_current_A_m2":-p,
              "positive_ion_flux_m2_s":np.zeros(3)}
    report=independent.independent_physics_row(system,state,None,0.,reported=reported)
    assert not report["checks"]["electron_current_A_m2_state_content_matches"]["passed"]
    assert not report["checks"]["electron_current_A_m2_reported_content_matches"]["passed"]
    assert not report["passed"]


def test_coordinated_solver_and_published_constant_current_is_not_independent_evidence(fixture):
    system,state,previous=fixture
    state.current_n=np.ones(3) # Uniform: the forged solver spread and report both say zero.
    reported={"electron_current_A_m2":state.current_n,"hole_current_A_m2":state.current_p,
              "positive_ion_flux_m2_s":state.positive_flux}
    report=independent.independent_physics_row(system,state,None,0.,reported=reported)
    assert report["metrics"]["internal_face_current_spread_relative"]==0.
    assert report["checks"]["electron_current_A_m2_reported_content_matches"]["passed"] is False
    assert report["passed"] is False


def test_contact_control_volume_error_can_fail_with_perfect_internal_agreement(fixture):
    system,state,previous=fixture
    previous.positive*=1e22
    state.positive=previous.positive.copy()
    system.common_dc_state.positive_ion_density_m3=previous.positive.copy()
    state.coordinate[4]=1e-5 # Change only the physical left endpoint ion population.
    state.positive[0]*=np.exp(1e-5)
    report=independent.independent_physics_row(system,state,previous,1.)
    assert report["checks"]["internal_face_current_spread_relative"]["passed"]
    assert report["checks"]["contact_internal_current_spread_relative"]["passed"] is False
    assert report["checks"]["charge_balance_normalized"]["passed"] is False


def test_inventory_is_recomputed_from_original_population_and_geometry(fixture):
    system,state,previous=fixture
    state.positive*=1.001
    system.positive_targets=np.array([3.003]) # Forged solver inventory target is ignored.
    report=independent.independent_physics_row(system,state,None,0.)
    assert report["checks"]["inventory_relative_drift"]["passed"] is False
    assert report["arrays"]["initial_positive_inventory_m2"]==[3.]


def test_geometry_is_not_taken_from_solver_widths(fixture):
    system,state,previous=fixture
    system.widths*=2
    report=independent.independent_physics_row(system,state,None,0.)
    assert report["checks"]["physical_geometry_matches"]["passed"] is False
    assert report["arrays"]["positive_inventory_m2"]==[3.]


def test_interface_guard_can_fail_when_internal_and_contact_guards_pass(fixture,monkeypatch):
    system,state,previous=fixture
    system.interface_count=1
    system.interface_faces=(1,)
    system.left_nodes,system.right_nodes=(1,),(2,)
    system.local_slice=slice(10,16)
    system.material.eps_r=np.ones(4)
    system.material.iface_qss_left_distances_m=(.5,)
    system.material.iface_qss_right_distances_m=(.5,)
    state.coordinate=np.zeros(16);previous.coordinate=np.zeros(16)
    state.current_n=np.ones(3)
    monkeypatch.setattr(independent,"_currents",lambda *args:(np.ones(3),np.zeros(3),np.zeros(3),np.array([[1.,2.]])))
    report=independent.independent_physics_row(system,state,previous,1.)
    assert report["checks"]["internal_face_current_spread_relative"]["passed"]
    assert report["checks"]["contact_internal_current_spread_relative"]["passed"]
    assert report["checks"]["interface_current_spread_relative"]["passed"] is False
    assert report["reasons"]==["interface_current_spread_relative"]


def test_internal_face_guard_is_evaluated_not_inferred_from_solver_flag(fixture,monkeypatch):
    system,state,previous=fixture
    state.current_n=np.array([1.,1.01,1.])
    monkeypatch.setattr(independent,"_currents",lambda *args:(state.current_n,np.zeros(3),np.zeros(3),np.empty((0,2))))
    report=independent.independent_physics_row(system,state,previous,1.)
    assert report["checks"]["internal_face_current_spread_relative"]["passed"] is False
    assert report["checks"]["interface_current_spread_relative"]["passed"]


def test_zero_plus_is_not_a_finite_step_current_certificate(fixture):
    system,state,previous=fixture
    report=independent.independent_physics_row(system,state,None,0.)
    for key in independent.FINITE_STEP_METRICS:
        assert report["checks"][key]=={"applicable":False,"passed":None}
    with pytest.raises(ValueError,match="positive dt"):
        independent.independent_physics_row(system,state,None,1.)

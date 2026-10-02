"""Real-system ownership, cache invalidation and same-call evidence checks."""

import copy
from dataclasses import replace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts.check_r1_v7_precision_consumers import prepare_real_system
from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as precision
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from perovskite_sim.physics.compensated import DD


@pytest.fixture(scope="module")
def real_system():
    with threadpool_limits(1):
        system,state,_,_=prepare_real_system()
    return system,state


def same_words(left,right):
    assert set(left)==set(right)
    for name in left:
        np.testing.assert_array_equal(left[name].hi,right[name].hi,err_msg=name)
        np.testing.assert_array_equal(left[name].lo,right[name].lo,err_msg=name)


def test_repeated_interleaved_rebase_voltage_and_fault_evaluations(real_system):
    system,initial=real_system
    owner,previous=system.rebase(initial)
    zero=np.zeros(owner.dimension)
    first=owner.evaluate(zero,0.)
    coordinate=zero.copy()
    coordinate[owner.potential_slice.start+3]=1e-5
    coordinate[owner.positive_slice.start+2]=-1e-5
    other=owner.evaluate(coordinate,0.)
    assert other.fine is not first.fine
    same_words(first.fine,owner.evaluate(zero,0.).fine)
    owner.precision_fault="diffusion"
    faulty=owner.evaluate(coordinate,0.)
    assert np.any(faulty.fine["positive_flux_m2_s"].hi != other.fine["positive_flux_m2_s"].hi)
    owner.precision_fault="none"
    same_words(other.fine,owner.evaluate(coordinate,0.).fine)
    lifted,local=owner.rebase(first)
    lifted.set_voltage_lift(.005,local)
    lifted.evaluate(zero,.005)
    same_words(first.fine,owner.evaluate(zero,0.).fine)
    assert "ion" not in first.fine and "chemical" not in first.fine
    assert owner._fine_evaluation_cache["fine"] is owner._fine_work


def test_mutating_a_material_array_invalidates_frozen_wrappers(real_system):
    system,initial=real_system
    owner,_=system.rebase(initial)
    original=np.asarray(owner.material.D_ion_face).copy()
    owner.material=replace(owner.material,D_ion_face=original.copy())
    initial_constant=owner._refresh_fine_constants()["ion_diffusion"]
    owner.material.D_ion_face[:]*=1.01
    changed_constant=owner._refresh_fine_constants()["ion_diffusion"]
    assert changed_constant is not initial_constant
    np.testing.assert_array_equal(initial_constant.hi,original)
    np.testing.assert_array_equal(changed_constant.hi,original*1.01)


def test_reference_quantization_keeps_the_actual_physical_previous(real_system):
    system,initial=real_system
    owner=copy.copy(system)
    fine=dict(initial.fine)
    fine["positive_m3"]=precision.put(fine["positive_m3"],7,fine["positive_m3"][7]+DD(1000.))
    seeded,_=system.rebase(replace(initial,fine=fine))
    # Recompute storage, sheet and currents before treating this complete
    # represented operator state as the physical previous for the unit test.
    changed=seeded.evaluate(np.zeros(seeded.dimension),0.)
    before={k:precision.pair_words(v) for k,v in changed.fine.items()}
    normal,local=owner.rebase(changed)
    assert normal.rebase_evidence is None
    same_words(normal._fine_reference,{k:fine[k] for k in precision.REFERENCE_FIELDS})
    owner.enable_reference_quantization()
    working,local=owner.rebase(changed)
    assert working._step_reference is local
    assert local.fine is changed.fine
    assert {k:precision.pair_words(v) for k,v in changed.fine.items()}==before
    assert set(working._fine_reference)==set(precision.REFERENCE_FIELDS)
    for value in working._fine_reference.values():
        assert np.all(value.lo==0)
    evidence=working.rebase_evidence
    assert evidence["physical_previous_unchanged"]
    assert evidence["physical_previous_identity"]==evidence["reference_before_identity"]
    assert evidence["reference_after_identity"]!=evidence["physical_previous_identity"]
    evaluated=working.evaluate(np.zeros(working.dimension),0.)
    expected=evaluated.fine["storage"]-local.fine["storage"]
    np.testing.assert_array_equal(working.storage_increment(evaluated,local),expected.hi)
    assert np.any(expected.hi!=0)
    assert "storage" not in working._fine_reference
    assert "sheet_charge_C_m2" not in working._fine_reference


def test_eliminated_evidence_uses_one_old_seed_and_actual_final_fields(real_system,monkeypatch):
    system,initial=real_system
    owner,_=system.rebase(initial)
    state=owner.evaluate(np.zeros(owner.dimension),0.)
    original=owner.system.evaluate_quasi_fermi_increments_defect_ion_combined
    calls=[]
    def captured(*args,**kwargs):
        result=original(*args,**kwargs)
        calls.append(result.phi.copy())
        return result
    monkeypatch.setattr(owner.system,"evaluate_quasi_fermi_increments_defect_ion_combined",captured)
    diagnostics=owner.eliminated_operator_diagnostics(state,0.)
    assert len(calls)==1
    assert all("relative_error" in value for value in diagnostics.values())
    evidence=diagnostics.precision_evidence
    assert evidence["state_identity"]==precision.fine_identity(state.fine)
    assert evidence["solve"]["used_direct_phi"] is False
    assert evidence["solve"]["converged"] is True
    assert 1<=evidence["solve"]["iterations"]<=12
    assert evidence["solve"]["last_correction_max_abs_V"]<1e-28
    exported={k:DD(v["hi"],v["lo"]) for k,v in evidence["fields"].items()}
    recomputed=owner._poisson_pair(exported["constraint_phi_V"],exported["n_m3"],exported["p_m3"],
                                    exported["positive_m3"],exported["sheet_charge_C_m2"])
    np.testing.assert_array_equal(exported["poisson_residual_C_m2"].hi,recomputed.hi)
    np.testing.assert_array_equal(exported["poisson_residual_C_m2"].lo,recomputed.lo)


def test_actual_initial_context_has_upstream_inputs_and_applied_coordinate(real_system):
    system,initial=real_system
    with threadpool_limits(1),precision.precision_context():
        # Resolve after the context patches existing module aliases.
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step as build
        result=build(system,initial,.005,policy=r1_policy(.1))
    context=result.event["precision_arithmetic_context"]
    assert context["inputs_derived_from_final_state"] is False
    np.testing.assert_array_equal(context["qf_anchors"]["log_n0"],system.system.log_n0)
    assert context["trace_state_arithmetic"]["kind"]=="raw_binary64_local_solver_input"
    plus=context["zero_plus_context"]
    assert plus["trace_state_arithmetic"]["kind"]=="recorded_log_update"
    assert plus["actual_update"]["fine_state_identity"]==context["zero_plus_state_identity"]
    assert len(plus["actual_update"]["coordinate"])==system.dimension
    assert np.any(np.asarray(plus["actual_update"]["voltage_lift_V"]["hi"])!=0)
    assert result.system.rebase_evidence is None

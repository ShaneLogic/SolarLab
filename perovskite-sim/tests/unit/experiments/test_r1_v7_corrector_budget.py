"""A corrector restart must not reset an original per-step search budget."""
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import CompensatedR1System, _precision_solve_step
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy


def test_corrector_restarts_receive_only_remaining_nonmonotone_budget():
    system=object.__new__(CompensatedR1System)
    system.interface_count=0
    system.precision_constraint_corrections=0
    system.storage_scale=lambda *a:np.ones(1)
    system.poisson_scale=lambda *a:np.ones(1)
    system.local_algebraic_scale=lambda *a:np.empty(0)
    calls=[]
    def original(owner,coordinate,previous,voltage,dt,policy,**kw):
        calls.append((policy.maximum_newton_iterations,policy.maximum_near_acceptance_nonmonotone_steps))
        nonmonotone=1 if len(calls)<3 else 0
        return SimpleNamespace(coordinate=np.asarray(coordinate).copy()),1,.01,0.,2,nonmonotone
    system.eliminated_operator_diagnostics=lambda *a:{
        key:{"relative_error":0. if len(calls)==3 else 1.}
        for key in ("positive_ion_flux","positive_ion_rate")}
    system.residual_and_jacobian=lambda coordinate,*a:(np.array([0.,1.]),sparse.eye(2,format="csr"),SimpleNamespace(coordinate=coordinate))
    previous=SimpleNamespace(storage=np.ones(1))
    result=_precision_solve_step(original,system,np.ones(2),previous,.005,1.,r1_policy(.1),check_jacobian=False)
    assert [budget for _,budget in calls]==[2,1,0]
    assert calls[0][0]>calls[1][0]>calls[2][0]
    assert result[5]==2
    assert system.precision_constraint_corrections==2

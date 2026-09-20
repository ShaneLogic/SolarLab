"""Synthetic trust-boundary/role controls; not actual scientific evidence."""
from copy import deepcopy
from decimal import Decimal
import pytest

from scripts import verify_r1_v9_precision as verify


def sample():
    pair = lambda value: {"hi":[0.,value,0.],"lo":[0.,0.,0.]}
    fields = {name:pair(0.) for name in (
        "phi_V","dqfn_V","dqfp_V","n_m3","p_m3","positive_m3",
        "occupancy","trace_potential_V","trace_state_m3","sheet_charge_C_m2")}
    other = deepcopy(fields)
    other.update(phi_V=pair(.1),constraint_phi_V=pair(.1),poisson_residual_C_m2=pair(0.),
                 positive_flux_m2_s=pair(0.),positive_rate_m3_s=pair(0.),boundary_flux_m2_s=pair(0.))
    return {"state":{"fields":fields},"physics_reconstruction":{"eliminated_precision":{"fields":other}}}


@pytest.fixture
def fake_arithmetic(monkeypatch):
    monkeypatch.setattr(verify,"verify_v8_sides",lambda *a,**k:{"qualified":True,
        "direct":{"qualified":True},"eliminated":{"qualified":True},"upstream_construction":None,
        "solve_provenance_passed":True,"input_identity_passed":True,"shared_inputs_passed":True})


def test_self_reported_json_is_not_actual_recomputation(fake_arithmetic):
    row=sample();context={"source":"synthetic"}
    for proof in (None,{"verified":True,"actual_recomputation":True,"binding":verify.record_binding(row)}):
        result=verify.verify_two_sides({},row,context=context,recomputation=proof)
        assert result["constitutive_checks_passed"] and not result["independent_path_verified"]
        assert not result["qualified"]
    with pytest.raises(TypeError,match="actual recomputation"):
        verify.ActualRecomputation(verify.record_binding(row),context,"self_report")


@pytest.mark.parametrize("mutation",("phi_only","all_direct","low_word_bypass"))
def test_saved_copy_and_low_word_bypass_do_not_match_actual_witness(fake_arithmetic,mutation):
    row=sample();context={"source":"synthetic"}
    witness=verify.capture_actual_recomputation(lambda:deepcopy(row),context=context)
    assert verify.verify_two_sides({},row,context=context,recomputation=witness)["qualified"]
    fields=row["physics_reconstruction"]["eliminated_precision"]["fields"]
    fields["phi_V"]=deepcopy(row["state"]["fields"]["phi_V"])
    if mutation != "phi_only":
        fields["constraint_phi_V"]=deepcopy(fields["phi_V"])
    if mutation=="low_word_bypass":
        fields["phi_V"]["lo"][1]=1e-30
        fields["constraint_phi_V"]["lo"][1]=1e-30
    result=verify.verify_two_sides({},row,context=context,recomputation=witness)
    assert result["constitutive_checks_passed"] and not result["qualified"]


def test_actual_equal_solution_is_allowed_outside_fixed_v8_sentinel(fake_arithmetic):
    row=sample();fields=row["physics_reconstruction"]["eliminated_precision"]["fields"]
    fields["phi_V"]=deepcopy(row["state"]["fields"]["phi_V"])
    fields["constraint_phi_V"]=deepcopy(fields["phi_V"])
    context={"source":"synthetic_independent_equal_solution"}
    witness=verify.capture_actual_recomputation(lambda:deepcopy(row),context=context)
    assert verify.verify_two_sides({},row,context=context,recomputation=witness)["qualified"]
    assert not verify.verify_two_sides({},row,context={"source":"different"},recomputation=witness)["qualified"]


def test_fixed_v8_guard_does_not_add_a_minimum_separation(fake_arithmetic):
    row=sample();fields=row["physics_reconstruction"]["eliminated_precision"]["fields"]
    fields["phi_V"]={"hi":[0.,0.,0.],"lo":[0.,1e-34,0.]}
    fields["constraint_phi_V"]=deepcopy(fields["phi_V"])
    context={"source_commit":verify.V8_SOURCE,"request_sha256":verify.V8_REQUEST}
    witness=verify.capture_actual_recomputation(lambda:deepcopy(row),context=context)
    assert verify.verify_two_sides({},row,context=context,recomputation=witness,fixed_v8_case=True)["qualified"]


def test_floor_and_zero_specimens_keep_original_gate_but_limit_relative_claim():
    weak=verify.floor_capability([0.],[1e-12],floor=1.,limit=1e-10,unit="m^-2 s^-1",floor_source="original")
    assert weak["original_gate_passed"] and weak["reference_relative_error"]=="1"
    assert not weak["relative_capability"] and weak["floor_active"]
    zero=verify.floor_capability([0.],[0.],floor=1.,limit=1e-10,unit="m^-2 s^-1",floor_source="original")
    assert zero["original_gate_passed"] and zero["strict_zero_observation"]
    assert zero["reference_relative_error"] is None and not zero["relative_capability"]
    signal=verify.floor_capability([2.00000000001],[2.],floor=1.,limit=1e-10,unit="m^-2 s^-1",floor_source="original")
    assert signal["relative_capability"] and signal["original_gate_passed"]
    assert Decimal(signal["reference_relative_bound_if_original_gate_passes"])>Decimal("1e-10")


def test_receipt_cannot_supply_its_own_trust_anchor():
    with pytest.raises(ValueError,match="external identity"):
        verify.bind_replay_receipt(b'{"actual_recomputation":true}',expected_sha256=None,context={},row_index=0)


def test_live_production_receipt_binds_v2_payload_and_json_copy_cannot_replace_it(fake_arithmetic):
    # The leaf issuer is used only to model a replay result in this unit test.
    # A real numerical receipt is exercised separately by integration checks.
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_replay_receipt import _issue_replay_receipt
    row=sample();row.update(substeps=1,time_s=0.)
    record=row['physics_reconstruction']['eliminated_precision'];f=record['fields']
    fixed={k:f[k] for k in ('dqfn_V','dqfp_V','positive_m3','occupancy','sheet_charge_C_m2')}
    record.update(schema='R1EliminatedPrecisionV2',shared_inputs={'fields':fixed},
        solve_inputs={'schema':'R1IndependentPoissonInputsV1','fixed_inputs':fixed,'preparation':{},'seed_phi_V':{},'seed_provenance':'synthetic'},
        ion_inputs={k:f[k] for k in ('phi_V','positive_m3','boundary_flux_m2_s')})
    solved={'phi_V':f['constraint_phi_V'],**{k:f[k] for k in ('n_m3','p_m3','poisson_residual_C_m2')}}
    record['solve']={'fixed_input_digest':verify.digest(fixed),'solve_inputs_digest':verify.digest(record['solve_inputs']),
        'solved_fields_digest':verify.digest(solved),'ion_input_digest':verify.digest(record['ion_inputs'])}
    binding={'row_index':0,'substeps':1,'time_s':0.,'direct_primary_digest':verify.state_identity(row['state']),
        **record['solve'],'record_digest':verify.digest(record)}
    context={'source_digest':'source','prepared_sha256':'prepared','result_sha256':'result','saved_rows_digest':verify.digest([row])}
    ledger={'schema':'R1ActualPhysicsReplayLedgerV1','representation':'float64-pair-v1',**context,
        'actual_recomputed_rows_digest':context['saved_rows_digest'],'row_bindings':[binding],
        'row_bindings_digest':verify.digest([binding])}
    receipt=_issue_replay_receipt(ledger)
    assert verify.verify_two_sides({},row,replay_receipt=receipt,row_index=0,context=context)['qualified']
    assert not verify.verify_two_sides({},row,replay_receipt=receipt.to_dict(),row_index=0,context=context)['qualified']
    f['phi_V']['lo'][1]=1e-30
    assert not verify.verify_two_sides({},row,replay_receipt=receipt,row_index=0,context=context)['qualified']


def test_state_dependent_rate_floor_is_not_misreported_as_a_noise_floor():
    report=verify.floor_capability([2.],[1.999999999999],floor=2.,limit=1e-6,unit='m^-3 s^-1',
                                  floor_source='max(direct_positive_rate_peak,1)')
    assert report['floor']=='2' and report['relative_capability_floor']=='1'
    assert report['relative_capability']

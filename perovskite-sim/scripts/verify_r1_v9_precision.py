"""V9 arithmetic, role, independent-recomputation and floor-capability checks.

No production numerical kernel is imported. A serialized solve flag is not
evidence of execution. Live witnesses must be made by an actual recomputation
callback; persisted replay witnesses require an externally supplied digest.
Neither is a replacement for source/request authorization at the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
from typing import Callable

from scripts.verify_r1_v7_precision import dec
from scripts.verify_r1_v8_precision import (
    arithmetic_comparison, canonical, digest, precision_fields, represented,
    state_identity, verify_two_sides as verify_v8_sides,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_replay_receipt import R1PhysicsReplayReceipt

V8_SOURCE = "8792c233b26f4debef3652c848930e6014957a25"
V8_REQUEST = "bce8c6d421e6e80d69149be5524d36c095aa93c412e63856839b193027a588d0"
V8_MANIFEST = "b35f23ad870cb32f1272e4b887dd708f6d6226ba1b923c55ade33421a78a917f"
_LIVE_TOKEN = object()


def record_binding(row):
    record = row["physics_reconstruction"]["eliminated_precision"]
    fields = record["fields"]
    shared = ("dqfn_V", "dqfp_V", "positive_m3", "occupancy", "sheet_charge_C_m2")
    solved = ("constraint_phi_V", "n_m3", "p_m3", "poisson_residual_C_m2")
    constitutive = ("phi_V", "positive_m3", "boundary_flux_m2_s")
    return {"direct_state_sha256": state_identity(row["state"]),
            "shared_inputs_sha256": digest({name: fields[name] for name in shared}),
            "solved_fields_sha256": digest({name: fields[name] for name in solved}),
            "constitutive_inputs_sha256": digest({name: fields[name] for name in constitutive}),
            "eliminated_record_sha256": digest(record)}


@dataclass(frozen=True, init=False)
class ActualRecomputation:
    """An in-process execution witness, not a decoder for user-supplied JSON."""
    bindings_json: str
    context_json: str
    origin: str

    def __init__(self, bindings, context, origin, *, _token=None):
        if _token is not _LIVE_TOKEN:
            raise TypeError("actual recomputation requires a live call or externally bound replay receipt")
        object.__setattr__(self, "bindings_json", canonical(bindings))
        object.__setattr__(self, "context_json", canonical(context))
        object.__setattr__(self, "origin", origin)


def capture_actual_recomputation(recompute: Callable, *, context):
    """Call the owner's independent reconstruction and bind its returned row.

    The trusted caller supplies an execution callback, never a callback that
    simply returns saved input. This function deliberately does not integrate.
    """
    actual_row = recompute()
    return ActualRecomputation(record_binding(actual_row), context, "actual_recomputation_call", _token=_LIVE_TOKEN)


def bind_replay_receipt(receipt_bytes, *, expected_sha256, context, row_index):
    """Inspect an externally anchored serialized ledger without granting trust.

    expected_sha256 must come from the validated producer/verification result,
    not from a self-reported hash within this receipt or the untrusted run.
    The returned mapping is not an ActualRecomputation/live production receipt.
    """
    if not expected_sha256 or hashlib.sha256(receipt_bytes).hexdigest() != expected_sha256:
        raise ValueError("actual replay receipt lacks its external identity binding")
    receipt = json.loads(receipt_bytes)
    if (receipt.get("schema") != "R1V9IndependentReplayBindingsV1"
            or receipt.get("context") != context or receipt.get("actual_recomputation") is not True):
        raise ValueError("actual replay receipt context or schema disagrees")
    entries = receipt.get("row_bindings", [])
    selected = [entry for entry in entries if entry.get("row") == row_index]
    if len(selected) != 1 or receipt.get("row_bindings_sha256") != digest(entries):
        raise ValueError("actual replay row is missing, ambiguous or has an invalid binding")
    return {"serialized_integrity_passed":True,"independent_path_verified":False,
            "binding":selected[0]["binding"],"scope":"serialized_ledger_is_not_a_live_replay_receipt"}


def archived_v8_witnesses(directory):
    """Reuse the externally pinned V8 complete replay, never create V9 proof."""
    from pathlib import Path
    from scripts.analyze_r1_v7_prototype import verify_manifest
    directory = Path(directory)
    if hashlib.sha256((directory/"ManifestV1.json").read_bytes()).hexdigest() != V8_MANIFEST:
        raise ValueError("historical V8 input differs from its externally frozen manifest")
    verify_manifest(directory)
    read = lambda name: json.loads((directory/name).read_text())
    summary, replay = read("SummaryV1.json"), read("PhysicsReplayV1.json")
    if (summary["source_commit"] != V8_SOURCE or summary["request_sha256"] != V8_REQUEST
            or not summary["four_predicates_passed"] or replay.get("checked_row_count") != 794
            or not all(replay.get(k) is True for k in
                       ("certified","complete","content_matches_recomputed","evidence_matches_equations"))):
        raise ValueError("historical V8 actual full replay evidence is incomplete")
    rows = [json.loads(line) for line in (directory/"AcceptedStepsV1.jsonl").read_text().splitlines()]
    context = {"source_commit":V8_SOURCE,"request_sha256":V8_REQUEST,"run_manifest_sha256":V8_MANIFEST,
               "historical_source":True,"scope":"previously_executed_immutable_V8_full_replay_not_V9_execution"}
    return context, [ActualRecomputation(record_binding(row),context,"externally_pinned_historical_V8_full_replay",_token=_LIVE_TOKEN)
                     for row in rows]


def floor_capability(actual, reference, *, floor, limit, unit, floor_source):
    """Describe the unchanged vector-peak gate and its relative claim scope."""
    with localcontext() as context:
        context.prec = 100
        a, b = represented(actual), represented(reference)
        if len(a) != len(b):
            raise ValueError("floor capability vector shapes disagree")
        f, tolerance = dec(floor), dec(limit)
        if f <= 0 or not 0 < tolerance < 1:
            raise ValueError("floor must be positive and tolerance must lie in (0,1)")
        error = max((abs(x-y) for x,y in zip(a,b)), default=Decimal(0))
        actual_peak = max(map(abs,a), default=Decimal(0))
        reference_peak = max(map(abs,b), default=Decimal(0))
        scale = max(f, actual_peak, reference_peak)
        allowance = tolerance*scale
        reference_relative = None if reference_peak == 0 else error/reference_peak
        # In this single-positive-ion lane the original rate floor is
        # max(actual direct rate peak,1), so it adds only a unit floor to
        # max(A,R). Do not mistake that state-dependent label for noise.
        capability_floor = Decimal(1) if floor_source == 'max(direct_positive_rate_peak,1)' else f
        relative_capability = reference_peak > 0 and reference_peak >= capability_floor
        relative_bound = (None if reference_peak == 0 else
                          tolerance*max(capability_floor,reference_peak)/((1-tolerance)*reference_peak))
        local = [{"index":i, "absolute_error":str(abs(x-y)), "reference_magnitude":str(abs(y)),
                  "reference_relative_error":None if y == 0 else str(abs(x-y)/abs(y))}
                 for i,(x,y) in enumerate(zip(a,b))]
        return {"absolute_error":str(error), "normalization_scale":str(scale),
                "floor":str(f), "floor_source":floor_source, "floor_active":max(actual_peak,reference_peak)<f,
                "actual_peak":str(actual_peak), "reference_peak":str(reference_peak),
                "relative_capability_floor":str(capability_floor),
                "original_limit":str(tolerance), "original_score":str(error/scale),
                "original_gate_passed":error <= allowance, "absolute_allowance":str(allowance), "unit":unit,
                "reference_relative_error":None if reference_relative is None else str(reference_relative),
                "relative_capability":relative_capability,
                "relative_capability_definition":"nonzero independent reference peak is at least the original floor",
                "reference_relative_bound_if_original_gate_passes":None if relative_bound is None else str(relative_bound),
                "bound_derivation":"E<=eps*max(floor,A,R), A<=R+E imply E/R<=eps*max(floor,R)/((1-eps)*R); no new acceptance gate",
                "signal_regime":"zero_reference" if reference_peak == 0 else (
                    "relative_sensitive" if relative_capability else "absolute_allowance_only"),
                "strict_zero_observation":reference_peak == 0 and actual_peak == 0,
                "local_errors":local, "scope":"unchanged_gate_capability_not_a_new_forward_error_bound"}


def role_check(record, *, path_transform=None):
    fields = record["fields"]
    if path_transform is None:
        return {"passed":canonical(fields["phi_V"]) == canonical(fields["constraint_phi_V"]),
                "kind":"healthy_identity", "scope":"actual_constitutive_phi_is_actual_constraint_solution"}
    if (path_transform.get("kind") != "add_single_potential" or type(path_transform.get("node")) is not int
            or path_transform.get("direction") not in (-1,1) or path_transform.get("amplitude_V") != 1e-13):
        raise ValueError("path transform must be the externally frozen single-node intervention")
    with localcontext() as context:
        context.prec = 100
        expected = represented(fields["constraint_phi_V"])
        node = path_transform["node"]
        if node != 7 or not 0 < node < len(expected)-1:
            raise ValueError("path transform does not match frozen interior node 7")
        expected[node] += dec(path_transform["direction"]*path_transform["amplitude_V"])
        result = arithmetic_comparison(fields["phi_V"], expected, voltage=True)
    return {**result,"kind":"externally_declared_single_node_transform"}


def verify_live_receipt(receipt,row,*,row_index,context):
    if not isinstance(receipt,R1PhysicsReplayReceipt):
        return False
    ledger=receipt.to_dict()
    required=('source_digest','prepared_sha256','result_sha256','saved_rows_digest')
    if (not isinstance(context,dict) or any(not context.get(k) or context[k]!=ledger.get(k) for k in required)
            or ledger.get('schema')!='R1ActualPhysicsReplayLedgerV1'
            or ledger.get('representation')!='float64-pair-v1'
            or ledger.get('saved_rows_digest')!=ledger.get('actual_recomputed_rows_digest')
            or ledger.get('row_bindings_digest')!=digest(ledger.get('row_bindings',[]))):
        return False
    entries=[x for x in ledger['row_bindings'] if x['row_index']==row_index]
    if len(entries)!=1:return False
    record=row['physics_reconstruction']['eliminated_precision'];fields=record['fields']
    if record.get('schema')!='R1EliminatedPrecisionV2':return False
    shared=('dqfn_V','dqfp_V','positive_m3','occupancy','sheet_charge_C_m2')
    solved={'phi_V':fields['constraint_phi_V'],**{k:fields[k] for k in ('n_m3','p_m3','poisson_residual_C_m2')}}
    expected={'row_index':row_index,'substeps':row['substeps'],'time_s':row['time_s'],
        'direct_primary_digest':state_identity(row['state']),
        'fixed_input_digest':digest({k:fields[k] for k in shared}),
        'solve_inputs_digest':digest(record['solve_inputs']),'solved_fields_digest':digest(solved),
        'ion_input_digest':digest(record['ion_inputs']),'record_digest':digest(record)}
    return entries[0]==expected


def v2_bindings(record):
    fields=record['fields'];solve=record['solve']
    names=('dqfn_V','dqfp_V','positive_m3','occupancy','sheet_charge_C_m2')
    expected_fixed={k:fields[k] for k in names}
    solved={'phi_V':fields['constraint_phi_V'],**{k:fields[k] for k in ('n_m3','p_m3','poisson_residual_C_m2')}}
    inputs=record.get('solve_inputs',{});ion=record.get('ion_inputs',{})
    return (record.get('shared_inputs',{}).get('fields')==expected_fixed
        and inputs.get('fixed_inputs')==expected_fixed
        and set(inputs)=={'schema','fixed_inputs','preparation','seed_phi_V','seed_provenance'}
        and solve.get('fixed_input_digest')==digest(expected_fixed)
        and solve.get('solve_inputs_digest')==digest(inputs)
        and solve.get('solved_fields_digest')==digest(solved)
        and solve.get('ion_input_digest')==digest(ion)
        and all(ion.get(k)==fields[k] for k in ('phi_V','positive_m3','boundary_flux_m2_s')))


def verify_two_sides(frozen, row, *, upstream=None, recomputation=None, context=None,
                     fixed_v8_case=False, path_transform=None,replay_receipt=None,row_index=None):
    """Separate constitutive arithmetic from independently evidenced execution."""
    record = row.get("physics_reconstruction", {}).get("eliminated_precision")
    arithmetic_row=row
    v2_consistent=True
    if isinstance(record,dict) and record.get('schema')=='R1EliminatedPrecisionV2':
        # V2 preserves the V1 arithmetic fields and adds execution bindings.
        # Adapt only the in-memory arithmetic view; never rewrite its artifact.
        v2_consistent=v2_bindings(record)
        shared_names=['dqfn_V','dqfp_V','positive_m3','occupancy','sheet_charge_C_m2']
        arithmetic_row={**row,'physics_reconstruction':{**row['physics_reconstruction'],
            'eliminated_precision':{**record,'schema':'R1EliminatedPrecisionV1',
                'shared_inputs':{**record['shared_inputs'],'fields':shared_names}}}}
    base = verify_v8_sides(frozen, arithmetic_row, upstream=upstream)
    roles = {"passed":False,"kind":"missing_record"} if record is None else role_check(record,path_transform=path_transform)
    duplicate = None
    if fixed_v8_case:
        if not context or context.get("source_commit") != V8_SOURCE or context.get("request_sha256") != V8_REQUEST:
            raise ValueError("the V8 noncopy sentinel requires its fixed historical source and request")
        duplicate = record is not None and canonical(record["fields"]["constraint_phi_V"]) == canonical(precision_fields(row["state"])["phi_V"])
    actual = isinstance(recomputation,ActualRecomputation)
    execution = (actual and context is not None and json.loads(recomputation.context_json)==context
                 and record_binding(row)==json.loads(recomputation.bindings_json))
    live_replay=verify_live_receipt(replay_receipt,row,row_index=row_index,context=context)
    execution=execution or live_replay
    provenance = all(base.get(name) is True for name in
                     ("solve_provenance_passed","input_identity_passed","shared_inputs_passed"))
    independent = bool(execution and roles["passed"] and duplicate is not True and provenance and v2_consistent)
    constitutive = all(base.get(side, {}).get("qualified") for side in ("direct","eliminated")
                       if base.get(side) is not None) and base.get("eliminated") is not None
    if base.get("upstream_construction") is not None:
        constitutive &= all(item["qualified"] for item in base["upstream_construction"].values())
    return {"schema":"R1V9TwoSideCheckV1", "constitutive_checks_passed":bool(constitutive),
            "independent_path_verified":independent, "qualified":independent and bool(constitutive),
            "independence_status":"verified_actual_recomputation" if independent else (
                "missing_actual_recomputation" if not actual and replay_receipt is None else "binding_or_role_mismatch"),
            "role_binding":roles, "fixed_V8_exact_copy_detected":duplicate,
            "v2_actual_input_digest_bindings":v2_consistent,
            "actual_recomputation_origin":"actual_production_physics_replay_receipt" if live_replay else recomputation.origin if actual else None,
            "source_request_context":context,"arithmetic":base,
            "physical_acceptance":"separate_original_certificate", "P2_qualified":False,
            "scope":"per_side_arithmetic_and_actual_independent_binding_not_full_P2"}

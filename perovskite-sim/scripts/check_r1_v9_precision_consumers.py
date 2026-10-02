"""Frozen V9 near-threshold and shared-input operator campaigns.

Only identified saved states are evaluated. No time integration is performed.
The shared-input specimen is intentionally outside the original request and
must pass constitutive arithmetic while failing that request's material bind.
"""
from __future__ import annotations

import argparse
from copy import deepcopy, copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.analyze_r1_v7_prototype import sha, write, verify_manifest
from scripts.check_r1_v8_precision_consumers import (
    current_source as original_source, decode, reconstruct_selected, select_rows, snapshot,
)
from scripts.verify_r1_v7_precision import _check_frozen, dec
from scripts.verify_r1_v8_precision import digest, precision_fields
from scripts.verify_r1_v9_precision import capture_actual_recomputation, floor_capability, verify_two_sides

CONTRACT_SHA = "f450fc485d9878b3a53496cbe8de4e4b5d815364b9f8d5b80f6da553e1e78397"


def current_source():
    result=original_source();project=Path(__file__).resolve().parents[1]
    for name in ('verify_r1_v9_precision.py','analyze_r1_v9_prototype.py','check_r1_v9_precision_consumers.py','diagnose_r1_v9_residuals.py'):
        result['source_files']['scripts/'+name]=sha(project/'scripts'/name)
    result['source_files_sha256']=digest(result['source_files'])
    return result


def runtime_identity():
    import scipy
    from threadpoolctl import threadpool_info
    return {'executable':sys.executable,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
            'threadpools':threadpool_info(),'numerical_thread_limit':1}


def load_contract(path, expected_sha256=CONTRACT_SHA):
    if expected_sha256 != CONTRACT_SHA or sha(path) != expected_sha256:
        raise ValueError("V9 validation contract differs from its frozen identity")
    return json.loads(Path(path).read_text())


def request_material_guard(observed_frozen, *, expected_material_sha256):
    _check_frozen(observed_frozen)
    matched = observed_frozen["content_sha256"] == expected_material_sha256
    return {"passed":matched, "expected_material_sha256":expected_material_sha256,
            "observed_material_sha256":observed_frozen["content_sha256"],
            "reason":None if matched else "original_request_material_binding_mismatch"}


def changed_oracle_coefficient(frozen, actual_diffusion):
    changed = deepcopy(frozen)
    changed["coefficients"]["D_ion_face_m2_s"] = [str(dec(x)) for x in actual_diffusion]
    changed["content_sha256"] = digest({k:v for k,v in changed.items() if k != "content_sha256"})
    _check_frozen(changed)
    return changed


def floor_specimens(contract):
    result=[]
    for entry in contract["floor_specimens"]:
        report=floor_capability(entry["observed"],entry["reference"],floor=entry["original_floor"],
            limit=entry["original_limit"],unit=entry["unit"],floor_source="unchanged_independent_oracle_unit_floor")
        relative=report["reference_relative_error"]
        expected=entry["expected_reference_relative_error"]
        matched=(report["original_gate_passed"]==entry["expected_absolute_gate_pass"]
                 and report["relative_capability"]==entry["expected_relative_capability"]
                 and ((relative is None and expected is None) or relative is not None and expected is not None and float(relative)==expected))
        result.append({"frozen_specimen":entry,"result":report,"expected_behavior_passed":matched})
    return result


def actual_sides(owner, state, voltage, frozen, upstream, context):
    captured={}
    def call():
        diagnostic=owner.eliminated_operator_diagnostics(state,voltage)
        row={"state":snapshot(state),"physics_reconstruction":{"eliminated_precision":diagnostic.precision_evidence}}
        captured.update(row=row,diagnostic=diagnostic)
        return row
    witness=capture_actual_recomputation(call,context=context)
    result=verify_two_sides(frozen,captured["row"],upstream=upstream,recomputation=witness,context=context)
    return {"verification":result,"actual_row":captured["row"],
            "original_ion_components":{k:{n:v for n,v in captured['diagnostic'][k].items()
                if n in ("normalization_scale","normalization_floor","maximum_absolute_difference","relative_error","unit")}
                for k in ("positive_ion_flux","positive_ion_rate")}}


def campaign(initial_system, zero_minus_system, selected, frozen, contract, *, context, require_exact=True):
    before=digest(frozen); upstream=zero_minus_system.precision_arithmetic_context()
    healthy={};near=[];shared=[]
    for label in contract["near_threshold_implementation_fault"]["case_keys"]:
        index,previous,row=selected[label]
        owner,state,reconstruction=reconstruct_selected(initial_system,previous,row,require_exact=require_exact)
        voltage=row["physics_reconstruction"]["voltage_V"]
        good=actual_sides(owner,state,voltage,frozen,upstream,context)
        healthy[label]={"row":index,"time_s":row["time_s"],"substeps":row["substeps"],
                        "reconstruction":reconstruction,"result":good}
        multiplier=contract["near_threshold_implementation_fault"]["multiplier"]
        changed=copy(owner)
        changed.material=replace(owner.material,D_ion_face=np.asarray(owner.material.D_ion_face)*multiplier)
        changed._fine_constants={}; changed._fine_evaluation_cache={};changed._fine_work={}
        actual=changed.evaluate(np.asarray(row["physics_reconstruction"]["coordinate"]),voltage)
        checked=actual_sides(changed,actual,voltage,frozen,upstream,context)
        arithmetic=checked["verification"]["arithmetic"]
        near.append({"case":label,"multiplier_binary64":multiplier,"multiplier_exact":str(dec(multiplier)),
            "actual_diffusion_m2_s":changed.material.D_ion_face.tolist(),"result":checked,
            "detected":good["verification"]["qualified"] and all(not arithmetic[side]["qualified"] for side in ("direct","eliminated")),
            "healthy_oracle_material_unchanged":digest(frozen)==before})
        if label in contract["shared_input_perturbation"]["case_keys"]:
            factor=contract["shared_input_perturbation"]["multiplier"]
            changed=copy(owner)
            changed.material=replace(owner.material,D_ion_face=np.asarray(owner.material.D_ion_face)*factor)
            changed._fine_constants={};changed._fine_evaluation_cache={};changed._fine_work={}
            changed_frozen=changed_oracle_coefficient(frozen,changed.material.D_ion_face)
            actual=changed.evaluate(np.asarray(row["physics_reconstruction"]["coordinate"]),voltage)
            checked=actual_sides(changed,actual,voltage,changed_frozen,upstream,context)
            guard=request_material_guard(changed_frozen,expected_material_sha256=frozen["content_sha256"])
            shared.append({"case":label,"multiplier_binary64":factor,"actual_diffusion_m2_s":changed.material.D_ion_face.tolist(),
                "changed_oracle_material":changed_frozen,"result":checked,"original_request_guard":guard,
                "expected_behavior_passed":checked["verification"]["constitutive_checks_passed"] and not guard["passed"],
                "scope":"consistently_changed_operator_input_not_the_original_physical_request"})
    floor=floor_specimens(contract)
    passed=(all(x['result']['verification']['qualified'] for x in healthy.values())
            and all(x['detected'] and x['healthy_oracle_material_unchanged'] for x in near)
            and all(x['expected_behavior_passed'] for x in shared)
            and all(x['expected_behavior_passed'] for x in floor) and digest(frozen)==before)
    return {"schema":"R1V9PrecisionCampaignV1","passed":passed,"healthy":healthy,"near_threshold":near,
            "shared_input":shared,"floor_specimens":floor,"new_trajectory_steps":0,"P2_qualified":False,
            "qualification":"named_operator_and_request_guard_requirements_only"}


def load_runtime(directory, request, prepared, *, historical=False):
    """Always use explicit pair backend; old packages provide raw inputs only."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
    directory=Path(directory)
    stack=load_device_from_yaml(directory/'SourceFixtureV1.yaml')
    binding=json.loads((directory/'ReferenceBindingV1.json').read_text())
    pair=(states.prepare_common_state(stack,request['case']['intervals'],binding,policy=r1_policy(),backend='pair')
          if historical else prepared)
    system,minus=states.verify_prepared_physics(pair,stack,binding,policy=r1_policy(),backend='pair')
    event=build_initial_step(system,minus,request['case']['amplitude_V'],policy=r1_policy(
        request['case']['nonlinear_factor'],time_substeps=tuple(request['case']['time_substeps'])),backend='pair')
    identity=pair.sha256 if hasattr(pair,'sha256') else pair['sha256']
    return system,minus,event,identity


def copied_result_negative(result,kind,*,row_index=1):
    """Predeclared internal-consistency-preserving eliminated-record copies."""
    if kind not in ('phi_only','all_direct_fields','low_word_bypass'):
        raise ValueError('unknown independent-path copy negative')
    changed=deepcopy(result);row=changed['accepted_steps'][row_index]
    direct=precision_fields(row['state']);record=row['physics_reconstruction']['eliminated_precision'];fields=record['fields']
    fields['phi_V']=deepcopy(direct['phi_V'])
    if kind!='phi_only':
        for name in tuple(fields):
            if name in direct:fields[name]=deepcopy(direct[name])
        fields['constraint_phi_V']=deepcopy(direct['phi_V'])
    before_low=fields['phi_V']['lo'][1]
    if kind=='low_word_bypass':
        low=float(np.nextafter(before_low,np.inf))
        high=fields['phi_V']['hi'][1]
        if (low==before_low or not np.isfinite(low) or high+low!=high+before_low
                or 0<abs(low)<np.finfo(float).tiny):
            raise ValueError('predeclared nextafter left the normal finite binary64-invisible domain')
        fields['phi_V']['lo'][1]=low;fields['constraint_phi_V']['lo'][1]=low
    fixed={k:fields[k] for k in ('dqfn_V','dqfp_V','positive_m3','occupancy','sheet_charge_C_m2')}
    record['shared_inputs']['fields']=deepcopy(fixed)
    record['solve_inputs']['fixed_inputs']=deepcopy(fixed)
    for name in ('phi_V','positive_m3','boundary_flux_m2_s'):record['ion_inputs'][name]=deepcopy(fields[name])
    solved={'phi_V':fields['constraint_phi_V'],**{k:fields[k] for k in ('n_m3','p_m3','poisson_residual_C_m2')}}
    record['solve'].update(fixed_input_digest=digest(fixed),solve_inputs_digest=digest(record['solve_inputs']),
        solved_fields_digest=digest(solved),ion_input_digest=digest(record['ion_inputs']))
    # Preserve the original floor definitions while synchronizing the duplicate
    # ion comparison outputs. This is not an output-only physics intervention;
    # it is an intentionally forged, internally resealed saved input record.
    for name,field in (('positive_ion_flux','positive_flux_m2_s'),('positive_ion_rate','positive_rate_m3_s')):
        item=row['physics_reconstruction']['eliminated_operator'][name]
        left=np.asarray(direct[field]['hi'])+np.asarray(direct[field]['lo'])
        right=np.asarray(fields[field]['hi'])+np.asarray(fields[field]['lo'])
        difference=left-right;peak=max(float(np.max(abs(left))),float(np.max(abs(right))))
        scale=max(peak,item['normalization_floor']);absolute=float(np.max(abs(difference)))
        item.update(direct=left.tolist(),eliminated=right.tolist(),difference=difference.tolist(),
            direct_maximum_absolute=float(np.max(abs(left))),eliminated_maximum_absolute=float(np.max(abs(right))),
            maximum_absolute_difference=absolute,normalization_scale=scale,floor_active=peak<item['normalization_floor'],relative_error=absolute/scale)
    row['physics_reconstruction']['eliminated_operator_error']=max(
        x['relative_error'] for x in row['physics_reconstruction']['eliminated_operator'].values())
    changed['sha256']=digest({k:v for k,v in changed.items() if k!='sha256'})
    return changed,{'kind':kind,'row_index':row_index,'substeps':row['substeps'],'time_s':row['time_s'],
        'low_word_rule':'one nextafter toward positive infinity at node 1' if kind=='low_word_bypass' else None,
        'old_low_V':before_low,'new_low_V':fields['phi_V']['lo'][1],
        'source_unchanged':changed['source']==result['source'],'direct_states_unchanged':all(a['state']==b['state'] for a,b in zip(changed['accepted_steps'],result['accepted_steps'])),
        'coordinates_unchanged':all(a['physics_reconstruction']['coordinate']==b['physics_reconstruction']['coordinate'] for a,b in zip(changed['accepted_steps'],result['accepted_steps'])),
        'internal_solved_and_ion_digests_resealed':True,'result_content_resealed':True}


def copy_campaign(directory,output,contract):
    """Actual D short-package replay of three resealed copied-side negatives."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import get_backend
    from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data
    from scripts.verify_r1_v7_precision import freeze_healthy_material
    from scripts.verify_r1_v9_precision import verify_two_sides
    directory,output=Path(directory),Path(output);output.mkdir(parents=True,exist_ok=False)
    read=lambda name:json.loads((directory/name).read_text())
    prepared,result,binding=read('PreparedV1.json'),read('ResultV1.json'),read('BindingV1.json')
    stack=load_device_from_yaml(directory/'FixtureV1.yaml')
    pair=get_backend('pair').decode_prepared(prepared)
    context={'source_digest':digest(result['source']),'prepared_sha256':prepared['sha256'],
             'result_sha256':result['sha256'],'saved_rows_digest':digest(result['accepted_steps'])}
    before=current_source();start=time.monotonic();report={'schema':'R1V9ActualCopiedSideReplayV1','passed':False,'cases':[],
        'new_trajectory_steps':0,'P2_qualified':False,'source_before':before,'input_context':context,
        'validation_contract_sha256':CONTRACT_SHA,
        'copy_case_rule':{'row_index':1,'node':1,'low_word_bypass':'one nextafter toward positive infinity; no amplitude scan'}}
    try:
        with threadpool_limits(1):
            report['runtime_identity']=runtime_identity()
            frozen=freeze_healthy_material(*build_r1_material(stack,16),source_identity=context['source_digest'])
            healthy=verify_r1_step_physics(stack,16,binding,pair,result,backend='pair')
            report['healthy_replay_certified']=healthy['certified']
            write(output/'ActualHealthyReplayV1.json',json_data(dict(healthy)))
            write(output/'ActualReceiptV1.json',healthy.replay_receipt.to_dict())
            upstream=result['initial_event']['precision_arithmetic_context']
            for kind in ('phi_only','all_direct_fields','low_word_bypass'):
                changed,mutation=copied_result_negative(result,kind)
                case_path=output/kind;case_path.mkdir()
                write(case_path/'ResultV1.json',changed);write(case_path/'MutationV1.json',mutation)
                write(case_path/'ManifestV1.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in case_path.iterdir()})
                verify_manifest(case_path)
                saved=json.loads((case_path/'ResultV1.json').read_text())
                if saved['sha256']!=digest({k:v for k,v in saved.items() if k!='sha256'}):raise ValueError('copied negative seal mismatch')
                checker=verify_two_sides(frozen,saved['accepted_steps'][1],upstream=upstream,
                    replay_receipt=healthy.replay_receipt,row_index=1,context=context)
                rejected=False;error=None
                try:verify_r1_step_physics(stack,16,binding,pair,saved,backend='pair')
                except Exception as exc:
                    error={'type':type(exc).__name__,'message':str(exc)}
                    rejected=type(exc).__name__=='R1PhysicsValidationError' and 'state equations 1' in str(exc)
                report['cases'].append({'mutation':mutation,'strict_checker':checker,'actual_replay_rejected':rejected,'rejection':error,
                    'passed':rejected and not checker['qualified'] and all(mutation[k] for k in ('source_unchanged','direct_states_unchanged','coordinates_unchanged'))})
            report['passed']=report['healthy_replay_certified'] and all(x['passed'] for x in report['cases'])
    except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)}
    after=current_source();report.update(source_after=after,source_unchanged=before==after,wall_seconds=time.monotonic()-start)
    report['passed'] &= before==after
    write(output/'ResultV1.json',report)
    write(output/'ManifestV1.json',{p.relative_to(output).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}
        for p in output.rglob('*') if p.is_file() and p!=output/'ManifestV1.json'})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--contract-sha256',default=CONTRACT_SHA)
    parser.add_argument('--historical-v8',action='store_true')
    parser.add_argument('--copy-campaign',action='store_true')
    args=parser.parse_args();contract=load_contract(args.contract,args.contract_sha256)
    directory=args.run_dir.resolve();output=args.output.resolve()
    if output==directory or output.is_relative_to(directory):raise ValueError('output must be outside sealed input')
    if args.copy_campaign:
        report=copy_campaign(directory,output,contract)
        print(json.dumps({k:report.get(k) for k in ('passed','error','source_unchanged','wall_seconds')},indent=2))
        return 0 if report['passed'] else 1
    verify_manifest(directory)
    read=lambda name:json.loads((directory/name).read_text())
    request,prepared,frozen=read('RequestV1.json'),read('PreparedV1.json'),read('FrozenHealthyMaterialV1.json')
    selected=select_rows(directory);output.mkdir(parents=True,exist_ok=False)
    before=current_source();start=time.monotonic()
    context={'source_commit':before['commit'],'source_files_sha256':before['source_files_sha256'],
             'request_sha256':sha(directory/'RequestV1.json'),'contract_sha256':args.contract_sha256,
             'historical_input':args.historical_v8}
    report={'schema':'R1V9PrecisionCampaignV1','passed':False,'P2_qualified':False,'context':context}
    write(output/'FrozenInputsV1.json',{'context':context,'contract':contract,'frozen_material':frozen,
        'selected_rows':{label:{'row':i,'previous':p,'state':r} for label,(i,p,r) in selected.items()}})
    try:
        with threadpool_limits(1):
            report['runtime_identity']=runtime_identity()
            system,minus,event,preparation=load_runtime(directory,request,prepared,historical=args.historical_v8)
            report.update(campaign(event.system,system,selected,frozen,contract,context=context,require_exact=True))
            report['actual_preparation_sha256']=preparation
    except Exception as exc:
        report['error']={'type':type(exc).__name__,'message':str(exc)}
    after=current_source();report.update(wall_seconds=time.monotonic()-start,source_before=before,source_after=after,
        source_unchanged=before==after,historical_input=args.historical_v8)
    report['passed'] &= before==after
    write(output/'ResultV1.json',report)
    write(output/'ManifestV1.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()})
    print(json.dumps({k:report.get(k) for k in ('passed','error','wall_seconds','P2_qualified')},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

"""V9 saved-record arithmetic, independent binding and capability analysis.

The programmatic entry accepts the actual replay receipt in the same process
that performed replay. Serialized booleans cannot establish that provenance.
The historical CLI reuses only the externally pinned V8 replay artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from scripts.analyze_r1_v7_prototype import compare_state_arithmetic, original_gate, sha, write, verify_manifest
from scripts.analyze_r1_v8_prototype import initial_checks, validate_budget
from scripts.verify_r1_v8_precision import canonical, digest, precision_fields, state_identity
from scripts.verify_r1_v9_precision import archived_v8_witnesses, floor_capability, verify_two_sides
from scripts.check_r1_v9_precision_consumers import load_contract, floor_specimens


def capability_intervals(records):
    """Observed consecutive row intervals; never interpolate a time horizon."""
    groups={}
    for item in records:
        for side,quantities in item['capability'].items():
            for quantity,value in quantities.items():
                key=f"{item['substeps']}:{side}:{quantity}"
                groups.setdefault(key,[]).append((item['row'],item['time_s'],value['signal_regime']))
    result={}
    for key,values in groups.items():
        segments=[]
        for index,time_s,regime in values:
            if not segments or segments[-1]['regime'] != regime:
                segments.append({'first_row':index,'last_row':index,'time_start_s':time_s,'time_end_s':time_s,
                                 'regime':regime,'rows':1})
            else:
                segments[-1].update(last_row=index,time_end_s=time_s,rows=segments[-1]['rows']+1)
        result[key]=segments
    return result


def validate_analysis_context(rows,prepared,result,context,*,historical=False):
    """Bind the caller's receipt context to these actual supplied inputs."""
    if rows != result.get('accepted_steps'):
        raise ValueError('analysis rows differ from the supplied result')
    for name,value in (('prepared',prepared),('result',result)):
        if value.get('sha256') != digest({k:v for k,v in value.items() if k!='sha256'}):
            raise ValueError('analysis '+name+' content seal is invalid')
    expected={'source_digest':digest(result['source']),'prepared_sha256':prepared['sha256'],
              'result_sha256':result['sha256'],'saved_rows_digest':digest(rows)}
    if not historical and any(context.get(key)!=value for key,value in expected.items()):
        raise ValueError('analysis context does not describe the supplied source/preparation/result/rows')
    return expected


def analyze_records(rows,frozen,prepared,result,*,context,output,budget,recomputations=None,
                    replay_receipt=None,fixed_v8_case=False):
    """No solver is called here. Call after actual replay for production proof."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    validate_budget(budget)
    actual_context=validate_analysis_context(rows,prepared,result,context,historical=fixed_v8_case)
    initial=initial_checks(frozen,prepared,result)
    upstream=result.get('initial_event',{}).get('precision_arithmetic_context',{})
    start=time.monotonic();records=[];failures=[];previous=None;states=set();anchors=regular=0
    with (output/'RowsV1.jsonl').open('w') as stream:
        for index,row in enumerate(rows):
            item={'row':index,'time_s':row['time_s'],'substeps':row['substeps'],'capability':{}}
            try:
                witness=None if recomputations is None else recomputations[index]
                verified=verify_two_sides(frozen,row,upstream=upstream,recomputation=witness,
                    context=context,fixed_v8_case=fixed_v8_case,replay_receipt=replay_receipt,row_index=index)
                item['constitutive_checks_passed']=verified['constitutive_checks_passed']
                item['independent_path_verified']=verified['independent_path_verified']
                item['qualified']=verified['qualified']
                item['role_binding']=verified['role_binding']
                item['copy_detected']=verified['fixed_V8_exact_copy_detected']
                item['original_gate']=original_gate(row,budget['comparison']['original_gate'])
                fields={'direct':precision_fields(row['state']),
                        'eliminated':row['physics_reconstruction']['eliminated_precision']['fields']}
                for side in ('direct','eliminated'):
                    oracle=verified['arithmetic'][side]['oracle']
                    item['capability'][side]={name:floor_capability(fields[side][name],oracle[name],
                        floor=1.,limit=1e-10,unit='m^-3 s^-1' if 'rate' in name else 'm^-2 s^-1',
                        floor_source='independent_oracle_original_unit_floor')
                        for name in ('positive_flux_m2_s','positive_rate_m3_s','boundary_flux_m2_s')}
                item['original_path_capability']={}
                for name,field in (('positive_ion_flux','positive_flux_m2_s'),('positive_ion_rate','positive_rate_m3_s')):
                    saved=row['physics_reconstruction']['eliminated_operator'][name]
                    item['original_path_capability'][name]=floor_capability(fields['direct'][field],fields['eliminated'][field],
                        floor=saved['normalization_floor'],limit=1e-6,unit=saved['unit'],
                        floor_source='fixed_unit_floor' if name=='positive_ion_flux' else 'max(direct_positive_rate_peak,1)')
                    item['original_path_capability'][name]['reference_scope']='other_actual_path_not_independent_healthy_oracle'
                    item['original_path_capability'][name]['recorded_original_gate']=saved
                states.add(state_identity(row['state']))
                if row['time_s']==0:
                    anchors+=1
                    item['arithmetic_passed']=canonical(precision_fields(row['state']))==canonical(initial['zero_plus_fields'])
                else:
                    regular+=1
                    arithmetic=compare_state_arithmetic(previous,row,frozen,budget)
                    item['arithmetic_passed']=arithmetic['passed']
                    item['arithmetic']=arithmetic
                item['passed']=item['qualified'] and item['original_gate']['passed'] and item['arithmetic_passed']
                if not item['passed']:failures.append({'row':index,'reason':'required_check_failed','verification':verified})
            except Exception as exc:
                item.update(passed=False,error={'type':type(exc).__name__,'message':str(exc)})
                failures.append({'row':index,'reason':'analysis_error','error':item['error']})
            stream.write(canonical(item)+'\n');records.append(item);previous=row
    passed=bool(rows) and not failures and initial['qualified']
    report={'schema':'R1V9SavedEvidenceAnalysisV1','passed':passed,'constitutive_checks_passed':bool(records) and all(x.get('constitutive_checks_passed') for x in records),
        'independent_path_verified':bool(records) and all(x.get('independent_path_verified') for x in records),
        'source_request_context':context,'actual_input_bindings':actual_context,'accepted_events':len(rows),'unique_primary_states':len(states),
        'initial_anchors':anchors,'regular_steps':regular,'rows_by_substeps':dict(Counter(str(x['substeps']) for x in rows)),
        'initial_arithmetic':initial,'failed_rows':len(failures),'first_failure':failures[0] if failures else None,
        'capability_intervals':capability_intervals(records),'new_trajectory_steps':0,'P2_qualified':False,
        'historical_V8_evidence':fixed_v8_case,'wall_seconds':time.monotonic()-start,
        'scope':'named_saved_record_checks_not_complete_production_or_global_error_qualification'}
    write(output/'FailuresV1.json',failures);write(output/'ResultV1.json',report)
    write(output/'ManifestV1.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget-file',type=Path,required=True);parser.add_argument('--budget-sha256',required=True)
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--historical-v8',action='store_true')
    args=parser.parse_args();directory=args.run_dir.resolve();output=args.output.resolve()
    if output==directory or output.is_relative_to(directory):raise ValueError('output must be outside the sealed input')
    if not args.historical_v8:raise ValueError('production analysis must receive an actual replay receipt through analyze_records')
    if sha(args.budget_file)!=args.budget_sha256:raise ValueError('budget external identity mismatch')
    contract=load_contract(args.contract);specimens=floor_specimens(contract)
    if not all(x['expected_behavior_passed'] for x in specimens):raise ValueError('frozen floor specimen behavior failed')
    context,witnesses=archived_v8_witnesses(directory)
    read=lambda name:json.loads((directory/name).read_text())
    rows=[json.loads(line) for line in (directory/'AcceptedStepsV1.jsonl').read_text().splitlines()]
    report=analyze_records(rows,read('FrozenHealthyMaterialV1.json'),read('PreparedV1.json'),read('ResultV1.json'),
        context=context,output=output,budget=json.loads(args.budget_file.read_text()),recomputations=witnesses,fixed_v8_case=True)
    print(json.dumps({k:report[k] for k in ('passed','accepted_events','unique_primary_states','initial_anchors','regular_steps','P2_qualified')},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

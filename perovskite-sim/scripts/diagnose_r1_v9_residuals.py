"""Bounded V8-point residual diagnostics under the explicit V9 backend.

State, reference, physical previous, coordinate, dt, material and scales are
saved. Only evaluations are performed. Decimal independently covers storage,
ionic divergence, Poisson and trace electrostatics; carrier-rate/local-carrier
terms are explicitly held at their recorded binary64 values, not falsely
presented as a complete independent carrier residual or Newton error bound.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.analyze_r1_v7_prototype import sha, write, verify_manifest
from scripts.check_r1_v8_precision_consumers import reconstruct_selected, snapshot
from scripts.check_r1_v9_precision_consumers import current_source, load_contract, load_runtime, runtime_identity
from scripts.verify_r1_v7_precision import dec, evaluate_state
from scripts.verify_r1_v8_precision import as_snapshot, digest, precision_fields, represented


def vector_difference(actual,expected):
    a,b=[dec(x) for x in actual],[dec(x) for x in expected]
    if len(a)!=len(b):raise ValueError('residual vectors have different lengths')
    errors=[abs(x-y) for x,y in zip(a,b)]
    return {'maximum_absolute_scaled_difference':str(max(errors,default=Decimal(0))),
            'actual_inf_norm':str(max(map(abs,a),default=Decimal(0))),
            'reference_inf_norm':str(max(map(abs,b),default=Decimal(0))),
            'component_differences':[str(x-y) for x,y in zip(a,b)],
            'scope':'observed_same_declared_scales_not_a_tolerance_or_error_bound'}


def decimal_components(state,previous,frozen,upstream,record,*,quantized=False):
    """Supported independent components with unchanged explicit other terms."""
    with localcontext() as context:
        context.prec=100
        source=precision_fields(state);old=precision_fields(previous)
        convert=lambda x:represented(x['hi']) if quantized else represented(x)
        f={k:convert(v) for k,v in source.items()};p={k:convert(v) for k,v in old.items()}
        active_faces=[j for j,x in enumerate(frozen['coefficients']['D_ion_face_m2_s']) if dec(x)>0]
        active=sorted({x for face in active_faces for x in (face,face+1)})
        nt=represented(upstream['sheet_inputs']['trap_density_m2'])
        eq=represented(upstream['sheet_inputs']['equilibrium_occupancy'])
        static=represented(upstream['sheet_inputs']['static_sheet_charge_C_m2'])
        q=dec(frozen['constants']['q_C'])
        f['sheet_charge_C_m2']=[s+q*n*(e-o) for s,n,e,o in zip(static,nt,eq,f['occupancy'])]
        def storage(v):return v['n_m3'][1:-1]+v['p_m3'][1:-1]+[a*b for a,b in zip(nt,v['occupancy'])]+[v['positive_m3'][j] for j in active]
        current_storage,old_storage=storage(f),storage(p)
        point={k:[str(x) for x in v] for k,v in f.items()}
        oracle=evaluate_state(frozen,point,precision=100)
        rate=[dec(x) for x in record['physics_reconstruction']['rate']]
        interior=len(f['phi_V'])-2;ion_start=2*interior+len(nt)
        for j,node in enumerate(active):rate[ion_start+j]=dec(oracle['positive_rate_m3_s'][node])
        dt=dec(record['dt_s']);storage_residual=[a-b-dt*c for a,b,c in zip(current_storage,old_storage,rate)]
        poisson=[dec(x) for x in oracle['poisson']['residual_C_m2']]
        local=[dec(x) for x in record['physics_reconstruction']['local_residual']]
        caps=upstream['trace_geometry']['capacitances_F_m2'];jumps=represented(upstream['trace_geometry']['jump_V'])
        for k,((left,right),pair,jump) in enumerate(zip(frozen['coefficients']['interface_nodes'],caps,jumps)):
            cl,cr=represented(pair);tl,tr=f['trace_potential_V'][2*k:2*k+2]
            local[6*k]=tr-tl-jump
            local[6*k+1]=cl*(tl-f['phi_V'][left])+cr*(tr-f['phi_V'][right])-f['sheet_charge_C_m2'][k]
        evidence=record['physics_reconstruction']
        scales=[dec(x) for key in ('storage_scale','poisson_scale_C_m2','local_algebraic_scale') for x in evidence[key]]
        raw=storage_residual+poisson+local
        return {'scaled':[str(x/s) for x,s in zip(raw,scales)],'unscaled':[str(x) for x in raw],
            'scales':[str(x) for x in scales], 'quantized_primary_inputs':quantized,
            'independently_evaluated':['storage_from_population_and_trap_inputs','positive_ion_flux_and_rate','Poisson','trace_potential_and_Gauss'],
            'held_recorded_binary64_terms':['electron_rate','hole_rate','trap_rate','four_local_carrier_balance_terms_per_interface'],
            'units':{'carrier_ion_storage':'m^-3','trap_storage':'m^-2','Poisson':'C m^-2',
                     'local_potential':'V','local_Gauss':'C m^-2','local_carrier_balance':'m^-2 s^-1'},
            'scope':'partial_independent_equation_evaluation_with_explicit_held_terms_not_full_Newton_error_bound'}


def diagnose(initial_system,rows,frozen,contract,*,baseline_rows=None):
    results=[]
    for index in contract['same_point_residuals']['V8_rows']:
        previous,row=rows[index-1],rows[index]
        if previous['substeps']!=row['substeps'] or previous['time_s']>=row['time_s']:
            raise ValueError('fixed residual point lacks its actual same-tier previous')
        owner,state,identity=reconstruct_selected(initial_system,previous,row,require_exact=True)
        evidence=row['physics_reconstruction'];z=np.asarray(evidence['coordinate']);voltage=evidence['voltage_V']
        scales=tuple(np.asarray(evidence[name]) for name in ('storage_scale','poisson_scale_C_m2','local_algebraic_scale'))
        actual_previous=owner._step_reference
        actual,_,evaluated=owner.residual_and_jacobian(z,voltage,actual_previous,row['dt_s'],*scales)
        if precision_fields(snapshot(evaluated))!=precision_fields(row['state']):
            raise ValueError('fixed point changed during residual evaluation')
        upstream=owner.precision_arithmetic_context()
        high=decimal_components(row['state'],previous['state'],frozen,upstream,row)
        low=decimal_components(row['state'],previous['state'],frozen,upstream,row,quantized=True)
        inputs={'state':row['state'],'reference':{k:{'hi':v.hi.tolist(),'lo':v.lo.tolist()} for k,v in owner._fine_reference.items()},
            'physical_previous':snapshot(actual_previous),'coordinate':z.tolist(),'dt_s':row['dt_s'],'voltage_V':voltage,
            'frozen_material':frozen,'scales':{name:array.tolist() for name,array in zip(('storage','Poisson','local'),scales)},
            'upstream':upstream}
        item={'V8_row':index,'substeps':row['substeps'],'time_s':row['time_s'],'reconstruction':identity,
            'fixed_inputs':inputs,'fixed_inputs_sha256':digest(inputs),'actual_scaled_residual':actual.tolist(),
            'saved_scaled_residual':evidence['scaled_residual_vector'],
            'same_pair_algorithm_reconstruction':vector_difference(actual,evidence['scaled_residual_vector']),
            'decimal_fixed_inputs':high,'decimal_quantized_inputs':low,
            'equation_arithmetic_and_held_terms':vector_difference(actual,high['scaled']),
            'input_quantization_on_supported_components':vector_difference(low['scaled'],high['scaled']),
            'Newton_iterations_performed':0,'strict_case_restored':False}
        if baseline_rows is not None:
            hits=[r for r in baseline_rows if (r['substeps'],r['time_s'])==(row['substeps'],row['time_s'])]
            if len(hits)!=1:raise ValueError('baseline point selection is absent or ambiguous')
            other=hits[0]['physics_reconstruction']
            unscaled=other['storage_residual']+other['poisson_residual_C_m2']+other['local_residual']
            baseline_same_scales=[dec(a)/dec(b) for a,b in zip(unscaled,np.concatenate(scales))]
            item['cross_trajectory_context']={'comparison':vector_difference(baseline_same_scales,actual),
                'baseline_state_sha256':digest(hits[0]['state']),
                'scope':'different_actual_states_and_histories_with_pair_scales_not_a_same_point_error_bar'}
        results.append(item)
    return {'schema':'R1V9SamePointResidualDiagnosticsV1','points':results,'new_trajectory_steps':0,'new_Newton_iterations':0,
            'all_fixed_points_reconstructed':all(all(x['reconstruction']['fields'].values()) for x in results),
            'P2_qualified':False,'strict_N32_N256_restored':False,
            'preparation_scope':'fresh_initial_constraint_reconstruction_before_point_evaluations_may_iterate',
            'scope':'three_frozen_N16_V8_points_no_transfer_to_missing_strict_case_low_words_or_histories'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--baseline-run',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
    args=parser.parse_args();contract=load_contract(args.contract);directory=args.run_dir.resolve();output=args.output.resolve()
    if output==directory or output.is_relative_to(directory):raise ValueError('output must be outside the sealed run')
    verify_manifest(directory)
    read=lambda name:json.loads((directory/name).read_text())
    rows=[json.loads(x) for x in (directory/'AcceptedStepsV1.jsonl').read_text().splitlines()]
    baseline=None
    if args.baseline_run:
        verify_manifest(args.baseline_run)
        baseline=[json.loads(x) for x in (args.baseline_run/'AcceptedStepsV1.jsonl').read_text().splitlines()]
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic();before=current_source()
    report={'schema':'R1V9SamePointResidualDiagnosticsV1','all_fixed_points_reconstructed':False,'P2_qualified':False}
    try:
        with threadpool_limits(1):
            report['runtime_identity']=runtime_identity()
            system,minus,event,_=load_runtime(directory,read('RequestV1.json'),read('PreparedV1.json'),historical=True)
            report.update(diagnose(event.system,rows,read('FrozenHealthyMaterialV1.json'),contract,baseline_rows=baseline))
    except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)}
    after=current_source();report.update(wall_seconds=time.monotonic()-start,source_before=before,source_after=after,
        source_unchanged=before==after,input_manifest_sha256=sha(directory/'ManifestV1.json'),contract_sha256=sha(args.contract))
    write(output/'ResultV1.json',report)
    write(output/'ManifestV1.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()})
    print(json.dumps({k:report.get(k) for k in ('all_fixed_points_reconstructed','error','source_unchanged','wall_seconds')},indent=2))
    return 0 if report['all_fixed_points_reconstructed'] and report['source_unchanged'] else 1


if __name__=='__main__':raise SystemExit(main())

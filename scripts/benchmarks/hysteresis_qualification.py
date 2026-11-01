"""Prepare exact HI refinement requests or bound existing saved device snapshots.

Both operations are nonintegrating. Admission requests are deliberately false;
the reference harness still requires a separate Root-issued authorization.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time


SAVED_REVIEW_SHA256 = '26e0a903b5be27656e07f80e2c82db5f65b3a51931203431d7739eeae89f7772'
CASES = {'D0':'hi_diffusivity_00', 'C0':'hi_density_00'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data, *, compact=False):
    with Path(path).open('x') as stream:
        json.dump(data,stream,indent=None if compact else 2,allow_nan=False)
        stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def source_packet(repo):
    paths = ['scripts/benchmarks/hysteresis_reference.py','scripts/benchmarks/hysteresis_current_bounds.py',
             'scripts/benchmarks/hysteresis_qualification.py',
             'perovskite-sim/perovskite_sim/solver/mol.py',
             'perovskite-sim/perovskite_sim/experiments/jv_sweep.py',
             'perovskite-sim/perovskite_sim/physics/continuity.py',
             'perovskite-sim/perovskite_sim/physics/poisson.py',
             'perovskite-sim/perovskite_sim/physics/recombination.py',
             'perovskite-sim/perovskite_sim/constants.py']
    return {str(repo/p):sha(repo/p) for p in paths}


def saved_bindings(repo, archive):
    stage = archive/'test/ArchitectureRefactor/Stage00'
    review = stage/'HysteresisSavedDiagnosticsReviewV1.json'
    if sha(review) != SAVED_REVIEW_SHA256:
        raise ValueError('accepted saved-data review identity changed')
    previous = read(review)
    bindings = read(previous['binding']['path'])
    if sha(previous['binding']['path']) != previous['binding']['sha256']:
        raise ValueError('accepted review binding changed')
    pins = bindings['all_rechecked_source_input_hashes']
    result = {}
    for case in CASES:
        run = archive/'results/RefactorHysteresis'/case/'HistoryDiagnosticAttempt01'
        initial = stage/'HysteresisInitialPacketAttempt01'/case
        files = {'plan':run/'RunPlan.json','states':run/'PhysicalStates.json',
                 'packet':initial/'Packet.json','arrays':initial/'Arrays.npz'}
        for path in files.values():
            if sha(path) != pins[str(path)]:
                raise ValueError(f'saved input changed: {path}')
        plan = read(files['plan'])
        if sha(plan['raw_input']['path']) != plan['raw_input']['sha256']:
            raise ValueError('original request changed')
        # The harness alone has the separately requested numerical-policy delta.
        # Every original production/backend source remains the actual saved source.
        for path,expected in plan['source']['files'].items():
            if path != 'scripts/benchmarks/hysteresis_reference.py' and sha(repo/path) != expected:
                raise ValueError(f'original physical source changed: {path}')
        result[case] = {key:{'path':str(path),'sha256':sha(path)} for key,path in files.items()}
        result[case]['original_harness_sha256'] = plan['source']['files']['scripts/benchmarks/hysteresis_reference.py']
    return result


def bound_saved(repo, archive, output):
    import ast
    import numpy as np
    from threadpoolctl import threadpool_info
    import _decimal
    bounds = module('hi_saved_current_bounds',repo/'scripts/benchmarks/hysteresis_current_bounds.py')
    start = time.monotonic()
    sources = source_packet(repo)
    inputs = saved_bindings(repo,archive)
    output.mkdir(parents=True,exist_ok=False)
    q = None
    for item in ast.parse((repo/'perovskite-sim/perovskite_sim/constants.py').read_text()).body:
        if isinstance(item,ast.Assign) and isinstance(item.value,ast.Constant):
            if any(isinstance(t,ast.Name) and t.id=='Q' for t in item.targets):
                q = float(item.value.value)
    if q is None:
        raise ValueError('pinned elementary-charge literal unavailable')
    records, summaries = [], []
    priority = {'dark_seed':0,'dark_prebias':1,'forward_ramp':2,'reverse_ramp':3}
    for case,files in inputs.items():
        packet = read(files['packet']['path'])
        with np.load(files['arrays']['path'],allow_pickle=False) as data:
            arrays = {k:data[k] for k in data.files}
        state = read(files['states']['path'])
        case_rows = []
        ordered = sorted(state['phase_limits'].items(),key=lambda item:priority.get(item[0],4))
        for phase,points in ordered:
            for side,point in points.items():
                row = bounds.qualify_point(point,arrays,packet,q)
                row.update({'case':case,'phase_side':side})
                records.append(row);case_rows.append(row)
        summaries.append({'case':case,'points':len(case_rows),
                          'max_current_error_upper_A_m2':max(r['max_total_current_error_upper_A_m2'] for r in case_rows),
                          'max_Ddot_error_upper_A_m2':max(r['max_Ddot_error_upper_A_m2'] for r in case_rows),
                          'first_ramp_current_errors_A_m2':{r['phase']:r['left_current_error_upper_A_m2'] for r in case_rows if r['phase'].endswith('_ramp') and r['phase_side']=='start'}})
    if sources != source_packet(repo):
        raise ValueError('qualification source drift')
    result = {'schema':'solarlab.hi-device-pointwise-enclosures.v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
              'status':'computed_bounds_require_independent_review','points':records,'case_summaries':summaries,
              'sources':sources,'inputs':inputs,'accepted_saved_review_sha256':SAVED_REVIEW_SHA256,
              'enclosure_arithmetic':{'precision':bounds.PRECISION,'Taylor_terms':bounds.SERIES_TERMS,
                'exponential':'range-reduced positive expm1 Taylor series with explicit geometric remainder; no library exp used',
                'decimal_implementation':{'origin':_decimal.__spec__.origin,
                  'binding_path':getattr(_decimal,'__file__',sys.executable),
                  'binding_sha256':sha(getattr(_decimal,'__file__',sys.executable)),
                  'embedded_in_interpreter':getattr(_decimal,'__file__',None) is None}},
              'runtime':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
                         'scipy':importlib.metadata.version('scipy'),'threadpools':threadpool_info()},
              'continuous_trajectory_error':None,'scientific_gate_awards':[],
              'native_time_advances':0,'new_initial_states':0,'wall_s':time.monotonic()-start,
              'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write(output/'Result.json',result)
    print(json.dumps({'points':len(records),'cases':summaries,'wall_s':result['wall_s']}))


def prepare(repo, archive, output, first_step, test_receipt):
    import numpy as np
    from contextlib import ExitStack
    from unittest.mock import patch
    start = time.monotonic()
    hr = module('hi_prepared_reference',repo/'scripts/benchmarks/hysteresis_reference.py')
    if any(os.environ.get(k) != '1' for k in hr.THREAD_KEYS):
        raise ValueError('set all six thread controls to one before preparation')
    saved = saved_bindings(repo,archive)
    initial_sources = source_packet(repo)
    receipt = read(test_receipt)
    if receipt.get('success') is not True or receipt.get('integrator_step_attempts') != 0:
        raise ValueError('a successful bounded nonintegrating test receipt is required')
    for path, expected in receipt.get('sources', {}).items():
        if sha(path) != expected:
            raise ValueError('focused test source no longer matches preparation')
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(repo/'perovskite-sim'))
    from backend.main import stack_from_dict
    from perovskite_sim.experiments import jv_sweep as jv
    from perovskite_sim.experiments.waveform_jv import uniform_absorber_generation
    from perovskite_sim.solver import mol
    contract = hr.checked_json(repo/hr.CONTRACT,hr.CONTRACT_SHA256)
    predicate = contract['error_budget']['required_power_precondition']
    predicate_path = archive/predicate['executable_source']['path']
    if sha(predicate_path) != predicate['executable_source']['sha256']:
        raise ValueError('frozen power predicate changed')
    power = module('hi_frozen_power_precondition',predicate_path)
    # Distinct settings: three spatial, two additional temporal and two additional
    # joint solver-tolerance runs. Observation refinement adds no integration.
    settings = [('space61',60,2,2),('space121',120,2,2),('space241',240,2,2),
                ('time0',120,0,2),('time1',120,1,2),('solver0',120,2,0),('solver1',120,2,1)]
    runs,meshes,eligibility = [],[],[]
    guards = []
    with ExitStack() as stack:
        for target in ('scipy.integrate.solve_ivp','scipy.integrate._ivp.base.OdeSolver.step'):
            guards.append(stack.enter_context(patch(target,side_effect=AssertionError('time integration forbidden'))))
        for owner,name in ((jv,'solve_equilibrium'),(jv,'run_transient'),(mol,'run_transient')):
            guards.append(stack.enter_context(patch.object(owner,name,side_effect=AssertionError('initial state or time integration forbidden'))))
        shared_source,shared_environment = None,None
        for case,case_id in CASES.items():
            bundle = hr.load_inputs(repo,archive,case_id)
            device = stack_from_dict(bundle['request']['device'])
            for grid in (60,120,240):
                x = jv.build_electrical_grid(device,grid)
                jv.require_thick_layer_interface_resolution(x,device,N_grid=grid,allow_underresolved_grid=False)
                material = jv.build_material_arrays(x,device)
                material = replace(material,G_optical=uniform_absorber_generation(x,device,bundle['request']['params']['waveform']['uniform_generation_rate_m3_s']))
                if len(x) != grid+1:
                    raise ValueError('actual electrical mesh size differs from frozen ladder')
                arrays = {'initial.x_m':x,'material.poisson_factor.C':material.poisson_factor.C,
                          'material.poisson_factor.h_cell':material.poisson_factor.h_cell}
                names = ('N_A','N_D','P_ion0','eps_r','D_n_face','D_p_face','D_ion_face','G_optical',
                         'ni_sq','tau_n','tau_p','n1','p1','B_rad','C_n','C_p','dx_cell','chi','Eg','N_t_node')
                arrays.update({'material.'+name:getattr(material,name) for name in names})
                if grid == 60:
                    with np.load(saved[case]['arrays']['path'],allow_pickle=False) as old:
                        if not all(np.array_equal(value,old[name]) for name,value in arrays.items()):
                            raise ValueError('source-built coarse mesh/material differs from saved input')
                path = output/f'{case}Mesh{grid+1}.npz'
                np.savez(path,**arrays)
                series = 1/sum((1/Fraction.from_float(float(c)) for c in material.poisson_factor.C),Fraction(0))
                meshes.append({'case':case,'requested_N':grid,'actual_nodes':len(x),'path':str(path),'sha256':sha(path),
                               'initial_state_constructed':False,'material_compiled':True,
                               'C_series_exact_F_m2':str(series),'C_series_display_F_m2':float(series),
                               'device_area_m2':None,'current_units':'A/m^2; no total-device area invented'})
            for name,grid,time_level,solver_level in settings:
                plan = hr.build_plan(repo,archive,case_id,grid,time_level,solver_level,prebias_first_step_s=first_step)
                if shared_source is None:
                    shared_source,shared_environment = plan['source'],plan['environment']
                if plan['source'] != shared_source or plan['environment'] != shared_environment:
                    raise ValueError('source/environment changed during ladder preparation')
                path = output/f'{case}_{name}_RunPlan.json'
                write(path,plan,compact=True)
                admission_path = output/f'{case}_{name}_AdmissionRequest.json'
                admission = {'schema':'solarlab.hysteresis_reference_admission.v1','execution_authorized':False,
                  'identity_sha256':plan['identity_sha256'],'resources':plan['resources'],'root_authorization_message':None,
                  'bound_analytic_test_receipt_sha256':sha(test_receipt),'purpose':'unqualified_reference_diagnostic',
                  'state':'prepared; Root review and exact compute admission required'}
                write(admission_path,admission)
                destination = archive/'results/RefactorHysteresis'/case/f'DeviceQualificationAttempt01_{name}'
                if destination.exists():
                    raise ValueError('future output already exists; prepare a new named attempt')
                argv = [sys.executable,'-B',str(repo/'scripts/benchmarks/hysteresis_reference.py'),'run',
                  '--repo',str(repo),'--archive',str(archive),'--case',case_id,'--grid',str(grid),
                  '--step-level',str(time_level),'--solver-level',str(solver_level),'--admission',str(admission_path),'--output',str(destination)]
                if first_step is not None:
                    argv += ['--prebias-first-step-s',repr(first_step)]
                runs.append({'case':case,'setting':name,'plan':{'path':str(path),'sha256':sha(path)},
                  'identity_sha256':plan['identity_sha256'],'admission_request':{'path':str(admission_path),'sha256':sha(admission_path)},
                  'command':argv,'shell_command':'env '+' '.join(k+'=1' for k in hr.THREAD_KEYS)+' '+shlex.join(argv),
                  'execution_authorized':False})
            observations = read(archive/'results/RefactorHysteresis'/case/'HistoryDiagnosticAttempt01/Observations441.json')
            old_power = next(m for m in observations['metrics'] if m['observable']=='instantaneous_ramp_side_v1')['sampled_Pmax_W_m2']
            unknown = power.qualify_reference_power(old_power['forward'],old_power['reverse'],None,None,**predicate['arguments'])
            if unknown['eligible'] or unknown['status'] != 'unknown':
                raise ValueError('missing independent uncertainty was promoted to eligibility')
            eligibility.append({'case':case,'power_predicate_result':unknown,'independent_power_errors_W_m2':None})
    if any(g.call_count for g in guards) or initial_sources != source_packet(repo):
        raise ValueError('preparation advanced/initialized or source drifted')
    if len(runs) != 14 or len(meshes) != 6:
        raise ValueError('incomplete D0/C0 preparation')
    manifest = {'schema':'solarlab.hi-independent-refinement-preparation.v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'D0/C0 only; remaining eleven frozen sentinels are not qualified by these two devices',
      'sources':initial_sources,'original_saved_bindings':saved,'runtime_environment':shared_environment,'source_tree':shared_source,
      'runs':runs,'meshes':meshes,'prebias_first_step_s':first_step,
      'axis_definitions':{'space':['space61','space121','space241'],'time':['time0','time1','space121'],
                         'joint_solver_tolerance':['solver0','solver1','space121'],
                         'observation_counts':[111,221,441],'observation_additional_runs':0,'ramp_control_intervals':440},
      'eligibility':eligibility,'power_predicate':predicate,
      'first_next_command':runs[0]['shell_command'],'all_execution_authorizations_false':True,
      'no_admission_self_awarded':True,'new_equilibrium_or_time_calls':0,'guards_called':sum(g.call_count for g in guards),
      'remaining':'Independent Root review/admission; time/state/nonlinear/space and complete observable error qualification remain pending.',
      'wall_s':time.monotonic()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write(output/'Manifest.json',manifest)
    print(json.dumps({'runs':len(runs),'meshes':len(meshes),'wall_s':manifest['wall_s'],'native_or_initialization_calls':0}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('bound-saved','prepare'))
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prebias-first-step-s',type=float)
    parser.add_argument('--test-receipt',type=Path)
    args=parser.parse_args(argv)
    if args.operation=='bound-saved':
        if args.prebias_first_step_s is not None:
            parser.error('saved-state bounds do not change numerical startup policy')
        bound_saved(args.repo.resolve(),args.archive.resolve(),args.output.resolve())
    else:
        if args.test_receipt is None:
            parser.error('prepare needs a successful focused test receipt')
        prepare(args.repo.resolve(),args.archive.resolve(),args.output.resolve(),args.prebias_first_step_s,args.test_receipt.resolve())


if __name__=='__main__':
    main()

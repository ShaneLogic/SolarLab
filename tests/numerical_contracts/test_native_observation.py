"""Source-contract/map controls; synthetic packets never claim an IDA run."""

from dataclasses import asdict, replace
from fractions import Fraction
from hashlib import sha256
from functools import lru_cache
from pathlib import Path
import copy
import importlib.util
import json
import os

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import ContractError
from scripts.benchmarks.coupled_device_prototype import (
    AffineCoupledSlab, AffineSamplingContext, AffineVoltageMap, ProtocolSegment,
    SlabDefinition, VoltageLiftAdapter, VoltageLiftSegmentAdapter, protocol_from_plan,
)
from scripts.benchmarks.interval_observation import AcceptedClock, BallIntegrator, Polynomial, SlabPathObserver, rational
from scripts.benchmarks.native_observation import (
    compare_physical_readback, input_evaluation_bounds, prepare_physical_polynomial,
    propagated_input_error, read_native_basis_packet, prepare_native_frame_map,
    require_frame_successor, voltage_current_sensitivity, AcceptedIntervalObserver,
    observation_record, restore_observation_record, readback_current_evidence,
    prepare_interval_observation_policy, validate_interval_observation_admission,
)


REPO = Path(__file__).resolve().parents[2]
PLAN = Path(os.environ['REAL_DEVICE_PLAN'])
CASES = ('S0NeutralPublicDeviceV1', 'DynamicAcceptorIonPublicDeviceV1')
HEADER = '04988c332e2d004b4e838ffca847990f353f68d41cb225f1b9085b3a8bed52ce'
STAMP_NAMES = ('IDAGetNumSteps', 'IDAGetCurrentTime', 'IDAGetLastStep',
               'IDAGetLastOrder', 'IDAGetCurrentStep', 'IDAGetCurrentOrder')


def _charge_refinement_request():
    """Schema-only input: never a native request or physical case."""
    return {'map_identity': '1'*64, 'segments': [{'id': 'synthetic'}],
            'controls': {'rtol': 1e-8}, 'budgets': {'native_steps': 200000, 'charge_C': 1.0},
            'observation_times': {'synthetic': [0.0, 1.0]}, 'quadrature': {}}


def test_charge_refinement_default_and_parent_binding():
    from scripts.benchmarks.native_observation import prepare_charge_accumulator, CHARGE_REFINEMENT
    from scripts.benchmarks.coupled_device_prototype import digest
    request = _charge_refinement_request()
    before = copy.deepcopy(request)
    arguments = dict(binding_identity='a'*64, header_sha256='b'*64, backend_modules={})
    legacy = prepare_interval_observation_policy(request, **arguments)
    assert prepare_interval_observation_policy(request, charge_refinement=None, **arguments) == legacy
    assert prepare_charge_accumulator(request, legacy) is None
    refined = prepare_interval_observation_policy(request, charge_refinement=CHARGE_REFINEMENT, **arguments)
    assert request == before
    assert refined['parent_policy_sha256'] == digest(legacy)
    assert {k: refined[k] for k in legacy if k != 'schema'} == {k: v for k, v in legacy.items() if k != 'schema'}
    request['interval_observation'] = refined
    accumulator = prepare_charge_accumulator(request, refined)
    assert accumulator.policy.source_identity == digest(request)
    assert accumulator.policy.quantum == Fraction(1, 2**120)
    assert accumulator.policy.max_terms*accumulator.policy.quantum == accumulator.policy.max_slack_per_channel
    assert accumulator.policy.upper_integer_bits == 65675 < 131072
    request['controls']['rtol'] *= 2
    with pytest.raises(ContractError, match='charge_policy_changed'):
        prepare_charge_accumulator(request, refined)


@pytest.mark.parametrize('fault', ['name', 'cells', 'quantum', 'channels', 'extra', 'parent', 'null'])
def test_charge_refinement_rejects_tampering(fault):
    from scripts.benchmarks.native_observation import prepare_charge_accumulator, CHARGE_REFINEMENT
    request = _charge_refinement_request()
    policy = prepare_interval_observation_policy(request, binding_identity='a'*64, header_sha256='b'*64,
        backend_modules={}, charge_refinement=CHARGE_REFINEMENT)
    refinement = policy['charge_refinement']
    if fault == 'name': refinement['name'] = 'fixed8'
    if fault == 'cells': refinement['partition_cells'] = 2
    if fault == 'quantum': refinement['upper_sum']['quantum_bits'] = 119
    if fault == 'channels': refinement['upper_sum']['channel_ids'].pop()
    if fault == 'extra': refinement['unbound'] = True
    if fault == 'parent': policy['parent_policy_sha256'] = 'f'*64
    if fault == 'null': policy['charge_refinement'] = None
    with pytest.raises(ContractError, match='charge_(policy|refinement)'):
        prepare_charge_accumulator(request, policy)


def _synthetic_charge_observer(monkeypatch, source='a'*64):
    from types import SimpleNamespace
    from scripts.benchmarks.interval_observation import Enclosure
    import scripts.benchmarks.native_observation as module

    observer = object.__new__(AcceptedIntervalObserver)
    zeros = (Fraction(0),)*3
    inputs = {'raw_charge_integral_error_C': zeros, 'tangent_charge_integral_error_C': zeros}
    monkeypatch.setattr(module, 'propagated_input_error', lambda *_: inputs)
    monkeypatch.setattr(module, 'endpoint_charge_mismatch', lambda *_: zeros)
    monkeypatch.setattr(module, 'public_action_enclosures', lambda values: values)
    path = SimpleNamespace(identity='b'*64, clock=SimpleNamespace(payload=lambda: {'segment_id': 'synthetic'}))
    observer.prepared = SimpleNamespace(path=path)
    observer.frame = SimpleNamespace(identity='c'*64)
    observer.binding = SimpleNamespace(context=SimpleNamespace(request_sha256=source))
    observer.arithmetic = object()
    values = (Enclosure(0), Enclosure(Fraction(-1, 7), Fraction(1, 1000)), Enclosure(0))
    calls = []
    def integrate(*args, **kwargs):
        calls.append(kwargs)
        return {'raw_polynomial': values, 'same_state_affine_tangent': values,
                'raw_tangent_charge_L1_upper_C': zeros, 'raw_tangent_L1_upper_bounds': (),
                'terminal_partition': {'synthetic_routing_fixture': True}}
    model = SimpleNamespace(linear_action=lambda name, *args, **kwargs:
                            (Enclosure(0),)*(1 if name == 'body_charge' else 2))
    observer.observer = SimpleNamespace(model=model, path=path, integrate=integrate,
        affine={'body_charge': {'left': (0,), 'right': (0,)}, 'metal_charge': {'left': (0, 0), 'right': (0, 0)}},
        strip_current_debit=lambda _: {key: zeros for key in ('raw_polynomial', 'same_state_affine_tangent')})
    return observer, calls


def test_charge_refinement_actual_routing_signed_evidence_and_nonresetting_prefix(monkeypatch):
    from scripts.benchmarks.interval_observation import ChargePrefix
    from scripts.benchmarks.bounded_observation import UpperAccumulator, verify_upper_prefix, UpperStepCertificate
    from scripts.benchmarks.native_observation import charge_upper_policy, charge_upper_term
    observer, calls = _synthetic_charge_observer(monkeypatch)
    def prefixes(): return {key: ChargePrefix.start((1,)*3) for key in ('raw_polynomial', 'same_state_affine_tangent')}
    exact, old = observer.charge_evidence((None,), (None, None), prefixes(), absolute_error=Fraction(1, 10000), charge_budget=1)
    assert calls == [{}] and 'upper_sum' not in old and 'terminal_partition' not in old
    accumulator = UpperAccumulator(charge_upper_policy('a'*64))
    current, certificates, terms = prefixes(), [], []
    for index in range(1, 4):
        current, evidence = observer.charge_evidence((None,), (None, None), current,
            absolute_error=Fraction(1, 10000), charge_budget=1, charge_accumulator=accumulator)
        assert calls[-1] == {'partition_cells': 4}
        upper = evidence['upper_sum']
        term = charge_upper_term('a'*64, 'c'*64, 'b'*64, evidence['clock'], evidence['ledgers'])
        assert term.interval_sha256 == upper['interval_sha256']
        assert upper['index'] == index and accumulator.count == index
        terms.append(term)
        certificates.append(UpperStepCertificate(upper['policy_sha256'], index, term,
            upper['upper_numerators'], upper['rounded_terms'], upper['slack_upper_bounds']))
        for key, ledger in evidence['ledgers'].items():
            assert ledger['signed_integral_C'][1].center == Fraction(-1, 7)
            for name in ('signed_integral_C', 'signed_saved_defect_C', 'interval_reference_error_C', 'interval_total_bound_C'):
                assert ledger[name] == old['ledgers'][key][name]
            assert current[key].absolute_defects[0] == current[key].absolute_defects[2] == 0
            assert 0 <= current[key].absolute_defects[1]-index*exact[key].absolute_defects[1] < index*accumulator.policy.quantum
        # A fresh segment observer shares this accumulator; its history cannot reset.
        observer, calls = _synthetic_charge_observer(monkeypatch)
    retained = []
    verified = verify_upper_prefix(accumulator.policy, certificates, terms, retain=retained.append)
    assert verified.count == 3 and len(retained) == 3
    assert all(v <= accumulator.policy.max_slack_per_channel for v in verified.slack_upper_bounds)
    with pytest.raises(ContractError, match='upper_prefix_reset'):
        observer.charge_evidence((None,), (None, None), prefixes(), absolute_error=Fraction(1, 10000),
                                  charge_budget=1, charge_accumulator=accumulator)


def test_charge_refinement_rejects_foreign_source_before_observation(monkeypatch):
    from scripts.benchmarks.interval_observation import ChargePrefix
    from scripts.benchmarks.bounded_observation import UpperAccumulator
    from scripts.benchmarks.native_observation import charge_upper_policy
    observer, calls = _synthetic_charge_observer(monkeypatch)
    prefixes = {key: ChargePrefix.start((1,)*3) for key in ('raw_polynomial', 'same_state_affine_tangent')}
    with pytest.raises(ContractError, match='upper_source_policy'):
        observer.charge_evidence((None,), (None, None), prefixes, absolute_error=Fraction(1, 10000), charge_budget=1,
                                  charge_accumulator=UpperAccumulator(charge_upper_policy('d'*64)))
    assert calls == []


def test_charge_refinement_retains_large_signed_integer_words(monkeypatch):
    from scripts.benchmarks.interval_observation import ChargePrefix, Enclosure
    from scripts.benchmarks.native_history import decode_integer_values, INTEGER_HEX_TAG
    from scripts.benchmarks.native_observation import charge_upper_term
    observer, _ = _synthetic_charge_observer(monkeypatch)
    prefixes = {key: ChargePrefix.start((1,)*3) for key in ('raw_polynomial', 'same_state_affine_tangent')}
    _, evidence = observer.charge_evidence((None,), (None, None), prefixes,
                                          absolute_error=Fraction(1, 10000), charge_budget=1)
    # Encoding-only fixture, not a physical/arithmetic acceptance record.
    signed = Fraction(-1, 2**20000+3)
    ledgers = evidence['ledgers']
    ledgers['raw_polynomial']['signed_saved_defect_C'] = (Enclosure(signed), Enclosure(0), Enclosure(0))
    term = charge_upper_term('a'*64, 'c'*64, 'b'*64, evidence['clock'], ledgers)
    assert INTEGER_HEX_TAG.encode() in term.signed_evidence
    decoded = decode_integer_values(json.loads(term.signed_evidence), {'max_integer_bits': 65536})
    value = decoded['ledgers']['raw_polynomial']['signed_saved_defect_C']['tuple'][0]['center']
    assert Fraction(*value) == signed
    ledgers['raw_polynomial']['signed_saved_defect_C'] = (Enclosure(-signed), Enclosure(0), Enclosure(0))
    assert charge_upper_term('a'*64, 'c'*64, 'b'*64, evidence['clock'], ledgers).interval_sha256 != term.interval_sha256


@lru_cache(maxsize=2)
def independent_test_support(filename):
    # The old simulator also has a package named tests. Resolve this test's
    # own support file explicitly instead of depending on package shadowing.
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location('solarlab_observation_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(request, **values):
    def encode(value):
        if isinstance(value, Fraction): return [value.numerator, value.denominator]
        if isinstance(value, bytes): return {'bytes_sha256': sha256(value).hexdigest(), 'length': len(value)}
        if isinstance(value, np.ndarray): return value.tolist()
        if hasattr(value, 'payload'): return value.payload()
        raise TypeError(type(value).__name__)
    if filename := os.environ.get('NATIVE_MAP_CASE_LOG'):
        with Path(filename).open('a') as file:
            file.write(json.dumps({'test': request.node.nodeid, 'new_native_steps': 0, **values},
                                  default=encode, allow_nan=False)+'\n')


def context(case):
    model = AffineCoupledSlab(SlabDefinition.from_plan(PLAN, REPO, case, area=0.37), 8)
    mapping = AffineVoltageMap(model)
    segments = protocol_from_plan(PLAN, case)
    declaration = {'case_id': case, 'segments': [asdict(s) for s in segments],
                   'controls': {'test_scope': 'no_solver'}, 'budgets': {}, 'voltage_lift_map': mapping.payload()}
    sampling = AffineSamplingContext(model, declaration)
    return mapping, tuple(VoltageLiftSegmentAdapter(VoltageLiftAdapter(mapping), sampling, s) for s in segments)


def raw_polynomials(mapping):
    raw = [Polynomial() for _ in mapping.columns]
    for name in ('n_m3', 'p_m3', 'phi_V'):
        offset = mapping.model.layout.offsets[name]
        sign = -1 if name == 'p_m3' else 1
        for local, row in enumerate(range(offset.start, offset.stop)):
            raw[row] = Polynomial((sign*Fraction(local, 2**16), sign*Fraction(local, 2**20)))
    return tuple(raw)


def middle_clock(segment):
    tn = float(segment.start+(segment.end-segment.start)*0.75)
    previous = float(segment.start+(segment.end-segment.start)*0.5)
    return AcceptedClock(previous, tn, tn-previous, 1, 2, 1, segment.start, segment.end, segment.id)


@pytest.mark.parametrize('case', CASES)
def test_exact_map_and_public_full_word_rate(case, request):
    mapping, bindings = context(case)
    binding = bindings[1]
    clock = middle_clock(binding.segment)
    raw = raw_polynomials(mapping)
    prepared = prepare_physical_polynomial(binding, raw, clock)
    z = np.asarray([float(p.at(0)) for p in raw]); z[0] = -0.0
    zdot = np.asarray([float(p.derivative().at(0)/clock.hused) for p in raw])
    inputs, input_rate = binding.segment.inputs(float(clock.tn))
    point, _ = mapping.trial(z, float(clock.tn), inputs)
    reading = compare_physical_readback(binding, prepared, point, z, zdot, side='continuous')
    ev = prepared.input_errors[0]['uniform_evaluation_bound']
    for row, error in enumerate(reading['physical_state_absolute_error']):
        assert error <= abs(rational(mapping.lift[row, 0]))*ev+2*reading['state_action_error_bounds'][row]
    expected_rates = tuple(abs(rational(mapping.columns[i])*(rational(zdot[i])-p.derivative().at(0)/clock.hused))
                           for i, p in enumerate(raw))
    assert reading['physical_rate_absolute_error'] == expected_rates
    assert reading['raw_z_words'] == z.tobytes()
    assert len(reading['physical_rate_words']) == 4
    assert prepared.path.clock.hused != 1
    with pytest.raises(ContractError, match='event_side'):
        compare_physical_readback(binding, prepared, point, z, zdot, side='left')
    log(request, case=case, clock=clock.payload(), mapped_path=prepared.path.identity,
        point_identity=point.identity, voltage_bound=ev, rate_errors=expected_rates,
        raw_word_hash=sha256(reading['raw_z_words']).hexdigest(), actual_backend_executed=False)


@pytest.mark.parametrize('case', CASES)
def test_actual_full_protocol_inputs_and_zero_operations(case, request):
    mapping, bindings = context(case)
    rows = []
    for binding in bindings:
        clock = middle_clock(binding.segment)
        bounds = input_evaluation_bounds(binding.segment, clock)
        starts, _ = binding.segment.inputs(binding.segment.start)
        for fraction in (0., 0.125, 0.5, 0.75, 1.):
            t = binding.segment.start+(binding.segment.end-binding.segment.start)*fraction
            actual, _ = binding.segment.inputs(t)
            for index, row in enumerate(bounds):
                line = rational(starts[index])+row['stored_slope']*(rational(t)-rational(binding.segment.start))
                assert abs(rational(actual[index])-line) <= row['uniform_evaluation_bound']
        if binding.segment.id != bindings[1].segment.id or case == CASES[1]:
            assert all(row['uniform_evaluation_bound'] == 0 for row in bounds)
        prepared = prepare_physical_polynomial(binding, raw_polynomials(mapping), clock)
        propagation = propagated_input_error(binding, prepared)
        assert propagation['native_admitted'] is False
        assert all(v == 0 for v in propagation['affine_observable_error']['storage'])
        assert propagation['affine_observable_error']['body_charge'] == (0,)
        assert propagation['raw_charge_integral_error_C'][1:] == (0, 0)
        rows.append({'segment': binding.segment.id, 'inputs': [dict(x) for x in bounds],
                     'voltage_error_V': propagation['voltage_error_V'],
                     'raw_current_error_A': propagation['raw_terminal_current_error_A'],
                     'forcing_errors': propagation['individual_carrier_generation_error_particles_s']})
    log(request, case=case, complete_original_protocol_end_s=bindings[-1].segment.end, rows=rows)


@pytest.mark.parametrize('case', CASES)
def test_correlated_voltage_error_reaches_carrier_and_ion_currents(case, request):
    mapping, bindings = context(case); binding = bindings[1]
    prepared = prepare_physical_polynomial(binding, raw_polynomials(mapping), middle_clock(binding.segment))
    observer = SlabPathObserver(mapping.model, prepared.path)
    carrier, ion = voltage_current_sensitivity(mapping, observer)
    arithmetic = BallIntegrator(bits=512)
    delta = Fraction(1, 2**40)
    phi = mapping.model.layout.offsets['phi_V']
    errors = []
    for sign in (-1, 1):
        fields = dict(prepared.path.fields)
        fields['phi_V'] = tuple(p+sign*delta*rational(mapping.lift[phi.start+i, 0])
                                for i, p in enumerate(fields['phi_V']))
        shifted = replace(prepared.path, fields=fields,
                          inputs=(prepared.path.inputs[0]+sign*delta, prepared.path.inputs[1]),
                          coefficient_identity='explicit_non_native_sensitivity_control')
        perturbed = SlabPathObserver(mapping.model, shifted)
        for u in (-1, Fraction(-1, 2), 0):
            for family, coefficients in enumerate((carrier, ion)):
                for i, coefficient in enumerate(coefficients):
                    difference = arithmetic.point(lambda x, a: perturbed.fluxes(arithmetic, x, a)[family][i]
                                                                -observer.fluxes(arithmetic, x, a)[family][i], u)
                    assert abs(difference.center) <= coefficient*delta+difference.radius
                    errors.append({'sign': sign, 'coordinate': u, 'family': family, 'face': i,
                                   'difference': difference, 'bound': coefficient*delta})
    log(request, case=case, explicit_perturbation_V=delta, face_checks=errors,
        scope='uniform sensitivity of the fixed physical formulas; not a native path')


def packet(*, size=3, previous=0.0, tn=0.25, hused=None, owner='synthetic-owner-A', generation=1,
           nsteps=1, previous_y=None, endpoint_y=None, previous_rate=None, endpoint_rate=None):
    """Clearly synthetic interface fixture, never a historical/native record."""
    hused = float(tn-previous) if hused is None else float(hused)
    z0 = np.zeros(size) if previous_y is None else np.asarray(previous_y)
    z1 = np.zeros(size) if endpoint_y is None else np.asarray(endpoint_y)
    phi = np.vstack((z1, z1-z0)); psi = np.array([hused])
    rates = np.asarray((z1-z0)/hused)
    source = ('sksundae.ida.accepted-step-observation.v1', 'scikit-sundae=1.1.3',
              'explicit-synthetic-unit-fixture-no-build', 'no-extension-executed', ('double',),
              HEADER, sha256(b'synthetic config').hexdigest(), ('no-native-library',))
    binding_id = sha256(repr(source).encode()).hexdigest()
    stamp = {'nsteps': nsteps, 'tn': float(tn), 'hused': hused, 'kused': 1, 'hh': hused,
             'kk': 1, 'statuses': tuple((name, 0) for name in STAMP_NAMES)}
    def point(time, steps, values, supplied_rate):
        y, yp = values.tobytes(), (rates if supplied_rate is None else np.asarray(supplied_rate, dtype=float)).tobytes()
        return {'owner': owner, 'generation': generation, 'nsteps': steps, 'internal_t': float(time),
                'output_t': float(time), 'native_status': 0, 'raw_y': y, 'raw_yp': yp,
                'identity': (owner, generation, steps, float(time).hex(), float(time).hex(), sha256(y+yp).hexdigest())}
    prev, end = point(previous, nsteps-1, z0, previous_rate), point(tn, nsteps, z1, endpoint_rate)
    result = {'schema': source[0], 'binding': {'source': source, 'identity': binding_id},
              'owner': owner, 'generation': generation, 'native_before': copy.deepcopy(stamp),
              'native_after': copy.deepcopy(stamp), 'step_key': (binding_id, owner, generation, nsteps,
              float(tn).hex(), hused.hex(), 1, hused.hex(), 1), 'predecessor': prev, 'endpoint': end,
              'dtype': '<f8', 'shape': (size,), 'raw_y': end['raw_y'], 'raw_yp': end['raw_yp'],
              'dky': (end['raw_y'], end['raw_yp']), 'error_weights': np.ones(size).tobytes(),
              'estimated_local_errors': np.zeros(size).tobytes(),
              'vector_statuses': (('IDAGetDky[0]',0),('IDAGetDky[1]',0),('IDAGetErrWeights',0),('IDAGetEstLocalErrors',0)),
              'native_left_terms': (float(tn), -hused), 'clock_gap_terms': (float(tn), -hused, -float(previous)),
              'basis': {'kind': 'ida75_phi_psi_copy_v1', 'dtype': '<f8', 'phi_shape': (2,size), 'phi': phi.tobytes(),
                        'psi_shape': (1,), 'psi': psi.tobytes(), 'status': 0, 'source_header_sha256': HEADER,
                        'source_config_sha256': source[6], 'uround': 2.**-52, 'build_identity': binding_id},
              'predecessor_dky': {'query_t': float(previous), 'query_t_hex': float(previous).hex(),
                         'orders': (0,1), 'buffers': (prev['raw_y'], prev['raw_yp']), 'statuses': (0,0),
                         'native_eligible': True, 'step_before': copy.deepcopy(stamp), 'step_after': copy.deepcopy(stamp)},
              'event_metadata': {'synthetic_fixture_only': True}}
    return result


def read(value):
    return read_native_basis_packet(value, expected_binding_identity=value['binding']['identity'],
                                    expected_header_sha256=HEADER, size=value['shape'][0])


def test_packet_copies_all_words_and_retains_error_metadata(request):
    raw = packet(endpoint_y=np.array([-0.0, .125, .25]))
    frame = read(raw)
    raw['endpoint']['raw_y'] = b'corrupted after read'
    assert frame.snapshot['raw_y'] == np.array([-0.0, .125, .25]).tobytes()
    assert frame.snapshot['basis']['phi'] == np.vstack(([-0.0,.125,.25],[-0.0,.125,.25])).tobytes()
    changed = packet(endpoint_y=np.array([-0.0,.125,.25]))
    changed['estimated_local_errors'] = np.ones(3).tobytes()
    assert read(changed).identity != frame.identity
    with pytest.raises(TypeError): frame.snapshot['owner'] = 'changed'
    log(request, packet_identity=frame.identity, origin='synthetic interface fixture',
        native_build_or_trajectory_proven=False, raw_words=frame.snapshot['raw_y'])


@pytest.mark.parametrize('corruption', ['no_basis','stamp_name','vector_order','failure_bytes','dtype','step_key','endpoint_hash','nonfinite_phi'])
def test_packet_rejects_incomplete_or_inconsistent_contract(corruption, request):
    value = packet()
    if corruption == 'no_basis': value.pop('basis')
    elif corruption == 'stamp_name':
        for key in ('native_before','native_after'):
            value[key]['statuses'] = (('WrongGetter',0),*value[key]['statuses'][1:])
    elif corruption == 'vector_order': value['vector_statuses'] = tuple(reversed(value['vector_statuses']))
    elif corruption == 'failure_bytes':
        value['predecessor_dky'].update(statuses=(-25,0), native_eligible=False)
    elif corruption == 'dtype': value['dtype'] = '>f8'
    elif corruption == 'step_key': value['step_key'] = ('wrong',*value['step_key'][1:])
    elif corruption == 'endpoint_hash': value['endpoint']['identity'] = (*value['endpoint']['identity'][:-1],'wrong')
    elif corruption == 'nonfinite_phi': value['basis']['phi'] = np.full((2,3),np.nan).tobytes()
    with pytest.raises(ContractError) as error:
        read(value)
    log(request, corruption=corruption, actual_rejection=str(error.value), synthetic_only=True)


def test_packet_source_required_and_strip_getter_failure_retained(request):
    mapping, bindings = context(CASES[1]); binding = bindings[1]
    value = packet(size=mapping.model.layout.size, previous=.125, tn=.1875, hused=.0625-2.**-56)
    with pytest.raises(ContractError, match='missing_expected_source_pins'):
        read_native_basis_packet(value, expected_binding_identity=None, expected_header_sha256=HEADER, size=mapping.model.layout.size)
    frame = read(value); prepared = prepare_native_frame_map(binding, frame)
    assert prepared.path.clock.strip == Fraction(1,2**56)
    assert prepared.path.clock_policy == 'declared_polynomial_extension'
    value['predecessor_dky'].update(statuses=(-25,0), buffers=(None,value['predecessor_dky']['buffers'][1]),native_eligible=False)
    failure_frame = read(value)
    assert failure_frame.snapshot['predecessor_dky']['statuses'][0] == -25
    with pytest.raises(ContractError, match='strip_getter_not_eligible'):
        prepare_native_frame_map(binding, failure_frame)
    log(request, strip=prepared.path.clock.strip, failed_getter_status_retained=-25,
        native_getter_actually_run=False)


def test_fresh_solver_owner_is_not_a_logical_generation_alias(request):
    a = read(packet(previous=.75, tn=1., owner='synthetic-A', generation=4, nsteps=8))
    b = read(packet(previous=1., tn=1.25, owner='synthetic-B', generation=1, nsteps=1))
    left = ProtocolSegment('first',0.,1.,(0.,0.),(0.,0.))
    right = ProtocolSegment('second',1.,2.,(0.,0.),(0.,0.))
    require_frame_successor(a,b,previous_segment=left,current_segment=right)
    wrong = read(packet(previous=1.,tn=1.25,owner='synthetic-B',generation=5,nsteps=1))
    with pytest.raises(ContractError, match='unbound_restart_epoch'):
        require_frame_successor(a,wrong,previous_segment=left,current_segment=right)
    log(request, original_owner_generation=[a.owner,a.generation],next_owner_generation=[b.owner,b.generation],
        native_execution=False, logical_segment_generation_not_substituted=True)


def supplied_pair(mapping, binding, native, predecessor=None):
    """A supplied-state observation only: no solver, time step or preparation."""
    m = mapping.model
    inputs, adot = binding.segment.inputs(native['time'])
    point, increment = mapping.trial(native['z'], native['time'], inputs, predecessor=predecessor)
    evaluation = m.observation_evaluation(point)
    raw = m.observe(point, mapping.bind_rate(point, native['z'], native['zdot'], adot), adot,
                    evaluation=evaluation)
    tangent = m.observe(point, m.tangent_rate(point, adot, evaluation=evaluation), adot,
                        'physical_tangent', evaluation=evaluation)
    return point, increment, raw, tangent, {'record_sha256': 'synthetic supplied-point fixture'}


def native_row(frame, key):
    p = frame.snapshot[key]
    return dict(time=p['internal_t'], z=np.frombuffer(p['raw_y'],dtype='<f8').copy(),
                zdot=np.frombuffer(p['raw_yp'],dtype='<f8').copy(),success=True,status=p['native_status'],
                message='synthetic interface fixture; no native solver was run')


@pytest.mark.parametrize('restart', ['fresh_owner', 'same_owner_reinit'])
def test_frame_map_clock_event_and_prefix_compose(restart, request):
    from scripts.benchmarks.interval_observation import ChargePrefix, Enclosure

    mapping, bindings = context(CASES[1]); m = mapping.model
    a, b = bindings[:2]
    zeros = np.zeros(m.layout.size); dz = zeros.copy(); dz[0] = 2.**-30
    first = read(packet(size=m.layout.size, previous=0., tn=a.segment.end, owner='A'))
    owner, generation = ('B',1) if restart == 'fresh_owner' else ('A',2)
    second = read(packet(size=m.layout.size, previous=b.segment.start, tn=.125,
                         owner=owner,generation=generation,endpoint_y=dz))
    arithmetic = BallIntegrator()
    previous = AcceptedIntervalObserver(a,first,arithmetic=arithmetic)
    current = AcceptedIntervalObserver(b,second,arithmetic=arithmetic,previous=previous)
    assert previous.prepared.path.clock.owner == 'A'
    assert (current.prepared.path.clock.owner,current.prepared.path.clock.generation) == (owner,generation)
    assert first.snapshot['endpoint']['raw_y'] == second.snapshot['predecessor']['raw_y']
    assert first.snapshot['endpoint']['raw_yp'] != second.snapshot['predecessor']['raw_yp']
    left_native, right_native = native_row(second,'predecessor'), native_row(second,'endpoint')
    current.require_endpoint(left_native,predecessor=True); current.require_endpoint(right_native)
    wrong = dict(left_native,z=left_native['z'].copy()); wrong['z'][0] = 2.**-50
    with pytest.raises(ContractError,match='endpoint_mismatch'):
        current.require_endpoint(wrong,predecessor=True)
    prefix = ChargePrefix.start((1,1,1))
    prefix, _ = prefix.append((Enclosure(Fraction(1,8)),)*3,(Enclosure(0),)*3,(0,0,0))
    prefix, _ = prefix.append((Enclosure(Fraction(-1,16)),)*3,(Enclosure(0),)*3,(0,0,0))
    assert prefix.intervals == 2 and prefix.absolute_defects == (Fraction(3,16),)*3
    # The subsequent accepted step must retain this owner's epoch and the
    # actual predecessor rate, not manufacture one from its new polynomial.
    third = read(packet(size=m.layout.size,previous=.125,tn=.15,owner=owner,generation=generation,
                        nsteps=2,previous_y=dz,endpoint_y=dz,
                        previous_rate=np.frombuffer(second.snapshot['endpoint']['raw_yp'],dtype='<f8')))
    successor = AcceptedIntervalObserver(b,third,arithmetic=arithmetic,previous=current)
    assert successor.prepared.path.clock.nsteps == 2
    restored = read(restore_observation_record(observation_record(second.snapshot)))
    assert restored.identity == second.identity
    log(request,restart=restart,clocks=[previous.prepared.path.clock.payload(),current.prepared.path.clock.payload(),
                                     successor.prepared.path.clock.payload()],
        prefix_absolute=prefix.absolute_defects,all_source_bytes_restored=True,
        raw_rates_at_kink_different=True,native_build_or_trajectory_proven=False)


def manufactured_map():
    """Independent review's one simultaneous moving path, using real SI data."""
    from perovskite_sim.constants import Q
    independent = independent_test_support('interval_fraction_oracle.py')
    expect, p, SOURCE_SHA256 = independent.expect, independent.p, independent.SOURCE_SHA256

    mapping, bindings = context(CASES[1]); binding=bindings[1]; m=mapping.model; d=m.definition
    H, t0, tn = Fraction(1,32),Fraction(1,8),Fraction(5,32)
    starts, slopes = binding.segment.inputs(binding.segment.start)
    voltage = p(rational(starts[0])+rational(slopes[0])*(t0-rational(binding.segment.start)),rational(slopes[0])*H)
    N=rational(d.trap_density)/4
    fields=dict(n=p(N,N/16),p=p(N,-N/32),c=p(rational(d.ion_initial),rational(d.ion_initial)/64),
                f=p(Fraction(1,4),Fraction(1,128)),voltage=voltage)
    k=dict(q=rational(Q),area=rational(d.area),H=H,dx=tuple(map(rational,m.dx)),
           volumes=tuple(map(rational,m.geometry.volumes)),cn=rational(d.capture_n),cp=rational(d.capture_p),
           n1=rational(d.n1),p1=rational(d.p1),c0=rational(d.ion_initial),Nt=rational(d.trap_density),
           mu_n=rational(d.mu_n),mu_p=rational(d.mu_p),D=rational(d.diffusion_ion),vt=rational(d.vt),
           epsilon=rational(d.epsilon))
    expected=expect(k,fields)
    in_u=lambda row: Polynomial((row[0]+row[1],row[1]))
    desired={name:(in_u(fields[key]),)*m.count for name,key in
             [('n_m3','n'),('p_m3','p'),('c_m3','c'),('f','f')]}
    ell=sum(k['dx'],Fraction(0))
    desired['phi_V']=tuple(-in_u(voltage)*sum(k['dx'][:i],Fraction(0))/ell for i in range(m.count))
    a=(in_u(voltage),Polynomial())
    raw=[]
    for variable in m.layout.variables:
        root=m.field(m.reference,variable.id)
        for i in range(variable.shape[0]):
            row=m.layout.offsets[variable.id].start+i
            reference=sum((rational(word[i]) for word in root.words),Fraction(0))
            input_part=sum((rational(mapping.lift[row,j])*(a[j]-rational(mapping.reference_inputs[j]))
                            for j in range(2)),Polynomial())
            raw.append((desired[variable.id][i]-reference-input_part)/rational(mapping.columns[row]))
    clock=AcceptedClock(t0,tn,H,1,2,1,binding.segment.start,binding.segment.end,binding.segment.id)
    prepared=prepare_physical_polynomial(binding,raw,clock)
    assert all(actual==target for name,row in desired.items() for actual,target in
               zip(prepared.path.fields[name],row,strict=True))
    return mapping,binding,prepared,expected,k,fields,SOURCE_SHA256


def test_simultaneous_physical_path_against_independent_fraction_oracle(request):
    mapping,binding,prepared,expected,k,fields,oracle_source=manufactured_map()
    observer=SlabPathObserver(mapping.model,prepared.path); arithmetic=BallIntegrator()
    # A work goal for exact polynomial oracle arithmetic, not a physical gate.
    goal=Fraction(1,2**160)
    result=observer.integrate(arithmetic,goal)
    actual_changes=tuple(b-a for name in ('body_charge','metal_charge') for a,b in
                         zip(observer.affine[name]['left'],observer.affine[name]['right'],strict=True))
    assert actual_changes==tuple(expected['charge_deltas_C'])
    actual_defects={key:tuple(change-value.center for change,value in zip(actual_changes,result[key],strict=True))
                    for key in ('raw_polynomial','same_state_affine_tangent')}
    for key,values in [('raw_polynomial',expected['raw_integrals_C']),
                       ('same_state_affine_tangent',expected['tangent_integrals_C'])]:
        for actual,reference in zip(result[key],values,strict=True):
            assert abs(actual.center-reference)<=actual.radius
    for key,expected_key in [('raw_polynomial','raw_defects_C'),('same_state_affine_tangent','tangent_defects_C')]:
        for defect,reference,ball in zip(actual_defects[key],expected[expected_key],result[key],strict=True):
            assert abs(defect-reference)<=ball.radius
        indices=(0,) if key=='raw_polynomial' else (0,1,2)
        assert all(abs(actual_defects[key][i])>result[key][i].radius for i in indices)
    inventory=observer.affine['ion_inventory']
    assert inventory['right'][0]-inventory['left'][0]==expected['ion_inventory_change_particles']
    evaluate=lambda coefficients,s: sum((v*s**i for i,v in enumerate(coefficients)),Fraction(0))
    point_checks=[]
    for s in (Fraction(0),Fraction(1,2),Fraction(1)):
        gauss=prepared.path.action(mapping.model.linear_forms['state']['gauss_defect'])[0].at(s-1)
        assert gauss==evaluate(expected['gauss_defect_C'],s) and gauss!=0
        with arithmetic.flint.ctx.workprec(arithmetic.bits):
            currents=observer.currents(arithmetic,arithmetic.number(s-1))
            for key,reference_key in [('raw_polynomial_current','raw_current_A'),
                                      ('same_state_affine_tangent_current','tangent_current_A')]:
                for i,value in enumerate(currents[key]):
                    bound=arithmetic.enclosure(value); exact=evaluate(expected[reference_key][i],s)
                    assert abs(bound.center-exact)<=bound.radius
                    point_checks.append(dict(s=s,key=key,side=i,actual=bound,expected=exact))
    assert expected['ion_inventory_change_particles']>0
    assert expected['raw_defects_C'][0]!=0 and expected['raw_defects_C'][1:]==[0,0]
    assert any(v!=0 for v in expected['gauss_defect_C'])
    assert all(v!=0 for v in expected['tangent_defects_C'])
    assert prepared.path.inputs[0].derivative().at(0)/k['H']!=0
    for name in ('n_m3','p_m3','c_m3','f','phi_V'):
        assert any(p.derivative().at(0)!=0 for p in prepared.path.fields[name])
    # A separately named algebraic history of supplied polynomial endpoints.
    native=lambda u: dict(time=float(prepared.path.clock.tn+prepared.path.clock.hused*u),
                          z=np.array([float(p.at(u)) for p in prepared.raw_polynomials]),
                          zdot=np.array([float(p.derivative().at(u)/k['H']) for p in prepared.raw_polynomials]))
    left=supplied_pair(mapping,binding,native(-1)); right=supplied_pair(mapping,binding,native(0),left[0])
    left_read=compare_physical_readback(binding,prepared,left[0],native(-1)['z'],native(-1)['zdot'],side='continuous')
    right_read=compare_physical_readback(binding,prepared,right[0],native(0)['z'],native(0)['zdot'],side='continuous')
    assert len(left_read['physical_rate_words'])==len(right_read['physical_rate_words'])==4
    # The representable synthetic endpoint packet is a different, explicitly
    # labelled path. It must reject the manufactured inventory drift too.
    rounded=read(packet(size=mapping.model.layout.size,previous=float(prepared.path.clock.predecessor),
                        tn=float(prepared.path.clock.tn),previous_y=native(-1)['z'],endpoint_y=native(0)['z']))
    domains=AcceptedIntervalObserver(binding,rounded,arithmetic=arithmetic).path_domain_evidence(
        original_native_request(CASES[1])['budgets'])
    assert not domains['passed'] and not domains['checks']['ion_inventory']
    log(request,oracle_source_sha256=oracle_source,actual_SI_parameters=k,fields_in_s=fields,
        expected=expected,actual_charge_changes=actual_changes,actual_charge_defects=actual_defects,
        integrals=result,point_checks=point_checks,
        source_map=mapping.identity,source_reference=mapping.reference_identity,path_identity=prepared.path.identity,
        native_coefficients_acquired=False,DAE_or_conservation_pass=False,
        rounded_packet_inventory_gate_rejected=True,
        explicit_negative_results=['nonzero ion inventory change','nonzero Gauss','nonzero body and tangent defects'])


@pytest.mark.parametrize('case',CASES)
def test_point_source_readback_bounds_and_current_gate(case,request):
    decimal_currents=independent_test_support('test_interval_observation.py').decimal_currents

    mapping,bindings=context(case); binding=bindings[1]
    clock=middle_clock(binding.segment)
    prepared=prepare_physical_polynomial(binding,raw_polynomials(mapping),clock)
    time=float(clock.tn)
    z=np.array([float(p.at(0)) for p in prepared.raw_polynomials])
    zdot=np.array([float(p.derivative().at(0)/clock.hused) for p in prepared.raw_polynomials])
    # A deliberately different supplied rate is preserved and bounded, not
    # replaced by the polynomial or a tangent before observing currents.
    zdot[mapping.model.layout.offsets['phi_V'].stop-1]+=2.**-24
    native=dict(time=time,z=z,zdot=zdot)
    pair=supplied_pair(mapping,binding,native)
    evidence=readback_current_evidence(binding,prepared,pair,z,zdot,BallIntegrator(),1.e-300)
    assert not evidence['passed']
    assert evidence['input_error_included_once'] and not evidence['native_rate_or_state_changed']
    actual=evidence['readback']['physical_state_enclosures']
    rate_words=tuple(np.frombuffer(word,dtype='<f8') for word in evidence['readback']['physical_rate_words'])
    values={}
    for name,offset in mapping.model.layout.offsets.items():
        values[name]=tuple(Polynomial((actual[i].center,
                         clock.hused*sum((rational(word[i]) for word in rate_words),Fraction(0))))
                          for i in range(offset.start,offset.stop))
    inputs,adot=binding.segment.inputs(time)
    at_point=replace(prepared.path,fields=values,
                     inputs=tuple(Polynomial((rational(a),rational(d)*clock.hused)) for a,d in zip(inputs,adot)),
                     coefficient_identity='independent Decimal point oracle; not a trajectory')
    reference=decimal_currents(mapping.model,at_point,0)
    for row,expected in zip((evidence['raw_reference_A'],evidence['same_state_exact_tangent_reference_A']),
                            (reference[0][1],reference[1][1]),strict=True):
        for value,exact in zip(row,expected,strict=True):
            assert abs(value.center-Fraction(exact))<=value.radius+Fraction(1,10**85)
    log(request,case=case,evidence=evidence,oracle='independent Decimal120 finite-volume point equations',
        oracle_roundoff_allowance=Fraction(1,10**85),native_steps=0)


def original_native_request(case):
    """Historical input for unchanged budgets/comparisons, never a live map."""
    inputs=json.loads(Path(os.environ['INTERVAL_NATIVE_REQUESTS']).read_text())
    return json.loads(Path(inputs[case]).read_text())


def current_controller_input(case):
    """One source-aware factory shared by every new controller/history test."""
    from scripts.benchmarks.coupled_device_prototype import prepare_voltage_lift_native_request, digest

    original=original_native_request(case)
    model=AffineCoupledSlab(SlabDefinition.from_plan(PLAN,REPO,case),8)
    mapping=AffineVoltageMap(model); segments=protocol_from_plan(PLAN,case)
    previous_paths=json.loads(Path(os.environ['LIFT_PREVIOUS_REQUESTS']).read_text())
    previous=json.loads(Path(previous_paths[case]).read_text())
    proposal=prepare_voltage_lift_native_request(mapping,segments,previous)
    for name in ('segments','budgets','controls','observation_times','quadrature'):
        assert proposal[name]==original[name]
    assert digest({k:v for k,v in proposal['weight_certificate'].items() if k!='map_identity'})==digest(
        {k:v for k,v in original['weight_certificate'].items() if k!='map_identity'})
    return model,mapping,segments,proposal,original,previous


@pytest.mark.parametrize('case',CASES)
def test_interval_errors_prefix_and_reconstruction_history(case,monkeypatch,request):
    from scripts.benchmarks.interval_observation import ChargePrefix, endpoint_charge_mismatch
    from scripts.benchmarks.coupled_device_prototype import VoltageLiftHistory
    import scripts.benchmarks.native_observation as implementation

    mapping,bindings=context(case); binding=bindings[1]; m=mapping.model
    selected=middle_clock(binding.segment)
    frame=read(packet(size=m.layout.size,previous=float(selected.predecessor),tn=float(selected.tn),nsteps=2))
    observer=AcceptedIntervalObserver(binding,frame,arithmetic=BallIntegrator())
    left_native=native_row(frame,'predecessor'); right_native=native_row(frame,'endpoint')
    left=supplied_pair(mapping,binding,left_native); right=supplied_pair(mapping,binding,right_native,left[0])
    budget=original_native_request(case)['budgets']['charge_C']
    prefixes={name:ChargePrefix.start((budget,)*3) for name in ('raw_polynomial','same_state_affine_tangent')}
    before=dict(prefixes)
    after,evidence=observer.charge_evidence(left,right,prefixes,absolute_error=rational(budget)/4096,
                                           charge_budget=budget)
    assert prefixes==before and all(p.intervals==1 for p in after.values())
    endpoint=endpoint_charge_mismatch(m,observer.prepared.path,left[0],right[0])
    inputs=propagated_input_error(binding,observer.prepared)
    for name,ledger in evidence['ledgers'].items():
        terms=ledger['error_components_C']
        for i in range(3):
            assert terms['endpoint'][i]>=endpoint[i]
            assert ledger['interval_reference_error_C'][i]==sum(row[i] for row in terms.values())
            assert ledger['prefix_reference_error_C'][i]==ledger['interval_reference_error_C'][i]
        # Input-at-endpoint uncertainty has not been appended a second time.
        assert set(terms)=={'endpoint','input','clock_strip','raw_tangent_departure','integration'}
        assert terms['input']==inputs['raw_charge_integral_error_C' if name=='raw_polynomial'
                                      else 'tangent_charge_integral_error_C']
    sample=observer.reconstruction(float((selected.predecessor+selected.tn)/2))
    assert sample['native_output'] is False
    history=VoltageLiftHistory(mapping)
    p,delta,rate,saved=history.build_sample(sample['z'],sample['time'],binding.segment.inputs(sample['time'])[0],
        left[0],sample['zdot'],binding.segment.inputs(sample['time'])[1],origin='declared_polynomial')
    restored=history.restore(history.reference_record,saved,left[0])
    assert restored[0].identity==p.identity
    assert saved['origin']=='declared_polynomial' and len(saved['physical_rate_words_hex'])==4
    corrupted=dict(inputs); corrupted.pop('raw_charge_integral_error_C')
    monkeypatch.setattr(implementation,'propagated_input_error',lambda *args:corrupted)
    with pytest.raises(ContractError,match='missing_input_error_term:raw_charge_integral_error_C'):
        observer.charge_evidence(left,right,prefixes,absolute_error=rational(budget)/4096,charge_budget=budget)
    log(request,case=case,observation=evidence,prefixes_unchanged_on_input_object=True,
        sample_origin=saved['origin'],history_original_bytes_restored=True,
        missing_input_term_rejected=True,full_device_qualification=False)


@pytest.mark.parametrize('case',CASES)
def test_controller_requires_new_source_bound_review_before_backend(case,request):
    from scripts.benchmarks.coupled_device_prototype import (
        digest, run_voltage_lift_native_pilot,
    )
    import sys

    # The old request belongs to its original frozen source location. Build
    # a new source-bound request using the public material/geometry checks;
    # never rewrite its historical model/map identities to make it validate.
    m,mapping,segments,proposal,original,previous=current_controller_input(case)
    ready=copy.deepcopy(proposal)
    pins={str(REPO/'scripts/benchmarks'/name):sha256((REPO/'scripts/benchmarks'/name).read_bytes()).hexdigest()
          for name in ('coupled_device_prototype.py','native_observation.py','interval_observation.py')}
    admission={'request_sha256':digest(proposal),'map_identity':mapping.identity,
               'voltage_lift_native_authorized':True,'coordinator_message':'msg_unit_fixture_no_native_authority',
               'source_sha256':pins}
    with pytest.raises(ContractError,match='full_word_native_observation_policy_unqualified'):
        run_voltage_lift_native_pilot(mapping,segments,proposal,admission,lambda _:pytest.fail('unadmitted output'))
    assert 'sksundae' not in sys.modules
    policy=prepare_interval_observation_policy(proposal,binding_identity='a'*64,header_sha256=HEADER,
                                               backend_modules={})
    proposal['interval_observation']=policy
    admission.update(request_sha256=digest(proposal),interval_observation_authorized=True,
                     interval_observation_policy_sha256=digest(policy))
    with pytest.raises(ContractError,match='independent_review_missing'):
        run_voltage_lift_native_pilot(mapping,segments,proposal,admission,lambda _:pytest.fail('unreviewed output'))
    assert 'sksundae' not in sys.modules
    assert proposal['segments']==original['segments'] and proposal['budgets']==original['budgets']
    assert proposal['observation_times']==original['observation_times'] and proposal['quadrature']==original['quadrature']
    assert proposal['controls']==original['controls']
    assert digest({k:v for k,v in proposal['weight_certificate'].items() if k!='map_identity'})==digest(
        {k:v for k,v in original['weight_certificate'].items() if k!='map_identity'})
    assert proposal['map_identity']!=original['map_identity']
    if folder:=os.environ.get('INTERVAL_PREPARED_REQUESTS'):
        destination=Path(folder)/(case+'.json');destination.parent.mkdir(exist_ok=True)
        with destination.open('x') as f:
            json.dump(ready,f,indent=2,allow_nan=False);f.write('\n')
    log(request,case=case,complete_protocol_s=segments[-1].end,all_original_limits_retained=True,
        old_request_digest=digest(original),new_request_digest=digest(ready),
        previous_request_digest=digest(previous),new_source_path=m.definition.source_path,
        old_source_path=original['numeric_packet']['definition']['source_path'],
        missing_policy_and_review_rejected=True,native_backend_imported=False,native_steps=0)


@pytest.mark.parametrize('case',CASES)
def test_current_request_metadata_keeps_all_numeric_words(case,request):
    from scripts.benchmarks.coupled_device_prototype import digest

    m,mapping,segments,proposal,original,previous=current_controller_input(case)
    paths=json.loads(Path(os.environ['INTERVAL_PRIOR_PREPARED_REQUESTS']).read_text())
    before=json.loads(Path(paths[case]).read_text())
    def numeric_words(value,path=()):
        if isinstance(value,dict):
            return [row for key,item in sorted(value.items()) for row in numeric_words(item,(*path,key))]
        if isinstance(value,(list,tuple)):
            return [row for i,item in enumerate(value) for row in numeric_words(item,(*path,str(i)))]
        if type(value) is float:return [(path,'float',value.hex())]
        if type(value) in (int,bool):return [(path,type(value).__name__,str(value))]
        if isinstance(value,str) and value.lstrip('-').startswith('0x'):
            return [(path,'stored_hex_word',value)]
        return []
    before_words,after_words=numeric_words(before),numeric_words(proposal)
    assert before_words==after_words
    assert 'declared_polynomial' in proposal['history_policy']
    assert 'counterfactual' in proposal['projection_policy']['interval']
    assert 'E32+abs' not in proposal['projection_policy']['interval']
    assert 'request.interval_observation' in proposal['observation_policy_authority']
    assert proposal['full_word_consumer_qualification']['native_policy_admitted'] is False
    if folder:=os.environ.get('INTERVAL_PREPARED_REQUESTS'):
        destination=Path(folder)/(case+'.json');destination.parent.mkdir(exist_ok=True)
        with destination.open('x') as f:json.dump(proposal,f,indent=2,allow_nan=False);f.write('\n')
    log(request,case=case,numeric_word_count=len(after_words),all_numeric_words_equal=True,
        previous_request_digest=digest(before),current_request_digest=digest(proposal),
        original44_digest=digest(original),original_affine_digest=digest(previous),
        actual_source_path=m.definition.source_path,actual_map_identity=mapping.identity,
        complete_protocol_s=segments[-1].end,native_admitted=False)

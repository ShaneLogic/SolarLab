# Experiment protocol schema

`perovskite_sim.experiments.protocol.ExperimentProtocol` records the physical
history and measurement sampling that define an experiment. It is immutable,
JSON serializable, and has a canonical SHA-256 hash.

The protocol is evidence of **what was requested and executed**. It is not a
steady-state, mesh, tolerance, conservation, or external-validation
certificate. In particular, a finite-time `dc_settle` remains finite-time even
when its integrator succeeds.

## Required evidence

Every protocol records:

- experiment kind and schema version;
- initial-state source;
- pre-bias voltage, soak time, and per-point dwell where applicable;
- ordered dark/light history, including generation source or hash;
- active device temperature;
- scan axis, direction, endpoints, and voltage rate where applicable;
- AC DC bias, amplitude, cycles, extraction cycles, and time sampling;
- DC settle criterion and any declared residual/current thresholds;
- the full state-advancing open-circuit search, where applicable: coarse
  bracket, per-step dwell, bisection tolerance/count/dwell, final settle,
  warm-start and fallback policy;
- exact output sampling values;
- whether the history came from an implicit compatibility call.

Nested collections are tuples of frozen dataclasses. Construction and parsing
reject non-finite values, missing schema keys, and unknown schema keys.

## Compatibility and research-strict modes

Existing experiment calls retain their numerical defaults. With no protocol
argument, the result carries a generated declaration with
`implicit_legacy_protocol=True`:

```python
result = run_jv_sweep(stack, n_points=40, v_rate=1.0)
assert result.protocol.implicit_legacy_protocol
```

This output is reproducible metadata, but it is not an explicit research
history. Research-strict execution requires a protocol built independently
from the same requested inputs:

```python
from perovskite_sim.experiments.jv_sweep import (
    build_jv_experiment_protocol,
    run_jv_sweep,
)

protocol = build_jv_experiment_protocol(
    stack,
    n_points=40,
    v_rate=1.0,
    V_max=1.3,
)
result = run_jv_sweep(
    stack,
    n_points=40,
    v_rate=1.0,
    V_max=1.3,
    experiment_protocol=protocol,
    protocol_mode="research_strict",
)
```

The execution compares all protocol fields except the legacy provenance flag.
A mismatch fails before the solver runs. An adaptive J-V `V_max` ladder cannot
be predeclared exactly and therefore remains compatibility-only; a strict run
must use a fixed voltage window.

Equivalent builders exist for TPV, EQE, Suns-Voc, and impedance:

- `build_tpv_experiment_protocol`
- `build_eqe_experiment_protocol`
- `build_suns_voc_experiment_protocol`
- `build_impedance_experiment_protocol`

TPV and Suns-Voc embed a frozen `VocSearchProtocol`. Their numerical
`_find_voc` path reads the coarse grid, dwell times, bisection limit/tolerance,
final settle and warm-start behavior from that object, so changing any
state-advancing search field changes the protocol hash. J-V scan history
records all executed point dwells on both branches (`2 * n_points * dwell`),
including the first point of each branch.

The direct endpoints and asynchronous `/api/jobs` path apply the same semantic
gate. For J-V, impedance, TPV, Suns-Voc and EQE, an explicit protocol that does
not match normalized request parameters is rejected with HTTP 422 before the
job registry receives a worker.

The existing impedance result retains its Phase 0 `ImpedanceProtocol` for API
compatibility. The unified schema is available at
`result.protocol.experiment_protocol`.

## Serialization and hashing

```python
payload = protocol.to_dict()
canonical_json = protocol.to_json()
sha256 = protocol.protocol_hash

restored = ExperimentProtocol.from_json(canonical_json)
assert restored.protocol_hash == sha256
```

Canonical JSON sorts mapping keys, uses compact separators, normalizes signed
zero, and rejects NaN and infinity. The hash changes when an execution-defining
field changes, including voltage samples, illumination history, AC amplitude,
or a fixed-generation array hash.

## Evidence boundary

A protocol hash allows results with identical history to be grouped and
results with different history to be rejected from one convergence ladder. It
does not promote a result beyond its numerical certificate. Publication and
validation records must still carry the configuration hash, Git commit,
environment, grid, tolerance policy, solver acceptance, conservation and
positivity diagnostics, contact certificate, evidence tier, and limitations.

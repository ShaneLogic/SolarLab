# External Series/Shunt DC Layer

## Scope

The external-circuit layer maps a certified intrinsic J-V curve into terminal
coordinates without changing drift-diffusion contacts or material parameters.
It uses area-normalized resistance (`ohm m2`) and the photovoltaic convention
that delivered current is positive:

```text
J_shunt   = V_junction / R_shunt
J_terminal = J_device - J_shunt
V_terminal = V_junction - J_terminal * R_series
```

`R_shunt=None` disables the leakage branch. `R_series=0` and a disabled shunt
are the exact zero-coupling limit: terminal arrays and the default 1-sun
metrics are preserved exactly.

## Python API

```python
from perovskite_sim.experiments.external_circuit import (
    ExternalCircuitProtocol,
    apply_external_circuit,
)

circuit = ExternalCircuitProtocol(
    series_resistance_ohm_m2=5e-4,
    shunt_resistance_ohm_m2=0.2,
)
terminal = apply_external_circuit(intrinsic_jv_result, circuit)
```

The result keeps junction voltage, intrinsic device current, shunt current,
series drop, terminal voltage/current/power, both branch metrics, the canonical
circuit hash, and a content hash of the complete source result. Arrays are read-only.
By default an incomplete or uncertified intrinsic `JVResult` is rejected.
The result also records `incident_power_W_m2` and a mapping hash that binds the
source curve, circuit protocol, metric convention, and incident power.

## Backend API

`POST /api/jv/external-circuit` accepts the normal `JVRequest` fields plus
`external_circuit_protocol` and `incident_power_W_m2`. The circuit payload must
be the exact output of `ExternalCircuitProtocol.to_dict()`; missing and unknown
keys return HTTP 422 before a drift-diffusion solve starts. The response is a
separate `ExternalCircuitJVResult`, so the existing `/api/jv` response and the
workstation's default intrinsic path remain unchanged.

## Evidence Boundary

This checkpoint is a DC algebraic post-processing layer. It does not change the
state-advancing voltage history, solve a circuit-coupled transient DAE, add
contact resistance, or infer resistance from a material/contact calibration.
If the terminal-voltage mapping folds or changes branch orientation, it fails
closed instead of sorting points into an apparently physical curve.

Unit tests pin the two balance equations, reverse-branch orientation, exact
zero-coupling values, canonical protocol round-trip/hash, immutable evidence,
source-curve binding, metric trends, backend fail-closed behavior, and
invalid/folded input rejection.

## Numerical Certificate

The pre-registered `external-series-shunt-dc-v1` matrix ran 9/9 cells at source
commit `301291b` without solver or quality-gate failures, but correctly returned
`partial` (certificate
`3eb975c938cd8bae2d022a37a4f6d2732a149d4dd63bb62ae7f4e0042ab5dc70`).
Its full 0--1.2 V junction-sampled trace includes a deep-forward-injection point
near `-17.8 kA/m2`, producing a `3.55 V` series drop. At the finest tolerance,
N=30 to N=40 changed that tail by `0.008398 V` and `0.167966` in the
250 A/m2-normalized current, above the frozen v1 limits. Those results were not
discarded and the v1 limits were not relaxed.

The resolved `external-series-shunt-dc-operating-quadrant-v2` lane instead
pre-registers the device-operating claim directly: both branches are linearly
sampled on 21 fixed `V_terminal/Voc_terminal` points from 0 to 1, with separate
terminal Voc, Jsc, FF, and PCE observables. The source-clean 20/30/40 grid by
1/0.1/0.01 componentwise-atol matrix at commit `2392ba3` completed 9/9 cells,
0 failed, 0 missing, and is `certified`:

- run ID: `eb651533c03a91cc41ff4c71a93030e560cabb8276ba56e79451ca6fb7bea52c`
- certificate SHA-256: `a9f6d63a229ec613594d78d73a2ac94e6d0aea756c10e960948dc31123e1bf26`
- protocol SHA-256: `4c70336e3bba3591e92b83f1240f554c0222d9fcb311781a97a818c66f2d6eae`
- finest-grid differences: normalized operating trace `0.001592`, Voc
  `0.2841 mV`, Jsc `0.1335%`, FF `5.87e-5`, and PCE `0.0943%`
- finest-grid tolerance differences: at most `1.32e-11` for the normalized
  operating trace and negligible for the scalar metrics
- every cell: intrinsic and external certificates true, current/voltage
  balance errors exactly zero, zero-coupling exact, both branches monotonic,
  both Voc values bracketed, and PCE loss between about 10.83% and 10.91%

This certificate is internal convergence and algebraic-balance evidence for
the frozen synthetic `R_series=2e-4 ohm m2`, `R_shunt=0.2 ohm m2` protocol. It
does not validate those values against a measured device, certify the excluded
high-injection tail, or add circuit dynamics, self-heating, capacitance,
inductance, or distributed sheet resistance.

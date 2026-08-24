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
invalid/folded input rejection. A registered grid/parameter certificate is the
next evidence layer; until then this is an internally tested research path,
not a validated parasitic model for a measured device.

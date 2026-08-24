# 2D J-V Execution Protocol

## Scope

The 2D J-V driver has a canonical, immutable execution protocol. The
historical frozen-ion call remains compatible and returns a visibly implicit
protocol. The research-only mobile-ion and two-sided interface-SRH modes are
public only through an explicit matching protocol and
`protocol_mode="research_strict"`.

This protocol records a finite-time ascending voltage history. It is not the
1D forward/reverse hysteresis protocol and does not invent a scan rate. Every
listed voltage receives one full fixed-voltage dwell, including the first
point at 0 V.

## Bound Fields

The `jv-2d-execution-protocol-v1` document binds:

- the exact voltage array and dwell time per point;
- dark equilibrium or 1 ms illuminated 1D initial-state preparation;
- illumination source and temperature;
- exact x/y coordinates and physical grain-boundary geometry;
- frozen or single-positive-mobile-ion state topology;
- frozen or blocking ion boundaries and ohmic/Robin carrier contacts;
- off or two-sided cross-node interface SRH;
- carrier-only or carrier/ion/displacement current composition;
- instantaneous fixed-voltage endpoint sampling with `dV/dt=0`;
- Radau relative and absolute tolerances, max-step divisor, RHS budget,
  bisection budget, and ion-inventory tolerance;
- snapshot retention and the implicit/explicit provenance flag.

Canonical JSON uses sorted keys, finite numbers, and exact nested schemas.
Unknown or missing fields fail closed. SHA-256 changes when any execution
field changes.

## Python API

```python
from perovskite_sim.solver.tolerances import ComponentwiseAtol
from perovskite_sim.twod.experiments.jv_sweep_2d import (
    build_jv_2d_execution_protocol,
    run_jv_sweep_2d,
)

kwargs = dict(
    lateral_length=100e-9,
    Nx=4,
    V_max=0.2,
    V_step=0.05,
    lateral_bc="neumann",
    ion_dynamics="single_mobile",
    interface_srh="two_sided_cross_node",
    settle_t=1e-9,
    atol=ComponentwiseAtol(),
)
protocol = build_jv_2d_execution_protocol(stack, microstructure, **kwargs)
result = run_jv_sweep_2d(
    stack,
    microstructure,
    **kwargs,
    jv_2d_protocol=protocol,
    protocol_mode="research_strict",
)
```

`JV2DResult` returns the resolved protocol, protocol hash through the protocol
object, complete mobile-current reports, per-point ion inventory/bound
diagnostics, and per-point interface surface-rate/clamp reports.

## Backend Contract

`POST /api/jobs` with `kind="jv_2d"` accepts `ion_dynamics`,
`interface_srh`, `componentwise_atol`, `protocol_mode`, and
`jv_2d_protocol` in `params`. Extended topology is checked before worker
submission. Missing strict acknowledgement, schema errors, topology
incompatibility, and protocol/execution mismatch return HTTP 422 and no job
is created.

The backend response includes canonical protocol/hash, compact terminal
current evidence, ion diagnostics, interface clamp counts, and `P_ion` in
saved snapshots. The workstation remains on the compatibility frozen lane
until a protocol-preview/acknowledgement workflow is added.

## Evidence Boundary

Unit tests cover strict round-trip, unknown fields, semantic contradictions,
hash sensitivity, default compatibility, strict gating, three-block
reabsorption slicing, and backend pre-submit mismatch rejection. Real small
two-point Radau runs cover mobile current and the combined
grain-boundary/interface/mobile topology.

This is public execution provenance, not a numerical certificate. A new
content-addressed grid/tolerance lane must still close J-V convergence, ion
inventory, positive state, interface clamp status, current decomposition, and
all-face Maxwell-current gates.

# Ion-aware impedance reference engine

Status: `INTERNAL_TESTED_REFERENCE` as of 2026-08-21. This is not yet an
`INTERNAL_CERTIFIED` impedance lane and is not external validation.

## Scope

`perovskite_sim.experiments.ion_aware_impedance` implements Phase 2 steps 2
and the reference half of step 3. It consumes one exact
`IonAwareDCResult`, re-evaluates its full-MOL DC certificate, and constructs
the small-signal system

```text
(j*omega*M - J) delta_u = b*delta_V
```

with central finite differences. The implementation is deliberately a
reference operator: later analytic or structured Jacobians must reproduce
its columns and responses before they can replace it.

## State and storage contract

The coordinate is a dimensionless log-density increment,
`y_i = y_dc,i * exp(u_i)`. It includes only actual dynamic densities:

- interior electron and hole nodes under fixed-density contacts;
- carrier boundary nodes only when that side uses a finite Robin contact;
- positive-ion nodes with positive reference density and diffusivity;
- negative-ion nodes under the same rule in dual-ion mode.

Fixed contact values and structural zero-ion nodes are excluded. Dynamic
interface-state blocks remain unsupported. Every callback reconstructs the
physical density state, evaluates the full MoL RHS, and re-solves eliminated
Poisson. Consequently, Poisson has no artificial storage row, while its
global state and voltage derivatives enter `J`, `b`, terminal conduction,
and dielectric displacement.

For log increments, the reference mass matrix must be diagonal with
`M_ii = y_dc,i`. The protocol gates both normalized diagonal error and
off-diagonal leakage. Electron, hole, positive-ion and negative-ion storage
responses are integrated with the same dual-cell weights as the conservation
operators.

## Current contract

All impedance currents use the passive convention. The result exposes
all-face complex responses for:

- electron conduction;
- hole conduction;
- positive-ion charge current;
- negative-ion charge current when active;
- total conduction;
- dielectric displacement;
- total admittance and impedance.

The generic `solver.small_signal` result now also retains the finite-difference
`M`, `J`, voltage forcing derivatives, conduction/displacement responses, and
named current components. Existing callbacks that omit components retain the
old numerical path. Named components must sum to the evaluated conduction
current; when present, their differentiated sum is the discrete definition of
total conduction, avoiding a second cancellation-sensitive subtraction. The
ion-aware certificate still gates the returned decomposition closure.

## Protocol and fail-closed gates

`IonAwareImpedanceProtocol` is frozen, strict-schema, canonical JSON. Its
SHA-256 binds:

- DC voltage, illumination and effective temperature;
- exact DC protocol SHA-256 and packed final-state SHA-256;
- ordered frequencies and nominal AC amplitude;
- state/voltage finite-difference steps and refinement ladder;
- current-continuity, linear-solve, perturbation, mass, inventory, and
  current-decomposition limits.

The default finite-difference ladder is `1`, `0.5`, `0.25`. The final response
must pass all-face relative admittance spread `<=5e-4`, componentwise backward
error `<=1e-10`, mass error `<=1e-8`, blocking-ion inventory response
`<=1e-8`, and current-decomposition closure `<=1e-7`. The last two finite-
difference levels must change impedance magnitude by `<=1%` and phase by
`<=0.5 deg`.

Before any linearization, execution rejects a stale DC state hash, mismatched
stack/grid/history/temperature, a currently invalid DC residual certificate,
unsupported interface-state topology, non-positive active densities, or a
stencil crossing the ion site limit. Contact thermodynamics remains a
separate strict axis: the current IonMonger deck is
`compatible_unverified`, so a numerically passing response still has
`certified=false`.

## Usage

```python
import numpy as np

from perovskite_sim.experiments.ion_aware_impedance import (
    build_ion_aware_impedance_protocol,
    run_ion_aware_impedance,
)

protocol = build_ion_aware_impedance_protocol(
    certified_dc_state,
    np.logspace(-3, 3, 25),
    delta_V=0.01,
)
response = run_ion_aware_impedance(
    certified_dc_state.x,
    stack,
    protocol,
    dc_state=certified_dc_state,
)
```

This API is intentionally opt-in and is not yet routed through the legacy
`run_impedance` method selector or backend.

## Current evidence

The real single-ion IonMonger N13 probe at 0.9 V, one sun and frequencies
`1e-3`, `1`, `1e3 Hz` reported:

| Check | Observed | Limit |
|---|---:|---:|
| all-face admittance spread | `2.90e-8` | `5e-4` |
| componentwise backward error | `2.50e-16` | `1e-10` |
| minimum reciprocal condition | `2.01e-6` | machine-singularity gate |
| mass diagonal relative error | `3.80e-11` | `1e-8` |
| mass off-diagonal relative error | `0` | `1e-8` |
| positive-ion inventory response | `1.71e-14` | `1e-8` |
| final-step magnitude change | `7.93e-8` | `1e-2` |
| final-step phase change | `4.92e-7 deg` | `0.5 deg` |

A symmetric dual-ion N13 probe also passed, with both ionic current/storage
blocks active, maximum inventory response `1.07e-13`, all-face spread
`2.07e-8`, and backward error `2.49e-16`. The focused impedance/DC suite is
`85 passed, 1 deselected`; the existing c-Si QF slow regression is `8 passed`.
The repository default suite is `2028 passed, 2 skipped, 263 deselected`.

The N61 single-ion performance probe used 138 dynamic coordinates and 30
frequencies from `1e-4` to `1e6 Hz`, with all three finite-difference levels.
On the recorded single-thread environment, DC preparation took `1.120 s` and
the frequency reference solve took `0.276 s`. It passed with all-face spread
`6.93e-7`, backward error `5.87e-16`, minimum reciprocal condition
`8.76e-10`, inventory response `1.35e-12`, and exact returned current-
decomposition closure. These timings are a local baseline, not a cross-host
performance certificate.

These are implementation tests and probes, not a registered grid/frequency
certificate.

## Remaining Phase 2 work

1. Exact discrete Poisson sensitivity, analytic SG transport, local bulk,
   defect-free, clamp-inactive cross-node/projected, and positive-density
   shared-occupancy/additive-two-sided interface reaction, selective-contact,
   and differentiable CT/PF mobility blocks are implemented; replace the
   remaining unsupported interface/nonlocal frozen-potential differences.
   See
   [ion-aware-structured-jacobian-comparison.md](ion-aware-structured-jacobian-comparison.md).
2. Add device-timescale frequency-window assessment without changing user
   frequencies.
3. Register grid, finite-difference and frequency-coverage matrices and mint
   per-frequency certificates.
4. Cross-check selected frequencies against transient lock-in using the exact
   same DC state and protocol.
5. Only after those gates pass, route the method through `run_impedance`, the
   backend, and frontend diagnostics.

External IonMonger/Driftfusion artifacts and experimental spectroscopy remain
separate validation layers.

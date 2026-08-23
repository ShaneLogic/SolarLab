# Ion-aware impedance reference engine

Status: `REGISTERED_CONTENT_ADDRESSED_CANDIDATE` as of 2026-08-23.
This is a canonical internal numerical certificate with a registered runner
lane; its release artifact and external validation remain separate evidence.

## Scope

`perovskite_sim.experiments.ion_aware_impedance` implements Phase 2 step 2
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
  current-decomposition limits;
- frequency-envelope margin and maximum log-frequency sampling gap.

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

## Frequency-window evidence

Protocol v2 calls the same shared
`experiments.impedance_frequency` assessment used by the public impedance
engine. Each contiguous positive- or negative-ion active region is evaluated
independently using its median diffusivity and equilibrium density, mean
permittivity, and finite-volume region length. The screening scales are

```text
lambda_D = sqrt(epsilon V_T / (q P))
f_dielectric = D / (2 pi lambda_D^2)
f_blocking = D / (pi L lambda_D)
f_diffusion = D / (2 pi L^2).
```

These are model-derived order-of-magnitude frequencies, not fitted circuit
constants. The assessment preserves the historical blocking-frequency bracket
flag and separately reports whether the full diffusion/blocking/dielectric
envelope is bracketed. For each region it recommends

```text
f_min <= min(f_diffusion, f_blocking, f_dielectric) / 10^margin
f_max >= max(f_diffusion, f_blocking, f_dielectric) * 10^margin,
```

with protocol defaults `margin=1 decade` and maximum sampling gap `0.5
decades`. Coverage requires both margins and the sampling-gap gate for every
active region and ion species. The result records per-region bracket flags,
recommended bounds, observed gap, and warnings. It never inserts, removes, or
moves a requested frequency. An uncovered window leaves the linear solve's
`numerically_certified` flag intact but sets the separate
`frequency_window_certified=false`, so combined certification cannot be
claimed from a high-frequency-only sweep.

## Per-frequency and grid certificates

The sweep certificate no longer relies only on global extrema. Every requested
frequency receives an immutable `IonAwareImpedanceFrequencyCertificate` with:

- all-face relative admittance spread, reciprocal condition and componentwise
  backward error;
- the magnitude and phase change for every adjacent state/voltage stencil
  pair, with the finest pair controlling promotion;
- positive- and negative-ion blocking-inventory response;
- current-decomposition closure and the complex electron, hole, positive-ion,
  negative-ion, and net-charge storage responses.

`IonAwareImpedanceGridProtocol` then binds at least three ordered DC states.
Its canonical SHA-256 covers each exact grid-coordinate SHA-256, each packed DC
state and DC-history protocol, the common frequency/stencil request, and the
grid limits. Equal node counts on different clustered grids are therefore not
interchangeable evidence. Execution rejects reordered or replaced grids/states
before AC linearization. All adjacent grid comparisons are retained, while the
finest two grids control promotion at every frequency using default limits of
2% in `|Z|` and 1 degree in phase. Numerical, frequency-window, and contact
thermodynamic certification remain separate axes.

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
`2.07e-8`, and backward error `2.49e-16`. Both species receive independent
frequency evidence.

For IonMonger N30, the shared assessment reports `lambda_D=1.467 nm`,
`f_diffusion=1.008e-5 Hz`, `f_blocking=5.487e-3 Hz`, and
`f_dielectric=0.7470 Hz`. The default one-decade policy therefore recommends
approximately `1.008e-6` through `7.470 Hz`. A `10 Hz--100 kHz` request is
returned unchanged and explicitly remains uncovered.

The N61 single-ion performance probe used 138 dynamic coordinates and 30
frequencies from `1e-4` to `1e6 Hz`, with all three finite-difference levels.
On the recorded single-thread environment, DC preparation took `1.120 s` and
the frequency reference solve took `0.276 s`. It passed with all-face spread
`6.93e-7`, backward error `5.87e-16`, minimum reciprocal condition
`8.76e-10`, inventory response `1.35e-12`, and exact returned current-
decomposition closure. These timings are a local baseline, not a cross-host
performance certificate.

The real IonMonger grid ladder uses actual node counts `61/91/121`, 29 fixed
frequencies from `1e-6` through `10 Hz` at 0.25-decade spacing, and the same
canonical DC history on every grid. All 29 per-frequency certificates passed.
The finest `91 -> 121` comparison had maximum `|Z|` change `0.311%` and phase
change `0.0349 deg`, below the declared runtime limits of `2%` and `1 deg`;
all three grids also passed full frequency-window coverage. Combined physical
certification remains false because the source deck's contact thermodynamics
is `compatible_unverified`.

These are implementation tests and canonical runtime evidence. A
content-addressed runner lane named
`ionmonger-ion-aware-impedance-resolved-v1` now binds the `60/90/120`
nominal-grid matrix, external finite-difference factors `1/0.5/0.25`, and all
29 frequency points. Magnitude convergence uses a pointwise relative norm so a
large impedance elsewhere in the spectrum cannot hide one unconverged
frequency. The release execution and external comparison remain separate
deliverables.

## Remaining Phase 2 work

1. Exact discrete Poisson sensitivity, analytic SG transport, local bulk,
   defect-free, clamp-inactive cross-node/projected, and positive-density
   shared-occupancy/additive-two-sided and interior-root QSS interface
   reaction, selective-contact, and differentiable CT/PF mobility blocks are
   implemented; replace the remaining unsupported interface/nonlocal
   frozen-potential differences.
   See
   [ion-aware-structured-jacobian-comparison.md](ion-aware-structured-jacobian-comparison.md).
2. Execute the registered canonical grid/stencil/frequency matrix and mint its
   content-addressed release artifact.
3. Cross-check selected frequencies against transient lock-in using the exact
   same DC state and protocol.
4. Only after those gates pass, route the method through `run_impedance`, the
   backend, and frontend diagnostics.

External IonMonger/Driftfusion artifacts and experimental spectroscopy remain
separate validation layers.

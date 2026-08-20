# Ion-aware structured Jacobian comparison

Status: `INTERNAL_TESTED_COMPARISON` as of 2026-08-21. This is a validation
path for the ion-aware impedance reference engine. It is not yet a fully
analytic Jacobian, a registered numerical certificate, or external
validation.

## Purpose

`perovskite_sim.experiments.ion_aware_structured_jacobian` tests the most
important global chain rule in the eliminated-Poisson formulation before a
production structured Jacobian is introduced. It compares two independently
constructed operators:

1. the existing nonlinear callback, which re-solves Poisson at every finite-
   difference stencil;
2. a semi-analytic callback, which differentiates the exact discrete Poisson
   solve once and evaluates transport, recombination, contacts, and current at
   the resulting frozen potential.

Both paths use the same certified DC state, state layout, voltage convention,
material cache, frequencies, current decomposition, and adaptive per-column
log-density stencil.

## Exact Poisson block

For each active log-density coordinate `u_j`, the charge derivative is

```text
d rho / d u_j = s_j * q * y_dc,j
```

at that coordinate's node, with `s_j = -1, +1, +1, -1` for electrons,
holes, positive ions, and negative ions. The code solves the already-factored
finite-volume Poisson matrix for every charge derivative. The voltage
derivative uses zero bulk charge and the exact right-boundary derivative
`-junction_polarity`.

The certificate directly evaluates the componentwise backward error of every
state and voltage sensitivity solve. Poisson therefore does not pass merely
because the final impedance happens to agree.

## Adaptive column stencil

A single log-density step is not numerically meaningful across this device.
On the N61 IonMonger state, maximum potential sensitivity spans roughly
`1e-52` to `2.5e2 V` per unit log increment. The comparison protocol chooses

```text
h_j = clip(target_potential_step / max(abs(d phi / d u_j)), h_min, h_max)
```

with defaults `target_potential_step=1e-9 V`, `h_min=1e-5`, and
`h_max=1e-3`. The full-Poisson and structured callbacks are both expressed in
the same scaled coordinates, so each operator column uses the same physical
stencil. The maximum ion step is checked against the site-occupancy limit
before linearization.

## Comparison contract

The mass block uses the exact affine tangent of `y_dc * exp(u)`. The rate rows
are scaled by operating storage, matching the frequency solver. Columns are
grouped separately as electron, hole, positive ion, and negative ion so a
fast ionic block cannot hide a carrier error.

Every column must pass either its declared self-relative error gate or a
`1e-6` absolute-error gate normalized to that species group's dominant
column. Columns smaller than `1e-4` of the group scale are additionally
listed in `bounded_weak_columns`; columns whose relative comparison is noisy
but whose group-normalized error is bounded are listed in
`absolute_bounded_columns`. A column that satisfies neither gate appears in
`failed_columns` and fails the certificate. Final impedance magnitude and
phase are independently compared with the converged three-level reference
response.

Default gates include:

| Quantity | Limit |
|---|---:|
| Poisson componentwise backward error | `2e-12` |
| mass column error | `2e-7` |
| group-normalized absolute column error | `1e-6` |
| storage-voltage derivative error | `1e-12` |
| storage-scaled rate column error | `5e-5` |
| total conduction column error | `1e-4` |
| displacement-charge column error | `1e-5` |
| named current-component column error | `1e-4` |
| impedance magnitude error | `1e-4` |
| impedance phase error | `1e-3 deg` |

Unknown protocol fields, a mismatched impedance protocol hash, invalid step
bounds, an occupancy-crossing stencil, a failed reference certificate, or a
failed comparison gate all fail closed. Diagnostic mode returns the complete
failed evidence without promoting it.

## Current evidence

The real N13 single-ion and symmetric dual-ion integrations pass. The N61
single-ion probe used 138 dynamic coordinates and 30 frequencies from
`1e-4` to `1e6 Hz`. Its adaptive steps ranged from `1e-5` to `1e-3`, with a
median `4.13e-5`. Observed maxima were:

| Check | N61 observed | Limit |
|---|---:|---:|
| Poisson backward error | `4.16e-13` | `2e-12` |
| mass column error | `1.67e-7` | `2e-7` |
| rate column error | `2.61e-7` | `5e-5` |
| conduction column error | `8.22e-5` | `1e-4` |
| displacement column error | `7.59e-7` | `1e-5` |
| component column error | `5.71e-5` | `1e-4` |
| impedance magnitude error | `6.45e-8` | `1e-4` |
| impedance phase error | `7.78e-7 deg` | `1e-3 deg` |

On the recorded single-thread run, the existing three-level N61 reference had
a median wall time of `0.259 s`; the complete comparison, including the
reference, exact Poisson sensitivities, adaptive operator reference, and
structured solve, took `0.438 s` (`1.69x`). This is acceptable for validation
but is not a production performance improvement.

An N91 probe also passes the dual column gate. Its Poisson backward error is
`1.05e-12`; the largest self-relative hole-current discrepancy is `5.94e-4`,
while that column's error normalized to the dominant hole block is below
`1e-6`. The classification is retained as absolute-bounded evidence rather
than reported as a false high-accuracy relative comparison.

The focused small-signal and structured suite is `32 passed`; the expanded
DC/current/conservation domain is `99 passed`; the existing slow c-Si QF
frequency regression is `8 passed`. The repository default suite is
`2028 passed, 2 skipped, 263 deselected`.

## Remaining work

1. Implement analytic Scharfetter-Gummel carrier and ion flux derivatives.
2. Implement analytic bulk and interface recombination derivatives with
   explicit capability gates for non-smooth regularization branches.
3. Replace the remaining frozen-potential central differences block by block,
   preserving these per-column comparisons.
4. Introduce sparse or matrix-free assembly only after analytic parity passes.
5. Complete frequency-window coverage and transient lock-in cross-checks
   before routing ion-aware impedance through public backend or frontend APIs.

Contact thermodynamics, external IonMonger or Driftfusion comparison, and
experimental impedance validation remain separate evidence axes.

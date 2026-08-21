# Ion-aware structured Jacobian comparison

Status: `INTERNAL_TESTED_ANALYTIC_TRANSPORT` as of 2026-08-21. This is a
validation path for the ion-aware impedance reference engine. Poisson and
Scharfetter-Gummel transport are analytic; reaction, contact, and unsupported
non-smooth closures are not. It is not yet a fully analytic production
Jacobian, a registered numerical certificate, or external validation.

## Purpose

`perovskite_sim.experiments.ion_aware_structured_jacobian` tests the global
chain rule in the eliminated-Poisson formulation before a production sparse
Jacobian is introduced. Protocol v2 retains three independently constructed
objects:

1. the existing nonlinear callback, which re-solves Poisson at every finite-
   difference stencil;
2. a frozen-potential finite-difference operator, used as a block-local
   transport reference;
3. a hybrid structured operator with exact discrete Poisson sensitivities,
   analytic carrier/ion SG face currents and matching conservative flux-
   divergence rate rows, while reaction and contact terms remain the frozen-
   potential finite differences.

All paths use the same certified DC state, state layout, voltage convention,
material cache, frequencies, current decomposition, and adaptive per-column
log-density stencil. Existing transient, DC, and current APIs continue to use
their original numerical paths; the analytic derivatives are opt-in here.

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

with defaults `target_potential_step=1e-9 V`, `h_max=1e-3`, and `h_min` bound
to the final impedance refinement level. For the default reference this is
`1e-5 * 0.25 = 2.5e-6`, not the coarse first level. The full-Poisson and
structured callbacks are both expressed in
the same scaled coordinates, so each operator column uses the same physical
stencil. The maximum ion step is checked against the site-occupancy limit
before linearization.

## Analytic transport block

`bernoulli_derivative` uses the cancellation-free power series near zero and
an `exp(-x)` form at large positive argument. Electron and hole face-current
Jacobians include direct left/right density derivatives and both potential
derivatives, with the fixed band-edge offsets retained in the SG argument.

The ion block covers both implemented steric laws:

- the legacy whole-flux steric multiplier, including its concentration
  derivative;
- the physical diffusion-only lattice-gas chemical potential;
- positive and negative charge signs;
- single-ion, distinct-site dual-ion, and shared-site dual-ion cross
  derivatives.

Each face derivative is chained through the exact Poisson sensitivity and the
per-column log-density scale. The same analytic face-current correction is
inserted into the corresponding electron, hole, positive-ion, or negative-ion
continuity divergence. Reaction, generation, interface recombination, and
contact rows remain the independently evaluated finite-difference remainder.
This paired replacement is required: replacing current alone produced a
measured low-frequency all-face spread of `6.66e-1`; replacing its matching
conservative divergence reduced it below `4e-7` on N13/N61/N91.

The analytic lane fails closed for a field-dependent mobility, an active or
near-switching thermionic cap, a smoothed thermionic cap, exclusive interface
transport, incomplete dual-ion arrays, or an active face on a steric clipping
kink. A zero-diffusivity structural face is allowed because every derivative
there is identically zero.

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
| analytic SG transport column error | `5e-6` |
| analytic SG transport voltage error | `5e-6` |
| impedance magnitude error | `1e-4` |
| impedance phase error | `1e-3 deg` |

Unknown protocol fields, a mismatched impedance protocol hash, invalid step
bounds, an occupancy-crossing stencil, a failed reference certificate, or a
failed comparison gate all fail closed. Diagnostic mode returns the complete
failed evidence without promoting it.

## Current evidence

The real N13 single-ion and symmetric dual-ion integrations pass. The N61
single-ion probe uses 138 dynamic coordinates. Its adaptive steps range from
`2.5e-6` to `1e-3`, with a median `4.13e-5`. A three-frequency probe spanning
`1e-4` to `1e6 Hz` observed:

| Check | N61 observed | Limit |
|---|---:|---:|
| Poisson backward error | `4.27e-13` | `2e-12` |
| storage-scaled rate column error | `1.67e-7` | `5e-5` |
| conduction self-relative error | `2.70e-4` | dual gate |
| conduction group-normalized error | `3.37e-8` | `1e-6` |
| analytic transport self-relative error | `1.90e-4` | dual gate |
| analytic transport group-normalized error | `3.00e-8` | `1e-6` |
| displacement group-normalized error | `2.24e-9` | `1e-6` |
| impedance magnitude error | `2.11e-8` | `1e-4` |
| impedance phase error | `5.50e-7 deg` | `1e-3 deg` |
| all-face admittance spread | `1.68e-7` | reference protocol |

The structured comparison itself took `0.76 s` in the recorded single-thread
N61 probe after the DC state was available. This is validation evidence, not a
production performance claim; the code still assembles dense matrices and
retains finite-difference reaction work.

An N91 probe also passes the dual column gate. Its Poisson backward error is
`1.03e-12`; the largest self-relative analytic hole-current discrepancy is
`1.22e-3`, while its error normalized to the dominant hole block is
`2.45e-7`. The classification is retained as absolute-bounded weak-column
evidence rather than reported as false high-relative-accuracy evidence. Its
impedance magnitude error is `6.92e-8` and all-face spread is `3.87e-7`.

The formula and N13 structured unit layer is `35 passed`; the real single-ion,
symmetric dual-ion, N61, and N91 integration layer is `4 passed`. The expanded
related domain is `112 passed, 5 deselected`. The repository default suite is
`2049 passed, 2 skipped, 263 deselected`.

## Remaining work

1. Implement analytic bulk and interface recombination derivatives with
   explicit capability gates for non-smooth regularization branches.
2. Implement analytic selective-contact and field-mobility derivatives, or
   retain explicit capability gates where a smooth tangent is unavailable.
3. Replace the remaining frozen-potential reaction differences block by block,
   preserving these per-column comparisons.
4. Introduce sparse or matrix-free assembly only after analytic parity passes.
5. Complete frequency-window coverage and transient lock-in cross-checks
   before routing ion-aware impedance through public backend or frontend APIs.

Contact thermodynamics, external IonMonger or Driftfusion comparison, and
experimental impedance validation remain separate evidence axes.

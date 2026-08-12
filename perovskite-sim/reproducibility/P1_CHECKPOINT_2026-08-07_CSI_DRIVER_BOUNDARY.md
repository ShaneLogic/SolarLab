# P1 checkpoint: c-Si J-V driver capability boundary (2026-08-07)

> Historical checkpoint: the final phase decision and verification totals are
> governed by `P1_CLOSURE_2026-08-07.md` and `p1_gaps.yaml`. The root-cause
> evidence and fail-closed driver boundary below remain current, while the
> remaining-work list records the state at the time of this checkpoint.

## Decision

The shipped `configs/cSi_homojunction.yaml` is production-routable only through
the cancellation-safe quasi-Fermi J-V driver. The transient and algebraic
drivers remain available solely behind an explicit diagnostic override.

This checkpoint resolves the operational capability boundary. It does not
close `csi-transient-jv-grid-envelope` and does not claim that the transient or
algebraic numerical root cause has been repaired.

## Root cause isolated

The same residual-certified N=200, V=0, one-sun QF state was evaluated through
both current representations. No seed, voltage, illumination, material, or
physical state was changed:

```text
split-QF max normalized residual             5.8220e-13
split-QF electron/hole continuity [A/m2]     5.8026e-10 / 3.5129e-09
split-QF all-face current spread [A/m2]      3.4140e-10
density-SG max peak-scaled residual          33.0819
density-SG electron/hole continuity [A/m2]   3.18478 / 7.0973e-04
```

The general RHS therefore destroys an already certified state when split QF
reference/increment information is collapsed into absolute `n,p,phi`. This is
a state/current representation failure in the high-conductivity emitter, not
evidence that the QF seed needs a longer transient settle. The executable
regression is
`tests/regression/test_csi_generic_state_cancellation.py`.

## Implemented contract

- `DeviceStack.jv_solver_policy` accepts `general` or
  `cancellation_safe_qf_required`; unknown values fail during construction.
- YAML loading, inline backend loading, backend serialization, frontend types,
  and the config editor preserve the policy.
- The c-Si preset declares `cancellation_safe_qf_required`.
- Transient J-V, algebraic steady-state J-V, and algebraic Voc fail after the
  electrical-grid guard and before a solve starts.
- `allow_unvalidated_driver=True` is explicit, emits a
  `JVCertificationWarning`, and marks the run as diagnostic-only.
- The backend exposes `solver="quasi_fermi"` without silently substituting it
  for another requested solver. Dark QF requests fail because that path has no
  dark-J-V certificate.
- Direct API grid/capability failures return HTTP 422.
- QF results carry per-point residual, electron/hole continuity-current,
  all-face current-spread, and Poisson diagnostics in `JVPointStatus`.
- The frontend offers an explicit three-way solver selector, blocks
  unsupported c-Si decomposition/spatial combinations, preserves the c-Si
  `electrical_grid`, and rejects a grid below `simulation_hints.min_N_grid`.

## Reproducibility registration

`csi-jv-driver-capability-boundary` is registered as an internal numerical
software-capability contract. The c-Si byte and semantic hashes were updated.
Default `jv_solver_policy="general"` is omitted from semantic canonicalization,
so behaviorally unchanged historical configs retain their prior hashes.

The reconstructed pre-policy c-Si semantic hash is still:

```text
bd78c5657031c06d84dcd7457f1685d7b9af93f4938f06fd4b0d13793cb2e64f
```

## Verification completed

```text
28 passed, 1 deselected in 0.60 s
4 passed in 0.76 s
Python py_compile: pass
Ruff isolated targeted check: pass
TypeScript 6.0.2 isolated transpile (6 changed files): pass
```

The 28 tests cover driver policy, explicit override warnings, QF dispatch and
point certificates, API 422 behavior, and every c-Si grid-envelope node except
the unrelated IonMonger thin-film comparison. The four matrix tests cover
bidirectional benchmark/config links, partial-status evidence, gap status, and
P1 gap contracts.

The root-cause extension additionally passes:

```text
1 passed in 6.44 s
Ruff targeted check: pass
```

## Environment-limited verification

Full matrix/config traversal and the remaining thin-film comparison are not
rerun in this checkpoint because 21 shipped config files are OneDrive
`dataless` placeholders. Frontend Vitest/build are also not rerun because 15
source files and the local `node_modules` toolchain are `dataless`. These are
environment materialization blockers, not observed test failures.

## Remaining physics work

1. Keep `csi-transient-jv-grid-envelope` open. Preserve split QF information in
   a dynamic mass-matrix or equivalent compensated general-driver state; the
   high-conductivity cancellation root cause is now isolated.
2. Require independent N=200/300/400 point, curve, continuity, photon-budget,
   Voc, FF, and PCE certificates before changing the c-Si policy.
3. Keep the QF model boundary explicit: local homojunction, ideal ohmic
   contacts, Beer-Lambert optics, simplified interface SRV, and no external
   terminal J-V validation.
4. Continue the external c-Si C-V geometry match, Lin tandem comparison, and
   external IonMonger/Driftfusion curve cross-check as separate P1 work.

The P0 baseline and its frozen files are unchanged.

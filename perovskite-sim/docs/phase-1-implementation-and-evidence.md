# Phase 1 implementation and evidence

Initial implementation: 2026-08-14

Final Phase 1 verification: 2026-08-20

This report closes the engineering work in Phase 1 of the
[physics and numerics hardening roadmap](plans/2026-08-14-physics-numerics-hardening-plan.md).
It deliberately separates implementation completion from numerical
certification. A lane is reported as `certified`, `partial`, or `failed`
exactly as its immutable certificate records; an implemented runner or a
completed matrix is not itself a physics pass.

## Outcome

All four Phase 1 work packages are implemented:

| Work package | Implemented outcome | Default-path effect |
|---|---|---|
| P1.1 refinement certificates | Frozen lane registry, real adapters, resumable content-addressed runner, protocol/source/environment provenance, fail-closed certificate schema | None until a lane runner is invoked |
| P1.2 numerical health and positivity research | Trial/final state and SRH diagnostics, strict terminal gates, split-step pre-clip evidence, edge-regime probes, opt-in log-density coordinate | Observational report only; historical density coordinate remains default |
| P1.3 controlled RHS regularization | Compact-support Poole-Frenkel, thermionic-cap and interface-density policies plus real-device width ladders | New widths default to zero; legacy steady-state TE behavior remains compatible when policy is omitted |
| P1.4 experiment protocol | Frozen canonical protocol, SHA-256 identity, strict round-trip/mismatch checks, legacy labeling, integration in J-V/TPV/EQE/Suns-Voc/impedance and backend forwarding | Legacy calls remain accepted and are labeled `implicit_legacy_protocol` |

The global Phase 2 entry gate is **not** cleared merely by this engineering
closure. The frozen Phase 1.1 matrices below retain their unsuccessful lanes
as evidence and identify the remaining solver/physics work.

## P1.1 numerical refinement evidence

The registry contains the five roadmap lanes and one versioned c-Si resolved
companion. Every row is a real 3 x 3 execution, not a dry-run or injected
adapter test. Artifacts live under the ignored local directory
`outputs/numerical-refinement/`; promotion into a tracked reference remains a
separate review action.

| Lane | Status | Completed cells | Run ID | Certificate SHA-256 | Exact boundary |
|---|---|---:|---|---|---|
| `scaps-mirror-frozen-ion-ss` | `failed` | 0/9 | `3c6daeb4d0730b6a4dae57f2e830d39d3304e7d18635aaa8d1d0a27f93c710cf` | `b32f8607ccddea74bea7ac1fcede339d6d9acf3881ec7178116128820c297b42` | All nine residual steady-state cells failed; no convergence claim |
| `ionmonger-mobile-ion-transient` | `partial` | 9/9 | `f055334117e4ee3f6ad4693bb4a17b26800e2f38826a371beaa6f76ddc6fab0a` | `da06261a3b09d0fc28653fb3d1827b78a056cdecbd904806b690aaac2ffd56dc` | Grid/tolerance hysteresis, normalized terminal-current trace and reverse Voc gates failed; terminal positivity and zero-floor diagnostic quality gates failed |
| `csi-qf-frequency-domain` | `failed` | 6/9 | `eea468113e7f717562ab664df1b0adc6fea8fd0320c1e517a72bf0dfbe49d518` | `e953a8f80e9db2f75bde97cf137441b72311214ecf3e28e59d25ba39f28b7186` | All N=100 cells fail the explicit Debye-resolution guard; the failure is retained as a negative grid test |
| `csi-qf-frequency-domain-resolved-v2` | `certified` | 9/9 | `823cb5c20c45901eb890735d2f49a71e8366038220659e9a42a0a527b8190ec1` | `be658e9642e6bdac78c98fd0e4cad018992677b2bd8b03883ea832254a52f3de` | N=200/300/400 and all FD-step factors pass observable and quality gates |
| `twod-uniform-limit` | `failed` | 6/9 | `73a495f71a2e8e833b7b1f1769bcf102a7960e560cfc8bb0eddece4ff48bf342` | `fe133c89f13a8bcef9a37fdf4d15c28dfec0e412d199d62a47684a6c46c66ae4` | All multiplier-4 cells failed at the first 0.1 V step after reaching the registered `max_nfev`; no fine-grid parity claim |
| `interface-recombination-charge-off` | `failed` | 0/9 | `892bd79e839134d2cda27662a151bd887a44e1865e296320fbe1c47eae7dfaba` | `0206db5c7ef4b617a456b53b48f9a865b188ea561b3c4b3c731a830723aa4367` | All nine residual steady-state cells failed; interface electrostatic trap charge remains parked |

The original c-Si lane is intentionally not coerced to `partial`: N=100 has
a first p-base cell of about 72.18 nm versus a local Debye length of about
40.89 nm, a ratio of 1.765 above the registered 1.5 guard. The resolved-v2
lane uses N=200/300/400 without changing the observable or quality contract.
This preserves both the negative under-resolution evidence and a usable
internally certified c-Si result.

The SCAPS/interface failures likewise remain `failed`. A read-only QF probe
showed that prolongation can help N=90 converge, but the N=60 to N=90 Voc
change was about 2.01 mV, above the registered 1 mV gate. It was therefore not
introduced as a passing companion after seeing the result.

### Post-Phase-1 interface reference closure (2026-08-23)

The failed legacy shared-node interface lane above remains historical negative
evidence. Phase 3 replaced its execution contract with an explicitly
uncalibrated, contact-consistent two-sided QF charge-off reference. Under
source commit `29c94b4`, run
`d0dc822393290d892e7118bcb7fabd4214b5584815f51ff9ff24f663822687e4`
completed 9/9 cells and produced internally certified certificate
`0a4fdebdf18eb0237eaa1a4bef599872745697d148461f6de25d10a6985a950b`.
This closes the charge-off grid/tolerance and dark-occupancy entry gates; it
does not enable interface sheet charge or revise the historical Phase 1
certificate.

The 2D matrix completed all nine execution attempts. Its multiplier-1 and
multiplier-2 cells completed, while every multiplier-4 tolerance cell failed
at `V_app=0.1 V` with `actual nfev=200001` against the registered 200000
limit. Their wall times were 2028.62 s, 1816.01 s and 1964.36 s. The result is
therefore an execution `failed` certificate, not a `partial` parity envelope.
It exposes the dense implicit 2D path as a solver-scalability boundary that
requires a new versioned strategy rather than a post-hoc threshold change.

The certificate environment records macOS 26.6.1. A 2026-08-20 read-only
replay plan on macOS 26.6.2 retained the same source fingerprint
`544278287c5483ad5d67da9f79a1c99cd440dbc69421b8e1c088cdaee530a9a5`,
registry definition and executor hashes, but correctly minted a different run
ID because the platform identity changed. The 26.6.1 artifacts remain the
frozen evidence cited here; they are not silently resumed into the newer
environment.

See [numerical-refinement-certificates.md](numerical-refinement-certificates.md)
for the registry, state machine, protocol provenance, matrix definitions and
reproduction commands.

## P1.2 numerical health and positivity

`run_transient` now observes raw implicit RHS trials and accepted terminal
states without changing the equations. Its immutable report covers each state
block, negative and non-finite trial counts, non-finite RHS events, exact bulk
and interface SRH denominator minima, and terminal strict gates. Production
J-V results propagate accepted-step diagnostics into point status and the
mobile-ion refinement quality gates.

The split-step lane separately records initial, raw implicit-trial, raw
terminal, projected terminal and final ion bounds and finite-volume
dual-cell-weighted inventories. It also gates final electron, hole and active
interface-state positivity. Research strict mode evaluates the raw terminal
before historical clipping, so clipping cannot manufacture a pass. Both
positive and negative mobile-ion species and inactive structural zeros are
covered.

The opt-in `state_coordinates="research_log_density"` prototype maps active
electron, hole, ion and interface-state densities through `y = s exp(z)` while
keeping inactive structural-ion zeros linear. All external solver boundaries
remain in physical density units. A real N=10 device refinement found the fine
density/log terminal-current difference of 0.004320 A/m2 inside the larger
0.015992 A/m2 same-coordinate refinement change. It remains a research lane:
very small dark-equilibrium densities can drive `f/y` beyond the finite
exponential range, where the implementation fails closed, and the transform
does not enforce an ion-site upper bound.

See [numerical-health-diagnostics.md](numerical-health-diagnostics.md) for the
coordinate tolerance map, units, split-step contract, measured overhead and
known limitations.

## P1.3 regularization evidence

Three explicit compact-support policies cover the non-smooth Poole-Frenkel
field factor, thermionic-emission magnitude cap and interface density
projection. Each policy carries a transition width with units, exactly recovers
the hard expression outside its compact band and defaults to the historical
zero-width behavior. The legacy steady-state logistic TE cap is retained when
the new policy is omitted.

The final `[w, 0.5w, 0.25w, 0]` real-device certificates are generated by:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 python scripts/run_regularization_ladders.py all
```

| Study | Status | Certificate SHA-256 |
|---|---|---|
| `poole-frenkel-device` | `certified` | `1966b9b537e836dec5153714a53740c9626cbce3205ee70b26d1a8658f6137d1` |
| `thermionic-cap-device` | `certified` | `7a5c9c6235b79e169e75f9cee5cb918d7a56b79c768148cfff526c687ad00dbd` |
| `interface-density-device` | `certified` | `aed4f4310de01d0cc359f1c939f1cd24c23f21236d0356baa0b216efafb2f102` |

These short registered trajectories establish wiring and width sensitivity on
their frozen grids. They do not certify the constitutive model against an
external solver or experiment and do not establish a steady operating point.

## P1.4 protocol contract

`ExperimentProtocol` and its nested history, scan, AC, settle and sampling
objects are frozen and finite-validated. Canonical JSON has a stable SHA-256;
round-trip parsing rejects unknown or missing fields. Explicit protocols must
match the actual execution fields, so metadata cannot describe a different
numerical history. Research-strict calls reject an implicit history, while
legacy calls receive an explicit `implicit_legacy_protocol` label.

J-V hysteresis, TPV, EQE, Suns-Voc and impedance results carry the protocol.
TPV and Suns-Voc include a frozen `VocSearchProtocol` for every state-advancing
coarse/bisection/settle step, and J-V duration includes every dwell actually
executed on both branches. The backend forwards explicit protocol documents
and rejects request/protocol mismatches before an asynchronous job is
submitted. Numerical steady-state continuation lanes do not invent scan rates:
they use a separate canonical numerical execution contract, while the 2D
comparison records matched 1D and 2D subprotocols.

## Verification

Verification on macOS 26.6.2 used single-threaded BLAS/OpenMP settings for the
Python runs:

- the Phase 1 focused Python suite passed with `359 passed, 17 deselected` in
  74.08 s;
- the repository's default Python suite passed in one run with `1973 passed,
  2 skipped, 263 deselected` in 181.23 s. The repository default excludes
  tests marked `slow`; the real registered matrices above provide the
  expensive Phase 1 execution evidence instead of being hidden inside this
  count;
- the frontend suite passed with `399 passed` across 29 files; the two
  impedance evidence files passed `26/26` after replacing a one-microtask mock
  race with an observable wait condition;
- TypeScript `tsc --noEmit`, Vite production build, Python `compileall`, scoped
  Ruff checks and `git diff --check` passed. Vite emitted only its existing
  large-chunk advisory;
- the P0 reproducibility test initially encountered OneDrive dataless Git
  objects. Hydrating only the referenced objects changed no source or manifest;
  the exact test and the final full Python run then passed.

The immutable refinement and regularization artifacts were executed on their
recorded macOS 26.6.1 environment. Current-environment unit/regression success
does not rewrite those certificates, and real lane execution is evidence in
addition to, not a substitute for, the test suite.

## Evidence boundary and next work

- `certified` here means internal numerical convergence for one frozen
  code/config/protocol/environment contract. It is not SCAPS/IonMonger parity,
  experimental validation, parameter identifiability or publication novelty.
- Interface trap electrostatic charge remains `PARKED`. Its equilibrium
  reference, sign, gauge, Gauss jump and outer Poisson coupling are not closed.
- The resolved ion-free c-Si frequency lane is internally certified; it is not
  an ion-aware impedance certificate.
- The IonMonger lane provides a complete sensitivity envelope but remains
  `partial`; it must not be promoted until terminal positivity/zero-floor and
  terminal-pair convergence gates close.
- SCAPS mirror and interface steady lanes need solver/continuation work before
  they can be promoted. Their failed artifacts are retained.
- Phase 2 and Phase 3 physics are not implemented by this report. Their entry
  work starts from these recorded failures and certificates rather than from a
  blanket Phase 1 pass claim.

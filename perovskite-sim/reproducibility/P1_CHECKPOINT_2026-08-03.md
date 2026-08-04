# P1 Continuation Checkpoint (2026-08-03)

This checkpoint continues `P1_CHECKPOINT_2026-08-01.md`. It does not replace
the frozen P0 baseline; machine-readable status remains authoritative in
`config_benchmark_matrix.yaml` and `p1_gaps.yaml`.

## Frozen P0 Boundary

- Base commit: `c23e5b9beb3c356250ea32dcb09c78dc45ba28ec`
- Frozen patch SHA-256:
  `58166a458047984bf85ead3cc5c5b5e29b2c6dc22aa851682a8ca81ef314d82a`
- The P0 patch was not refreshed. Reconstruction still verifies all 15 files.

## CIGS Gap Closure

- Root cause: the n-left/p-right stack used a positive Poisson boundary and
  the p-left external-voltage/current mapping. `compute_V_bi()` remains signed;
  `junction_polarity` now maps the configured magnitude and reported current.
- An isolated failed 4-leg refinement no longer erases a successful 2-leg
  result; a finer 8-leg result must still agree in terminal current and every
  state block before acceptance.
- Contact reservoirs now use endpoint `ni_sq`, including temperature and
  outer-layer grading, instead of the scalar front material `ni`.
- The two shipped CIGS `V_bi` magnitudes are synchronized to their current
  material endpoints. This makes the graded/ungraded contact models comparable.
- The registered protocol is `V_max=1.05 V`, 43 points, `dV=25 mV`, and one
  BLAS thread. A 24-point/50 mV N=120 reverse scan is a documented conditioning
  counterexample, not the production protocol.

Uncontacted baseline, forward branch:

| Nominal grid | Actual intervals | Voc (V) | Jsc (A/m2) | FF | PCE (%) |
|---:|---:|---:|---:|---:|---:|
| 40 | 39 | 0.954965 | 397.6155 | 0.837118 | 31.7861 |
| 80 | 78 | 0.955111 | 397.3653 | 0.836824 | 31.7598 |
| 120 | 120 | 0.955141 | 397.3531 | 0.836702 | 31.7552 |

The 78-to-120 changes are 0.0299 mV in Voc, 0.0031 percent in Jsc,
0.0145 percent in FF, and 0.0144 percent in PCE.

At the production rung, the 120-interval ungraded Robin stack and 160-interval
graded stack both certify. Grading changes Voc from 0.956142 V to 1.008605 V
and Jsc from 397.4401 to 398.4837 A/m2. Both directions bracket Voc, obey the
incident-photon ceiling, have physical FF/PCE, and have negligible hysteresis.

This closes `cigs-2um-graded-notch` as an **internal numerical and qualitative
trend** gap. Both configs remain `partial`: `Nc300/Nv300`, the SI provenance of
`ni/n1/p1`, composition-dependent absorption, measured contact work functions,
interface parameters, and an external J-V/QE/band-diagram dataset are absent.
The idealized 31.8-33.7 percent PCE is not an experimental validation.

## c-Si QF Full J-V Closure

- The old equal-per-layer mesh forced the 300 nm n+ emitter onto sub-angstrom
  cells while leaving the 180 um base junction cell comparatively coarse. The
  executable `electrical_grid` config now allocates total intervals 1:4 and
  uses alpha=2/3 by layer, giving 40/160, 60/240, and 80/320 intervals on the
  N=200/300/400 ladder.
- The manual positive `V_bi` magnitude is synchronized to
  `abs(compute_V_bi()) = 0.8928964399850017 V`; the signed source value remains
  negative for this n-left junction.
- A new opt-in QF-potential steady-state solver eliminates Poisson at every
  residual evaluation and evaluates SG face currents with cancellation-safe
  `expm1` identities. Independent review found that the first implementation
  still cancelled the ordinary SG divergence against a correction term. The
  corrected implementation assembles generation/recombination on a
  zero-transport material and adds the stable QF divergence directly. All
  contact faces now enter the current-spread gate, density exponent clipping
  is rejected rather than hidden, and thermionic faces fail before Newton.

The registered voltage protocol uses 42 common points: a 25 mV base grid from
0 to 0.600 V, 5 mV MPP refinement from 0.480 to 0.530 V, and sub-mV open-circuit
refinement. Each grid starts independently at 0 V with the full illumination
ramp; later voltages use only the preceding certified QF state as a warm start.

| N_grid | Layer intervals | Jsc (A/m2) | Voc (V) | FF | PCE (%) |
|---:|---:|---:|---:|---:|---:|
| 200 | 40 / 160 | 356.970188 | 0.592163 | 0.824722 | 17.4334 |
| 300 | 60 / 240 | 356.888189 | 0.592500 | 0.824894 | 17.4429 |
| 400 | 80 / 320 | 356.842862 | 0.592684 | 0.824938 | 17.4470 |

The N=300-to-400 maximum pointwise current change is 2.9995 A/m2, 0.8406
percent of fine-grid Jsc, and contracts from 6.4817 A/m2. Finest-pair Jsc,
Voc, FF, and PCE changes are 0.0127 percent, 0.184 mV, 0.0054 percent, and
0.0236 percent. The nested 10 mV versus 5 mV MPP check changes PCE by at most
0.0553 percent. Across the final registered run, worst continuity and all-face
current-spread certificates are below 5.5e-8 A/m2; the worst normalized
Poisson residual is below 2.5e-11.

This closes `csi-qf-jv-grid-convergence` for the restricted internal QF model.
The solver is not wired into the default driver and rejects ions, selective
contacts, non-local photon recycling, and thermionic interfaces. The default
transient/algebraic c-Si ladder and external c-Si validation remain open.

## c-Si C-V Fail-Closed Audit

- The former Mott-Schottky smoke used `N_grid=30`. Its first p-base cell was
  about 289 nm (`7.07` local Debye lengths), comparable to the full
  `0.29-0.37 um` analytic depletion width over the sampled bias range.
- The resulting `5.7e-7 F/m2` capacitance was near the geometric capacitance of
  the full 180 um wafer, and its `1/C2` span was only 1.66 percent. Returning
  `NaN` under the 5 percent identifiability guard was physically correct; the
  old test's demand for finite `V_bi` and `N_eff` was contradictory.
- Impedance now shares the executable weighted grid and Debye guard with J-V
  and steady-state. Failed dark DC preconditioning, non-positive capacitive
  susceptance, non-depletion slope, and flat `1/C2` data all fail closed.
- This mesh repair is not the physical root-cause fix. A single N=200 point at
  `V=-0.2 V`, 100 kHz, three cycles still returned `5.723e-7 F/m2` after
  618.7 s.
- The root cause is now localized. Three certified dark QF states give a
  carrier-inventory differential capacitance of `2.886e-4 F/m2`, consistent
  with the abrupt-junction depletion scale. A 1 mV step transfers
  `2.887e-7 C/m2`, but the 100 kHz staircase uses 250 ns intervals and samples
  conduction current only at each endpoint. The ns-scale charging pulse has
  already decayed from hundreds of A/m2 to about `1.2e-5 A/m2`, while the
  displacement term is an interval average. Mixing these time definitions
  removes the depletion response and leaves the whole-wafer geometric term.
- `V_bi` polarity, the static junction response, and the lock-in fit are not
  the root cause. Dark DC preconditioning still needs a residual certificate
  as an independent secondary gap.

The analytic wrapper recovers known `V_bi` and `N_eff`, so the post-processing
contract is covered. `csi-mott-schottky-convergence` remains open for the real
solver and requires grid/frequency/amplitude/cycle convergence.

## Current P1 Status

Closed:

1. `scaps-low-doping-etl`
2. `ionmonger-residual-ss-96` (terminal-current claim only)
3. `cigs-2um-graded-notch` (internal numerical/trend claim only)
4. `csi-qf-jv-grid-convergence` (restricted internal QF model only)

Open:

1. `lin2019-tandem-jsc-pce`
2. `csi-transient-jv-grid-envelope` (default drivers only)
3. `external-solver-curve-crosscheck`
4. `csi-mott-schottky-convergence`

## Verification At This Checkpoint

- Reproducibility verifier: P0 reconstruction passed; matrix covers 28 configs,
  18 resources, 16 benchmark contracts, and 3 schemas.
- Reproducibility tests: `35 passed`.
- Mott-Schottky/impedance fail-closed unit tests: `20 passed`; affected real
  impedance slow paths: `3 passed, 1 deselected`.
- Electrical-grid parser, allocation, round-trip, and guard tests: `35 passed`.
- c-Si QF source-only audit tests: `8 passed, 1 deselected`; registered
  42-point full J-V ladder: `6 passed in 169.85s`.
- Complete CIGS module: `10 passed in 279.28s`, including the thickness,
  production-grid, graded-trend, physical-bound, and registry-observation gates.
- Junction/refinement state-machine tests: `31 passed`.
- Grading/contact/temperature focused tests: `32 passed, 2 deselected`.
- BLAS marker-expression tests: `8 passed`; `slow or not slow` now retains the
  required single-thread limit instead of silently oversubscribing dense LU.
- `compileall` and `git diff --check` passed.
- Frontend: production build passed; `377 passed` across 27 Vitest files.
- Complete default suite:
  `1516 passed, 2 skipped, 239 deselected, 12 warnings in 152.30s`.
- Complete slow suite:
  `234 passed, 4 skipped, 1518 deselected, 1 xfailed, 4 warnings in 6211.11s`.

## Next Stage

Keep the default-driver c-Si runtime/convergence gap separate. For C-V, do not
spend a 200/300/400 ladder on the current endpoint-sampled branch. Implement a
charge-conservative interval current or, preferably, a frequency-domain
linearized small-signal solve about a residual-certified DC state; first close
the one-point depletion-capacitance and charge-balance gates. The other
research validation work remains the provenance-preserved external-solver
curve freeze and Lin tandem current matching; neither is resolved by the
internal QF certificate.

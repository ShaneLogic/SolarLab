# P1 Continuation Checkpoint (2026-08-04)

This checkpoint continues `P1_CHECKPOINT_2026-08-03.md`. It does not replace
or refresh the frozen P0 baseline. Machine-readable status remains
authoritative in `config_benchmark_matrix.yaml` and `p1_gaps.yaml`.

## Frozen Boundary

- Starting implementation commit: `b52d8f2`.
- P0 base commit: `c23e5b9beb3c356250ea32dcb09c78dc45ba28ec`.
- Frozen patch SHA-256:
  `58166a458047984bf85ead3cc5c5b5e29b2c6dc22aa851682a8ca81ef314d82a`.
- P0 reconstruction still verifies all 15 frozen files. No P0 hash or patch
  was changed.

## c-Si Small-Signal Root Fix

The c-Si transient impedance probe sampled conduction current at the end of
each staircase interval while using an interval-average displacement current.
The ns-scale charging pulse had already decayed at the endpoint, so the
reported capacitance collapsed to the approximately `5.72e-7 F/m2` geometric
capacitance of the 180 um wafer.

The new explicit `quasi_fermi_frequency` method:

1. Requires a residual-certified dark QF operating point.
2. Preserves the reference and increment parts of each QF potential so tiny
   high-conductivity gradients are not lost by absolute-value subtraction.
3. Forms central-difference storage, rate, conduction-current, and
   displacement-charge derivatives.
4. Solves `(i*omega*M - A) du = (b - i*omega*mV) dV` with LAPACK matrix
   equilibration and iterative refinement.
5. Reports every face's total admittance, reciprocal condition estimate, and
   direct componentwise backward error.
6. Requires `0 < delta_V < 20 mV` and fails closed outside the audited local,
   ion-free QF capability envelope.

The default `transient` impedance method is unchanged. This work does not
claim that its endpoint-sampling defect is repaired.

## Registered Numerical Evidence

The registered dark protocol uses biases `-0.3` through `+0.2 V`, frequencies
10 kHz/100 kHz/1 MHz, and the configured N=200/300/400 weighted grid ladder.

| N_grid | C(-0.2 V, 100 kHz) (F/m2) | 100 kHz Mott intercept (V) | N_eff (m-3) |
|---:|---:|---:|---:|
| 200 | 2.886247e-4 | 0.771663 | 9.4928e21 |
| 300 | 2.907073e-4 | 0.755249 | 9.4660e21 |
| 400 | 2.919108e-4 | 0.756214 | 9.5537e21 |

- At N=200 and -0.2 V, the frequency-domain result is within 4.8 percent of
  the abrupt-junction depletion estimate and over 400 times the geometric
  capacitance.
- Maximum 100 kHz C(V) change contracts from 1.2407 percent (200 to 300) to
  0.4123 percent (300 to 400).
- The finest-pair fitted-intercept change is below 1 mV; the N_eff change is
  below 1 percent.
- On N=400, the maximum capacitance change from 10 kHz to 1 MHz is
  `4.82e-5` relative. The fitted intercepts are
  `0.756209/0.756214/0.756265 V` across those frequencies.
- Every retained point has positive capacitive susceptance. The complete
  matrix passes the `5e-4` all-face admittance-spread gate and `1e-10`
  componentwise linear-solve backward-error gate.
- Halving the state and voltage finite-difference steps leaves the 100 kHz
  capacitance inside the registered stability limit. Changing nominal AC
  amplitude from 10 mV to 5 mV leaves the linear response invariant.

## Decision Boundary

The old geometric-capacitance failure is resolved only for the restricted QF
frequency-domain path. `csi-mott-schottky-convergence` remains **open** because
the converged Mott intercept, about `0.756 V`, is about `0.137 V` below the
configured `0.892896 V` contact-potential magnitude. No external c-Si C-V
curve is frozen, and the new operator does not yet include mobile ions,
selective contacts, thermionic interfaces, or non-local photon recycling.

The Mott-Schottky automatic window residual gate is now 1 percent rather than
10 percent. This excludes a smooth forward-injection tail that otherwise
shifts the extrapolated intercept by more than 0.1 V.

## Reproducibility Matrix

- Added `csi-qf-frequency-domain-cv` as an internal numerical benchmark.
- The c-Si config remains `partial`, now linked to four explicit contracts:
  loader, grid envelope, QF J-V convergence, and QF frequency-domain C-V.
- Verifier coverage is now 28 configs, 18 resources, 17 benchmark contracts,
  and 3 schemas.
- The four P1 gaps remain open: Lin tandem current matching, default-driver
  c-Si J-V convergence, external-solver curve provenance, and the residual
  c-Si Mott-intercept interpretation/external-validation gap.

## Verification

- P0 reconstruction and reproducibility verifier passed.
- Reproducibility matrix tests: `35 passed`.
- New c-Si frequency-domain C-V regression: `6 passed in 26.13s`.
- Existing c-Si QF 42-point J-V ladder: `6 passed in 173.95s`.
- Affected QF/impedance/Mott/small-signal unit tests:
  `36 passed, 1 deselected in 3.36s`.
- Complete default suite:
  `1524 passed, 2 skipped, 245 deselected, 12 warnings in 151.49s`.
- `compileall` and `git diff --check` passed.

The complete historical slow suite was not repeated. The two slow lanes whose
shared QF implementation changed were run explicitly: the existing full J-V
ladder and the new full C-V matrix.

## Next Stage

Do not generalize the AC operator first. Audit the remaining intercept gap by
comparing frequency-domain capacitance with bias derivatives of electrode and
depletion charge, then separate finite n+/p-junction potential partition,
contact-reservoir, and thermal-correction contributions. Freeze an independent
c-Si C-V reference only after that internal identity closes. General ion-aware
frequency-domain variables remain the subsequent extensibility task.

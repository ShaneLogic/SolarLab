# R1 long-window operator criterion decision V2

Date: 2026-09-16. Status: development disposition; replacement criterion not
approved. Supersedes V1's description of the probes, without changing any
equation, material parameter, iteration limit or production acceptance gate.

## Decision

The measured mobile-ion long-window tails remain **not certifiable under the
current relative operator criterion** (`maximum_eliminated_operator_relative_error
= 1e-6`). Preserve both the original metric and the original failures. This
document does not authorize a denominator floor, automatic pass, omitted
component, or relaxed threshold. Development trajectories may be recorded with
failed certificates; they do not become accepted scientific results.

## Corrections to V1

The independent review of candidate `97d9ef3141c9fb433d534e8f769223ecfaaff6d2`
reproduced the conditioning diagnosis and corrected its scope:

- In the N16/D/0.01, +5 mV diagnostic, the trajectory maximum is 2.0. The late
  per-row value at 10 s is approximately 1.0430806348152415. The 100 s probe
  instead ends near 0.6667; V1's claim of the same asymptotic value was false.
  Finite-time observations are not proof of an asymptotic limit.
- The cited trap-storage usage 0.9766564315966633 is one D-case observation,
  not a bound across controls. The independent review reports approximately
  99.86% for C and 98.16% in a separate nonlinear-residual comparison. These
  metrics and scopes must remain separate; none is permission to relax a gate.
- Rate and flux comparisons independently fail at weak net signal. The
  1 s component maxima are approximately 1.009234e-4 (ion rate) and
  7.135546e-6 (ion flux). Changing only the rate denominator is insufficient.

These numbers are inherited development diagnostics from
`results/OneDimensionalMechanism/R1StageTwoAcceptanceV2/README.md` and its
`ReviewV1/` records in the external project archive. They are not new formal
measurements or a proof that the physical solution is correct.

## Work required for a replacement proposal

1. Derive separate, dimensioned absolute rate and flux error budgets from the
   equations and intended accuracy, without fitting a floor to failed traces.
2. Record absolute discrepancies, the existing relative metric, signal scales
   and resolution status. An unresolved signal is not an agreement.
3. Use independent analytic or high-precision equilibrium, weak-signal and
   resolved-signal controls. Inject genuine rate and flux errors independently.
4. Compare the proposal over independent spatial, temporal and nonlinear axes
   and all required controls; independently examine trap-storage consistency.
5. Independently review and freeze the formula and constants before using them
   for formal long-window acceptance. Historical failed bundles stay failed.

## Evidence boundary

Revision-five execution and acceptance pin this document's bytes independently
of the bundle. The pin makes the declared decision stable; it is not independent
approval of the development candidate. Numerical recertification of a saved
state cannot reconstruct historical timestamps, environment or optimizer
history. Such fields remain recorded provenance under the stated runtime trust
assumptions, with no claim of independent historical verification.

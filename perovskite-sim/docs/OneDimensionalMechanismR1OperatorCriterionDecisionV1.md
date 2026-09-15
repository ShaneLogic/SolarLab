# R1 long-window operator criterion decision V1

Date: 2026-09-15. Status: frozen development disposition; replacement criterion
not approved. Applies to the unchanged R1-1/R1-2 physical equations and the
existing `maximum_eliminated_operator_relative_error = 1e-6` gate.

## Decision

The measured nu_I=1 long-window tails are **not certifiable under the current
operator criterion**. Keep the gate and failed certificates unchanged. Treat
the demonstrated failure as a criterion applicability/conditioning issue,
not as an established defect in the physical solver that must be repaired
before the same relative threshold can become meaningful.

This closes the decision requirement by explicitly recording the window as
uncertifiable. It does not close long-window R1-2 acceptance. No replacement
threshold, automatic pass, denominator floor, waived component or change to
solver iteration limits is authorized by this document. A computation may
continue to preserve diagnostic trajectories while its certificate fails.

## Evidence and scope

The independent review of `84f9131123869a8f55a3589d0f3c435c2add8b35` records
these reproducible development-class observations for N16/D/factor 0.01,
+5 mV, 12 intervals per decade and 1/2/4 subdivisions:

| Observation | Recorded value and implication |
|---|---|
| Window ending at 1 s | 766 records; only final certificate reason `eliminated_operator_error`, maximum `1.0092335580336034e-4` against `1e-6` |
| Finest-level component maxima over the 1 s window | Ion-rate error about `1.009234e-4`; ion-flux error about `7.135546e-6`. These are separate component maxima, not necessarily from the same state. Changing the rate normalization alone is insufficient. |
| Cancellation estimate on one ion face | Forward/reverse SG terms cancel by about `3.063e11`; binary64 flux arithmetic estimate is `3.38e-6` of the remaining signal, already above the comparison threshold. This is a state-specific estimate, not a universal exact error bound. |
| Time dependence | The ion-rate discrepancy numerator remains within roughly one decade while the denominator decreases by about 13.2 decades over the extended diagnostic range. |
| Window ending at 10 s | The trajectory maximum is `2.0`; only two finest-level rows attain that value. The late per-row value is approximately `1.0430806348152415`, not a sustained `2.0` plateau. The 100 s probe retains the same maximum and asymptotic value. |
| Other gates | All recorded per-step physical gates pass for this case; trap-storage maximum reaches `0.9766564315966633` of its budget at 10/100 s and remains a separate, nearly binding check. |

The underlying metric compares independent operators using a scale tied to
the net rate/flux. Near equilibrium that scale can approach arithmetic noise.
The contract `OneDimensionalMechanismR1DynamicsV1.md`, common-state section,
already states that a relative error normalized by a vanishing signal has no
useful equilibrium precision meaning. The observed tail meets that concern
in both rate and flux. It does not establish an impossibility theorem for all
alternative arithmetic or algebraically equivalent implementations.

Evidence is retained in the project results archive
`OneDimensionalMechanism/R1StageTwoAcceptanceV1/ReviewV1/LaneReportsV1.json`
and `RefutationsV1.json` (LongWindow entries). The refutation corrects the
original lane's sustained-saturation claim. The original delivery's 16-point
minimal case remains in `R1StageTwoPrerequisitesV1/NumericalDiagnosisV1/`.
These are development diagnostics, not controlled formal long-window runs;
the current formal CLI exposes only the short functional times.

## Required work before revising the criterion

1. Define independently dimensioned absolute rate/flux error budgets and the
   intended signal-resolved regime. Derive scales from equations and physical
   error budgets, not by fitting a floor to the failing traces.
2. Retain distinct ion-rate, ion-flux, carrier and electrostatic components,
   their absolute discrepancies, normalization scales and resolution status.
   An unresolved comparison must not be reported as agreement.
3. Verify equilibrium, weak-signal and resolved-signal cases with independent
   high-precision or analytic controls. Inject genuine operator errors in both
   rate and flux and show the new procedure still detects them.
4. Preserve the old metric alongside any proposed replacement; measure the
   effect on three grids, independent time/nonlinear axes and other controls.
   Assess the nearly binding trap-storage gate independently.
5. Freeze the proposed formula and constants as a separately reviewed policy
   revision before generating formal long-window evidence. Existing failed
   bundles remain historical failures and must not be relabelled.

Short-trace provenance and comparison-tool work may proceed. Full-window
completion, tail-to-DC agreement and time-to-AC reconstruction remain open
R1-2 deliverables; none is implied by this decision.

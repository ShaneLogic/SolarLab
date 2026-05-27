# Phase E2 Sprint 3 Day 6-7 — thin-shell validation gate

**Status:** MIXED — primary criterion PASS, CBO regression FAIL
**Branch:** `e2-thin-shell-srh`
**Input:** `outputs/scaps_validation_e2_thin_shell_w2/report.md`

## Result summary (shell width = 2)

| Sweep | Legacy (ba10b10) | BBD-active | **Thin-shell w=2** | SCAPS | Verdict |
|---|---|---|---|---|---|
| Base V_oc | 1.069 V | 1.054 V | **1.088 V** | 1.168 V | within envelope ✓ |
| CBO range | 782 mV | 847 mV | **16 mV** | 918 mV | **CBO COLLAPSES** ✗ |
| ETL doping range | 1075 mV | 1542 mV | **17 mV** | 137 mV | **88 % closure** ✓✓ |
| PVK doping | 34 | 21 | 48 mV | 34 mV | range overshoots |
| PVK/ETL interface N_t | 210 | 208 | 206 mV | 282 mV | unchanged (74 %) |
| Bulk N_t | 0 | 0 | 0 mV | 39 mV | unchanged |
| Bulk E_t | 2 | 1 | 3 mV | 0 mV | flat both |

Plus: 23/23 SCAPS-subset tests pass with env unset (no legacy regression).

## RFC pass criteria check

| Criterion | Required | Measured | Pass? |
|---|---|---|---|
| ETL doping range ≤ 200 mV | ≤ 200 | **17 mV** | **PASS** ✓ |
| CBO closure ≥ 80 % | ≥ 80 | **1.7 %** (collapsed 782 → 16 mV) | **FAIL** ✗ |
| Base V_oc ∈ [1.05, 1.25] V | bounded | 1.088 V | PASS |
| No new test failures | 0 | 0 | PASS |

**One pass, one hard fail.** Same gate logic as BBD: any single hard
fail kills the prototype.

## Diagnosis — complementary failures

BBD and thin-shell are complementary:

| | CBO closure | ETL doping range |
|---|---|---|
| Legacy E1.5 | 85 % | 1075 mV (over) |
| BBD | 92 % ✓ | 1542 mV ✗✗ (worse) |
| Thin-shell w=2 | 1.7 % ✗✗ | 17 mV ✓ |
| SCAPS target | 100 % | 137 mV |

**BBD adds bulk-doping sensitivity to fix CBO.**
**Thin-shell removes ALL bulk sensitivity to fix ETL doping.**

The right model needs SOME interface-band-offset sensitivity (for CBO)
but LESS bulk-doping sensitivity than E1.5 (for ETL doping). Neither
prototype hit the right balance — they sit at opposite ends.

Why thin-shell kills CBO: it uses n[i]·p[i] at non-idx shell nodes,
which are deep in their respective bulks. PVK-side has tiny n; ETL-side
has tiny p. The product is dominated by bulk values that do NOT depend
on band offset. So changing CBO does not change the interface SRH
rate measurably.

Why BBD over-sensitive on ETL doping: exp(-Δφ/V_T) factor amplifies
depletion-zone widening at low N_D_ETL (documented in Sprint 2 Day
2-3 findings).

## Three paths forward

### (a) Hybrid: cross-carrier at idx + thin-shell at neighbours

Apply BOTH:
- Legacy E1.5 rate at idx (gives CBO sensitivity)
- Thin-shell rate at shell neighbours (gives ETL doping insensitivity)
- Tune weighting α to balance.

Conceptually: SCAPS' Pauwels-Vanhoutte interface SRH has a sharp
peak at the interface plane (CBO-sensitive) PLUS a thin distributed
tail (less doping-sensitive). The hybrid models this directly.

Risk: another guess without paper guidance. May land at a third
opposite-extreme failure mode. Effort: 3-5 days for prototype + gate.

### (b) Paper acquisition + faithful Pauwels-Vanhoutte implementation

Acquire ref [13] Pauwels & Vanhoutte 1978 J.Phys.D 11, 649-667 via
HKUST library ILL or partner. Implement the literal formula.

Risk: low formula risk, high schedule risk (2-4 weeks total).
Probability of success: high IF formula extracted correctly.

### (c) Park ETL doping as structural (Phase G/F precedent)

Accept that ETL doping over-sensitivity is structural like Phase G
base V_oc gap (74 mV residual) and Phase F PVK doping direction.
Ship the partner report with the current main-branch state. CBO
closure at 85 % remains the strongest closure.

Risk: low (no new code). Cost: cannot close the ETL doping gap.

## Recommendation

**(a) hybrid prototype** is cheapest direct test. Goal: find a single
α ∈ [0, 1] where:
- CBO closure ≥ 70 % (some regression from BBD's 92 %, acceptable)
- ETL doping range ≤ 300 mV (some regression from thin-shell's 17 mV)

If (a) fails, escalate to (b). The (c) park option is always available
as the final fallback.

## Sprint 4 schedule (if user picks hybrid)

| Day | Deliverable |
|---|---|
| 1 | Hybrid design RFC + branch `e2-hybrid-srh` off main |
| 2 | RED tests for hybrid contract |
| 3-4 | GREEN env-var-gated prototype (~30 LoC + 5 tests) |
| 5 | α probe sweep ∈ {0.0, 0.25, 0.5, 0.75, 1.0} |
| 6-7 | Validation gate run + findings doc |
| 8-10 | Ship if pass; else escalate to (b) |

## Convention

- Branch `e2-thin-shell-srh` is parked as failure evidence (same as
  `e2-bbd-face-density`). Neither merges to main.
- Validation gate findings doc + the thin-shell prototype commit
  (405cc12) form the audit trail.
- Future hybrid / paper-based work cuts off main `20d8661` (current
  main HEAD after the BBD gate doc cherry-pick).

**Related:** [[project-scaps-validation-parked]], Phase E2 design RFC
`2026-05-27-e2-design-rfc.md`, BBD gate
`2026-05-27-e2-sprint2-day2-3-validation-gate.md`, thin-shell RFC
`2026-05-27-e2-thin-shell-srh-rfc.md`.

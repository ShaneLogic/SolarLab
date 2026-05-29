# Phase E2 Sprint 2 Day 2-3 — BBD validation gate

**Status:** FAIL on primary criterion → pivot per RFC fallback path
**Branch:** `e2-bbd-face-density`
**Input:** `outputs/scaps_validation_e2_bbd/report.md`

## Result summary

| Sweep | Legacy (ba10b10) | BBD-active | SCAPS | Verdict |
|---|---|---|---|---|
| Base V<sub>oc</sub> | 1.069 V | 1.054 V | 1.168 V | within ±10 % envelope ✓ |
| CBO range | 782 mV | **847 mV** | 918 mV | **improved** (85 % → 92 % closure) ✓ |
| ETL doping range | 1075 mV | **1542 mV** | 137 mV | **WORSE by 43 %** ✗ |
| PVK doping range | 34 mV | 21 mV | 34 mV | range shrunk |
| PVK/ETL interface N<sub>t</sub> | 210 mV | 208 mV | 282 mV | unchanged (74 % closure) |
| Bulk N<sub>t</sub> | 0 mV | 0 mV | 39 mV | unchanged (still blocked) |
| Bulk E<sub>t</sub> | 2 mV | 1 mV | 0 mV | flat both |

Plus: 23/23 SCAPS-subset tests pass with env unset (no legacy regression).

## RFC pass criteria check

| Criterion | Required | Measured | Pass? |
|---|---|---|---|
| ETL doping range ≤ 200 mV | ≤ 200 mV | 1542 mV | **FAIL** |
| CBO closure ≥ 80 % | ≥ 80 % | 92 % | PASS |
| Base V<sub>oc</sub> ∈ [1.05, 1.25] V | bounded | 1.054 V | PASS |
| No new test failures | 0 | 0 | PASS |

**Primary criterion FAIL.** Sprint 2 Day 2-3 gate is a hard fail.

## Why did probe predict 2.8× sensitivity REDUCTION but full sweep shows 43 % INCREASE?

Day 3.5 probe measured `d log(np) / d log(N_D_ETL)` at single V<sub>oc</sub>
point over 3 decades (1e16→1e19). Showed BBD np 2.8× LESS sensitive
than E1.5 — direction-right.

But full SCAPS sweep covers 10 decades (1e8→1e18) including the
low-N<sub>D</sub> Fermi-pinning regime. Day 3.5 explicitly flagged this:

> "Probe at single V<sub>oc</sub> point — cannot capture low-N<sub>D</sub> Fermi-pinning
>  regime where bulk of V<sub>oc</sub> range originates."

At low N_D_ETL (1e8-1e14 cm⁻³) the ETL/PVK depletion zone widens
massively into ETL, so Δφ across the interface neighbourhood grows
strongly with the doping change. BBD multiplies by
exp(-Δφ/V<sub>T</sub>), which AMPLIFIES that doping sensitivity rather than
damping it. The single-V<sub>oc</sub> probe found a coincidental near-cancellation
between n and p exponentials at the high-N<sub>D</sub> end — the cancellation
collapses at low N<sub>D</sub> and the depletion-zone term takes over.

In other words: the BBD formula is **physically incorrect** at the
ETL doping low-N<sub>D</sub> limit, because it conflates the band-bending Δφ
(which is what depletes the bulk-side density) with a face-density
correction. SCAPS' actual interface-plane density (per ref [13]
Pauwels-Vanhoutte 1978) must use a different formulation that does
NOT amplify Δφ in this way.

## What BBD did improve

- **CBO closure 85 % → 92 %.** The depletion factor correctly captures
  the band-offset-induced density step. BBD is right for CBO.
- **Base V<sub>oc</sub> kept within envelope.** No physical-envelope violation.
- **No legacy regressions.** Env-gating works correctly.

So BBD is not categorically wrong — it gets the band-offset physics
right but adds extra ETL-doping sensitivity that does not exist in
SCAPS. The right model probably has the band-offset depletion factor
but NOT the bulk doping sensitivity.

## Decision: pivot to thin-shell volumetric SRH

Per Phase E2 design RFC fallback path (Day 1 audit Section "Fallback
path: thin-shell volumetric SRH"):

> "Treat interface SRH as a volumetric source over a thin shell
>  (~1 nm) at the interface node, with R = N_t_volumetric · σ · v<sub>th</sub>
>  · ... evaluated using the solver's own n[idx], p[idx] at the
>  interface node (NOT eval_n_idx / eval_p_idx). The shell width
>  and N<sub>t</sub> conversion are calibrated to SCAPS' areal N<sub>t</sub>."

Why thin-shell may avoid the BBD failure mode: it uses n[idx], p[idx]
at the interface NODE (where the solver already harmonic-means
between layers), so it does not pick up the bulk-side N_D_ETL
sensitivity that BBD amplified. The interface-node density is set
by the heterojunction electrostatics + Q-Fermi splittings, not by
bulk N<sub>D</sub> directly.

Estimated effort: 2 weeks (Sprint 3 + Sprint 4 = Sprint 2.5
extension).

## Alternative pivot options

| Option | Description | Effort | Risk |
|---|---|---|---|
| **Thin-shell volumetric SRH** | Per RFC fallback — use n[idx], p[idx] at interface node, scale by shell width | 2 weeks | medium (Newton stability at the heterointerface node) |
| **Hybrid BBD-for-CBO + cross-carrier-for-doping** | Keep BBD active only for the CBO sweep; revert to cross-carrier for ETL doping | 3 days | high (per-sweep physics gating violates immutability principle) |
| **Investigate Pauwels-Vanhoutte 1978 paper directly** | Block on library ILL | 2 weeks ILL + 1-2 weeks implementation | low risk on formula correctness, high schedule risk |
| **Accept BBD CBO win + park ETL doping** | Revert BBD prototype; ship CBO improvement separately if any path exists | 1 day | low — but doesn't close any sweep |

## Recommendation

**Thin-shell volumetric SRH (Sprint 3)** is the structured fallback
per RFC. Two-week effort. Probability of success: medium. If it also
fails, escalate to Pauwels-Vanhoutte paper acquisition (Sprint 4).

Alternative: surface to partner via Phase H-style report with this
validation gate finding + ask whether they want the thin-shell pivot
OR a Pauwels-Vanhoutte paper-acquisition wait OR park the ETL doping
problem as "structural" like Phase G/F base V<sub>oc</sub>.

## Next actions

- Commit this findings doc + the BBD prototype as the validation
  evidence.
- Revert the prototype call in `assemble_rhs` so default behaviour
  is bit-identical? OR keep it gated and surface to user for
  pivot decision.
- If user picks thin-shell: cut `e2-thin-shell-srh` branch off main,
  write Phase E2.2 design RFC for thin-shell.

## Convention

- Validation outputs land in `outputs/scaps_validation_e2_bbd/` (with
  PNG sweep overlays). Do NOT commit the PNG blobs — only the
  findings doc + script edits.
- The BBD prototype commit (4162862) stays on the branch as evidence
  of the failed gate. Branch will NOT merge to main.

**Related:** [[project-scaps-validation-parked]], Phase E2 design RFC
`2026-05-27-e2-design-rfc.md`, Day 3.5 N<sub>D</sub> sweep findings
`2026-05-27-e2a-sprint1-day3p5-nd-sweep.md`.

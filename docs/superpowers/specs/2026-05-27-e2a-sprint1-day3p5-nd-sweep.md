# Phase E2a Sprint 1 Day 3.5 — N_D_ETL sweep BBD sensitivity

**Status:** probe results (investigation-only)
**Branch:** `e2a-scaps-source-audit`
**Input:** `outputs/e2a_probe_nd_sweep.txt`

## Goal

Test Day 3 hypothesis: does the BBD (band-bending depletion) candidate
recover SCAPS' lower ETL doping sensitivity (parked-memory claim:
SolarLab 1075 mV vs SCAPS 137 mV V<sub>oc</sub> range, 8× over-sensitive)?

## Method

Sweep `etl_doping_cm3` ∈ {1e16, 1e17, 1e18, 1e19} cm⁻³ on
scaps_mirror.yaml at V<sub>oc</sub> baseline. Record E1.5 vs BBD face densities
+ np products.

## Results

| N_D_ETL [cm⁻³] | solver V<sub>oc</sub> | E1.5 n_face | E1.5 p_face | E1.5 np | BBD n_face | BBD p_face | BBD np |
|---|---|---|---|---|---|---|---|
| 1e16 | 1.054 | 3.35e23 | 9.12e22 | 3.06e46 | 3.52e23 | 3.21e23 | 1.13e47 |
| 1e17 | 1.054 | 4.57e23 | 7.92e22 | 3.62e46 | 4.76e23 | 2.44e23 | 1.16e47 |
| 1e18 | 1.060 | 1.34e24 | 5.21e22 | 6.97e46 | 1.37e24 | 1.09e23 | 1.49e47 |
| 1e19 | 1.071 | 1.00e25 | 2.16e22 | 2.19e47 | 1.00e25 | 2.19e22 | 2.19e47 |

### Sensitivity comparison (across 3 decades 1e16→1e19)

|  | rise of n_face | drop of p_face | rise of np | d log(np)/d log(N_D_ETL) |
|---|---|---|---|---|
| **E1.5** | 30× | 4.2× | 7.1× | 0.28 / decade |
| **BBD** | 28× | 14.6× | 1.95× | 0.10 / decade |

**Ratio E1.5 / BBD: 2.8× sensitivity reduction.**

### V<sub>oc</sub> movement under current E1.5 solver

Solver V<sub>oc</sub> rises 17 mV across the probed 3-decade range (1.054→1.071).
Parked-memory claim of 1075 mV range refers to the full SCAPS sweep
1e8→1e18 cm⁻³ (10 decades), with the bulk of the range coming from
low N<sub>D</sub> where Fermi-pinning dominates — not from the high-N<sub>D</sub> regime
probed here.

## Diagnosis

BBD np sensitivity to N_D_ETL is **2.8× LOWER** than E1.5 cross-carrier
across the probed 3-decade range. This is the **right direction** for
closing the SCAPS over-sensitivity gap.

But the partial reduction (2.8× vs needed 8×) suggests:

1. **BBD alone is not a complete fix.** Remaining ~3× sensitivity gap
   likely comes from low-N<sub>D</sub> regime where Fermi-pinning + Robin contact
   physics matter (not captured by BBD at single V<sub>oc</sub> point).

2. **Individual n_face / p_face sensitivities are LARGER in BBD.**
   n rises 28× (vs E1.5 30×) and p drops 14.6× (vs E1.5 4.2×). The
   cancellation in np product is what reduces sensitivity. If the
   actual SRH path is in the asymptotic regime where R ∝ min(n,p)
   (large-injection limit), then BBD p drop dominates → R drops 14×,
   V<sub>oc</sub> rises 70 mV across 3 decades. Need solver-wired prototype to
   measure actual SRH sensitivity rather than np proxy.

3. **Cancellation could be coincidental.** BBD assumes Boltzmann
   within-layer projection; SCAPS' actual interface-plane density
   comes from thermionic-emission boundary which includes a
   quasi-Fermi step (ref [13] Pauwels-Vanhoutte). Without the original
   paper, we cannot confirm BBD = SCAPS' actual model. Probe shows
   BBD is in the right ballpark but cannot prove formula match.

## Decision

**Proceed to Phase E2 design RFC with caveat:** the RFC must include
a 1-day **solver-wired BBD prototype** step BEFORE committing to the
full multi-week refactor. The prototype:

1. Add a `MaterialArrays.use_bbd_face` boolean flag (or env var).
2. Modify `_apply_interface_recombination` to compute n_face, p_face
   via BBD when flag is True; keep E1.5 cross-carrier as default.
3. Re-run the SCAPS validation suite (CBO + ETL doping + bulk + PVK
   doping + base J-V) with flag=True.
4. **Pass criterion:** ETL doping V<sub>oc</sub> range closes to SCAPS 137 mV ±50 %
   (200 mV envelope) AND CBO closure stays ≥80 % AND base V<sub>oc</sub> within
   10 % envelope.
5. **Fail criterion:** ETL doping range still >300 mV OR CBO regresses
   <70 % OR base V<sub>oc</sub> moves outside [1.05, 1.25] V envelope.

If pass → ship as Phase E1.17 / E2 (~150-300 LoC, 1-2 weeks).
If fail → fall back to thin-shell volumetric SRH option from Day 1 audit.

## Next actions

- **Day 4-5:** Write Phase E2 design RFC with the prototype-first strategy
  documented above. Include risk analysis if N<sub>D</sub> probe insensitivity is
  coincidental rather than physical.
- **Phase E2 Sprint 2 Day 1:** Implement solver-wired BBD prototype.
  Pass/fail decision lands in 1 day.
- **Phase E2 Sprint 2 Days 2-10:** Either Phase E1.17 BBD ship (pass)
  or pivot to thin-shell volumetric SRH (fail).

## Convention

Probe-only — no `perovskite_sim/` touch. Findings doc + script
extension only.

**Related:** [[project-scaps-validation-parked]], Day 1 audit
`2026-05-27-e2a-scaps-source-audit.md`, Day 3 findings
`2026-05-27-e2a-sprint1-day3-findings.md`.

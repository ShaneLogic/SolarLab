# Phase E10 — root-cause SolarLab↔SCAPS matching workflow

**Goal:** match ALL 10 PDF sweep trends AND absolute values (V_oc/J_sc/FF/PCE)
against `1R-Parameters.xlsx`, by fixing **root causes** in the numerical
algorithm + physical models — **never by unphysical hacks.**

## Physics gate (every iteration must pass)

A change is accepted only if ALL hold:
1. `J_sc ≤ SQ(Eg)` (≈27.5 mA/cm² for 1.53 eV); J_sc ≤ photogeneration.
2. `V_oc ≤ V_bi`; V_oc rises with generation, falls with recombination.
3. Recombination ≥ 0 at illuminated forward bias (no spurious sources);
   SRH/Auger/radiative use their textbook forms.
4. Energy conservation in optics (R+T+A=1); detailed balance (R=0 at dark eq).
5. Sweep directions physically correct (more defects → lower V_oc, etc.).
6. No regression of already-physical/matched sweeps or the test suite.

**If matching SCAPS requires violating the gate → physics wins; document the
SCAPS-side non-physicality instead of chasing it.**

## Root causes (ranked) and the PHYSICAL fix for each

| # | Root cause | Symptom | Physical fix | Hardness |
|---|---|---|---|---|
| R1 | Interface SRH samples **bulk-interior** densities w/ bulk-asymptotic ni_eff² | HTL/PVK spurious generation (band-aided by NOGEN clamp), Nd_ETL under-sensitive, bulk-N_t masked, ~15mV V_oc | **interface-plane carrier sampling** (QSS reduction of the interface-plane state; replaces the clamp) | hard |
| R2 | TMM absorber **under-generation** | J_sc 240 vs 263 (−9%) | audit reflection / n,k / spectrum; fix if it's an error (NOT if SCAPS ignores real reflection) | medium |
| R3 | base V_oc **ceiling** (zero-recomb 1199 vs 1249) | −50mV everywhere | contact-flux / carrier-statistics audit vs SCAPS | hard |

## Loop protocol (per iteration)
1. Pick the highest-leverage unsolved root cause.
2. Diagnose with a probe (numbers, not assumptions).
3. Implement the physical fix (env-gated until validated).
4. **Physics gate** check + regression test.
5. Regenerate the affected figure(s); update scorecard.
6. Update `scaps_validation_report.md`; commit; push.
7. Repeat until all 10 sweeps match (trend + absolute) within tolerance OR a
   gap is proven to be a SCAPS-side non-physicality / fundamental model
   difference (then documented, not hacked).

## Success criteria
- All 10 sweeps: direction match + range closure ≥ 70% (trend).
- Absolute: V_oc/J_sc/FF/PCE median |Δ| within ~5% (or documented physics limit).
- Every shipped result passes the physics gate.

## Iteration log
- (E10.1) … see commits below.

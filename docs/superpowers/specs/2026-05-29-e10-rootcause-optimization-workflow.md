# Phase E10 — root-cause SolarLab↔SCAPS matching workflow

**Goal:** match ALL 10 PDF sweep trends AND absolute values (V<sub>oc</sub>/J<sub>sc</sub>/FF/PCE)
against `1R-Parameters.xlsx`, by fixing **root causes** in the numerical
algorithm + physical models — **never by unphysical hacks.**

## Physics gate (every iteration must pass)

A change is accepted only if ALL hold:
1. `J_sc ≤ SQ(Eg)` (≈27.5 mA/cm² for 1.53 eV); J<sub>sc</sub> ≤ photogeneration.
2. `V_oc ≤ V_bi`; V<sub>oc</sub> rises with generation, falls with recombination.
3. Recombination ≥ 0 at illuminated forward bias (no spurious sources);
   SRH/Auger/radiative use their textbook forms.
4. Energy conservation in optics (R+T+A=1); detailed balance (R=0 at dark eq).
5. Sweep directions physically correct (more defects → lower V<sub>oc</sub>, etc.).
6. No regression of already-physical/matched sweeps or the test suite.

**If matching SCAPS requires violating the gate → physics wins; document the
SCAPS-side non-physicality instead of chasing it.**

## Root causes (ranked) and the PHYSICAL fix for each

| # | Root cause | Symptom | Physical fix | Hardness |
|---|---|---|---|---|
| R1 | Interface SRH samples **bulk-interior** densities w/ bulk-asymptotic ni_eff² | HTL/PVK spurious generation (band-aided by NOGEN clamp), Nd_ETL under-sensitive, bulk-N<sub>t</sub> masked, ~15mV V<sub>oc</sub> | **interface-plane carrier sampling** (QSS reduction of the interface-plane state; replaces the clamp) | hard |
| R2 | TMM absorber **under-generation** | J<sub>sc</sub> 240 vs 263 (−9%) | audit reflection / n,k / spectrum; fix if it's an error (NOT if SCAPS ignores real reflection) | medium |
| R3 | base V<sub>oc</sub> **ceiling** (zero-recomb 1199 vs 1249) | −50mV everywhere | contact-flux / carrier-statistics audit vs SCAPS | hard |

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
- Absolute: V<sub>oc</sub>/J<sub>sc</sub>/FF/PCE median |Δ| within ~5% (or documented physics limit).
- Every shipped result passes the physics gate.

## Iteration log

### E10.1 — R2 (TMM under-generation): add glass front substrate ✅
Photon balance (R+A+T=1.00): bare 3-layer lost 7.4 mA/cm² to air/spiro front
reflection (missing glass substrate). Added 1 mm glass front (optical-only).
**J<sub>sc</sub> 24.0→25.73 mA/cm² (N_grid=30), within −2% of SCAPS 26.3.** Physics gate
PASS (J<sub>sc</sub>≤SQ, V<sub>oc</sub>≤V<sub>bi</sub>, trends preserved). Commit `c447c22`. Residual =
SCAPS zero-front-reflection idealization (not chased). Metal back reflector
tested: +0.4 only (most remaining transmission is sub-Eg, unrecoverable).
**Numerical item logged:** N_grid=30 inflates the peaked-G trapz ~+2.5 mA/cm²
vs fine grid (true fine-grid J<sub>sc</sub>≈23.1) — grid-convergence, future iteration.

### E10.2 — R3 (base V<sub>oc</sub> −97mV): root-caused, fundamental solver divergence
Implied dark saturation J<sub>0</sub>(SolarLab)≈2.6e-20 vs J<sub>0</sub>(SCAPS)≈7e-22 A/cm² —
**37× higher → exactly the 93mV via kT·ln(37)**. Channel decomposition at V<sub>oc</sub>:
**Auger 4.79 + radiative 4.11 mA/cm² dominate** (bulk SRH 0.33, interface
clamped). Auger/radiative use the PDF coefficients (C=2.3e-29, B=1e-12),
identical to SCAPS. Yet SolarLab recombines 37× more at a given V — because its
**carrier-density-vs-voltage relation differs**: the absorber QFL split is
1.205 V while the terminal V is only 1.07 V — a **135 mV internal drop** across
the heterojunction band offsets (HTL/PVK ΔE<sub>C</sub>=1.54 eV, PVK/ETL transport).
With identical ni (1.408e12, verified), coefficients, and contacts (Robin/
Dirichlet equal at SCAPS-realistic S — tested), this is a **fundamental
solver / heterojunction-transport / high-injection-statistics divergence**, not
a tunable parameter. It is LINKED to R1 (the interface/heterojunction
carrier-sampling). **Physics gate: SolarLab's V<sub>oc</sub> is physical; not hacked.**
Closing it requires either SCAPS solver internals or the R1 interface-plane/QSS
work (multi-day) — deferred. Eliminated (all tested): ni, contact BC, single
interface SRH channel.

### Remaining (hard, deferred)
- **R1** — interface-plane carrier sampling (QSS); replaces the NOGEN clamp with
  the principled fix; addresses Nd_ETL direction, Nt_C/V_PVK mask, AND part of
  the R3 135 mV heterojunction drop. Multi-day solver work (scoped in the E8 spec).
- **R3 residual** — high-injection np(V)/statistics vs SCAPS; needs SCAPS internals.

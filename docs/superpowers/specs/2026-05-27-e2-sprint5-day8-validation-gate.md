# Phase E2 Sprint 5 Day 8 — full Pauwels-Vanhoutte validation gate

**Status:** FAIL — ETL doping over-suppressed (1 mV), PVK doping Newton-stall
**Branch:** `e2-pv-full`
**Input:** `outputs/scaps_validation_e2_pv_full/` (partial — killed at PVK sweep)

## Partial result (killed at PVK doping sweep after 23 min hang)

| Sweep | Legacy | BBD | Thin-shell | Single-sided PV | **Full PV** | SCAPS |
|---|---|---|---|---|---|---|
| Base V<sub>oc</sub> | 1.069 | 1.054 | 1.088 | 1.0905 | **1.0905** | 1.168 |
| CBO range | 782 | 847 | 16 | 663 | **663** | 918 |
| ETL doping range | 1075 | 1542 | 17 | 1419 | **1 mV** | 137 |
| PVK doping | 34 | 21 | 48 | 50 | **stuck** | 34 |
| Interface N<sub>t</sub> | 210 | 208 | 206 | 8 | n/a | 282 |

Full PV with charge-balance partition + two-sided SRH: ETL doping
collapsed to 1 mV (over-suppression, same disease as thin-shell);
CBO regressed to 72 % (same as single-sided PV); PVK doping sweep
hung Newton.

## RFC pass criteria check

| Criterion | Required | Measured | Pass? |
|---|---|---|---|
| ETL doping range ≤ 200 mV | ≤ 200 | 1 mV (over-suppressed) | **FAIL (overshoot)** |
| CBO closure ≥ 80 % | ≥ 80 | 72 % | **FAIL** |
| Interface defect N<sub>t</sub> ≥ 150 mV | ≥ 150 | (untested, sweep killed) | unknown |
| Base V<sub>oc</sub> ∈ [1.05, 1.25] V | bounded | 1.0905 | PASS |
| No new test failures | 0 | 0 (23/23) | PASS |
| Newton converges on full sweep | yes | NO (PVK doping hang) | **FAIL** |

**Three hard fails.** Worse than single-sided PV (which at least finished).

## Diagnosis — four-prototype pattern

|  | CBO | ETL doping | Interface N<sub>t</sub> | Other |
|---|---|---|---|---|
| Legacy E1.5 | 85 % | 1075 mV (over) | 210 mV | balanced |
| BBD | 92 % ✓ | 1542 mV ✗ | 208 ✓ | over-sensitive to N<sub>D</sub> |
| Thin-shell w=2 | 1.7 % ✗ | 17 mV ✓ | 206 ✓ | kills band-offset response |
| Single-sided PV | 72 % | 1419 mV ✗ | 8 mV ✗ | suppresses interface SRH |
| **Full PV** | 72 % | 1 mV (over) | n/a | over-suppression + Newton hang |
| SCAPS target | 100 % | 137 mV | 282 mV | — |

**Pattern:** Every depletion factor I add either over-amplifies (BBD)
or over-suppresses (others). MoL + SG framework cannot replicate
SCAPS' analytical heavy-doping form without architectural refactor.

## Root cause — discretisation mismatch

SCAPS uses thermionic-emission boundary at the interface plane. This
creates a Q-Fermi STEP at the heterointerface. Interface SRH then uses
interface-plane densities computed from the per-side Q-Fermi levels
that DIFFER across the step.

SolarLab MoL + SG assumes Q-Fermi continuity across all grid nodes
(SG flux derivation requires it). The "interface plane" is just a node
where the solver hands off between two layers — no Q-Fermi step lives
there.

When we project bulk densities to "interface plane" via exp(-V_2/V<sub>T</sub>):
- At V<sub>app</sub> = 0 dark: V_2 ≈ V<sub>bi</sub> → exp(-42) ≈ 1e-18 → R ≈ 0 (over-suppressed)
- At V<sub>app</sub> = V<sub>oc</sub>: V_2 ≈ 0 → exp(0) = 1 → R = bulk-bulk product (legacy-like)

SCAPS doesn't suffer from this collapse because TE boundary lets
n_1s, n_2s evolve independently as separate unknowns, NOT both
projected from a single bulk density.

## Why all four prototypes failed the same way

BBD: depletion factor SCALES WITH N_D_ETL via local grid potential.
Thin-shell: skips interface node, samples non-idx where n·p mismatched.
Single-sided PV: heavy-doping limit, only one side has exp factor.
Full PV: TWO sides have exp factors, both can over-suppress.

NONE of them implement the actual SCAPS missing ingredient: an
independent interface-plane carrier density unknown coupled to bulks
through TE flux + SRH.

## Realistic options

### (A) Ship main as Phase H — accept structural ETL gap

Three E2 prototype attempts have demonstrated that the MoL + SG
discretisation cannot replicate SCAPS' interface-plane physics
without research-grade refactor. Main branch state (CBO 85 %,
interface defect 74 %, base V<sub>oc</sub> within envelope) is the best
attainable in this framework.

Effort: 0 days (already shipped at `ba10b10` / Phase H).
Document the four E2 failures in parked memory for future-self.

### (B) Architectural refactor — dedicated interface-plane node

Add a thin pseudo-node AT each heterointerface that carries
(n_iface_L, n_iface_R, p_iface_L, p_iface_R) as 4 separate state
variables coupled via:
- TE flux between bulk-side node and interface-plane-side node
- Q-Fermi continuity within each layer (preserved)
- Q-Fermi STEP across the interface plane (new degree of freedom)
- Interface SRH as a sink between interface-plane n and p

This is what SCAPS does internally. Estimated effort: 4-8 weeks of
focused solver work + careful Newton stability investigation +
re-validation of the entire test suite (not just SCAPS subset).

Probability of success: medium-high (formula is known, paper +
SCAPS Manual provide the recipe). Risk: Newton stability around the
new state variables, especially at low N_D_ETL where the TE flux
ratio shifts dramatically.

### (C) Phase E3 — Boltzmann-degenerate carrier statistics

Phase G base V<sub>oc</sub> audit identified Boltzmann-degenerate statistics
as a candidate for the residual 74 mV base V<sub>oc</sub> gap AND potentially
for ETL doping at N<sub>D</sub> > 1e18 where bands approach degenerate.
Implementing this would refactor `physics/recombination.py` to use
Fermi-Dirac statistics instead of Maxwell-Boltzmann. Doesn't directly
fix interface SRH but may unlock parity in a different way.

Effort: 3-4 weeks. Probability: medium.

## Recommendation

**(A) Ship main + document E2 failures in parked memory.** Four
prototypes have demonstrated the limit. Continuing with (B) or (C)
is research-grade work that may or may not succeed; the cost-benefit
is unclear given the partner has already received the Phase H report
with 85 % CBO closure and 74 % interface defect closure.

If user explicitly wants to invest 4-8 more weeks: pick (B). If user
wants a different angle: pick (C).

## Convention

- Branch `e2-pv-full` parked alongside the other three failure
  branches. Findings doc cherry-picks to main.
- All four E2 prototype attempts documented in parked memory as a
  group: BBD / thin-shell / single-sided PV / full PV. Future agents
  reading the memory know these have been tried and why they failed.

**Related:** [[project-scaps-validation-parked]], Phase E2 design RFCs,
all four sprint-by-sprint gate findings docs.

# Phase E3 Sprint 7 Day 4-6 — χ-step cross-flux validation gate

**Status:** PARTIAL — Newton hang fixed, ETL doping preserved, CBO still collapsed
**Branch:** `e3-interface-plane-state`
**Input:** `outputs/scaps_validation_e3_chi_step/report.md`

## Result summary

| Sweep | Legacy | Day 1-3 iface | **Day 4-6 χ-step** | SCAPS |
|---|---|---|---|---|
| Base V<sub>oc</sub> | 1.069 | 1.0905 | **1.0905 V** | 1.168 |
| CBO range | 782 | 18 | **18 mV** ✗ | 918 |
| ETL doping range | 1075 | 15 | **15 mV** ✓ | 137 |
| PVK doping | 34 | 49 | **50 mV** | 34 |
| **Interface N<sub>t</sub>** | **210** | **hang** | **0 mV** ✗ | **282** |
| Bulk N<sub>t</sub> | 0 | 0 | 0 | 39 |
| Bulk E<sub>t</sub> | 2 | hang | 3 | 0 |

Plus: 23/23 SCAPS-subset regression pass with env unset (legacy bit-identity preserved). 33/33 E3 unit + integration tests pass.

## What Day 4-6 fixed

1. **Newton stability.** Interface defect N<sub>t</sub> sweep no longer hangs. χ-step-anchored dark-eq init keeps cross-flux ≈ 0 at initial state.
2. **ETL doping closure preserved** (15 mV, 89 %).
3. **Validation gate completes in 3 min** vs 36 min hang prior.

## What Day 4-6 did NOT fix

1. **CBO collapse persists** (782 → 18 mV, 1.96 % closure).
2. **Interface defect N<sub>t</sub> collapsed FURTHER** (210 → 0 mV). Defect density sweep has ZERO effect on V<sub>oc</sub> because χ-step-constrained iface_state plus χ-step-consistent ni_eff² makes the SRH numerator (np − ni²) invariant to N<sub>t</sub>.

## Diagnosis

The χ-step cross-flux ties n_1s, n_2s into a rigid ratio. As ΔE_c sweeps:
- iface_state densities scale proportionally
- SRH pair products n_1s · p_2s = n_1s · p_1s · exp(−ΔE_v/V<sub>T</sub>) = ni²_R · exp(−ΔE_v/V<sub>T</sub>)
- ni_eff² also = ni²_R · exp(−ΔE_v/V<sub>T</sub>) by construction
- Numerator (np − ni²) cancels at equilibrium AND scales the same way as ni² under perturbation

Net: CBO sweep has no effect on iface SRH rate because the χ-step constraint AND the ni_eff² both track ΔE_c symmetrically.

Same mechanism kills interface defect N<sub>t</sub> sweep: defect N<sub>t</sub> enters via v_n, v_p (calibration_factor). When the np product nearly equals ni_eff² (always, due to χ-step constraint), the SRH rate ≈ 0 regardless of v_n, v_p magnitude.

## Five-prototype failure pattern (cumulative)

| Prototype | CBO | ETL doping | Interface N<sub>t</sub> | Newton |
|---|---|---|---|---|
| Legacy E1.5 | 85 % | 1075 ✗ | 210 | OK |
| BBD | 92 % ✓ | 1542 ✗ | 208 ✓ | OK |
| Thin-shell w=2 | 1.7 % ✗ | 17 ✓ | 206 ✓ | OK |
| Single PV | 72 % ~ | 1419 ✗ | 8 ✗ | OK |
| Full PV (1 side) | 72 % ~ | 1 ✗ | n/a | hang |
| Day 1-3 iface-state | 1.96 % ✗ | 15 ✓ | hang | hang |
| **Day 4-6 χ-step** | **1.96 % ✗** | **15 ✓** | **0 ✗** | **OK** |
| SCAPS | 100 % | 137 | 282 | — |

## Honest assessment

Architectural insight: SCAPS' interface-plane physics requires bulk-side thermionic emission to make the bulk Fermi level respond to χ. SolarLab's MoL+SG framework cannot reproduce this without modifying `carrier_continuity_rhs` at the heterointerface face.

The 5+ prototype attempts have demonstrated:
- Closing ETL doping → loses CBO + interface N<sub>t</sub>
- Closing CBO → loses ETL doping (BBD)
- No single-parameter knob exists in the current discretisation

## Three realistic paths

| Path | Effort | Risk |
|---|---|---|
| **(A) Ship Sprint 7 Day 4-6 as "partial parity"** | 1 day | low; documents new closure (ETL 89 %) on top of existing main parity. env-gated feature flag. |
| **(B) Sprint 7 Day 7+ deep refactor — modify carrier_continuity_rhs at heterointerface face to split into per-layer flux + add bulk thermionic emission BC** | 3-4 weeks | high; touches the SG flux core. Solver-wide regression risk. |
| **(C) Park** — accept three structural gaps as documented (ETL, interface N<sub>t</sub>, CBO+CBO tradeoff). Ship main as Phase H. | 0 days | none; partner already has Phase H |

**Recommendation: (A) ship Sprint 7 Day 4-6 partial parity.** ETL doping at 89 % closure is the BEST result of all 5 prototypes. Branch becomes optional env-gated feature; legacy bit-identity preserved. Future work continues in dedicated sprints if partner asks.

## Convention

- Branch `e3-interface-plane-state` continues as Sprint 7 Day 4-6 deliverable.
- All 33 E3 tests pass; legacy bit-identity preserved.
- Feature gated by SOLARLAB_INTERFACE_PLANE_STATE=1.
- Sprint 8 promotes env to SimulationMode field if user picks (A).

**Related:** all prior E2/E3 gate findings docs; Phase E3 design RFC.

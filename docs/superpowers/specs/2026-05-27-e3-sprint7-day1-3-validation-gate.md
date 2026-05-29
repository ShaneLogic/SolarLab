# Phase E3 Sprint 7 Day 1-3 — interface-plane state validation gate (PARTIAL)

**Status:** MIXED — ETL doping closure +88 % but CBO collapsed, Newton hang on interface defect N<sub>t</sub> sweep
**Branch:** `e3-interface-plane-state`
**Input:** `outputs/scaps_validation_e3_iface_plane/` (partial — killed at interface defect N<sub>t</sub> sweep after 36 min hang)

## Partial result vs all five prior prototypes

| Sweep | Legacy | BBD | Thin-shell | Single PV | Full PV | **Iface-state** | SCAPS |
|---|---|---|---|---|---|---|---|
| Base V<sub>oc</sub> | 1.069 | 1.054 | 1.088 | 1.0905 | 1.0905 | **1.0905** | 1.168 |
| CBO range (mV) | 782 ✓ | 847 ✓ | 16 ✗ | 663 ~ | 663 ~ | **18 ✗** | 918 |
| ETL doping range | 1075 ✗ | 1542 ✗ | 17 ✓ | 1419 ✗ | 1 ✗ | **15 ✓** | 137 |
| PVK doping | 34 | 21 | 48 | 50 | stuck | **49** | 34 |
| Interface N<sub>t</sub> | 210 | 208 | 206 | 8 ✗ | n/a | **hang ✗** | 282 |
| Bulk N<sub>t</sub> | 0 | 0 | 0 | 0 | 0 | **0** | 39 |

## RFC gate criteria check

| Criterion | Required | Measured | Pass? |
|---|---|---|---|
| ETL doping ≤ 200 mV | ≤ 200 | 15 | **PASS** ✓ |
| CBO closure ≥ 80 % | ≥ 80 | 1.96 % | **FAIL** |
| Interface defect N<sub>t</sub> ≥ 150 mV | ≥ 150 | hang (untested) | **FAIL** (Newton stability) |
| Base V<sub>oc</sub> ∈ [1.05, 1.25] V | bounded | 1.0905 | PASS |
| 23/23 SCAPS-subset tests | PASS | 23/23 | PASS |
| Full slow regression | PASS | not run yet | unknown |

## What worked

1. **Architecture is sound.** State-vec extension, TE flux, two-sided SRH, dark-eq init, env-gated dispatch — all wire correctly through assemble_rhs. Legacy bit-identity preserved when env unset.
2. **ETL doping closure: ~89 %.** From 1075 mV (legacy 8× over) to 15 mV (SCAPS 137 ±50 %). Best of any prototype.
3. **PVK doping in range:** 49 mV vs SCAPS 34 mV (143 % closure — overshoot but bracketed).
4. **Base V<sub>oc</sub> within envelope:** 1.0905 V.

## What broke

### CBO collapse (782 → 18 mV)

Same failure mode as thin-shell w=2. When interface SRH is decoupled from bulk band-bending (legacy `_apply_interface_recombination` skipped under env=1), the interface-plane state evolves to a quasi-equilibrium that has weak ΔE_c sensitivity. The TE flux + Boltzmann projection captures band-bending V_1/V_2 but NOT the χ step at the heterointerface (paper eq 15: cross-side β_c = N_c1/N_c2 · exp(−ΔE_c/V<sub>T</sub>) factor).

**Missing ingredient:** paper eq 15 — cross-side density relation `n_1s = β_c · exp(−ΔE_c/V_T) · n_2s'`. The current TE flux primitive treats each side's state independently without the χ-step coupling. Sprint 7 Day 4-10 needs to either:

(a) Add a CROSS-INTERFACE TE flux that respects the χ step (paper eq 14a barrier factor).
(b) Couple n_1s and n_2s via algebraic constraint at SS (Sze-style abrupt heterojunction).

### Interface defect N<sub>t</sub> sweep Newton hang

At certain N<sub>t</sub> values (likely 1e15-1e17 cm⁻²), the SRH consumption rate exceeds the TE refill rate by orders of magnitude. The iface_state ODE becomes super-stiff: τ_SRH << τ_TE.

Newton bisection budget (10 levels) exhausts. BDF fallback engages but also struggles. Process hangs at 100 % CPU.

**Mitigation options:**
1. Cap interface_state values to physical floors (n, p ≥ 0 already done; add upper cap?).
2. Quasi-steady-state algebraic reduction — solve d(iface_state)/dt = 0 algebraically instead of integrating.
3. Tighter v_th_eff calibration — current 1e-2 m/s may need to drop further at high N<sub>t</sub>.

## Design choices that landed

- `_DEFAULT_V_TH_MS = 1.0e-2` — surface-recombination-velocity scale rather than thermal 1e5. Textbook v<sub>th</sub> makes Newton diverge at diode knee (V<sub>app</sub> ≈ 0.5 V). Sprint 7 Day 4+ revisits with QSS reduction.
- Bulk drain coupling — TE flux subtracted from bulk dn/dp at eval_n/eval_p nodes. Without this, V<sub>oc</sub> was bit-identical to legacy (interface-state was decoupled from bulk dynamics).
- Legacy `_apply_interface_recombination` skipped when N_iface_state > 0 — full replacement, not addition.

## Sprint 7 Day 4-10 plan

| Day | Deliverable |
|---|---|
| 4 | Diagnose CBO collapse — add probe script measuring sensitivity to ΔE_c via paper eq 15 factor |
| 5-6 | Add cross-interface χ-step coupling: TE flux for n_1s↔n_2s with exp(−ΔE_c/V<sub>T</sub>) factor (paper eq 14a) |
| 7 | Newton stability fix — QSS algebraic reduction for iface_state when SRH rate >> Radau timestep budget |
| 8 | Full validation gate re-run |
| 9-10 | Iterate if close-but-miss; escalate to user if still failing |

## Convention

Branch `e3-interface-plane-state` continues; Sprint 7 Day 4+ commits build on Sprint 7 Day 1-3.

**Related:** [[project-scaps-validation-parked]], Phase E3 design RFC
`2026-05-27-e3-interface-plane-state-rfc.md`, Pauwels-Vanhoutte formula
extraction `2026-05-27-e2-pauwels-vanhoutte-formula.md`.

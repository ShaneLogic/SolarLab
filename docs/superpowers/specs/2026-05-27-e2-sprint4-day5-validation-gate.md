# Phase E2 Sprint 4 Day 5 — Pauwels-Vanhoutte validation gate

**Status:** THREE hard fails. Heavy-doping limit collapses interface SRH.
**Branch:** `e2-pauwels-vanhoutte`
**Input:** `outputs/scaps_validation_e2_pauwels_vanhoutte/report.md`

## Result summary

| Sweep | Legacy main | BBD | Thin-shell w=2 | **PV heavy-doping** | SCAPS | Verdict |
|---|---|---|---|---|---|---|
| Base V<sub>oc</sub> | 1.069 | 1.054 | 1.088 | **1.0905 V** | 1.168 | within envelope ✓ |
| CBO range | 782 | 847 | 16 | **663 mV** | 918 | **regressed 85 % → 72 %** ✗ |
| ETL doping range | 1075 | 1542 | 17 | **1419 mV** | 137 | **WORSE than legacy** ✗ |
| PVK doping range | 34 | 21 | 48 | 50 mV | 34 | range overshoots |
| **Interface defect N<sub>t</sub>** | **210** | 208 | 206 | **8 mV** | 282 | **COLLAPSED 74 % → 3 %** ✗ |
| Bulk N<sub>t</sub> | 0 | 0 | 0 | 0 | 39 | unchanged (still blocked) |
| Bulk E<sub>t</sub> | 2 | 1 | 3 | 3 | 0 | flat both |

Plus: 23/23 SCAPS-subset regression pass (legacy bit-identity confirmed).

## RFC pass criteria check

| Criterion | Required | Measured | Pass? |
|---|---|---|---|
| ETL doping range ≤ 200 mV | ≤ 200 | 1419 mV | **FAIL** |
| CBO closure ≥ 80 % | ≥ 80 | 72 % | **FAIL** |
| Base V<sub>oc</sub> ∈ [1.05, 1.25] V | bounded | 1.0905 V | PASS |
| No new test failures | 0 | 0 | PASS |

**Two hard fails on primary criteria + one collapsed sweep (interface
defect N<sub>t</sub>).** Worse than the BBD or thin-shell gates.

## Diagnosis — heavy-doping limit kills interface SRH

PV factor at V<sub>app</sub> sweep points:
```
V_app = 0:     V_2 = V_bi ≈ 1.1 V    →  exp(-V_2/V_T) = exp(-42) ≈ 1e-18
V_app = 1.0:   V_2 = 0.1 V           →  exp(-3.86)    ≈ 0.021
V_app = 1.05:  V_2 = 0.05 V          →  exp(-1.93)    ≈ 0.145
V_app = 1.10:  V_2 = 0               →  exp(0)        = 1.0
```

Factor ramps from ~0 at low bias to 1 near V<sub>bi</sub>. Suppresses p_iface
across the entire useful V range of the sweep. Interface SRH rate
collapses → interface defect sweep collapses (210 → 8 mV) → CBO sweep
weakens → only ETL doping survives because it shifts V<sub>bi</sub> directly.

The heavy-doping assumption is mathematically equivalent to assuming
ALL band-bending happens in PVK with full V<sub>bi</sub> − V<sub>app</sub> drop. For
SCAPS-mirror (PVK light-doped 1e14, ETL heavy-doped 1e18) this is
qualitatively correct. The QUANTITATIVE problem: the factor decays
exponentially while the rate computation expects a more moderate
depletion (closer to factor 0.1-0.5 not 1e-18-0.1).

The paper's full formulation uses V_1, V_2 self-consistently with the
SRH occupancy AND the thermionic-emission boundary AND the diffusion
current — solving these as a coupled system gives a different
effective factor. The heavy-doping limit truncates too aggressively.

## Three-prototype failure pattern

| Prototype | CBO | ETL doping | Interface N<sub>t</sub> |
|---|---|---|---|
| BBD (local Δφ) | 92 % ✓ | 1542 mV ✗ | 208 mV ✓ |
| Thin-shell w=2 | 1.7 % ✗ | 17 mV ✓ | 206 mV ✓ |
| PV heavy-doping | 72 % ~ | 1419 mV ✗ | **8 mV ✗** |
| Legacy E1.5 | 85 % | 1075 mV ✗ | 210 mV |
| SCAPS target | 100 % | 137 mV | 282 mV |

Each prototype breaks a DIFFERENT physics knob. Legacy is the most
balanced even though it doesn't close ETL doping. **No single-knob
fix exists** within the current solver's discretisation framework.

## Three paths forward

### (d) Full Pauwels-Vanhoutte (NOT heavy-doping limit)

Solve V_1, V_2 self-consistently per RHS call. Requires implementing
the full coupled system from eqs (12)-(19) + (A1)-(A7). Multi-week.
High risk: the paper itself only solves analytically under simplifying
assumptions; numerical implementation in our solver needs careful
Newton stability work near the heterointerface.

Estimated effort: 3-5 weeks. Probability of success: medium.

### (c) Park ETL doping as structural (Phase G/F precedent)

Accept the 8× over-sensitivity in ETL doping as a structural limit of
the current SolarLab interface SRH discretisation. Document in
partner report alongside the Phase G base V<sub>oc</sub> gap (74 mV) and Phase
F PVK doping direction. Ship main-branch state as the final
deliverable.

Effort: 1-2 days for partner report + memory update.

### (e) Hybrid PV with milder depletion factor

Calibrate the PV factor empirically: `pv_factor = exp(-α·V_2/V_T)`
with α ∈ [0, 1] tuned to balance interface defect sensitivity against
CBO closure. α = 0 reduces to legacy; α = 1 is full heavy-doping
limit. Probe α ∈ {0.1, 0.3, 0.5} to find balance.

Risk: empirical calibration; not principled. Smells like the
calibration_factor trick again.

Effort: 1-2 days for probe + commit.

## Recommendation

**(c) Park option.** Three prototype paths have been explored. Each
fails a different criterion. The fundamental issue is that SCAPS uses
a different solver convention (thermionic emission + SRH on interface
plane) that our Method-of-Lines + Scharfetter-Gummel framework cannot
replicate without research-grade refactor.

Ship the main-branch state (CBO 85 %, PVK/ETL N<sub>t</sub> 74 %, base V<sub>oc</sub>
within envelope) as the final partner deliverable. Document the ETL
doping over-sensitivity, PVK doping direction, and bulk N<sub>t</sub> mismatch
as known limits requiring SCAPS source access.

Alternative: (d) full Pauwels-Vanhoutte. Only justified if partner
explicitly requests deeper investigation budget.

## Convention

- Branch `e2-pauwels-vanhoutte` parked as failure evidence alongside
  `e2-bbd-face-density` and `e2-thin-shell-srh`. None merge to main.
- Findings doc cherry-picks to main as audit trail.
- Three-prototype failure pattern documented for future-self / new
  team members reading parked memory.

**Related:** [[project-scaps-validation-parked]], Phase E2 design RFC
`2026-05-27-e2-design-rfc.md`, BBD gate, thin-shell gate, PV formula
`2026-05-27-e2-pauwels-vanhoutte-formula.md`.

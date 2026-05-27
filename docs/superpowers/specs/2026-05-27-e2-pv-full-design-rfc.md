# Phase E2 Sprint 5 — Full Pauwels-Vanhoutte design RFC

**Status:** design (no code change yet)
**Branch:** `e2-pv-full` (off main `515bafb`)
**Predecessor failures:** BBD (Sprint 2), thin-shell (Sprint 3),
PV heavy-doping single-sided (Sprint 4) — see parked branches +
gate findings.

## Goal

Solve SolarLab ↔ SCAPS parity properly. Implement faithful Pauwels-
Vanhoutte 1978 interface SRH without heavy-doping shortcut. Target
all four RFC gate criteria:
- ETL doping range ≤ 200 mV (legacy 1075 mV; SCAPS 137 mV)
- CBO closure ≥ 80 % (legacy 85 %)
- Base V_oc ∈ [1.05, 1.25] V envelope
- No regression on 23/23 SCAPS-subset tests

## Why Sprint 4 PV failed — three missing ingredients

| Ingredient | Sprint 4 PV | Full PV (Sprint 5) |
|---|---|---|
| V_1, V_2 partition | V_1 = 0, V_2 = V_bi − V_app (heavy-doping shortcut) | Solved from charge balance: V_1 / V_2 = (ε_2·N_2)/(ε_1·N_1) |
| SRH paths | Single-sided (j_s2 only — PVK-side hole capture) | Two-sided: j_s = j_s1 + j_s2 per eq (7) |
| Quasi-Fermi at interface | Assumed flat across heterojunction | Properly split via TE boundary (eq 14a/b) |
| Suppression behavior at low V_app | exp(−V_bi/V_T) ≈ 1e-18 (over-suppresses) | TE-modulated; reduces to bulk-mode-coupling not exp-decay |

Single missing ingredient = single failure mode. All three missing
together = three observed failure modes across BBD / thin-shell /
single-sided PV.

## Architecture overview

### Static cache (build_material_arrays)

Per interface k, compute and cache on `MaterialArrays`:

```python
# Charge-balance partition of V_bi between layers:
eps_1, N_1 = mat.eps_r[idx_ETL_bulk], stack.layers[ETL].N_D  # heavy side
eps_2, N_2 = mat.eps_r[idx_PVK_bulk], stack.layers[PVK].N_A  # light side
denom = eps_1 * N_1 + eps_2 * N_2
mat.interface_V_partition_1[k] = eps_2 * N_2 / denom  # fraction in heavy side
mat.interface_V_partition_2[k] = eps_1 * N_1 / denom  # fraction in light side

# Band-offset factors (per eq 15):
mat.interface_chi_step[k]  = chi[idx_R] - chi[idx_L]  # ΔE_c for electron
mat.interface_Eg_step[k]   = Eg[idx_R] - Eg[idx_L]    # ΔE_g

# N_c, N_v ratios (per eq 15 β_c):
mat.interface_beta_c[k] = N_c_ETL / N_c_PVK
mat.interface_beta_v[k] = N_v_PVK / N_v_ETL

# Equilibrium interface densities (single hole Fermi level reference):
mat.interface_n_1s0[k] = n_eq_ETL · exp(V_1_eq)
mat.interface_p_2s0[k] = p_eq_PVK · exp(-V_2_eq)
```

### Per-RHS computation (`_apply_interface_recombination`)

For each interface k with V_app from `assemble_rhs` kwarg:

```python
# Step 1 — V_1, V_2 at this V_app (depends only on doping + V_bi):
V_total = mat.V_bi_eff - V_app
V_1 = mat.interface_V_partition_1[k] * V_total  # ETL band-bending
V_2 = mat.interface_V_partition_2[k] * V_total  # PVK band-bending

# Step 2 — interface-plane n, p on each side (eqs 8, 9, 11):
n_1s = n[bulk_ETL] * exp(-V_1 / V_T)  # ETL-side electron at interface
p_1s = p[bulk_ETL] * exp(+V_1 / V_T)  # ETL-side hole at interface
n_2s = n[bulk_PVK] * exp(+V_2 / V_T)  # PVK-side electron at interface
p_2s = p[bulk_PVK] * exp(-V_2 / V_T)  # PVK-side hole at interface

# Step 3 — two-sided Shockley-Read SRH (eqs 12, 13):
# j_s1 — capture electron from ETL CB + hole from PVK side:
R_s1 = (n_1s * p_2s - ni_sq_eff) /
       ((n_1s + n_1) / v_p_1 + (p_2s + p_1) / v_n_1)
# j_s2 — capture electron from PVK CB + hole from ETL side:
R_s2 = (n_2s * p_1s - ni_sq_eff) /
       ((n_2s + n_1) / v_p_2 + (p_1s + p_1) / v_n_2)
R_s_total = R_s1 + R_s2

# Step 4 — volumetric loss at interface node:
R_vol = R_s_total / mat.dx_cell[idx]
dn[idx] -= R_vol
dp[idx] -= R_vol
```

### Surface velocities — keep existing per-interface v_n, v_p

Use the existing `ifaces[k] = (v_n, v_p)` for both j_s1 and j_s2.
Phase E1.6 `calibration_factor` still multiplies. Two-sided rate
naturally scales 2× vs single-sided when V_1 ≈ V_2 (symmetric doping);
heavily asymmetric stacks have one side dominate by orders of
magnitude.

## Validation predictions

| Sweep | Legacy | Full PV expected |
|---|---|---|
| Base V_oc | 1.069 V | ~1.04-1.09 V (down from 1.069 if R increases at V_oc) |
| CBO range | 782 mV | 700-900 mV (preserves; V_1+V_2=V_bi tracks ΔE_c) |
| ETL doping | 1075 mV | 200-400 mV (charge partition kills bulk-doping sensitivity beyond log-V_bi) |
| Interface defect N_t | 210 mV | 200-280 mV (two-sided rate restores defect sensitivity) |
| PVK doping | 34 mV | 30-60 mV (range preserved) |

If predictions land within these envelopes, Sprint 5 succeeds and
ships.

## Sprint 5 schedule

| Day | Deliverable | Effort |
|---|---|---|
| 1 (now) | Design RFC + branch cut | 0.5d |
| 2-3 | RED tests + MaterialArrays cache extension | 1.5d |
| 4-6 | GREEN — `_apply_pauwels_vanhoutte_full` implementation | 3d |
| 7 | Local regression + ad-hoc V_oc + N_D_ETL spot probe | 1d |
| 8 | Full SCAPS validation gate | 1d |
| 9-10 | Iterate: parameter tuning if gate close-but-miss | 2d |
| 11-13 | Ship: env-var → InterfaceDefect.recombination_model field promotion, scaps_mirror.yaml update, frontend round-trip plumbing | 3d |
| 14 | Partner report update + memory write | 1d |

**Total: 14 days budgeted, 10-15 working days realistic.**

## Risk analysis

| Risk | Likelihood | Mitigation |
|---|---|---|
| Two-sided rate over-suppresses interface defect closer to BBD failure | medium | RED test pins interface defect sweep ≥150 mV before declaring GREEN |
| V_1/V_2 partition charge balance formula wrong for our heterojunction stack | medium | Cross-check against textbook (Sze) + numerical comparison vs analytical p+n |
| Newton convergence breaks at low N_D_ETL where charge partition shifts | medium | Sprint 5 Day 7 probe at N_D_ETL ∈ {1e14, 1e16, 1e18}; v_max_max_attempts already provides retry |
| ETL doping over-sensitivity is structural (not closed even with full PV) | medium-high | Sprint 5 Day 8 gate measures; if fail, escalate to Phase E3 carrier-statistics refactor (Boltzmann-degenerate) |
| Two-sided rate doubles up calibration_factor effect | low | Sprint 5 Day 9 re-fits calibration_factor; target cf ∈ [0.5, 5] |
| PVK doping direction still flipped | low (Phase F precedent) | Acceptable scope — flag as known limit |

## What's different vs Sprint 4 failed prototype

Single-sided PV (Sprint 4) had:
```
n_iface = n[idx+1]  # bulk ETL, no projection
p_iface = p[idx-1] * exp(-V_2/V_T)
R = interface_recombination(n_iface, p_iface, ...)  # ONE rate
```

Full PV (Sprint 5) has:
```
n_1s = n[bulk_ETL] * exp(-V_1/V_T)
p_1s = p[bulk_ETL] * exp(+V_1/V_T)
n_2s = n[bulk_PVK] * exp(+V_2/V_T)  
p_2s = p[bulk_PVK] * exp(-V_2/V_T)
R = R_s1(n_1s, p_2s) + R_s2(n_2s, p_1s)  # TWO rates summed
```

Plus V_1 ≠ 0 because charge partition isn't 100% on light side.

## Pass / fail criteria (Sprint 5 Day 8 gate)

### PASS
- ETL doping ≤ 200 mV ✓
- CBO ≥ 80 % closure ✓
- Base V_oc ∈ [1.05, 1.25] V ✓
- Interface defect N_t ≥ 150 mV (not collapsed) ✓
- 23/23 SCAPS-subset regression passes ✓

### FAIL (any of)
- ETL doping > 300 mV
- CBO < 70 % closure
- Interface defect N_t collapsed (< 100 mV)
- Base V_oc outside envelope
- > 2 regression failures

If FAIL: escalate to Phase E3 (carrier statistics refactor) or
park per Phase F/G precedent.

## Convention

- Branch `e2-pv-full` cut off main `515bafb`. Commits ship in atomic
  steps per superpowers TDD skill.
- Env var `SOLARLAB_PV_FULL=1` activates during prototype phase
  (Days 4-8). Promoted to per-interface YAML field after Day 8 gate
  passes (Days 9-13).
- Static cache extensions land on `MaterialArrays` as new frozen
  tuple fields; mutator stays in `build_material_arrays`.
- TDD: RED test for each sub-step (V_1/V_2 partition, two-sided rate,
  legacy bit-identity).

**Related:** [[project-scaps-validation-parked]], Phase E2 design
RFC `2026-05-27-e2-design-rfc.md`, Sprint 4 PV gate failure
`2026-05-27-e2-sprint4-day5-validation-gate.md`, PV formula
extraction `2026-05-27-e2-pauwels-vanhoutte-formula.md`.

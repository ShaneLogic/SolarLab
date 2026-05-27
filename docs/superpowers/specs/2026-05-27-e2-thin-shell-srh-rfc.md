# Phase E2 Sprint 3 design RFC — thin-shell volumetric SRH

**Status:** design RFC (no code change yet)
**Branch:** `e2-thin-shell-srh` (cut off main `20d8661`)
**Predecessor:** Phase E2 BBD validation gate FAILED on ETL doping
(`docs/superpowers/specs/2026-05-27-e2-sprint2-day2-3-validation-gate.md`)
**Parked branch:** `e2-bbd-face-density` retained as failure evidence;
not merged.

## Goal

Close the SCAPS ETL doping over-sensitivity (1075 mV → target ≤ 200 mV)
WITHOUT regressing the CBO closure (currently 85 % on main).

## Why BBD failed and what thin-shell might fix

BBD multiplied bulk-side density by `exp(-Δφ/V_T)`. At low N_D_ETL the
depletion zone widens INTO ETL, Δφ grows, and the exp factor amplifies
N_D sensitivity rather than damping it. ETL doping range got WORSE
(1075 → 1542 mV).

Thin-shell volumetric SRH samples n, p across multiple grid nodes around
the interface — naturally captures the depletion-zone shape rather than
forcing it through a single-point exp factor. Hypothesis: the
distributed sampling averages out the BBD low-N_D blow-up.

## Architecture

### Definition

Treat interface SRH as a volumetric source over a shell of `n_shell`
grid nodes centred on the interface node:

```
shell_indices = [idx - n_shell//2, ..., idx + n_shell//2]
shell_width = sum(dx_cell[i] for i in shell_indices)  # m
N_t_vol     = N_t_areal / shell_width                  # m^-3
```

For each node `i` in the shell, evaluate per-node SRH:

```
R_i = N_t_vol · σ · v_th · (n[i]·p[i] - ni_sq_eff[i])
                         / (n[i] + p[i] + n1 + p1)
dn[i] -= R_i
dp[i] -= R_i
```

### Differences from current E1.5 cross-carrier

| Aspect | E1.5 (current) | Thin-shell |
|---|---|---|
| Sample point | single — n[idx+1], p[idx-1] | distributed — n[i], p[i] per shell node |
| Cross-carrier | YES (n ETL, p PVK) | NO (same-node n·p) |
| Volume conversion | R_s / dx_cell[idx] | R_vol = N_t_vol · ... directly |
| Density consistency | mismatched bulk samples | self-consistent per-node |
| Newton risk | low | medium (multi-node depletion is sensitive) |

### Negative-p artifact at idx

Phase A1 probe showed `p[idx]` can be slightly negative at the
PVK/ETL interface (-2.3e21 m⁻³) due to state-vector reconstruction
with ions. Thin-shell uses `n[i]·p[i]` per node; the negative-p node
will give a NEGATIVE recombination rate (carrier GENERATION instead
of loss).

**Mitigation:** clamp per-node n, p to `[0, +inf)` before the SRH
evaluation, OR exclude the interface node itself from the shell and
use only the PVK-side and ETL-side neighbours. The latter is closer
to the "thin-shell" intent — the SRH centres physically lie just OFF
the interface plane, in the bulk material.

Pick the OFF-interface-node approach: shell = [idx-1, idx+1] for
`n_shell=2`. Generalises to [idx-2, idx-1, idx+1, idx+2] for
`n_shell=4`. Always SKIP idx itself.

### LoC estimate

- `solver/mol.py`: add `_apply_thin_shell_recombination` (~50 lines).
  Gated by env var `SOLARLAB_THIN_SHELL_SRH=N` (where N is shell
  width in nodes; 0 or unset = legacy E1.5 path).
- `_apply_interface_recombination`: keep E1.5 path; add early-return
  dispatch to thin-shell when env active.
- `assemble_rhs`: no change beyond the existing phi= plumbing.

**Total: ~50-80 LoC + 5-8 RED tests.**

### Test surface

1. **Legacy bit-identity** when env unset (mirror BBD env-unset test).
2. **Env=2 activates shell width 2** — V_oc moves from legacy.
3. **Env=4 activates shell width 4** — V_oc moves further than env=2.
4. **Env=non-int / =0 takes legacy path** (defensive).
5. **Finite JV under env=2** — no NaN/Inf.
6. **np product non-negative at shell nodes** — clamp validation.
7. **CBO direction preserved** at shell=2.

### Validation gate (Sprint 3 Day 6-7)

Same RFC criteria as BBD:
- ETL doping range ≤ 200 mV (PRIMARY)
- CBO closure ≥ 80 % (must not regress)
- Base V_oc ∈ [1.05, 1.25] V
- No new test failures

If fail → escalate to Pauwels-Vanhoutte paper acquisition (Sprint 4
option (c) from gate findings).

## Sprint 3 schedule

| Day | Deliverable |
|---|---|
| 1 (now) | RFC committed (this doc) |
| 2 | RED test file `test_e2_thin_shell_srh_prototype.py` |
| 3-4 | GREEN implementation in `solver/mol.py` with env-var gate |
| 5 | Local regression run + ad-hoc V_oc probe |
| 6-7 | Validation gate run + findings doc |
| 8-10 | Ship (if pass) — promote to InterfaceDefect.recombination_model field, scaps_mirror.yaml update, partner report update |

## Risk analysis

| Risk | Likelihood | Mitigation |
|---|---|---|
| Thin-shell fails ETL doping gate same as BBD | medium | Different physics surface (distributed sampling); RFC gate at Sprint 3 Day 6-7 |
| Newton convergence breaks at low N_D | medium | Phase E1.9 v_max_max_attempts already mitigates; test on full SCAPS sweep before declaring GREEN |
| Negative-p artifact pollutes shell | known | Skip idx node; clamp per-node n, p |
| Shell width parameter has no good default | medium | Probe n_shell ∈ {2, 4, 6} during prototype to find sweet spot |
| Thin-shell regresses CBO (band-offset physics) | low-medium | Same gate criteria as BBD — CBO regression is hard fail |

## Convention

- Branch `e2-thin-shell-srh` off main. Sprint 3 Day 1 commit only
  this RFC.
- TDD: RED test (Day 2) → GREEN implementation (Day 3-4) → validation
  gate (Day 6-7). One atomic commit per Day.
- Env-var gating (Sprint 3.1 prototype) → InterfaceDefect data-model
  field promotion (Sprint 3.2 if pass), mirroring the BBD RFC E2.1 → E2.2
  flow.
- Validation outputs land in `outputs/scaps_validation_e2_thin_shell/`
  with shell-width suffix for partner-readable history.

## Open question

Shell-width default: n_shell=2 (minimal — just immediate neighbours of
idx) is safest. n_shell=4 picks up more of the depletion zone but
risks coupling with the bulk SRH path. Sprint 3 Day 5 ad-hoc probe
sweeps n_shell ∈ {2, 4, 6} on scaps_mirror.yaml and reports V_oc
movement. Default chosen based on probe outcome.

**Related:** [[project-scaps-validation-parked]], Phase E2 design RFC
`2026-05-27-e2-design-rfc.md`, BBD validation gate
`2026-05-27-e2-sprint2-day2-3-validation-gate.md`.

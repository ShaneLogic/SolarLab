# Phase E4 Sprint 8 — split-interface-flux design RFC

**Status:** design (no code change yet)
**Branch:** `e4-split-interface-flux` (off `e3-interface-plane-state` Sprint 7 Day 4-6 commit `c203d12`)
**Predecessors:** five failed prototypes — BBD / thin-shell / single PV / full PV / iface-state χ-step. See gate findings docs.

## Goal

Close SolarLab ↔ SCAPS parity by refactoring `physics/continuity.py:carrier_continuity_rhs` at heterointerface faces. Split the single chi/Eg-aware SG flux into two per-layer half-fluxes coupled to interface-plane state via thermionic-emission boundary condition (paper eq 14a). Adds the MISSING bulk-side TE BC ingredient that all five prior prototypes lacked.

## Current architecture

Single SG flux at heterointerface face `f` (between idx and idx+1):
```
J_n[f] = sg_fluxes_n(phi_n, n, dx, D_n, V_T)[f]       # one flux per face
# Optional TE cap when |ΔE_c| > 0.05 eV (current path)
```

Bulk continuity at idx, idx+1 uses J<sub>n</sub>[f] as the boundary face flux for both bulk nodes. No interface-plane state participation.

## Proposed architecture

At heterointerface face f connecting layer L (idx) and layer R (idx+1):

1. **Two half-fluxes** instead of one:
   ```
   J_n_L_half[f] = single-layer SG flux from idx into interface plane on L side
   J_n_R_half[f] = single-layer SG flux from interface plane on R side into idx+1
   ```
   Each uses ONLY the local-layer chi, Eg, D — no cross-layer harmonic averaging.

2. **Interface plane carries 4 state densities** (Phase E3 done):
   `n_L_iface[k], p_L_iface[k], n_R_iface[k], p_R_iface[k]`

3. **TE boundary couples bulk to iface state** (paper eq 14a):
   ```
   J_n_TE_L = v_th · (n[idx]   · exp(-V_1/V_T) − n_L_iface[k])
   J_n_TE_R = v_th · (n[idx+1] · exp(-V_2/V_T) − n_R_iface[k])
   ```
   where V_1, V_2 are charge-balance band-bending partitions.

4. **Cross-side TE within iface plane** (Sprint 7 Day 4-6 work, already in place):
   ```
   J_n_cross = v_cross · (n_R_iface · exp(-ΔE_c/V_T) − n_L_iface)
   ```

5. **Interface SRH on iface state** (Sprint 6 done):
   ```
   R_s1 = SRH(n_L_iface, p_R_iface, ni²_eff_chi, ...)
   R_s2 = SRH(n_R_iface, p_L_iface, ni²_eff_chi, ...)
   ```

Net result: bulk Fermi level responds to χ via TE BC at the heterointerface face. CBO sweep affects bulk J<sub>n</sub> through the TE rate factor exp(-V_1/V<sub>T</sub>) which depends on band-bending V_1, which depends on χ via charge-balance partition. The MISSING ingredient that killed all five prior prototypes.

## File touches

| File | Sprint 8 | Sprint 9 |
|---|---|---|
| `physics/continuity.py` | new `split_interface_flux()` helper; `carrier_continuity_rhs` gates on env+mat.N_iface_state | adds J_n_TE_L/R wiring |
| `solver/mol.py` | passes new `interface_state` params dict key for chi/Eg-aware split | removes legacy `_apply_interface_recombination` call when split active |
| `physics/interface_plane.py` | unchanged | `compute_interface_te_fluxes` reads bulk n[idx], n[idx+1] for the bulk-anchored projection |
| `tests/unit/physics/test_continuity_split.py` | NEW — RED tests pinning split-flux contract | — |
| `tests/integration/test_e4_split_validation.py` | — | NEW — gate criteria |

## Sprint 8 schedule

| Day | Deliverable |
|---|---|
| 1 (this commit) | RFC + branch cut |
| 2-3 | RED tests for split_interface_flux + dispatch gate |
| 4-7 | GREEN: per-layer half-flux implementation |
| 8-10 | Legacy bit-identity probe (env unset must reproduce current SG flux) + iteration |

## Sprint 9 schedule

| Day | Deliverable |
|---|---|
| 1-3 | TE BC wiring: J_n_TE_L/R from interface_plane.py into continuity.py params |
| 4-7 | Combined flow: bulk SG → iface state via TE → SRH → cross-side TE |
| 8-10 | Full SCAPS validation gate + iteration |

## Pass / fail criteria (Sprint 9 Day 10 gate)

| Criterion | Required |
|---|---|
| ETL doping range | ∈ [60, 270 mV] (SCAPS 137 ± 100 %) |
| CBO closure | ≥ 50 % of SCAPS 918 mV |
| Interface defect N<sub>t</sub> | ≥ 150 mV (not collapsed) |
| Base V<sub>oc</sub> | ∈ [1.05, 1.25] V envelope |
| 23/23 SCAPS-subset | PASS |
| Slow regression (TMM) | PASS |
| Newton stability | full sweep completes |

If gate fails: escalate to user. Three remaining options will then be:
- Continue with deeper refactor (probably modify SG flux primitive itself)
- Park and ship main
- Try a completely different solver architecture (Newton-Krylov?)

## Risk analysis

| Risk | Likelihood | Mitigation |
|---|---|---|
| Splitting SG flux breaks IonMonger benchmark | medium | Gate behind env+mat.N_iface_state; legacy path bit-identical |
| 2D solver (`twod/continuity_2d.py`) needs same split | low | Defer 2D port to Sprint 11 if Sprint 9 gate passes |
| TE BC rate coupling makes Newton stiff | high | Use small v_th_TE first (1e-2 m/s) like Sprint 7 ; iterate |
| Charge conservation lost across heterointerface | medium | RED test pinning ∂(n+p)/∂t = G - R - (J_in - J_out)/dx |
| Half-flux convention sign error | medium | RED test pinning legacy bit-identity at chi=Eg=0 |

## Convention

- Branch `e4-split-interface-flux` cut off `e3-interface-plane-state` Sprint 7 Day 4-6 commit. Keeps the iface_state machinery from Sprint 6 + 7.
- Env var `SOLARLAB_INTERFACE_PLANE_STATE=1` activates BOTH iface_state AND split-flux. Sprint 10 promotes to `SimulationMode.use_interface_plane_state`.
- TDD: RED → GREEN → REFACTOR per superpowers TDD skill.
- Atomic commits with `Constraint:` / `Rejected:` / `Confidence:` / `Directive:` trailers.

**Related:** [[project-scaps-validation-parked]], all five prior prototype gate findings docs, Phase E3 design RFC + Sprint 7 Day 4-6 χ-step gate.

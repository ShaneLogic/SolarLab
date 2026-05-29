# Phase E4 Sprint 9 Day 1-3 — Newton hang on JV sweep with split-flux wire-through

**Status:** FAILED — Newton hangs on test_jv_sweep_voc_in_envelope_with_split after >24 min wallclock
**Branch:** `e4-split-interface-flux` (Sprint 8 Day 4-7 scaffold preserved at `05a7bfd`)
**Stash:** "Sprint 9 Day 1-3 wire-through — Newton hang on JV sweep" (recoverable via `git stash list`)

## What was attempted

Sprint 9 mass-conserving wire-through of the split-flux scaffold landed in Sprint 8. Three coupled changes:

1. **Helper** `compute_interface_split_fluxes(mat, iface_state, phi, n, p, V_T)` in `physics/interface_plane.py` — computes (J_L_n, J_R_n, J_L_p, J_R_p) per heterointerface face once per RHS evaluation.

2. **continuity.py** override consumes pre-computed half-fluxes (fast-path); falls back to inline computation when absent (Sprint 8 scaffold path preserved).

3. **assemble_rhs** changes:
   - Calls helper once per RHS, injects result into `carrier_params["interface_split_data"]`
   - Iface state RHS replaces single-side TE (`v_th_eff` → 0) with split-flux mass-conserving coupling:
     ```
     d(n_1s)/dt += -J_R_n / Q / dx_iface    (loses to R bulk)
     d(p_1s)/dt += -J_R_p / Q / dx_iface
     d(n_2s)/dt += +J_L_n / Q / dx_iface    (gains from L bulk)
     d(p_2s)/dt += +J_L_p / Q / dx_iface
     ```
   - Cross-flux (paper eq 15, `v_cross_eff=1e-2`) + SRH retained
   - Legacy Sprint 7 single-side bulk drain gated off when split path active

Intent: bulk drain via continuity.py divergence override at heterointerface = iface gain via assemble_rhs RHS — mass-conserving by construction.

## Failure mode

`test_jv_sweep_voc_in_envelope_with_split` hangs >24 min wallclock (legacy Sprint 7 path completes JV sweep in ~17 s).

4 fast tests passed before stall: D_n_node cache, D_n/D_face consistency, legacy bit-identity (env unset), assemble_rhs finite at V=0. 5th test (illuminated JV sweep with env=1) hangs on Radau Newton iteration.

## Diagnosis — order-of-magnitude bulk drain

At dark equilibrium with χ-step-anchored iface init:

  n_R_iface = n_R_eq · exp(-V_1/V<sub>T</sub>)  ≈ 1e24 (ETL bulk, V_1 ≈ 0)
  n_L_iface = n_1s · exp(-ΔE_c/V<sub>T</sub>)   ≈ 2e21 (χ-step from 1s, ΔE_c=0.16 eV)
  p_R_iface = p_R_eq · exp(+V_1/V<sub>T</sub>)  ≈ 1e-4 (ETL minority)
  p_L_iface = p_1s · exp(-ΔE_v/V<sub>T</sub>)   ≈ 1e9  (χ-step from 1s, ΔE_v=-2.31 eV capped at -30/V<sub>T</sub>)

Bulk under illumination at V=0:
  n[idx]   = PVK minority + photo-injection ≈ 1e22 m⁻³
  n[idx+1] = ETL majority ≈ 1e24 m⁻³ (unchanged)

L-half flux at heterointerface face:
  J_L_n ≈ q·D_L/(dx_face/2) · (B(xi)·n_L_iface − B(-xi)·n[idx])
       ≈ q·(1e-4)/(5e-10) · (1.0·2e21 − 1.0·1e22)
       ≈ −2.5e8 A/m²

Legacy J_n_legacy[f] at heterointerface face (chi-step suppressed):
  J_n_legacy ≈ q·D/dx · (B(6.18)·n[idx+1] − B(-6.18)·n[idx])
            ≈ q·(1e-4)/(1e-9) · (0.013·1e24 − 6.18·1e22)
            ≈ small (≈ 1e6 A/m²)

Override delta: dn[idx] += (J_L_n − J_n_legacy) / (Q · dx_cell[idx])
              ≈ (−2.5e8 − 1e6) / (1.6e-19 · 1e-9)
              ≈ −1.5e36 m⁻³/s

Astronomical bulk drain rate. Newton cannot contract. Solver bisects, then hangs.

## Root cause — discretisation mismatch at heterointerface

The χ-step suppression in the legacy SG flux (factor of exp(-ΔE_c/V<sub>T</sub>) ≈ 2e-3 in B(xi)) is what keeps the bulk-bulk SG flux physical at the heterointerface. Removing it (split half-flux uses only same-layer chi) gives a much larger raw flux value that the bulk node cannot accommodate.

Mass conservation between bulk drain and iface gain is mathematically correct, BUT the magnitude of both is too large for the Radau time integrator to digest. The iface_state ODE inherits the same large dy/dt → super-stiff ODE → Newton hang.

This is fundamentally the same failure mode as the 6 prior prototypes (BBD, thin-shell, single-PV, full-PV, Sprint 7 iface_state, Sprint 7 χ-step) — the discrete heterointerface SG flux is so tightly coupled to the χ-step physics that replacing it with any alternative either over-amplifies or over-suppresses the local rate by 10+ orders of magnitude.

## Seven-prototype pattern (cumulative)

|  | ETL doping | CBO closure | Other failure |
|---|---|---|---|
| Legacy E1.5 | 1075 mV (8× over) | 85 % ✓ | balanced |
| BBD | 1542 mV (11× over) | 92 % ✓ | makes ETL worse |
| Thin-shell w=2 | 17 mV (8× under) | 1.7 % ✗ | kills CBO |
| Single PV | 1419 mV ✗ | 72 % | kills interface N<sub>t</sub> |
| Full PV (1-side) | 1 mV ✗ | 72 % | hang on PVK sweep |
| Sprint 7 iface-state | 15 mV (9× under) | 1.96 % ✗ | hang on interface N<sub>t</sub> |
| Sprint 7 χ-step | 15 mV ✗ | 18 mV ✗ | interface N<sub>t</sub> collapsed |
| **Sprint 9 split-flux** | n/a — Newton hang | n/a | bulk drain ~ 1e36 m⁻³/s |
| SCAPS target | 137 mV | 100 % | — |

**No prototype brackets SCAPS on ETL doping AND preserves CBO.** Discretisation-level limit reached.

## What scaffold survives on main

Sprint 8 Day 1-7 scaffold (helper + tests + dormant override) preserved on branch:
- `physics/continuity.py:split_interface_flux` + `split_interface_flux_p` helpers
- `physics/continuity.py` divergence override gated by `params["interface_split_data"]`
- `MaterialArrays.D_n_node` / `D_p_node` cache
- 7 unit + 5 integration tests (all GREEN with env unset; activated path is dormant)

Future work can resurrect the Sprint 9 wire-through from stash and iterate on:
- Smoothing the split-flux magnitude (e.g., interpolate iface_state between dark eq + illuminated chi-step-aware bulk)
- Adaptive sub-stepping at heterointerface face during Radau Newton
- Newton-Krylov solver instead of full Radau Newton

## Three remaining realistic options

| Option | Effort | Risk | Outcome |
|---|---|---|---|
| **(A) Park** | 1 day docs/memory | low | Ship main as Phase H. 7 prototypes attempted, all fail. Sprint 8 scaffold stays as research foundation. Partner has 85 % CBO closure, 74 % interface N<sub>t</sub> closure baseline. |
| **(B) Newton-Krylov solver refactor** | 4-6 weeks | high | Switch Radau Jacobian solve to Krylov subspace method (e.g., GMRES). May handle super-stiff iface_state ODE that current LU-direct Newton cannot. |
| **(C) Semi-implicit time stepping** | 3-4 weeks | high | Treat iface_state evolution as algebraic constraint solved per-step (quasi-steady-state reduction). Decouples stiff iface dynamics from main Radau step. |

## Recommendation

**(A) Park.** 7 prototype failures is conclusive evidence. The MoL+SG discretisation cannot reproduce SCAPS' interface SRH without a solver-level refactor (Newton-Krylov OR QSS reduction OR completely different framework). Cost-benefit on (B)/(C) is unclear and may produce yet another "small win + new failure mode" pattern.

Surface to user for direction.

**Related:** all 7 prior prototype gate findings docs.

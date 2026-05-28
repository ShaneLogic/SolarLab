# Phase E8 — interface-plane band-bending projection

**Status:** implemented, env-gated (`SOLARLAB_IFACE_PROJ=1`), default OFF.
**Date:** 2026-05-29
**Closes:** the Nd_ETL trend gap that E1–E7 left parked as "requires
multi-week SG-face-density refactor" — without the refactor.

## Core problem (re-derived independently of the parked narrative)

The production E1.5 cross-carrier interface SRH (`solver/mol.py:
_apply_interface_recombination`) samples the **bulk-interior** densities
`n[idx+1]` (ETL side) and `p[idx-1]` (PVK side) directly. SCAPS uses the
band-bending-depleted **interface-plane** densities. The two differ by the
local Boltzmann factor `exp(±Δφ/V_T)`. Because that factor scales with
band bending — which scales with `V_bi` (hence `N_D_ETL`) and with `V_app`
— a *constant* attenuation (the v1 `calibration_factor=1e-4`, or the v2
`σ=1e-19` substitute) cannot track it. Frozen suppression ⇒ V_oc cannot
respond to ETL doping ⇒ the **Nd_ETL under-sensitivity** (30 % closure).

## Why 7 prior prototypes failed and this one does not

`failed-prototype/e2-bbd-face-density` (commit 4162862) projected the
densities exactly as here:

```
n_eval = n[idx+1]·exp((φ[idx]−φ[idx+1])/V_T)
p_eval = p[idx-1]·exp(−(φ[idx]−φ[idx-1])/V_T)
R = interface_recombination(n_eval, p_eval, ni_sq_eff, …)   # ← BUG
```

but left `ni_sq_eff` **un-projected**. At dark equilibrium
`n_eval·p_eval = (nR_eq·pL_eq)·F ≠ ni_sq_eff`, so the SRH numerator
`(n·p − ni²)` is non-zero at equilibrium → spurious recombination →
Newton blow-up at V_app ≈ 0.08 V. That single missing detailed-balance
term is what sank the whole architectural-refactor narrative.

**Fix:** co-project `ni_sq_eff` by the *same* combined factor
`F = fac_n·fac_p = exp((φ[idx-1]−φ[idx+1])/V_T)` (the `φ[idx]` term
cancels). Then the numerator is exactly `F·(n·p − ni²)` → identically zero
at equilibrium for all bias. No new DOF, no DAE block, no flux
restructure — a per-RHS algebraic correction, categorically different from
the 7 state-adding prototypes (all archived `failed-prototype/*`).

Implementation: `solver/mol.py:_apply_interface_recombination`, gated on
`SOLARLAB_IFACE_PROJ=1`. No-op when a config has no interface defects
(eval nodes collapse to `idx`, `F=1`). Off-path bit-identical — 17/17
SCAPS-subset integration tests green.

## Measured impact (scaps_mirror_v2.yaml, run_scaps_v2_regression.py)

| Sweep | OFF (parked) | ON (E8) | Verdict |
|---|---|---|---|
| CHI_ETL (CBO) | 83.0 % | **92.2 %** | +9 pp |
| **Nd_ETL** (ETL doping) | 29.8 % | **54.4 %** | **+25 pp — range nearly doubled; first lever in E1–E7 to move it** |
| Nt_PVK ETL (interface) | 108.6 % | 106.3 % | −2 pp, still passing |
| Nt_C_PVK (PVK bulk) | 0.2 % | 0.1 % | unchanged — still masked |
| Base J-V V_oc | 1.0808 V | 1.0590 V | −22 mV (J_sc 333→272 A/m², *closer* to SCAPS 263) |

Bracket counts identical across all sweeps (14/14, 8/11, 6/7, 7/7) →
Newton-stable, no crashes, the E2-BBD V=0.08 failure is gone.

### Trend-shape notes

- **Nd_ETL**: the high-doping arm (1e16→1e20) now rises steeper, matching
  SCAPS's monotonic increase. The low-doping arm (1e13→1e16) still dips —
  a *separate* contact/V_bi root cause at low ETL doping (the same region
  that fails to bracket below 1e13), not an interface-SRH problem.
- **Nt_C_PVK**: projection *lowers* the interface SRH ceiling
  (1.072→1.052 V), so bulk N_t stays masked. Unmasking needs the ceiling
  *raised*, which is in direct tension with the CBO/Nd_ETL gains (one
  interface-SRH model drives all of them). Confirms the E7 conclusion that
  Nt_C_PVK is genuinely blocked on SCAPS's Auger/radiative/interface
  formulation (partner data), not an in-tree lever.

## Reproduce

```bash
cd perovskite-sim
python scripts/run_scaps_v2_regression.py --out-dir ../outputs/scaps_proj_baseline       # OFF
SOLARLAB_IFACE_PROJ=1 python scripts/run_scaps_v2_regression.py --out-dir ../outputs/scaps_proj_on  # ON
```

## Open integration decision

The hook is env-gated for safety. Promoting it (SimulationMode flag /
config opt-in / default-on for heterojunction stacks) is a separate
decision because it shifts the SCAPS-mirror base V_oc (1.08→1.06) and
J_sc (333→272), which would require updating the
`test_scaps_mirror_baseline.py` envelope guards and the partner-facing
parity numbers.

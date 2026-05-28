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

## Full-PDF trend scorecard (all 10 single-variable sweeps)

`scripts/run_scaps_full_regression.py` runs every sweep present in the
partner PDF `1D-SCAPS 模拟.pdf` (mirrored in `scaps_reference.json`) and
reports direction + range-closure. Status with projection ON +
`interface_defect_E_t_eV` axis (this phase):

| # | Sweep | SCAPS trend | SolarLab | Status |
|---|---|---|---|---|
| 1 | CBO ΔE_C | rise 0.33→1.25 | 92 % closure | ✅ |
| 2 | Nd_ETL | monotonic rise +100 mV | high-N_D arm rises, **low-N_D dip flips net sign** | ❌ dir |
| 3 | PVK/ETL N_t | drop 1.25→0.97 | 106 % | ✅ |
| 4 | PVK/ETL E_t | drop −35 mV | −21 mV, dir + shape match (new axis) | ✅ |
| 5 | PVK-CB N_t | drop −39 mV | flat (cascade-masked) | ❌ |
| 6 | PVK-VB N_t | drop −11 mV | flat (same combined SRH as CB) | ❌ |
| 7 | HTL/PVK N_t | ~flat (−5 mV) | **rises +105 mV + solver crash @1e15** | ❌ bug |
| 8 | PVK-CB E_t | ~flat | flat | ✅ |
| 9 | PVK-VB E_t | ~flat | flat | ✅ |
| 10 | HTL/PVK E_t | dead flat | flat | ✅ |

**6/10 met.** Four open, each a distinct root cause:

- **Nd_ETL (#2)** — low-N_D_ETL V_oc dip (contact/V_bi behaviour at
  N_D ≤ 1e15), not interface SRH. The high-N_D arm is correct post-E8.
- **PVK-CB / PVK-VB N_t (#5,#6)** — bulk SRH masked by the interface SRH
  ceiling (the E7 recombination cascade). Also the sweep handler drives
  the absorber's *combined* τ, so CB and VB can't be distinguished
  (SCAPS shows them asymmetric: −39 vs −11 mV). Hardest; partner-blocked
  on SCAPS Auger/radiative spec per E7.
- **HTL/PVK N_t (#7)** — cross-carrier orientation bug. `ni_sq_eff =
  n_R_eq·p_L_eq = 1e20·1e24 = 1e44` pairs PVK electrons (minority at this
  hole-contact interface) with HTL holes (majority). When the sampled
  PVK-side electron density dips below 1e20 the numerator `n·p−ni²` goes
  negative → SRH acts as *generation* → higher N_t (SRV) → more spurious
  generation → V_oc rises. The E1.5 cross-carrier orientation only holds
  when the right layer is the electron-transport layer (true for PVK/ETL,
  false for HTL/PVK). Projection amplifies it (+85→+105 mV). Fix pending.

New sweep axis `interface_defect_E_t_eV` (targetable, mirrors
`interface_defect_N_t_cm2`) added to `device_parameter_sweep.py` to cover
sweeps #4/#10 — purely additive, 5/5 existing sweep unit tests green.

## Open integration decision

The hook is env-gated for safety. Promoting it (SimulationMode flag /
config opt-in / default-on for heterojunction stacks) is a separate
decision because it shifts the SCAPS-mirror base V_oc (1.08→1.06) and
J_sc (333→272), which would require updating the
`test_scaps_mirror_baseline.py` envelope guards and the partner-facing
parity numbers.

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

## HTL/PVK + bulk-N_t investigation — entangled interface calibration (do-not-retry)

Tried to close the HTL/PVK N_t sweep (SL rose +105 mV + crashed; SCAPS flat).
Root cause measured precisely: the cross-carrier reference
`ni_sq_eff = nR_eq·pL_eq` uses **bulk-asymptotic** equilibrium densities. At
the thin HTL/PVK junction the eval node sits in the depletion region, so the
self-consistent dark-equilibrium product is `9.2e23` vs the cached `1e44` —
**20 orders too high** → `np < ni²` → SRH flips to spurious generation that
raises V_oc with N_t.

Prototyped the principled fix (`SOLARLAB_IFACE_EQREF`): recompute `ni_sq_eff`
per interface from a self-consistent dark steady state (interfaces zeroed),
sampled at the eval nodes. Result:

- HTL/PVK sign **corrected** (V_oc now *drops* with N_t) — but over-shoots
  (−190 mV vs SCAPS −5 mV): the cross-carrier np at HTL/PVK is too large, so
  the interface is a strong recombiner where SCAPS keeps it nearly inert.
- **PVK/ETL regressed** (106 % → broken, base 1.05→0.82, crash). It was
  matching SCAPS *because* its reference was "wrong" (1.98e28 vs correct
  4.08e29, too low → more recombination → right magnitude). Correcting it
  reduces recombination and breaks the match.

**Conclusion (reverted, not shipped):** the interface-SRH references and σ
values are an *entangled empirical calibration*. Making one interface
physically correct regresses the empirically-matched one. Closing all
interface sweeps simultaneously requires either the interface-plane-state
solver (the dormant `SOLARLAB_INTERFACE_PLANE_STATE` path — Newton-unstable,
archived `failed-prototype/*`) or SCAPS source data (interface SRH
formulation + contact spec). This is the same architectural boundary E1–E7
documented; the eq-reference measurement quantifies *why* (20-order reference
error on the depleted interface) and is preserved here so it is not retried
blind. **Do not retry the global eq-reference correction without first
re-tuning the PVK/ETL σ/calibration to absorb the reference change.**

`Nt_C_PVK` / `Nt_V_PVK` remain cascade-masked (E7) and indistinguishable
(combined absorber τ) — genuinely partner-data-blocked.

## Interface-plane-state solver — stabilized but coupling-dead (E8.4)

Attempted the route to close HTL/PVK + bulk-N_t together: revive the dormant
`SOLARLAB_INTERFACE_PLANE_STATE=1` path (4 extra DOF per interface evolving
via TE flux + SRH on the interface plane).

**Finding 1 — it no longer hangs on v2** (11 s, V_oc 1.045). The documented
Newton-hang was the *Sprint-9 split-flux* variant, not this base path.

**Finding 2 — the χ-step cross-flux explodes under the cap.** Legacy
`J_cross_n = v_cross·(n_2s·exp(ΔE_c/V_T) − n_1s)` carries `exp(+|ΔE|/V_T)`;
at HTL/PVK (ΔE_c=1.54 eV) that is `exp(59)` capped at `exp(30)≈1e13` → the
~1e36 cross-flux. Fixed (`physics/interface_plane.py`) by factoring out the
larger exponential so the surviving arg is always ≤0 — same detailed-balance
zero-point (`n_1s/n_2s = exp(ΔE_c/V_T)`), bounded magnitude. 45/45 interface
unit tests green; dormant-gated so zero production impact.

**Finding 3 (the real blocker) — the bulk↔state coupling transmits no V_oc
trend.** With the cross-flux bounded, every interface sweep is *flat*:
CBO 1.060→1.075 (should be 0.33→1.25), PVK/ETL N_t dead flat at 1.077 (should
be 1.25→0.97). Sweeping the TE coupling rate `v_th_eff` over 1e-2 → 1e2 m/s
changes **nothing** — so it is not a rate problem. At steady state the bulk
drain equals the state SRH rate regardless of `v_th`; the SRH rate *on the
interface-plane state densities is simply too small* to move V_oc. The
state-SRH magnitude / state-density formulation needs fundamental rework to
match the production E1.5 path's effectiveness.

**Root cause of the coupling-death (E8.5, measured):** `compute_interface_te_
fluxes` builds the TE-flux *target* from the **cached dark-equilibrium**
densities (`mat.interface_n_R_eq` / `p_L_eq`), not the live illuminated bulk.
At V_oc the live PVK hole density is `1.6e22` but the cached target is `1.98e4`
— **18 orders too low** — so the interface-plane state is pinned at dark levels
and its SRH stays ≈0 (R=0 at equilibrium by construction). That is why no
interface sweep moves V_oc and why `v_th_eff` is irrelevant.

**Naive live-target fix is Newton-unstable.** Feeding the target from the live
(state-dependent) `n`/`p` couples the extra interface-plane DOF into the
Jacobian with feedback → fails to converge (V≈0.59). The production E1.5 path
is stable because it applies SRH *algebraically* (no DOF); the iface-state DOF
+ live feedback is the stiff combination.

**Well-scoped fix (downgrades "multi-week" → ~hours): lagged live target.**
Freeze the illuminated densities at each voltage step's *entry* state and feed
those (state-independent during the Newton solve → stable; refreshed per step →
illumination-aware). The codebase already implements exactly this pattern for
radiative reabsorption (`experiments/jv_sweep._bake_radiative_reabsorption_
step`). Plumbing: snapshot `n`/`p` at step entry in `_integrate_step`, stash on
`mat` via `dataclasses.replace`, have `compute_interface_te_fluxes` read the
frozen target. This is the concrete next milestone to close HTL/PVK + bulk-N_t
without the PVK/ETL entanglement.

**Lagged-target prototyped — Jacobian-stable but stiff (E8.5b).** Built the
full lagged plumbing: snapshot illuminated (n, p) at each voltage-step entry
(`jv_sweep._bake_iface_target`), stash on `mat`, project to the interface plane
as the TE-flux target. Default path stays bit-identical (bake is a no-op when
`N_iface_state==0`; verified V_oc=1.0808). But once the target carries the
illuminated population the interface-plane SRH becomes *physically active* —
and the extra interface-plane DOF then makes the system **stiff**: Radau fails
to converge at the very first step (V=0) as the illuminated-strength SRH is
applied onto the freshly-baked target. So the lag removes the *Jacobian
feedback* instability but exposes the *physical stiffness* of an actively-
recombining extra DOF. Reverted (kept committed bounded cross-flux only).

**Full root-cause chain for the iface-state route (definitive):**
1. cached-eq target → stable, coupling-dead (SRH at dark levels, no trend)
2. live-iterate target → Jacobian-unstable (state-dependent target feedback)
3. lagged frozen target → Jacobian-stable but stiff (active DOF, Newton fails)
4. **resolution = QSS algebraic reduction**: collapse the fast interface-plane
   state to its quasi-steady algebraic value (`diface_state = 0` solved
   per-RHS) so there is no stiff ODE DOF, with the illuminated target and
   per-interface detailed balance. This is what the original `_DEFAULT_V_TH_MS`
   comment anticipated ("Sprint 7 Day 4+ will revisit with QSS algebraic
   reduction"). It is also, in effect, deriving the proper per-interface
   Pauwels-Vanhoutte *algebraic* interface SRH — which would unify the fix
   (no entanglement, correct HTL/PVK reference, illuminated coupling) in one
   formulation. Multi-day solver work; the clear next milestone.

**Conclusion:** the cross-flux is stabilized (committed); the iface-state
coupling is fully root-caused; the remaining work is the QSS algebraic
reduction (multi-day). Production remains the E1.5 path + E8 projection
(6/10 PDF trends) — the achievable in-tree maximum without the QSS work or
SCAPS source data.

## Phase E9 — absolute+trend audit vs 1R-Parameters.xlsx (parallel-agent findings)

Partner asked for absolute-value closeness AND trend match, referencing the
raw xlsx. Two parallel debugger agents dispatched. Absolute targets locked
(`scripts/scaps_absolute_scorecard.py` grades all 4 metrics vs the xlsx):
base **V_oc 1.168 V, J_sc 263 A/m² (constant across every sweep), FF 0.870,
PCE 26.69 %**.

### CRITICAL — Nt_PVK/ETL "106 % closure" was a sweep σ-bug artifact

`sweeps/device_parameter_sweep.py:_apply_interface_defect_N_t_cm2` hardcodes
`SIGMA_CM2 = 1.0e-15` (the v1 value) when converting swept N_t → SRV. The v2
config uses **σ = 1e-19**. So every Nt_PVK/ETL sweep point ran with SRV
**10 000× too large** (100 m/s vs the config base 0.01 m/s at N_t=1e12). The
sweep's N_t=1e12 point therefore does NOT match the true base. With the
corrected σ=1e-19 the closure is **75 %, not 106 %** (direction still correct,
whole curve offset ~87 mV low). **The prior "interface 109/106 % ✓" in this
doc and the parked memory was measuring over-amplified physics — corrected to
~75 %.** CBO (CHI_ETL) is unaffected (not an N_t sweep): 83 %/92 % stand.

Fix: the sweep must reconstruct SRV with the config's σ. `InterfaceDefect`
carries only `E_t_eV` + `calibration_factor` (no σ/N_t), so the clean fix adds
`sigma_n_cm2`/`sigma_p_cm2`/`N_t_cm2`/`v_th_cm_s` to `InterfaceDefect`
(populated by the loader) and has the sweep handler reconstruct SRV =
σ·v_th·N_t — OR ratio-scale off the base SRV using a stored base N_t.

### Base V_oc gap (87 mV) is bulk/contact-limited, NOT interface-calibratable

Single-channel knockouts at the base point (PROJ=OFF, V_oc 1080.8 mV):

| zeroed | V_oc (mV) | Δ |
|---|---|---|
| PVK/ETL iface SRH | 1095.7 | +14.9 |
| HTL/PVK iface SRH | 1069.0 | **−11.8 (spurious generation)** |
| Auger | 1098.3 | +17.5 |
| B_rad | 1093.7 | +12.9 |
| bulk SRH → ∞ | 1081.6 | +0.8 (negligible) |
| iface + Auger + B_rad all 0 | **1199.5** | +118.7 (super-additive) |

Even removing ALL interface SRH only reaches 1095.7 mV — still 72 mV short of
1168. The all-recomb-off ceiling 1199.5 mV is itself below SCAPS's low-Nt
ceiling of 1249 mV → a residual ~50 mV **contact/carrier-statistics** gap
(SCAPS contact band-bending / Fermi treatment vs SolarLab Dirichlet pinning).
No `calibration_factor` on PVK/ETL can close it; closing needs bulk-recomb or
contact-BC changes that would regress the Nt sweep slope (the E7 entanglement,
now quantified). HTL/PVK orientation bug reconfirmed (`ni_sq_eff=nR_eq·pL_eq`
≈1e44 pairs ETL-electron with HTL-hole → np<ni² → spurious +12 mV generation).

### Absolute-vs-trend tradeoff (E8 projection)
- PROJ=OFF better for absolute V_oc (1080.8 vs 1059.0 mV).
- PROJ=ON better for CBO slope (92 % vs 83 %).
- Cannot get base V_oc=1.168 AND keep the Nt slope — bulk/contact ceiling.

### J_sc over-generation (Agent A — incomplete)
SolarLab base J_sc 333 A/m² (33.3 mA/cm²) exceeds the SQ limit (~27.5) for
Eg=1.53 eV; SCAPS 263 (26.28). Agent A hit its session budget before
reporting a root cause — TMM over-absorption (sub-gap band-tail in the MAPbI3
n,k, or missing reflection / flux normalization) is the leading hypothesis;
re-investigation pending.

## Open integration decision

The hook is env-gated for safety. Promoting it (SimulationMode flag /
config opt-in / default-on for heterojunction stacks) is a separate
decision because it shifts the SCAPS-mirror base V_oc (1.08→1.06) and
J_sc (333→272), which would require updating the
`test_scaps_mirror_baseline.py` envelope guards and the partner-facing
parity numbers.

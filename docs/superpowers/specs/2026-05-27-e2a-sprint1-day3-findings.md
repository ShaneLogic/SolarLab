# Phase E2a Sprint 1 Day 3 — Band-bending depletion probe findings

**Status:** probe results (no code change to perovskite_sim/)
**Branch:** `e2a-scaps-source-audit`
**Inputs:** `outputs/e2a_probe_bbd.txt`

## Hypothesis under test (Day 1 audit)

SCAPS samples interface-plane n, p with a **band-bending depletion**
factor `exp(-Δφ/V_T)` applied within the bulk side, NOT bulk-interior
densities (SolarLab E1.5) and NOT bulk-density × χ-step factor (E1.6 v2
Boltzmann-from-Fermi). Candidate (d):

```
n_face = n[idx+1] · exp((φ[idx] − φ[idx+1])/V_T)   # ETL side, electron majority
p_face = p[idx-1] · exp(-(φ[idx] − φ[idx-1])/V_T)  # PVK side, hole minority but positive
```

Only same-layer φ band-bending; no χ step crossing ⇒ photo-injection
safe (does not amplify Q-Fermi splitting through χ).

## Probe results (V<sub>oc</sub> baseline + ΔE<sub>C</sub> ∈ {−0.5, +0.3} V)

| Probe point | Δφ(M−R) | Δφ(M−L) | E1.5 np | BBD np | BBD/E1.5 |
|---|---|---|---|---|---|
| V<sub>oc</sub> baseline (ΔE<sub>C</sub>=0)    | +0.5 mV   | −19.8 mV  | 7.4e46 | 1.6e47 | 2.2× |
| Cliff (ΔE<sub>C</sub>=−0.5)         | +1.3 mV   | −51.2 mV  | 3.0e47 | 2.2e48 | 7.6× |
| Spike (ΔE<sub>C</sub>=+0.3)         | −5.4 mV   | +125.3 mV | 7.6e41 | 4.8e39 | 0.006× |

**Key observations:**

1. BBD produces **physically sensible positive values** at all three
   probe points. No machine-epsilon collapse (vs SG-Selberherr at
   1e−5 m⁻³), no exp(ΔE<sub>V</sub>/V<sub>T</sub>) blow-up (vs Boltzmann-from-Fermi at
   1e+35).

2. **Cliff direction is preserved.** Cliff side (ΔE<sub>C</sub>=−0.5) yields
   BBD np = 2.2e48 — 14× rise from baseline. E1.5 only rises 4×.
   More sensitivity to cliff direction means a stronger V<sub>oc</sub> cliff —
   which is what SCAPS shows.

3. **Spike side under-shoots BBD.** At ΔE<sub>C</sub>=+0.3 the spike Δφ
   reverses (Δφ_M−L = +125 mV positive), giving p_face = p_L ·
   exp(−4.8) = p_L · 8e−3. The np product collapses to 4.8e39 — 5
   orders below baseline. E1.5 only drops 5×. This is the right
   direction for a V<sub>oc</sub> spike (less recombination at the favourable
   band offset) but the magnitude may overshoot.

## Caveats

- **Negative p densities at ETL interior** (idx_R) persist from
  Phase A1 — known artifact of state-vector reconstruction in
  heavily-doped ETL where ion residual gives tiny negative p. BBD
  uses p[idx_L] (PVK side, positive) so this artifact does not
  pollute the candidate. SG-Selberherr DID use p[idx_R] and
  collapsed for this reason.

- **Probe does NOT test N_D_ETL sensitivity yet.** Day 1 hypothesis
  says BBD should recover SCAPS' 8× ETL doping sensitivity (the
  remaining gap on the ETL doping sweep). Need next probe extension:
  parametrize N_D_ETL ∈ {1e16, 1e17, 1e18} cm⁻³ and check
  d(BBD np) / d(log N_D_ETL).

- **Three-layer scaps_mirror only.** HTL/PVK interface untested. If
  partner adds the SCAPS PDF's HTL/PVK Gaussian defect, the candidate
  must symmetrise correctly across both interfaces.

## Decision matrix

| Option | Probe status | Recommendation |
|---|---|---|
| Keep E1.5 cross-carrier | OK but blocks 3 sweeps | rejected |
| SG-Selberherr | REJECTED (Phase A1 + A2) | rejected |
| Boltzmann-from-Fermi (E1.6 v2) | REVERTED | rejected |
| **BBD band-bending depletion** | Physically sensible; direction matches SCAPS | **proceed to Phase E2 design RFC** |
| Thin-shell volumetric SRH | Untested probe-wise | fallback if BBD fails Day 4 N_D_ETL sweep |

## Next actions

- **Day 3.5:** extend probe with N_D_ETL sweep ∈ {1e16, 1e17, 1e18} cm⁻³.
  Check d log(np_BBD) / d log(N_D_ETL) vs SCAPS PDF sensitivity 8×.
- **Day 4-5:** if N_D_ETL sweep validates BBD, write Phase E2 design RFC
  proposing `_apply_interface_recombination` switch from cross-carrier
  to BBD-projected face densities. Estimate ~50-100 LoC + 5-8 RED tests.
  If N_D_ETL sweep fails, fall back to thin-shell volumetric SRH option.

## Convention

- All work in `outputs/` and `docs/superpowers/specs/` only. No
  `perovskite_sim/` touch until Phase E2 design RFC approved.

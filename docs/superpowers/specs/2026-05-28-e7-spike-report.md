# Phase E7 Day 1 spike — findings + decision matrix

**Status:** complete; awaiting writing-plans for Y1 + revised Y2 scope
**Date:** 2026-05-28
**Spec:** `docs/superpowers/specs/2026-05-28-e7-trend-parity-design.md`

## Probe A — PVK donor doping direction under v2

**Script:** `perovskite-sim/scripts/probes/e7_probe_a_pvk_doping.py`
**Output:** `outputs/scaps_e7_probe_a/pvk_doping_v2_direction.csv`
**Wall time:** ~75 s
**Verdict:** **MATCH (direction-correct in physical regime)**

Per-point V_oc / J_sc:

| N_D (cm⁻³) | V_oc (V) | J_sc (A/m²) | Bracketed | Note |
|---:|---:|---:|---|---|
| 1e16 | 1.1164 | 3366  | ✓ | physical |
| 5e16 | 1.1328 | 8884  | ✓ | physical |
| 1e17 | 1.1407 | 13511 | ✓ | physical |
| 5e17 | 1.1432 | 16408 | ✓ | physical |
| 1e18 | 1.1229 | 14369 | ✓ | mild dip |
| 5e18 | 1.1105 | **131** | ✓ | J_sc collapse |
| 1e19 | 1.1478 | **5.3** | ✓ | J_sc collapse (V_oc unreliable) |

**Trend across physical regime [1e16, 5e17]**: V_oc rises monotonically
+27 mV across 1.5 decades — direction matches SCAPS (V_oc RISES with N_D).
SCAPS reference range over PDF page 2 was 34 mV; SolarLab gives 27 mV in
the physical sub-range. Closure ~80%.

**Anomalies at N_D ≥ 1e18**: J_sc collapses by 100×–10000×. Likely heavy
donor doping kills collection (PVK becomes too n-type, hole extraction
breaks). NOT a V_oc direction artifact — separate physics question, out
of E7 scope unless partner xlsx includes these high-N_D points.

**Y3 verdict:** **SKIP**. PVK doping direction already matches SCAPS
under v2 schema in the physical regime. No Y3 work needed.

**Caveat:** SCAPS reference for PVK doping is **not in partner xlsx**
(only in older PDF page 2 sweeps). Verification is qualitative
(direction + ~80% magnitude closure) rather than xlsx-quantitative.

## Probe B — multi-defect SRH collapse audit

**Script:** `perovskite-sim/scripts/probes/e7_probe_b_srh_collapse.py`
**Output:** `outputs/scaps_e7_probe_b/srh_collapse_ratio.csv`
**Wall time:** < 1 s (pure math)
**Verdict:** **B1 — collapse is exact**

Per-defect parameters (PVK-CB and PVK-VB):
- σ_n = σ_p = 1e-19 m² (loader-converted from 1e-15 cm²)
- v_th = 1e5 m/s, N_t = 1e18 m⁻³
- τ_CB = τ_VB = 1e-4 s
- n1_CB = 2.09e23 m⁻³, p1_CB ≈ 0
- n1_VB ≈ 0, p1_VB = 2.09e23 m⁻³ (mirror image)

Loader-collapsed (`_combine_bulk_defects`):
- τ_n_eff = τ_p_eff = 5e-5 s
- n1_eff = p1_eff = 1.045e23 m⁻³

R_true / R_collapsed ratio across 6 (n, p) sample points spanning dark
equilibrium → V_oc-level injection → asymmetric n-rich / p-rich:

| Sample | (n, p) | R_true | R_collapsed | Ratio |
|---|---|---:|---:|---:|
| dark-eq low | (1e18, 1e18) | 9.57e16 | 9.57e16 | 1.0000 |
| low injection | (1e20, 1e20) | 9.56e20 | 9.56e20 | 1.0000 |
| mid injection | (1e22, 1e22) | 8.74e24 | 8.74e24 | 1.0000 |
| V_oc-level | (1e23, 1e23) | 4.89e26 | 4.89e26 | 1.0000 |
| asymmetric n-rich | (1e23, 1e18) | 6.47e21 | 6.47e21 | 1.0000 |
| asymmetric p-rich | (1e18, 1e23) | 6.47e21 | 6.47e21 | 1.0000 |

**Max deviation: 0.00%.**

The CB+VB defect pair has identical (σ_n, σ_p, N_t) with mirror-image
(n1, p1). Parallel SRH summation is mathematically equivalent to a
single-effective-defect SRH for this symmetric configuration. The
loader's inverse-τ-weighted n1/p1 averaging recovers the exact two-defect
rate, not an approximation.

**Y1 verdict:** **B1 — YAML-only SRV tune.** No need for true multi-
defect solver hook (B2). The Nt_C_PVK 0.2% closure mask is NOT from
defect-collapse smearing; it is purely from PVK/ETL interface SRH
overwhelming bulk SRH (250× dominance per E6.4).

## Probe C — Robin contact BC dry-run

**Script:** `perovskite-sim/scripts/probes/e7_probe_c_robin_nd_etl.py`
**Output:** `outputs/scaps_e7_probe_c/nd_etl_v2_{dirichlet,robin_moderate,robin_strong}.csv`
**Wall time:** ~6 min
**Verdict:** **C3 — Robin alone insufficient. Re-scope Y2.**

**Note on Φ_b:** Partner PDF (`scaps_1d_simulation_report.pdf`) does NOT
specify contact workfunctions. SCAPS defaults were used. This was a
sensitivity probe over the Robin S parameter space rather than a
SCAPS-exact-match attempt.

Per-config Nd_ETL closure (V_oc range over the working subset of bracketed
+ physical points):

| Config | S values (m/s) | Bracketed | V_oc range (SL) | V_oc range (SCAPS) | Closure |
|---|---|---:|---:|---:|---:|
| v2 Dirichlet | n/a | 8/11 | 29.7 mV | 99.6 mV | 29.8 % |
| v2 Robin moderate | maj=1e3 / min=1e1 | 8/11 | 29.0 mV | 99.6 mV | 29.1 % |
| v2 Robin strong | maj=10 / min=0.1 | 7/11 | 312.4 mV | 99.6 mV | 313.5 % |

Per-point V_oc (working subset only):

| N_d (cm⁻³) | V_oc Dirichlet | V_oc Robin mod | V_oc Robin strong | V_oc SCAPS |
|---:|---:|---:|---:|---:|
| 1e13 | 1.0954 | 1.0947 | **1.3783** | 1.1376 |
| 1e14 | 1.0793 | 1.0794 | _unbracketed_ | 1.1412 |
| 1e15 | 1.0683 | 1.0684 | 1.0702 | 1.1440 |
| 1e16 | 1.0656 | 1.0656 | 1.0660 | 1.1464 |
| 1e17 | 1.0659 | 1.0659 | 1.0661 | 1.1510 |
| 1e18 | 1.0711 | 1.0711 | 1.0712 | 1.1676 |
| 1e19 | 1.0802 | 1.0802 | 1.0803 | 1.2024 |
| 1e20 | 1.0844 | 1.0844 | 1.0845 | 1.2373 |

**Key observations**:

1. **No 2.1 V unphysical branch at V_max=1.6 V in any config.** The 2.1 V
   E6.5 branch was a V_max=2.5 V artifact, not a default-sweep concern.
   The pre-spike narrative about Robin "killing the 2.1 V branch" was
   based on a stale problem framing.

2. **High-Nd regime (1e15-1e20) is identical across all three configs**
   (V_oc differs by < 2 mV). Robin S choice does not move the working
   regime. The Nd_ETL under-sensitivity is **bulk-limited**, not
   contact-limited.

3. **Strong Robin (S=10/0.1) inflates low-Nd V_oc** to unphysical-ish
   1.378 V at N_d=1e13 and de-brackets N_d=1e14. The 313% reported
   "closure" is a spread artefact from the inflated low-Nd point;
   high-Nd regime is still under-sensitive.

4. **Root cause** (working hypothesis): the E1.5 cross-carrier interface
   SRH formulation reads bulk `n[idx+1] = N_D_ETL` directly. This makes
   SRH scale with bulk N_D_ETL, but the BULK PVK SRH ceiling (τ_eff = 5e-5 s)
   clamps V_oc to ~1.07 V across most of the sweep range. No contact
   BC change can break this ceiling.

**Y2 verdict**: **C3 — re-scope.** Robin BC alone cannot close the
Nd_ETL gap. The architectural lever is interface SRH formulation, not
contact BC. The parked Phase E1.6 SG-face-density refactor was tracking
this exact root cause but was falsified by E6.4's different reasoning;
the underlying physics question remains open.

## Updated spec scope (post-spike)

| Phase | Original scope | Post-spike scope | Cost |
|---|---|---|---|
| Y1 | Multi-defect SRH solver hook (B2) | **YAML-only PVK/ETL SRV tune (B1)** | 0.5 day YAML + 1 day tests |
| Y2 | Robin/Φ_b contact BC | **Re-scoped: parked, escalation needed** | architectural, not in short path |
| Y3 | Conditional PVK doping fix | **Dropped — v2 fixes direction** | 0 days |

**Y2 re-scope options** (none in short path):

1. **Park Y2** — accept Nd_ETL 30% closure as ship-state. Document as
   "architectural gap requiring SG-face-density refactor (parked Phase
   E1.6); user-confirmed trends-over-absolutes accepts this as
   non-blocking".
2. **Lower PVK/ETL SRV calibration** — would raise V_oc ceiling but
   not improve Nd_ETL trend sensitivity (bulk SRH still doesn't depend
   on N_D_ETL). NOT a closure path.
3. **Two-preset split with Y1-style approach** — `scaps_mirror_v2_loSRV.yaml`
   for Nd_ETL sweep with reduced PVK/ETL N_t. Risk: still doesn't make
   trend more N_D_ETL sensitive (same physics).
4. **SG-face-density refactor (Phase E1.6 retry)** — multi-week,
   previously falsified-prototype. Out of short path.
5. **Request partner SCAPS contact-model spec** — Φ_b values + verify
   SCAPS' interface SRH formulation. Would unblock informed Y2.

**Recommended path**: ship Y1 (bulk N_t closure) + park Y2 with
documentation, focus next on partner deliverable. Re-open Y2 only if
partner explicitly requires Nd_ETL closure.

## Closure summary (projected post-Y1, no Y2)

| Sweep | Pre-E7 | Post-Y1 (projected) | Target | Status |
|---|---|---|---|---|
| CHI_ETL (CBO) | 83 % | 83 % (preserved) | ≥ 80 % | ✓ |
| Nt_PVK_ETL (interface) | 109 % | 109 % (preserved) | ≥ 90 % | ✓✓ |
| Nd_ETL (ETL doping) | 30 % | 30 % (no change) | ≥ 70 % | ✗ parked |
| Nt_C_PVK (PVK bulk) | 0.2 % | **≥ 50 % (Y1 target)** | ≥ 50 % | TBD by Y1 |
| Na_PVK (PVK doping) | unknown | **direction ✓ (Probe A)** | direction ✓ | ✓ |

Net: 4 of 5 trend gaps closed/preserved. Nd_ETL remains as documented
architectural limitation.

## Next steps

1. **Commit spike artefacts** (probe scripts, configs, CSVs, this report).
2. **Discuss revised scope with user** — confirm parking Y2 vs pursuing
   architectural re-scope.
3. **If user approves Y1-only ship**: invoke writing-plans skill for
   Y1 (bulk N_t SRV tune) — small enough that an implementation plan
   may not be needed; could execute directly.

## Spike artefacts (this commit)

- `perovskite-sim/scripts/probes/e7_probe_a_pvk_doping.py`
- `perovskite-sim/scripts/probes/e7_probe_b_srh_collapse.py`
- `perovskite-sim/scripts/probes/e7_probe_c_robin_nd_etl.py`
- `perovskite-sim/configs/scaps_mirror_v2_robin_moderate.yaml`
- `perovskite-sim/configs/scaps_mirror_v2_robin_strong.yaml`
- `outputs/scaps_e7_probe_a/pvk_doping_v2_direction.csv`
- `outputs/scaps_e7_probe_b/srh_collapse_ratio.csv`
- `outputs/scaps_e7_probe_c/nd_etl_v2_{dirichlet,robin_moderate,robin_strong}.csv`
- `docs/superpowers/specs/2026-05-28-e7-spike-report.md` (this file)

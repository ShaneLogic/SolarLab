# Scoping: matching SCAPS interface recombination (Pauwels-Vanhoutte) in SolarLab

**Date:** 2026-06-08  ·  **Status:** scope (path A) — no code changed
**Question:** what exactly does SolarLab need to change to reproduce SCAPS's interface
recombination, and is building it worth the effort vs documenting the difference?

---

## 1. What SCAPS does (authoritative)

**Method** — SCAPS manual §3.9 (p.45) + ref [13]:
- Interface recombination = **Pauwels-Vanhoutte theory** — a two-velocity extension of
  Shockley-Read-Hall for interface states. Ref: H.J. Pauwels & G. Vanhoutte, *J. Phys. D* (1978).
- Interface **transport** = thermionic emission; interface thermal velocity `v_th = min`
  of the two neighbouring layers' thermal velocities.
- Reported as a **two-sided current** `jn1/jn2` (electrons left/right) + `jp1/jp2` (holes
  left/right) → `jifr` (manual p.80-81).

**Rate (neutral single-level interface defect, no tunnelling — our config):**
```
R_if = (n_s·p_s − n_i²) / ( (n_s + n1)/S_p + (p_s + p1)/S_n )      [areal, cm⁻² s⁻¹]
S_n = σ_n · v_th · N_it ,  S_p = σ_p · v_th · N_it
n1  = N_C·exp(−(E_C−E_t)/kT) ,  p1 = N_V·exp(−(E_t−E_V)/kT)
```
**Crucially `n_s, p_s` are the carrier densities AT THE INTERFACE PLANE** — i.e. the
band-bending-suppressed values at the metallurgical junction, NOT the bulk densities a
node away.

**Parameters** (partner PDF `1D-SCAPS 模拟.pdf` p.1, already in `configs/scaps_mirror_v2.yaml`):
HTL/PVK & PVK/ETL interfaces — neutral, σ_n=σ_p=1e-19 cm², E_t=0.6 eV, N_t=1e12 cm⁻³
(HTL/PVK single level; PVK/ETL Gaussian, char-energy 0.1 eV, peak 5.64e8). Tunnelling off.

---

## 2. What SolarLab does now (production, E1.5)

`physics/recombination.py:44` — identical SRH surface form:
```
R_s = (n·p − ni_sq_eff) / ( (n + n1)/v_p + (p + p1)/v_n )          [m⁻² s⁻¹]
```
Wired in `solver/mol.py:_apply_interface_recombination` (entry ~L1159) with build-side
caches set in `build_material_arrays` (~L749-858):
- `v_n = σ_n·v_th·N_t`, `v_p = σ_p·v_th·N_t`  — **same surface-velocity def as SCAPS S_n/S_p ✓**
- `n1, p1` = E_t-aware via `srh_n1_p1_from_trap_depth(reference="below_cb")` — **✓**
- `n = n[idx+1]` (transport-side interior), `p = p[idx-1]` (absorber-side interior)
  — **BULK-INTERIOR sampling ✗ ← the divergence**
- `ni_sq_eff = n_R_eq · p_L_eq` (equilibrium cross-product, so the numerator vanishes at
  dark equilibrium) — detailed-balance-correct but calibration-entangled (see §5).
- `nogen` clamp (`R_s ≥ 0`, default on) suppresses spurious generation from the
  bulk-asymptotic `ni_sq_eff`.

---

## 3. The diff — ONE primary divergence

| term | SCAPS P-V | SolarLab E1.5 | match? |
|---|---|---|---|
| rate form | two-velocity surface SRH | identical | ✅ |
| S / v velocities | σ·v_th·N_t | σ·v_th·N_t | ✅ |
| n1 / p1 | E_t-aware | E_t-aware | ✅ |
| **density n_s, p_s** | **interface plane** | **bulk interior (idx±1)** | ❌ **THE gap** |
| ni² reference | n_i² (interface plane) | n_R_eq·p_L_eq (bulk) | ⚠ entangled |
| TE self-limiting | v_th-bounded | none (clamp only) | ⚠ secondary |

Bulk-interior vs interface-plane densities differ by the band-bending Boltzmann factor
`exp(±Δφ/V_T)`. Under forward bias / at V_oc the interface-plane densities are much lower,
so **SolarLab over-counts interface recombination → too much leak → low V_oc and the
wrong response to doping/CBO sweeps.** This single sampling-location error is the root of
the recombination-wall parity gaps (base V_oc, Nd_ETL magnitude, CBO plateau, Nt_C_PVK mask).

---

## 4. The minimal faithful change is ALREADY CODED

`solver/mol.py:1257-1271` — the **E8 Boltzmann projection** (`SOLARLAB_IFACE_PROJ=1`,
env-gated, **default OFF**) does exactly the missing step:
```
fac_n = exp((φ[idx] − φ[eval_n])/V_T);  fac_p = exp(−(φ[idx] − φ[eval_p])/V_T)
n_eval *= fac_n;  p_eval *= fac_p;  ni_sq_eff *= fac_n·fac_p   # co-projected → R=0 at eq
```
This projects the bulk-interior eval densities onto the interface plane — i.e. makes
SolarLab sample `n_s, p_s` like SCAPS. It is **Newton-stable and shipped** (off-path
bit-identical). Measured on scaps_mirror_v2 (project memory): **CBO 83→92 %, Nd_ETL 30→54 %**,
J_sc 333→272 (closer), base V_oc 1.08→1.06.

`SOLARLAB_IFACE_QSS=1` (`_qss_interface_R`, L1123) adds the thermionic-emission
self-limiting (`v_th·δ = SRH(proj−δ)`, R = v_th·δ) — the §3 "TE self-limiting" row. More
physical (no clamp) but lowers base V_oc further; E11.3 found it does NOT additionally
close the 3 gaps.

**So "build the P-V fix" is ~90 % done.** What remains is integration + the trade-off, not
new solver math.

---

## 5. What's NOT solved by flipping the flag (honest)

1. **Base V_oc absolute gets slightly WORSE.** Projection removes over-counted leak at the
   sampling level but lowers base V_oc ~20 mV (1.08→1.06), widening the absolute gap. Under
   the trends-over-absolutes bar this is acceptable (trends improve), but it is a real trade.
2. **Nt_C_PVK stays masked.** Bulk-trap response is gated by the recombination ceiling
   cascade (interface SRH > Auger > radiative > bulk SRH); projection lowers the interface
   ceiling but doesn't expose bulk SRH.
3. **Calibration entanglement.** `ni_sq_eff = n_R_eq·p_L_eq` is tuned so PVK/ETL matches;
   a global self-consistent eq-reference (the reverted `SOLARLAB_IFACE_EQREF`) fixed HTL/PVK
   but broke PVK/ETL. The interface references + σ are an entangled empirical calibration.
4. **N_t units check.** Partner PDF lists interface N_t as **1e12 cm⁻³** (volumetric), but
   SCAPS interface defects are natively **areal (cm⁻²)**. Verify the loader's σ·v_th·N_t maps
   to SCAPS's S_n with consistent units before trusting absolute magnitudes.

---

## 6. Recommendation

The fix is far smaller than "multi-day solver build" — the interface-plane projection that
matches SCAPS's P-V density sampling **already exists and is Newton-stable**. The real task
is a **decision + validation**, not new physics:

**Path B-lite (recommended if pursuing parity):**
1. Promote `SOLARLAB_IFACE_PROJ` from env-gate to a `SimulationMode` flag (default OFF →
   LEGACY/IonMonger bit-identical; opt-in for SCAPS-parity runs).
2. Tests FIRST: legacy off-path bit-identical; IonMonger-preset regression unchanged;
   golden-master (`scripts/qss_golden_master.py`) drift = 0 with flag off.
3. Re-baseline the SCAPS test-guards (`test_scaps_mirror_*`) to the projection-on numbers.
4. Resolve §5.4 (N_t units) and document the §5.1 base-V_oc trade + §5.2/5.3 limits.
~1-day task, mostly tests + a flag, low IonMonger risk (default off).

**Path C (document):** Write the parity report stating the identified cause precisely —
"SolarLab production samples bulk-interior densities; SCAPS Pauwels-Vanhoutte samples the
interface plane; the `SOLARLAB_IFACE_PROJ` projection closes most of the trend gap (CBO
83→92 %, Nd_ETL 30→54 %) at a ~20 mV base-V_oc cost; full Nt_C_PVK closure is
calibration-entangled." Ship under the trends-over-absolutes bar.

Either way the multi-day E11-QSS / interface-plane-state-solver work is **not** required for
the trend gains — that was the over-estimate. It remains the path only for the residual
Nt_C_PVK mask and the no-clamp physical purity, which E11.3 already showed don't move the
parity needle.

---

## 7. Key files
- `perovskite_sim/physics/recombination.py:28` — `interface_recombination` (the SRH form)
- `perovskite_sim/solver/mol.py:1159` — `_apply_interface_recombination` (E1.5 + E8 proj + E11 QSS)
- `perovskite_sim/solver/mol.py:749-858` — interface eval-node / n1,p1 / ni_sq_eff build
- `configs/scaps_mirror_v2.yaml` — partner params
- SCAPS manual `/tmp/scaps_manual.pdf` §3.9 p.45 (+ ref [13] p.159); partner `docs/partner/1D-SCAPS 模拟.pdf` p.1

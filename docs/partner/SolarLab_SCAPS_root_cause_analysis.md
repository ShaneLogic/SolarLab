# SolarLab vs. SCAPS-1D: A Drift-Diffusion Cross-Validation and Root-Cause Analysis of the Base-Model Discrepancy

## Abstract

We compare two one-dimensional drift-diffusion (DD) solar-cell simulators — **SolarLab** (the Python `perovskite_sim` library under development in this repository) and **SCAPS-1D** (Burgelman/University of Gent, the de-facto reference) — on a partner "Base Model" perovskite stack: glass / spiro-OMeTAD hole-transport layer (HTL, 20 nm) / MAPbI₃-like absorber (800 nm, E_g = 1.53 eV) / TiO₂ electron-transport layer (ETL, 25 nm). Both codes are drift-diffusion solvers; neither is "experiment," so we anchor absolute realism against the literature. At the base point, SolarLab reports V_oc = 1.072 V vs. SCAPS 1.1676 V (a −96 mV / −8% deficit), J_sc = 25.73 vs. 26.282 mA/cm² (−2%), FF = 85.6% vs. 86.99% (−1.4 pp), and PCE = 22.1% vs. 26.69% (−4.6 pp). Across ten device-parameter sweeps inherited from the SCAPS reference, only two levers close acceptably under a trend-first standard (conduction-band-offset 83% closure; PVK/ETL interface trap density 109%), and one shows an outright direction reversal (ETL donor doping). Through an adversarial three-lens review (physics-correctness, numerical-algorithm, data-consistency), we find that **the prior gap report's central claim — a 37× dark-saturation-current prefactor driven by quasi-Fermi-level dissipation across band offsets — is not supported by the conduction-band-offset sweep data**, which show the gap *widening* (not closing) as the offset is removed. The two robustly surviving root causes are: (i) the **frozen Poisson built-in-potential boundary** (`stack.V_bi = 1.30 V`, hard-wired, while the band-derived `V_bi_eff` is computed but routed only to the sweep range), which severs the ETL-doping → V_oc lever and produces the direction reversal (3/3 lenses held); and (ii) the **PCE/FF self-consistency anomaly**, which is a *reporting artifact* (cross-run cell-splicing of figures-of-merit, not a solver defect; classification held, mechanism corrected). The −96 mV V_oc deficit itself remains *physical but contested in mechanism*: candidate drivers (hard-Dirichlet ohmic contacts, thermionic-emission flux capping, and the default interface-recombination model differing from SCAPS's Pauwels–Vanhoutte formulation) each survived only one of three lenses. Critically, against experiment, **SolarLab's 1.072 V sits near the measured-device median (1.05–1.13 V) while SCAPS's 1.1676 V is at the published champion ceiling**; under the repository's trend-over-absolutes house rule, we recommend exactly one cheap correctness fix (re-extract all four figures-of-merit from a single self-consistent J–V array to eliminate the PCE/FF splice) and explicitly recommend *against* the multi-week contact-BC / V_bi refactors needed to chase absolute V_oc parity.

---

## 1. Methodology

### 1.1 The two solvers

Both SolarLab and SCAPS-1D solve the standard one-dimensional semiconductor drift-diffusion system: Poisson's equation coupled to electron and hole continuity equations, with Scharfetter–Gummel (SG) exponentially-fitted flux discretization [1, 5]. They differ in numerical strategy and several physical boundary treatments, summarized here and detailed in §3.

**SCAPS-1D** solves the DD set in the variables (ψ, E_Fn, E_Fp) — electrostatic potential and the two quasi-Fermi levels — using a **Gummel iteration with Newton–Raphson substeps** on a non-uniform finite-difference mesh with two coincident nodes at each metallurgical interface and adaptive refinement keyed to exp(qψ/kT), exp(E_Fn/kT), and the R/G ratios (SCAPS Manual, Feb. 2016, pp. 39–41) [1, 6]. Carrier statistics are **Boltzmann by default**; Fermi–Dirac degeneracy and band-gap narrowing are explicitly not implemented (Manual p. 31). Interface recombination uses **Pauwels–Vanhoutte theory** (an SRH extension where both adjacent-layer band edges enter the effective rate), and interface *transport* is by **thermionic emission** with a thermal velocity equal to the smaller of the two adjacent layers, which always leaves a finite quasi-Fermi-level step at the interface even at zero band offset (Manual p. 30) [8].

**SolarLab** solves Poisson plus the two continuity equations by a **method-of-lines (MOL) pseudo-transient** scheme: space is discretized with exponentially-fitted SG finite-volume fluxes; Poisson is a single prefactored tridiagonal solve per right-hand-side (RHS) call (no inner Newton loop); and time is integrated to steady state with `scipy.integrate.solve_ivp(Radau)` (`perovskite-sim/perovskite_sim/solver/mol.py:1577–1622, 1741`). Convergence control is entirely Radau's adaptive local-truncation-error machinery (rtol = 1e-4, atol = 1e-6), a `max_step` cap near flat-band, a `max_nfev` abort (`_JV_RADAU_MAX_NFEV = 100_000`), a BDF last-resort fallback after bisections, and an `isfinite` RHS guard. Carrier statistics are **Boltzmann only**; n_i is supplied per-layer (in the validated path derived from the density of states via n_i = √(N_c·N_v)·exp(−E_g/2kT), matching the SCAPS convention; `loader.py:252`).

### 1.2 The partner Base Model stack

| Layer | Material | Thickness | E_g (eV) | χ (eV) | Doping |
|---|---|---|---|---|---|
| HTL | spiro-OMeTAD | 20 nm | (wide-gap) | 2.40 | p-type |
| Absorber | MAPbI₃-like | 800 nm | 1.53 | 3.94 | intrinsic/lightly doped |
| ETL | TiO₂ | 25 nm | (wide-gap) | 4.10 | n-type donor |

The conduction-band offset at the PVK/ETL junction is ΔE_C = χ_PVK − χ_ETL = 3.94 − 4.10 = **−0.16 eV** (a "cliff"). Interface defects are **neutral**, statistics are **Boltzmann**, and **tunnelling is OFF** in the partner configuration (default; the intra-band and band-to-band tunnelling checkboxes are unset, so transport over the cliff is pure thermionic emission) [1, 2]. The SolarLab mirror configuration is `perovskite-sim/configs/scaps_mirror_v2.yaml` (χ values at lines 89/124; V_bi = 1.30 at line 40; `mode: fast` at line 42, which keeps thermionic-emission capping ON and selective contacts OFF).

### 1.3 Data sources

- **SCAPS reference figures-of-merit and ten sweeps:** `perovskite-sim/tests/integration/scaps_reference.json` (base_model: V_oc 1.1676 V, J_sc 26.282 mA/cm², FF 86.99%, PCE 26.69%; plus 10 parameter sweeps with per-point V_oc).
- **Partner SCAPS report and parameter table:** `docs/superpowers/references/scaps_1d_simulation_report.pdf`, `docs/superpowers/references/scaps_1r_parameters.xlsx`.
- **SCAPS formulation:** `docs/SCAPS Manual february 2016.pdf` (111 pp.).
- **SolarLab sweep results (V_oc per point, SolarLab vs. SCAPS):** `outputs/scaps_full_off/summary.json` (interface projection OFF — the default path), `outputs/scaps_full_on/summary.json` (band-aware projection ON), `outputs/scaps_full_ifacestate/summary.json` (true Pauwels–Vanhoutte interface-plane state variant).
- **SolarLab full-FOM self-consistent artifact:** `perovskite-sim/outputs/scaps_validation/report.md` and `sweep_*.csv`.
- **Prior analysis (tested, not assumed):** `docs/partner/SolarLab_SCAPS_gap_analysis.md`.

### 1.4 Evaluation standard: trend-first

Per the repository house rule, **parity is judged by trend fidelity (direction + shape of each sweep) first, and absolutes second**. We do not propose multi-week refactors to chase absolute V_oc/J_sc/FF/PCE parity. Closure is reported as a span ratio (SolarLab V_oc range ÷ SCAPS V_oc range over the sweep); we explicitly flag where this span-ratio metric masks a shape (curvature) mismatch (§4, Nt_PVK_ETL) or where the SolarLab sweep variable differs from the one in `report.md` (§4.1, CHI_ETL). Direction match is reported separately. Every quantitative claim below traces to one of the data sources in §1.3 or to a cited reference; literature values used only for realism-positioning are flagged as such.

---

## 2. Base-Model Comparison

### 2.1 Figures of merit

| Metric | SolarLab | SCAPS-1D | Δ (SolarLab − SCAPS) |
|---|---|---|---|
| V_oc (V) | 1.072 | 1.1676 | **−0.096 V (−8.2%)** — primary discrepancy |
| J_sc (mA/cm²) | 25.73 † | 26.282 | −0.55 (−2.1%) — optical |
| FF (%) | 85.6 | 86.99 | −1.4 pp |
| PCE (%) | 22.1 | 26.69 | −4.6 pp |

Sources: `tests/integration/scaps_reference.json` (SCAPS base_model); task FOM table and `outputs/scaps_full_off/summary.json` (SolarLab base point, V_oc 1.0711 V at the true base ΔE_C = −0.16 eV).

† **The J_sc = 25.73 mA/cm² cell is a foreign value spliced from a different (N_grid = 30) run; SolarLab's genuine self-consistent base-point J_sc is 23.96 mA/cm². See §2.2 and §5.3.** It is retained here only because it is the as-quoted task tuple; every per-FOM analysis below uses 23.96.

### 2.2 PCE/FF self-consistency check (called out explicitly)

In any single, self-consistent J–V evaluation, the four FOMs are algebraically locked: with FF ≡ P_mpp/(V_oc·J_sc) and PCE ≡ P_mpp/(1000 W·m⁻²), it follows identically that **PCE = V_oc · J_sc · FF** (`perovskite-sim/perovskite_sim/experiments/jv_sweep.py:118–120`). The two reference tools obey this lock to within rounding:

- **SCAPS:** 1.1676 × 26.282 × 0.8699 = 26.69% = reported 26.69% (residual < 0.01 pp). ✔
- **SolarLab `report.md`:** 1.0905 × 23.96 × 0.8802 = 22.99% = reported 22.99% (residual < 0.01 pp). ✔

The **task-quoted SolarLab tuple does not obey the lock**: 1.072 × 25.73 × 0.856 = **23.61%**, yet PCE is quoted as 22.1% — a **+1.51 pp residual that is mathematically impossible from a single `compute_metrics` call**. This is therefore a **reporting artifact** (a splice of FOM cells from two different runs), not a physics or solver defect. The adversarial review (§5) located the foreign cell precisely: SolarLab's genuine self-consistent base-point rows carry **J_sc ≈ 23.96 mA/cm²** (= 239.567 A/m² in the CSV artifacts), and a real, locked SolarLab point gives V_oc 1.0709 / J_sc 23.957 / FF 0.862 / **PCE 22.1%**. The **25.73 mA/cm² figure is the contaminant**: it originates from a different, later optimization run at N_grid = 30, where the peaked optical-generation profile inflates the trapezoidal J_sc integral by roughly +2.5 mA/cm² (`docs/superpowers/specs/2026-05-29-e10-rootcause-optimization-workflow.md`). The partner table glued that coarse-grid J_sc onto a fine-grid V_oc/FF/PCE triple. **Consequence: the genuine SolarLab PCE is ≈ 22.1–22.3%, and the self-consistent value is *not* 23.6%** — quoting 23.6% would re-commit the very splice. (This corrects the prior gap report's framing, which had assumed PCE was the stale cell.)

### 2.3 Three-way positioning against experiment

Neither solver is experiment; the question is which *absolute* is more representative of a fabricated MAPbI₃/spiro/TiO₂ n-i-p cell. The literature for E_g ≈ 1.53 eV:

- **Radiative Shockley–Queisser V_oc ceiling:** ≈ 1.30–1.33 V at 1.53 eV; one detailed-balance study reports an SQ V_oc ceiling of ~1.25 V with a measured device reaching 1.18 V (94.4% of SQ) [12, 15]. The Auger limit lies within < 2 mV of the radiative limit, so SQ is the correct reference [14].
- **Champion measured V_oc:** A mesoporous-TiO₂/spiro champion delivers J_sc 24.6 mA/cm², **V_oc 1.16 V**, FF 0.73, PCE 20.8% [16] — this brackets SCAPS's 1.1676 V as a *champion*, not a typical, value. Best-in-class low-loss stacks reach V_oc 1.21 V at 1.53 eV (certified 23.09% PCE, non-radiative V_oc loss ≈ 0.10 V), defining the physical ceiling for this exact gap [17].
- **Typical measured V_oc:** 1.05–1.13 V; single-crystal MAPbI₃ cells > 21% PCE sit at V_oc 1.0–1.1 V [18]. The typical non-radiative V_oc deficit is 150–280 mV below SQ [13].

**Positioning:** SCAPS 1.1676 V implies a non-radiative deficit of only ~130–160 mV (champion-grade interface passivation), consistent with the published champion ceiling but optimistic for a routine stack with a 20 nm spiro HTL and 25 nm TiO₂. SolarLab 1.072 V implies a ~230–260 mV deficit, squarely in the typical-device band. On PCE, **SCAPS's 26.69% exceeds the certified single-junction MAPbI₃ record (~21–23%)** and lies above the SQ-derated practical ceiling, whereas SolarLab's ~22.1% is realistic. *Conclusion: on absolutes, SolarLab is nearer the measured-device median; SCAPS is nearer the published champion. Neither is "wrong"; SolarLab's absolute is the more conservative/representative of a typical fabricated cell.* (Literature V_oc values are well-established and cited; the SQ ceiling spread of 1.25–1.33 V reflects bandgap-definition variance and is flagged as a moderate-confidence band.)

---

## 3. SCAPS vs. SolarLab: Physical-Model and Numerical-Algorithm Comparison

| Aspect | SolarLab (code file:line) | SCAPS-1D (manual page / DOI) |
|---|---|---|
| **DD system / variables** | Poisson + 2 continuity; carrier-density variables (n, p, φ). Poisson FV form d/dx(ε₀ε_r dφ/dx) = −ρ, harmonic-mean face permittivity, Dirichlet BCs (`poisson.py:111–182`). Continuity dn = (∇·J_n)/q − R + G; dp = −(∇·J_p)/q − R + G (`continuity.py:212–213`). | Poisson + 2 continuity in (ψ, E_Fn, E_Fp); J_n = μ_n·n·∇E_Fn, J_p = μ_p·p·∇E_Fp (QFL-gradient form), Manual p. 39 eq. (14)–(15) [1]. |
| **Flux discretization** | Exponentially-fitted Scharfetter–Gummel finite-volume: Bernoulli B(x) = x/(eˣ−1) with small/large branches; flux qD/h·(B(ξ)n_{i+1} − B(−ξ)n_i) (`fe_operators.py:9–81`). True SG, not a plain FE gradient. | Scharfetter–Gummel-style flux on non-uniform FD mesh, 2 coincident nodes/interface (Manual p. 39–40) [5]. |
| **Solver / convergence** | Method-of-lines pseudo-transient; Poisson = single LU-cached tridiagonal solve per RHS (no inner Newton); carriers advanced by Radau to steady state; rtol 1e-4, atol 1e-6, max_step cap, BDF fallback (`mol.py:1577–1622, 1741`). | Gummel scheme with Newton–Raphson substeps; adaptive mesh refinement on exp(qψ/kT), R, G ratios (f_max ≈ 1.60 ≈ 12 meV) (Manual p. 40) [1, 6]. |
| **Carrier statistics** | Boltzmann only; n_i a per-layer scalar, n_i² = n_i² in every R term (`parameters.py:17, 108–109`). No Fermi–Dirac/degeneracy. n_i derived from DOS in the validated path (`loader.py:252`). | Boltzmann by default; Fermi–Dirac, degeneracy, band-gap narrowing **not** implemented (Manual p. 31). n_i² = N_c·N_v·exp(−E_g/kT). |
| **Bulk SRH** | R = (np − n_i²)/(τ_p(n+n₁) + τ_n(p+p₁)) (`recombination.py:5–10`); n₁,p₁ from trap-depth helper (`device.py:25–26`). Two PVK defects combined by parallel SRH 1/τ_tot = Σ1/τ_i (`loader.py:23–32`). | SRH-through-defects; τ = 1/(σ·N_t·v_th). Neutral defect → only τ enters the rate, not the space charge (Manual p. 24, 26) [9]. |
| **Auger / radiative** | Auger (C_n n + C_p p)(np − n_i²) (`recombination.py:20–25`); radiative B_rad(np − n_i²) with per-RHS Yablonovitch P_esc photon-recycling reabsorption (`recombination.py:13–17`). | SRH + radiative + Auger bulk channels; no per-RHS photon-recycling reabsorption (Manual recombination sections). |
| **Interface SRH (Pauwels–Vanhoutte)** | **Default**: velocity-form face-SRH R = (np − n_i²)/((n+n₁)/v_p + (p+p₁)/v_n), v = σ·v_th·N_t (`recombination.py:28–44`; `loader.py:196–202`), sampled at bulk-interior cross-carrier nodes (n from transport side, p from absorber side) with n_i²_eff = n_R,eq·p_L,eq and an R ≥ 0 clamp (`mol.py:1188–1234`). A **true** Pauwels–Vanhoutte interface-plane-state model (n_1s, p_1s, n_2s, p_2s + TE coupling + two-sided SRH) exists but is env-gated OFF (`interface_plane.py:49–268`). | Single Pauwels–Vanhoutte interface-state SRH: electrons in layer-1 CB and holes in layer-2 VB recombine; **both** band edges set the effective n, p in the SRH denominator (Manual p. 30; Pauwels & Vanhoutte, J. Phys. D 11 (1978) 649) [8]. |
| **Contact BC** | **Default hard Dirichlet ohmic** pin: dn[0] = dn[−1] = 0, carriers held at doping-equilibrium n_R = N_D, p_R = n_i²/N_D (S → ∞ ohmic) (`continuity.py:216–223`; `mol.py:874–888`). Optional finite-S Robin path J = ±qS(n − n_eq) exists but is supplied by no shipped preset (`contacts.py:62–134`). | Metal work-function Φ_m for majority carriers, or "flat bands" (Φ_m solved for flatband); shallow doping in eq. (1)–(3); wavelength-dependent reflection/transmission filter at the contact (Manual p. 9–10) [1]. |
| **Poisson built-in potential** | **Frozen manual scalar** `stack.V_bi` = 1.30 V used at the boundary φ_right = stack.V_bi − V_app (`mol.py:1334, 1731`; config line 40). Band-derived V_bi_eff = compute_V_bi() (Fermi-level difference, **tracks** χ_ETL and N_D) is computed but routed **only** to the V_max sweep range (`mol.py:895, 1054`; `device.py:186`). | Built-in potential set self-consistently by the work-function/flat-band contact electrostatics; moves with band edges and doping (Manual p. 9–10) [1]. |
| **Band-offset transport / tunnelling** | Vacuum level continuous (φ_n = φ + χ, φ_p = φ + χ + E_g); at \|ΔE_C\| or \|ΔE_V\| > 0.05 eV the SG flux is **capped** to the Richardson–Dushman thermionic-emission bound (`continuity.py:144–181`). TE applied as a **flux ceiling on top of SG**, not as the interface BC. | Thermionic emission **as** the interface transport BC (always a finite QFL step); intra-band and band-to-band tunnelling are optional checkboxes, **OFF** in the partner model (Manual p. 30–31; Burgelman & Marlein, TSF 515 (2007) 6276) [2]. |
| **Optics** | TMM front reflection over glass→spiro→PVK; wavelength-resolved R(λ); glass index-matching recovers most of the bare-stack Fresnel loss (`optics.py`; config lines 47–51). Finite-thickness absorber transmission and a dropped multi-pass geometric series mean net optical loss exceeds pure front R (`optics.py:482–488`). | Internal generation g(x) from AM1.5G (1000 W/m²); **front reflection = 0** unless a reflection/filter .spe is supplied (none in the partner model) → near-ideal absorption (Manual p. 10, 36–37). |
| **FOM extraction** | J_sc = interp(0, V, J); V_oc by linear interpolation at the first J sign-crossing; FF = P_mpp/(V_oc·J_sc); PCE = P_mpp/1000 (`jv_sweep.py:105–120`). Terminal J for the sweep uses the contact face J_faces[0] (carries displacement transients), not the charge-conserving median (`jv_sweep.py` `_compute_current` vs. `_compute_current_ss:377`). | V_oc·J_sc·FF self-consistent (PCE 26.69 = 1.1676·26.282·0.8699). |

**Key model-level divergences feeding the base-point discrepancy:** (1) the **frozen V_bi boundary** (SolarLab) vs. the doping-/band-tracking flatband contact (SCAPS); (2) the **TE flux-cap-on-top-of-SG** (SolarLab) vs. **TE-as-the-BC** (SCAPS); (3) the **default bulk-projected cross-carrier velocity-SRH interface model** (SolarLab) vs. **single-plane Pauwels–Vanhoutte both-band-edge SRH** (SCAPS); (4) **hard-Dirichlet ohmic contacts** (SolarLab default) vs. flatband work-function contacts (SCAPS); (5) **TMM front reflection + finite-absorber transmission** (SolarLab) vs. **R = 0 ideal optics** (SCAPS).

---

## 4. Step-by-Step Sweep Analysis (10 sweeps)

The SolarLab side uses the default path (interface projection OFF, `outputs/scaps_full_off/summary.json`); ON figures (`scaps_full_on`) are noted where they matter. "Closure" = SolarLab V_oc span ÷ SCAPS V_oc span; "Dir" = direction match; "Base gap" = SolarLab − SCAPS V_oc at the sweep's base point.

| # | Sweep (lever) | SCAPS span (mV) | SolarLab span (mV) | Closure | Dir | Base gap (mV) | Group |
|---|---|---|---|---|---|---|---|
| 1 | CHI_ETL (CBO) | 918.0 | 761.7 ‡ | 83.0% | ✔ | −96.5 | matched (slope) |
| 2 | Nt_PVK_ETL (iface N_t) | 226.7 | 246.2 | 108.6% | ✔ | −338.7 | divergent (shape/abs) |
| 3 | Nd_ETL (ETL donor) | 99.6 | 29.7 | 29.8% | **✘** | −96.5 | divergent (direction) |
| 4 | Nt_HTL_PVK (iface N_t) | 5.2 | 85.0 | 1633% | **✘** | −69.3 | divergent (over-sensitive) |
| 5 | Nt_C_PVK (bulk N_t, CB) | 38.6 | 0.1 | 0.2% | ✔ | −95.9 | divergent (dead) |
| 6 | Nt_V_PVK (bulk N_t, VB) | 10.8 | 0.1 | 0.6% | ✔ | −95.8 | divergent (dead) |
| 7 | Et_PVK_ETL (iface trap E) | 34.5 | 0.0 | 0.0% | **✘** | −96.5 | divergent (dead) |
| 8 | Et_C_PVK (bulk E_t, CB) | 0.4 | 3.4 | 787% | ✔ | −96.0 | divergent (over-sensitive) |
| 9 | Et_V_PVK (bulk E_t, VB) | 0.4 | 3.4 | 783% | ✔ | −95.8 | divergent (over-sensitive) |
| 10 | Et_HTL_PVK (iface trap E) | 0.0 | 0.0 | flat | flat | −96.5 | matched (both flat) |

‡ **The 761.7 mV SolarLab span is the absolute-χ-drag sweep variable in `summary.json` (the ETL affinity itself is dragged across the full range). It is *not* the same lever as the 18 mV span quoted in `report.md`, which sweeps the band-aligned ΔE_C offset (FF stays ~0.83 and the junction keeps working). The two SolarLab numbers are the same code on two different sweep-variable definitions; the reconciliation is detailed in §4.1 and §5.6.**

### 4.1 Matched / acceptable

**Sweep 1 — CHI_ETL (conduction-band offset).** SCAPS's dominant V_oc lever (918 mV range over 14 points). SolarLab reproduces 83% of the span and the direction is correct, the **best trend match**. *But the gap is not constant:* it is −7 mV in the deep-cliff regime (ΔE_C = −1.0 to −0.5 eV, where the real cliff barrier limits V_oc in both tools and they agree), −96.5 mV at the base (−0.16 eV), and grows to −163 to −166 mV at flat-band/spike (ΔE_C ≥ 0). SolarLab's V_oc *saturates* near 1.085 V for ΔE_C ≥ −0.1 eV (6.0 mV spread) while SCAPS keeps climbing to 1.25 V (33 mV further). Projection ON raises slope closure to 92% but lowers base V_oc by ~19 mV. **A note on the two SolarLab spans:** the `summary.json` value used in the table above is **761.7 mV** because that sweep drags the absolute electron affinity χ_ETL across its full range (which eventually destroys the junction at extreme negative offset); `report.md` reports an **18 mV** span for the *same* code because it sweeps the band-aligned ΔE_C with the rest of the band structure shifted self-consistently, so the device keeps working and V_oc moves little. Both are correct for their respective sweep-variable definitions; they are not in conflict. **Inline root cause (contested):** the saturation signature is consistent with the frozen-V_bi boundary failing to track χ_ETL, *and/or* the TE flux cap dissipating QFL in the spike regime — the two mechanisms could not be cleanly separated (see §5).

**Sweep 10 — HTL/PVK interface trap *energy* (Et_HTL_PVK).** SCAPS itself is dead here (0.0 mV) and SolarLab is dead (0.0 mV). Both flat → trivially "matched." Not diagnostic.

### 4.2 Divergent

**Sweep 3 — Nd_ETL (ETL donor doping): DIRECTION REVERSAL.** SCAPS rises monotonically +99.6 mV with doping (sub-ideal ~14 mV/decade); SolarLab nets **−11 mV** (closure 29.8%, dir_match = false). This is the cleanest, best-supported root cause in the study (3/3 lenses held). **Inline root cause:** SolarLab freezes the Poisson boundary at `stack.V_bi` = 1.30 V (`mol.py:1334, 1731`), so raising N_D cannot lift the built-in potential. The chi-/doping-aware `V_bi_eff = compute_V_bi()` (`device.py:103–134`) *does* track N_D but is routed only to the sweep range, never the boundary. The only residual N_D channel is the ohmic pin n_R = N_D, p_R = n_i²/N_D (`mol.py:874–888`): raising N_D *lowers* the pinned minority-hole density, slightly *suppressing* QFL splitting → wrong sign. The raw rows show the predicted non-monotonic signature: V_oc falls 1.0954 → 1.0656 V (1e13 → 1e16 cm⁻³, pin tightening) then drifts up only to 1.0844 V (1e20) from second-order electrostatics. The very-low-N_D rows (≤ 1e12 cm⁻³) fail to converge (V_oc = 0), an extraction failure separate from the V_bi mechanism. *Caveat surfaced in review:* the SCAPS lever's dominant low-doping gain is actually FF/extraction recovery (~7.7 mV/dec, far below the 59.5 mV/dec ideal flatband slope), with a genuine V_bi lift only in the top ~2 decades — SolarLab severs both the V_bi lever and the FF-recovery lever. Classification: **both (numerical boundary-condition severance + physical contact choice)**; confidence high; survived 3/3.

**Sweep 2 — Nt_PVK_ETL (PVK/ETL interface trap density): SPAN MATCHES, SHAPE DOES NOT.** The headline 108.6% "closure" is a **span-ratio artifact**. Per-decade, the two curves have *opposite curvature*: SCAPS is plateau-then-knee (−1 to −12 mV/dec below 1e11 cm⁻², then −68 to −79 mV/dec at 1e12–1e13), while SolarLab is **front-loaded** (−76 mV/dec from the first decade, saturating to −9 mV/dec exactly where SCAPS is steepest; SolarLab also returns NaN at N_t = 1e15). The base gap is −338.7 mV — the worst of any sweep — so the absolute is badly over-recombining. **Inline root cause:** SolarLab's velocity-form face-SRH puts N_t only in the denominator via v = σ·v_th·N_t (`recombination.py:28–44`), giving a velocity-limited→density-limited crossover (front-loaded then saturating), which is the wrong shape for an interface-state SRH whose recombination-active occupancy turns on near a threshold N_t; SCAPS's single-plane Pauwels–Vanhoutte rate stays plateaued until the interface density competes with bulk extraction (the late knee). The over-recombination at low N_t reflects sampling near-majority bulk-interior densities (`mol.py:1188–1234`) so (np − n_i²) is large even at negligible N_t. The true interface-plane-state path (`interface_plane.py`) is mis-calibrated (the ifacestate variant collapses sensitivity to 2%). Classification: **both (physical interface-SRH model + numerical sampling location)**; the prior framing of "slope reproduced, only absolute wrong" did **not** survive — the shape is the part that fails.

**Sweep 4 — Nt_HTL_PVK (HTL/PVK interface trap density): OVER-SENSITIVE.** SCAPS moves only 5.2 mV; SolarLab moves 85 mV (1633% closure, dir_match = false). Same velocity-SRH model as Sweep 2, here far too responsive at the HTL/PVK plane — consistent with the cross-carrier interior-node sampling over-weighting this interface.

**Sweeps 5, 6 — Nt_C/V_PVK (PVK bulk trap density): DEAD.** SCAPS moves 38.6 / 10.8 mV; SolarLab moves 0.1 / 0.1 mV (closure 0.2% / 0.6%). The SRH rate form and the parallel-defect combination are coded correctly (`recombination.py:5–10`; `loader.py:23–32`), so the term is right but not rate-limiting at the operating point. **Inline root cause (contested):** the bulk channel is sub-dominant. Adversarial review showed the cleaner physics is **carrier-density suppression** — SolarLab's absorber n·p at V_oc is low (the same root as the −96 mV base offset: lower QFL splitting → small np), so (np − n_i²) is tiny and the N_t-linear prefactor is invisible. The "transport-ceiling-masks-bulk" framing did *not* survive: a single series ceiling would suppress the E_t lever equally, yet SolarLab is N_t-dead but E_t-*hyper*sensitive (3.4 mV vs. SCAPS 0.4 mV, ~8×; see Sweeps 8–9), which demands that n₁,p₁ still matter in the denominator — i.e. carriers are starved, not masked. Both tools are *also* base-flat below 1e13 cm⁻³ (100 µs lifetime), so much of SCAPS's 38.6 mV comes from the single 1e14→1e15 decade. Classification: physical-model; the dead bulk response is a *symptom* of the V_oc-suppression root, not an independent bug.

**Sweep 7 — Et_PVK_ETL (interface trap energy): DEAD + DIRECTION MISMATCH.** SCAPS moves 34.5 mV with trap position; SolarLab moves 0.0 mV. The interface-SRH denominator's n₁,p₁ are insensitive to trap position in the default path — a real but secondary defect, distinct from the N_t-magnitude question.

**Sweeps 8, 9 — Et_C/V_PVK (PVK bulk trap *energy*): SolarLab MILDLY OVER-SENSITIVE.** SCAPS is essentially dead (0.4 / 0.4 mV) while SolarLab moves **3.4 / 3.4 mV** — an ~8× over-sensitivity, so the span-ratio closure metric reads **787% / 783%** (it is not "flat" on the SolarLab side). The "both flat" description is true only for SCAPS. This is the same E_t-hypersensitivity that §5.5's verifier uses (3.4 vs. 0.4 mV, "E_t-hypersensitive") as evidence that SolarLab's absorber carriers are *starved* rather than masked: because n₁,p₁ ~ n_i·exp(±E_t/kT) still matter in the SRH denominator, a small trap-energy shift moves V_oc, which can only happen when n·p at V_oc is low. Direction matches and the absolute movement is tiny (3.4 mV), so this is a low-severity divergence — but it is *not* a clean match.

---

## 5. Root-Cause Findings

Each finding carries its classification, mechanism, evidence, rejected alternatives, confidence, cost of forcing agreement, and — critically — its **adversarial survival** across three lenses (physics-correctness, numerical-algorithm, data-consistency). Causes that did not survive a majority of lenses are marked **TENTATIVE**.

### 5.1 V_oc −96 mV primary deficit — TENTATIVE (mechanism contested; 1/3 lenses held)

- **Classification:** physical-model (boundary/transport), with a numerical reporting cross-check that is separable.
- **Proposed mechanism (as advanced):** SolarLab's hard-Dirichlet ohmic outer contacts pin the minority carrier at doping-equilibrium (infinite surface-recombination velocity), injecting a parasitic contact-recombination current absent in SCAPS's finite-velocity flatband contacts; the implied dark-saturation prefactor ratio J0,SL/J0,SCAPS ≈ 33–42× translates to V_T·ln(≈33–42) = 90–96 mV ≈ the observed deficit (`continuity.py:216–223`; `mol.py:874–892`).
- **Evidence for:** The J0 ratio and its V_T·ln translation are arithmetically verified (J0,SL ≈ 2.5e-20, J0,SCAPS ≈ 6.2e-22 A/cm²; V_T·ln(41.7) = 96.5 mV = the exact base gap). The deficit is genuinely physical (reproduced by the validated sweeps, FF healthy at 0.878 in the self-consistent run, so not a terminal-face artifact). V_bi headroom (band-derived 1.294 V ≈ manual 1.30 V) and n_i (Boltzmann-consistent) are excluded as drivers. The TE cap does *not* bind on the forward light current (v_TE/v_diff ≈ 1045).
- **Why it is TENTATIVE (refutations that held):**
 1. *Physics lens (REFUTED):* A hard-Dirichlet ohmic pin is the **same** S → ∞ minority condition SCAPS's flatband ohmic contact uses, so by itself it cannot produce a code-to-code difference. The implied J0 for both tools is consistent with a **bulk-SRH-limited** diode (both operating points sit between the radiative floor J0 ≈ 2.5e-23 and the SRH level), so the 33–42× is the ratio of two SRH-dominated diodes — better explained by the two codes computing **different interface SRH rates** (SolarLab's default bulk-projected cross-carrier SRH vs. SCAPS Pauwels–Vanhoutte both-band-edge SRH), which is untested here.
 2. *Data-consistency lens (REFUTED):* Reading the full 14-point CHI_ETL rows shows the gap is **monotone in CBO** (−7 mV at the deep cliff → −163 mV at flat-band), not a constant contact plateau. Agreement to −7 mV at the deep cliff cannot arise from an S → ∞ sink that is unchanged across the sweep. Projection ON (a pure interface-recombination lever) moves base V_oc by −19 mV, which a metal-contact sink should be insensitive to. Both point to a **band-offset / interface-transport** mechanism, not a contact sink. The "33×" figure is also numerically loose; the exact ratio is 41.7×.
 3. *Numerical lens (HELD):* The deficit is not a discretization/FOM artifact — V_oc is face-choice-insensitive at the crossing, the PCE/FF inconsistency is V_oc-orthogonal, and the TE cap does not bind. The base deficit is better quoted as **−77 to −96 mV** (the self-consistent `report.md` run gives −77 mV: 1.0905 vs. 1.1676), with ~19 mV run-to-run spread.
- **Net status:** The *symptom* (an elevated effective J0 of ≈ 40× → ≈ 96 mV) and the *exonerations* (V_bi, n_i, TE-cap, FOM-extraction as drivers of the base V_oc) are robust. The *attribution* to a contact-specific minority sink is **not established and probably wrong**; the better-supported (but not independently confirmed) origin is the **difference between SolarLab's default interface-SRH model and SCAPS's Pauwels–Vanhoutte interface-state + thermionic-emission interface BC**. The prior gap report's "37× J0 from QFL dissipated ~135 mV across band offsets" is **falsified at the base point by the CBO-sweep sign** (gap widens, not closes, as the offset is removed).
- **Confidence:** medium that the cause lies in interface SRH/transport; low that it is the contact BC specifically.
- **Missing experiment (the one that would settle it):** a controlled finite-S contact re-run and a per-channel R(x) recombination-budget dump at V_oc; neither was executed (budget-limited).
- **Cost of forcing agreement:** switching to finite-S selective/flatband contacts or routing V_bi_eff into the boundary would re-baseline every regression, change FF/J_sc on all presets, risk Radau stiffness from the Robin time constant dx/S, and tune SolarLab *off* the experimental median (1.05–1.13 V) *toward* SCAPS's champion ceiling (1.168 V) — a fidelity loss against literature. **Not recommended.**

### 5.2 Nd_ETL direction reversal — ROBUST (3/3 lenses held)

- **Classification:** both (numerical boundary-condition severance, compounded by the physical contact choice).
- **Mechanism:** The Poisson right BC is hard-wired to `stack.V_bi` = 1.30 V (`mol.py:1334, 1731, 1759`); the N_D-dependent `V_bi_eff = compute_V_bi()` (`mol.py:895`, via `_fermi_level` reading N_D) flows only to V_max, never the boundary. With V_bi frozen, raising N_D cannot lift the built-in potential; the only residual channel is the ohmic pin (p_R = n_i²/N_D ↓ as N_D ↑), which slightly suppresses QFL splitting → wrong sign.
- **Evidence:** Direction data `sl_dir = −0.0109`, `sc_dir = +0.0996`, closure 29.8%, dir_match = false (`outputs/scaps_full_off/summary.json`). Predicted non-monotonic SolarLab signature (fall 1.0954 → 1.0656, then rise to 1.0844) confirmed in raw rows. Projection ON preserves the wrong sign (`sl_dir = −0.0111`) — interface recombination is downstream of, and cannot restore, the severed lever. All three lenses independently confirmed the code path and the fall-then-rise signature; the ON/OFF discriminator passed (an interface-recombination root cause would have flipped direction under ON; it does not).
- **Rejected alternatives:** the −96 mV base offset (flat across the sweep, cannot cause a slope-sign error); the TE cap (ΔE_C fixed under N_D); n_i (per-layer fixed, shifts absolute not derivative sign). All correctly excluded.
- **Confidence:** high. One nuance surfaced in review: SCAPS's headline Nd_ETL benefit is largely FF/extraction recovery at low doping (7.7 mV/dec ≪ 59.5 mV/dec ideal), not a clean flatband V_bi lift; SolarLab severs both levers. The very-low-N_D (≤ 1e12 cm⁻³) SolarLab V_oc = 0 rows are a convergence/extraction failure, not part of the V_bi mechanism.
- **Cost of forcing agreement:** routing V_bi_eff (or a flatband Robin contact) into the boundary recovers the +N_D direction but shifts the base-point V_bi off 1.30 V, perturbs the calibrated base offset and every other sweep's base V_oc, breaks the documented IonMonger-convention invariant and the 17/17 guard/regression baselines, and reintroduces the contact time-constant the hard pin avoids. **Real defect, worth flagging; the BC rework is the multi-week refactor the house rule warns against — not recommended for parity.**

### 5.3 PCE/FF self-consistency anomaly — ROBUST classification, CORRECTED mechanism

- **Classification:** **reporting-artifact** (not physics, not solver). This top-level label held across all three lenses.
- **Mechanism (corrected):** FF and PCE both derive from the same P_mpp scalar (`jv_sweep.py:118–120`), so PCE ≡ V_oc·J_sc·FF in any single run. The task tuple violates this by +1.51 pp, definitionally a multi-source pairing. **The foreign cell is J_sc (25.73 mA/cm²), not PCE.** SolarLab's genuine self-consistent base rows carry J_sc = 23.957 mA/cm² and PCE ≈ 22.1% (locked: 1.0709·23.957·0.862 = 22.1%); the 25.73 figure comes from a separate N_grid = 30 optimization run whose peaked-G trapezoidal integral inflates J_sc by ~+2.5 mA/cm². The prior gap report's framing (PCE was the stale cell; "true" PCE ≈ 23.6%) is **wrong** — re-quoting 23.6% would re-commit the splice.
- **Evidence:** Algebraic lock verified in source; `report.md` (22.99% locked) and SCAPS (26.69% locked) both self-consistent to < 0.01 pp; the contaminated tuple appears only in partner-facing prose (`docs/partner/SolarLab_SCAPS_gap_analysis.md`), never in any pipeline output; the J_sc back-solve (PCE 22.1% needs J_sc ≈ 24.08 at fixed V_oc/FF, matching the real 23.96 within rounding) pinpoints J_sc as the spliced cell.
- **Rejected alternatives:** an FF physics defect (would move FF and PCE together, preserving the lock); the terminal-face J_faces[0] noise (real sweep CSVs are self-consistent to 0.000 pp using that very path, so it is not the contaminant here); the −96 mV V_oc physics (orthogonal — PCE and FF are algebraically locked).
- **Confidence:** high on classification and on J_sc-as-contaminant.
- **Cost of forcing agreement:** **near-zero.** Re-extract all four FOMs from one J–V array in a single `compute_metrics` call and quote that self-consistent tuple. This is the one cheap correctness fix recommended below.

### 5.4 J_sc −2% (front optics) — partially TENTATIVE

- **Classification:** expected-physical-residual (optical assumption mismatch, not a numerical bug). This held.
- **Mechanism (corrected):** SolarLab computes a real front reflectance via TMM (glass/spiro/PVK); SCAPS uses R = 0 by default. *But the "−2.10% = (1−R)" closure in the prior framing is built on the spliced J_sc = 25.73.* The genuine validated SolarLab J_sc is **23.96 mA/cm²** (`report.md`; sweep CSVs), so the real deficit vs. SCAPS 26.282 is **−2.33 mA/cm² = −8.8%**, not −2%. A ~5% glass-stack front reflection (config lines 47–51) cannot account for 8.8%; the residual ~3–4 pp is **absorption-side** — finite-thickness 800 nm transmission and a dropped multi-pass geometric series in `optics.py:482–488`, plus parasitic HTL/ETL/glass absorption — which SCAPS's near-unity internal absorption recovers.
- **Status:** the *direction and classification* (SolarLab < SCAPS, optical, decoupled from the V_oc transport gap) survive; the *magnitude and single-cause "front reflection" attribution* did **not** (3/3 refuted). Possible numerical contributors to the absorption residual (AM1.5G spectral quadrature resolution, trapezoidal optical-depth integration on the tanh grid, the `_inv2x2` determinant-guard clamp) were flagged but not excluded.
- **Confidence:** high that the residual is optical and V_oc-orthogonal; low that it is pure front reflection.
- **Cost of forcing agreement:** disabling TMM reflection (R = 0) would discard a physically real loss and make SolarLab's J_sc less representative of fabricated cells (champion MAPbI₃ J_sc ≈ 24.6 mA/cm²; SolarLab 23.96 already realistic). **Not recommended** — but the J_sc table cell should be quoted as 23.96, not 25.73 (see §5.3).

### 5.5 Bulk-trap insensitivity (Nt_C/V_PVK dead) — TENTATIVE mechanism (1/3 held)

- **Classification:** physical-model.
- **Status:** The SRH form and wiring are correct (held across lenses); the dead N_t response is real. The proposed "transport ceiling masks a sub-dominant parallel bulk channel" mechanism was **REFUTED on physics and numerical lenses**: it fails the high-N_t limit (SolarLab shows zero bend where SCAPS reaches 30 mV/dec) and the N_t-dead / E_t-hypersensitive asymmetry (N_t closure 0.2% but E_t span 3.4 mV vs. SCAPS 0.4 mV, ~8× — see Sweeps 8–9), which together indicate **carrier-density suppression** (small np at V_oc) rather than series masking. The data-consistency lens *held* (the attenuated-but-faithful, monotonically-un-masking SolarLab response rules out a wiring bug). **Net: dead bulk response is a symptom of the same V_oc-suppression root as §5.1, not an independent defect.**
- **Confidence:** medium that it is a symptom; the precise mechanism shares the §5.1 uncertainty.

### 5.6 CHI_ETL absolute saturation (frozen V_bi vs. TE cap) — TENTATIVE (1/3 held)

- **Classification:** both (numerical BC + physical TE cap), primary contested.
- **Status:** The frozen-V_bi code asymmetry is real and confirmed (numerical lens HELD; it is a *documented* IonMonger-convention choice, not an accidental bug). But the *electrostatic-cap mechanism* was **REFUTED on physics and data lenses**: SolarLab saturates at ~1.085 V, ~215 mV **below** the frozen 1.30 V boundary, so V_bi is not the binding constraint; and projection ON moves the saturation ceiling while the Poisson boundary is byte-identical, proving the ceiling is set by **interface recombination**, not the BC. The physics lens argues the surviving chi-coupled interior term is the **TE flux cap** (the demoted "compounding" factor). **Net: the saturation is recombination/QFL-limited, not electrostatically clamped; the frozen-V_bi bug is real but inert for V_oc at the base point (its live consequence is the Nd_ETL reversal, §5.2).** Note that the **761.7 mV SolarLab span** quoted for this sweep in `summary.json` is the absolute-χ-drag variable; `report.md`'s **18 mV** span is the band-aligned ΔE_C variable on the same code (see §4.1).

---

## 6. Synthesis and Remediation

### 6.1 Genuine physical-model differences (real, by design or by formulation)

1. **Interface-recombination model.** SolarLab's default velocity-form bulk-projected cross-carrier SRH (`recombination.py:28–44`; `mol.py:1188–1234`) is *not* SCAPS's single-plane Pauwels–Vanhoutte both-band-edge SRH [8]. This drives the Nt_PVK_ETL shape mismatch (§4.2/Sweep 2), the Nt_HTL_PVK over-sensitivity (Sweep 4), the Et_PVK_ETL deadness (Sweep 7), and is the **most likely true origin of the −96 mV base deficit** (§5.1). A true Pauwels–Vanhoutte path exists (`interface_plane.py`) but is mis-calibrated/off.
2. **Thermionic-emission treatment.** SolarLab caps the SG flux at the Richardson bound for \|ΔE_C\| > 0.05 eV (`continuity.py:144–181`); SCAPS uses TE *as* the interface BC (Manual p. 30) [8]. This shapes the CHI_ETL spike-regime gap (§5.6) and is the surviving candidate for that saturation.
3. **Contact statistics.** SolarLab default = hard-Dirichlet ohmic (S → ∞); SCAPS = flatband work-function contacts (Manual p. 9–10). Same S → ∞ minority limit, so *not* by itself a code-to-code V_oc driver (§5.1, refuted) — but the frozen-V_bi coupling makes it the live cause of the Nd_ETL reversal (§5.2).
4. **Optics.** SolarLab TMM front reflection + finite-absorber transmission vs. SCAPS R = 0 (§5.4). A real, bounded physical residual.
5. **Carrier-density suppression at V_oc.** The low absorber np that masks the bulk-trap lever (§5.5) is a *consequence* of (1)/(2), not an independent difference.

### 6.2 Numerical-algorithm artifacts (true code-level issues)

1. **Frozen Poisson V_bi boundary** (`mol.py:1334, 1731`): the band-/doping-derived `V_bi_eff` is computed but never used at the boundary. **Live defect** — produces the Nd_ETL direction reversal (§5.2). Documented IonMonger convention, so a deliberate design liability rather than an accidental error, but a liability with a real cost.
2. **PCE/FF FOM splice** (reporting layer, not solver): cross-run cell-mixing in partner-facing tables (§5.3). **The only cheap-to-fix correctness issue.**
3. **Terminal-face current convention** (`jv_sweep.py` `_compute_current` = J_faces[0]): carries displacement transients vs. the charge-conserving median used by the SS probe. *Not* the V_oc driver and *not* the PCE-splice contaminant (verified), but a latent FF/P_mpp noise source worth standardizing on the median.
4. **Low-N_D convergence failures** (Nd_ETL ≤ 1e12 cm⁻³ → V_oc = 0): an extraction/convergence gap at the lever extreme, orthogonal to the direction reversal.

### 6.3 Expected / bounded residuals (do not chase)

- **J_sc −2% (table) / −8.8% (true):** optical, V_oc-orthogonal, partially absorption-side; SolarLab's 23.96 mA/cm² is *more* realistic than SCAPS's R = 0 idealization (§5.4).
- **V_oc −96 mV:** physical; SolarLab sits at the experimental median, SCAPS at the champion ceiling (§2.3). Under trend-first, document — do not tune.
- **FF −1.4 pp:** consequent on V_oc and the terminal-face convention; not independently meaningful once the FOM splice is corrected.

### 6.4 Prioritised remediation (honest)

| Priority | Action | Class | Recommendation |
|---|---|---|---|
| **P0 (do now)** | Re-extract all four FOMs from a **single** J–V array in one `compute_metrics` call; quote SolarLab's self-consistent base tuple (V_oc 1.071 / J_sc 23.96 / FF ~0.862 / **PCE 22.1%**). Stop quoting J_sc 25.73 (foreign N_grid=30 cell). | Reporting fix | **DO** — near-zero risk, removes the −4.6 pp PCE artifact and the +1.51 pp self-consistency violation. |
| **P1 (cheap, optional)** | Standardize the swept J–V on the charge-conserving median face (`_compute_current_ss`) instead of J_faces[0]; regenerate any text citing 85.6%/22.1% as a matched pair. | Numerical hygiene | **DO if convenient** — small FF/P_mpp denoising, no physics change. |
| **P1 (documentation)** | Document the frozen-V_bi boundary as the cause of the Nd_ETL direction reversal; note `V_bi_eff` is intentionally routed only to V_max (IonMonger convention). | Documentation | **DO** — honest disclosure; do not silently hot-fix. |
| **P2 (investigate, not fix)** | Run the missing controlled experiments for the −96 mV: a finite-S contact re-run and a per-channel R(x) budget dump at V_oc, to settle interface-SRH-rate vs. contact-BC attribution. | Investigation | **OPTIONAL** — diagnostic only. |
| **NOT RECOMMENDED** | Route `V_bi_eff` into the Poisson boundary; switch defaults to finite-S flatband contacts; recalibrate the true Pauwels–Vanhoutte interface-plane model as default; disable TMM front reflection. | Costly refactor | **DO NOT** — each re-baselines the 17/17 guard suite and every sweep's base point, risks Radau stiffness, and (for V_oc) tunes SolarLab *away* from the experimental median toward SCAPS's optimistic champion ceiling. Violates the trend-over-absolutes house rule for an absolute-parity gain that is not physically more correct. |

### 6.5 Bottom line

SolarLab reproduces SCAPS's **dominant V_oc lever (conduction-band offset) at 83% span fidelity with correct direction**, which is the trend that matters most. Its absolute V_oc (1.072 V) and PCE (~22.1%) are *more representative of fabricated MAPbI₃ devices* than SCAPS's champion-ceiling absolutes. The genuine defects are: one **reporting splice** (P0, fix now — it is the source of the headline −4.6 pp PCE gap and the J_sc table inconsistency) and one **frozen-boundary direction reversal** on ETL doping (document, do not refactor). The prior report's central physical claim — a 37× J0 from QFL dissipated across band offsets — does **not survive** the conduction-band-offset sweep data and should be retracted in favour of the interface-SRH-model difference as the better-supported (though not yet independently confirmed) origin of the −96 mV deficit.

---

## 7. References

1. M. Burgelman, P. Nollet, S. Degrave. "Modelling polycrystalline semiconductor solar cells." *Thin Solid Films* **361–362**, 527–532 (2000). — Core SCAPS drift-diffusion formulation; Gummel/Newton solver.
2. M. Burgelman, J. Marlein. "Analysis of graded band gap solar cells with SCAPS." *Thin Solid Films* **515**(15), 6276–6278 (2007). — Intra-band tunnelling at heterointerfaces (OFF in the partner model).
3. K. Decock, S. Khelifi, M. Burgelman. "Modelling multivalent defects in thin-film solar cells." *Thin Solid Films* **519**(21), 7481–7484 (2011). — Defect/grading algorithms.
4. J. Niemegeers, M. Burgelman. Thermionic-emission interface transport and numerical scheme (1998). — Heterojunction boundary condition.
5. D. L. Scharfetter, H. K. Gummel. "Large-signal analysis of a silicon Read diode oscillator." *IEEE Trans. Electron Devices* **16**(1), 64–77 (1969). — Exponentially-fitted flux discretization.
6. *SCAPS Manual* (February 2016), University of Gent. — pp. 9–10 (contacts/flatband eq. 1–3), pp. 24/26 (neutral defects; τ usage), pp. 30–31 (interfaces: thermionic emission + Pauwels–Vanhoutte; tunnelling), pp. 39–41 (DD eq. 14–15, Gummel + Newton, FD mesh), p. 31 (Boltzmann default), pp. 36–37 (optics, R = 0 default).
7. W. Shockley, W. T. Read. "Statistics of the recombinations of holes and electrons." *Phys. Rev.* **87**(5), 835–842 (1952). — Bulk SRH theory.
8. R. J. Pauwels, G. Vanhoutte. "The influence of interface state and energy barriers on the efficiency of heterojunction solar cells." *J. Phys. D: Appl. Phys.* **11**(5), 649 (1978). — Interface-state (both-band-edge) recombination theory.
9. J. Nelson. *The Physics of Solar Cells.* Imperial College Press (2003). — SRH/recombination, diode/J0, device physics reference.
10. P. Calado, A. M. Telford, D. Bryant, et al. "Evidence for ion migration in hybrid perovskite solar cells with minimal hysteresis." *Nat. Commun.* **7**, 13831 (2016) / Driftfusion. — Open-source perovskite DD reference.
11. N. E. Courtier, J. M. Cave, J. M. Foster, A. B. Walker, G. Richardson. "IonMonger: a free and fast planar perovskite solar cell simulator with coupled ion vacancy and charge carrier dynamics." *J. Comput. Electron.* **18**, 1435–1449 (2019). — V_bi-as-free-parameter convention; TE handling.
12. "Effect of perovskite thickness on electroluminescence and solar cell conversion efficiency." *J. Phys. Chem. Lett.* (2020). doi:10.1021/acs.jpclett.0c02363. — Measured V_oc 1.18 V = 94.4% of SQ at 1.53 eV. *(Moderate confidence on the SQ-ceiling value, which varies with bandgap definition.)*
13. K. Tvingstedt, O. Malinkiewicz, A. Baumann, et al. "Radiative efficiency of lead iodide based perovskite solar cells." *Sci. Rep.* **4**, 6071 (2014). doi:10.1038/srep06071. — Non-radiative V_oc deficit / EL reciprocity baseline (150–280 mV).
14. L. M. Pazos-Outón, T. P. Xiao, E. Yablonovitch. "Fundamental efficiency limit of lead iodide perovskite solar cells." *J. Phys. Chem. Lett.* **9**(7), 1703–1711 (2018). doi:10.1021/acs.jpclett.7b03054. — Radiative ≈ Auger limit (< 2 mV).
15. W. E. I. Sha, X. Ren, L. Chen, W. C. H. Choy. "The efficiency limit of CH₃NH₃PbI₃ perovskite solar cells." *Appl. Phys. Lett.* **106**, 221104 (2015). arXiv:1506.09003. — SQ/detailed-balance V_oc and PCE ceiling for MAPbI₃.
16. Mesoporous-TiO₂/spiro champion (luminescent-perovskite cell): J_sc 24.6 mA/cm², V_oc 1.16 V, FF 0.73, PCE 20.8%. *Energy Environ. Sci.* (2016); PMC4705040. — Direct n-i-p champion bracket for SCAPS's 1.1676 V.
17. Low-loss MAPbI₃ stack: V_oc 1.21 V at 1.53 eV, certified 23.09% PCE, ΔV_oc,nr ≈ 0.10 V. *Sci. China Mater.* (2025). doi:10.1007/s40843-025-3457-3. — Physical V_oc ceiling for this exact bandgap.
18. Single-crystal MAPbI₃ cells > 21% PCE, V_oc 1.0–1.1 V. *ACS Energy Lett.* **4**(5), 1258 (2019). doi:10.1021/acsenergylett.9b00847. — Typical-device V_oc regime (where SolarLab's 1.072 V sits).

---

*Notation: V_oc = open-circuit voltage; J_sc = short-circuit current density (mA·cm⁻²); J0 = dark-saturation current density (A·cm⁻²); FF = fill factor; PCE = power-conversion efficiency; V_T = kT/q ≈ 25.85 mV at 300 K; ΔE_C = conduction-band offset; N_D = donor density (cm⁻³); N_t = trap density (bulk cm⁻³ / interface cm⁻²); τ = SRH lifetime; n_i = intrinsic carrier density (cm⁻³); SG = Scharfetter–Gummel; TE = thermionic emission; QFL = quasi-Fermi level; SRH = Shockley–Read–Hall; SQ = Shockley–Queisser; MOL = method of lines. SolarLab code paths are relative to `perovskite-sim/perovskite_sim/` unless otherwise noted; sweep data from `outputs/scaps_full_{off,on,ifacestate}/summary.json` and `perovskite-sim/outputs/scaps_validation/`.*

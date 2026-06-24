<!-- Generated 2026-06-24 from a 29-agent workflow cross-referencing the Zotero SCAPS collection (14 refs) against the SolarLab codebase. -->

# SolarLab Physics-Validation Gap Analysis

## 1. What SolarLab has already validated

SolarLab has demonstrated a credible drift-diffusion core against SCAPS-1D across several axes: a settled-state **energy-band diagram** with true quasi-Fermi levels (equilibrium flat E_F, qV splitting under bias — `band_diagram.py:74-117`, tests at `test_band_diagram.py:22-42`); **spatial snapshots** of φ, E, n, p, P, ρ (`spatial.py:9-25`); **J-V curves** with forward/reverse hysteresis, metrics extraction, and current decomposition (`jv_sweep.py`); **2D defect-parameter scans** (Nt×Et, Nt×ΔE_C interface maps reproduced structurally vs SCAPS per project memory); and a documented **SCAPS trend-parity campaign** where sweep direction/magnitude match (V_oc gap root-caused to the DOS-fold QFL-step term, PCE 26.68 ≈ SCAPS 26.69 with the fix). What is conspicuously *not yet visualized* is the spatial charge/recombination/current-conservation layer that the reference simulator papers (Driftfusion, IonMonger, Courtier EES) lean on most heavily — even though, per the audit, most of those quantities are already extractable.

## 2. Gap matrix

| Test / quantity | Physics it verifies | References that do it | SolarLab status | Spatial-charge relevance | Effort |
|---|---|---|---|---|---|
| Recombination profile R(x) decomposed by mechanism | SRH/Auger/radiative/interface loss localization | Courtier EES (Fig 11), Calado 2016 | **glue** (`recombination.py:5-44`; sum-only discard at `continuity.py:215-218`) | high | small |
| Optical generation G(x) and net U(x)=G−R | photogeneration vs loss balance, collection | Driftfusion, ∂PV | **glue** (`mol.py:383` G cached; `recombination.py:47`) | high | small |
| Spatial current components Jn/Jp/Jion/Jdisp(x) + charge-conservation check | current continuity, charge conservation across x | Verschraegen (J vs x), Courtier (E_bulk) | **now** (`jv_sweep.py:202`, tested `test_current_decomposition.py:44-49`) | high | trivial |
| Quasi-Fermi splitting (E_Fn−E_Fp)(x) vs bias | QFL splitting → qV, carrier collection | Calado 2016, Pauwels, Courtier | **now** (`band_diagram.py:108-114`, tested) | high | trivial |
| Poisson self-consistency residual dE/dx=ρ/εε₀ | electrostatic solver correctness | Driftfusion (depletion ρ,E,φ), Courtier asymptotic | **glue** (`spatial.py:18-24`, `poisson.py:119-126`) | high | trivial |
| Mobile-ion redistribution P(x,t) ↔ hysteresis | ion migration screening, hysteresis origin | IonMonger, Calado 2016, Courtier EES | **now** (`jv_sweep.py:177-199,787-808`; `hysteresis_index`) | high | trivial |
| Energy-band diagram, settled QFLs | band alignment, built-in field, blocking | all simulator + theory papers | **now** (`band_diagram.py:74-117`) | high | trivial |
| EQE(λ) / IPCE + AM1.5G J_sc cross-check | spectral collection, optical-electrical consistency | Alshaikh, Kogo, Liu, ∂PV | **now** (`eqe.py:193`, 11 tests) | medium | trivial |
| C(V) / C(f) / Mott-Schottky → V_bi, N_eff | depletion electrostatics, doping extraction | IonMonger (impedance), Pitarch-Tena | **now** (`mott_schottky.py:237`, `impedance.py:130`) | medium | trivial |
| Impedance Nyquist/Bode | RC timescales, ionic vs electronic | IonMonger, Pitarch-Tena, Courtier EES | **now** (plots wired `impedance.ts`); **glue** for equiv-circuit/C_μ fit | medium | small |
| Diode ideality n (dark-JV & Suns-Voc) + J_0 | recombination mechanism, ideality | Calado 2016 (HI/ideality), Pauwels | **now** (dark, `dark_jv.py:35-59`); **glue** (Suns-Voc n) | medium | small |
| Voc(T) → activation energy E_A | dominant recombination pathway | (implicit in theory papers) | **now** (`voc_t.py:68`) | low | trivial |
| TPV / TPC / CE decay-time τ | recombination kinetics, transient response | Calado 2016 (TPV), IonMonger, Courtier (current decay) | **now** TPV (`tpv.py:206`); **glue** TPC/CE | high | small |
| Intra-band / thermionic-field tunneling (TFE) through CB spike | tunneling enhancement past band-offset spike | Verschraegen 2007 (SCAPS method) | **new-physics** (`fe_operators.py:84-98` pure TE, no WKB) | high | large |
| Continuously graded-bandgap absorber χ/Eg(x) → J-V/V_oc | grading-driven quasi-field, Jsc/Voc trade-off | Burgelman-Marlein 2008 (SCAPS), CIGS | **glue** (transport core ready `continuity.py:132-142`; arrays flat-broadcast `mol.py:646-647`) | high | medium |
| **Extra in A, absent from B:** | | | | | |
| Hysteresis-factor vs scan-rate sweep curve | scan-rate-dependent ion hysteresis | Courtier EES (Figs 5,8,9), IonMonger | **glue** (re-run `run_jv_sweep` at v_rate sweep, read `hysteresis_index`) | medium | small |
| Spatial-convergence / runtime benchmark (error vs N) | numerical scheme order, efficiency | Courtier AMM 2018, IonMonger perf | **glue** (refine N_grid, compare against fine baseline) | low | small |
| Non-Boltzmann statistical integrals (Fermi-Dirac/Gauss-Fermi) | degenerate/disordered carrier statistics | IonMonger 2.0 | **new-physics** (Boltzmann-only; FD parked per memory — hurts SCAPS parity) | low | medium |
| Steric/volume-exclusion ion Debye-layer saturation | nonlinear PNP crowding | IonMonger 2.0 | **glue/now** (Blakemore steric already in `J_ion`, `jv_sweep.py:268`) | high | small |
| Depletion-approximation analytic J-V cross-check | textbook validation anchor | Driftfusion Sect. 5 | **glue** (run doped p-n, compare to Shockley) | medium | small |

## 3. Tier 1 — do now (spatial-charge focused), ranked

These lead with spatial charge / recombination / current-conservation because that is the stated focus, and every one is already extractable (status `now`/`glue`).

**1. Spatial current-conservation panel: Jn(x), Jp(x), Jion(x), Jdisp(x) and Σ=const.**
- *Shows:* each carrier's current at every mesh face, and that the total is flat across x at steady state.
- *New angle:* this is the cleanest possible demonstration of charge conservation — the defining invariant of any DD solver — and it visualizes *where* in the device current converts from electronic to ionic. None of the SolarLab figures to date show this; Verschraegen 2007 and Courtier EES treat the spatial current/field as the core diagnostic.
- *Recipe:* `x = multilayer_grid(electrical_layers(stack))`; `mat = build_material_arrays(x, stack)`; `y_ss = solve_illuminated_ss(x, stack, V_app)`; `cc = compute_current_components(x, y_ss, stack, V_app, mat=mat)` (`jv_sweep.py:202`). Plot `cc.J_n, cc.J_p, cc.J_ion, cc.J_total` (A/m², (N−1,) faces). Conservation metric: `std(cc.J_total)/mean(abs(cc.J_total)) → 0` (tested at `test_current_decomposition.py:44-49`).
- *Figure:* four stacked current-component traces vs depth at SC and near-MPP, with a residual subplot showing |J_total − ⟨J_total⟩| ≈ machine-zero.

**2. Mechanism-resolved recombination profile R(x) = R_SRH + R_rad + R_Auger + R_interface.**
- *Shows:* where carriers are lost and by which channel, under dark/light/bias.
- *New angle:* SolarLab currently *computes the decomposition and discards it* (`continuity.py:215-218` folds only the sum). Surfacing it directly answers the V_oc-gap root-cause story spatially (bulk vs interface SRH) — the exact diagnostic Courtier EES Fig 11 uses to find the optimal TL doping, and the mechanism Calado 2016 ties to hysteresis.
- *Recipe:* helper mirroring `extract_spatial_snapshot`: settle, get n,p,φ; `R_srh = srh_recombination(n,p,mat.ni_sq,mat.tau_n,mat.tau_p,mat.n1,mat.p1)`, `R_rad = radiative_recombination(...)`, `R_aug = auger_recombination(...)` (all `recombination.py:5-44`). For solver-exactness, apply the `het_recomb_despike` blend (`continuity.py:203-212`) before the bulk rates; add the interface channel via `_apply_interface_recombination` (`mol.py:1414`) converted to volumetric /dx_cell.
- *Figure:* stacked-area R(x) by channel on a log y-axis across the stack, at V=0 and V≈V_oc, light vs dark.

**3. Generation–recombination balance U(x) = G(x) − R(x).**
- *Shows:* the net source/sink term that the continuity equation integrates — where the device generates vs recombines.
- *New angle:* directly visualizes the collection problem; pairs the TMM optical profile with the loss profile in one frame. ∂PV and Driftfusion both anchor validation on G/U-type spatial balance.
- *Recipe:* `G = mat.G_optical` (TMM, `mol.py:383`) or `beer_lambert_generation(x, mat.alpha, stack.Phi)`; `R = total_recombination(n,p, mat.ni_sq,...)` (`recombination.py:47`); `U = G − R`. (Note: U ≠ exact dn/dt due to TE caps/despike, per audit caveat — label as bulk G−R.)
- *Figure:* G(x), R(x), and U(x) overlaid vs depth at MPP.

**4. Poisson self-consistency residual dE/dx − ρ/(ε₀ε_r) ≈ 0.**
- *Shows:* the electrostatic solver is internally consistent at every interior node.
- *New angle:* a credibility/verification figure — Driftfusion's depletion-region ρ/E/φ checks and Courtier's asymptotic-vs-numerical agreement are exactly this. Cheap insurance against reviewer skepticism.
- *Recipe:* from a `SpatialSnapshot` (φ, E, ρ) + `mat`, use the discretization-faithful form: `flux_face = C*(-(φ[1:]-φ[:-1]))` with `C = mat.poisson_factor.C`, node residual `= diff(flux_face) − ρ[1:-1]*mat.poisson_factor.h_cell` (`poisson.py:45-126`); expect ~1e-10 relative. Interior nodes only (Dirichlet boundaries).
- *Figure:* residual vs x (≈1e-10 line) plus the coarse continuum check showing the harmonic-mean interface artifact, to make the point honestly.

**5. Mobile-ion P(x,t) movie ↔ hysteresis.**
- *Shows:* vacancy redistribution to the contacts during a sweep, and the resulting forward/reverse J-V asymmetry.
- *New angle:* this is *the* signature physics of IonMonger / Calado 2016 / Courtier EES — ion-driven field screening as the hysteresis mechanism, shown spatially. SolarLab already produces every frame.
- *Recipe:* `run_jv_sweep(stack, v_rate=…, save_snapshots=True, decompose_currents=True)`; `snapshots_fwd[k].P` vs `snapshots_fwd[k].x` at bias `[k].V_app`, time `t_k = k·dt`; tie to `JVResult.hysteresis_index` and `decomp_fwd.J_ion` (`jv_sweep.py:177-199,787-808`). Sweep v_rate to show scan-rate dependence.
- *Figure:* P(x) frames at several biases (fwd vs rev) beside the hysteretic J-V, annotated with HI.

**6. Quasi-Fermi-level splitting (E_Fn−E_Fp)(x) vs bias.**
- *Shows:* QFL splitting growing toward qV in the absorber, with interface drops.
- *New angle:* converts the existing band diagram into a *carrier-collection* diagnostic — the QFL-gradient driving force Calado 2016 and Pauwels emphasize, and the direct visual of the DOS-fold QFL-step root cause from project memory.
- *Recipe:* loop `bd = compute_band_diagram(stack, V_app, illuminated=True)`; profile `bd.E_Fn − bd.E_Fp`; absorber-mean `np.nanmean((bd.E_Fn−bd.E_Fp)[mask])` (`band_diagram.py:108-114`, invariant tested at `test_band_diagram.py:34-42`).
- *Figure:* splitting profile vs depth at V = 0, 0.4, 0.8, ~V_oc.

**7. Hysteresis-factor vs scan-rate curve.**
- *Shows:* HF peaking at an intermediate scan rate.
- *New angle:* the headline result of Courtier EES (Figs 5/8/9) and IonMonger — a quantitative, falsifiable ionic-transport signature, not just a single hysteretic loop.
- *Recipe:* re-run `run_jv_sweep` at a log-spaced `v_rate` array; record `JVResult.hysteresis_index` per rate.
- *Figure:* HF vs log(scan rate), one curve per TL doping if desired.

## 4. Tier 2 — needs small new code

All capabilities below have the underlying physics in-solver; only a thin post-processor or field-add is missing.

- **Suns-Voc ideality factor n_eff + J_0.** Dark-JV ideality is already `now` (`dark_jv.py:35-59`). For Suns-Voc, run `run_suns_voc(stack, suns_levels=…)` and add `n_eff = polyfit(ln(suns), V_oc, 1)[0]/V_T` (`test_suns_voc.py:148-150`); promote to a `SunsVocResult.n_ideality` field. Enables the standard R_s-free ideality cross-check.
- **Equivalent-circuit / chemical-capacitance fit from impedance.** Nyquist/Bode and Mott-Schottky are `now`; the gap is a Randles/Voigt fitter. Add a `scipy.optimize.least_squares` post-processor over `ImpedanceResult.Z` with `Z = R_s + R_rec/(1+jωR_recC)` (form already in `test_impedance_randles.py`), extract C_μ from the low-f limit (`impedance.py:130`).
- **TPC and CE transients.** TPV is `now` (`tpv.py:206`). Clone it: TPC holds V_app=0 (skip the OC root-find), CE steps V_oc→0 and integrates ∫J dt. Reuse `run_transient`, the `dataclasses.replace(G_optical…)` pulse trick, and `_fit_decay_tau`. ~150-line `experiments/tpc_ce.py` + backend branch.
- **Voc(T) / E_A figure.** Already `now` (`voc_t.py:68`) — just needs a config with temperature-scaling active (FAST/FULL tier) to be physically meaningful, then plot V_oc vs T with the T→0 intercept = E_A.
- **EQE spectrum + integrated J_sc cross-check.** `now` for `*_tmm.yaml` configs (`eqe.py:193`); the "figure" is just calling `compute_eqe` and overlaying integrated J_sc on the J-V J_sc — matches Alshaikh/Kogo/Liu validation.
- **Steric ion Debye-layer saturation profile.** The Blakemore steric term is already in `J_ion` (`jv_sweep.py:268`); surfacing the saturated P(x) near a contact at high bias (IonMonger 2.0 test) is a plotting helper over existing snapshots.

## 5. Tier 3 — needs new physics (NOT yet capable)

- **Intra-band / thermionic-field-emission tunneling through a CB spike (Verschraegen 2007).** **SolarLab cannot do this today.** `thermionic_emission_flux` (`fe_operators.py:84-98`) is pure Richardson-Dushman over-the-barrier emission — the barrier term is `exp(−ΔE/V_T)` with no WKB/transmission factor; grep for `tunnel|wkb|tfe` returns zero solver hits. Required: a tunneling kernel (Padovani-Stratton E_00, or a WKB integral over the spike reconstructed from χ(x)/E_C(x) and local field), a new `m*_tun` per-layer YAML param, threading onto `MaterialArrays`, and a `use_tfe_tunneling` `SimulationMode` flag. The spike geometry inputs (E_C from band diagram, E and pileup from `SpatialSnapshot`) already exist — only the kernel is missing. **Effort: large.**
- **Continuously graded-bandgap absorber χ/Eg(x) (Burgelman-Marlein 2008).** **Not capable as-shipped, but no new transport physics needed.** The SG core already reads χ/Eg as per-node arrays and forms the grading quasi-field automatically (`continuity.py:132-142`); the blocker is that `mol.py:646-647` flat-broadcasts `chi[mask]=p.chi`. Required: new `Eg_grade_delta`/`chi_grade_delta` MaterialParams + loader plumbing, populate graded arrays mirroring the trap-profile pattern (`mol.py:701-726`), and critically recompute ni²(x)/n1(x)/p1(x) from the graded Eg. **Effort: medium** — sits at the glue/new-physics boundary; flag it as "build required, but solver-ready."
- **Non-Boltzmann statistical integrals (Fermi-Dirac / Gauss-Fermi, IonMonger 2.0).** Boltzmann-only today. Per project memory, FD *hurts* SCAPS parity (SCAPS is Boltzmann) and perovskite is non-degenerate — **deprioritize**; only an opt-in per-layer flag for degenerate contacts/TCO/tandem is justified. **Effort: medium, low value for current goal.**

## 6. Recommended immediate next 2 figures

**Figure A — Spatial current-conservation + mechanism-resolved recombination, side by side (Tier 1 #1 + #2).**
*Justification:* highest value-per-effort and dead-center on the stated spatial-charge focus. The current-conservation half is **trivial** (`compute_current_components` is public and tested — zero new physics, ~10 lines of plotting), and it delivers the single most rigorous solver-credibility statement (Σ J = const across x) that no SolarLab figure currently makes. The recombination half (**small** glue) surfaces a decomposition the solver *already computes and throws away* (`continuity.py:215-218`), and it directly visualizes the bulk-vs-interface SRH split that the project's own V_oc-gap root-cause analysis hinges on — answering "where is the loss" spatially, exactly as Courtier EES Fig 11 does. Together they convert the existing scalar SCAPS-parity story into a spatial one.

**Figure B — Ion P(x,t) redistribution beside the hysteretic J-V, with HF-vs-scan-rate inset (Tier 1 #5 + #7).**
*Justification:* this is the marquee perovskite-specific physics — ion-migration-driven hysteresis — and it is the dominant validation angle across *three* of the reference simulator papers (IonMonger, Calado 2016 Nat. Commun., Courtier EES). It is **trivial** to extract (`run_jv_sweep(save_snapshots=True)` already returns `snapshots_fwd/rev[k].P` and `hysteresis_index`), so it is near-zero marginal cost given the J-V infrastructure already in place. Showing P(x) screening the field during the sweep, tied quantitatively to HI and its scan-rate dependence, demonstrates that SolarLab reproduces the *mechanism* (not just the loop) — the strongest possible parity claim against the ionic-DD reference codes.

Report files referenced (all under `/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim/`): `perovskite_sim/experiments/jv_sweep.py`, `perovskite_sim/experiments/band_diagram.py`, `perovskite_sim/physics/recombination.py`, `perovskite_sim/physics/continuity.py`, `perovskite_sim/physics/poisson.py`, `perovskite_sim/solver/mol.py`, `perovskite_sim/experiments/eqe.py`, `perovskite_sim/experiments/mott_schottky.py`, `perovskite_sim/experiments/impedance.py`, `perovskite_sim/experiments/tpv.py`, `perovskite_sim/experiments/voc_t.py`, `perovskite_sim/discretization/fe_operators.py`.
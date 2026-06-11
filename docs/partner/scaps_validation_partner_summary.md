---
title: "Validation of the SolarLab Device Simulator against SCAPS-1D"
header-includes: |
  \usepackage{float}
  \makeatletter
  \let\oldfigure\figure
  \let\endoldfigure\endfigure
  \renewenvironment{figure}[1][]{\oldfigure[htbp]}{\endoldfigure}
  \makeatother
  \renewcommand{\topfraction}{0.95}
  \renewcommand{\bottomfraction}{0.95}
  \renewcommand{\textfraction}{0.05}
  \renewcommand{\floatpagefraction}{0.80}
  \setcounter{topnumber}{3}
  \setcounter{bottomnumber}{3}
  \setcounter{totalnumber}{4}
  \setlength{\intextsep}{8pt plus 2pt minus 2pt}
  \setlength{\floatsep}{8pt plus 2pt minus 2pt}
  \setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
---

# Summary

This report documents the validation of SolarLab — an in-house one-dimensional
device simulator coupling drift–diffusion transport, the Poisson equation, and
mobile-ion migration with transfer-matrix optics — against the established
reference solver SCAPS-1D. The test case is the partner's three-layer perovskite
Base Model, and the comparison spans the base current–voltage operating point
together with all eleven single-variable parameter sweeps recorded in the
partner workbook. This revision (2026-06-11) supersedes the 2026-05-29 report:
since then the dominant open-circuit-voltage discrepancy was root-caused to a
missing effective-density-of-states term in the heterojunction transport
discretisation, the correction was implemented and verified
(`dos_band_potentials`), and the validation protocol was refined (denser voltage
grid; a detailed-balance-ceiling validity guard on degenerate sweep points). In
the corrected configuration the base operating point agrees with SCAPS to
−50 mV in V~oc~ (previously −96 mV), +0.9 percentage points in FF, and −2 % in
J~sc~. Across the eleven sweeps, six match SCAPS (in direction and shape, or
flat in both solvers), two match partially, and three remain open; every
physically well-posed result satisfies the governing bounds. Notably, the
perovskite bulk-defect sweeps — previously flat — now reproduce the SCAPS
V~oc~/PCE descent in direction after a sweep-wiring defect was found and fixed
(Section 5). The remaining open items trace to one named residual — the
interface-recombination channel — and are documented rather than tuned away.

# 1. Introduction

SolarLab solves the coupled drift–diffusion and Poisson system for electrons,
holes, and mobile ionic species in one spatial dimension, using a
Scharfetter–Gummel discretization advanced in time by an implicit Radau
integrator, with optical generation supplied by a transfer-matrix (TMM) optical
stack. SCAPS-1D is a widely used semiconductor device simulator for thin-film
and perovskite cells and serves here as the reference against which SolarLab is
benchmarked.

The validation device is the partner Base Model: a hole-transport layer (spiro,
20 nm), a methylammonium lead iodide absorber (MAPbI~3~, 800 nm, bandgap
E~g~ = 1.53 eV), and an electron-transport layer (TiO~2~, 25 nm). All material
and defect parameters, and the reference sweep data, are taken from the partner
workbook `1R-Parameters.xlsx`.

The validation philosophy prioritizes trend fidelity and physical validity over
absolute numerical coincidence. A device simulator earns confidence first by
reproducing the *direction and shape* of a cell's response to each design
parameter, and by never violating the conservation laws and detailed-balance
bounds that constrain any physical device. Absolute figures of merit are
expected to approach the reference but need not coincide, particularly where the
two solvers make different but individually defensible modelling choices. This
report applies that standard throughout.

# 2. Methodology

SolarLab mirrors the SCAPS Base Model through the `scaps_mirror_v2.yaml` device
definition (partner layer stack, doping, mobilities, defect levels, optical
constants, and a glass front substrate completing the optical stack). Each of
the eleven single-variable sweeps varies one parameter across the workbook range
while holding all others fixed, and the figures of merit (V~oc~, J~sc~, FF, PCE)
are compared directly against the SCAPS values.

Two protocol elements were corrected since the previous revision. First, the
voltage grid: V~oc~ is extracted by interpolating the J–V zero crossing, and the
previous 20-point grid's linear interpolation across the exponential diode knee
under-read V~oc~ by 10–16 mV; the pipeline now uses a 40-point grid, verified
against a steady-state bisection. Second, a physical-validity guard: sweep
points whose extracted V~oc~ reaches the absorber's detailed-balance
(radiative-limit) ceiling — which no physical curve can cross — are flagged as
degenerate and excluded from trend statistics, while remaining visible in the
figures; this concerns the two lowest ETL-doping points, where the
essentially-undoped ETL cannot form a junction and the J–V collapses
(FF $\approx$ 0.3–0.6).

The headline configuration adds the `dos_band_potentials` transport correction
(Section 5): with Boltzmann statistics the heterostructure drift-diffusion
potentials carry effective-density-of-states terms ($V_T\ln N_C$,
$-V_T\ln N_V$) that the discretisation previously omitted, imposing a spurious
$kT\ln(\mathrm{DOS\ ratio})$ quasi-Fermi-level step at each heterojunction
(137 mV total on this stack). The correction is exact, parameter-free, and
verified against the analytic detailed-balance ceiling; the flag-off reference
results are tabulated in Appendix B.

Every simulation is checked against physical gates: J~sc~ below the
Shockley–Queisser limit for E~g~ = 1.53 eV; V~oc~ below the detailed-balance
ceiling; non-negative recombination at illuminated forward bias; and optical
balance R + T + A = 1.

# 3. Base operating point

| Metric | SolarLab 2026-05-29 | SolarLab corrected | SCAPS | Residual |
|---|---|---|---|---|
| V~oc~ (V) | 1.072 | **1.118** | 1.168 | −50 mV |
| J~sc~ (mA/cm^2^) | 25.73 | 25.70 | 26.28 | −2 % |
| FF (%) | 85.6 | **87.9** | 87.0 | +0.9 pp |
| PCE (%) | 23.6 | **25.26** | 26.69 | −1.4 pp |

Table 1. Base operating-point comparison on the partner Base Model.

The previous −96 mV V~oc~ shortfall decomposed into a 10–16 mV grid artifact, a
137 mV transport-discretisation omission, and compensating model differences;
with both fixes the residual is −50 mV, attributed to the interface
recombination channel (Section 5). The fill factor now matches SCAPS within a
point; the −2 % J~sc~ residual is the physical front-surface reflection that
SCAPS idealises away. The remaining PCE gap is the product of the V~oc~ and
J~sc~ residuals.

# 4. Sweep-by-sweep comparison

The eleven sweeps are shown as overlays of SolarLab (solid blue) against SCAPS
(dashed red), four figures of merit per panel, all in the corrected
configuration. Grouped by outcome:

**Matched (six).** The ETL/PVK conduction-band-offset sweep (Figure 1)
reproduces the SCAPS recombination cliff and recovery with 80 % of the SCAPS
V~oc~ range (734 of 918 mV). The PVK donor-doping sweep (Figure 11) — a
direction mismatch in the previous revision — now matches SCAPS in both
direction and magnitude (39 vs 34 mV, including the fill-factor and efficiency
collapse at the degenerate 10^18^ cm^−3^ point). The HTL/PVK interface
defect-density sweep (Figure 4) is near-flat in both solvers (0 vs 5 mV), and
the three defect-level sweeps that SCAPS holds flat — perovskite CB, perovskite
VB, and HTL/PVK (Figures 8–10) — are flat in SolarLab as well ($\le$ 0.3 mV both).

**Partial (two).** The PVK/ETL interface defect-density sweep (Figure 2) matches
in direction and saturating shape with 53 % of the SCAPS V~oc~ range (149 of
282 mV; 72 % with the transport correction off — see Section 5 for why the
correction trades interface-sweep closure for base-point accuracy). The
perovskite-VB bulk defect-density sweep (Figure 7) now matches SCAPS in
direction with 41 % of its range (4.4 of 10.8 mV) after the sweep-wiring fix
of Section 5.

**Open (three).** The ETL donor-doping sweep (Figure 5): the two degenerate
low-doping points are excluded by the validity guard, and on the well-posed arm
(10^14^–10^20^ cm^−3^) SolarLab is essentially flat (11 mV) where SCAPS rises by
100 mV — the direction is within noise of flat, so the trend is counted open
rather than matched. The PVK/ETL interface defect-level sweep (Figure 3)
responds only weakly (2 of 35 mV). The perovskite-CB bulk defect-density sweep
(Figure 6) now responds in the correct direction but covers only 11 % of the
SCAPS range (4.3 of 38.6 mV). All three open items share one mechanism
(Section 5).

![ETL/PVK conduction-band offset (ΔE~C~).](../figures/scaps_validation/sweep_CHI_ETL.png){width=86%}

![PVK/ETL interface defect density (N~t~).](../figures/scaps_validation/sweep_Nt_PVK_ETL.png){width=86%}

![PVK/ETL interface defect level (E~t~).](../figures/scaps_validation/sweep_Et_PVK_ETL.png){width=86%}

![HTL/PVK interface defect density (N~t~).](../figures/scaps_validation/sweep_Nt_HTL_PVK.png){width=86%}

![ETL donor doping (N~D~). Grey open markers: degenerate points excluded by the
detailed-balance-ceiling guard.](../figures/scaps_validation/sweep_Nd_ETL.png){width=86%}

![Perovskite conduction-band bulk defect density (N~t~).](../figures/scaps_validation/sweep_Nt_C_PVK.png){width=86%}

![Perovskite valence-band bulk defect density (N~t~).](../figures/scaps_validation/sweep_Nt_V_PVK.png){width=86%}

![Perovskite conduction-band bulk defect level (E~t~).](../figures/scaps_validation/sweep_Et_C_PVK.png){width=86%}

![Perovskite valence-band bulk defect level (E~t~).](../figures/scaps_validation/sweep_Et_V_PVK.png){width=86%}

![HTL/PVK interface defect level (E~t~).](../figures/scaps_validation/sweep_Et_HTL_PVK.png){width=86%}

![PVK donor doping (N~D,PVK~).](../figures/scaps_validation/sweep_Nd_PVK.png){width=86%}

```{=latex}
\clearpage
```

| # | Sweep | SolarLab range | SCAPS range | Status |
|---|---|---|---|---|
| 1 | ETL/PVK conduction-band offset | 734 mV | 918 mV | match (80 %) |
| 2 | PVK donor doping | 39 mV | 34 mV | match |
| 3 | HTL/PVK interface N~t~ | 0 mV | 5 mV | match (near-flat both) |
| 4 | PVK-CB bulk E~t~ | 0.3 mV | 0.4 mV | match (flat both) |
| 5 | PVK-VB bulk E~t~ | 0.3 mV | 0.4 mV | match (flat both) |
| 6 | HTL/PVK interface E~t~ | 0 mV | 0 mV | match (flat both) |
| 7 | PVK/ETL interface N~t~ | 149 mV | 282 mV | partial (53 %) |
| 8 | ETL donor doping | 11 mV | 100 mV | open (flat vs rising) |
| 9 | PVK/ETL interface E~t~ | 2 mV | 35 mV | open |
| 10 | PVK-VB bulk N~t~ | 4.4 mV | 11 mV | partial (41 %) |
| 11 | PVK-CB bulk N~t~ | 4.3 mV | 39 mV | open (direction match, 11 %) |

Table 2. Per-sweep scorecard, corrected configuration (V~oc~ ranges over the
physically well-posed points).

# 5. Analysis

**The resolved item: the base V~oc~ root cause.** The 2026-05-29 report carried
a −96 mV V~oc~ deficit of then-unknown mechanism. It has since been traced to
the transport discretisation: the Scharfetter–Gummel flux carried the band-edge
potentials but not the effective-density-of-states terms, imposing spurious
quasi-Fermi-level steps of $kT\ln 25 = 83$ meV (HTL/PVK) and $kT\ln 8 = 54$ meV
(PVK/ETL) — 137 mV total, matching the solver's measured quasi-Fermi-level
profile exactly and verified constructively (with the correction the
radiative-only configuration reaches its analytic detailed-balance ceiling,
1.2535 V). Two prior hypotheses — quasi-Fermi-level dissipation across the band
offsets and the contact boundary condition — were tested and refuted in the
process. The full account is in `SolarLab_SCAPS_gap_analysis_corrected.pdf`.

**The bulk-defect sweeps: a sweep-wiring defect, found and fixed.** The
previous revision reported the perovskite CB/VB bulk defect-density sweeps as
fully masked (0 mV). Most of that flatness was an artifact: the sweep machinery
ratio-scaled the bulk SRH lifetime against a hardcoded absolute reference of
10^16^ cm^−3^, while the configuration's base lifetime corresponds to its
declared 10^12^ cm^−3^ defects — so every swept point in the partner range ran
with a *longer* lifetime than the baseline and the recombination knob was never
actually turned. With the lifetime now ratio-scaled off the configuration's own
declared density, both sweeps reproduce the SCAPS V~oc~/PCE descent in
direction (VB: 41 % of the SCAPS range; CB: 11 %). The residual magnitude gap
is the genuine physics remainder: the swept trap is shallow (0.1 eV below the
conduction band, hence weakly recombining even in SCAPS), and SolarLab's higher
interface-recombination floor partially masks the response.

**The remaining residual: one interface channel.** With the transport
corrected, the −50 mV base residual and the remaining open sweeps reduce to the
interface-recombination channel. The PVK/ETL interface dominates SolarLab's
recombination at the corrected operating point; its strength (i) sets the −50 mV
base offset, (ii) absorbs the ETL-doping response that SCAPS expresses as a
+100 mV V~oc~ rise, (iii) partially masks the bulk-defect responses discussed
above, and (iv) damps the interface-E~t~ response. The interface-sweep closure also explains the one
regression in Table 2: lifting the 137 mV transport floor raises the V~oc~
baseline into a regime where the interface channel saturates differently, so the
PVK/ETL N~t~ closure moves from 72 % (correction off) to 53 % (on) — the
correction trades some interface-sweep range for base-point accuracy, with the
direction preserved. Matching SCAPS's interface behaviour exactly would require
adopting its interface-recombination formulation (two-sided carrier capture with
SCAPS's reference-density conventions), which is documented as future work, not
approximated by tuning.

**ETL low-doping regime.** At N~D~ $\le$ 10^12^ cm^−3^ the ETL is effectively an
insulator; SolarLab's transient solver cannot establish the steady state there
and its ideal-ohmic contact pins degenerate (the excluded grey points). A
SCAPS-style flat-band contact mode (`flat_band_contacts`) was implemented and
eliminates the unphysical pseudo-crossings; reproducing SCAPS's absolute V~oc~
in that regime additionally requires a direct steady-state solve, which is an
engine-level difference, documented and bounded.

**Short-circuit current (−2 %).** Front-surface reflection that SolarLab's
transfer-matrix optics retains and SCAPS idealises away; kept deliberately.

# 6. Conclusion and outlook

With the effective-DOS transport correction the base operating point agrees with
SCAPS to −50 mV in V~oc~ (from −96 mV), +0.9 pp in FF, and −2 % in J~sc~, and
six of the eleven parameter-sweep trends match with two partial. The three open
trends share a single named mechanism — the interface-recombination channel —
rather than being independent defects, and every physically well-posed result
satisfies the governing bounds. We recommend the corrected configuration
(`dos_band_potentials` on the SCAPS-mirror device) as the validated baseline.
The natural next step, if closer interface-sweep agreement is required, is to
adopt SCAPS's two-sided interface-recombination formulation; the groundwork
(interface-plane projection, flat-band contacts) is already in the codebase
behind explicit flags.

# Appendix A: Reproducibility

```
cd perovskite-sim
SOLARLAB_DOS_BAND=1 python scripts/run_scaps_full_regression.py \
    --out-dir outputs/full_dos_ON          # 10-sweep trend verdicts
SOLARLAB_DOS_BAND=1 python scripts/run_scaps_validation.py \
    --config configs/scaps_mirror_v2.yaml --out-dir outputs/dos_ON
SOLARLAB_DOS_BAND=1 python scripts/scaps_validation_figures.py \
    --out ../docs/figures/scaps_validation   # regenerate the overlays
```

Full technical detail is in `docs/scaps_validation_report.md`; the V~oc~
root-cause analysis is in `SolarLab_SCAPS_gap_analysis_corrected.pdf`.

# Appendix B: Reference results with the transport correction off

| Sweep | SolarLab range (off) | SCAPS range | Verdict (off) |
|---|---|---|---|
| ETL/PVK conduction-band offset | 780 mV | 918 mV | match (85 %) |
| PVK donor doping | 35 mV | 34 mV | direction mismatch at 10^18^ |
| PVK/ETL interface N~t~ | 203 mV | 282 mV | match (72 %) |
| PVK/ETL interface E~t~ | 8 mV | 35 mV | weak (22 %) |
| ETL donor doping | 23 mV | 100 mV | open (flat vs rising) |
| HTL/PVK interface N~t~ | 0 mV | 5 mV | near-flat both |
| bulk N~t~ (CB/VB) | 65 mV | 39/11 mV | over-responds (less masked) |
| bulk + HTL/PVK E~t~ | $\le$ 2 mV | $\le$ 0.4 mV | flat both |

Table 3. Flag-off reference (base V~oc~ 1.079 V at the corrected grid). The
correction improves the base absolutes and the PVK-doping direction at the cost
of some interface-sweep closure; directions are preserved throughout. Note the
bulk-N~t~ sweep *over*-responds with the transport correction off (65 mV vs the
SCAPS 39/11 mV): the lower flag-off V~oc~ baseline sits further from the
interface-recombination ceiling, so the bulk traps are less masked there — the
corrected-configuration 4 mV (Table 2) is the physically consistent value.

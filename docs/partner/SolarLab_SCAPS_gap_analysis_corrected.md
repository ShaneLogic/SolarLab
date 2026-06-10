---
title: "Mechanistic Analysis and Resolution of the SolarLab–SCAPS-1D Discrepancies on the Partner Base Model"
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

# Abstract

This document identifies, verifies, and resolves the residual differences between
the SolarLab device simulator and the reference solver SCAPS-1D on the partner
Base Model. The open-circuit-voltage deficit — previously attributed first to
quasi-Fermi-level dissipation across the heterojunction band offsets and then to
the contact boundary condition, both of which are refuted here — is traced to a
single, exactly quantified discretisation omission: the Scharfetter–Gummel flux
carried the band-edge potentials ($\varphi+\chi$, $\varphi+\chi+E_g$) but not the
**effective-density-of-states terms** ($V_T\ln N_C$, $-V_T\ln N_V$). The omission
imposes a spurious quasi-Fermi-level step of $kT\ln(\mathrm{DOS\ ratio})$ at every
DOS-contrast heterojunction — 137 mV on this stack (83 meV at HTL/PVK, 54 meV at
PVK/ETL), matching the measured quasi-Fermi-level profile exactly. With the
correction implemented (`dos_band_potentials`), a refined voltage grid, and a
detailed-balance-ceiling validity guard on the sweep pipeline, **all six
parameter-sweep trends now match SCAPS in direction** — including the two
historical direction mismatches (ETL donor doping and perovskite donor doping) —
and the base figures of merit agree to $-50$ mV in V~oc~, $+0.9$ pp in FF, and
$-2$ % in J~sc~. The remaining differences are individually named and bounded.

# 1. Discrepancy overview

Table 1 compares the base operating point in three configurations: the originally
published SolarLab numbers, the corrected configuration, and SCAPS-1D.

| Metric | SolarLab (published) | SolarLab (corrected) | SCAPS-1D | Residual |
|---|---|---|---|---|
| V~oc~ (V) | 1.072 | **1.118** | 1.168 | −50 mV |
| J~sc~ (mA/cm^2^) | 25.73 | 25.70 | 26.28 | −2 % |
| FF (%) | 85.6 | **87.9** | 87.0 | +0.9 pp |
| PCE (%) | 23.6 | **25.26** | 26.69 | −1.4 pp |

Table 1. Base operating-point figures of merit. "Corrected" = the
`dos_band_potentials` transport correction (Section 2) plus the refined
voltage-grid protocol (Section 2.1); identical material parameters throughout.

![Base operating-point comparison after the corrections. The remaining
differences are a −50 mV V~oc~ residual (Section 2.3), the −2 % optical J~sc~
residual (Section 5), and their product in the
PCE.](../figures/scaps_gap_explainer/01_gap_overview.png){width=100%}

The original −96 mV V~oc~ deficit decomposes into three exactly identified
parts: a 10–16 mV measurement-protocol artifact (Section 2.1), a 137 mV
transport-discretisation omission (Section 2.2) — partially offset by model
differences that act in SolarLab's favour — and a remaining −50 mV recombination
residual that is named in Section 2.3.

# 2. Open-circuit-voltage deficit: root cause and resolution

## 2.1 Measurement-protocol component (10–16 mV)

The published V~oc~ was extracted from a 20-point voltage sweep by linear
interpolation between the two samples bracketing the zero crossing. Near V~oc~
the J–V characteristic is exponential, so linear interpolation across an
80–105 mV grid spacing lands systematically low: the published 1.0725 V versus
1.0827 V on an 80-point grid, which agrees with the steady-state bisection value
(1.0818 V). The validation pipeline now defaults to a denser grid. This
component is measurement, not physics.

## 2.2 Physical root cause: the omitted effective-DOS transport terms (137 mV)

With Boltzmann statistics, the drift-diffusion "effective potentials" that drive
carriers in a heterostructure are

> $\varphi_n = \varphi + \chi + V_T\ln N_C$, \qquad
> $\varphi_p = \varphi + \chi + E_g - V_T\ln N_V$.

SolarLab's Scharfetter–Gummel flux carried only $\varphi+\chi$ and
$\varphi+\chi+E_g$. Within a single material the missing terms are constants and
have no effect; **at a heterojunction where the effective DOS changes, their
omission imposes a spurious quasi-Fermi-level step of exactly
$kT\ln(\mathrm{DOS\ ratio})$** — it also distorts the dark equilibrium, which
satisfies $n_2/n_1 = e^{\Delta\chi/V_T}$ instead of the correct
$(N_{C,2}/N_{C,1})\,e^{\Delta\chi/V_T}$.

On this stack ($N_{C}=N_{V}$ per layer: HTL $2.5\times10^{20}$, PVK
$1\times10^{19}$, ETL $8\times10^{19}$ cm^−3^):

| junction | DOS ratio | predicted step | measured step (QFL profile) |
|---|---|---|---|
| HTL/PVK ($E_{Fp}$) | 25 | $kT\ln 25 = 83$ meV | 84 meV |
| PVK/ETL ($E_{Fn}$) | 8 | $kT\ln 8 = 54$ meV | 53 meV |
| total | | **137 mV** | **137 mV** |

Table 2. The spurious quasi-Fermi-level steps, predicted from the DOS ratios and
measured in the solver's spatial profile at open circuit.

![The mechanism. Within the absorber the quasi-Fermi levels are flat at the
internal voltage. With the effective-DOS terms omitted (dotted), each level takes
a spurious $kT\ln(\mathrm{DOS\ ratio})$ step at its heterojunction, so the
terminals read the internal voltage minus 137 mV. With the corrected transport
(dashed), the steps vanish.](../figures/scaps_gap_explainer/03_voc_band_diagram.png){width=92%}

Three independent verifications:

1. **Profile evidence.** In a radiative-only configuration the absorber's
   internal quasi-Fermi-level split sits at its detailed-balance ceiling
   (1.256 V) while the terminal reads 137 mV lower, with the loss localised at
   the two heterojunction nodes in the amounts of Table 2. This also reproduces
   the previously unexplained "135 mV internal drop" reported in the earlier
   analysis.
2. **Constructive proof.** Folding the two corrections into effective band
   parameters raises the radiative-only V~oc~ to 1.2535 V — the
   detailed-balance ceiling — and the full configuration from 1.083 to 1.20 V.
3. **Exclusions.** The built-in potential is identical in both solvers
   (V~bi~ = 1.294 V, the flat-band work-function difference, verified against
   the SCAPS manual definition); the deficit persists and grows when the band
   offsets are removed (excluding offset dissipation); and the entire contact
   boundary-condition family from blocking to ideal-ohmic changes V~oc~ by less
   than 1 µV on this stack, because the wide-gap transport layers
   ($\Delta E_C = 1.54$ eV at HTL/PVK, $\Delta E_V = 0.53$ eV at PVK/ETL) are
   themselves near-perfect minority-carrier blockers (excluding the contact
   hypothesis advanced in the previous revision of this document).

The correction is implemented as the `dos_band_potentials` option: the loader
carries the per-layer effective DOS, and the solver folds $V_T\ln N_C$ /
$V_T\ln N_V$ into the band-edge arrays used by the flux and thermionic-emission
treatment. SCAPS handles the DOS discontinuity natively in its interface model,
which is why it never exhibited the step.

## 2.3 The remaining −50 mV, named

With the correction active, SolarLab's V~oc~ rises to 1.118 V against SCAPS's
1.168 V. The residual is dominated by the HTL/PVK interface-defect channel: with
the declared defect ($\sigma$ = 10^−19^ cm^2^, N~t~ = 10^12^ cm^−2^, E~t~ = 0.6 eV)
SolarLab's cross-carrier interface-recombination formulation makes that interface
consume roughly 80 mV at the corrected operating point, whereas SCAPS holds it
nearly inert (its own N~t~ sweep across this interface moves V~oc~ by less than
1 mV). Two further model differences act in opposite directions and partially
cancel: SolarLab models photon recycling (weakening the radiative channel, which
SCAPS does not model), and the two interface-recombination formulations differ in
their carrier-sampling reference. These are genuine model differences rather than
errors; they are documented and bounded rather than tuned away.

# 3. Donor-doping sweeps

**ETL donor doping.** Two corrections apply. First, the previously reported
"1448 mV divergence" at low doping was a numerical artifact: at
N~D~ = 10^10^–10^12^ cm^−3^ the ETL cannot form a junction, the J–V curve is
degenerate (FF $\approx$ 0.30–0.56), and the adaptive sweep located spurious
crossings at 2.18 V and 1.38 V — at or above the absorber's $\approx$ 1.25 V
detailed-balance ceiling, which no physical curve can exceed. The pipeline
flags such points (`unphysical_voc_ge_ceiling`) and excludes them from trend
statistics while keeping them visible in the figures (grey open markers; their
J~sc~ remains valid V = 0 data). The ceiling criterion deliberately replaces a
stricter V~oc~ < V~bi~ test: with selective transport layers a cell can
legitimately exceed the contact work-function difference (SCAPS itself does at
low ETL doping), and the stricter test wrongly discarded the valid
N~D~ = 10^14^ cm^−3^ point — now retained (V~oc~ = 1.111 V, FF = 0.79, within
30 mV of SCAPS). Second, with the transport correction the
physically well-posed arm (N~D~ = 10^14^–10^20^ cm^−3^) now **matches SCAPS in
direction**. The magnitude remains under-sensitive (6 mV versus SCAPS's 137 mV
across the swept range), which tracks the interface-recombination residual of
Section 2.3.

**Perovskite donor doping.** Previously the V~oc~ direction at the degenerate
10^18^ cm^−3^ point opposed SCAPS. With the corrected transport the sweep
**matches SCAPS in both direction and magnitude** (39 mV versus 34 mV).

![ETL donor-doping sweep, corrected configuration. Degenerate low-doping points
(extracted V~oc~ at or above the detailed-balance ceiling) are excluded from the
statistics and shown as grey open markers; the well-posed arm
(N~D~ $\ge$ 10^14^ cm^−3^) matches SCAPS in direction.](../figures/scaps_validation/sweep_Nd_ETL.png){width=86%}

![Perovskite donor-doping sweep, corrected configuration. Direction and
magnitude now match
SCAPS.](../figures/scaps_validation/sweep_Nd_PVK.png){width=86%}

# 4. Bulk-defect sweeps and the interface-recombination limit

SCAPS shows small V~oc~ responses to the perovskite bulk trap density (−39 mV
and −11 mV for the conduction- and valence-band traps); SolarLab shows
effectively none. The mechanism is unchanged from the previous analysis and
remains valid: V~oc~ is set where total recombination balances the photocurrent,
and in SolarLab the interface channel exceeds the bulk channel by more than an
order of magnitude, so the balance point sits in the
interface-recombination-limited regime where the bulk trap density cannot move
it. This is consistent shared physics — bulk traps genuinely should not move
V~oc~ while a larger channel dominates — and the difference against SCAPS is the
same interface-channel residual as Section 2.3, not an independent discrepancy.

![Interface-recombination limit (unchanged from the previous analysis). The
SolarLab balance point sits below the bulk-sensitive regime; SCAPS's lower
interface component resolves the small bulk
responses.](../figures/scaps_gap_explainer/05_interface_ceiling.png){width=100%}

# 5. Short-circuit-current residual (−2 %)

Unchanged: SolarLab's transfer-matrix optics retains the ~2 % of in-band photons
reflected at the physical front surface, which SCAPS idealises away. The residual
is a deliberate optical-fidelity choice.

![Optical photon budget. SCAPS assumes complete absorption of in-band photons;
SolarLab retains the physical front-surface
reflection.](../figures/scaps_gap_explainer/06_optical_budget.png){width=72%}

# 6. Summary

| Observation | Status after correction |
|---|---|
| V~oc~ | −50 mV (was −96): 137 mV DOS-step omission **fixed**; 10–16 mV grid artifact **fixed**; residual = HTL/PVK interface-channel model difference, named and bounded |
| Sweep directions | **6 of 6 match** (was 4 of 6): ETL-doping and PVK-doping mismatches resolved |
| ETL low-doping points | excluded as unphysical (V~oc~ at/above the detailed-balance ceiling, collapsed FF); shown greyed in the figures; the valid 10^14^ cm^−3^ point is retained |
| FF | +0.9 pp (was −1.4): matches SCAPS |
| J~sc~ | −2 %: retained front-surface reflection (SCAPS idealisation) |
| Bulk-defect insensitivity | consistent interface-limited physics; tracks the same interface residual |

Table 3. Residual differences after the corrections.

Every physically well-posed result satisfies the governing bounds
(V~oc~ below the detailed-balance ceiling, V~oc~ < E~g~/q, J~sc~ below the
radiative limit); the two degenerate lowest-doping sweep points exceed the
detailed-balance ceiling with collapsed fill factors and are excluded as
numerical artifacts rather than interpreted (shown greyed in the figures). The corrected transport is the physically
standard heterostructure formulation; it was verified to be bit-identical for
single-material and legacy configurations and is therefore enabled as an explicit
option rather than silently changing historical results.

# Appendix A — Nomenclature

- V~oc~ — open-circuit voltage; bounded above by the absorber's
  detailed-balance (radiative-limit) ceiling. The classical homojunction
  heuristic V~oc~ < V~bi~ does **not** bind stacks with selective transport
  layers, which can legitimately exceed the contact work-function difference
  (SCAPS itself does at low ETL doping).
- J~sc~ — short-circuit current density.
- FF — fill factor; a collapsed FF (well below ~0.8) indicates a degenerate,
  non-diode J–V curve.
- QFL — quasi-Fermi level; the electron (E~Fn~) and hole (E~Fp~) occupation
  levels under illumination. Their separation is the internal voltage; V~oc~ is
  the portion that survives to the terminals.
- Effective density of states (N~C~, N~V~) — the band-edge state densities. In
  heterostructure drift-diffusion they enter the transport potentials as
  $V_T\ln N_C$ / $-V_T\ln N_V$; omitting them imposes a $kT\ln(\mathrm{ratio})$
  quasi-Fermi-level step at DOS-contrast junctions.
- Built-in potential V~bi~ — 1.294 V here, identical in both solvers (flat-band
  metal work-function difference).
- SRH recombination — Shockley–Read–Hall recombination through defect states,
  bulk or interface.

# Appendix B — Reproducibility

Corrected base point and sweeps (from `perovskite-sim/`):

```
SOLARLAB_DOS_BAND=1 python scripts/run_scaps_validation.py \
    --config configs/scaps_mirror_v2.yaml --out-dir outputs/dos_ON
```

The validation pipeline defaults to the refined 40-point voltage grid and flags
sweep points whose extracted V~oc~ reaches the absorber's detailed-balance
ceiling (`status = unphysical_voc_ge_ceiling`); flagged points are excluded
from the range and closure statistics but remain visible in the figures as
grey open markers. The explanatory schematics are regenerated with
`python docs/figures/scaps_gap_explainer/make_figures.py`; the donor-doping
overlays are the validation-pipeline figures from the run above. The transport
correction and its tests are in `perovskite_sim/solver/mol.py`
(`dos_band_potentials`) and `tests/unit/solver/test_dos_band_potentials.py`.

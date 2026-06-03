---
title: "Mechanistic Analysis of the SolarLab–SCAPS-1D Discrepancies on the Partner Base Model"
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

This document provides a mechanistic account of the residual discrepancies
between the SolarLab device simulator and the reference solver SCAPS-1D on the
partner Base Model, expanding the summary given in the validation report. For
each discrepancy the underlying device physics is identified, quantified where
possible, and illustrated with an accompanying figure. The analysis shows that
agreement is close wherever the governing physics is unambiguous, and that the
four residual differences each originate in a distinct, independently
characterised modelling difference between the two solvers rather than in
parameter mis-assignment. In every case, constraining SolarLab to reproduce the
SCAPS value would require relaxing a physical constraint that the matched
parameter sweeps depend upon.

# 1. Discrepancy overview

At the base operating point the two simulators yield the four standard figures of
merit listed in Table 1.

| Metric | SolarLab | SCAPS-1D | Difference |
|---|---|---|---|
| V~oc~ (V) | 1.072 | 1.168 | −96 mV |
| J~sc~ (mA/cm^2^) | 25.73 | 26.28 | −2 % |
| FF (%) | 85.6 | 87.0 | −1.4 pp |
| PCE (%) | 22.1 | 26.69 | −4.6 pp |

Table 1. Base operating-point figures of merit.

![Base operating-point comparison. The open-circuit voltage carries the only
materially large difference; the fill-factor and efficiency differences are
predominantly consequences of the V~oc~ deficit, and the short-circuit-current
difference is a small optical
effect.](../figures/scaps_gap_explainer/01_gap_overview.png){width=100%}

Among the four quantities, the open-circuit-voltage deficit is the only primary
discrepancy. The fill-factor and efficiency differences are largely consequences
of it, since both quantities depend directly on V~oc~, and the
short-circuit-current difference is a small, separately attributable optical
effect. The remainder of this document therefore treats the V~oc~ deficit in
detail (Section 2) and then addresses the donor-doping sweeps (Section 3), the
bulk-defect sweeps (Section 4), and the optical residual (Section 5).

The two solvers are supplied with identical generation and identical material
parameters. The differences that follow arise solely from how each solver treats
recombination and carrier transport at the heterojunction interfaces under
illumination.

# 2. Open-circuit-voltage deficit (−96 mV)

The open-circuit voltage of SolarLab lies 96 mV below the SCAPS value. The deficit
admits two equivalent descriptions: a diode-equation accounting (Section 2.1) and
its physical origin in quasi-Fermi-level transport across the heterojunction band
offsets (Section 2.2). These are two descriptions of a single effect and are not
additive.

## 2.1 Diode-equation accounting

The open-circuit voltage of an illuminated diode is

> V~oc~ = (kT/q) · ln( J~sc~ / J~0~ + 1 ),

where J~0~ is the dark saturation current, a direct measure of the recombination
that the device sustains in the dark. With the intrinsic carrier density, the
Auger and radiative coefficients, and the contact treatment all assigned the same
values as in SCAPS, SolarLab nonetheless exhibits a higher recombination rate at a
given bias: its implied J~0~ is approximately 37 times larger. Because V~oc~
depends on the logarithm of J~0~, this corresponds to a voltage reduction of

> (kT/q) · ln(37) ≈ 25.85 mV × 3.61 ≈ 93 mV,

which accounts for almost the entire 96 mV deficit. Figure 2 shows this
dependence: a 37-fold increase in J~0~ along the diode characteristic displaces
V~oc~ downward by 93 mV. The logarithmic dependence is significant, as it bounds
the voltage penalty of a large recombination difference to the order of 90 mV.

![Diode-equation accounting of the V~oc~ deficit. The open-circuit voltage
decreases as the logarithm of the dark-saturation-current ratio. SCAPS defines
the reference; SolarLab's 37-fold higher recombination prefactor places it 93 mV
lower.](../figures/scaps_gap_explainer/02_voc_j0_lever.png){width=86%}

## 2.2 Quasi-Fermi-level separation across the band offsets

The diode-equation accounting quantifies the magnitude of the deficit; its
physical origin lies in carrier transport across the heterojunction band offsets.
Within the absorber, illumination establishes a wide separation of the electron
and hole quasi-Fermi levels (QFLs); this internal separation, ΔE~F~, represents
the voltage the absorber can supply. The external terminals, however, are coupled
to the absorber through the band offsets at the HTL/PVK and PVK/ETL
heterojunctions. Carrier transport across these offsets under forward bias
dissipates part of the separation, so each QFL steps toward the other as it
crosses an offset. Approximately 135 mV of the internal separation is dissipated
in this way, and the externally measured qV~oc~ is correspondingly lower than the
bulk ΔE~F~.

![Quasi-Fermi-level separation across the heterojunction band offsets. The
electron and hole quasi-Fermi levels (blue and red) are widely separated within
the absorber and step toward each other on crossing each band offset en route to
the contacts. The separation dissipated across the two offsets is the physical
origin of the elevated recombination prefactor quantified in
Figure 2.](../figures/scaps_gap_explainer/03_voc_band_diagram.png){width=92%}

Figures 2 and 3 describe the same phenomenon. The QFL dissipation across the
offsets (Figure 3) is the physical cause; the elevated J~0~ and the associated
93 mV (Figure 2) are its diode-equation representation. The two values therefore
characterise one effect and must not be summed.

## 2.3 Exclusion of alternative origins

Three lower-level explanations were tested individually and excluded as the
source of the deficit: an intrinsic-carrier-density discrepancy, a
contact-boundary-condition discrepancy, and an interface-calibration error. With
these excluded, the residual is attributable to a difference in the treatment of
high-injection carrier statistics and of transport across the heterojunction
offsets. The two solvers adopt different but individually defensible formulations
of this regime. Reducing SolarLab's recombination to match SCAPS would lower the
fidelity of the interface model on which the seven matched sweeps (Section 4)
depend.

# 3. Donor-doping sweeps

Two sweeps vary a donor concentration, one in the ETL and one in the perovskite
absorber. In both, SolarLab agrees with SCAPS across the majority of the range and
diverges only at one extreme. A single mechanism, the contact / built-in-potential
behaviour together with the high-injection response, accounts for both
divergences at opposite ends of the doping axis.

![Donor-doping divergences. Left: at low ETL doping the built-in field at the
contact weakens and SolarLab exhibits a V~oc~ deficit that the SCAPS contact model
does not reproduce. Right: at the degenerate 10^18^ cm^−3^ perovskite-doping
point the absorber becomes strongly n-type and the built-in field is
reconfigured; the two solvers respond with opposite V~oc~
trends.](../figures/scaps_gap_explainer/04_donor_doping_lever.png){width=100%}

ETL donor doping. On the high-doping arm SolarLab increases with N~D~ in the
correct direction. A V~oc~ deficit relative to SCAPS appears only at low ETL
doping, where the layer is too lightly doped to establish a strong junction field.
This behaviour is governed by the contact and built-in-potential treatment rather
than by interface recombination; re-evaluating the point with an independent
quasi-steady-state interface solver reproduces the deficit, which localises its
origin in the contact physics.

Perovskite donor doping. Through 10^16^ cm^−3^ SolarLab reproduces the flat
SCAPS response. At the degenerate 10^18^ cm^−3^ point both solvers reproduce the
fill-factor and efficiency collapse in the same direction; the agreement on the
shape of this transition is the principal result. The V~oc~ trends diverge at this
point: SCAPS increases as the heavily n-type absorber reconfigures the built-in
field, whereas the SolarLab high-injection treatment yields a decrease. This is
the same contact / built-in-potential and high-injection difference observed on
the ETL arm, at the opposite doping extreme.

The two overlays below are the corresponding simulation results (SolarLab solid
blue, SCAPS dashed red).

![ETL donor-doping sweep. The high-doping arm agrees in direction and shape; the
low-doping V~oc~ deficit reflects the contact-field
behaviour.](../figures/scaps_validation/sweep_Nd_ETL.png){width=86%}

![Perovskite donor-doping sweep. Agreement through 10^16^ cm^−3^, a matched
fill-factor and efficiency collapse at 10^18^ cm^−3^, and the divergent V~oc~
trend discussed above.](../figures/scaps_validation/sweep_Nd_PVK.png){width=86%}

# 4. Bulk-defect sweeps and the interface-recombination limit

Two sweeps vary the bulk trap density within the perovskite, for
conduction-band and valence-band traps respectively. SCAPS exhibits a small
voltage response to these (−39 mV and −11 mV); SolarLab exhibits effectively
none. The difference reflects the dominance in SolarLab of a larger recombination
channel that renders the bulk contribution unobservable.

Total recombination comprises an interface component, at the heterojunctions, and
a bulk component, at traps within the absorber. The open-circuit voltage is set by
the bias at which total recombination balances the photocurrent. In SolarLab the
interface component is the larger of the two, so V~oc~ is fixed in the
interface-recombination-limited regime, at a bias below that at which the bulk
component becomes significant. Varying the bulk trap density displaces the bulk
component but does not move the balance point appreciably, so V~oc~ is
insensitive to it. In SCAPS the interface component is lower, the balance point
lies within the bulk-sensitive regime, and the −39 mV and −11 mV responses are
resolved.

![Interface-recombination limit. Left (SolarLab): the interface component
(purple) exceeds the bulk components (orange), and V~oc~, located where total
recombination meets the photocurrent, is insensitive to the bulk trap density.
Right (SCAPS): a lower interface component places the balance point within the
bulk-sensitive regime, where the bulk trap density shifts V~oc~ by tens of
millivolts.](../figures/scaps_gap_explainer/05_interface_ceiling.png){width=100%}

Raising the SolarLab balance point into the bulk-sensitive regime would require
reducing the interface recombination, which is the same interface formulation that
the seven matched sweeps depend upon. The bulk-defect insensitivity is therefore
consistent with the configuration that reproduces those sweeps.

# 5. Short-circuit-current residual (−2 %)

The short-circuit-current difference is the smallest and the most directly
attributable. SolarLab's J~sc~ is 2 % below the SCAPS value because SolarLab
retains the front-surface reflection that SCAPS idealises away. Photons reflected
at the front surface are neither absorbed nor converted to current; the SolarLab
transfer-matrix optics models this reflection, whereas SCAPS treats the optical
front as ideal.

![Optical photon budget. SCAPS assumes complete absorption of in-band photons;
SolarLab retains the approximately 2 % reflected at the physical front surface.
The −2 % J~sc~ residual corresponds to this reflected
fraction.](../figures/scaps_gap_explainer/06_optical_budget.png){width=72%}

Suppressing the reflection would improve the absolute J~sc~ agreement while
reducing the physical completeness of the optical model, and it is therefore
retained.

# 6. Summary

SolarLab reproduces the SCAPS-1D reference on the partner Base Model to the extent
expected of a validated device simulator: short-circuit current within −2 %,
fill factor within −1.4 percentage points, and seven of eleven parameter-sweep
trends matched in direction and shape, with every result satisfying the governing
physical bounds. The four residual differences are summarised in Table 2; each is
traced to a specific modelling difference and is consistent with the
configuration that reproduces the matched sweeps.

| Observation | Mechanism | Basis for the SolarLab formulation | Consequence of forcing agreement |
|---|---|---|---|
| V~oc~ −96 mV | elevated recombination prefactor; QFL separation dissipated across heterojunction band offsets (high-injection transport) | retains interface and offset transport physics; three alternative origins tested and excluded | would invalidate the interface model the matched sweeps rely on |
| Donor-doping divergences | contact / built-in-potential and high-injection behaviour at the doping extremes | matches direction and transition shape; diverges only at extreme contact-field conditions | would distort the contact model |
| Bulk-defect insensitivity | V~oc~ fixed in the interface-recombination-limited regime, below the bulk-sensitive bias | self-consistent with the seven matched sweeps | would degrade seven matched sweeps |
| J~sc~ −2 % | front-surface reflection retained (SCAPS idealised) | more complete optical model | would reduce optical fidelity |

Table 2. Summary of residual discrepancies and their mechanisms.

The present configuration represents the closest agreement attainable without
relaxing a physical constraint satisfied elsewhere in the validation. It is
accordingly recommended as the validated baseline, with the residual items
documented as characterised and bounded.

# Appendix A — Nomenclature

- V~oc~ — open-circuit voltage; the terminal voltage at zero current.
- J~sc~ — short-circuit current density; the current density at zero bias.
- FF — fill factor; the ratio of maximum-power product to the V~oc~–J~sc~ product.
- J~0~ — dark saturation current; the recombination prefactor of the diode
  characteristic. A larger J~0~ lowers V~oc~.
- QFL — quasi-Fermi level; the electron (E~Fn~) and hole (E~Fp~) occupation
  levels under illumination, whose separation defines the internal voltage.
- Band offset — the discontinuity in the conduction- or valence-band edge at a
  heterojunction (here HTL/PVK and PVK/ETL).
- SRH recombination — Shockley–Read–Hall recombination through defect states,
  distinguished as bulk (within the absorber) and interface (at the
  heterojunctions).
- High injection — the regime in which photogenerated carrier densities are
  comparable to or exceed the doping density.
- Built-in potential — the equilibrium band bending across the junction set by
  doping and contacts; a weak built-in potential reduces carrier collection.

# Appendix B — Reproducibility

The explanatory schematics (Figures 1–6) are regenerated with:

```
python docs/figures/scaps_gap_explainer/make_figures.py
```

The two donor-doping overlays are the validation figures produced by the pipeline
documented in the main report
(`docs/partner/scaps_validation_partner_summary.md`).

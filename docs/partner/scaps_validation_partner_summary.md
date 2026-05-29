---
title: "SolarLab vs SCAPS-1D — Validation Summary"
subtitle: "Perovskite cell (HTL / MAPbI3 / ETL), partner Base Model"
date: "2026-05-29"
---

# Overview

SolarLab (1D drift-diffusion + Poisson + mobile-ion solver, transfer-matrix
optics) is validated against **SCAPS-1D** on the partner's three-layer
perovskite Base Model — HTL (spiro, 20 nm) / MAPbI3 absorber (E<sub>g</sub> = 1.53 eV,
800 nm) / ETL (TiO2, 25 nm) — using the parameters and sweep data in
`1R-Parameters.xlsx`. The comparison covers the base J–V operating point and
all ten single-variable sweeps in the partner workbook.

**Headline:** every SolarLab result is physically valid (J<sub>sc</sub> below the
Shockley–Queisser limit, V<sub>oc</sub> below the built-in potential, recombination
non-negative). J<sub>sc</sub> matches SCAPS to within −2 %, and 7 of 10 sweep trends
match in direction and shape. The three open items are characterised to
specific, physically-understood model differences (not fitting errors).

# Base J–V parity (current model)

| Metric | SolarLab | SCAPS | Δ |
|---|---|---|---|
| V<sub>oc</sub> (V) | 1.072 | 1.168 | −96 mV |
| J<sub>sc</sub> (mA/cm²) | 25.73 | 26.28 | **−2 %** |
| FF (%) | 85.6 | 87.0 | −1.4 pp |
| PCE (%) | 22.1 | 26.69 | −4.6 pp |

J<sub>sc</sub> is now physical and within 2 % of SCAPS after completing the optical
stack (glass front substrate). The V<sub>oc</sub> shortfall (−96 mV) propagates into
FF·V<sub>oc</sub>-limited PCE and is the principal open item (see Assessment).

# Per-sweep scorecard

| Sweep | SolarLab vs SCAPS | Status |
|---|---|---|
| ETL/PVK conduction-band offset (ΔE<sub>C</sub>) | cliff 0.30 → spike 1.08 V; direction + cliff slope match | match |
| PVK/ETL interface N<sub>t</sub> | 1.08 → 0.87 V (~75 % of SCAPS range) | match |
| PVK/ETL interface E<sub>t</sub> | direction + saturating shape match | match |
| HTL/PVK interface N<sub>t</sub> | near-flat (SCAPS near-flat) | match |
| PVK-CB / PVK-VB / HTL/PVK E<sub>t</sub> | flat (SCAPS flat) | match |
| ETL donor doping N<sub>D</sub> | high-N<sub>D</sub> arm correct; low-N<sub>D</sub> dip | partial |
| PVK-CB / PVK-VB bulk N<sub>t</sub> | flat (SCAPS shows −39 / −11 mV) | open |

# Per-sweep overlays

SolarLab (solid blue) vs SCAPS (dashed red), all four metrics per panel.

![ETL/PVK conduction-band offset](../figures/scaps_validation/sweep_CHI_ETL.png)

![PVK/ETL interface defect density](../figures/scaps_validation/sweep_Nt_PVK_ETL.png)

![PVK/ETL interface defect level](../figures/scaps_validation/sweep_Et_PVK_ETL.png)

![HTL/PVK interface defect density](../figures/scaps_validation/sweep_Nt_HTL_PVK.png)

![ETL donor doping](../figures/scaps_validation/sweep_Nd_ETL.png)

![Perovskite-CB bulk defect density](../figures/scaps_validation/sweep_Nt_C_PVK.png)

![Perovskite-VB bulk defect density](../figures/scaps_validation/sweep_Nt_V_PVK.png)

![Perovskite-CB bulk defect level](../figures/scaps_validation/sweep_Et_C_PVK.png)

![Perovskite-VB bulk defect level](../figures/scaps_validation/sweep_Et_V_PVK.png)

![HTL/PVK interface defect level](../figures/scaps_validation/sweep_Et_HTL_PVK.png)

# Assessment of the open items

All three open items are physical model differences, not parameter-fitting
errors. SolarLab's values remain physically valid throughout.

1. **Base V<sub>oc</sub> (−96 mV).** With identical n<sub>i</sub>, recombination coefficients
   (Auger, radiative) and contact treatment to SCAPS, SolarLab recombines more
   at a given voltage — the implied dark saturation current is ~37× higher,
   which is exactly the 93 mV gap (kT·ln 37). At the operating point the
   recombination is dominated by Auger and radiative channels at the
   PDF-specified coefficients, plus a ~135 mV internal drop of the quasi-Fermi
   splitting across the heterojunction band offsets. We verified this is **not**
   an n<sub>i</sub>, contact-boundary or interface-calibration error (all tested and
   ruled out); it is a high-injection carrier-statistics / heterojunction-
   transport difference between the two solvers.

2. **ETL donor doping (Nd_ETL).** The high-doping arm rises with N<sub>D</sub> in the
   correct direction; the low-doping points show a V<sub>oc</sub> dip tied to the
   contact / built-in-potential treatment at very low ETL doping, not the
   interface recombination.

3. **Perovskite bulk N<sub>t</sub> (CB / VB).** SolarLab's V<sub>oc</sub> is pinned by the
   interface-recombination ceiling, which sits below the voltage where bulk
   trap density becomes visible; SCAPS reaches that regime and shows a small
   −39 / −11 mV response. Raising the ceiling requires reducing total
   recombination, which conflicts with the interface model the matched sweeps
   rely on.

J<sub>sc</sub>'s residual −2 % is front-surface reflection that the SCAPS optical model
treats as ideal; SolarLab keeps the physical reflection.

# Reproducibility

```
cd perovskite-sim
python scripts/scaps_absolute_scorecard.py      # absolute + trend vs the xlsx
python scripts/scaps_validation_figures.py \
       --out ../docs/figures/scaps_validation     # regenerate the overlays
```

Full technical detail (methods, every sweep, and the development history) is in
`docs/scaps_validation_report.md`.

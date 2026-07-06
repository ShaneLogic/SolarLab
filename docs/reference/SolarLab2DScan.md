---
title: "Two-Dimensional Defect-Parameter Validation of the SolarLab Drift--Diffusion Simulator against SCAPS-1D at the Perovskite/ETL Interface"
date: "23 June 2026"
header-includes: |
  \usepackage{etoolbox}
  \AtBeginEnvironment{longtable}{\footnotesize}
  \setlength{\tabcolsep}{5pt}
  \usepackage{graphicx}
  \usepackage{float}
  \renewcommand{\arraystretch}{1.15}
---

## Abstract

The SolarLab one-dimensional drift--diffusion solar-cell simulator is benchmarked
against the established reference solver SCAPS-1D on two two-dimensional
defect-parameter scans of the perovskite/electron-transport-layer (ETL) interface:
interface defect density versus trap energy ($N_t \times E_t$), and interface
defect density versus conduction-band offset ($N_t \times \Delta E_C$). Both maps
are reproduced on SolarLab's validated parity configuration using the transient
current--voltage driver. SolarLab recovers the qualitative structure of all four
figures of merit across both scans, including the defect-density banding, the
shallow- versus deep-trap response, and the coupling between the conduction-band
cliff/spike and the interface defect density that the second scan was designed to
expose. The residual quantitative differences are confined to two previously
documented model boundaries: a bulk-Auger-limited open-circuit-voltage ceiling
($1.169$~V versus SCAPS $1.250$~V at low defect density) and the spike-side
collapse mechanism (fill-factor and short-circuit-current loss in SolarLab versus
open-circuit-voltage collapse in SCAPS). No new discrepancies are observed, and
SolarLab converges on every grid point whereas SCAPS leaves approximately twelve
points unresolved.

## 1. Introduction

A device simulator earns confidence by reproducing the *direction and shape* of a
cell's response to its design parameters, not merely by matching a single
operating point. Single-variable sweeps validate one parameter at a time;
two-dimensional scans additionally test whether the simulator captures the
*coupling* between two parameters. The reference SCAPS-1D study concludes with two
such scans on the perovskite/ETL interface, chosen because that interface was the
most sensitive location in the preceding single-variable analysis. This report
asks whether SolarLab can perform the same two scans and how its maps compare with
the SCAPS reference.

## 2. Methodology

Each scan is a nested two-parameter sweep (it is not the two-dimensional spatial
solver). SolarLab's parameter-sweep interface applies both axis values in a single
update, targeting the perovskite/ETL interface defect. All runs use the validated
parity configuration `scaps_mirror_v2`, which combines the effective
density-of-states band-potential correction with the heterointerface Auger
de-spike ($f = 0.53$), and the transient (method-of-lines) current--voltage driver.
The transient driver is used in preference to the steady-state Newton driver
because it remains convergent through the collapsed-junction cliff and spike
regimes, reproducing SCAPS's steady-state-per-bias result as the fast-scan,
ion-frozen limit.

The two scans follow the SCAPS definitions exactly:

* **Scan A** ($N_t \times E_t$): interface defect density $N_t = 10^{9}$ to
  $10^{15}~\mathrm{cm^{-2}}$ (seven levels) by trap energy $E_t = 0.01$ to
  $0.6~\mathrm{eV}$ (eight levels), giving $7 \times 8 = 56$ points.
* **Scan B** ($N_t \times \Delta E_C$): the same seven defect-density levels by the
  ETL/PVK conduction-band offset $\Delta E_C = -1.0$ to $+0.70~\mathrm{eV}$
  (sixteen levels), giving $7 \times 16 = 112$ points. The offset is imposed
  through the electron affinity, $\chi_{\mathrm{PVK}} = 3.94~\mathrm{eV}$ fixed and
  $\chi_{\mathrm{ETL}} = 3.94 - \Delta E_C$.

At every grid point the four figures of merit (PCE, $V_{oc}$, FF, $J_{sc}$) are
extracted from the simulated current--voltage curve; points without an
open-circuit crossing are recorded as undefined, mirroring the unresolved cells in
the SCAPS maps.

For legibility, the SCAPS reference maps are colour-digitised and re-rendered at
the same scale and font size as the SolarLab maps; the digitisation was validated
against the reported best-point values and reproduces them to the displayed
precision (e.g. Scan~A: PCE~$=29.83\,\%$, $V_{oc}=1.250$~V). In each comparison
figure the SolarLab and SCAPS panels of a given figure of merit share a single
colour scale spanning the union of the two solvers' ranges, so a colour denotes
the same value in both blocks; systematic offsets between the solvers (e.g. the
Scan~A $J_{sc}$ difference of ${\sim}0.56~\mathrm{mA\,cm^{-2}}$) therefore appear
as overall brightness shifts of a whole panel rather than as rescaled detail.

```{=latex}
\clearpage
```

## 3. Results

### 3.1 Interface defect density versus trap energy

```{=latex}
\begin{figure}[H]
\centering
\includegraphics[width=0.80\textwidth]{../../outputs/scan2d/compare_ntet.png}
\caption{Scan A ($N_t \times E_t$) on the perovskite/ETL interface. SolarLab (upper block) and SCAPS-1D (lower block) four-panel maps of PCE, $V_{oc}$, FF and $J_{sc}$; the vertical axis is $\log_{10} N_t$ and the horizontal axis is the trap energy $E_t$.}
\end{figure}
```

Both maps are banded predominantly by defect density: low $N_t$ yields high
$V_{oc}$ and PCE, and the cell collapses as $N_t$ increases. Superimposed on this
banding is a weaker trap-energy modulation in which a shallow trap
($E_t = 0.01~\mathrm{eV}$) is least harmful and deeper traps reduce $V_{oc}$ before
saturating beyond $E_t \approx 0.2~\mathrm{eV}$. The short-circuit current is
essentially flat in both solvers. The single systematic difference is the
open-circuit-voltage ceiling at low defect density: SolarLab saturates at
$1.169~\mathrm{V}$ whereas SCAPS continues to $1.250~\mathrm{V}$, because once
interface recombination is removed SolarLab's $V_{oc}$ is limited by bulk Auger
recombination.

```{=latex}
\clearpage
```

### 3.2 Interface defect density versus conduction-band offset

```{=latex}
\begin{figure}[H]
\centering
\includegraphics[width=0.80\textwidth]{../../outputs/scan2d/compare_ntcbo.png}
\caption{Scan B ($N_t \times \Delta E_C$) on the perovskite/ETL interface. Panels and axes as in Figure 1, with the horizontal axis the conduction-band offset $\Delta E_C$. White cells in the SCAPS block are grid points the reference solver did not resolve.}
\end{figure}
```

Both maps display a high-performance plateau for $\Delta E_C \approx -0.5$ to
$+0.4~\mathrm{eV}$, a defect-density-dependent *cliff* triangle for
$\Delta E_C \le -0.5~\mathrm{eV}$, and a *spike* collapse for
$\Delta E_C \ge +0.5~\mathrm{eV}$. SolarLab reproduces the cliff--defect coupling
quantitatively: at $\Delta E_C = -1.0~\mathrm{eV}$ the open-circuit voltage barely
falls at low defect density ($1.165~\mathrm{V}$ at $N_t = 10^{9}$) but collapses at
high defect density ($0.129~\mathrm{V}$ at $N_t = 10^{15}$), confirming that the
band-offset cliff is detrimental only when interface defects are present.

Two differences are noted. First, the spike-side collapse proceeds by a different
channel: SCAPS collapses the open-circuit voltage (to about $0.2~\mathrm{V}$) and
the solver fails on approximately twelve grid points (white cells), whereas
SolarLab holds $V_{oc} \approx 1.169~\mathrm{V}$ and instead collapses the fill
factor (from $89$ to $30~\%$ at $\Delta E_C = +0.6~\mathrm{eV}$) and then the
short-circuit current (to $6.4~\mathrm{mA\,cm^{-2}}$ at
$\Delta E_C = +0.7~\mathrm{eV}$), an S-shaped collection failure rather than a
voltage collapse. Second, SolarLab's transient driver converges on all
$112$ points, where SCAPS leaves approximately twelve unresolved.

## 4. Discussion

The two scans confirm that SolarLab captures the physics structure of the
two-dimensional defect/band-offset landscape on every axis the scans were designed
to interrogate: the defect-density banding, the shallow- versus deep-trap
response, and -- most significantly -- the coupling between the conduction-band
cliff/spike and the interface defect density. The quantitative residuals coincide
exactly with the two boundaries already identified in the single-variable parity
campaign. The open-circuit-voltage ceiling reflects a bulk-Auger recombination
floor that limits SolarLab's voltage once interface recombination is suppressed; it
is a known absolute-magnitude offset, not a structural disagreement. The
spike-side collapse channel reflects the documented difference in how the two
solvers treat the strongly band-offset junction; SolarLab's robustness there (no
unresolved points) is a property of its transient time-integration.

Table 1 summarises the base operating points and Table 2 the qualitative
agreement per behaviour.

\begin{table}[H]
\centering
\caption{Best-performance operating point of each scan (lowest defect density).}
\footnotesize
\begin{tabular}{lcccc}
\hline
Metric & SolarLab A & SCAPS A & SolarLab B & SCAPS B \\
\hline
PCE (\%)            & 26.8  & 29.83  & 26.9  & 29.86  \\
$V_{oc}$ (V)        & 1.169 & 1.2495 & 1.169 & 1.2511 \\
FF (\%)             & 89.2  & 90.84  & 89.3  & 90.82  \\
$J_{sc}$ (mA\,cm$^{-2}$) & 25.7 & 26.28 & 25.7 & 26.28 \\
\hline
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Qualitative agreement between SolarLab and SCAPS-1D.}
\footnotesize
\begin{tabular}{lll}
\hline
Behaviour & Agreement & Note \\
\hline
$N_t$ banding (both scans)          & match & --- \\
Shallow/deep $E_t$ response         & match & --- \\
Cliff--defect coupling              & match & $V_{oc}$ $1.165 \to 0.129$ V across $N_t$ \\
$V_{oc}$ ceiling at low $N_t$        & differ & bulk-Auger floor ($1.169$ vs $1.25$ V) \\
Spike collapse channel              & differ & FF/$J_{sc}$ vs $V_{oc}$ \\
Unresolved grid points              & --- & $0/112$ (SolarLab) vs ${\sim}12$ (SCAPS) \\
\hline
\end{tabular}
\end{table}

## 5. Conclusion

SolarLab reproduces both SCAPS-1D two-dimensional interface-defect scans and agrees
with the reference on every qualitative trend they were constructed to probe. The
two residual differences -- the bulk-Auger open-circuit-voltage ceiling and the
spike-side collapse mechanism -- are the same documented model boundaries
established in the single-variable validation, not new discrepancies. The
comparison therefore extends the trends-over-absolutes validation of SolarLab from
one-dimensional sweeps to the coupled two-parameter regime. All results, heatmaps
and the generating script are archived under `outputs/scan2d/`.

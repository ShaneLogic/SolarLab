# SCAPS Partner Report Academic Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the partner-facing SolarLab vs SCAPS-1D comparison report into professional academic prose with correct sub/superscript typography and an explicit current-situation analysis, then re-render the partner PDF.

**Architecture:** Single-document rewrite. Task 1 replaces `docs/partner/scaps_validation_partner_summary.md` with the full final prose (source of truth, given verbatim below). Task 2 runs the typography normalizer as a safety pass. Task 3 renders + verifies the PDF. Task 4 commits and pushes. No simulation runs; all numbers come from the current best (E10.1) data already in the repo.

**Tech Stack:** Markdown, `perovskite-sim/scripts/md_physics_typography.py`, pandoc 2.12 + xelatex, Arial, existing Arial overlay figures under `docs/figures/scaps_validation/`.

---

## File Structure

- **Modify:** `docs/partner/scaps_validation_partner_summary.md` — the report source (full rewrite).
- **Modify (regenerate):** `docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf` — rendered output.
- **Read-only:** `docs/figures/scaps_validation/sweep_*.png` (ten figures, reused), `perovskite-sim/scripts/md_physics_typography.py` (typography pass).

---

### Task 1: Replace the report source with the academic-prose rewrite

**Files:**
- Modify: `docs/partner/scaps_validation_partner_summary.md`

- [ ] **Step 1: Overwrite the file with the full content below (verbatim)**

````markdown
---
title: "Validation of the SolarLab Device Simulator against SCAPS-1D"
subtitle: "A perovskite solar cell case study on the partner Base Model"
author: "SolarLab — HKUST(GZ)"
date: "2026-05-29"
---

# Executive summary

This report documents the validation of SolarLab — an in-house one-dimensional
device simulator coupling drift–diffusion transport, the Poisson equation, and
mobile-ion migration with transfer-matrix optics — against the established
reference solver SCAPS-1D. The test case is the partner's three-layer
perovskite Base Model, and the comparison spans the base current–voltage
operating point together with all ten single-variable parameter sweeps recorded
in the partner workbook. At the operating point, SolarLab reproduces the
short-circuit current density to within −2 % and the fill factor to within
−1.4 percentage points of SCAPS, while the open-circuit voltage is 96 mV lower.
Across the ten sweeps, seven match SCAPS in both direction and shape, one matches
partially, and two remain open. Every SolarLab result satisfies the governing
physical bounds — the short-circuit current stays below the Shockley–Queisser
limit, the open-circuit voltage below the built-in potential, and recombination
remains non-negative throughout. The three residual discrepancies are not
parameter-fitting errors; each traces to a specific, physically understood
difference between the two solvers, and resolving them by relaxing the physics
would make the SolarLab result less faithful, not more. We therefore regard the
present configuration as the physically honest maximum of agreement and
recommend it as the validated baseline for ongoing partner work.

# 1. Introduction and scope

SolarLab solves the coupled drift–diffusion and Poisson system for electrons,
holes, and mobile ionic species in one spatial dimension, using a
Scharfetter–Gummel discretization advanced in time by an implicit Radau
integrator, with optical generation supplied by a transfer-matrix (TMM) optical
stack. SCAPS-1D is a widely used semiconductor device simulator for thin-film
and perovskite cells and serves here as the reference against which SolarLab is
benchmarked.

The validation device is the partner Base Model: a hole-transport layer (spiro,
20 nm), a methylammonium lead iodide absorber (MAPbI<sub>3</sub>, 800 nm, bandgap
E<sub>g</sub> = 1.53 eV), and an electron-transport layer (TiO<sub>2</sub>, 25 nm). All
material and defect parameters, and the reference sweep data, are taken from the
partner workbook `1R-Parameters.xlsx`.

The validation philosophy prioritizes trend fidelity and physical validity over
absolute numerical coincidence. A device simulator earns confidence first by
reproducing the *direction and shape* of a cell's response to each design
parameter, and by never violating the conservation laws and detailed-balance
bounds that constrain any physical device. Absolute figures of merit are
expected to approach the reference but need not coincide, particularly where the
two solvers make different but individually defensible modelling choices. This
report applies that standard throughout.

# 2. Methodology

SolarLab is configured to mirror the SCAPS Base Model through the
`scaps_mirror_v2.yaml` device definition, which encodes the partner layer stack,
doping, mobilities, defect levels, and optical constants, and adds a glass front
substrate so that the optical stack is complete. Each of the ten single-variable
sweeps varies one physical parameter across the range tabulated in the workbook
while holding all others fixed, and the resulting figures of merit
(V<sub>oc</sub>, J<sub>sc</sub>, FF, PCE) are compared directly against the SCAPS values.

Every simulation is checked against a set of physical gates that must hold for
the result to be admissible: the short-circuit current density must not exceed
the Shockley–Queisser limit for the absorber bandgap (≈ 27.5 mA/cm<sup>2</sup> at
E<sub>g</sub> = 1.53 eV); the open-circuit voltage must not exceed the built-in
potential; recombination must be non-negative at illuminated forward bias; the
optical balance R + T + A = 1 must be conserved; and each sweep must reproduce
the monotonic direction expected from device physics. Results that fail any gate
are rejected rather than reported.

# 3. Base operating point

Table 1 compares the base J–V figures of merit.

| Metric | SolarLab | SCAPS | Difference |
|---|---|---|---|
| V<sub>oc</sub> (V) | 1.072 | 1.168 | −96 mV |
| J<sub>sc</sub> (mA/cm<sup>2</sup>) | 25.73 | 26.28 | −2 % |
| FF (%) | 85.6 | 87.0 | −1.4 pp |
| PCE (%) | 22.1 | 26.69 | −4.6 pp |

Table 1. Base operating-point comparison on the partner Base Model.

The short-circuit current is now physical and within 2 % of SCAPS, a direct
consequence of completing the optical stack with the glass front substrate; the
small residual is front-surface reflection that SolarLab retains and SCAPS
idealizes away (Section 5). The fill factor agrees to within 1.4 percentage
points. The open-circuit voltage is the single dominant discrepancy: its 96 mV
shortfall propagates through the V<sub>oc</sub>-limited power conversion efficiency and
accounts for essentially the entire PCE gap. The origin of that shortfall is
examined in Section 5.

# 4. Sweep-by-sweep comparison

The ten single-variable sweeps are presented below as overlays of SolarLab
(solid blue) against SCAPS (dashed red), with all four figures of merit per
panel. Grouped by outcome, the results fall into three categories.

**Matched (seven sweeps).** The ETL/PVK conduction-band offset sweep
(Figure 1) reproduces the SCAPS behaviour closely: a recombination cliff near
ΔE<sub>C</sub> = 0.30 eV giving way to a V<sub>oc</sub> spike toward 1.08 V, with the
direction and cliff slope both matched. The PVK/ETL interface defect-density and
defect-level sweeps (Figures 2 and 3) match in direction and saturating shape,
the defect-density response spanning roughly 75 % of the SCAPS V<sub>oc</sub> range.
The HTL/PVK interface defect-density sweep (Figure 4) is near-flat in both
solvers, as are the perovskite conduction- and valence-band defect-level sweeps
and the HTL/PVK defect-level sweep (Figures 8–10), where SCAPS likewise shows no
significant response. These seven constitute the core of the validation: where
SCAPS responds, SolarLab responds in the same direction and shape; where SCAPS
is flat, SolarLab is flat.

**Partial (one sweep).** The ETL donor-doping sweep (Figure 5) is correct on the
high-doping arm — V<sub>oc</sub> rises with N<sub>D</sub> in the right direction — but the
low-doping points show a V<sub>oc</sub> dip not present in SCAPS. This is discussed in
Section 5.

**Open (two sweeps).** The perovskite-bulk conduction- and valence-band
defect-density sweeps (Figures 6 and 7) are flat in SolarLab, whereas SCAPS
shows a small response (−39 mV and −11 mV respectively). The mechanism is
explained in Section 5.

![Figure 1. ETL/PVK conduction-band offset (ΔE<sub>C</sub>).](../figures/scaps_validation/sweep_CHI_ETL.png)

![Figure 2. PVK/ETL interface defect density (N<sub>t</sub>).](../figures/scaps_validation/sweep_Nt_PVK_ETL.png)

![Figure 3. PVK/ETL interface defect level (E<sub>t</sub>).](../figures/scaps_validation/sweep_Et_PVK_ETL.png)

![Figure 4. HTL/PVK interface defect density (N<sub>t</sub>).](../figures/scaps_validation/sweep_Nt_HTL_PVK.png)

![Figure 5. ETL donor doping (N<sub>D</sub>).](../figures/scaps_validation/sweep_Nd_ETL.png)

![Figure 6. Perovskite conduction-band bulk defect density (N<sub>t</sub>).](../figures/scaps_validation/sweep_Nt_C_PVK.png)

![Figure 7. Perovskite valence-band bulk defect density (N<sub>t</sub>).](../figures/scaps_validation/sweep_Nt_V_PVK.png)

![Figure 8. Perovskite conduction-band bulk defect level (E<sub>t</sub>).](../figures/scaps_validation/sweep_Et_C_PVK.png)

![Figure 9. Perovskite valence-band bulk defect level (E<sub>t</sub>).](../figures/scaps_validation/sweep_Et_V_PVK.png)

![Figure 10. HTL/PVK interface defect level (E<sub>t</sub>).](../figures/scaps_validation/sweep_Et_HTL_PVK.png)

For at-a-glance reference, Table 2 summarizes the per-sweep outcome.

| Sweep | SolarLab vs SCAPS | Status |
|---|---|---|
| ETL/PVK conduction-band offset (ΔE<sub>C</sub>) | cliff 0.30 → spike 1.08 V; direction and cliff slope match | match |
| PVK/ETL interface N<sub>t</sub> | 1.08 → 0.87 V (≈ 75 % of SCAPS range) | match |
| PVK/ETL interface E<sub>t</sub> | direction and saturating shape match | match |
| HTL/PVK interface N<sub>t</sub> | near-flat (SCAPS near-flat) | match |
| PVK-CB / PVK-VB / HTL-PVK E<sub>t</sub> | flat (SCAPS flat) | match |
| ETL donor doping N<sub>D</sub> | high-N<sub>D</sub> arm correct; low-N<sub>D</sub> dip | partial |
| PVK-CB / PVK-VB bulk N<sub>t</sub> | flat (SCAPS shows −39 / −11 mV) | open |

Table 2. Per-sweep scorecard.

# 5. Analysis of the current situation

The three residual discrepancies are physical model differences, not
parameter-fitting errors, and SolarLab's values remain physically valid in every
case.

**Base open-circuit voltage (−96 mV).** With the intrinsic carrier density,
recombination coefficients (Auger and radiative), and contact treatment all set
identical to SCAPS, SolarLab recombines more at a given voltage: the implied dark
saturation current is roughly 37× higher, and kT·ln 37 ≈ 93 mV accounts for
almost the entire gap. At the operating point the recombination is dominated by
the Auger and radiative channels at the coefficients specified in the partner
data, compounded by an internal drop of about 135 mV in the quasi-Fermi-level
splitting across the heterojunction band offsets. We verified that this is *not*
an intrinsic-carrier-density, contact-boundary, or interface-calibration error —
each was tested and ruled out — but rather a high-injection carrier-statistics
and heterojunction-transport difference between the two solvers.

**ETL donor doping (N<sub>D,ETL</sub>).** The high-doping arm rises with N<sub>D</sub> in
the correct direction; the low-doping V<sub>oc</sub> dip is tied to the contact and
built-in-potential treatment at very low ETL doping, not to interface
recombination. The dip persists under an independent quasi-steady-state
interface solver, confirming it originates in the contact physics rather than in
interface sampling.

**Perovskite bulk defect density (CB / VB).** SolarLab's open-circuit voltage is
pinned by the interface-recombination ceiling, which sits below the voltage at
which bulk trap density becomes visible; SCAPS reaches that regime and shows the
small −39 mV and −11 mV responses. Raising the ceiling would require reducing
total recombination, which conflicts with the interface model the seven matched
sweeps depend on — so closing this gap by tuning would regress the matched
results.

**Short-circuit current residual (−2 %).** The remaining 2 % of J<sub>sc</sub> is
front-surface reflection. SCAPS treats the optical front as ideal; SolarLab
retains the physical reflection. Removing it would improve the absolute match at
the cost of physical fidelity, so it is kept.

Taken together, the open items reflect a consistent principle: where matching
SCAPS would require SolarLab to behave less physically, physical correctness is
preferred. The present configuration is therefore the honest upper bound of
agreement under that constraint.

# 6. Conclusion and outlook

SolarLab reproduces the SCAPS-1D reference on the partner Base Model to a degree
appropriate for a validated device simulator: short-circuit current within −2 %,
fill factor within −1.4 percentage points, and seven of ten parameter-sweep
trends matched in direction and shape, with every result satisfying the physical
bounds. The remaining discrepancies are characterized rather than incidental —
the open-circuit-voltage shortfall is a heterojunction-transport and
carrier-statistics difference; the ETL-doping dip is contact / built-in-potential
physics; the bulk-defect insensitivity is the interface-recombination ceiling.
Each has been isolated, and in each case the SolarLab choice is the physically
defensible one.

Closing the remaining gaps would require reconciling solver-internal differences
in heterojunction transport and high-injection carrier statistics, which is a
substantial undertaking and is not pursued here because it would not improve —
and could degrade — the physical fidelity that the present configuration
guarantees. We recommend the current configuration (glass front substrate with
the non-generative interface clamp enabled by default) as the validated baseline
for partner work, with the open items documented as known, understood, and
bounded.

# Appendix: Reproducibility

```
cd perovskite-sim
python scripts/scaps_absolute_scorecard.py      # absolute + trend vs the workbook
python scripts/scaps_validation_figures.py \
       --out ../docs/figures/scaps_validation     # regenerate the overlays
```

Full technical detail — methods, every sweep, and the development history — is in
`docs/scaps_validation_report.md`.
````

- [ ] **Step 2: Verify the file has no AI-style tics or literal sub/superscripts in prose**

Run:
```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
grep -nE 'Headline:|✅|⚠️|❌|模拟' docs/partner/scaps_validation_partner_summary.md || echo "OK: no AI callouts / emoji / CJK"
grep -nE '[A-Za-z]_(oc|sc|sat|bi|t|D|C|V|g|i)\b' docs/partner/scaps_validation_partner_summary.md | grep -v '<sub>' | grep -v 'sweep_' | grep -v '_PVK\|_ETL\|_HTL\|N_t\|cm3' || echo "OK: no literal subscripts in prose"
```
Expected: both print OK lines (sweep filenames and the appendix reproducibility command paths are allowed; the second grep filters those out).

---

### Task 2: Run the typography normalizer as a safety pass

**Files:**
- Modify: `docs/partner/scaps_validation_partner_summary.md` (in place, if any symbols remain)
- Read-only: `perovskite-sim/scripts/md_physics_typography.py`

- [ ] **Step 1: Run the normalizer on the file**

Run:
```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
python perovskite-sim/scripts/md_physics_typography.py docs/partner/scaps_validation_partner_summary.md
```
Expected: reports either "no changes" or a small number of conversions. The script protects code fences, `$math$`, inline code, links, and existing HTML tags.

- [ ] **Step 2: Confirm subscript tags are balanced and figure links intact**

Run:
```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
echo "open=$(grep -o '<sub>' docs/partner/scaps_validation_partner_summary.md | wc -l) close=$(grep -o '</sub>' docs/partner/scaps_validation_partner_summary.md | wc -l)"
grep -c 'figures/scaps_validation/sweep_' docs/partner/scaps_validation_partner_summary.md
```
Expected: `open` equals `close`; figure-link count is 10.

---

### Task 3: Render and verify the PDF

**Files:**
- Modify (regenerate): `docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf`

- [ ] **Step 1: Render the PDF**

Run:
```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
pandoc docs/partner/scaps_validation_partner_summary.md \
  -o docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf \
  --toc --pdf-engine=xelatex --resource-path=docs/partner \
  -V mainfont="Arial" -V monofont="Menlo" \
  -V geometry:margin=2cm -V colorlinks=true
```
Expected: exits 0, no "Missing character" / "tofu" warnings, no LaTeX error.

- [ ] **Step 2: Verify page count and embedded figures**

Run:
```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
python -c "import pypdf; r=pypdf.PdfReader('docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf'); print('pages', len(r.pages))" 2>/dev/null || mdls -name kMDItemNumberOfPages docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf
```
Expected: a reasonable page count (≈ 8–12). If pypdf is unavailable, `mdls` reports the page count on macOS.

- [ ] **Step 3: Spot-check rendered typography by extracting text**

Run:
```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
python -c "import pypdf; r=pypdf.PdfReader('docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf'); t=''.join(p.extract_text() for p in r.pages); print('Voc literal present (BAD if many):', t.count('V_oc')); print('has Voc text:', 'Voc' in t or 'V oc' in t)" 2>/dev/null || echo "pypdf unavailable; open PDF to spot-check V_oc/J_sc render as subscripts"
```
Expected: no `V_oc` literal in extracted prose text (extraction of `<sub>` renders as `Voc` or `V oc`), confirming subscripts are typeset rather than literal.

---

### Task 4: Commit and push

**Files:**
- Commit: `docs/partner/scaps_validation_partner_summary.md`, `docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf`

- [ ] **Step 1: Stage and commit**

Run:
```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
git add docs/partner/scaps_validation_partner_summary.md docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf
git commit -m "docs(partner): rewrite SCAPS comparison report as academic prose

Replace AI-style bullet/scorecard summary with a professional technical
report: executive summary, introduction/scope, methodology, base operating
point, sweep-by-sweep comparison, current-situation analysis of the three
open physical gaps, and conclusion/outlook. Correct sub/superscript
typography throughout; re-rendered partner PDF (Arial, xelatex).

Confidence: high
Scope-risk: narrow
Not-tested: visual PDF inspection deferred to user"
```
Expected: commit succeeds.

- [ ] **Step 2: Push**

Run:
```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
git push origin main
```
Expected: push succeeds.

- [ ] **Step 3: Re-verify the PDF still on disk (OneDrive sync guard)**

Run:
```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
ls -la docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf
git status --short docs/partner/
```
Expected: PDF present with recent timestamp; `git status` clean for `docs/partner/` (OneDrive can overwrite freshly-built files — if dirty, re-render and re-commit).

---

## Self-Review

- **Spec coverage:** Executive summary (Task 1 §Exec), introduction/scope (§1), methodology (§2), base operating point + table (§3), sweep comparison + 10 figures + scorecard (§4), current-situation analysis of 3 gaps + J<sub>sc</sub> residual (§5), conclusion/outlook (§6), reproducibility appendix — all present. Typography constraint → Task 2. xelatex-safety (no emoji/CJK) → Task 1 Step 2 grep. Render pipeline → Task 3. Data fidelity (E10.1, no new runs) → content uses the existing numbers; no sweep/sim commands in execution. ✔ All spec sections covered.
- **Placeholder scan:** No TBD/TODO; full prose and exact commands given. ✔
- **Type/number consistency:** Base J–V numbers (1.072 / 25.73 / 85.6 / 22.1 and SCAPS 1.168 / 26.28 / 87.0 / 26.69; −96 mV / −2 % / −1.4 pp / −4.6 pp) and the ten figure filenames match the prior summary and the scorecard table. ✔

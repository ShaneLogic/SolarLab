# SolarLab vs SCAPS-1D partner report — academic rewrite (design)

**Date:** 2026-05-29
**Author:** Xuan-Yan Chen
**Status:** approved (design phase)

## Problem

The partner-facing comparison report (`solarlab_scaps_comparison_report.pdf`,
E6-era, and its successor `docs/partner/scaps_validation_partner_summary.md`)
reads "like the AI": bullet/scorecard-heavy, hedged phrasing, `Headline:`
callouts, and inconsistent sub/superscript typography. It needs to be rewritten
as an academic, professional **technical report** in flowing analytical prose,
with correct super/subscript typography throughout, and an explicit analysis of
the current validation situation.

## Decision

- **Target deliverable:** rewrite `docs/partner/scaps_validation_partner_summary.md`
  in place, then re-render the partner PDF in `docs/partner/`.
- **Style/depth:** professional technical report (~4–6 pp). Executive-summary
  prose plus one analytical section per topic. Replace bullet scorecards with
  prose; retain at most two compact tables (base J–V + at-a-glance scorecard).
- The old `outputs/scaps_analysis/solarlab_scaps_comparison_report.pdf` (gitignored
  E6 artifact) is left as-is; it is superseded by the docs/partner deliverable.

## Document structure

1. **Executive summary** — one prose paragraph. What was validated, the headline
   (J<sub>sc</sub> within −2 %, FF within −1.4 pp, 7 of 10 sweep trends matched,
   V<sub>oc</sub> −96 mV; every result physically valid), and the bottom line for the
   partner.
2. **Introduction & scope** — SolarLab (1D drift-diffusion + Poisson + mobile-ion
   solver, transfer-matrix optics) vs SCAPS-1D; the three-layer Base Model
   (spiro 20 nm / MAPbI<sub>3</sub> 800 nm, E<sub>g</sub> = 1.53 eV / TiO<sub>2</sub> 25 nm) from
   `1R-Parameters.xlsx`; validation philosophy — trend fidelity and physical
   validity are the primary bar, absolute values need only approach.
3. **Methodology** — brief prose: the SCAPS-mirror configuration
   (`scaps_mirror_v2.yaml`), the ten single-variable sweeps, and the enforced
   physics gates (J<sub>sc</sub> ≤ Shockley–Queisser limit, V<sub>oc</sub> ≤ V<sub>bi</sub>,
   recombination ≥ 0, energy conservation R + T + A = 1).
4. **Base operating point** — retain the one 4-row J–V table
   (V<sub>oc</sub> / J<sub>sc</sub> / FF / PCE + Δ). Replace caption bullets with prose
   interpreting the match and the V<sub>oc</sub>-limited PCE.
5. **Sweep comparison** — prose grouped by outcome: matched (ETL/PVK
   conduction-band offset cliff, PVK/ETL interface N<sub>t</sub>/E<sub>t</sub>, HTL/PVK and
   PVK-bulk flats), partial (ETL donor doping N<sub>D,ETL</sub>), and open (perovskite
   bulk N<sub>t</sub>, CB/VB). The ten Arial overlay figures embedded as numbered
   Figures. One compact scorecard table retained for at-a-glance reference.
6. **Analysis of the current situation** — the three open items presented as
   characterized *physical model differences* (not parameter-fit errors):
   - base V<sub>oc</sub> (−96 mV): ~37× higher implied dark saturation current
     (kT·ln 37 ≈ 93 mV), Auger + radiative dominated at the PDF coefficients,
     plus a ~135 mV quasi-Fermi-splitting drop across the heterojunction band
     offsets; ruled out n<sub>i</sub>, contact-boundary, and interface-calibration
     causes;
   - N<sub>D,ETL</sub>: high-doping arm correct; low-doping V<sub>oc</sub> dip tied to the
     contact / built-in-potential treatment, not interface recombination;
   - perovskite bulk N<sub>t</sub>: V<sub>oc</sub> pinned by the interface-recombination
     ceiling, which sits below the bulk-trap-visible regime SCAPS reaches.

   Plus the J<sub>sc</sub> −2 % residual (physical front-surface reflection SolarLab
   keeps; SCAPS treats the optical front as ideal). Frame why physics-honest
   results are preferred over chasing absolute parity.
7. **Conclusion & outlook** — current best (E10.1: glass front substrate + NOGEN
   clamp default) is the physically-honest maximum; what closing each remaining
   gap would require (e.g., heterojunction-transport / SCAPS solver-internal
   differences); recommendation for the partner.
8. **Reproducibility** (appendix) — the two regeneration commands.

## Constraints

- **Typography:** real `<sub>`/`<sup>` tags in the markdown source for every
  physics symbol; never `_`/`^` literals in user-facing text. Run
  `perovskite-sim/scripts/md_physics_typography.py` as a safety normalization
  pass after authoring.
- **xelatex-safe:** no emoji, no CJK characters (e.g. 模拟) in the rendered
  source — they break the xelatex pipeline.
- **Figures:** reuse the existing ten Arial overlays under
  `docs/figures/scaps_validation/sweep_*.png`. No figure regeneration unless the
  underlying data changed (it has not since E10.1).
- **Data fidelity:** all numbers come from the current best (E10.1) results
  already in `scaps_validation_partner_summary.md` and `docs/scaps_validation_report.md`.
  No new simulation runs; this is a writing task.

## Render pipeline

```
pandoc docs/partner/scaps_validation_partner_summary.md \
  -o docs/partner/SolarLab_SCAPS_validation_2026-05-29.pdf \
  --toc --pdf-engine=xelatex --resource-path=docs/partner \
  -V mainfont="Arial" -V monofont="Menlo" \
  -V geometry:margin=2cm -V colorlinks=true
```

## Success criteria

- Report reads as professional academic prose — no `Headline:` callouts, no
  bullet scorecards as the primary content, no AI hedging tics.
- Every physics symbol renders with correct sub/superscript in the PDF
  (spot-check V<sub>oc</sub>, J<sub>sc</sub>, MAPbI<sub>3</sub>, mA/cm<sup>2</sup>, N<sub>D,ETL</sub>).
- All ten figures render; PDF has no missing-glyph "tofu" boxes.
- Section set matches the structure above; the "Analysis of the current
  situation" section explicitly covers the three gaps + J<sub>sc</sub> residual.
- Numbers consistent with the current-best (E10.1) data.

## Out of scope

- Regenerating figures or re-running simulations.
- Touching the old `outputs/scaps_analysis/` artifact.
- Changing the full technical report `docs/scaps_validation_report.md` (the
  partner summary links to it for detail).

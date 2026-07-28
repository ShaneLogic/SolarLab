# SolarLab Research Documentation

Project-wide research notes, validation reports, plans, and source materials.
For the simulator library and how to run it, see the [root README](../README.md)
and `perovskite-sim/CLAUDE.md`.

> **Scope split.** This `docs/` holds **cross-tree** material (SCAPS validation,
> the user manual, research plans/specs). Library-specific dev docs (benchmarks,
> phase plans) live under `perovskite-sim/docs/`.

## Layout

```
docs/
├── reference/     SolarLab vs SCAPS-1D validation deliverables
│                  (see "Reference reports" below)
├── manual/        SolarLab technical user manual (md + tex + pdf, figures, slides)
├── figures/       ScapsSolarlabCompare/ — SCAPS sweep overlays (2026-07-02 run).
│                  The report figure sets were dropped 2026-07-28: the reports
│                  ship as PDFs with their figures already embedded
├── plans/         cross-tree design + implementation plans
├── superpowers/   spec/plan history (specs/, plans/, references/)
├── autoloop/      autonomous research-loop ledger
└── *.md, *.pdf    top-level notes (see "Loose docs" below)
```

> **Naming:** SCAPS-1D is treated as a **reference** simulator (a validation
> baseline), never a "partner." Put new SCAPS-comparison docs under `reference/`.

## Reference reports (current, canonical)

| File | What |
|------|------|
| `SolarLab2DScan` (`.md` + `.pdf`) | 2D defect-parameter (Nt×Et, Nt×ΔE_C) validation |
| `SolarLabVerifyFormal260702.pdf` | SCAPS vs SolarLab (f=0.53) across all 11 sweeps — physical-model & numerical-algorithm attribution; publication-style figures |

The other reports (physics diagnostics, de-spike/interface closure, the
2026-07-02 transient-vs-steady-state comparison, the mechanistic gap analysis,
the interface-SRH scope note, the `_archive/` snapshots and the GAPReasoning
decks) were removed on 2026-07-28. They are in git history at `c7d4ee9` —
restore with `git checkout c7d4ee9 -- docs/reference`.

### Rendering a report
```bash
python perovskite-sim/scripts/md_physics_typography.py <file>.md   # idempotent sub/sup pass
pandoc <file>.md -o <file>.pdf --toc --pdf-engine=xelatex \
  --resource-path=docs/reference \
  -V mainfont="Arial" -V monofont="Menlo" -V geometry:margin=2cm -V colorlinks=true
```
Decks: the GAPReasoning `.pptx` files and their `build_deck.py` were removed on
2026-07-28 (`git checkout c7d4ee9 -- docs/reference/GAPReasoning` to get them
back; the per-figure generators they call need `c8e4fde`).

## Loose docs

| File | What |
|------|------|
| ~~`scaps_validation_report.md`~~ | Removed 2026-07-28. Regenerate with `perovskite-sim/scripts/run_scaps_validation.py`; the findings are in `reference/SolarLabSCAPSGapAnal.pdf` |
| `SolarLab_validation_gap_analysis_2026-06-24.md` | Physics-validation gap analysis (Zotero × codebase) |
| `solarlab_manual_source_dossier.md` | Source dossier for the user manual |
| `manual/SCAPSManual2016.pdf` | SCAPS-1D reference manual (input) |
| `docker-development.md` | Docker dev environment notes |

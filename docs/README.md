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
│   ├── *.md / *.pdf        current reports (see "Reference reports" below)
│   ├── Reasoning/     slide decks + figure-generation scripts
│   └── _archive/           superseded report versions (kept for history)
├── manual/        SolarLab technical user manual (md + tex + pdf, figures, slides)
├── figures/       shared SCAPS sweep figure sets (validation / ss_compare /
│                  despike_compare / gap_explainer) — referenced by reports
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
| `SolarLabPhyVerify260624.md` (+ `PhysicsVerify260624.pdf`) | Depth-resolved physics diagnostics |
| `SolarLab2DScan` | 2D defect-parameter (Nt×Et, Nt×ΔE_C) validation |
| `SolarLabDespikeIface` | De-spike + interface-plane closure decomposition |
| `SolarLabValid260702.pdf` (+ `ScapsValidSum.md`) | Transient vs steady-state interface-states comparison (2026-07-02: transient Nd_ETL contact-reservoir fix + CBO sweep extended to +1.0 eV) |
| `SolarLabVerifyFormal260702.pdf` | SCAPS vs SolarLab (f=0.53) across all 11 sweeps — physical-model & numerical-algorithm attribution; publication-style figures |
| `SolarLabSCAPSGapAnal` | Mechanistic gap analysis (referenced by the 06-22 summary) |
| `SCAPSIfaceSRH.md` | Interface-SRH scope note (cited from `device.py`) |

Superseded versions (06-15 validation, root-cause analysis) live in `reference/_archive/`.

### Rendering a report
```bash
python perovskite-sim/scripts/md_physics_typography.py <file>.md   # idempotent sub/sup pass
pandoc <file>.md -o <file>.pdf --toc --pdf-engine=xelatex \
  --resource-path=docs/reference \
  -V mainfont="Arial" -V monofont="Menlo" -V geometry:margin=2cm -V colorlinks=true
```
Decks: `python docs/reference/Reasoning/build_deck.py`.

## Loose docs

| File | What |
|------|------|
| `scaps_validation_report.md` | Internal SCAPS-mirror validation report (from `run_scaps_validation.py`); uses `figures/scaps_validation/` |
| `SolarLab_validation_gap_analysis_2026-06-24.md` | Physics-validation gap analysis (Zotero × codebase) |
| `solarlab_manual_source_dossier.md` | Source dossier for the user manual |
| `SCAPS Manual february 2016.pdf` | SCAPS-1D reference manual (input) |
| `docker-development.md` | Docker dev environment notes |

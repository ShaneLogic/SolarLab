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
│   ├── ppt_root_cause/     slide decks + figure-generation scripts
│   ├── _archive/           superseded report versions (kept for history)
│   ├── 1D-SCAPS 模拟.pdf    ⎫ SCAPS source INPUTS (reference device + params)
│   └── 1R-Parameters.xlsx  ⎭
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
| `SolarLab_SCAPS_parity_status_2026-06-23` | Campaign capstone — parity verdict |
| `SolarLab_physics_verification_report_2026-06-24` | Depth-resolved physics diagnostics |
| `SolarLab_SCAPS_2Dscan_comparison_2026-06-23` | 2D defect-parameter (Nt×Et, Nt×ΔE_C) validation |
| `SolarLab_SCAPS_despike_interface_findings` | De-spike + interface-plane closure decomposition |
| `SolarLab_SCAPS_validation_2026-06-22` (+ `scaps_validation_reference_summary_2026-06-22.md`) | Transient vs steady-state interface-states comparison |
| `SolarLab_SCAPS_gap_analysis_corrected` | Mechanistic gap analysis (referenced by the 06-22 summary) |
| `SCAPS_interface_SRH_scope.md` | Interface-SRH scope note (cited from `device.py`) |

Superseded versions (06-15 validation, root-cause analysis) live in `reference/_archive/`.

### Rendering a report
```bash
python perovskite-sim/scripts/md_physics_typography.py <file>.md   # idempotent sub/sup pass
pandoc <file>.md -o <file>.pdf --toc --pdf-engine=xelatex \
  --resource-path=docs/reference \
  -V mainfont="Arial" -V monofont="Menlo" -V geometry:margin=2cm -V colorlinks=true
```
Decks: `python docs/reference/ppt_root_cause/build_deck.py`.

## Loose docs

| File | What |
|------|------|
| `scaps_validation_report.md` | Internal SCAPS-mirror validation report (from `run_scaps_validation.py`); uses `figures/scaps_validation/` |
| `SolarLab_validation_gap_analysis_2026-06-24.md` | Physics-validation gap analysis (Zotero × codebase) |
| `solarlab_manual_source_dossier.md` | Source dossier for the user manual |
| `SCAPS Manual february 2016.pdf` | SCAPS-1D reference manual (input) |
| `docker-development.md` | Docker dev environment notes |

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
├── manual/        SolarLab technical manual source, dated PDFs, figure generator
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

## Current manual

| Artifact | Role |
|------|------|
| [`manual/SolarLabManual260811.pdf`](manual/SolarLabManual260811.pdf) | Current 88-page published edition, dated 2026-08-11 |
| [`manual/solarlabPubManual.md`](manual/solarlabPubManual.md) | Single editable source of truth |
| [`manual/generate_manual_figures.py`](manual/generate_manual_figures.py) | Generates matched PDF/PNG figure pairs and fails closed on changed evidence |
| [`manual/README.md`](manual/README.md) | Exact build, evidence-input, and publication workflow |

The current manual figure set is:

| Figure | README use and claim boundary |
|------|------|
| `architecture_flow` | Architecture overview; replaces the earlier manual flow PNG |
| `device_contact_boundary` | Electrical coordinate and potential sources; replaces the old README `device_structure` view |
| `band_interface_convention` | Band bending and interface closures; replaces the old `band_diagram` + `transport_equations` pair |
| `solver_topology` | Driver-specific variables and checks; replaces the old `solver_pipeline` view |
| `csi_qf_convergence` | Restricted-QF internal numerical evidence, not an external c-Si fit |
| `cbo_interface_validation` | Internal grid pass plus failed external SCAPS-shape gate |
| `twod_scope` | Registered 1D/2D parity domain and excluded interface/ion physics |

Each name has a vector `.pdf` for the manual and a web-renderable `.png` for
GitHub. The existing `perovskite-sim/docs/images/ui_layout.png` has no 260811
manual replacement and remains the current README UI diagram.

## Reference reports (current, canonical)

| File | What |
|------|------|
| `SolarLab2DScan` (`.md` + `.pdf`) | 2D defect-parameter (Nt×Et, Nt×ΔE_C) validation |
| `SolarLabVerifyFormal260702.pdf` | SCAPS vs SolarLab (f=0.53) across all 11 sweeps — physical-model & numerical-algorithm attribution; publication-style figures |

The other reports (physics diagnostics, de-spike/interface closure, the
2026-07-02 transient-vs-steady-state comparison, the mechanistic gap analysis,
the interface-SRH scope note, the `_archive/` snapshots and the GAPReasoning
decks) were removed on 2026-07-28. They are in git history at `b773569` —
restore with `git checkout b773569 -- docs/reference`.

### Rendering a report
```bash
python perovskite-sim/scripts/md_physics_typography.py <file>.md   # idempotent sub/sup pass
pandoc <file>.md -o <file>.pdf --toc --pdf-engine=xelatex \
  --resource-path=docs/reference \
  -V mainfont="Arial" -V monofont="Menlo" -V geometry:margin=2cm -V colorlinks=true
```
Decks: the GAPReasoning `.pptx` files and their `build_deck.py` were removed on
2026-07-28 (`git checkout b773569 -- docs/reference/GAPReasoning` to get them
back; the per-figure generators they call need `551615d`).

## Loose docs

| File | What |
|------|------|
| `manual/PauwelsVanhoutte1978.pdf` | Interface-state reference paper used by the manual |
| `manual/SCAPSManual2016.pdf` | SCAPS-1D reference manual (input) |
| `docker-development.md` | Docker dev environment notes |

### Removed loose docs

All of these were write-once prose: no code, test, fixture or build step read
them, so removing them changes nothing that runs.

| File | Was | Get it back |
|------|-----|-------------|
| `SolarLab_validation_gap_analysis_2026-06-24.md` | Physics-validation gap analysis (Zotero × codebase). Superseded by `reference/SolarLabVerifyFormal260702.pdf` | `git checkout db700fe -- docs/SolarLab_validation_gap_analysis_2026-06-24.md` (removed 2026-07-29) |
| `solarlab_manual_source_dossier.md` | Source dossier the user manual was drafted from; its content is in the manual itself | `git checkout db700fe -- docs/solarlab_manual_source_dossier.md` (removed 2026-07-29) |
| `scaps_validation_report.md` | SCAPS validation report | Regenerate with `perovskite-sim/scripts/run_scaps_validation.py` (removed 2026-07-28) |

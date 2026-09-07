# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

Single project tree: `perovskite-sim/` (the whole simulator) plus `docs/figures/`
(only the PNGs the root README embeds). **`perovskite-sim/` has its own `CLAUDE.md`** with exhaustive architecture notes (solver hot paths, TMM optics, backend SSE pattern, frontend panel structure, test BLAS-pinning gotcha, etc.). **Always read `perovskite-sim/CLAUDE.md`** — it is the authoritative guide.

Note: parallel `perovskite-sim-phase2b/` worktree was removed once tandem v1 (PR #11) and Phase 2b Layer Builder UI (PR #2) merged into `main`. Short-lived feature isolation now uses `.worktrees/<name>/` (gitignored).

## Archived documents (outside the repo)

Since 2026-09-07 the repo holds only code, tests, configs, the physics contract
docs under `perovskite-sim/docs/`, and the README figures. Everything else lives in
the iCloud folder `~/Library/Mobile Documents/com~apple~CloudDocs/projects/solarlab/`
(its `README.md` is the index; files there use two-word CamelCase names, and
`RenameMap.md` maps the old repo-era names to the new ones):

| iCloud folder | Was in the repo |
|---|---|
| `plans/specs/`, `plans/impl/` | `docs/superpowers/{specs,plans}`, `docs/plans`, `perovskite-sim/docs/{plans,superpowers}` |
| `plans/autoloop/`, `plans/audit/`, `plans/notes/` | autoloop ledger, physics audit, benchmark + study notes |
| `develop/manual/` | `docs/manual` (manual source, dated PDFs, figure generator) |
| `figures/` | `docs/figures/ScapsSolarlabCompare`, `docs/superpowers/figures`, old README images |
| `results/` | `outputs/` → `Scan2d`, `Spatial`, `ScapsAnalysis`, `Autoloop`, `IonResults`, `DenseIonResults`; `perovskite-sim/outputs/` → `CaladoRepro`, `NumericalRefinement`, `InterfaceCbo`, `ScapsValid`, `ScapsValidE17`, `RegLadders`; `notebooks/outputs` → `Notebooks`. Run-instance folders inside keep their original parameter names |
| `test/` | `docs/reference` reports (`TwodScan`, `FormalVerify`), `ValidEvidence1/2.md`, `P1Checkpoints/` (reproducibility P1 notes) |
| `reference/` | SCAPS manual, Pauwels-Vanhoutte paper, SCAPS reference report + xlsx |

Generated outputs are gitignored (`outputs/`, `perovskite-sim/outputs/`); regenerate
with the scripts under `perovskite-sim/scripts/` and file the result in `results/`.
The one input file code still reads, `scaps_1r_parameters.xlsx`, stays at
`perovskite-sim/reproducibility/`.

## Git

`origin`: `github.com/ShaneLogic/SolarLab.git`, default branch `main`. Commits land directly on `main` in this project (no PR workflow enforced locally). Use `git push origin main` after committing.

## Common Commands

Run from **inside `perovskite-sim/`**, not from the SolarLab root:

```bash
cd perovskite-sim

# Python — the marker split is the non-obvious part
pytest                                 # unit + integration, excludes -m slow (~2 min)
pytest -m slow                         # slow regression (258 tests, ~2 h, BLAS auto-pinned)

# Backend — must run from the SolarLab root so --app-dir resolves
uvicorn backend.main:app --host 127.0.0.1 --port 8000 \
    --app-dir perovskite-sim --reload
```

See `perovskite-sim/CLAUDE.md` for the full command reference, architecture deep-dive, and known gotchas (Radau `max_step` cap near flat-band, `_JV_RADAU_MAX_NFEV`, RHS finite-check, TMM `_inv2x2` det guard, tandem series-matching, YAML 1.1 scientific-notation coercion, SSE streaming pattern, BLAS thread pinning for the slow suite).

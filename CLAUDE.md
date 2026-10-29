# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

Project trees: `perovskite-sim/` (Python simulator and backend), root `web/`
(the frontend), plus `docs/figures/`
(only the PNGs the root README embeds). **`perovskite-sim/` has its own `CLAUDE.md`** with exhaustive architecture notes (solver hot paths, TMM optics, backend SSE pattern, frontend panel structure, test BLAS-pinning gotcha, etc.). **Always read `perovskite-sim/CLAUDE.md`** — it is the authoritative guide.

Note: parallel `perovskite-sim-phase2b/` worktree was removed once tandem v1 (PR #11) and Phase 2b Layer Builder UI (PR #2) merged into `main`. Short-lived feature isolation now uses `.worktrees/<name>/` (gitignored).

## New files: where they go and what they are called

Decide placement first, then the name.

**Folder names.** Inside the repo every folder is lowercase `snake_case` or a single
lowercase word, like the code tree (`perovskite_sim/`, `tests/fixtures/configs/`,
`docs/figures/`, `docs/scaps_collection_templates/`); never CamelCase, never hyphens.
In the archive the root-level folders keep the user's lowercase scheme (`present/`,
`develop/`, `plans/`, `figures/`, `results/`, `test/`, `reference/`, `deck-src/`) and
every folder below them is multi-word CamelCase with no dates (`plans/Specs/`,
`develop/Manual/Figures/`, `results/CaladoRepro/Alignment/`, run folders `RunV7`).

**Repo (`SolarLab/`, pushed to GitHub)** — only what the simulator needs to run,
test, and be understood:

| What | Where | Name |
|---|---|---|
| Python / TypeScript / MATLAB code, tests, backend, frontend | language folders | language convention (`snake_case.py`, `test_*.py`, kebab-case `.ts`) — never CamelCase |
| Shipped device presets (the research entry points the frontend catalogue lists) | `perovskite-sim/configs/` + `web/src/preset-catalog.ts` | existing scheme (`snake_case.yaml`) |
| Test-only / historical device presets | `perovskite-sim/tests/fixtures/configs/` (never served by the API; API tests use the `serve_all_presets` fixture) | existing scheme |
| Package data, tooling files | `perovskite_sim/data/`, tool-fixed names | as the tool or existing scheme requires |
| Physics contract / policy / protocol / schema / certificate docs that code, tests or configs cite | `perovskite-sim/docs/` | multi-word CamelCase, versions as `V1`, `V2` (`TwodTransportContract.md`, `ExplicitDefectSchemaV2.md`) |
| Registries and data the reproducibility chain reads | `perovskite-sim/reproducibility/` | CamelCase (`ConfigBenchmarkMatrix.yaml`); re-pin the sha256 in the matrix after any edit |
| Figures the root README embeds | `docs/figures/` | CamelCase, no date; regenerate under a new `V<n>` and update the README link |
| Tutorial notebooks and run-by-path benchmark scripts | `perovskite-sim/notebooks/` | CamelCase (`JvHysteresis.ipynb`, `IonmongerBenchmark.py`) |

**Archive (iCloud `projects/solarlab/`, never in git)** — everything else:

| What | Folder |
|---|---|
| Design specs / implementation plans / audits / study notes | `plans/Specs`, `plans/Impl`, `plans/Audit`, `plans/Notes` |
| Manual source, built PDFs, manual figures | `develop/Manual` (`ManualV<n>.pdf`, bump `n` per edition) |
| Technical reviews, comparison reports, explorers | `develop/` |
| Decks | folder root while being edited, then `present/`; old versions `present/Archive/` |
| Validation reports, evidence, certification notes | `test/` |
| Third-party papers, manuals, reference data | `reference/` |
| Sweep / run outputs, figure sets, notebook outputs | `results/<Study>/` (scripts write to the gitignored `outputs/` first; move the finished run over) |
| Explanatory figures not embedded in the README | `figures/<Set>/` |

Archive names: multi-word CamelCase for every file and non-root folder, no dates,
timestamps or hashes — chronology is `V1`, `V2`, …; decimals as `0p4`; index files stay
`README.md`. Add a row to the archive's `README.md` when you add a folder. Do not rewrite
references inside text files with bare-word patterns (see `RenameMapFull.md` for why).

## Archived documents (outside the repo)

Since 2026-09-07 the repo holds only code, tests, configs, the physics contract
docs under `perovskite-sim/docs/`, and the README figures. Everything else lives in
the iCloud folder `~/Library/Mobile Documents/com~apple~CloudDocs/projects/solarlab/`
(its `README.md` is the index; every file and non-root folder there uses multi-word
CamelCase with no dates — versions are V1, V2, ... — and `RenameMap.md` + `RenameMapFull.md`
map the old repo-era names to the new ones):

| iCloud folder | Was in the repo |
|---|---|
| `plans/Specs/`, `plans/Impl/` | `docs/superpowers/{specs,plans}`, `docs/plans`, `perovskite-sim/docs/{plans,superpowers}` |
| `plans/Autoloop/`, `plans/Audit/`, `plans/Notes/` | autoloop ledger, physics audit, benchmark + study notes |
| `develop/Manual/` | `docs/manual` (manual source, PDFs as ManualV1–V5, figure generator) |
| `figures/` | `docs/figures/ScapsSolarlabCompare`, `docs/superpowers/figures`, old README images |
| `results/` | `outputs/` → `Scan2d`, `Spatial`, `ScapsAnalysis`, `Autoloop`, `IonResults`, `DenseIonResults`; `perovskite-sim/outputs/` → `CaladoRepro`, `NumericalRefinement`, `InterfaceCbo`, `ScapsValid`, `ScapsValidE17`, `RegLadders`; `notebooks/outputs` → `Notebooks`. Run-instance folders inside are CamelCase too, e.g. `CaladoRepro/Alignment/PaperContactsAtol1N60`, `Driftfusion/RunV7` |
| `test/` | `docs/reference` reports (`TwodScan`, `FormalVerify`), `ValidEvidence1/2.md`, `P1Checkpoints/` (reproducibility P1 notes) |
| `reference/` | SCAPS manual, Pauwels-Vanhoutte paper, SCAPS reference report + xlsx |

Generated outputs are gitignored (`outputs/`, `perovskite-sim/outputs/`); regenerate
with the scripts under `perovskite-sim/scripts/` and file the result in `results/`.
The one input file code still reads, `ScapsParams.xlsx`, stays at
`perovskite-sim/reproducibility/`.

## Git

`origin`: `github.com/ShaneLogic/SolarLab.git`, default branch `main`.

The user authorized the full refactor on 2026-10-05. All refactor development,
staging, commits and pushes belong on **`devel`**. Do not modify, commit to, or
push `main`. A single coordinator-designated integrator owns branch changes,
staging, commits and synchronization in the shared checkout. Other workers must
verify the repository path, actual branch and HEAD before editing their assigned
files; they must not switch branches or commit concurrently. Preserve existing
changes and unrelated untracked files; never reset or stash them to start work.

The automatic delivery endpoint is a fully validated, reproducible release and
a review-ready **`devel` → `main` PR**. Only the user may merge it. Agents must
not merge, enable auto-merge, bypass branch protection, or claim `main` has been
refactored before the human merge. Resolve safe integration conflicts and rerun
affected checks on `devel` before requesting that final review. Any per-task
branch advice in older plans is superseded for this shared refactor run.

Follow archive `plans/Impl/ArchitectureRefactorExecutionV2.md` (V1 is frozen
history) and keep the sole active state at archive
`test/ArchitectureRefactor/ExecutionStateV2.json` with the append-only event log
`test/ArchitectureRefactor/EventLogV2.jsonl`. Worker reports are evidence, not
additional activity ledgers. Repository-required guidance stays in Git; plans,
evidence and scientific run artifacts stay in the archive. Use the supported Orca
task/message/lifecycle workflow and respect each Dispatch's file ownership.

The refactor commander and sole integrator is the user-appointed Claude Code
session `0537b216-df76-4292-bb73-949175abc111` (model `claude-opus-5-5`, effort
`max`, Orca terminal `term_a33ad446-8ee7-473b-b4b7-00aea210c562`), appointed by
the user on 2026-10-29 and bound to Run `run_b01db3902366` at consumer
generation 6. It replaces Codex session `01a180c3-f826-7873-8d0b-44b2e40899c4`
(Orca terminal `term_3c65f4db-0a9b-41c7-b337-956eff7204dd`), which is now deputy
and standby commander. The deputy makes no Git mutations, dispatches or state
writes; it takes over only when the commander has been unreachable for more
than 2 hours. Execution is authorized. The runnable new-core 1D research
release has been delivered (research packages up to V8). The next delivery is
the user-selected engineering milestone (option A, 2026-10-29): one unified
layout/term/assembly layer serving steady, transient and AC analyses, a
one-command green test suite, full CI coverage, and the existing new-core
capabilities migrated onto it. Functionality not yet migrated stays explicitly
behind the legacy facade, and G9P remains the final endpoint. The commander
sets the product time-integration route deadline (proposed 2026-11-08). If IDA
is not qualified by then, a new-core adaptive BDF becomes the product default
and IDA becomes optional; final scientific gates are unchanged.
Then continue the remaining approved scope. Clearly distinguish implementation,
verification, and scientific qualification; legacy execution is not new-core completion.
Routine reversible implementation and sequencing decisions belong to the commander.
Use approximately 15-minute investigation checkpoints; after two checkpoints
without useful evidence or progress, change the assumption, scope, or method.
Ask the user only about changes to final scientific standards, core delivery
scope, added resource cost, or merging main. Existing workers keep actual runtime `gpt-6-astra`, reasoning
`max`, Full Access and network enabled; sessions added after 2026-10-14 may
instead run `claude-opus-5-5` at effort `max` (user-approved). Every
participant's actual model/effort/permissions are verified once per session start
from its own runtime record (Codex turn_context or Claude transcript); a requested
setting or UI label alone is insufficient. Scientific thresholds change only
prospectively after independent review and before the affected runs; user-facing
acceptance targets change only with explicit user approval; old failures are
never relabelled. R1 L2/L3 qualification is a research track decoupled from the
product release.

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

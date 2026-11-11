# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

The active package lives in `src/solarlab/`, with the service in
`src/solarlab_server/`, root `tests/`, `examples/` and `web/`. Read
[Development Guide](docs/DevelopmentGuide.md) for the current extension and
validation workflow. `perovskite-sim/` is the retained legacy engine and has
its own `CLAUDE.md`; read that guide before working on legacy behavior.

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
| Operational developer and extension guidance | `docs/` | Multi-word CamelCase, such as `DevelopmentGuide.md` |
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

`origin` is `github.com/ShaneLogic/SolarLab.git`; remote `main` retains the
published history. The user's November 10, 2026 consolidation keeps two local
development branches and workspaces:

- **`devel`**: the shared new-core engineering and integration baseline.
- **`AIMC`**: the proposed microscopic aging model, based on the current
  `devel` core. Its research design is not an implemented model capability.

Use one active writer per checkout. Verify the actual repository path, branch
and HEAD before editing or staging. Preserve unrelated user changes; do not
reset or stash them to make room. Shared engineering changes belong on
`devel`; model-specific changes belong on `AIMC` until their contracts and
validation support integration.

The former commander/deputy/worker terminal assignments were retired by the
user's consolidation request. Do not restart that roster or treat old
"in progress" terminal labels as current work. Future delegation follows the
user's current instructions and recorded model preferences. Live user
directions supersede historical staffing assignments.

The sole active state remains archive
`test/ArchitectureRefactor/ExecutionStateV2.json`, with append-only events in
`test/ArchitectureRefactor/EventLogV2.jsonl`. The snapshot, pending-work
decisions, source/evidence recovery material and session indexes are under
`test/ArchitectureRefactor/WorkspaceConsolidationV1/`. Older formal task
counts and independently pinned results retain their historical scope; they
are not a current implementation percentage.

The unified layout/term/assembly engineering milestone A was delivered and
merged through PR #26 on November 1, 2026. Supported new-core 1D and steady
2D entry points use that assembly layer; unmigrated features remain behind
the explicit legacy boundary. Backward Euler remains the general transient
default. Explicit BDF routes exist, but CND2 method-selection/default-switch
work is a separate unmerged candidate.

For the next CND2 review, use
`Stage04/P04CoreV1/Core1D/AdaptiveBDFV1/Cnd2ReviewScopeDecisionV1.json` in the
archive: the November 2 user decision adopts the interface trust standard.
Earlier strict-scope failures remain historical. V4's committed-tree CI was
recorded green, while independent acceptance under the new standard remains
outstanding. November 8 was an earliest permitted switch date, not an
automatic switch; retain every applicable prerequisite.

Scientific thresholds and resource limits are unchanged. Distinguish working
implementation, engineering verification and scientific qualification.
G2/C2, the applicable hysteresis/SCAPS reference work, refinement and final
G9P acceptance remain separate. R1 L2/L3 research qualification remains
decoupled from product engineering delivery.

Do not modify or push remote `main`, merge a release PR, enable auto-merge
or bypass branch protection. Only the user merges into `main`. Keep a
review-ready `devel` change reproducible without reintroducing old worktree
paths, duplicate numerical implementations or unreviewed candidate defaults.

## Common Commands

### Root developer test command

With the locked editable all-extras development environment active and `web/`
dependencies installed (`uv sync --frozen --all-extras` and `npm ci --prefix web`),
run from the SolarLab root:

```bash
python scripts/test_all.py
```

This runs every ordinary **root pytest selection** in `.github/workflows/ci.yml`
on the active Python, with the matching `PYTEST_CI_JOB`. Each selection gets a
fresh process, including the isolated `scan_execution` and `run_client` groups;
BLAS uses one thread. The runner reads CI's discovery/isolation policy, keeps all
failures visible, continues through the remaining groups, prints one final
per-job/overall summary, and exits nonzero if any selection fails. It writes
JUnit reports and `summary.json` under `outputs/test_all/` (`--output-dir` overrides
that path). Existing reasoned skips and xfails (strict marks and conditional runtime xfails) remain visible in the totals.
The 768 MiB pool limit is unchanged; do not substitute a single root `pytest`
process, whose accumulated imports count toward the pools' process-family RSS.

This local command uses the existing installation; CI still checks its Python
matrix, package installation modes, binding probe, lint/types and web test/build
steps. **legacy suite: not part of the green claim** — the separate
`perovskite-sim` suite is outside this command. Optional scheduled slow/coverage jobs retain
their own commands and scope.

### Legacy commands

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

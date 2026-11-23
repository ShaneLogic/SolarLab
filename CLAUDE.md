# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

The active shipped product and research companion packages live in `src/solarlab/`,
`src/solarlab_server/`, `src/solarlab_research/`, and `web/` (cut over to V2 execution
at `P09-02` and packaged in `solarlab==0.1.0` wheels/sdists), alongside root `tests/`,
`examples/`, `docs/`, `reproducibility/`, and `scripts/`. Read
[Development Guide](docs/DevelopmentGuide.md) for the current extension and
validation workflow, and [Retired Models & Legacy Witness Ledger](docs/RetiredModelsV1.md)
for retired legacy entrypoints and historical provenance. The legacy `perovskite-sim/`
tree has been completely retired and removed from `devel` (its final pre-slim snapshot
remains permanently addressable in Git at tag `archive/pre-slim-be973ac5`, commit
`be973ac5e1aa46c5c21eaa79c42b402d25a76eb3`). Local `.claude/` skills and
`skills-lock.json` remain gitignored on disk.

Short-lived feature isolation uses `.worktrees/<name>/` (gitignored).

## New files: where they go and what they are called

Decide placement first, then the name.

**Folder names.** Inside the repo every folder is lowercase `snake_case` or a single
lowercase word, like the code tree (`src/solarlab/`, `tests/fixtures/configs/`,
`docs/contracts/`, `docs/figures/`); never CamelCase, never hyphens.
In the archive the root-level folders keep the user's lowercase scheme (`present/`,
`develop/`, `plans/`, `figures/`, `results/`, `test/`, `reference/`, `deck-src/`) and
every folder below them is multi-word CamelCase with no dates (`plans/Specs/`,
`develop/Manual/Figures/`, `results/CaladoRepro/Alignment/`, run folders `RunV7`).

**Repo (`SolarLab/`, pushed to GitHub)** — only what the simulator needs to run,
test, and be understood:

| What | Where | Name |
|---|---|---|
| Python / TypeScript code, tests, server, frontend | `src/`, `tests/`, `scripts/`, `web/` | language convention (`snake_case.py`, `test_*.py`, kebab-case `.ts`) — never CamelCase |
| Shipped device presets (the research entry points the frontend catalogue lists) | `src/solarlab/materials/data/presets/` + `web/src/preset-catalog.ts` | existing scheme (`snake_case.yaml`) |
| Test-only / historical device presets | `tests/fixtures/configs/` (never served by the API) | existing scheme |
| Package optical / template / default catalog data | `src/solarlab/materials/data/` | as the packaged loader requires |
| Physics contract / policy / protocol / schema / certificate docs | `docs/contracts/` and `docs/` | multi-word CamelCase, versions as `V1`, `V2` (`TwodTransportContract.md`, `ExplicitDefectSchemaV2.md`) |
| Active reference contracts and benchmark baselines | `reproducibility/` | CamelCase (`ScapsComparisonContractV1.json`, `HysteresisReferenceContractV2.json`, `ScapsParams.xlsx`) |
| Operational developer and extension guidance | `docs/` | Multi-word CamelCase, such as `DevelopmentGuide.md` |
| Retired models, falsified closures, and historical witness ledger | `docs/RetiredModelsV1.md` | Multi-word CamelCase with version suffix (`RetiredModelsV1.md`) |
| Figures the root README embeds | `docs/figures/` | Versionless multi-word CamelCase, no date (`ValidationScapsOverview.png`) |

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

Generated outputs are gitignored (`outputs/`, `var/`). Active benchmark and
qualification scripts live under `scripts/benchmarks/`, and the SCAPS reference
workbook `ScapsParams.xlsx` lives at `reproducibility/ScapsParams.xlsx`.

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
steps. Optional scheduled slow/coverage jobs retain their own commands and scope.

### Product CLI & Server commands

From the locked root environment (`.venv/bin/`) or an installed `solarlab==0.1.0` wheel:

```bash
# Verify packaged resources, configuration schema, and absence of legacy packages
solarlab check-install --require-no-legacy

# Run a 1D ion-free steady-state J-V sweep on the packaged classical_pin preset
solarlab steady-jv --output outputs/steady_jv.json

# Launch the V2 FastAPI/SSE server (serves compiled _web_dist SPA when built)
solarlab-server --store-dir ./var/solarlab/store --host 127.0.0.1 --port 8000

# Research companion CLI entry points (src/solarlab_research/)
solarlab-cbo-scan --help
solarlab-r1-short-window --help
solarlab-r1-verify-sources --help
```

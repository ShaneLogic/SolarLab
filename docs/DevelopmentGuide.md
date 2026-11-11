# Development Guide

The continuing development branch is `devel`. `AIMC` is the separate branch
for the proposed microscopic aging model. The Python package remains
`solarlab==0.1.0`; a working development baseline is not a declaration of
complete scientific or release qualification.

## Start from the root package

The root `pyproject.toml` and `uv.lock` define the new-core environment.
With uv and Node installed, use:

```sh
uv sync --frozen --all-extras
npm ci --prefix web
.venv/bin/python scripts/test_all.py
```

The test runner reproduces the ordinary root pytest selections and their
isolated process groups from `.github/workflows/ci.yml`. It retains skipped
and expected-failure cases in its summary. It does not run the separate
legacy suite, every CI installation mode, the full Python matrix, the
standalone TypeScript build, or the optional slow/coverage campaigns.

Run the frontend checks separately when frontend or cross-language contracts
change:

```sh
SOLARLAB_PYTHON="$PWD/.venv/bin/python" npm --prefix web run test:run
npm --prefix web run build
```

Start a scientific workflow with the existing [1D examples](../examples/research_1d/README.md)
or [2D examples](../examples/research_2d/README.md). Their input restrictions,
run commands and interpretation limits are part of the example contract.
Use a fresh local RunStore for a new calculation; use its reader to inspect
an existing calculation.

## The code to extend

| Layer | Location | Responsibility |
| --- | --- | --- |
| Input and material declarations | `src/solarlab/config/`, `materials/` | Explicit units, validated input and immutable material declarations |
| Device compilation | `src/solarlab/device/` | Resolve an admitted declaration into a bound device |
| Local physical laws | `src/solarlab/physics/` | Constitutive values and derivatives, independent of experiment orchestration |
| Geometry and discretization | `src/solarlab/mesh/`, `discretization/` | Control volumes, connectivity and stable numerical primitives |
| Equation composition | `src/solarlab/equations/assembly/` | Shared `SystemLayout`, `Term`, storage and `Assembler` contracts |
| Numerical solution | `src/solarlab/solvers/` | Nonlinear, time and linear solvers acting on equation contracts |
| Experiment orchestration | `src/solarlab/experiments/`, `sweeps/` | Protocols, preparation, execution and measurements |
| Persistence and identity | `src/solarlab/io/`, `reproducibility/` | RunStore, source/input identities, original attempts and lossless readers |
| Service and browser | `src/solarlab_server/`, `web/` | Registered execution, HTTP/events and client contracts |
| Legacy engine | `perovskite-sim/` and `solarlab_server/legacy_engine/` | Explicit compatibility routes for capabilities not yet migrated |
| Research companion | `src/solarlab_research/` | Reserved companion namespace; active runnable research entry points are currently in `examples/` |

The import contracts in `pyproject.toml` enforce these boundaries. In
particular, the core does not import the server, research companion or legacy
engine; solvers do not import experiment orchestration. Keep optional quantum
or GPU backends outside ordinary core imports.

## Current behavior and limits

The supported 1D steady, transient and small-signal entry points and the
supported 2D steady entry points use the unified equation assembly by default.
Their explicit original-equation opt-out is
`with equation_implementation(LEGACY): ...`, or
`NEWCORE_EQUATIONS=legacy` for an independently started process. This opt-out
selects the original new-core equation classes; it does not select the
separate `perovskite_sim` engine.

Backward Euler remains the general transient default. Adaptive BDF1/2 is an
explicit, bounded research route; its admitted electronic and single-positive-ion
adapters already use assembled equations. Equation implementation and time
integration method are separate choices.

CND2 adds admission-bound method selection and the proposed BDF default
switch. Its V4 candidate is preserved outside the production branch. The
November 2, 2026 decision requires a fresh review under the interface trust
standard; a recorded green committed-tree CI run does not complete that
review. November 8 was an earliest permitted switch date, not an automatic
deadline that changes the running default.

Remaining scientific work includes the applicable G2/C2 reference evidence,
hysteresis and SCAPS reference validation, mesh/time/voltage refinement, and
the final G9P release gates. Existing bounded working examples and historical
legacy results retain their own scope. Neither cleanup nor a green test suite
promotes those scientific gates.

## Add a physical model without duplicating the solver

1. Define the physical state, units, conserved inventory, supported input
   domain and unavailable cases before adding a numerical path.
2. Add the local law and its derivatives in `physics/`; resolve declared
   parameters through configuration/material/device compilation.
3. Bind the law through the existing layout and term/assembly contracts.
   Extend storage, residuals and Jacobians together when the model adds a
   dynamic state. Do not copy a steady or time solver into a model-specific
   module.
4. Add the experiment and measurement protocol above the solver layer.
   Preserve source, input, method and model identities in persisted results;
   record failed attempts without replacing them with accepted results.
5. Verify zero-effect and limiting cases, units, conservation, derivative
   consistency and unsupported-input refusals. Compare unchanged quantities
   with the existing implementation; bind any new scientific claim to its
   own declared reference and refinement evidence.
6. Add service/UI registration only after the Python contract and saved
   reader are usable. Keep optional runtime dependencies optional.

The executable local recombination extension in the 1D workflow is the
smallest existing example of this pattern. Broader dynamic defects and
microscopic charge/event coupling need their own contracts and tests.

## The AIMC branch

AIMC currently contains design and analytic research, not an implemented VMC,
atomistic event or device-coupling engine. Its branch-specific
`docs/AimcIntegration.md` records the agreed objective and next development
steps. Advance shared engineering changes from `devel` before changing the
physical model. Do not merge design status into production capability flags.

The existing requirement is a real electronic VMC solve after each accepted
ion/vacancy position update. A cache, surrogate, or coarsened event schedule
must not silently replace that requirement. The contact, electronic charge
sectors, event mechanism, error targets and resource measurements still need
to be frozen before a target simulation is claimed.

## Git and evidence workflow

Use `devel` for shared engineering work and `AIMC` for the new model.
Keep one active writer per checkout. Commit only cohesive, reviewed changes;
check the actual branch and HEAD before staging. Remote `main` is retained
as published history and is not rewritten by workspace cleanup.

Code, test fixtures and the operational guidance needed to use the repository
belong in Git. Research reports, raw outputs, historical candidate patches and
session recovery indexes belong in the existing SolarLab archive. Local
`.codex/` settings and `work/ignis/` runs are not production source.

The sole execution-state record remains archive
`test/ArchitectureRefactor/ExecutionStateV2.json`; its older formal task counts
must not be read as a current implementation percentage. The consolidation
snapshot, per-worktree decisions, checked restoration material and validation
receipts are under
`test/ArchitectureRefactor/WorkspaceConsolidationV1/`.

Use that folder's restoration tool to reopen a historical candidate in a fresh
directory. Do not reapply an old patch wholesale to the current branch merely
because a retired checkout had uncommitted files. Preserve each candidate's
base commit and review standard when continuing it.

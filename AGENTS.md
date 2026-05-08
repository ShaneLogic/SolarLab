# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Repository Layout

SolarLab is a meta-repo containing two sibling project trees for the same simulator:

```
SolarLab/
├── perovskite-sim/                      Primary tree — main branch work lands here
├── perovskite-sim-phase2b/perovskite-sim/   Parallel worktree for Phase 2b frontend (Layer Builder UI)
└── docs/                                Cross-tree superpowers plans + specs
```

Both trees are full copies of the same project (Python `perovskite_sim` library + FastAPI backend + Vite/TS frontend + pytest suite). They are **not** a monorepo with shared packages — each has its own `pyproject.toml`, `frontend/node_modules`, `configs/`, and `tests/`. Changes do not automatically propagate between them.

**Each tree has its own `AGENTS.md`** with exhaustive architecture notes (solver hot paths, TMM optics, backend SSE pattern, frontend panel structure, test BLAS-pinning gotcha, etc.). **Always read the `AGENTS.md` inside the tree you are actually editing** — it is the authoritative guide for that tree.

## Which Tree To Work In

- **`perovskite-sim/`** — default for physics, solver, backend, configs, and most test work. This is what `main` tracks and what gets pushed to `origin`.
- **`perovskite-sim-phase2b/perovskite-sim/`** — only when the task explicitly concerns the Phase 2b Layer Builder UI (custom stacks, drag/drop layer visualizer, user preset save-as). Its spec and plan live in `docs/superpowers/{specs,plans}/` at the root and in the tree itself.

If unsure, check `git status` and recent commits — work in the tree whose files are actually being modified.

## Git

The whole SolarLab root is **one git repository** (`origin`: `github.com/ShaneLogic/SolarLab.git`, default branch `main`). Both nested trees are tracked in the same repo; there are no submodules. `git status` at the root shows changes from either tree. `perovskite-sim-phase2b/` is currently untracked in `.gitignore` terms — verify before committing Phase 2b work.

Commits land directly on `main` in this project (no PR workflow enforced locally). Use `git push origin main` after committing.

## Common Commands

Run these from **inside the tree you are working in**, not from the SolarLab root:

```bash
cd perovskite-sim    # or perovskite-sim-phase2b/perovskite-sim

# Python
pip install -e ".[dev]"
pytest                                 # unit + integration, excludes -m slow
pytest -m slow                         # slow regression (TMM baselines, ~27 s, BLAS auto-pinned)
pytest path/to/test.py::test_name      # single test

# Backend (run from SolarLab root so --app-dir resolves)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 \
    --app-dir perovskite-sim --reload

# Frontend
cd frontend && npm install && npm run dev    # http://127.0.0.1:5173
npm run build                                # tsc + vite build
```

See the nested `AGENTS.md` for the full command reference, architecture deep-dive, and known gotchas (Radau `max_step` cap near flat-band, `_JV_RADAU_MAX_NFEV`, RHS finite-check, TMM `_inv2x2` det guard, tandem series-matching, YAML 1.1 scientific-notation coercion, SSE streaming pattern, BLAS thread pinning for the slow suite).

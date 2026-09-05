# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

Single project tree: `perovskite-sim/` (the whole simulator) plus `docs/`
(cross-tree superpowers plans + specs). **`perovskite-sim/` has its own `CLAUDE.md`** with exhaustive architecture notes (solver hot paths, TMM optics, backend SSE pattern, frontend panel structure, test BLAS-pinning gotcha, etc.). **Always read `perovskite-sim/CLAUDE.md`** — it is the authoritative guide.

Note: parallel `perovskite-sim-phase2b/` worktree was removed once tandem v1 (PR #11) and Phase 2b Layer Builder UI (PR #2) merged into `main`. Short-lived feature isolation now uses `.worktrees/<name>/` (gitignored).

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

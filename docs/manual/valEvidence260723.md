# Validation Evidence — 2026-07-23

Python evidence pass at commit `35e2f51` (quiet machine, load ~4-5,
`OPENBLAS` default; slow suite BLAS-pinned by `tests/conftest.py`).

| Gate | Command | Result | Runtime |
|---|---|---|---:|
| Python default suite | `pytest` | 1101 passed, 3 skipped, 1 xfailed | 1098 s |
| Python slow suite | `pytest -m slow` | 99 passed, 2 failed (pre-existing), 4 skipped | 4856 s |
| Python validation suite | `pytest -m validation` | 22 passed | 443 s |
| Frontend build | `npm run build` | passed (tsc clean; 86 modules; interactive terminal, I/O-bound wall time) | 380 s |
| Frontend unit tests | `npm run test:run` | 27 files, 371 tests passed (git checkout on local disk; in-place run stalls on the cloud-sync FS) | 1.34 s |

## Known failing slow tests (pre-existing)

Both fail identically at the 2026-07-22 baseline commit `bb24449` — they
pre-date the 2026-07 review-fix revision:

1. `tests/regression/test_grading_cigs_notch.py::test_cigs_back_grading_raises_voc_without_jsc_collapse`
   — coupled-solver non-convergence at V_app = 0.4 V on the graded-CIGS
   sweep (solver robustness on that configuration).
2. `tests/integration/test_autoloop_boulder.py::test_boulder_sweep_real`
   — stale expectation: the `trend:Nd_ETL:V_oc` gap the test awaits no
   longer exists after the interface-channel calibration.

## Context

This pass supersedes the 2026-05-19 pass at `43c81d7` (647/72/18 tests).
Between the two passes the suite grew (1105 default / 105 slow / 22
validation collected) and a ~150x ion-coupled sweep performance regression
introduced on 2026-05-29 (`6d37952`, e9.3 global interface clamp) was
found via A/B timing + git bisect and fixed by defect-scoping the clamp
(`35e2f51`); `ionmonger_benchmark` full J-V restored to 4.1 s with metrics
at the exact pinned envelope (V_oc 1.1932, J_sc 231.7, FF 0.7774).

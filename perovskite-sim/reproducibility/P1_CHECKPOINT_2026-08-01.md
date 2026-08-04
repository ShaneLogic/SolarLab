# P1 Checkpoint (2026-08-01)

This checkpoint records the final workspace state before the user-requested
pause. It is not a new certified baseline and does not replace the
machine-readable matrix or gap registry.

## Frozen P0 Boundary

- Base commit: `c23e5b9beb3c356250ea32dcb09c78dc45ba28ec`
- Frozen patch SHA-256:
  `58166a458047984bf85ead3cc5c5b5e29b2c6dc22aa851682a8ca81ef314d82a`
- `python scripts/verify_reproducibility.py --json` reconstructs the P0 state
  and verifies all 15 frozen file hashes.
- The P0 patch was not refreshed after P1 work started.
- Git index and refs are read-only in this managed workspace, so no commit or
  tag has been created.

## P1 State

- The shipped-config registry covers 28 YAML files, 3 schemas, 18 runtime
  resources, and 15 benchmark contracts with byte and semantic hashes.
- The IonMonger frozen-ion, short-circuit terminal-current mesh gap is closed
  on the residual-certified 48/72/96 interval ladder. The finest current
  change is 0.0262 percent. Spatial-profile, ionic-equilibrium, and full J-V
  convergence are not claimed.
- The c-Si preset now has a solver-consistent carrier-Debye grid guard. The
  generic under-resolved mesh fails before integration and the synchronous API
  reports HTTP 422. The 200/300/400 ladder still lacks a certified J-V branch.
- The Lin 2019 comparison uses the source-reported 300/800 nm absorber
  thicknesses. Voc, Jsc, FF, and PCE pass the broad paper window, while the
  sub-cell current mismatch remains 15.13 percent versus the 2 percent gate.
- CIGS 0.5/1.0 um ungraded ideal-contact continuation fails closed at the
  pinned 0.533333/0.466667 V points before Robin, grading, or `N_mult` is
  enabled. No CIGS J-V branch is certified.
- Courtier/IonMonger and Calado/Driftfusion presets remain disclosed
  calibrated reproductions. Exact publication-era external curves, revisions,
  and source protocols are not frozen locally.

Authoritative status and acceptance criteria:

- `reproducibility/config_benchmark_matrix.yaml`
- `reproducibility/p1_gaps.yaml`

Open P1 gaps at this checkpoint:

1. `cigs-2um-graded-notch`
2. `lin2019-tandem-jsc-pce`
3. `csi-transient-jv-grid-envelope`
4. `external-solver-curve-crosscheck`

Closed P1 gaps at this checkpoint:

1. `scaps-low-doping-etl`
2. `ionmonger-residual-ss-96` (terminal-current claim only)

## Completed Verification

- Final default Python lane:
  `1463 passed, 2 skipped, 231 deselected, 12 warnings`.
- Complete final-tree slow lane (2026-08-03):
  `223 passed, 4 skipped, 1465 deselected, 4 xfailed, 4 warnings` in
  `6012.50s (1:40:12)`. All four warnings are the existing `np.trapz`
  deprecations in the dual-ion conservation test; no `RuntimeWarning` was
  emitted.
- Validation lane: `22 passed, 1672 deselected`.
- Reproducibility matrix: `35 passed`.
- Final P1 focused lanes:
  - Ion mesh: `4 passed`.
  - Lin tandem: `6 passed, 1 xfailed`.
  - CIGS: `2 passed, 3 xfailed`.
  - c-Si guard plus API contract: `6 passed`.
  - J-V branch rejection/recovery: `30 passed`.
- Frontend: `377 passed`; TypeScript and Vite production build passed.
- `compileall`, `git diff --check`, and Ruff on the changed P1 core files
  passed.
- Project-wide Ruff is not a green gate: it currently reports 497 existing
  style findings across legacy code, notebooks, and tests.

The `pytest -m slow` worker that was active at the pause request remains
discarded. On 2026-08-03, the entire 231-node collection was rerun from the
beginning against the final code tree with BLAS/OpenMP limited to one thread.
It completed successfully with the exact result recorded above, so the pending
full-slow verification gate is closed.

## Revalidation Command

From the `perovskite-sim` project root, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pytest -q -m slow -W error::RuntimeWarning -p no:cacheprovider
```

After it completes, rerun:

```bash
python scripts/verify_reproducibility.py --json
pytest -q tests/reproducibility/test_matrix.py \
  -W error::RuntimeWarning -p no:cacheprovider
python -m compileall -q perovskite_sim backend scripts tests
git diff --check
```

Then review `git status --short -- perovskite-sim` from the SolarLab Git root.
Do not refresh the frozen P0 patch and do not stage generated outputs or local
research materials.

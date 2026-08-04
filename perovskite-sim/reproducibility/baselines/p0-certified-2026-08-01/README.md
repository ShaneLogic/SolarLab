# P0 Certified Baseline (2026-08-01)

This directory freezes the certified P0 source state independently of the
mutable working tree. The intended annotated tag is
`p0-certified-2026-08-01`, but the managed session exposed `.git` as read-only,
so no commit or tag is claimed here.

`p0.patch` is a full-index binary-capable patch from
`c23e5b9beb3c356250ea32dcb09c78dc45ba28ec` to the P0 state.
`manifest.yaml` records the patch hash, every changed file hash, the
runtime environment, exact-final verification lanes, and the three explicit
known xfails. Generated outputs and the autoloop ledger are outside the P0
scope.

## Reconstruct

Run from the SolarLab Git root, using a disposable directory:

```bash
tmpdir=$(mktemp -d)
git archive c23e5b9beb3c356250ea32dcb09c78dc45ba28ec | tar -x -C "$tmpdir"
cd "$tmpdir"
git apply /absolute/path/to/perovskite-sim/reproducibility/baselines/p0-certified-2026-08-01/p0.patch
```

The reconstructed `perovskite-sim/` files must match the SHA-256 values in
`manifest.yaml`. The project-level verification command performs the same hash
checks without changing the working tree:

```bash
python scripts/verify_reproducibility.py --baseline p0-certified-2026-08-01
```

After P1 starts, the current worktree is expected to differ from some P0 file
hashes. Use `--check-p0-worktree` only on a reconstructed P0 tree.

When Git metadata becomes writable, create the real boundary in a clean clone
or disposable branch based at
`c23e5b9beb3c356250ea32dcb09c78dc45ba28ec`, before applying any P1 changes:

```bash
git apply /absolute/path/to/p0.patch
git add -- perovskite-sim/perovskite_sim/experiments \
  perovskite-sim/perovskite_sim/solver/illuminated_ss.py \
  perovskite-sim/tests/conftest.py \
  perovskite-sim/tests/regression/test_mesh_convergence.py \
  perovskite-sim/tests/regression/test_twod_validation.py \
  perovskite-sim/tests/unit/experiments \
  perovskite-sim/tests/unit/solver/test_illuminated_ss.py
git commit -m "fix(numerics): freeze P0 certified solver baseline"
git tag -a p0-certified-2026-08-01 -m "P0 certified numerical baseline"
```

Verify the staged file list before committing. Do not include `outputs/` or
`docs/autoloop/ledger/`.

# Reproducibility Registry

This directory is the machine-readable boundary between a passing test and a
scientific claim. A configuration is not externally validated merely because
it loads or produces a finite J-V curve.

## Files

- `baselines/p0-certified-2026-08-01/` reconstructs the P0 solver state from
  Git commit `c23e5b9beb3c356250ea32dcb09c78dc45ba28ec` and a SHA-256 pinned
  patch.
- `schema_registry.yaml` assigns every shipped YAML to one explicit loader and
  unit convention.
- `config_benchmark_matrix.yaml` covers every shipped config, runtime optical
  resource, benchmark command, evidence level, and known limitation.
- `p1_gaps.yaml` gives each open or closed P1 gap a reproduction command,
  current evidence, next experiment, and acceptance contract.
- `P1_CHECKPOINT_2026-08-05_EXTERNAL_CV.md` is the latest human-readable
  continuation checkpoint; earlier checkpoint files remain immutable
  historical notes.

## Verification

Run from the `perovskite-sim` project root:

```bash
python scripts/verify_reproducibility.py --json
pytest -q tests/reproducibility/test_matrix.py
```

The verifier fails if any of the following drifts:

- reconstruction of the full 40-character P0 base with `git archive`, patch
  dry-run/application, or any frozen reconstructed-file hash;
- the set or byte-level SHA-256 of the 29 shipped YAML configs;
- the normalized loader semantic hash of any config;
- the set or byte-level SHA-256 of the AM1.5G and n,k resources;
- exact n,k CSV-to-manifest stem coverage;
- declared schema, loader, required fields, finite/domain checks, benchmark
  config/node links, or external source links;
- either direction of the config-to-benchmark mapping;
- standard-YAML versus backend-inline semantics, or SCAPS round-trip semantics.

Calibrated external reproductions additionally pin the observed metrics and
absolute regression tolerances in the matrix itself. Their executable tests
read that same contract, enforce the absorbed-photon budget, and record the
single-threaded reference environment. The matrix separately records the
local and publication protocols, calibrated parameters and targets, and
non-calibration checks. A calibrated reproduction is always `partial`: a broad
paper window or a regression against its own calibrated output is not an
independent validation certificate.

The Lin 2019 partial comparison likewise pins its reported local observables
with narrow regression tolerances in addition to the broader paper windows.
Those pins prevent a stale registry; they do not upgrade proxy optical or
electrical inputs into an external validation dataset.

## Evidence Levels

- `certified`: all attached claims pass and the evidence is not solely a
  calibrated literature reproduction.
- `partial`: some internal or external claims pass, with recorded limitations.
- `load_only`: schema/load checks pass; no physical-result claim is made.
- `demo`: illustrative model wiring, not an external validation case.
- `unvalidated`: an explicit open gap prevents a physical claim.

External labels are deliberately narrower:

- `calibrated_reproduction` reproduces selected literature observables after
  disclosed calibration; it is `calibration_only`, not a holdout prediction.
- `partial_external_comparison` has at least one unresolved model, data, or
  protocol mismatch.

## P1 Numerical Lanes

`test_mesh_convergence.py` is a fixed finite-time protocol study on an exact
24/48/96 interval ladder. It must not be cited as a residual steady-state
certificate.

`test_p1_steady_state_mesh_convergence.py` is the residual-certified frozen-ion
carrier lane. It certifies 48/72/96 actual intervals at short circuit by
prolonging the certified 48-interval state and solving the fine rungs in
relative-log coordinates. Every rung retains the same residual and 0.1 A/m2
continuity-current gates; the finest terminal-current change is 0.0262 percent.
This closes the recorded residual-certified short-circuit terminal-current
mesh gap, not spatial-profile, ionic-equilibrium, or full J-V convergence.

The CIGS lane uses a distinct 25 mV J-V protocol. It certifies the n-left
orientation, a 39/78/120-interval ungraded grid ladder, and a production
120/160-interval ungraded/graded trend comparison. Those are internal
numerical and qualitative claims only; the matrix deliberately keeps both
CIGS configs `partial` because material, contact, optical, and external-curve
provenance are incomplete.

The c-Si frequency-domain C-V lane is also explicit and opt-in. It linearizes
the audited local QF model about residual-certified dark states, uses an
equilibrated complex solve with iterative refinement, and certifies the
N=200/300/400, 10 kHz/100 kHz/1 MHz depletion-bias matrix. Independent DC
electron and hole inventory derivatives reproduce the terminal capacitance,
and the p-n fit uses the two-edge `2kT/q` correction. The resulting 0.782 V
apparent intercept is 0.111 V below the configured contact potential, within
the published 0.1-0.4 V distributed-carrier range. This resolves the internal
interpretation. A second config now maps van Nijen et al.'s published Gaussian
p+/n device without fitting and compares against content-addressed 2-D
Sentaurus admittance data. Its local N=400/600/800 curve is converged, but the
1-D capacitance is 19.7-35.6 percent higher over 0-0.2 V. Because the source
has partial-width 2-D contacts and no public input deck, this is
`partial_external_comparison`, not pointwise parity. Both configurations stay
`partial` and the P1 gap stays open. The lane does not certify the general
endpoint-sampled transient path or unsupported ion, contact, interface, or
non-local physics.

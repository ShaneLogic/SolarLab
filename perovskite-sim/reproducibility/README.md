# Reproducibility Registry

This directory is the machine-readable boundary between a passing test and a
scientific claim. A configuration is not externally validated merely because
it loads or produces a finite J-V curve.

## Current Research Scope

On 2026-09-05, 50 bundled presets were deleted. Only `scaps_mirror_v2.yaml`
and `calado2016_fig1f.yaml` remain. The 52-preset matrix, numerical refinement
registry and P1 records below describe the historical checkout. The full
matrix verifier and old preset-dependent tests require that historical
checkout; they are not a current passing-suite claim.

Current loading, API and protocol-helper checks are:

```bash
python -m pytest -q tests/reproducibility/test_research_presets.py tests/unit/backend/test_scaps_inline_config.py tests/unit/experiments/test_plot_calado_fig1f.py
```

Future studies must define new inputs and acceptance criteria. Existing
historical metrics must not be transferred to another preset.

## Files

- `baselines/p0-certified-2026-08-01/` reconstructs the P0 solver state from
  Git commit `c23e5b9beb3c356250ea32dcb09c78dc45ba28ec` and a SHA-256 pinned
  patch.
- `schema_registry.yaml` assigns every shipped YAML to one explicit loader and
  unit convention.
- `config_benchmark_matrix.yaml` covers every shipped config, runtime optical
  resource, benchmark command, evidence level, and known limitation.
- `p1_gaps.yaml` gives each closed or explicitly P2-deferred P1 item a
  reproduction command, current evidence, and unchanged acceptance contract.
- `P1_CLOSURE_2026-08-07.md` is the P1 phase closeout. Earlier checkpoint
  files remain immutable historical notes.

## Historical Verification

Run from the `perovskite-sim` project root:

```bash
python scripts/verify_reproducibility.py --json
pytest -q tests/reproducibility/test_matrix.py
```

The verifier fails if any of the following drifts:

- reconstruction of the full 40-character P0 base, patch dry-run/application,
  or any frozen reconstructed-file hash. If a loose base blob is a OneDrive
  placeholder, the verifier may reverse-recover it only from the patch-pinned
  target blob, and requires the recovered Git SHA-1 to equal the base commit's
  exact blob ID before applying the patch forward. A target that is also a
  placeholder may use an explicitly declared, SHA-256-pinned baseline snapshot,
  whose bytes must still match the patch's full target Git blob ID;
- the set or byte-level SHA-256 of the shipped YAML configs;
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
Its source stack, reported thicknesses, physical photon budgets, and 0.508
percent sub-cell current match now pass. The proxy-optics budgets remain about
2.1 percent below the champion central Jsc, and absorber/contact inputs remain
partial. Those pins prevent a stale registry; they do not upgrade proxy optical
or electrical inputs into an external validation dataset.

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
`partial`; exact external C-V parity is explicitly deferred to P2. The lane
does not certify the general endpoint-sampled transient path or unsupported
ion, contact, interface, or non-local physics.

## Post-P1 Ion-Aware DC Certification

Phase 2 step 1 adds a dedicated fixed-bias mobile-ion DC preparation and
certificate. The historical 1 ms impedance preconditioner remains a
compatibility path and is not relabeled: at N30 and 0.9 V its ion area
residual is `9.656e-4 A/m2`, maximum ionic face current is
`4.827e-4 A/m2`, and terminal hole positivity fails.

The new lane advances the declared endpoint ladder and requires two
consecutive states to pass full-MOL carrier/ion area residuals, separate
positive/negative ionic face currents, all-face DC current spread,
dual-cell-weighted inventory drift, terminal positivity and site occupancy.
Every solver attempt retains numerical diagnostics; the physical protocol and
outer numerical controls have separate canonical hashes.

| Lane | Status | Run ID | Certificate SHA-256 |
|---|---|---|---|
| `ionmonger-ion-aware-dc-v1` (N30/60/90) | `partial` | `314c4b8cdaecb31b64180a204cf0bdc541f2779fa07d5b32d4c4dc59d90665f1` | `13716fd0ce4d2a588819f450c25061706489ffa736bedb654556d7c876eecfec` |
| `ionmonger-ion-aware-dc-resolved-v2` (N60/90/120) | `certified` | `ba1e7b8dbcb0b16695d03c4626555bc98458ca851fc4aad9611425f55663fe1f` | `fcc84e55b8e6138e9b52c4abb49260f623e747a08b9b2a64af215c63b8ea51e9` |

The v1 matrix is complete; only the N60-to-N90 maximum-site-occupancy
relative difference fails (`0.0154895 > 0.01`). The resolved-v2 lane keeps the
same contract and passes all grid, tolerance and per-cell quality gates; its
N90-to-N120 occupancy difference is `0.0070462`.

This is internal numerical DC evidence, not an external IonMonger match. The
source deck lacks endpoint effective-DOS data, so contact thermodynamics is
`compatible_unverified` and the combined physical certificate remains false.
Dynamic interface-state charge is excluded. See
[ion-aware-dc-certification.md](../docs/ion-aware-dc-certification.md) for the
full contract, values and next-step boundary.

## Post-P1 Physical-Interface CBO Campaign

The 2026-08-10 N=40/50/60 physical-QF-interface scan is separate from the
frozen `p1-closure-2026-08-07` tag. Its one-percent-$J_{\mathrm{sc}}$ critical
intervals have a 7.031 meV union against a 10 meV internal limit, and its local
QSS/current diagnostics pass. The normalized SCAPS-shape error is 0.4744
against a 0.05 external gate. The result therefore records
`numerical_certified=true` but top-level `certified=false`.

The source JSON currently lives at
`outputs/interface-cbo/scan-fermi-edge-qf-grid-40-50-60.json`, inside the
ignored local-output tree. The README-facing rendered artifact is
[`cbo_interface_validation.png`](../../docs/manual/figures/cbo_interface_validation.png).
A clean clone cannot regenerate that panel until the exact machine-readable
result is restored; a different scan is not an acceptable substitute.

## Combined 2D Mobile-Ion/Interface-SRH Certificate

The registered `twod-mobile-ion-interface-srh-v1` lane completed its
source-clean 4/6/8 matched-grid by 1/0.1/0.01 tolerance matrix at commit
`0c9eb26`. Run
`89d108b8817fb4af5d0749bd5848efada9dda99a1b559ed395a8cb0603eaa55b`
completed all 9 cells, and certificate
`b02bc4f8b3b5d470d599f6dacde746b26c263591aafecd14cc6c890a94b677dd`
has status `certified` with no unconverged dimension.

This closes internal convergence and conservation only for the explicit
Neumann-x, ohmic, blocking single-positive-ion, finite-width grain-boundary,
clamp-inactive cross-node `InterfaceDefect`, short-dwell slice. Its accelerated
ion and defect parameters are synthetic stress inputs. Dual ions, selective
contacts, interface charge/state, long-time hysteresis, external simulator
parity, measured-device validation, and material-parameter validation remain
outside the certificate. The complete contract and values are in
[`twod-combined-numerical-certificate.md`](../docs/twod-combined-numerical-certificate.md).

# One-Dimensional Mechanism R0 Protocol V1

This contract is consumed by `scripts/run_one_dimensional_mechanism_r0.py`.
The study specification and completed evidence belong in the iCloud archive,
under `plans/Specs/OneDimensionalMechanismStudySpecV1.md` and
`results/OneDimensionalMechanism/R0V1/` respectively.

## Source and Input

The solver source is commit `5b744b023ac708e50b6035616ca6f07cee3c9eb1`.
The runner archives that commit and imports it from a temporary local directory.
Uncommitted solver, optical, frontend and registry edits are not imported.
The source ZIP, every member's SHA-256, the runner and the study specification
are retained as separate artifacts. A future solver version requires a new
protocol version; this runner does not silently track `main`.

The unchanged fixture is
`tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml`,
SHA-256 `f617f230b2d9c144573394e38fcc313225dc84c7b38f7670b97e9a0a7cc12a24`.
It is a synthetic numerical reference, not a fitted or identified material.
The ion-free foundation p-i-n remains a separate historical numerical anchor.

## Fixed Measurement

- Dark, 300 K, two 100 nm electrical layers, one microscopic interface.
- Positive ions move only in the left absorber; the right layer retains its
  fixed ionic density and compensation. No ionic binding reaction is added.
- Four effective carrier exchange velocities are `None`, meaning pinned
  reservoir densities. Raw omitted values have this meaning only after the
  contact resolver has evaluated the mode and contact flags.
- Start with `build_electrical_grid(stack, 4)`: four total intervals, followed
  by `build_two_sided_trace_grid`. Save the actual grid and its hash.
- Use times `[0, 1e-8, 1e-6, 1e-4] s` and voltage records
  `[0, 0.05, 0.05, 0.05] V`, with the existing right-continuous step-and-hold
  protocol. The first record is the dark DC state, then the driver applies
  50 mV during the first and subsequent integration intervals.
- Construct the public protocol with time-step factor 1. Its complete existing
  nonlinear policy is serialized without changing acceptance limits.
- The research solver and public entry each prepare their own certified dark
  reference and operating state. Their identical-input outputs must agree
  bitwise in all seven recorded overlapping state/current fields.

The 50 mV historical check is not a linear AC experiment. The four observations
do not resolve a long-time ionic relaxation or the ideal-step current impulse.
No lifetime, hysteresis attribution, or Fourier transform is inferred from them.

## Acceptance and Evidence

Both current engine certificates, contact thermodynamics, microscopic binding,
the dark reference and the operating state must pass. No clipping is allowed.
Keep the existing discriminating checks: maximum occupancy change above
`1e-9` and positive-ion relative motion above `1e-5`. Keep the existing
ion-inventory drift bound `1e-9`; this R0 check does not claim the tighter
`1e-10` research target has been certified for a new campaign.

Archive the full raw result, public result, protocol, all resolved inputs,
environment, current/storage decomposition, local interface evidence, and
focused test XML/logs. Raw results contain output-time states and the engine's
diagnostics; this runner does not export every internally accepted substep.
R1 must add the corresponding complete trajectory recording where required.

The focused checks exercise public protocol validation and the existing
four-channel contact-resolution inheritance. They are not a new 2D study.
The complete Python/slow suites and formal spatial refinement matrices are
not rerun by R0. Preserve failures in the output directory and return nonzero;
never overwrite an existing run directory.

## Execution

From the SolarLab repository root, using the existing reference Python
environment (NumPy 2.1.3 / SciPy 1.15.3 for the initial R0 run):

```bash
/Users/shane/Applications/anaconda3/bin/python \
  perovskite-sim/scripts/run_one_dimensional_mechanism_r0.py \
  --output-dir perovskite-sim/outputs/one_dimensional_mechanism_r0/run_v1 \
  --study-spec "/Users/shane/Library/Mobile Documents/com~apple~CloudDocs/projects/solarlab/plans/Specs/OneDimensionalMechanismStudySpecV1.md"
```

The script pins thread variables before numerical imports and checks the
observed BLAS backend inside `threadpool_limits`. Subsequent reproductions
use a new output directory and retain their own environment identities.
The study specification is frozen before execution; results and completion
status are recorded separately so its preregistered hash remains meaningful.

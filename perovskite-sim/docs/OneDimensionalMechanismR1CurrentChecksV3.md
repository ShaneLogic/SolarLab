# R1 independent current assembly V1 — V3 declaration

This document records the finite-step contract introduced in V3 and the V4
regular-current and failed-iterate coverage. It does not amend the
frozen physical limits or independently approve a scientific result.

`CURRENT_METRIC_SEMANTICS = r1-v3-separate-internal-contact-interface` identifies
the restored original internal-face meaning. `transient_current_metrics()[4]`
and the existing `all_face_current_relative` certificate once again use only
the original internal-face set; physical contact closure and two-sided
interface closure retain their own names. Old 392/815 records retain their
producer's contact-inclusive metric definition and are not relabelled.

The contact-aware numerical stopping condition remains available through
`solver_current_metrics`. It is not an independent check or an additional
measurement. This stage retains its old numerical limit and the original
100/40/2 caps. Changing its tolerance alone cannot establish independence.

The new `independent_physics_row(system,state,previous,dt,reported=...)`
returns `R1IndependentPhysicsRowV1` under each
`accepted_steps[*].physics_reconstruction.independent_physics`.
Its precise fields are exported as `ROW_FIELDS`, `ARRAY_FIELDS`,
`BASE_CHECK_FIELDS`, `BASE_REPORTED_FIELDS` and `FINITE_REPORTED_FIELDS` by
`one_dimensional_mechanism_r1_independent_physics.py`.

## Inputs, construction and units

Inputs are the saved/reconstructed physical populations, potentials, local
trace populations and potentials, exact local coordinates, previous accepted
reference, applied boundary change, fixed material data, and physical grid.
The independently assembled answer does not use a stored/solver current,
metric, residual or storage increment. `reported` supplies comparison targets
only and is never used to construct a current.

Physical volumes are reconstructed from the grid, real contacts and true
interface positions and compared with the solver's volumes. Carrier currents
are reconstructed through the cancellation-safe SG/QF law; local bulk-to-trace
currents are reevaluated from trace populations. Ion currents are reevaluated
from the held populations and control settings. The shared constitutive
primitives are explicitly listed in every row: Bernoulli, ion-face flux,
material interface parameter construction and fixed-occupancy carrier law.
The independent assembly does not yet supply a full independent carrier/DAE
oracle. Analytic sign/equilibrium examples and existing Decimal work constrain
parts of those shared primitives, not every possible common implementation
error.

Finite-step population changes use the exact saved logarithmic/logit
increments and reference populations. They are evaluated directly here, not
by the solver storage helper. From them reconstruct volume charge change
and trap sheet-charge change. Potential and interface-trace changes are
decoded separately, including the dielectric boundary lift when present.
Face displacement and physical-contact Gauss extension are assembled without
calling solver metric, displacement or storage helpers.

All returned currents are in A/m² along positive x, before reporting polarity;
ion particle flux is in m⁻²s⁻¹, inventory in m⁻², widths in m. The total charge
rate includes physical endpoint volumes and the changing trap sheet charge.
Inventory compares each active component with the original prepared population
integrated using independently reconstructed volumes.

## Unchanged acceptance checks

| Metric | Definition | Limit |
|---|---|---:|
| internal_face_current_spread_relative | range of original internal total currents / max absolute total, floor 1e-20 A/m² | 2e-6 |
| contact_internal_current_spread_relative | same statistic over internal and physical-contact totals | 2e-6 |
| interface_current_spread_relative | largest left/right interface difference / each interface's max absolute total, floor 1e-20 A/m² | 2e-6 |
| charge_balance_normalized | absolute charge-rate minus left/right contact-conduction difference, using the existing max(charge rate, boundary difference, conduction, 1 A/m²) scale | 1e-10 |
| inventory_relative_drift | largest relative change of independently integrated component population | 1e-10 |

Finite-step closure checks are inapplicable at 0+; the raw conduction values
there are not a right-limit displacement/current certificate. Their checks
have `applicable=false, passed=null`. Inventory, finite populations, positive
carriers, occupancy and physical geometry remain checked. The ion site ceiling
remains strictly below 0.999. No clipping or new physical error floor is added.

Fresh carrier/ion currents must also match the state and applicable published
current arrays exactly under the anchored implementation. This checks wrong
solver and publisher arrays even when they are mutually consistent, provided
the shared constitutive implementation remains the trusted one. These
are content/implementation checks, not new physical tolerances. Metadata
classification and provenance remain separate responsibilities.

`passed` requires every applicable independent check. Verification aggregates
all saved rows at all refinement levels into `independent_physics_passed` and
includes failures in `physical_limit_violations`; a clean finest level cannot
erase a coarse-level violation. When physics evidence is requested, the main protocol checks each independent
row before accepting further steps. An independent violation stops the run
and is retained in the failed prefix. The final replay also vetoes any
independent failure through physical_limits_satisfied.

The same conservation constraint is counted once. Agreement with its solver
stopping analogue is not a second independent measurement, a trajectory error
bound, a linearity budget, or R1-2 acceptance.

## Separate assessments and regular right limits

Every independent report separates `content_consistent` and
`conservation_compliant`. `reference_accuracy_qualified` remains null with
`not_assessed_shared_constitutive_laws`; neither an agreeing reconstruction
nor a small conservation residual supplies a solution-error estimate.
The actual finite-step constants, dimensionless units and phase applicability
are exported for comparison with the machine-readable approved standard.
Original limits, normalization floors, original inventory and the strict
0.999 site ceiling remain unchanged.

`one_dimensional_mechanism_r1_independent_regular.py` reconstructs the fixed-
voltage regular current, including a nonzero-step 0+, from physical state.
It independently assembles carrier and ion continuity and sheet capture,
then solves differentiated Gauss law by integration with both potential
derivative boundaries fixed at zero. Physical half contact volumes and the
two sides of each charged interface are retained. It uses no stored rate,
stored current, solver derivative helper or solver Poisson factorization.
Its shared laws are the same listed constitutive primitives plus bulk
recombination; this is not a second complete device solver.

The result is saved under `regular_current.independent_physics` (and each
output `regular_currents` entry). Nonzero excitation uses the original
2e-6 relative current and 1e-10 charge gates. True zero excitation uses the
original absolute DC continuity/current and normalized residual gates;
relative-current checks remain explicitly inapplicable. Original algebraic
and differentiated-Poisson checks also remain in the production path.

The integrated-Gauss reconstruction and the production tridiagonal solve
can round differently. Their current-array content comparison reports its
machine-arithmetic allowance: 32 times node count times binary64 epsilon,
with the maximum conduction/displacement component as the absolute scale.
This allowance applies only to assembly content comparison; it does not
alter physical conservation limits, weak-signal normalization or response
error budgets. Exact content comparison remains in the finite-step assembly.

## Failed Newton state evidence

On a supported R1 Newton failure the original error retains an actual
`R1NewtonFailureWitnessV1`: previous and attempted physical state, exact
coordinate, independently recomputable residual scales and residual vector,
time, subdivision and independently assembled physical checks. A failure
to collect this witness is explicitly recorded and never replaces the
original failure. The generic successful solver equations and 100/40/2 caps
are unchanged.

`rebuild_failure_witness` replays the saved accepted prefix without time
integration, independently obtains the original scales and re-evaluates the
attempted coordinate. Recomputed terminal physics and passed terminal
physics are separate booleans. Iteration count, historical line search and
linear-solve backward error remain provenance, not reconstructed claims.

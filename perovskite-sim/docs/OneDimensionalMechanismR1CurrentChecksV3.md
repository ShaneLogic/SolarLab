# R1 independent current assembly V1 — V3 declaration

This document records the V3 implementation contract. It does not amend the
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
solver and publisher currents even when they are mutually consistent. These
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

# Frozen-Ion Snapshot Contract

`experiments/degradation.py:measure_frozen_ion_snapshot` measures the electronic
steady-state J-V curve at a supplied, fixed ionic configuration. This is a
separation of electronic and ionic time scales, not proof of ionic equilibrium
or a microscopic chemical degradation mechanism.

## Preserved Quantities

Both `D_ion` and `D_ion_neg` are set to zero. Layer composition, temperature,
initial compensating backgrounds, site capacities, and the inherited positive
and negative ion profiles are preserved. Optical substrates with no electrical
material parameters are retained without attempting to replace those parameters.

A negative species with nonzero population remains in `MaterialArrays` even
when its diffusivity is zero. The ordinary density state retains its negative
ion block, including the Poisson background and site-limit arrays. A declared
but initially empty negative block is also retained when measuring an inherited
state. Freezing mobility must not remove a charged species from electrostatics.

Every requested voltage starts from the same snapshot. After transient seeding
and after electronic steady-state solution, both ionic arrays must be exactly
equal to their inherited values; this also preserves their integrals on the
unchanged grid. Non-finite, negative, malformed or over-capacity states fail.
The measurement and current observer use the same frozen material cache.

## Electronic Acceptance

The requested finite-duration transient supplies an initial guess. Its success
does not establish electronic steady state. The existing `solve_steady_state`
then solves the same carrier continuity equations with frozen ions. It must
return a finite, converged state and a finite nonnegative continuity-current
bound within the requested limit, default 0.05 A/m2. The reported current is
also checked for spatial spread over all faces using that limit. Frozen-ion
current must vanish, and displacement current is omitted only for this
electronically stationary measurement.

`FrozenIonSnapshot` retains the voltage, current, accepted states, carrier
residual, continuity-current bound, spatial current spread and derived metrics.
`DegradationResult` retains the maximum current bounds/spreads per snapshot and
both ionic histories when profile storage is enabled. Failures propagate to
the experiment; no prior state or efficiency is substituted. The empirical
damage law and its lifetime mapping are not retuned by this repair.

These current bounds are numerical acceptance criteria, not measurement errors
or universal accuracy guarantees. Spatial resolution and weak-current scales
require their own study. A successful sparse voltage sweep does not establish
that its interpolated Voc or fill factor is independent of voltage sampling.

## Verification

- `tests/unit/experiments/test_degradation_frozen_ions.py` covers absent,
  positive, negative and dual ions; optical substrates; nonuniform profiles
  with nonzero flux before freezing; exact profile preservation and zero
  per-species frozen current; independence from the original diffusivities;
  and rejection of missing or invalid electronic convergence evidence.
- `tests/integration/test_degradation_snapshot_refinement.py` fixes the initial
  nonuniform dual-ion state and physical history, varies internal seed steps
  by factors 1/2/4, then separately uses voltage steps 20/10/5 mV. The last
  two levels must agree within 1 mV in Voc, 0.2% in Jsc and 0.5% in FF.
  `SOLARLAB_F4_EVIDENCE_PATH` optionally exports raw curves, accepted states,
  parameter/mesh data, numerical-library versions and source hashes as JSON.
  Its paired JUnit result supplies the pass/fail record.
- Existing degradation tests retain their original checks on the empirical
  performance trend, and the componentwise-tolerance forwarding test remains.
- TPV frozen-dual-ion refinement now asserts actual negative-population
  retention before solving; earlier passing runs without that assertion did
  not verify this species when the old material builder discarded it.

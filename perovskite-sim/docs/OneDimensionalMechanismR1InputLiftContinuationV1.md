# R1 Input Lift Continuation V1

The opt-in `r1-rebased-input-lift-dd-v1` representation can advance from an
explicit accepted state. This is an accepted-state continuation API; the
default baseline/pair protocol and cold-start preparation remain separate.

`from_accepted_state(system, previous)` creates a zero-coordinate reference.
It preserves all represented high/low words, including historical Gauss
residual evidence. It clears the preceding voltage lift; `advance_step` builds
the next lift from the next prescribed voltage. No nonlinear correction occurs
when creating the reference.

Use `one_dimensional_mechanism_r1_input_lift_continuous.advance_step` for one
prescribed step or `advance_steps` for a contiguous schedule. Each step records
`previous_time_s`, `time_s`, `dt_s`, and `voltage_V`. Keep the original interval
division for `dt_s`; rounded endpoint subtraction need not be bitwise equal.
The API uses the original Newton solver, line search and policy. It neither
retries failed steps nor changes the time grid.

Pass the original initial system as `scaling_system`. Its reference populations
and local scales remain fixed, while its original scale methods use the current
previous state, reconstructed Jacobians and actual step size. Do not reuse the
frozen scale arrays from a single-step failure diagnostic across a trajectory.

`one_dimensional_mechanism_r1_input_lift_codec` provides `save_checkpoint`,
`restore_checkpoint`, `import_snapshot`, and `replay_saved_step`. JSON checkpoints
carry every represented high/low word, shape, physical coefficient identity,
time, voltage and content hash. Restore rejects missing or inconsistent data.
It performs one same-state algebraic evaluation to rebuild local payloads and
Jacobians, checks derived fields, and creates a new zero-coordinate reference.
This evaluation is not a time advance. V37 snapshots lack a historical Newton
coordinate; importing them records that absence explicitly.

Saved replay reconstructs the previous and accepted states from checkpoint
bytes, recomputes every scaled equation residual and independently assembles
current/charge checks. It does not rerun the nonlinear trajectory. Coefficient
and floating-point environment changes can cause strict same-state restoration
to reject a checkpoint; they must not be hidden by relabeling its evidence.

The bounded V38 integration test resumes the V37 accepted state at
0.075334812716299 s, advances the six remaining original substeps through
0.1 s, and restores the saved checkpoint after the third step. Its acceptance
requires actual native gates and saved-byte residual/physics replay at every
step. This does not establish cold-start, the full protocol Jacobian and
eliminated-operator checks, 4/8/16 refinement, relative resource cost, or 100 s
qualification. Existing short-matrix and dependency evidence is not new-source
qualification for this representation.

## Physical electrostatic residual correction

The V42 implementation evaluates Poisson and the two interface electrostatic
rows directly from the complete represented DD primary fields, in one
canonical arithmetic order. A copied historical residual is no longer treated
as the physical value of the predecessor's Gauss operator. The predecessor's
physical residual is assembled separately; its difference from the saved
historical residual and the stable incremental calculation remain diagnostic
evidence. Actual represented voltage-lift values are included, without assuming
their sampled Laplacian is bitwise zero.

`physical_electrostatic_reference(system, previous)` reconstructs that physical
reference without initialization, a Newton step or mutation of the supplied
state. Source coefficients and predecessor identities bind the cache. This
bounded path currently requires the frozen zero-static-sheet, zero-prescribed-
jump interface model; unsupported sources are rejected. Newton thresholds,
the original eliminated comparison and independent physical gates are unchanged.

Canonical live residuals also preserve exact zero-coordinate rebase and
checkpoint replay. Historical checkpoints containing the old biased residuals
can fail strict restoration under the corrected source; they must remain
historical evidence, not be silently migrated or relabelled accepted.

V42 validation covers manufactured cases, checkpoint/rebase regressions and
fixed-state constraint solves at the saved V40 step13/14 inputs, compared with
independent Decimal references. Corrected potential, carrier density and trace
values are local algebraic results. They are not accepted coupled time steps,
and do not authorize resuming an old trajectory or claim 100 s qualification.

## Independent ion elimination reference

The V45 ion comparison retains the full high/low words of the fixed electron
and hole quasi-Fermi inputs, positive-ion populations and trap occupancy. Its
independent absolute Poisson solve uses the frozen, pre-trial coordinate origin
to define the rebased carrier relation. That origin is an input definition;
the current trial potential, carrier populations, fluxes, rates and historical
residuals are not reference inputs. This verifies the rebased input problem,
not the global cold-start quasi-Fermi relation.

The independent reference assembles absolute Poisson charge and computes the
steric Scharfetter-Gummel flux and finite-volume rate in DD arithmetic. It
does not call the live ion-flux implementation. The same call's independently
eliminated binary64 potential supplies only the initial guess. A bounded
12-correction solve must converge below the existing 1e-28 V correction limit.
The reference supports the frozen single-positive-ion, zero-static-sheet,
zero-prescribed-jump model and rejects unsupported inputs.

Only the positive-ion flux and rate reference channels change. The original
public high-word difference, normalization floor, maximum reduction and 1e-6
acceptance threshold are preserved. The other eleven channels and the original
binary64 ion comparisons remain evidence. The saved ion-rate channel contains
the copied reference inputs, independently solved fields, residual and hashes.
This precision correction changes no live physical equation, Newton policy,
time grid, checkpoint representation or resource budget. Passing a fixed-state
reference check does not establish 100 s trajectory or refinement qualification.

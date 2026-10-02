# R1 Input Lift Continuation V1

The opt-in `r1-rebased-input-lift-dd-v1` representation can advance from an
explicit accepted state. This is an accepted-state continuation API; the
default baseline/pair protocol and cold-start preparation remain separate.

`from_accepted_state(system, previous)` creates a zero-coordinate reference.
It preserves all represented high/low words, including the incremental Gauss
residual anchors. It clears the preceding voltage lift; `advance_step` builds
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

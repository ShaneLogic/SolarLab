# Transient Photovoltage Open-Circuit Contract

This contract describes `experiments/tpv.py` and `experiments/open_circuit.py`.
It covers the production one-dimensional density MoL equations, including
their single-ion and dual-ion states. It does not certify charged explicit
defects, WKB channels, or the parked interface-plane charge model for MoL.

## Circuit and Conservation

All current components use the solar output sign convention. Let `D_s` be
`-junction_polarity * epsilon * E`, and let `J_c` include electron, hole and
both ionic charge currents. Open circuit requires `J_c + dD_s/dt = 0`.
The geometric area capacitance `C_A` is positive for both device orientations.

At fixed material and mesh, Poisson is affine in state charge and boundary
voltage. Its terminal displacement change is

`delta_D_s = delta_D_charge - C_A * delta_V`.

Radau integrates the ordinary density equations and one extra coordinate
`u = -delta_D_s / C_A`, with `du/dt = J_c(left_face) / C_A`.
The voltage used by every density RHS is
`V = V_initial + u + delta_D_charge / C_A`.
The cached Poisson factor supplies both displacement terms. Carrier currents
come from the same capped face currents used by the continuity equations.
The additional coordinate's finite-difference derivative uses a thermal
voltage reference; a difference based only on its initially zero value can
otherwise disappear when added to the DC voltage.

Acceptance checks instantaneous Maxwell current on every face and
independent 3/5-point Gauss-Legendre integration of conduction current over
each accepted Radau interpolant. The interpolants are split at output samples.
For each interval and face, the charge defect is
`integral(J_c dt) + D_s(end) - D_s(start)`.
Quadrature-order disagreement is included in the recorded defect.

The accumulated absolute charge defect divided by `C_A` is a charge-equivalent
voltage error, not a bound on all density, waveform, or material uncertainty.
Separate temporal and tolerance refinements remain necessary. The default
absolute limit is 1 microvolt, and a resolved pulse also requires this error
to be below 1% of its amplitude. The 0.05 A/m2 current limit is an additional
ceiling, not the sole weak-signal acceptance test.

## Preparation and Pulse History

Short-circuit light preparation has a declared finite duration; ionic
equilibrium is not inferred. A coarse voltage continuation locates a current
sign change. Every subsequent root candidate starts from the identical
lower-bracket state and executes both declared dwell intervals. A bracketed
Brent solve, bounded by the existing search iteration limit, returns the exact
state evaluated at the accepted voltage. Initial instantaneous fixed-voltage
Maxwell current must be below 1e-4 A/m2. Missing brackets, failed integrations
and unacceptable roots raise; the historical minimum-current fallback is not
an open-circuit state and is rejected.

Pulse turn-off is an integration boundary. Sampling never changes the pulse
duration or sets the internal step size. An unpulsed control starts from the
same accepted state and voltage and uses identical phase durations. Results
retain both protocols and numerical settings. The pulse response is the
difference of these two voltage traces, so slow preparation-dependent drift
does not silently become a pulse decay.

## Decay and Failure

Accepted states and quadrature samples must be finite and nonnegative, respect
ionic capacities, and pass the numerical-health diagnostics. Failed trajectories
raise through the backend job; no old state or fitted lifetime substitutes for
them. Returned validity arrays describe accepted numerical trajectories, not
external physical validation.

A single-exponential fit requires at least five resolved points, an observed
e-fold, consistent sign and monotone decay within the numerical noise scale.
The log fit must have R-squared at least 0.995 and maximum voltage residual
at most 2% of the response amplitude. Unresolved signals, short windows and
non-single-exponential responses retain a reason and `tau=null`. A valid TPV
voltage relaxation time is not automatically a unique microscopic carrier
recombination lifetime. Both frontend plot modes preserve this distinction.

## Verification

- `tests/unit/experiments/test_open_circuit.py`: dielectric capacitance,
  missing-voltage-history counterexample, known conductance-capacitance decay,
  sampling independence, failure rejection, and single/dual-ion continuity.
- `tests/unit/experiments/test_tpv.py`: physical workflow, bracket rejection,
  fit identifiability and pulse endpoints.
- `tests/integration/test_tpv_open_circuit_refinement.py`: no-ion and frozen
  dual-ion p-i-n examples; three independent time levels, three tolerances,
  and pulse fractions 0.02, 0.01, 0.005 at fixed physical duration. The last
  two waveform levels must agree within 1% of the refined pulse amplitude;
  identifiable fitted times must agree within 1%.
- The synthetic `tests/fixtures/configs/tpv_physical_reference.yaml` has
  common positive band gap and physical DOS, Boltzmann mass action, and
  compatible semiconductor-work-function contacts. It is not a fitted
  material, an AM1.5G reference, or an experimental validation.

The shared `VocSearchProtocol` retains historical serialized option names for
reading old records. New executions require `fallback=error` and
`warm_start=fixed_lower_bracket_state`; the same rule applies to Suns-Voc.

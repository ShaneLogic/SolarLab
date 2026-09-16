# R1 controlled DC and small-signal response V1

This research contract is implemented by
`perovskite_sim/experiments/one_dimensional_mechanism_r1_response.py`.
It implements the same-equation DC/AC prerequisite of study §7 and §10;
it does not close the full R1-2 numerical or time-window study.

## Operating state and DC equations

`solve_controlled_dc` restores the approved common state and fixed reference,
selects the existing A-D controls, and solves the controlled transient
operator's carrier rates, trap rate, ion rates, Poisson equation and local
interface equations at the requested bias. The same physical volumes,
contacts, interface traces and inventory are retained. Frozen ion and trap
coordinates are constrained to zero relative to the common state; their
stored populations and electrostatic charge are retained. Active blocking
ion components replace one redundant rate row by their original inventory.
Every original rate remains observable in the returned physical checks.

The Newton limit is 100 and the line-search limit is 40 through the existing
R1 policy. No continuation, increased budget, altered reference/material or
loosened physical criterion is hidden in this entry point. Row norms only
condition the Newton system. Acceptance requires the unchanged physical DC
gates, inventory and full Gauss limits of 1e-10, exact frozen populations,
and the dimensionless conditioned residual no greater than 1e-12. The full
iteration history and physical residuals are retained; failure raises
`R1ResponseError` with its partial evidence.

`dc_conductance_study` independently solves +/-0.1, 0.05 and 0.025 mV around
the zero-bias common state. It reports all six states and the three central
differences. The finest pair uses 1e-8 S/m2 + 1% of the larger magnitude.
The reported continuity indicator divided by the voltage difference is a
diagnostic, not a proved absolute error bound. A smooth three-point result
does not prove grid convergence or time-domain linearity.

## Coupled linear equations and observations

For exp(+i omega t), the exact state-coordinate storage and rate Jacobians
already exposed by the controlled transient operator form M and A. The
Poisson and local algebraic rows have zero storage. The complex solve is

    (i omega M - A) u_hat = b - i omega m_V.

Only voltage forcing and output observations use central differences; their
three half widths are 1e-5, 5e-6 and 2.5e-6 V. The state-observation coordinate
half widths use those same dimensionless numbers. Actual maximum physical
population, occupancy and potential perturbations are reported by coordinate.
These numerical derivative increments are not experimental AC amplitudes.
State-coordinate finite differences also independently compare the direct
tangent using dimensionless equation-scaled column norms. Elementwise ratios
are preserved separately: tiny derivatives obtained by cancellation of large
rates can be unresolved even when the whole column is accurately determined.

The same frozen-population and component-inventory constraints apply to all
frequencies, including zero. The solve retains the residual of the original
unreplaced equations as well as the constrained equations. Full component
inventory perturbations in m-2/V, thermal-normalized errors, and the legacy
inventory response divided by total absolute population response are saved.

The output observes electron, hole and ion conduction independently at both
physical contacts, every internal face and both zero-volume interface traces.
The blocking contacts have zero ionic flux. Contact electron and hole
recombination corrections are opposite and do not add a new total-current
channel. Displacement is computed from the physical electrostatic charge at
these same locations and contributes i omega D_hat. Trap charge is already
included in electrostatics, and trap storage is not added a second time to
conduction. Local capture versus trap-storage balance is checked separately.
All currents use the study sign j = junction_polarity * J_x; units are S/m2.

An additional algebraic solve holds every dynamic population fixed across
an ideal step. Its displacement change gives the impulse capacitance. This
quantity stays separate from the regular current and DC conductance.

## Scope of checks

The result retains the existing 1e-10 linear backward-error threshold,
5e-4 physical-face relative spread, 2e-3 derivative refinement, 1e-3 capture
versus storage balance, 1e-7 decomposition and 1e-10 thermal inventory gates.
The legacy inventory gate is 1e-8; direct-tangent comparison uses the existing
3e-4 transient Jacobian-check limit. The zero-bias real part is reported as a
passivity sign observation, with the declared 1e-8 S/m2 absolute numerical
scale. It is not applied to arbitrary nonequilibrium release states.

Passing frequency points remain `numerically_eligible_frequency_points`.
They do not establish that the frequency window covers all relevant modes,
that spatial refinement passed, or that a finite voltage step is linear.
`frequency_window_complete` and `double_domain_consistent` therefore remain
false. Evidence needed to change those research conclusions is explicitly
listed in the result and must come from the full study.

`compare_transient_tail` compares a same-preparation, same-control, same-bias
finite-time state with the independently computed DC state on the same grid.
It checks carrier log differences and potential changes in the bulk and at
both interface traces, occupancy changes, ion distribution and regular
terminal current against study §10. The ionic centroid uses the same exact
first moment of the conservative spatial reconstruction in each active
component as the spatial comparison module. It keeps the
omitted infinite-time integral bound unknown even when all endpoint values
agree. Endpoint agreement cannot exclude a weak unresolved slow mode.

`compare_reconstructed_response` compares real and imaginary admittance
components using their actual common frequency axis and the §10 component
tolerances. An unknown early interval, tail or interpolation error remains
unbounded through the existing reconstruction module; matching central
values alone cannot pass the error-budget check. The other study conditions
(two smallest amplitudes, independent refinements, T and 10T, earlier start,
stricter integration and tail-state/DC agreement) remain required.

## Verification

Independent analytic resistor, capacitor and one-pole descriptors check
signs, units, algebraic constraints, voltage-dependent storage and the
displacement term. The pole is compared with the separate finite-window
integration routine, using analytic early/tail and interpolation bounds.
Physical N16 checks use the approved reference and original material: A-D
biased DC retains frozen species and inventory; the three central DC steps
are compared with direct zero-frequency response; AC checks include all
physical faces, trap capture/storage, direct tangents and impulse capacitance.
These tests check discrete equations, not real-material validity.

## Bounded development study runner

`scripts/run_one_dimensional_mechanism_r1_physics_study.py` connects the
prepared-state, controlled transient, physical recomputation, DC/AC and fixed
position comparison APIs. It explicitly records development execution and
does not claim the formal controlled launcher's source assurance. The source
snapshot, runner hash and exact inputs must match when resuming an output.

Examples from the `perovskite-sim` directory:

```sh
python scripts/run_one_dimensional_mechanism_r1_physics_study.py --output-dir outputs/r1_physics --section short
python scripts/run_one_dimensional_mechanism_r1_physics_study.py --output-dir outputs/r1_physics --resume --section matrix
python scripts/run_one_dimensional_mechanism_r1_physics_study.py --output-dir outputs/r1_physics --resume --section long
python scripts/run_one_dimensional_mechanism_r1_physics_study.py --output-dir outputs/r1_physics --resume --section dc-ac --section amplitude --section compare
```

Sections can be selected independently; needed preparations are dependencies.
`--case-filter` selects case names and `--max-cases` bounds newly started cases.
Prior successful or failed cases are verified and retained. Explicit
`--retry-failed` writes a new attempt directory; interrupted attempts also
remain intact when resuming. Every solver row is appended and flushed to a
JSON-lines trajectory before returning to the solver. Successful trajectories
are reconstructed physically; failed prefixes are audited when sufficient
state-coordinate evidence exists. Each case has a request, result or failure,
completion record and content manifest, and the study has a failure index.
Unavailable comparison inputs are recorded separately from a measured
numerical disagreement; their absence cannot produce a passing comparison.

The matrix section runs all 27 independent settings on the *short* window.
Its comparison section checks 54 adjacent pairs, changing only one numerical
axis at a time while holding the other two fixed. It compares fixed physical
positions, both regular terminal currents, and impulse-inclusive integrated
charge. Each grid's independently computed zero-bias current is subtracted
from current and from accumulated baseline charge. This does not establish
full-window convergence. The separate long section attempts the declared
N16 D 100-second window and preserves its failure if it cannot finish.

The DC section contains N16/32/64 A-D target-bias states and the three central
conductance differences. AC evaluates D on those three grids at the 45 study
frequencies plus zero, and the comparison section compares N16/32 and N32/64
real and imaginary components independently. The amplitude section computes N16 D short responses
at 10, 5 and 2.5 mV; its normalized comparison keeps the absolute numerical
error budget unknown and does not certify linearity. The run's completion
status and the scoped scientific-check outcome are separate, and the full
R1-2 study exit remains false until its outstanding studies and independent
acceptance are supplied.

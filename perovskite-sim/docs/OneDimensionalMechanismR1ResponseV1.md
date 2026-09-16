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
3e-4 transient Jacobian-check limit. The zero-bias real part of the actual
published admittance must be at least -1e-8 S/m2, using the existing absolute
numerical scale. This dissipated-power sign check participates in every
eligible-frequency verdict. No positive-sign condition is imposed on the
imaginary part or on nonequilibrium release states.

Passing frequency points remain `numerically_eligible_frequency_points`.
They do not establish that the frequency window covers all relevant modes,
that spatial refinement passed, or that a finite voltage step is linear.
The direct solve leaves `frequency_window_complete` and `double_domain_consistent`
false. The separate comparison derives per-frequency dual-domain eligibility
only from matching source/preparation/control/grid/operating-voltage identity,
verified AC content, and every independently supplied study prerequisite.
Absent prerequisites remain incomplete, even when the central curves agree.

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

## Response content verification

`assess_small_signal_response` rederives the per-frequency checks from the
actual published response and all three saved derivative levels. It checks
that each level equals conduction plus i omega displacement and that the
headline uses the finest level's left contact. Stored flags must agree with
the rederived flags; they cannot replace them. Both constrained and original
complex linear residual vectors are saved alongside their backward errors.

`verify_response_content` rebuilds DC, the three DC conductance differences,
or the complete AC calculation from a separately verified preparation,
source, reference and request. It compares every scientific field, including
states, outputs, metric values and booleans. Exact numeric replay is scoped
to the frozen execution runtime. The caller must independently anchor and
verify the input artifacts before invoking it; these numeric helpers do not
grant source provenance. Formal DC imports require the caller's external
preparation digest, forwarded through all conductance solves. The Richardson
conductance is an additional central-difference diagnostic, not an absolute
error bound or a replacement for the frozen study tolerances.

## DC amplitude endpoints

The native study section `amplitude-dc` runs `dc_amplitude_endpoint_study` for
the requested grids and controls. Each `R1DCEndpointAmplitudeStudyV1` record
contains an independently solved zero-bias baseline and the seven declared
positive amplitudes from 10 mV through 0.15625 mV. Every endpoint retains its
full DC state, physical checks, actual voltage, preparation, reference and
source. The published endpoint response is `(j_dc(a)-j_dc(0))/a` in S/m2.
The six adjacent-halving diagnostics report differences and their ratios to
the 1% response scale; those ratios are not linearity acceptance results.

No independent absolute current-error budget is supplied by this endpoint
study. Its error-budget fields remain `None`, and both `linearity_certified`
and `full_transient_linearity_certified` remain false. `dc_states_certified`
only reports successful discrete DC solves. A close pair of endpoints cannot
certify the intervening transient or satisfy the R1-2 amplitude-linearity gate.

Formal verification rebuilds the baseline and all seven DC endpoints under
the bound amplitude/grid/control/preparation/source request and compares every
scientific field, including the normalized responses and diagnostic flags.
It cannot accept a changed DC value, a relabeled amplitude or a different source
by resealing the artifact. A failed solve retains the completed endpoint prefix
and identifies the amplitude whose DC solve failed.

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

The original V1 matrix section ran all 27 independent settings on the *short* window.
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

## Version-two study execution and verification

The controlled launcher now accepts `--runner physics-study`. Combined with
the study's `--formal` flag, this executes its committed package and runner
bytes through the same frozen-source loader as the stage-one path. Direct
execution remains explicitly development class. This execution class alone
does not assert physical convergence or independent approval.

`--grids` declares the ordered subset of 16,32,64,128,256; `--matrix-controls`
declares D and optional A-C axis crosses. D retains the full Cartesian matrix;
extra grids are extensions, not replacements for missing base cases.
`--window full`, `--first-time-s` and `--last-time-s` select the existing
versioned logarithmic grid. `--extended-frequency` includes zero and the
protocol-limited 1e-6..1e10 Hz grid. These switches do not change the physical
budgets or allowed iteration caps.

Each case stores its exact request, outcome, bound input attempts and manifests,
and historical failure observation. The version-two failure register extends
the original registers without waivers. Preparation explicitly uses `r1_policy`
and its fixed 100/40/2 caps. Resume and `--verify` on a formal study require the
caller-selected root `--manifest-sha256`; verification additionally replays
physical contents and comparisons. Re-sealing a changed result does not make
it scientifically valid. Verification is read-only. The source commit,
declared grid/window and other study choices must match the original request.

An invocation's exit status and the study's required finest-level conditions
are separate: failed diagnostics remain in the failure index with nonzero
status, even when a finer comparison succeeds. Missing or unresolvable
dependencies yield an incomplete result. Coarser comparison failures do not
replace the specification's finest-pair rule. The study report never asserts
independent acceptance.

The amplitude section executes separate space/time/nonlinear crosses and
uses the sum of measured finest-pair current differences as an empirical
uncertainty estimate. It records coarser differences without treating them as
the finest-pair criterion. Halving continues until an eligible pair is found
or the declared minimum amplitude is reached. Window extensions use the
verified qualifying amplitude when available; a fallback 5 mV observation is
explicitly diagnostic and does not acquire a linearity prerequisite.

The reconstruction section binds its actual step, DC, AC, amplitude and
window evidence. It checks T/10T, earlier-start and stricter-quadrature
differences, and the same-amplitude DC tail. Raw-current uncertainty is kept
separate from baseline uncertainty; conductance uncertainty requires spatial
and derivative refinements. Unknown early/interpolation/infinite-tail/impulse
error bounds remain unknown and cannot be replaced by zeros. Frequency
reports distinguish numerically eligible sampled bands from the separate
same-state turnover-coverage requirement. Thus implementing this pipeline
does not imply that any frequency already has a complete double-domain budget.

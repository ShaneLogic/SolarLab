# R1 Fixed-Position Spatial Reconstruction V1

The pure adapter in `one_dimensional_mechanism_r1_spatial.py` implements spatial
sampling for the archived `OneDimensionalMechanismR1StudySpecV1.md` §10.2.
It does not solve equations, alter a trajectory, verify an artifact's provenance,
or certify convergence of a campaign. Inputs must retain their separate
preparation, source, physical-check and stage-acceptance evidence.

## Supported geometry and sampling

The R1 adapter supports the current two homogeneous electrical layers with
Maxwell–Boltzmann carrier statistics. It rejects graded layers, substrates,
inconsistent physical faces/widths and malformed state arrays. The generic SG
and conservative-ion primitives can also be tested independently.

Interior positions are exactly `x_left + k*L_layer/18`, `k=1..17`, in each
layer. These 34 positions do not depend on the numerical grid. Interface
positions are represented separately, with explicit `left` and `right`
components. No interpolation crosses the physical material interface.

For the current R1 snapshot, `trace_state_m3` is the local canonical order
`[n_left, p_left, n_right, p_right]`; it is not the aggregate QSS right-first
array. Saved local `trace_potential_V` is `[phi_left, phi_right]`.
The adapter inserts the appropriate exact trace as the endpoint of each
layer's carrier/potential interpolation. Outer physical contact nodes are
retained. This also covers sampling between the nearest bulk node and the
physical interface without borrowing a density from the other material.

## Carrier and potential interpolation

`sample_sg_density` wraps the existing tested
`validation.foundation_reference_refinement._sample_sg_density` primitive.
It reconstructs the constant-face-flux solution on each interval for a
linearly varying transport potential. Electrons use positive drift sign;
holes use negative drift sign. Constant within-layer affinity, bandgap and
DOS offsets cancel in potential differences, so the homogeneous-layer
adapter can use the electrostatic potential directly.

The interpolation weight at interval fraction `f` is
`w = f*B(xi)/B(f*xi)`, where `xi = sign*(phi_right-phi_left)/V_T` and `B` is
the stable Bernoulli function. Density is `(1-w)*n_left+w*n_right`.
Zero field reduces to linear density, not linear log-density. Zero current
reproduces a Boltzmann exponential. Independent manufactured solutions and
subinterval flux checks pin the two signs and nonzero-current behavior.

Electrostatic potential is linearly interpolated separately in each layer,
using the same physical-side trace endpoints. Positive carrier inputs below
`1 m^-3` remain valid; there is no numerical population floor. Nonfinite or
nonpositive sampled carrier densities raise rather than being clipped.

## Conservative ionic reconstruction

`ConservativeIonProfile` interprets each saved ion density as the average of
its physical control volume. This preserves the simulator's stored population
`sum(P_i*width_i)`. The reconstruction within cell `i` is
`P_i + slope_i*(x-center_i)`, where `center_i` is the physical cell midpoint.
Its correction integrates to zero in that cell.

Interior slopes use a minmod limiter on adjacent secants and the centered
secant; end slopes are one-sided. Slopes are additionally bounded so both
reconstructed cell endpoints are nonnegative. Density averages are never
clipped, rescaled or renormalized. Slopes do not use a neighbor across a
material interface or an active-ion-component boundary.

`sample(positions, side=...)` requires an explicit side when choosing which
limit is desired at a discontinuity. The default is the right limit; R1
interface output explicitly evaluates both. `moments(left,right)` integrates
the piecewise linear reconstruction analytically, including partial cells.
It returns `integral(P dx)` in m^-2 and `integral(x P dx)` in m^-1.

The ionic centroid is their ratio, in metres, over each active connected
component. This is the centroid of the declared conservative reconstruction;
it is not the nodal quadrature `sum(x_node*P_i*width_i)/sum(P_i*width_i)`.
Physical cell centers differ from nodes near boundaries and on nonuniform
meshes. This distinction and the reconstructed inventory are retained.
The reconstruction conserves cell populations but is a spatial approximation;
its accuracy still needs independent refinement checks.

## Step adapters and comparison

`sample_r1_state(prepared, state=None)` returns absolute sampled fields and
ionic moments, defaulting to the saved `0-` state. Arrays are defensive
read-only copies. `spatial_responses_from_step` verifies the preparation,
reference and interval identities, then selects one finest accepted snapshot
at each exact declared output time. Missing/duplicate matches raise. It never
interpolates time or substitutes a nearest accepted state.

Each response uses that same grid's own `0-` baseline:

| Quantity | Response supplied to the existing comparator |
|---|---|
| Potential | `phi(t)-phi(0-)` in V |
| Carrier | `ln(n(t))-ln(n(0-))`, likewise holes |
| Ion density | `(P(t)-P(0-))/P0` |
| Interface occupancy | `f(t)-f(0-)` |
| Ionic centroid | `centroid(t)-centroid(0-)` in m |

These state responses are not divided by voltage amplitude. A time coordinate
of zero denotes `0+`, and may have a nonzero potential response from `0-`.
Carrier and ion interface-side comparisons remain separate from interior
positions. The independent DC potential comparison uses the saved `0-`
potentials at the interior positions and both interface-side limits;
absolute carrier and ion profiles remain available from
`sample_r1_state` but are not silently assigned a new DC acceptance rule.

`compare_spatial_responses` requires equal stack/reference/source identities,
control, voltage amplitude and exact physical coordinates/times. It calls
the existing §10.2 pointwise comparator without changing any tolerance.
Both numerical policies and spatial intervals are returned, so the campaign
must establish whether only the intended numerical axis changed. Passing
this comparison does not itself prove that axis independence.
The source identity is compared when present in the saved records. Synthetic
pure-adapter fixtures may omit it; absence does not establish provenance.

No independent carrier uncertainty is supplied to this adapter; the result
explicitly leaves carrier signal resolution undetermined. A small difference,
zero difference, or passed absolute log-density rule cannot establish that a
weak carrier mechanism signal is resolved. A failed input trajectory is not
promoted to a certified one by successful reconstruction.

The analytic tests cover constant field, equilibrium, nonzero SG flux,
nonuniform-cell inventory and first moments, positivity limiting, interface
discontinuities, fixed coordinates and response normalization. A separately
marked real-solver test exercises a prepared R1 state and a short trajectory.
Full-window, amplitude, time, space, nonlinear and DC/AC campaign gates remain
outside this module.

## Single-axis convergence verdict

`compare_convergence_responses` is the separate acceptance-facing comparison.
It requires an explicit spatial, time or nonlinear axis and validates adjacent
coarse-to-fine settings against the frozen R1 policies. Exactly that axis may
change; all physical identities and actual output times must agree. Thus a
time or nonlinear comparison correctly uses the same spatial grid, while a
self-comparison cannot become a convergence pass. The pure comparator can
still report zero differences but always reports `convergence_passed: false`.

The acceptance-facing function requires the complete eight-quantity inventory:
DC potential, response potential, carrier log-density change, ion distribution
change, trap occupancy change, ion centroid change, regular current and
impulse-inclusive integrated charge. Every interior/interface-side report and
both electrical reports participate in the verdict. Missing quantities fail
validation. Both inputs must also have successful independent content/physics
verification; passing reconstruction never promotes a failed input. The caller
must bind those verification reports to the exact source-anchored artifacts.

This is a verdict for one comparison. It does not by itself establish complete
time-window coverage, the full three-axis campaign, a resolved weak mechanism
signal or the R1-2 stage exit.

## Complete electrical response construction

`step_current_charge_responses` constructs baseline-subtracted regular
current at every declared output time and both physical contacts, and charge
from the impulse plus the unique finest-level integrated charge at each
time. A caller-fixed expected time axis rejects omitted interior samples.
`compare_step_current_charge` applies the unchanged component budgets to
those complete arrays. Backward-Euler averages cannot replace the regular
current samples, and an equal final point cannot hide an earlier failure.

Single-axis comparison validates that the declared input parameter changes.
Equal responses after a genuine tolerance change indicate insensitivity in
the tested scope; they are not a literal self-comparison and are not
independent repetitions of a grid result determined by the same initial
current. DC bulk and interface-limit reports are subreports of the same
`dc_potential` required quantity.

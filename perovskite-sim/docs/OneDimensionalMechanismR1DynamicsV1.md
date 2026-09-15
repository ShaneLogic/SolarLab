# One-Dimensional Mechanism R1 Dynamics V1

This is the execution contract for the R1-1 common-state, A-D control and
ideal-step APIs and `scripts/run_one_dimensional_mechanism_r1_stage_one.py`.
It implements the bounded requirements of sections 4–6 of the preregistered
`plans/Specs/OneDimensionalMechanismR1StudySpecV1.md`, SHA-256
`f140505c3a7ec7c52fe38c46f36d53b876a87c2ee50512568cb62671b05c6a73`.
The R1-0 implementation parent is `316e7046f85f94d08fe2350f45664ee433318810`.
The complete study, convergence matrix, long-time completion, finite-amplitude
linearity and time/frequency comparison remain later-stage work. A successful
R1-1 execution cannot acquire those labels.

## Fixed physical input and reference

The only supported input is the versioned
`reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json`. Its physical
source is the unchanged fixture
`tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml`,
SHA-256 `f617f230b2d9c144573394e38fcc313225dc84c7b38f7670b97e9a0a7cc12a24`.
The fixture describes a synthetic numerical reference: one dimension, 300 K,
dark, one two-sided interface, one positive-ion population and background bulk
SRH. The controls are recorded separately from the material input.

R1-1 imports an existing validated `ReferenceBindingV1` from R1-0. It never
prepares a new reference. The versioned input pins
`fixed_reference_binding_sha256` to the approved binding-content identity
`0a6532a436dd07e6b27f01da3fc39aef906e6fd882057f0109c1bc5bdf77e39b`.
Preparation and import first validate the binding's internal hash, stack,
geometry and reference ladder, then compare its identity with this approved
value. CLI `prepare`, `zero-check` and `step` and the common-state APIs share
this requirement. A different self-consistent reference and a matching
externally prepared state are rejected before solving or importing them.
JSON whitespace and key order do not change binding identity; the separately
recorded file-byte hash continues to identify the exact artifact.
The reference occupancy remains fixed in
`sigma_t = -q N_t,s (f - f_ref)` for every grid, control and excitation.
The R1-0 geometry contract defines the reference ladder, binding validation,
physical control volumes and independent geometry evidence. A finite-grid
equilibrium may have nonzero initial sheet charge; it is retained.

Physical ion density, site capacity, compensation background and trap density
are unchanged by the controls or mesh. The physical geometry identity is
`physical_boundary_two_sided_v1`; each active layer inventory is evaluated
with the same widths used by its rate equations, DC constraints and charge
integrals. The mobile positive-ion inventory is `1e15 m-2` and the areal trap
density is `5e14 m-2`. External ion flux is zero. Right-layer positive ions and
both layers' compensation backgrounds remain present and fixed.

## One D preparation, copied into A-D

`prepare_common_state` solves the complete D system at dark zero bias using
the fixed reference and the selected physical grid. The saved
`R1CommonStateV1` is a certified equilibrium at `0-`, not an unspecified
finite-time preparation. Formal CLI grids are 16, 32 and 64 intervals. Each
grid solves the same physical preparation problem independently. A-D on one
grid use copies of exactly one prepared record.

The record retains the full stack, effective carrier contacts and their
thermodynamic certificate, reference binding, grid coordinates, control-volume
faces and widths, interface position, active ion nodes and connected
components, background charge, `n,p,P,f`, sheet charge and component
inventories. It also retains potential, both local interface carrier and
potential traces, capture fluxes, residuals, QF references, DC certificate,
preparation history, source file hashes, environment and a content hash.
The fixed carrier reservoirs remain those resolved from the source input.

Import verifies the record hash and the physical, grid, reference, contact,
history, specification and executing-package identities. It reconstructs and
compares the physical arrays, then evaluates the current D equations and the
selected control's remaining equations. An old `certified` flag is not an
import certificate. Executing-source identity includes the versioned dynamics
input as well as the package sources. A changed grid or reference, altered population, stale
package source, or excessive live residual is rejected. Import performs no
new DC preparation, population interpolation, or silent reset of sheet charge.
Local algebraic equations may be re-evaluated while the supplied dynamic
populations remain fixed.

The `zero-check` stage imports this same D state for all A-D by default.
It checks the remaining carrier and ion equations, inventory, trap capture
balance, local algebraic/Gauss residuals, full-domain Gauss law and the
eliminated operator. Near-zero equilibrium currents use the policy's absolute
DC continuity and face-current bounds. A relative current-spread statistic
normalized by a vanishing signal has no useful equilibrium precision meaning;
R1-1 does not relabel it as a passed `2e-6` relative-current check.

## Explicit A-D equations

For each active ion cell, `w_i dP_i/dt = nu_I (F_left - F_right)` and
`J_I = nu_I q F_I`. For the shared areal trap,
`N_t,s df/dt = nu_t (Gamma_n - Gamma_p)`. The carrier equations use
`nu_t Gamma_n` and `nu_t Gamma_p` separately, with negative net capture
representing release. The same occupancy enters storage, capture and sheet
charge. Four microscopic capture/release channels retain their original
left/right order, parameter units and emission references.

| Control | `nu_I` | `nu_t` | Retained dynamics |
|---|---:|---:|---|
| A | 0 | 0 | Carriers and background bulk SRH |
| B | 1 | 0 | A plus ion redistribution |
| C | 0 | 1 | A plus trap exchange and charge storage |
| D | 1 | 1 | Complete coupled system |

A/C preserve the prepared ion distribution pointwise, its electrostatic
charge and the compensation background. A/B preserve `N_t,s`, `f_init`,
`f_ref` and the initial sheet charge while disabling all microscopic trap
capture/release channels and trap evolution. Interface band offsets,
thermionic transport and background bulk SRH remain active. The controls
multiply rate, current and derivative paths explicitly; no YAML diffusion
coefficient or capture cross-section is set to zero to impersonate a control.

Required control evidence includes unchanged frozen populations, zero
disabled capture, nonzero initial sheet-charge retention, D's local
capture/storage balance and a fixed-reservoir single-trap exponential oracle
using the same microscopic channels. The local oracle supplements the
self-consistent device checks. `D-B` includes both trap recombination and
storage; it cannot all be called trap memory. `D-B-C+A` is a nonadditivity
diagnostic only after preparation and observation identities match. The E
quasi-steady approximation is outside R1-1 and has no control entry here.

## Ideal step, current direction and the short trace

Let `s = junction_polarity`, with `phi(L) = V_bi_bc - s V_app`.
Physical current `J_x` is positive along increasing x. The reported current
conjugate to applied voltage is `j = s J_x`; no absolute-value operation
changes its sign. Positive voltage changes produce positive capacitive
charging in this convention.

The initial event explicitly separates `0-` and `0+`. At `0+`, all dynamic
populations `n,p,P,f` and sheet charge retain their `0-` values. The code solves
the new fixed-population Poisson/interface algebraic conditions, saves the
contact charge jump and then starts integration from that right-limit state.
The first finite-time backward-Euler displacement current is a step average;
it is never substituted for an instantaneous current peak.

The fixed-population dielectric capacitance is
`C_infinity = (integral(dx/epsilon))^-1 = 4.4270939085e-4 F/m2`, using the
source `EPS_0 = 8.854187817e-12 F/m`. The ideal-step impulse charge is
`Q_imp = C_infinity * amplitude`; at 5 mV it is
`2.21354695425e-6 C/m2`. Both physical contact displacement jumps independently
check this signed impulse, while the full device charge remains continuous.
This oracle assumes the source's dielectric model with no additional dipole,
external circuit capacitance or high-frequency dielectric dispersion.

The regular current is obtained from the post-step state and its differential
rates at fixed voltage, excluding the mathematical voltage impulse. The
evidence separates that regular current, the ideal-step event and finite-step
averages. Physical contacts, both interface traces and ordinary faces retain
their measured decompositions and conservation checks; no postprocessing
projects them to a common current.

The R1-1 functional trace is `[0, 1e-9, 1e-8, 1e-6, 1e-4] s` and defaults to
a 5 mV step with 1/2/4 backward-Euler subdivisions. The time-zero trace entry
is the post-step regular state; the separate event stores `0-` and `0+`.
The CLI permits a finite positive amplitude below 20 mV, recording its exact
value without claiming linearity. The ten documented nonlinear tolerance
fields use factor 0.1 by default; selectable factors are 1/0.1/0.01/0.001.
Physical acceptance limits remain unchanged. The full `1e-9..1e2 s` window,
window extension to `1e5 s`, early extension, amplitude halving, target-bias
DC comparison and 27-combination convergence matrix belong to R1-2.

Transient physical gates retain Gauss and charge balance at `1e-10`, ion
inventory drift at `1e-10`, and nonzero-signal contact/internal total-current
spread at `2e-6`. The original nonlinear, local capture/Gauss, analytic
Jacobian, eliminated-operator, time-refinement and decomposition checks also
remain in force. Passing one short trace establishes only its recorded scope.

For every finite accepted step on every nested time grid, the local trap
storage error is explicitly checked and included in the final certificate.
For the single interface in this study, let
`E_t = q * abs(delta(N_t f) - dt * R_t)`, where `R_t` is the same controlled
net capture rate used by the carrier and trap equations. Preserve the
independently recorded `trap_storage_error_A_m2 = E_t / dt`.

The allowed step-charge error is
`B_t = min(q * eta * S_t, dt * epsilon_Q * J_scale)`.
`eta` is the original `maximum_scaled_nonlinear_residual`; `S_t` is the
actual trap-row storage scale used by Newton for that step, including its
absolute, relative and roundoff terms and original scaling reference.
`epsilon_Q` retains the existing charge-balance acceptance limit (at most
`1e-10`). `J_scale` is the existing charge-balance scale:
`max(abs(charge_rate), abs(Jc_left-Jc_right), max(abs(conduction)), 1 A/m2)`.
This applies the existing total-charge budget conservatively to one local
interface; it is an explicit additional local check, not a claim that the
study specification previously listed this separate limit. No tolerance is
chosen from the observed residuals, and no original physical gate is relaxed.

Each finite step saves its error, both budget components, the effective
charge/current limits, the normalized ratio `E_t/B_t` and the pass/fail result.
Non-finite errors or invalid budgets are rejected; the normalized ratio must
not exceed one. The final certificate aggregates the worst normalized ratio
over all nested levels and retains the maximum absolute current error as a
diagnostic; that maximum is not compared with an unrelated step's limit.
The `0+` record has no finite `dt` and labels this check not applicable.
Its algebraic and regular-current checks remain separate. A failed step is
saved before raising an error, with the specific trap-storage failure reason.

## CLI and sealed evidence

Run from `perovskite-sim` with the local environment. Use an already prepared
R1-0 binding, and a fresh output directory for each execution:

```sh
python scripts/run_one_dimensional_mechanism_r1_stage_one.py list
python scripts/run_one_dimensional_mechanism_r1_stage_one.py prepare --intervals 16 --reference outputs/one_dimensional_mechanism_r1/preparation_v1/ReferenceBindingV1.json --output-dir outputs/one_dimensional_mechanism_r1/common_state_v1
python scripts/run_one_dimensional_mechanism_r1_stage_one.py zero-check --intervals 16 --reference outputs/one_dimensional_mechanism_r1/preparation_v1/ReferenceBindingV1.json --prepared outputs/one_dimensional_mechanism_r1/common_state_v1/PreparedStateV1.json --output-dir outputs/one_dimensional_mechanism_r1/zero_check_v1
python scripts/run_one_dimensional_mechanism_r1_stage_one.py step --control D --intervals 16 --amplitude 0.005 --reference outputs/one_dimensional_mechanism_r1/preparation_v1/ReferenceBindingV1.json --prepared outputs/one_dimensional_mechanism_r1/common_state_v1/PreparedStateV1.json --output-dir outputs/one_dimensional_mechanism_r1/step_d_v1
python scripts/run_one_dimensional_mechanism_r1_stage_one.py verify --output-dir outputs/one_dimensional_mechanism_r1/step_d_v1
```

The runner observes single-thread BLAS, copies the canonical input, fixture,
contract, fixed reference and imported preparation, and records resolved
parameters, policy, hashes and execution environment. It reuses the R1 source
ZIP/manifest/patch and strict JSON/NPZ writer. Successful preparation produces
`PreparedStateV1.json`; zero checks produce `ZeroExcitationV1.json`; steps
produce `StepResultV1.json` plus incrementally saved `AcceptedStepsV1.json`.
Array-bearing payloads also generate compressed NPZ files.

A failed computation exits nonzero and seals `FailureV1.json`, completion
status, accepted records and any exception result. Non-finite failed numbers
are explicitly tagged in strict JSON and preserved in raw NPZ arrays.
The byte manifest covers every saved file except itself. `verify` checks
coverage, safe paths, byte counts and hashes and reports the recorded status;
a failed recorded run remains a nonzero verification result. Integrity
verification does not rerun equations or validate scientific claims.

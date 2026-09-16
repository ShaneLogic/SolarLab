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
These are allowed attempts, not a claim that every case converges or passes
physical checks. The versioned input records exact historical cases under
`known_nonconvergence` and, separately, `known_physical_gate_failures`, with
source commit, control, grid, amplitude, output times and time subdivisions.
At source 3dd7e0c, confirmed at 84f9131, all A-D controls at 16/32/64 intervals
fail during initialization at factor 0.001. Factor 0.01 also fails for D at
32 during integration and A-D at 64 during initialization. At 5 mV, B and D
at 64/factor-1.0 converge locally but fail the contact/internal-current gate.
The 48-cell short-trace slice contains 17 nonconvergence and two physical-gate
failures. A separate 16-cell amplitude probe includes two additional 10 mV
B/D/32/factor-1.0 failures, each failing charge balance and current spread.
These supplementary observations are not part of the 48-cell count.
These are observations at the named source, not predictions for untested
cases or later implementations. Repeated probes are deduplicated by their
full case conditions. The runner reports recurrence of the recorded failure
message substring; it does not compare the historical numeric metric values,
without skipping the computation, waiving a gate or changing its exit code.
The factor 0.001 remains the preregistered deepest extension; all failures
are retained. A new source or a different time window requires a new test.
Physical acceptance limits remain unchanged. The full `1e-9..1e2 s` window,
window extension to `1e5 s`, early extension, amplitude halving, target-bias
DC comparison and 27-combination convergence matrix belong to R1-2.

Transient physical gates retain Gauss and charge balance at `1e-10`, ion
inventory drift at `1e-10`, and nonzero-signal contact/internal total-current
spread at `2e-6`. The original nonlinear, local capture/Gauss, analytic
Jacobian, eliminated-operator, time-refinement and decomposition checks also
remain in force. Passing one short trace establishes only its recorded scope.

The current declaration in `OneDimensionalMechanismR1PhysicsProtocolV1.md`
freezes the current long-window disposition: the measured nu_I=1 tails are
not certifiable under the existing eliminated-operator criterion. In the
observed near-equilibrium states its ion-rate and ion-flux denominators fall
into a cancellation/roundoff regime. This is a criterion applicability issue;
the failed ratio alone is not evidence of a physical solver defect. The
existing 1e-6 gate remains enforced and these failures remain failures. No
absolute budget, normalization floor or replacement certificate is approved
by that decision. Future criterion changes require their own preregistered
error model, independent negative controls and review before formal use.

For every finite accepted step on every nested time grid, the local trap
storage error is explicitly checked and included in the final certificate.
For the single interface in this study, let
`E_t = q * abs(delta(N_t f) - dt * R_t)`, where `R_t` is the same controlled
net capture rate used by the carrier and trap equations. Preserve the
recorded finite-step `trap_storage_error_A_m2 = E_t / dt`.

The allowed step-charge error is
`B_t = min(q * eta * S_t, dt * epsilon_Q * J_scale)`.
`eta` is the original `maximum_scaled_nonlinear_residual`; `S_t` is the
actual trap-row storage scale used by Newton for that step, including its
absolute, relative and roundoff terms and original scaling reference.
`epsilon_Q` retains the existing charge-balance acceptance limit (at most
`1e-10`). `J_scale` is the existing charge-balance scale:
`max(abs(charge_rate), abs(Jc_left-Jc_right), max(abs(conduction)), 1 A/m2)`.
The first budget rechecks consistency with the declared Newton storage
equation. With the same equation and scale, its ratio equals the absolute
scaled trap-row residual divided by `eta`, so it does not independently
tighten an otherwise correct Newton acceptance. Solver and diagnostic share
the storage-increment and rate operators: a consistent error in those shared
operators can pass both arms. Previously observed detection of a particular
time-factor inconsistency does not imply detection of arbitrary incorrect
storage equations. Independent equation tests require a separate oracle.
The second budget
applies the existing total-charge budget conservatively to this single
interface; it adds a tighter local condition only when it is smaller than
the Newton budget. The `1 A/m2` normalization floor remains explicit.

The normalized limit one means compliance with the recorded policy, not a
fixed absolute physical precision. The Newton budget changes with nonlinear
tolerances, including their unscaled roundoff terms; the independent charge
budget is not multiplied by the nonlinear factor. R1-2 must keep numerical
consistency, physical error budgets and response convergence distinct. Any
new physical budget must be frozen before its runs. No tolerance is chosen
from observed residuals, and no original physical gate is relaxed.

Each finite step saves its error, both budget components, the effective
charge/current limits, the normalized ratio `E_t/B_t` and the pass/fail result.
It also records separate Newton-consistency and local-charge ratios, the
dominant budget, actual nonlinear tolerance fields, trap activity and whether
the current normalization floor is active. Summaries split branch counts
and maxima into all steps, active traps and nonzero-error subsets. These
counts describe the saved run, not universal evidence-strength percentages.
Non-finite errors or invalid budgets are rejected; the normalized ratio must
not exceed one. The final certificate aggregates the worst normalized ratio
over all nested levels and retains the maximum absolute current error as a
diagnostic; that maximum is not compared with an unrelated step's limit.
The `0+` record has `dt_s=0.0` and no positive elapsed integration interval.
Its legacy `trap_storage_error_A_m2=0.0` field is retained solely as an
explicitly labeled compatibility placeholder; the finite-step check is not
applicable and excludes it from summaries. Its algebraic and regular-current
checks remain separate.

Every row records solver acceptance separately from physical-check success.
All applicable gates, including their finite-status checks, are evaluated
before raising. Simultaneous violations retain every reason and raw metric.
The observer receives the complete row, including a failed row. If persistence
also fails, physical reasons remain primary and the write failure is recorded
separately; it cannot replace a Gauss/charge/trap failure with a JSON error.

## CLI and sealed evidence

Run from `perovskite-sim` with the local environment. Use an already prepared
R1-0 binding, and a fresh output directory for each execution:

```sh
python scripts/run_one_dimensional_mechanism_r1_stage_one.py list
python scripts/run_one_dimensional_mechanism_r1_stage_one.py prepare --development --intervals 16 --reference outputs/one_dimensional_mechanism_r1/preparation_v1/ReferenceBindingV1.json --output-dir outputs/one_dimensional_mechanism_r1/common_state_v1
python scripts/run_one_dimensional_mechanism_r1_stage_one.py zero-check --development --intervals 16 --reference outputs/one_dimensional_mechanism_r1/preparation_v1/ReferenceBindingV1.json --prepared outputs/one_dimensional_mechanism_r1/common_state_v1/PreparedStateV1.json --output-dir outputs/one_dimensional_mechanism_r1/zero_check_v1
python scripts/run_one_dimensional_mechanism_r1_stage_one.py step --development --control D --intervals 16 --amplitude 0.005 --reference outputs/one_dimensional_mechanism_r1/preparation_v1/ReferenceBindingV1.json --prepared outputs/one_dimensional_mechanism_r1/common_state_v1/PreparedStateV1.json --output-dir outputs/one_dimensional_mechanism_r1/step_d_v1
python scripts/run_one_dimensional_mechanism_r1_stage_one.py verify --output-dir outputs/one_dimensional_mechanism_r1/step_d_v1
```

These direct invocations explicitly produce development evidence. Controlled
execution uses the trusted launcher and a full source commit obtained outside
the output being verified. `R1_DEPENDENCY_PATH` names the trusted environment's
site-packages directory; its `.pth` files are not executed. For example:

```bash
python -I -S scripts/run_one_dimensional_mechanism_r1_controlled.py \
  --project "$PWD" --source-commit "$R1_SOURCE_COMMIT" \
  --dependency-path "$R1_DEPENDENCY_PATH" -- \
  prepare --intervals 16 --reference "$R1_REFERENCE" --output-dir "$R1_NEW_OUTPUT"
```

The same launcher prefix supports zero-check and step. These formal consumers
also require `--prepared-manifest-sha256` from the independently selected
passed preparation bundle. A standalone prepared JSON is not a formal input.
The complete parent is saved under `PreparationV1/`, including its manifest;
its source commit, execution class, protocol and full payload are verified
before import. Thus preparation checks, DC certificates and environment are
pinned even where physical-array reconstruction does not recompute them.
`execution_source()` includes execution class and commit/content identity.
Formal library imports require an externally selected prepared-payload digest;
the CLI obtains it only after checking the sealed parent. Source identity checks
can succeed for a developer candidate; independent approval remains separate.

The runner observes single-thread BLAS, copies the canonical input, fixture,
contract, fixed reference and imported preparation, and records resolved
parameters, policy, hashes and execution environment. It reuses the R1 source
ZIP/manifest/patch and strict JSON/NPZ writer. Successful preparation produces
`PreparedStateV1.json`; zero checks produce `ZeroExcitationV1.json`; steps
produce `StepResultV1.json` plus incrementally saved `AcceptedStepsV1.json`.
Array-bearing payloads also generate compressed NPZ files.

Formal R1 computation uses a controlled source-checkout launcher started by a
trusted interpreter with `-I -S`. Required source bytes must match blobs in
the explicitly supplied source commit; Git index flags, ignore rules and a
clean status report are not evidence of this equality. A constant in the
checked binding module pins both the complete research input and execution
contract digests. Updating either requires an explicit code/policy repin.
The launcher executes project modules from the verified frozen source bytes,
and archives the same bytes. It does not read project bytecode caches.
Dependency paths are explicitly supplied without site initialization or
processing `.pth` files. Merely recording PYTHONPATH or a loader does not
prevent a startup hook; `-B` prevents bytecode writes, not reads.

An unchanged clone of the allowed source content may be used, including one
inside an otherwise ignored directory. A clone's own HEAD does not grant
approval. Altered input/source content must fail the applicable content
checks regardless of location. `ExecutionSourceV1.json` distinguishes the
observed commit, supplied source anchor and execution class. The caller must
obtain approved source/input anchors independently. The trusted launcher,
interpreter, standard library, Git executable and dependencies remain trust
assumptions; no claim covers arbitrary replacement of them or memory mutation.

Direct CLI computation requires explicit `--development` unless entered by
the controlled launcher. Library calls in ordinary Python processes assume
a trusted caller and are development execution, without formal source
attestation. Development edits may be recorded; new package files must first
be declared to Git. Wheel-only computation remains outside this workflow.
General R0 engines retain structural validators; R1 study wrappers, including
future AC entry points, must enforce this study context before using them.

A failed computation exits nonzero and seals `FailureV1.json`, completion
status, accepted records and any exception result. Non-finite failed numbers
are explicitly tagged in strict JSON and preserved in raw NPZ arrays.
Accepted-step streams and failed-result JSON/NPZ files are replaced atomically
per file; this is not a transaction spanning an entire bundle. Completion records distinguish
observed rows, successfully persisted rows, finite solver steps and physically
passing finite steps. The legacy `accepted_record_count` now explicitly means
persisted rows, including initial records; it is not a scientific pass count.
When writing fails, the prior valid stream remains and `FailedResultV1` retains
the complete observed state whenever persistence is available.
All numeric leaves of preparation, physical, state and derived result records are checked
before hashing a successful result. Non-finite values fail with field paths;
None with an explicit not-applicable reason remains a valid missing value.
The byte manifest covers every saved file except itself. `verify` checks
coverage, safe paths, byte counts and hashes and reports the recorded status;
a failed recorded run remains a nonzero verification result. Integrity
verification does not rerun equations or validate scientific claims.

The default `verify --mode integrity` reports checksum consistency and recorded
status, explicitly leaving provenance unauthenticated. Rebuilding a manifest
after editing its bundle can satisfy this limited check. Acceptance identity
requires either `--mode acceptance --expected-manifest-sha256 DIGEST`, or
`--mode acceptance --ledger PATH --ledger-sha256 DIGEST --run-id ID`. The digest
must come from a separately trusted review channel or approved Git ledger,
not from the bundle being checked. A ledger must be outside the bundle and
match the independently supplied ledger digest. Its `R1EvidenceLedgerV1`
`entries[ID]` pins `manifest_sha256`, `stage`, `recorded_status`, `stage_scope`,
`source_manifest_sha256`, `study_input_sha256` and `reference_binding_sha256`.
Revision-4 acceptance reads and checks a ledger's top-level `source_commit`
and any per-entry `source_commit`, rather than merely displaying them. It requires
`--expected-source-commit` from the caller, or the top-level source commit in the
separately anchored ledger. Source bytes are compared with that exact commit's
Git blobs in the trusted verifier repository, independent of HEAD/index flags.
For consuming stages the caller also supplies `--prepared-manifest-sha256`;
the nested preparation bundle must match that anchor and the same source.
The runner checks those identities against the anchored bundle. Acceptance
defaults to externally selected evidence revision 4: a bundle cannot downgrade
the required revision by deleting or changing its own field. Explicit
`--required-evidence-revision 1`, `2` or `3` requests explicitly labelled legacy
verification without claiming revision-4 source or preparation-chain checks.
Revision 3 checks
required-source coverage, source ZIP contents, frozen input and contract bytes,
canonical reference payload identity and protocol identity agreement.
Revision 4 additionally checks the verifier's fixed contract digest, the
external source identity, source execution class and the parent preparation
chain. All parent payload bytes and source artifacts remain in the child
manifest. These
checks inspect anchored records; they do not independently observe a past
process or turn an untrusted caller-supplied digest into approval. A recorded
failed run still exits nonzero even when its bytes match the anchor.

Anchor matching does not itself grant independent approval, authenticate an
arbitrarily replaced verifier, rerun physical equations or certify convergence.
Candidate ledgers generated during development require separate approval
before serving as formal study trust anchors. Signatures may be added for
distribution identity later; they do not replace equation or convergence checks.

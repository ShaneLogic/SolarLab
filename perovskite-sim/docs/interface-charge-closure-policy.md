# Interface-charge closure policy

Status: D4-E3b protocol-bound charged quasi-Fermi J-V backend/frontend
integration implemented and tested. The exact illuminated zero-scan-rate slice
is available through `/api/jv` and `/api/jobs`; all other charged experiment
routes remain `PARKED`. The capability is not production-certified until the
D4-E3c source-clean matrix completes (2026-08-29).

D4-E0 (2026-08-28) added the canonical per-area microscopic document described
in `docs/explicit-interface-defect-schema-v1.md`. D4-E1 now requires that
document and rebuilds uncalibrated capture velocities from it in the QF dark
reference, backend endpoint, charged refinement, and stress adapters. D4-E2
minted complete charge-off, charged, and resolved device-stress certificates at
source commit `851406a`; earlier certificates remain historical baselines only.

SolarLab's default and general interface-state paths remain
recombination-only. The explicit research API and the narrowly gated charged
J-V route below provide a self-consistent occupancy-dependent sheet charge in
the QF outer Poisson system. The current v2 charged lane and v3 resolved-stress
lane are internally numerically certified. That status does not promote the
new J-V route to a production-certified or externally validated model.

## Configuration contract

The device schema recognizes two values:

```yaml
device:
  interface_charge_closure: "off"
```

and the reserved research intent:

```yaml
device:
  interface_charge_closure: "equilibrium_referenced"
  interface_charge_rebaseline_acknowledged: true
```

`off` is the default and preserves the historical material arrays and RHS.
The acknowledgement records that activating electrostatic trap charge
invalidates the historical SCAPS calibration. A charge-off reference config may
set it while establishing the fresh baseline; doing so changes its semantic
identity without enabling charge. It is not a general enable switch: ordinary
material assembly and every backend experiment other than the exact D4-E3b J-V
slice reject `equilibrium_referenced` with a `PARKED` capability error. The
research endpoint and charged J-V dispatcher both build charge-off material
arrays internally and thread the signed charge through the separate charged QF
system; neither enables the legacy production material path.

## Charge convention

For electron occupancy `f`, both supported trap characters have the same
equilibrium-referenced increment:

```text
acceptor: Delta sigma = [-q Nt f] - [-q Nt f_eq]
                      = -q Nt (f - f_eq)

donor:    Delta sigma = [q Nt (1-f)] - [q Nt (1-f_eq)]
                      = -q Nt (f - f_eq)
```

`equilibrium_referenced_interface_trap_charge()` returns this signed quantity
directly in C/m2. It exposes no arbitrary `-1/+1` multiplier. Absolute trap
charge is intentionally not implemented because it would additionally require
a neutral-occupancy convention, fixed countercharge, and whole-device charge
neutrality.

The legacy `MaterialArrays.iface_state_charge` scalar is retired. A manually
constructed non-zero value fails before Poisson rather than depositing charge
on the shared interface node.

## Local charged-Gauss primitive

The research-only two-sided element now exposes an explicit
`EquilibriumReferencedSheetCharge` contract and evaluates the coupled local
coordinates
`(phi_L, phi_R, log n_L, log p_L, log n_R, log p_R)`. Its Gauss row includes
the signed incremental sheet charge and an analytic
`d Delta sigma / d log(state)` obtained from the shared SRH occupancy law.
Central-difference tests cover both the occupancy derivative and the complete
2x6 electrostatic tangent. Separate positive/negative sheet-charge cases close
the Gauss law across unequal left/right permittivity to a normalized residual
below `1e-10`.

This primitive is not a device solve and does not unlock the configuration.
The local physics layer now also solves the two electrostatic equations and
four carrier balances jointly. Its analytic Jacobian includes SG half-flux
potential derivatives and the clamp-inactive Fermi-Richardson barrier slice.
It reports the IFT sensitivity of all six eliminated coordinates to the two
adjacent bulk potentials and four bulk log densities, including the resulting
`d Delta sigma / d bulk`. Both the complete local/bulk residual Jacobians and
the sheet-charge sensitivity after independently re-solving perturbed systems
match central differences; the latter uses the roadmap relative threshold
`1e-4`.

## Self-consistent research API

`build_equilibrium_referenced_interface_charge_dark_reference()` first solves
and certifies a charge-off, zero-bias dark state in the same two-sided topology.
It requires one uncalibrated canonical microscopic document per physical
interface, rebuilds `sigma * v_th * N_t` from that document, and takes the
sheet-charge inventory from its SI `total_density_m2`. It stores SHA-256
identities for every document, the grid, stack, and dark state together with
`f_eq`, `N_t`, and the reconstructed capture velocities. The dark-state hash
also binds interface transmission. A changed document, SRV, density,
transmission, grid, stack, or uncertified dark state is rejected before the
charged solve.

`solve_equilibrium_referenced_interface_charge_steady_state()` is the only
Python activation API. At each inner Poisson Newton iteration it jointly
eliminates the charged local interface state, distributes `Delta sigma` to the
two adjacent control volumes using the exact left/right dielectric
capacitances, and adds the IFT `d Delta sigma / d phi_bulk` to the banded
Poisson Jacobian. The converged charged local state is reused by continuity;
there is no post-processing-only charge path. The result stores `f_eq`, `f`,
`Delta sigma`, both trace shifts, normalized Gauss residual, and the scaled
local Jacobian condition for every interface.

The dark research result reuses the certified charge-off arrays exactly and
reports zero incremental charge/trace shift. Focused device tests cover dark,
bias, and light; the complete outer banded Jacobian, including a re-solved
local QSS for every perturbation, matches central differences at the roadmap
`1e-4` threshold.

The fail-closed backend exposure is
`POST /api/research/interface-charge/steady-state`. Each request must set
`research_acknowledged=true`, supply exactly one inline device or config path,
and use an uncalibrated `equilibrium_referenced` stack with explicit rebaseline
acknowledgement. The endpoint fixes the same certificate-compatible solver
controls used by the registered lane; callers cannot weaken residual gates.
It constructs and hashes the charge-off dark reference inside the request,
verifies contact thermodynamics on the same two-sided grid, then returns an
explicit evidence schema containing `f_eq`, `f`, `N_t`, `Delta sigma`, both
trace shifts, normalized Gauss residual, scaled local Jacobian condition,
continuity/current/Poisson bounds, contact evidence, all three state/reference
hashes, every microscopic document hash, and the reconstructed capture
velocities. Response assembly independently rechecks array alignment,
`Delta sigma = -q Nt (f-f_eq)`, the one-electron-per-trap bound, and finite
certificate fields.

This endpoint is labelled `internal_numerical_research` and always reports
`production_unlocked=false`. It remains separate from the D4-E3b J-V route.
Transient, QF impedance, 2D, and every other charged experiment continue to
fail through the material-assembly capability gate.

## Protocol-bound charged J-V core (D4-E3a)

`build_interface_charge_jv_protocol()` and `solve_interface_charge_jv()` add a
narrow Python execution core without removing the production material gate.
The canonical `interface-charge-jv-protocol-v1` represents an illuminated,
ion-free, ascending, zero-scan-rate quasi-Fermi branch. It starts at 0 V, uses
the two-sided Fermi-Dirac Richardson interface with unit transmission, applies
the equilibrium-referenced `-q N_t (f-f_eq)` law, and stops after the first
sampled open-circuit bracket. It contains no invented transient dwell or scan
rate. Unknown, missing, or duplicate JSON fields fail closed, and every
physical, numerical, and acceptance field contributes to its SHA-256.

One charge-off dark reference anchors the complete branch. Every charged state
records the dark-state, grid, and stack hashes; a continuation seed is rejected
before Newton when those identities, the equilibrium occupancy, charge law,
finite state arrays, interface topology, or contact certificate differ. Direct
voltage continuation is attempted first. Failed intervals may be bisected only
within the frozen minimum-step and bridge-count limits; bridge states are
audited but are not reported as requested voltage samples.

The acceptance contract retains per-point occupancy, sheet charge, trace
shift, Gauss residual, local Jacobian condition, interface/cell residual,
electron and hole continuity bounds, face-current spread, Poisson residual,
and contact QFL span. The contact certificate is an explicit opt-in to the QF
solver; its default remains off and does not change historical charge-off
numerics. The protocol fixes the maximum contact span at 5 meV and cannot
relax that repository gate.

The current real test is deliberately an uncalibrated synthetic fixture, not a
device-performance or SCAPS-parity reference. It reuses the D4-E2-supported
ETL-only `N_D=2e15 cm^-3`, N30 slice. Requested points from 0 to 0.2 V stop at
the first bracket, retaining `[0, 0.025, 0.05, 0.075, 0.1] V`; the corresponding
current changes from `1.3330e-2` to `-1.1643e-2 A/m2`, with
`V_oc=0.078835 V`. The worst normalized cell residual is `3.121e-7`, continuity
bound `3.121e-7 A/m2`, normalized Gauss residual `3.16e-16`, and scaled local
Jacobian condition `1.186e4`. These values demonstrate an internally closed
execution path only.

At the D4-E3a checkpoint this core was not exposed through `/api/jv`,
`/api/jobs`, or the frontend. D4-E3b adds that narrow integration without
changing the core protocol. Dark J-V, finite-rate hysteresis, current
decomposition, spatial snapshots, external circuits, TPV, Suns-Voc, EQE, EL,
impedance, 2-D, and degradation remain unsupported for charged interface
defects. D4-E3c must run a newly registered source-clean grid/tolerance
certificate before this slice can be called production-certified.

## Backend and frontend charged J-V integration (D4-E3b)

The J-V-specific stack gate is the only production-route exception to the
general charge-off material gate. Both synchronous `/api/jv` and asynchronous
`/api/jobs kind=jv` resolve the same canonical protocol before solving or
submitting a worker. A charged request must use the quasi-Fermi solver,
zero scan rate, illuminated operation, no mobile ions, no bulk-defect
composition, no legacy interface-state channel, the two-sided physical
interface boundary, and Fermi-Dirac Richardson transport. Unknown controls,
protocol mismatches, or a charged protocol attached to a charge-off device fail
with HTTP 422 before numerical work begins.

`configs/interface_charge_jv_research.yaml` is the runnable uncalibrated
two-layer companion to the electrostatic stress fixture. It is explicitly an
internal synthetic numerical reference, not a SCAPS, experimental, or device-
performance reference. The dispatcher constructs the shared electrical grid,
applies its resolution guard, converts it to the charged two-sided trace grid,
and returns the protocol-bound core result in a `JVResult`-compatible envelope.
Forward and reverse arrays intentionally contain the same stationary branch,
hysteresis is zero, and the renderer draws one curve rather than implying two
independent scan histories.

Both J-V frontends detect the charged closure and lock scan rate to zero,
solver to quasi-Fermi, interface topology to the physical two-sided boundary,
and transport to Fermi-Dirac Richardson. Decomposition and spatial-profile
requests are disabled. The result view displays protocol, dark-state, grid and
stack identities together with requested/bridge counts, occupancy, sheet
charge, trace shift, Gauss/cell/continuity bounds, contact span, local Jacobian
condition, and tolerance factor.

The workstation Device panel now initializes from the active persisted
workspace snapshot instead of independently loading an IonMonger preset. This
state-ownership fix is part of the charged-route safety boundary: before the
fix, the controls could describe a charged device while the submitted payload
came from a different preset. A real browser run at N30 and five points from
0 to 0.1 V completed 5/5 certified points with zero bridges and rendered one
non-empty curve plus the full evidence strip. Desktop layout verification is
clean. The existing whole-workstation Golden Layout remains unsuitable for a
phone-width viewport; that general responsive-shell limitation is recorded but
is outside D4-E3b.

The shipped J-V fixture is registered as `partial` in the strict
config/benchmark matrix and is bound to a dedicated protocol/API integration
benchmark. The exact registered command passes 40 tests, including real core,
synchronous endpoint, and asynchronous worker solves. The backend suite passes
152 tests with three slow tests deselected; the frontend passes 453 tests across
35 files, TypeScript checking, and a production Vite build. The fixed single-
thread complete Python suite passes 3139 tests with 2 skipped and 267 deselected.
An initial full run failed only because the new shipped config was absent from
the reproducibility matrix; adding its file/semantic hashes and bidirectional
benchmark mapping fixed that provenance failure, after which the 55 matrix
tests and the complete suite passed. This failure and repair are retained as
part of the checkpoint history.

D4-E3b provides an auditable execution and presentation route, not a new
numerical certificate. The D4-E3c matrix must bind the current source,
environment, protocol, grid and tolerance identities and finish with no failed,
missing or reused cells before the capability label can be promoted.

## Charge-off reference lane

`configs/interface_charge_reference.yaml` is the uncalibrated Phase-3
reference. It uses `semiconductor_work_function` contacts, explicitly records
the fresh-rebaseline acknowledgement, and removes the historical SCAPS
de-spike, contact-floor, barrier, and interface calibration factors. It makes
no SCAPS-parity claim.

The registered `interface-recombination-charge-off` lane uses the two-sided
trace topology and cancellation-safe quasi-Fermi steady solver. Before
measuring the illuminated J-V arc, it requires a contact thermodynamic
certificate and solves a dark reference in the same topology. Cell artifacts
store the per-interface dark occupancy, dark-state hash, signed capture flux,
carrier-balance defect, local QSS residual, Poisson residual, continuity
bound, and current-spread evidence. The 3x3 grid/tolerance certificate is a
required entry artifact; a partial or failed matrix does not unlock charge.
The absolute interface-state balance uses the same `1e-4 A/m2` conservation
contract as the device continuity bound; the independent normalized local QSS
gate remains `1e-7`.

The current source-clean single-threaded matrix at commit `851406a` is
internally `certified`: run
`6930cd68ca7f0d531c269321e719163bf4079c0d07762bfc7a7275c3f4678722`,
certificate
`a4b131062885c5cb89d56a1c3b81246dec8b7980ad35e0b89994c704e894057d`,
and protocol
`d423c42dcf486d40b2bc84c930a806de7aa838810f84885b10b7a4a203755048`.
All nine cells completed with no failed or missing artifacts. The terminal
grid differences were `0.3311 A/m2` for interface flux, `9.740e-4` for
normalized J-V, and `4.852e-6 V` for Voc; all tolerance differences were at
least four orders below their registered limits (interface flux six orders,
Voc five orders, and normalized J-V four orders).

## Charged research certificate lane

`configs/interface_charge_research.yaml` is a purpose-built two-layer,
single-interface numerical reference. Its contact reservoirs and Poisson drop
share the `semiconductor_work_function` gauge, its asymmetric grid clustering
is fixed at `(2, 3)`, and its `N_t=1e13 cm^-2` interface samples the upper end
of the registered trap-density law without carrying any SCAPS calibration.

The registered `interface-charge-equilibrium-referenced-v1` lane retains its
physical claim identifier and evaluates N30/N60/N120 and QF residual factors
1/0.5/0.25. Every cell rebuilds a
content-addressed charge-off dark reference, verifies exact charge-on/off dark
array identity, then independently solves a dark biased state and an
illuminated operating point through the charged public Python API. Artifacts
store contact, grid, stack, dark-state and target-state identities together
with the canonical documents and hashes, reconstructed capture velocities,
`f_eq`, `f`, `Delta sigma`, trace shifts, Gauss residual, local residual and
IFT condition evidence. D4-E1 raises the executor and protocol schemas to v2
and adds the `microscopic_defect_contract_verified` quality gate.

The current source-clean single-threaded matrix at commit `851406a` is
internally `certified`: run
`902b25e2443caf039fb535e136699abe0fa8b7b69d4b8a41a9516255a0a1583a`,
certificate
`3d510f06e381ac56fc66afd3fb59db5e7f88e685d84e5037e1d5c9f8c439d631`,
and protocol
`730ddfd06a484fc0156dfd6ba0968a08382c9871147f209ef464a0caebb031bd`.
All nine cells completed with no failure, missing cell or reuse. Terminal grid
differences were `7.594e-4` for current, `7.627e-7` for occupancy,
`8.436e-4` relative for sheet charge and `8.918e-6 V` for trace shift. The
largest tolerance difference was `6.275e-10` for sheet charge. Across the
matrix, the worst normalized Gauss residual was `9.313e-17`, local interface
residual `1.582e-12`, continuity bound `4.571e-9 A/m2`, current spread
`4.554e-9 A/m2`, and scaled local Jacobian condition `4.874e4`. Dark
incremental charge and trace shift are exactly zero in all cells; the
microscopic-document, reconstructed-kinetics, charge-law, and evidence-alignment
gates all pass. Pre-D4-E1 certificates are retained only as historical context.

## D4-E2 completion and production unlock conditions

D4-E2's internal research exit conditions are now content-addressed under one
frozen current source identity. Production and general backend experiment
unlock nevertheless remain unavailable: D4-E3 must separately bind a charged
J-V protocol, expose fail-closed backend/frontend evidence, and preserve all of
the following certificates without weakening their gates:

- contact-consistent residual-certified dark reference (completed in the
  charge-off certificate above);
- complete charge-off interface steady-state grid/tolerance matrix (completed
  in the charge-off certificate above);
- local two-sided Gauss jump and analytic sheet-charge tangent with
  discontinuous permittivity (completed), including the registered
  device-level outer-coupling certificate above;
- stored per-interface `f_eq` in the same topology and energy gauge (completed
  and consumed through a content-addressed dark-reference identity);
- occupancy-dependent sheet charge inside the outer Poisson residual and a
  verified analytic/IFT Jacobian (completed for the research steady-state
  Python lane);
- dark reference identity and charge/grid conservation gates (completed for
  the purpose-built research config).

The backend research endpoint and its explicit evidence schema are completed,
including microscopic document identity and reconstructed kinetics.
The broader `E_t`/CBO/`N_D`/`N_t` work is registered as
`interface-charge-device-stress-v1`: nine one-factor device variants evaluated
on N30/N60 and residual factors 1/0.5. Its source-clean four-cell matrix has no
failed or missing cell, but remains `partial`: the N30-to-N60 pointwise sheet-
charge difference is `1.4735%`, above the fixed `1%` gate. The resolved-v2
companion retains every physical point and gate and adds N90 as the terminal
grid. The current v3 executor keeps its central finite-difference Jacobian probe
fixed at `7e-6`; only the Newton residual and Poisson tolerances tighten with
the matrix factor. This avoids treating the truncation/roundoff-balanced probe
as a monotonic convergence tolerance. Its source-clean six-cell matrix at
commit `851406a` is internally `certified`: run
`f4b6e916a2ba1827e337c1b9111423497cf3ccc1405b61289a8a863be43935bc`,
certificate
`9f9da1910c43e8c799ce8f13ed71f8603e0150097045f8027d4a314eecba9531`,
and protocol
`9927fdc22b5fc526146447343d67e4edd1e7293d3bf485e05ab35172026adeb9`.
The initial `N_D=1e16 cm^-3` endpoint failed N30/factor=0.5, while an
intermediate `5e15 cm^-3` endpoint failed N60/factor=0.5. Both failed runs are
retained as unsupported boundaries. The registered nonzero endpoints are
`1e14` and `2e15 cm^-3`, a 20-fold span, without changing any residual gate.
N90/factor=0.5 closes all nine points and reduces the N60-to-N90 sheet-charge
difference at the limiting `N_D=2e15 cm^-3` illuminated target to about
`0.155%`. N120/factor=0.5 instead fails closed at that point's dark-bias target,
so N120 is explicitly outside the current solver basin rather than being used
to extend the claim.
For the terminal N60-to-N90 pair, current changes by `0.1006%`, sheet charge
by `0.1548%`, and the largest absolute trace-potential shift is `42.7 uV`.
The terminal tolerance differences are `4.262e-10` for current,
`1.092e-11` for target occupancy, `3.246e-8` for sheet charge, and
`1.750e-13 V` for trace shift. The worst cell has continuity bound
`3.639e-7 A/m2`, normalized cell residual `2.235e-7`, normalized Gauss
residual `9.154e-11`, and scaled local Jacobian condition `1.048e7`. The Gauss
gate passes its fixed `1e-10` limit but has limited margin and must remain
visible in downstream evidence.
All six cells retain exact dark charge-off identity, contact certification,
the signed charge law, occupancy bounds, and charge/barrier sign consistency.
The current v3 results, not the inherited pre-D4-E1 certificates, support the
resolved internal numerical claim.
The historical three-layer SCAPS-derived reference also remains an unresolved
illuminated stress case. Transient, impedance and 2D require the later unified
algebraic-state topology. None of these gaps may be hidden by enabling the
production material path, and the current internal certificate must not be
described as external validation.

The pure sign-law and two-sided electrostatic unit tests are prerequisites;
the device-level claim is limited to the frozen research config and protocol
identified by the certificate above.

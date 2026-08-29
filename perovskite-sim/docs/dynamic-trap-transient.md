# Dynamic trap occupancy transient

## Scope

D6-E0 establishes the local time-domain contract for one single-level,
monovalent trap. D6-E1 and D6-E2 add separate research-only bulk and
two-sided-interface device transients. D6-E3a adds bulk-defect/mobile-ion
coupling, and D6-E3b adds the corresponding two-sided-interface/mobile-ion
coupling. None of these checkpoints modifies the historical method-of-lines
state vector.

The supported charge transitions are acceptor and donor. Neutral transitions
are rejected on this charge-coupled path because a changing neutral occupancy
cannot close the free-carrier capture charge with a trap-storage charge.
Multivalent, amphoteric, metastable, and tunnelling transitions remain outside
this checkpoint.

## Topology decision

The inherited density-form MoL cannot be enabled by removing its existing
explicit-defect guard:

- its packed state has carrier and ion densities, plus an optional four-density
  interface-plane block, but no trap occupancy state;
- its Poisson assembly would omit charged-defect storage;
- its quasi-steady recombination sink removes electrons and holes together and
  cannot represent the finite-frequency capture imbalance;
- the interface-plane carrier densities are not a shared trap occupancy.

The dynamic route therefore uses an explicit occupancy ODE. The physical
occupancy is represented by the unbounded coordinate

```text
u = log(f / (1 - f)),     0 < f < 1.
```

No accepted-step clipping or endpoint repair is allowed. A non-finite
coordinate or a coordinate whose floating-point reconstruction saturates at
zero or one fails closed. The bulk-device route appends these coordinates to
its own opt-in DAE layout; the two-sided routes do the same for shared
interface occupancies rather than changing the legacy layout.

## Local equations

For electron reservoirs `j` and hole reservoirs `k`,

```text
C_n,j = c_n,j [n_j (1 - f) - n1_j f]
C_p,k = c_p,k [p_k f - p1_k (1 - f)]
df/dt  = sum(C_n) - sum(C_p) = a (1 - f) - b f
a      = sum(c_n n)  + sum(c_p p1)
b      = sum(c_n n1) + sum(c_p p)
lambda = a + b.
```

The exact constant-reservoir solution is

```text
f(t) = f_qss + [f(0) - f_qss] exp(-lambda t),
f_qss = a / lambda.
```

The logit RHS and scalar analytic Jacobian are evaluated directly from the
same capture law. At the QSS point, the carrier-density forcing and capture
occupancy derivatives are exactly the D5 frequency-domain tangent arrays.

For both acceptor and donor transitions,

```text
dQ_trap/dt = -q df/dt,
dQ_carrier/dt = q [sum(C_n) - sum(C_p)],
d(Q_trap + Q_carrier)/dt = 0.
```

The absolute trap charge differs: `Q_acceptor = -q f` and
`Q_donor = q (1 - f)`.

## APIs

- `physics.trap_transient.evaluate_trap_transient` evaluates non-equilibrium
  capture, occupancy/storage, charge, and local conservation.
- `physics.trap_transient.linearize_trap_transient` returns exact occupancy,
  logit, carrier-density, and capture derivatives.
- `physics.trap_transient.constant_reservoir_trap_trace` is the independent
  closed-form oracle.
- `solver.trap_transient.solve_local_trap_transient` integrates the logit ODE
  with Radau or BDF and an analytic 1-by-1 Jacobian, then certifies every output
  sample against the oracle and local charge balance.
- `experiments.bulk_defect_transient.run_bulk_defect_device_transient` advances
  carrier storage and occupied-trap storage with a backward-Euler index-1 DAE.
  Its coordinates are interior electron/hole quasi-Fermi increments, trap
  logit increments, and interior electrostatic-potential increments.
- `experiments.interface_defect_transient.run_interface_defect_device_transient`
  advances one shared areal occupancy per physical interface together with
  bulk carriers, bulk electrostatic potential, and retained two-sided trace
  variables.
- `experiments.bulk_defect_ion_transient.run_bulk_defect_ion_device_transient`
  adds positive/negative mobile-ion log-density storage to the bulk-defect DAE.
- `experiments.interface_defect_ion_transient.run_interface_defect_ion_device_transient`
  adds the same ion storage and analytic face-flux tangent to the retained
  two-sided-interface DAE.

## D6-E1 device closure

Poisson remains an algebraic row instead of being eliminated from the time
step. This preserves a sparse analytic Jacobian with four block types:

- local carrier storage derivatives;
- nearest-neighbour Scharfetter-Gummel current derivatives;
- separate electron/hole capture and local recombination derivatives;
- local trap-charge and electrostatic Poisson derivatives.

Each implicit step solves

```text
S(z[k]) - S(z[k-1]) - dt R(z[k]) = 0,
P(z[k], V[k]) = 0,
```

where `S = [n, p, Nt*f]`. The implementation uses a sparse direct Newton
solve and a non-clipping line search. A centered-difference audit of every
Jacobian column is retained in the result certificate. Componentwise storage
scales are separate for carriers and occupied traps; carrier rows also include
an explicitly derived floating-point floor for subtracting adjacent large face
currents at a DC root.

The terminal current is not augmented by a synthetic trap-current term:

```text
J_total[k] = Jn[k] + Jp[k]
             + polarity * eps * (E[k] - E[k-1]) / dt.
```

Trap charge changes Poisson and therefore the displacement term. Independently,
the solver compares the integrated change of `q(p-n) + rho_trap` against the
left/right conduction-current difference. It also reconstructs every returned
state with the eliminated-Poisson QF operator and requires the two operators to
agree.

The voltage history is right-continuous step-and-hold. Nested substeps are run
inside every requested output interval, and the last two levels must agree in
carrier/trap/potential state and terminal current. The initial state must be a
residual- and contact-thermodynamically-certified QF/DC state from the same
defect model.

D6-E1 deliberately accepts only non-spatial, single-level acceptor/donor bulk
species, no interface states, no mobile ions, no selective contacts, and no
non-local photon recycling. Unsupported physics fails before integration.

## D6-E2 two-sided-interface closure

The interface route retains six zero-volume algebraic variables per physical
interface: two electrostatic trace potentials and the four carrier trace log
densities `(n_Ls, p_Ls, n_Rs, p_Rs)`. One shared trap logit is differential.
For `N` device nodes and `K` interfaces, the sparse coordinate dimension is

```text
3 * (N - 2) + 7 * K.
```

The differential storage and algebraic rows are

```text
S = [n_interior, p_interior, Nt_area * f],
P_bulk(phi, n, p, sigma_if) = 0,
G_trace(phi_Ls, phi_Rs, sigma_if) = 0,
B_carrier(n_Ls, p_Ls, n_Rs, p_Rs, f) = 0.
```

The same four microscopic capture legs determine the electron source, hole
source, and `Nt_area * df/dt`. The incremental sheet charge is always

```text
sigma_if = -q * Nt_area * (f - f_eq),
```

and enters both the outer finite-volume Poisson rows and the local two-sided
Gauss law. No lumped shared-node interface charge is introduced.

The analytic sparse Jacobian covers bulk quasi-Fermi and potential columns,
both trace-potential columns, all four trace log-density columns, and the
explicit occupancy column. Newton uses a sparse direct solve and a non-clipping
line search. Local carrier scaling includes the one-way Fermi-Dirac supplies;
storage scaling includes a sparse coordinate-resolution bound for stiff rate
rows. These are residual scales only and do not alter the physical equations.

The interface face reported in the ordinary device face array is the left
bulk-to-left-trace observation. Its displacement current uses

```text
D_L = -eps_L * (phi_Ls - phi_L) / d_L.
```

The result also reports both interface sides independently. On the right,

```text
D_R = eps_R * (phi_Rs - phi_R) / d_R,
J_total,L = J_conduction,L + dD_L/dt,
J_total,R = J_conduction,R + dD_R/dt.
```

The certificate requires `J_total,L = J_total,R`, all ordinary face totals to
agree, and the integrated free-carrier plus sheet-charge change to match the
terminal conduction-current difference. Every returned state is also rebuilt
with the existing locally eliminated fixed-occupancy QF operator.

The initial dark reference and DC state must carry the same microscopic defect
document hashes and a contact-thermodynamic certificate. Missing microscopic
documents, mobile ions, simultaneous bulk defects, and a cross-node barrier
exactly on the piecewise clamp switching boundary fail before time integration.
The last restriction makes the analytic clamp-inactive-slice contract explicit
instead of assigning an arbitrary derivative at a non-differentiable point.

## D6-E3a bulk-defect/mobile-ion closure

The first combined transient lane keeps the D6-E1 carrier/trap/potential DAE
and adds one reference-relative log-density coordinate for every active node
of each mobile-ion species. Its differential and algebraic rows are

```text
S = [n_interior, p_interior, Nt*f, P_positive, P_negative],
P(phi, n, p, rho_trap, P_positive, P_negative) = 0.
```

Positive and negative ions share the same blocking-boundary finite-volume
operator used by the legacy MoL and D5 AC paths. The Poisson contribution is

```text
rho_ion = q * [(P_positive - P_positive,0)
               - (P_negative - P_negative,0)].
```

The conduction-current decomposition is evaluated from the same state:

```text
J_conduction = Jn + Jp + q*F_positive - q*F_negative,
J_total = J_conduction + polarity*eps*(E[k] - E[k-1])/dt.
```

There is no synthetic trap- or ion-storage terminal channel. Terminal charge
balance integrates free carriers, absolute trap charge, and ion charge over
the interior control volume bounded by the first and last reported transport
faces. Each contiguous mobile region separately has a full inventory
certificate using the solver's endpoint-inclusive dual-cell widths. The two
integrals intentionally have different boundaries, and positive and negative
inventories cannot cancel one another into a false pass.

The ion part of the sparse Jacobian uses `ion_face_flux_jacobian()`, including
shared-site cross derivatives. Trap occupancy remains a logit and active ions
remain log densities. A declared site ceiling below the implemented steric
clip is checked before flux evaluation; a clipping kink or coordinate
overflow fails closed. Positive-only, dual-ion, and negative-only layouts use
the same public adapter.

The current E3a nonlinear solver is the inherited sparse-direct Newton with a
non-clipping line search. On the five-node reference device, `D_ion=1e-20`
and `1e-14 m2/s` certify, including the slow-ion frozen limit. Deliberately
accelerated `1e-12` and `1e-10 m2/s` probes currently stall the line search at
a scaled residual near `1.86e6`; this is a recorded D6-E3c stiffness-solver
entry condition, not a certified range and not silently recovered.

## D6-E3b two-sided-interface/mobile-ion closure

The second combined lane retains all D6-E2 variables and appends one
reference-relative log-density coordinate per active positive or negative ion
node. For `N` nodes, `K` interfaces, and `M_ion` active ion coordinates, the
coordinate dimension is

```text
3 * (N - 2) + 7 * K + M_ion.
```

The shared storage and algebraic rows are

```text
S = [n_interior, p_interior, Nt_area*f, P_positive, P_negative],
P_bulk(phi, n, p, sigma_if, P_positive, P_negative) = 0,
G_trace(phi_Ls, phi_Rs, sigma_if) = 0,
B_carrier(n_Ls, p_Ls, n_Rs, p_Rs, f) = 0.
```

The same microscopic four-leg capture law drives carrier continuity and the
shared occupancy. The same ion densities drive Poisson, ion continuity, and
the analytic `ion_face_flux_jacobian()`. At the physical interface, the ionic
face current crosses the same geometrical face and is added identically to the
left and right carrier observations; it is not assigned to the trap storage.
The certificate independently checks the two interface totals, every ordinary
face total, the interior carrier/interface/ion charge balance, and full
per-component ion inventories.

Electrostatic displacement is evaluated from the potential increment before
the spatial difference,

```text
Delta E = -diff(phi[k] - phi[k-1]) / diff(x),
```

rather than subtracting two separately formed large electric fields. The local
interface capacitive drops use the same increment-first form. This is
algebraically identical but avoids unnecessary floating-point cancellation.
No current-closure value is projected or repaired after the solve.

The implementation is currently at an intermediate checkpoint. Positive-only,
dual-ion, and negative-only 10 mV traces certify, including full-column
Jacobian comparison and nested time refinement. A 0.1 mV probe remains
fail-closed on relative charge/current gates because the displacement-current
difference approaches the double-precision subtraction floor. That low-signal
case is an E3c numerical-range item, not a reason to relax the declared gates.

## D6-E3c source-bound timescale refinement

The source-bound E3c adapter executes the same E3b index-1 DAE in three frozen
timescale cases. The combined case uses the source ion diffusivity and capture
kinetics. The defect-dominated case changes only active-source-layer ion
diffusivity from `1e-14` to `1e-20 m2/s`. The ion-dominated case changes only
the microscopic electron and hole capture cross sections by a factor of
`1e-12`. Every override, source identity, time/voltage history, grid axis,
time-step axis, solver policy, observable gate, and quality gate is included in
one canonical protocol hash shared by all nine matrix cells.

The production candidate fixture confines mobile ions to the absorber. An
ion-inactive ETL retains its structural site density but has `D_ion=0`; this
avoids introducing a zero-density steric boundary as a differentiable state.
Ion diffusivity overrides apply only to source layers that are already mobile.
The positive-ion centroid integrates only the active component with the same
endpoint-inclusive dual-cell widths used by the finite-volume inventory.

Cross-grid transient comparisons are reference relative:

```text
Delta f(t)     = f(t) - f(0),
Delta x_ion(t) = x_ion(t) - x_ion(0),
Delta Q(t)     = Q(t) - Q(0).
```

Each `t=0` state must still be residual-, contact-, and operating-point
certified. Comparing these changes prevents grid-dependent coarse DC baselines
from being misclassified as transient nonconvergence. Terminal current remains
an absolute observable. Earlier v3 and v4 lanes are retained as immutable
partial evidence: v3 used absolute DC-sensitive centroid and charge, while v4
still used absolute interface occupancy and required an unnecessarily narrow
line-search-only stiffness outcome.

The accelerated `D_ion=1e-12 m2/s` probe is a fail-closed stiffness boundary,
not a physical solution. Protocol v2 accepts only a typed line-search stall or
a typed Newton iteration limit. Both outcomes must report a machine-readable
iteration and finite residual above the registered nonlinear acceptance.
Iteration-limit errors also report charge, all-face current, interface current,
and linear backward errors. Unknown or unstructured failures fail the quality
contract.

The v5 source-clean matrix at commit `ba54ce2` completed all nine cells without
reuse, failure, or missing artifacts and passed 339 of 339 certificate checks.
Certificate SHA-256 is
`9eab2f9e251b8d4c0f7f3f07e0baeea9bb6497126ef8d8111eba1803947e5beb`.
Its largest terminal grid/tolerance comparisons included `4.57e-10 C/m2`
integrated charge change against a `1e-9 C/m2` limit, `3.82e-13 m` centroid
shift against a `5e-11 m` limit, and `2.02e-6 A/m2` terminal current against a
`1e-5 A/m2` limit. The canonical protocol SHA-256 is
`7db9bc5d8a166d3f928bcb0810bfbfdce26de1741085771a511ece9144ee1438`.

## Evidence and remaining boundary

Unit tests cover strict boundedness, endpoint/non-finite rejection, one- and
two-sided reservoirs, donor/acceptor charge signs, detailed balance, analytic
centered differences, exact relaxation, tolerance refinement, fast-QSS and
slow-frozen limits, immutable evidence, and fail-closed certification gates.

The D6-E1 integration tests use real acceptor and donor devices and cover
bounded occupancy, charge sign, exact current decomposition, all-face current
closure, carrier-plus-trap charge balance, analytic-Jacobian comparison,
sparse-versus-dense matrix cardinality, nested time refinement, fast-QSS and
slow-frozen limits, immutable output, and over-strict fail-closed gates.

The D6-E2 integration tests use a contact-consistent two-layer heterojunction
with nonzero conduction- and valence-band steps. They cover shared bounded
occupancy, the exact sheet-charge law, four capture and bulk-flux legs,
left/right conduction and displacement currents, global charge balance,
full-column Jacobian finite differences, nested time refinement, fast-QSS and
slow-frozen limits, immutable output, provenance rejection, and over-strict
partial/fail-closed behavior.

The D6-E3a integration tests use one residual-certified combined defect/ion DC
state and cover positive-only, dual-ion, and negative-only layouts; exact
carrier/ionic/displacement current decomposition; endpoint-inclusive
component inventories; global carrier/trap/ion charge balance; all-face total
current; full-column analytic-Jacobian comparison; nested time refinement;
the slow-ion frozen limit; immutable output; pre-clipping rejection; and
over-strict partial/fail-closed behavior.

The D6-E3b integration tests cover the corresponding three ion layouts with
one microscopic two-sided interface, exact sheet charge and
carrier/ion/displacement decomposition, left/right interface totals, the
interior terminal-charge control volume, endpoint-inclusive component
inventories, full-column analytic Jacobian comparison, slow-ion separation,
immutable output, pre-clipping rejection, and over-strict fail-closed behavior.
That checkpoint is committed and pushed with 12 focused tests, 83 related
tests, and a pinned default suite of 3327 passes, 2 skips, and 267 deselections.

E3c adds cancellation-safe coordinate-to-storage increments, strict
source/case/protocol identity, the three timescale cases, and the 3 by 3
space/time refinement matrix. The v5 contract/registry set has 32 passing
tests. Dynamic-storage and bulk/interface trap/ion transient related tests have
76 passes. The first full-suite run correctly found that the new configs and
registry SHA were absent from the reproducibility matrix; after adding file and
semantic hashes plus bidirectional benchmark mappings, the focused provenance
set has 74 passes. The final fixed-thread full suite has 3358 passes, 2 skips,
267 deselections, and 12 pre-existing `np.trapz` deprecation warnings. The v3
and v4 partial certificates remain preserved.

This closes the internal D6-E3c numerical and physical-logic evidence. It is
not external SCAPS or experimental validation. Protocol-bound public experiment
results, backend preflight, and frontend controls belong to D6-E4.

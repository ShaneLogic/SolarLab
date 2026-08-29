# Dynamic trap occupancy transient

## Scope

D6-E0 establishes the local time-domain contract for one single-level,
monovalent trap. D6-E1 adds a separate research-only bulk-device transient,
and D6-E2 adds a research-only two-sided-interface device transient. None of
these checkpoints modifies the historical method-of-lines state vector.

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
its own opt-in DAE layout; D6-E2 will do the same for shared interface
occupancies rather than changing the legacy layout.

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

This is internal numerical and physical-logic evidence, not external SCAPS or
experimental validation. D6-E3 must separately close simultaneous dynamic
defects and mobile ions. Protocol hashes, backend preflight, frontend controls,
and source-clean production matrices belong to D6-E4.

# Dynamic trap occupancy transient

## Scope

D6-E0 establishes the local time-domain contract for one single-level,
monovalent trap. D6-E1 adds a separate research-only bulk-device transient.
Neither checkpoint modifies the historical method-of-lines state vector.

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

This is internal numerical and physical-logic evidence, not external SCAPS or
experimental validation. D6-E2 and D6-E3 must separately close shared
two-sided interface traps and mobile-ion coupling. Protocol hashes, backend
preflight, frontend controls, and source-clean production matrices belong to
D6-E4.

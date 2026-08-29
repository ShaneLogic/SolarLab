# Dynamic trap occupancy transient

## Scope

D6-E0 establishes the local time-domain contract for one single-level,
monovalent trap. It does not expose a device transient method and does not
modify the historical method-of-lines state vector.

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
zero or one fails closed. Device coupling will append these coordinates to a
new, opt-in transient layout in D6-E1/E2 rather than changing the legacy layout.

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

## Evidence and remaining boundary

Unit tests cover strict boundedness, endpoint/non-finite rejection, one- and
two-sided reservoirs, donor/acceptor charge signs, detailed balance, analytic
centered differences, exact relaxation, tolerance refinement, fast-QSS and
slow-frozen limits, immutable evidence, and fail-closed certification gates.

This evidence is local only. D6-E1 must still prove, in one device solve, that
bulk capture enters the separate electron/hole continuity equations, trap
charge enters Poisson, trap storage contributes to displacement/terminal
current, componentwise tolerances include occupancy coordinates, and the
structured Jacobian outperforms a dense finite-difference baseline. D6-E2 and
D6-E3 separately close two-sided interface traps and mobile-ion coupling.

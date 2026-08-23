# Research DAE Backbone

The Phase-4.1 DAE work starts with a deliberately narrow reference problem. It
does not replace the production method-of-lines transient and is not exposed
through an experiment or backend route.

## First Slice

The research coordinate is

```text
q = (log(n / n_ref), log(p / p_ref), phi).
```

Carrier continuity at interior nodes supplies `2(N-2)` differential rows.
The four ohmic carrier boundary conditions and all `N` Poisson rows supply
`N+4` algebraic rows. The residual reports these two classes separately and
uses explicit rate, potential, and finite-volume charge scales.

`build_consistent_initial_condition` performs two deterministic operations:

1. pin the four ohmic carrier boundary coordinates;
2. solve the existing prefactored finite-volume Poisson system exactly, then
   set the interior log-density derivatives from the physical carrier RHS.

The resulting coordinate/derivative pair is accepted only when every scaled
DAE row is below the caller's tolerance. Arrays in the returned report and
initial-condition certificate are read-only, and a SHA-256 digest binds the
numerical state.

## Analytic Baseline

The initial analytic surface is intentionally small:

- exact `dF/d(qdot)` for all differential rows;
- exact carrier-boundary and Poisson rows of `dF/dq`;
- independent central-difference references for both matrices.

The Poisson block is the same finite-volume operator used by `solver.mol`.
Tests project a perturbed carrier state onto the algebraic manifold and compare
the frozen-potential DAE carrier RHS with the existing eliminated-Poisson RHS.

## Time-Discrete Reference

`solver/dae_integrator.py` supplies a research-only backward-Euler reference.
The time discretization is applied to physical carrier density, so each
differential row contains `(n_new - n_old) / dt` or `(p_new - p_old) / dt`.
The log-density coordinate therefore enforces positivity without replacing the
finite-volume carrier balance by a log-state balance.

Each accepted step records differential and algebraic residuals separately,
integrated electron and hole balance defects in `A/m2`, nonlinear iterations,
residual/Jacobian evaluations, line-search backtracks, update scaling, and the
scaled Jacobian condition number. Predictor overflow, singular Jacobians,
stalled line searches, and iteration exhaustion fail with the exact step and
time.

The current Newton matrix is deliberately a dense correctness baseline: its
differential state rows use central differences, while algebraic state rows and
the derivative chain use the analytic blocks. On the illuminated single-layer
c-Si test, 2/4/8 time steps contract the terminal log-density error against a
high-accuracy Radau/MoL trajectory at first order. This is internal numerical
equivalence only. The dense baseline does not satisfy the Phase-4 cost-scaling
gate and cannot justify replacing the production transient.

## Structured Newton Checkpoint

`solver/dae_jacobian.py` assembles the smooth first-slice tangent directly as
CSR. It reuses the analytic Scharfetter-Gummel, CT/PF field-mobility, and
SRH/radiative/Auger derivatives already certified in the ion-aware operator.
The adapter adds only finite-volume divergence, log-density chain, storage,
ohmic boundary, and explicit-Poisson blocks. A non-smooth field-mobility point,
self-consistent photon recycling, heterojunction de-spiking, or interface cap
fails capability checks.

The explicit `jacobian_mode="structured_analytic"` path uses sparse LU and a
sparse one-norm condition estimate. `dense_central` remains the default
reference. Analytic active/inactive field-mobility matrices agree with an
independent full-residual stencil within a `1.2e-5` group-normalized envelope;
the limiting entry is a minority-carrier Poisson derivative below the stencil's
floating-point resolution.

On 2026-08-23, with OpenBLAS, OMP, and vecLib fixed to one thread, five-repeat
median one-step measurements were:

| nodes | dense central | sparse analytic | speedup | dense/structured RHS |
|---:|---:|---:|---:|---:|
| 9 | 25.34 ms | 3.35 ms | 7.6x | 221 / 5 |
| 17 | 46.95 ms | 3.93 ms | 12.0x | 413 / 5 |
| 33 | 115.94 ms | 6.84 ms | 17.0x | 996 / 6 |
| 65 | 242.78 ms | 9.41 ms | 25.8x | 1956 / 6 |

The 9-to-65-node wall-time growth is 9.58x for dense central and 2.81x for
sparse analytic. Timing is a workstation observation, not a CI threshold.
The executable gates use trajectory equivalence, linear CSR nonzero growth,
and deterministic residual-evaluation counts. Reproduce the wall-time table
with `scripts/benchmark_dae_jacobian.py` under the same thread controls.

## Capability Boundary

This slice fails closed for physical interfaces, `InterfaceDefect`, dynamic or
QSS interface states, selective contacts, single or dual mobile ions, nonzero
structural ion coordinates, and nonpositive carrier references. It supports a
single electrical layer with ohmic contacts only.

Those exclusions are evidence boundaries, not claims that the omitted physics
can be added by changing a flag. The next checkpoint must add a time-discrete
content-addressed refinement lane for the first-slice error, residual, balance,
and work gates before any ion or algebraic interface-state topology is
introduced.

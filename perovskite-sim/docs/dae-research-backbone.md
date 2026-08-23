# Research DAE Backbone

The Phase-4.1 DAE work proceeds through deliberately narrow, separately
certified reference problems. None replaces the production method-of-lines
transient or is exposed through an experiment or backend route.

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

## Content-Addressed First-Slice Certificate

The registered `no-ion-dae-transient-v1` lane combines intervals 8/16/32 with
three backward-Euler time-step factors. Because the carrier equations are
diffusive, each grid's base step count scales as `(N / 8)^2`; the three time
levels then use factor 1/0.5/0.25. This avoids mistaking increased spatial
stiffness for failed temporal convergence. Every cell runs the same strict
Radau/MoL reference plus dense-central and structured-analytic backward Euler.

On 2026-08-24, source-clean commit `985a234` completed all 9 cells under
single-threaded BLAS/OpenMP:

- run ID `b3030c044d4753f17aed89216983ff2ac7013f54a5865fc6fa1647200c05931c`;
- certificate `44807d654d815bc0daa4aeb7d3ecc48ecbcb17436209f4d56c62f64d14cf3c0b`;
- protocol `65a6a01d80078146bf01a7e94b071eaa2cb24ba94ee812a9304cd97aed660d67`;
- terminal time-refinement change in MoL log-density error `9.16464e-5`
  against the pre-registered `1.1e-4` limit;
- terminal grid change in that error `3.18170e-6`, while the physical carrier
  inventory response changes `3.38764%` against its explicit 4% limit;
- all-cell maximum differential/algebraic residual
  `9.32738e-10 / 1.97891e-16`, and electron/hole balance defect
  `1.03260e-8 / 4.06689e-13 A/m2`;
- dense/structured trajectory log-density difference at most `1.99840e-15`,
  structured RHS-work fraction at most `0.02546`, and CSR storage at most
  `17.91` nonzeros per node.

This is an internal numerical certificate for the frozen first-slice protocol,
not an external c-Si validation or authorization to replace production MoL.

## Single-Positive-Ion Slice

The next capability contract extends the state to

```text
q = (log(n / n_ref), log(p / p_ref), logit(P / P_lim), phi).
```

All `N` positive-ion nodes are differential variables. The ion continuity
operator uses the production finite-volume face flux with blocking external
boundaries and the exact dual-cell inventory. The shifted-logit map enforces
`0 < P < P_lim`; a stable logistic density difference avoids subtracting two
site densities near `1e25 m-3` in the backward-Euler storage row. Carrier
interior rows remain differential, while the four ohmic carrier boundaries and
all Poisson rows remain algebraic.

`solver/dae_ions.py` constructs and certifies the consistent initial condition.
`solver/dae_ion_integrator.py` supplies dense-central and explicit
`structured_analytic` backward-Euler paths. The structured tangent reuses the
carrier blocks above and chains `ion_face_flux_jacobian` through the ion logit;
it includes ion storage curvature and the Poisson-ion derivative. Blocking
finite-volume divergence, dual-cell ion inventory, carrier balance, ion
balance, site occupancy, and algebraic residual are reported separately.
Non-differentiable steric faces fail closed.

The registered `single-positive-ion-dae-transient-v1` lane freezes the MAPbI3
absorber slice of `configs/ionmonger_benchmark.yaml`, dark operation at `10 mV`
for `10 ms`, ohmic contacts, blocking positive ions, and the configured
diffusion-only steric law. It combines 8/16/32 intervals with three
backward-Euler step factors; the base step count again scales as `(N / 8)^2`.
Every cell runs a strict high-accuracy production Radau/MoL reference plus the
dense and structured DAE paths.

On 2026-08-24, source-clean commit `6e9a274` completed all 9 cells under
single-threaded BLAS/OpenMP:

- run ID `d71d74acc5574e920edf8e0edb05020c043f3e8a87b1c95c926aa14556658dab`;
- certificate `7538fa4acea081c1c51c5f75201dfec90ebc64bca9e4d62068cb440f97b627e8`;
- protocol `3f0b24b96136e42972b689258aea16c547ac4e18e948ec7aa269ca23af1f989c`;
- terminal grid/time-step changes in positive-ion relative error
  `5.53590e-11 / 9.84279e-11`, below the frozen `1.1e-10` limit;
- terminal grid/time-step log-density error changes
  `1.95655e-9 / 2.30604e-9`, below `1e-8`, and potential-error changes
  `5.00667e-11 / 5.94353e-11 V`, below `5e-10 V`;
- all-cell maximum carrier/positive-ion/algebraic normalized residual
  `4.31830e-10 / 1.33230e-16 / 1.34261e-16`;
- positive-ion inventory drift at most `3.14321e-16`, positive-ion relative
  motion at least `2.94123e-6`, and electron/hole balance defect at most
  `3.35000e-18 / 7.79983e-18 A/m2`;
- dense/structured trajectory difference at most `5.55112e-16` in log density,
  `0` in positive-ion density, and `4.33681e-19 V` in potential; structured
  RHS-work fraction at most `0.02557` and CSR storage at most `24.73` nonzeros
  per node.

This certificate establishes internal numerical equivalence and conservation
for the frozen single-positive-ion topology. It is not an external IonMonger
validation and does not authorize replacing the production transient.

## Capability Boundary

The no-ion slice fails closed for any mobile ion. The single-ion slice admits
one positive mobile ion only and fails closed for a negative mobile ion. Both
fail closed for physical interfaces, `InterfaceDefect`, dynamic or QSS
interface states, selective contacts, nonzero structural ion coordinates
outside their declared topology, and nonpositive carrier references. Both
support one electrical layer with ohmic contacts only.

Those exclusions are evidence boundaries, not claims that the omitted physics
can be added by changing a flag. The no-ion and single-positive-ion certificates
are separately complete. Any dual-ion or algebraic interface-state topology
must use a new capability contract and a new content-addressed lane; it cannot
inherit either certificate.

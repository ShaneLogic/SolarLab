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

## Dual-Mobile-Ion Slice

The dual-ion capability extends the coordinate to

```text
q = (log(n / n_ref), log(p / p_ref), eta_plus, eta_minus, phi).
```

For the registered shared-site topology, `eta_plus` and `eta_minus` are the two
log ratios in a three-state softmax over positive ion, negative ion, and
vacancy. This enforces positive species densities and
`P_plus + P_minus < P_lim` without post-step clipping. The ion storage block is
therefore a node-local coupled `2 x 2` mass matrix. Its exact coordinate
Jacobian and Hessian enter the residual and backward-Euler tangent, including
the cross-species storage curvature. Positive and negative flux divergence,
blocking-boundary inventory, motion, charge sign, and site vacancy are reported
separately.

`solver/dae_dual_ions.py` supplies the residual and consistent initial
condition; `solver/dae_dual_ion_integrator.py` supplies dense-central and sparse
structured backward Euler. The analytic CSR tangent includes both ion-face
flux derivatives, both Poisson charge signs, and the shared-softmax chain. The
dense and structured paths retain the same physical-density storage equation.

The registered `dual-mobile-ion-dae-transient-v1` lane freezes the same dark
`10 mV`, `10 ms`, single-layer MAPbI3 slice, now with two blocking mobile-ion
species and a shared finite-site limit. The negative-ion diffusion, density,
and site-limit values are synthetic protocol inputs used to exercise the
topology; they are not attributed to the source IonMonger publication. Each of
the 9 cells runs strict Radau/MoL plus dense and structured DAE paths.

On 2026-08-24, source-clean commit `2d6b32f` completed all 9 cells under
single-threaded BLAS/OpenMP:

- run ID `a5e50a9f6522bf1229e1f2b416caf9b5ba914e574ceca2a08a363e1627581cc2`;
- certificate `15a6a4dcf38db26e2fa78f41ede0d908ad154c267ce86de13eeaf7e6c1f050ab`;
- protocol `8693f4c2009eb4487978671e5452db8c5ba596e3b365d12c13663a9257af348d`;
- terminal grid/time-step changes in positive-ion relative error
  `5.49541e-11 / 9.77153e-11`, negative-ion relative error
  `4.75563e-12 / 8.46220e-12`, log-density error
  `2.10299e-9 / 2.48429e-9`, and potential error
  `5.38119e-11 / 6.40295e-11 V`;
- all-cell maximum carrier/positive-ion/negative-ion/algebraic normalized
  residual `3.44856e-10 / 4.67293e-16 / 1.67140e-16 / 8.19753e-16`;
- positive/negative inventory drift at most `2.95472e-16 / 3.08323e-16`,
  minimum relative motion `2.94124e-6 / 9.31920e-7`, and minimum shared-site
  vacancy fraction `0.9799999`;
- dense/structured trajectory difference at most `5.55112e-16` in log density,
  zero for both ion densities, and `8.67362e-19 V` in potential; structured
  RHS-work fraction at most `0.02056` and CSR storage at most `37.43` nonzeros
  per node.

This is an internal numerical certificate for one dual-ion topology, not
external IonMonger parity or a production-route certificate.

## Algebraic Interface-State Slice

The fourth capability slice uses two electrical layers and one replaced
heterojunction face. Its coordinate is

```text
q = (log(n / n_ref), log(p / p_ref), eta_n1s, eta_p1s,
     eta_n2s, eta_p2s, phi).
```

Carrier continuity rows remain differential. The four DOS-bounded interface
trace densities and Poisson potential are algebraic. At the interface, the
ordinary Scharfetter-Gummel face is removed from both adjacent carrier rows and
replaced by the same bulk-to-plane thermionic flux used in the four interface
state balances. The local analytic tangent differentiates adjacent bulk Fermi
projection, reciprocal cross-plane Fermi-Richardson exchange, and
shared-occupancy interface SRH with respect to all coupled bulk, interface, and
potential coordinates. This cross-block dependence is not the separately
configurable cross-node carrier-sampling feature; that feature remains rejected.

`solver/dae_interface_states.py` defines the residual and consistent initial
condition. `solver/dae_interface_jacobian.py` assembles the complete physical-
density backward-Euler CSR tangent, and `solver/dae_interface_integrator.py`
provides dense-central and structured-analytic Newton paths. Analytic
linearization is valid only while the projection exponent, Fermi activity,
positive-state, SRH occupancy, and DOS/logit clamps are inactive. Any active
clamp fails closed instead of silently using a derivative of the clamped branch.

The registered `algebraic-interface-state-dae-transient-v1` lane freezes a dark
`10 mV`, `10 ns`, two-layer ohmic slice of
`configs/interface_charge_research.yaml`, with one uncharged interface, no
mobile ions, four algebraic Fermi-Richardson states, and locally eliminated QSS
states in the production MoL reference. It combines 4/8/16 intervals per layer
with three backward-Euler step factors. Every cell runs strict Radau/MoL plus
dense and structured DAE paths.

On 2026-08-24, source-clean commit `008aef3` completed all 9 cells under
single-threaded BLAS/OpenMP:

- run ID `407927842820d9360b132aaad50fa97c5bec55b146bef75f86811edd06845cad`;
- certificate `21bb12e4655c60ce7c97ce2a3cf57617fc3c2cb667990dab244fc04dc4a53c89`;
- protocol `9ffcc7e0cf2adfa52192686563455558f9fc4afa3a1982736d87f1c03af75efd`;
- terminal grid/time-step interface-occupation changes
  `4.92480e-3 / 5.57687e-13`, below the frozen `2e-2` limit;
- all-cell maximum carrier/interface/algebraic normalized residual
  `4.97175e-8 / 2.55161e-14 / 2.55161e-14`;
- electron/hole/interface balance defects at most
  `8.47196e-15 / 1.50341e-15 / 1.45516e-15 A/m2`;
- dense/structured trajectory difference at most `6.61882e-12` in log density,
  `6.61504e-12` in interface-state relative density, and `2.01228e-16 V` in
  potential; structured RHS-work fraction at most `0.02185`, while CSR storage
  is `17.44 / 18.18 / 18.58` nonzeros per node on the three grids;
- all clamp-inactive, bounded-interface-state, positive-terminal-density, and
  strict MoL numerical-health gates passed.

This certificate establishes internal numerical equivalence and conservation
for that frozen algebraic-interface topology. It is not a production transient
certificate, external solver validation, SCAPS parity, or a general interface
physics certificate.

## Single-Ion Algebraic-Interface Topology

The fifth checkpoint selects the first combined topology instead of widening
either earlier certificate. Its coordinate is

```text
q = (log(n / n_ref), log(p / p_ref), logit(P / P_lim),
     eta_n1s, eta_p1s, eta_n2s, eta_p2s, phi).
```

`solver/dae_ion_interface_states.py` combines two electrical layers, one
positive mobile ion active at every node, blocking ion boundaries, one
uncharged physical interface, four algebraic Fermi-Richardson trace densities,
and ohmic carrier contacts. The residual reports carrier, positive-ion,
interface-state, and Poisson rows separately. It also records the dual-cell
ion-inventory residual and uses the production `assemble_rhs` path with the
same explicit interface response that defines the four algebraic balances.

The consistent projection pins carrier reservoirs, solves Poisson with
`P - P0`, then eliminates the four local trace densities. Exact
`dF/d(qdot)` includes physical carrier and finite-site ion storage, while the
first state tangent covers carrier boundaries and the ion-aware Poisson block.
Independent central stencils verify both analytic blocks.

`solver/dae_ion_interface_integrator.py` adds the next correctness reference:
physical-density backward Euler for carriers and the bounded positive ion,
with the four interface traces and Poisson potential kept algebraic. Its dense
central Newton path reports carrier/ion/interface balances, dual-cell ion
inventory, residual work, conditioning, and an immutable trajectory digest.
On the two-layer `10 mV`, `10 ms` integration test, 2/4/8 backward-Euler steps
contract toward a strict production Radau/MoL eliminated-QSS reference with
successive error ratios `0.50019` to `0.50050`. At eight steps the terminal
carrier-log, positive-ion-relative, interface-relative, and potential errors
are respectively `3.50049e-9`, `1.62786e-8`, `1.90141e-9`, and
`9.18213e-11 V`. The reference also exercises positive-ion relative motion
`9.82449e-5` and interface-state relative motion `9.35783e-1`, while the
maximum relative ion-inventory drift is `1.13812e-16`.

`solver/dae_ion_interface_jacobian.py` then supplies the complete smooth
structured tangent. It maps the existing analytic carrier/interface block into
the combined coordinate without duplicating its projection, reciprocal
exchange, or shared-occupancy SRH formulas, then adds finite-site ion storage,
all ion-face flux derivatives, and the Poisson-ion column. Both the
diffusion-only and legacy whole-flux steric laws are covered. On 5/9/17-node
tests the CSR nonzero counts are `110/214/422`; the maximum scaled error against
an independent complete state stencil is `1.34873e-7`, and the complete
physical-density backward-Euler tangent differs by at most `3.02564e-7`.
Projection, occupation, SRH, DOS/logit, and steric differentiability clamps
remain explicit fail-closed boundaries.

The explicit `jacobian_mode="structured_analytic"` integrator path now factors
that CSR matrix with sparse LU while sharing the dense path's predictor,
physical-density residual, update limits, line search, and accepted-step
evidence. On the nontrivial `10 mV`, `10 ms`, two-step trajectory, dense and
structured paths both take 12 Newton iterations, while residual evaluations
fall from `976` to `16` (`0.01639` of dense work). Their maximum coordinate and
physical-state relative differences are `1.02141e-14`, interface-state relative
difference is `8.43769e-15`, and potential difference is `8.67362e-19 V`.
Across 5/9/17 nodes, dense residual evaluations grow `295/487/1016`, while the
structured path uses `7/7/8`.

A content-addressed refinement matrix remains the next checkpoint. Until that
source-clean matrix is complete, the combined topology remains
`INTERNAL_TESTED`, not `INTERNAL_CERTIFIED`.

The combined slice fails closed for dual ions, `InterfaceDefect`, configurable
cross-node sampling, interface charge, dynamic or two-sided interface states,
selective contacts, field mobility, photon recycling, and heterojunction
de-spiking. The two older builders continue to reject the combined stack, so
no prior certificate is silently reused.

## Capability Boundary

The no-ion slice fails closed for any mobile ion. The single-ion slice admits
one positive mobile ion only and fails closed for a negative mobile ion. The
dual-ion slice requires exactly one positive and one negative unit-charge
species; its registered lane additionally requires the diffusion-only
shared-site steric law. These three slices fail closed for physical interfaces,
`InterfaceDefect`, dynamic or QSS interface states, selective contacts, nonzero
structural ion coordinates outside their declared topology, and nonpositive
carrier references. They support one electrical layer with ohmic contacts only.

The ion-free algebraic-interface slice instead requires exactly two electrical
layers, one physical interface, four DOS-bounded Fermi-Richardson algebraic
states, ohmic contacts, charge-off electrostatics, and inactive clamps. It
fails closed for `InterfaceDefect`, configurable cross-node carrier sampling, dynamic
interface states, interface charge, two-sided trace geometry, mobile ions,
selective contacts, field-dependent mobility, photon recycling, and clamp-active
operating points.

Those exclusions are evidence boundaries, not claims that the omitted physics
can be added by changing a flag. The no-ion, single-positive-ion, dual-ion, and
algebraic-interface certificates are separately complete only for their frozen
topologies. The new combined residual has its own capability contract but no
structured-Newton or content-addressed certificate yet. Charged-interface,
`InterfaceDefect`, dual-ion/interface, and production-route topologies still
need separate contracts and lanes; none can inherit an earlier certificate.

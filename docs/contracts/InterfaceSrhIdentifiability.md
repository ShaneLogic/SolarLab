# Interface-SRH identifiability

## Purpose and evidence boundary

This opt-in workflow answers a narrower question than parameter fitting: given
the repository's interface-SRH formula, which parameter combinations are
structurally distinguishable from a frozen synthetic observable set? It keeps
two independent result flags:

- `analysis_certified`: the requested rank was observed, every multi-start and
  fixed-parameter profile completed, no forward call failed, and a full-rank
  scenario recovered its synthetic truth;
- `parameters_identifiable`: the weighted Jacobian is full column rank and its
  condition number is below the frozen protocol limit.

A rank-deficient analysis can therefore be certified while parameter
identifiability is false. This is intentional: correctly proving that a
parameter claim is unsupported is a successful analysis, not a solver failure.

The current forward slice uses production `interface_recombination` kinetics
and the research equilibrium-referenced charge law

```text
v_n, v_p = sigma_n, sigma_p * v_th * N_t * calibration_factor
Q_it = -q * N_t * (f - f_eq)
```

It does not run Poisson, drift-diffusion, J-V, TPV, impedance, ion transport,
or a measured-data likelihood. The YAML values used by the registered lane are
content-addressed formula inputs, not material truth.

## Frozen contract

`InterfaceSRHIdentifiabilityProtocol` contains all quantities that can alter
the inverse problem: estimated/fixed parameters and log bounds, synthetic truth,
carrier conditions, trap and intrinsic densities, capture cross sections,
thermal velocity, equilibrium occupancy, observable family and uncertainty,
noise seed and amplitude, finite-difference step, SVD threshold, rank and
condition gates, truth tolerance, multi-start coordinates, profile grid, solver
budget, and forward-failure penalty.

Canonical JSON and SHA-256 bind the protocol. The immutable result separately
stores the observed values, uncertainties, best fit, weighted Jacobian, Fisher
matrix/correlation, singular spectrum, numerical rank, canonical nullspace,
all fit attempts, fixed-parameter profiles, failure count, and result mapping
hash. Unknown, missing, nonfinite, inconsistent, or tampered evidence fails
closed.

The forward-failure policy is `penalize_and_invalidate`: a failed trial receives
a frozen large standardized residual so optimization can continue, but any
forward failure makes `analysis_certified=false`.

## Structural result

For the combined recombination-plus-charge observable family:

- with `N_t`, capture-cross-section scale, and calibration factor all free,
  kinetics constrain only their product while charge independently constrains
  `N_t`; the expected and observed rank is 2/3;
- the null direction has negligible `N_t` component and equal-magnitude,
  opposite capture-scale/calibration components;
- after capture scale is fixed, the remaining `N_t` and calibration parameters
  are rank 2/2 and recover noise-free synthetic truth.

This does not establish identifiability under realistic noise or model
discrepancy. In particular, it does not justify reporting the three free
quantities as independent material constants.

## Public entry points

Run the default synthetic protocol:

```bash
python scripts/run_interface_srh_identifiability.py \
  --out outputs/interface-srh-identifiability.json
```

Restrict the estimated set to the full-rank two-parameter slice:

```bash
python scripts/run_interface_srh_identifiability.py \
  --estimated-parameters trap_density_cm2 calibration_factor \
  --out outputs/interface-srh-identifiability-full-rank.json
```

The backend accepts an exact serialized protocol at
`POST /api/identifiability/interface-srh-synthetic` using
`{"protocol": {...}}`. Unknown request or protocol fields return HTTP 422.

## Numerical certificate

The immutable lane `interface-srh-identifiability-synthetic-v1` uses carrier
condition counts 5/7/9 and finite-difference step factors 1/0.5/0.25. The base
log10 step is `1e-3`. It runs both the rank-deficient and fixed-capture
scenarios in every cell.

Source-clean evidence at commit `12fc7cc`:

- run ID: `1069402cb1c90f32255aa6063a30c07500a4d92cb609ec3900cc78f3d9ea8f54`;
- certificate SHA-256:
  `b5fd5f2d784277c49b0b2720ad7f225b76586945bd2c9d2c772a34fa3dec5643`;
- protocol SHA-256:
  `343b648269636308348fce75181ad59484ecbd8a606c7fbddb1f79b4a3f91051`;
- 9/9 cells completed, with zero failed, missing, reused, or forward-failed
  cells;
- terminal grid differences were 0.9348% for the full-rank condition number,
  0.5085% for normalized full-rank singular values, 0.6269% for normalized
  nonzero rank-deficient singular values, and `5.37e-14` for the absolute
  nullspace vector;
- terminal tolerance differences were at most `1.66e-7` relative for singular
  values and `5.37e-14` absolute for the nullspace vector;
- the finest cell recovered log10 truth `(12, -1)`, had full-rank condition
  number `2.69276`, and reported rank-deficient absolute null vector
  approximately `(0, 0.707107, 0.707107)`.

The certificate closes only the pre-registered deterministic formula-local
slice. `InterfaceDefect`, `het_recomb_despike`, mobile ions, full-device
experiments, measured data, external solver comparison, experimental
validation, Bayesian inference, and uncertainty quantification remain outside
its claim.

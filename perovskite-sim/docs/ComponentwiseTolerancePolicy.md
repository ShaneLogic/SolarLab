# Componentwise absolute-tolerance policy

## Scope and compatibility boundary

SolarLab transient states are SI number densities whose magnitudes span many
orders. SciPy controls each local error component with

```text
atol_i + rtol * abs(y_i)
```

A single scalar `atol` therefore gives every carrier, ion, interface state,
and spatial node the same near-zero error floor. The opt-in
`ComponentwiseAtol` policy replaces only that floor with a reference-scaled
vector. It does not change the governing equations, `rtol`, Radau/BDF recovery
logic, state coordinates, or positivity behavior.

The default API remains the historical scalar (`1e-6` in 1D and `1e-8` in
2D). Existing calls pass that scalar to `solve_ivp` unchanged. Merely updating
SolarLab does not activate the new policy or change a baseline trajectory.

## Construction

For component `i` belonging to species `s`, the policy builds

```text
atol_i = refinement_factor
         * max(minimum_atol, species_fraction_s * reference_i)
```

The shipped opt-in policy uses `1e-12` for the carrier, ion, and interface
fractions, a `1e-6 m^-3` minimum, and a refinement factor of one. These are
starting values for a convergence study, not a universal accuracy certificate.

| State block | Reference density |
|---|---|
| 1D `n`, `p` | Local dark-neutral solution of `n - p = N_D - N_A` and `n p = n_i^2` |
| Positive ion `P` | Configured neutral-background profile `P_ion0` |
| Negative ion `P_neg` | Configured neutral-background profile `P_ion0_neg` |
| Interface-plane state | Magnitude of that block at the start of the integration interval |
| 2D `n`, `p` | Flattened local dark-neutral carrier references, in packed `(n, p)` order |

The builders reject non-finite/non-positive policy fields, incompatible array
shapes, missing dual-ion references, and packed states inconsistent with the
declared ion/interface layout.

## Usage

```python
from perovskite_sim.solver.tolerances import ComponentwiseAtol
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

policy = ComponentwiseAtol()
result = run_jv_sweep(stack, atol=policy)
```

The same object can be passed through experiments that forward `atol` to the
transient solver. It is expanded at the active `solve_ivp` boundary:

- 1D fully coupled carrier/ion/interface transients;
- the single- or dual-ion subproblem in `split_step`;
- 2D carrier transients;
- the local SciPy compatibility solver, which accepts the resolved vector.

## Minimum refinement study

Use the same physical deck, grid, initial state, bias/time protocol, solver
method, and `rtol` for all runs. Tighten only the absolute-tolerance vector:

```python
base = ComponentwiseAtol()
policies = (base, base.refined(0.1), base.refined(0.01))
```

For each level, record:

1. the policy fields and min/max generated `atol_i` for every state block;
2. solver success, status/message, RHS/Jacobian/LU counts when available, and
   recovery-path use;
3. the requested observables (for example `J_sc`, `V_oc`, `FF`, impedance, or
   TPV lifetime) and a normed comparison of the terminal internal state;
4. conservation, finiteness, site-occupancy, and experiment-specific physical
   acceptance checks.

A result is tolerance-resolved only when the two finest levels agree within a
predeclared observable and state envelope. A successful Radau return alone is
not a convergence certificate. If refinement changes the accepted J-V branch,
the branch/recovery diagnostics remain part of the evidence rather than being
averaged away.

## Non-goals

This policy does not enforce non-negative carrier states, regularize a
non-smooth RHS, make a dense finite-difference Jacobian cheaper, or establish
external physical validation. Those are separate numerical and scientific
gates.

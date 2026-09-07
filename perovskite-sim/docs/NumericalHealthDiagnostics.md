# Numerical health diagnostics

## Scope

The P1.2 diagnostics expose non-physical solver states without changing the
drift-diffusion equations. `run_transient` observes every Radau/BDF RHS trial
and attaches a `NumericalDiagnosticsReport` to the solver result. It records
block minima, negative and non-finite trial counts, exact production bulk and
interface SRH denominators, and terminal-state gates. Negative implicit trial
states are evidence, not accepted output, and do not fail strict mode by
themselves. Non-finite RHS values and non-positive terminal densities do.

This default diagnostic path is an observational contract, not a
positivity-preserving integrator. No density transform, projection, or
constitutive-law change is implied unless the separate research coordinate
below is explicitly selected.

## Research log-density coordinate

`run_transient(..., state_coordinates="research_log_density")` enables an
opt-in transient prototype. The default remains `state_coordinates="density"`:
the historical physical state, scalar or vector tolerances, RHS arithmetic and
solver return values are passed through unchanged.

For every physically active density component, the research coordinate is

```text
y_i(t) = s_i exp(z_i(t)),       s_i = y_i(0) > 0
dz_i/dt = f_i(y, t) / y_i.
```

Thus every representable active trial and returned density is finite and
strictly positive. Electron, hole, active positive/negative ion and active
interface-state blocks use this map. Ion nodes for which the material has zero
mobile-ion reference density are structural zeros rather than physical active
densities; they remain in linear density coordinates. Their exact zero is
therefore legal. The transform does not project either active or inactive
components.

All model boundaries remain physical: `assemble_rhs` receives densities in
`m^-3`, its result is interpreted as `m^-3 s^-1`, numerical-health diagnostics
observe those physical arrays, and `solution.y` is converted back to physical
density before return. A non-finite or non-positive active initial density, an
exponentially underflowing/overflowing trial, or a non-finite transformed RHS
fails closed with `LogDensityCoordinateError`.

### Absolute-tolerance map

The input `atol` is first resolved in physical density units, including the
normal `ComponentwiseAtol` construction. At the initial reference `s_i`, the
physical local-error budget is

```text
B_i = atol_i^y + rtol_y s_i.
```

The active dimensionless coordinate tolerance is chosen as

```text
atol_i^z = log1p(rtol_y + atol_i^y / s_i),
```

so a positive one-tolerance perturbation exactly satisfies
`s_i [exp(atol_i^z) - 1] = B_i`. For an inactive linear structural-ion
component, the mapped density tolerance is
`atol_i^u = atol_i^y + rtol_y |y_i(0)|`. The coordinate solver uses
`rtol_z = 100 eps` (about `2.22e-14` in float64): applying `rtol_y` to
`|z_i|` would measure distance from the arbitrary logarithmic reference, not
relative physical-density error. `state_coordinate_report` records the
physical input tolerances and reports active dimensionless and inactive
`m^-3` coordinate tolerances separately; it never combines their extrema.

This map is exact only at the initial reference. It is not a dynamic error
renormalization: after a density changes by many orders of magnitude, the
fixed log tolerance can be stricter or looser than the original physical
`atol + rtol |y|` budget.

### Small-device evidence and limitations

`tests/integration/test_log_density_coordinates.py` uses the shipped
`nip_MAPbI3` device with three intervals per electrical layer (10 nodes), an
illuminated steady-state start at 0 V, and a 10 ns transient at 0.01 V. On the
single-threaded 2026-08-14 development run, tightening physical `rtol` from
`3e-4` to `1e-4` gave:

| coordinate | terminal J at `3e-4` (`A m^-2`) | terminal J at `1e-4` (`A m^-2`) | fine `nfev` |
| --- | ---: | ---: | ---: |
| density | 303.143184 | 303.149319 | 71 |
| research log density | 303.129008 | 303.145000 | 81 |

The fine-coordinate difference was `0.004320 A m^-2`, inside the larger
same-coordinate refinement change of `0.015992 A m^-2`. One illustrative
single-thread wall-time sample was 11.1 ms for density and 17.8 ms for log
density; wall time is reported as a profile observation, not a test gate. The
test gates the refinement envelope and a broad `nfev` bound instead.

The current prototype has deliberate limits:

- A dark-equilibrium probe on the same grid contained active carrier densities
  down to `1e-24 m^-3`; `f_i/y_i` drove the first implicit log trial beyond the
  finite exponential range. It failed closed immediately rather than silently
  clipping. An illuminated preconditioned state did not hit this failure.
- The fixed initial reference, numerical Jacobian and `f_i/y_i` can worsen
  stiffness. There is no coordinate-aware analytic Jacobian or adaptive
  rescaling yet.
- Log coordinates enforce a lower positivity boundary only. They do not enforce
  ion site-occupancy upper bounds; that would require a separately audited
  bounded coordinate such as a logit map.
- Structural inactive-ion components are linear. Their exact zero is preserved
  only when the physical RHS preserves it, as the current production equations
  do.

This is therefore a research comparison lane, not a certified replacement for
the default density integrator and not a general positivity or convergence
proof.

## Split-step contract

The historical `split_step` path projects negative ion values inside its ion
RHS and clips its accepted ion terminal state to `[0, P_lim]`. Its default API
and arithmetic remain unchanged:

```python
state, success = split_step(x, state0, dt, stack)
```

Diagnostics are explicitly opt-in. `return_diagnostics=True` enables observe
mode and returns a third element:

```python
state, success, report = split_step(
    x, state0, dt, stack, return_diagnostics=True
)
```

The report keeps separate evidence for positive and negative ion species:

- raw initial, implicit-trial, raw ion-terminal, projected-terminal and final
  bounds;
- lower/upper projection entry counts and projection events;
- inventory before the step, before terminal clipping, after clipping and
  after carrier re-equilibration;
- relative inventory drift at the raw terminal and final returned state;
- final electron, hole and active interface-state minima and non-positive
  entry counts;
- ion-solver and carrier re-equilibration outcomes.

Ion inventory uses the same finite-volume invariant as the production ion
continuity operator: `sum(P * dual_cell_widths(x))`. It deliberately does not
use trapezoidal endpoint half-weights, which can report false drift when an
active boundary node exchanges density with its adjacent control volume.

Research-strict mode fails before a clipped state can be accepted:

```python
policy = SplitStepDiagnosticsPolicy.research_strict(
    maximum_relative_inventory_drift=1e-8,
)
state, success, report = split_step(
    x,
    state0,
    dt,
    stack,
    split_diagnostics=policy,
    return_diagnostics=True,
)
```

It always rejects a non-finite full state, an initially negative/over-limit
ion state, a raw terminal bound violation, a failed ion solve, a failed carrier
re-equilibration, a non-positive terminal electron/hole/active-interface
density, or inventory drift above the declared threshold. It cannot use
terminal clipping to manufacture a pass.

Negative and over-limit *implicit trial* states are always counted but are not
rejected by default, matching the transient diagnostic rule that nonlinear
solver exploration is distinct from an accepted terminal state. Certification
work that requires a stronger trial-state gate must explicitly set
`reject_negative_trial_states=True` and/or
`reject_overlimit_trial_states=True`. Non-finite trial states always fail
research-strict mode.

## Edge-regime probes

`tests/unit/solver/test_numerical_edge_regimes.py` evaluates the production RHS
on small real device grids for dark depletion, strong injection, deep cliff,
deep spike, 180 K operation, extremely low intrinsic density, high trap
density and a rapid 0 to 1.2 V bias step. The tests require finite RHS values,
positive finite SRH denominators and a strict diagnostics report. The rapid
bias test is an RHS health probe on both sides of the step; it is not a claim
that two samples resolve the physical transient.

## Evidence boundary

A passing report shows that the inspected internal solve did not hide a failed
terminal state or ion inventory behind projection. It does not establish grid
or tolerance convergence, external-solver agreement, experimental validity,
or a general positivity theorem. Those claims require their separate
certificate lanes.

# RHS regularization policy

## Scope

`RHSRegularization` is an explicit research policy for measuring sensitivity
to non-smooth constitutive closures. Its three declared widths default to zero,
and zero selects the corresponding hard/NumPy expression exactly. A
regularized trajectory is not automatically more physical, more accurate, or
positivity preserving.

There is one compatibility distinction. The transient/MoL and transient J-V
paths historically used the hard thermionic cap, so their omitted-policy
default is the zero-width limit. The legacy steady-state driver historically
used a global logistic cap with `te_softness=0.02`; omitting
`rhs_regularization` continues to use that exact legacy formula. Passing an
explicit `RHSRegularization` to the steady-state driver selects the new
compact-support implementation, including when its TE width is zero. Result
metadata therefore reports an explicit policy only when the caller supplied
one; it does not relabel the legacy logistic path as compact support.

The policy currently declares three widths with explicit units:

| Field | Units | Kink |
|---|---|---|
| `poole_frenkel_field_width_V_m` | V/m | `sqrt(abs(E))` at zero field |
| `interface_density_width_m3` | m^-3 | defensive `max(density, 0)` at zero |
| `te_cap_relative_width` | dimensionless | `min(abs(J_SG), abs(J_TE))` crossover |

## Compact-support contract

Every new smoothing is local. Outside its declared transition band it returns
the zero-width expression bit-for-bit:

- Poole-Frenkel is unchanged for `abs(E) >= field_width`;
- interface density projection is exactly zero below `-density_width` and
  exactly the input above `+density_width`;
- thermionic magnitude capping is the hard minimum when the relative
  separation exceeds `te_cap_relative_width`.

The transition polynomials match boundary derivatives and the TE cap always
keeps the Scharfetter-Gummel flux direction. A separately evaluated
thermionic bound can limit magnitude but cannot reverse current.

## Required ladder

An enabled policy must be evaluated with the identical device, grid, initial
state, physical protocol, solver and tolerances at widths

```text
w, 0.5*w, 0.25*w, 0
```

For every rung record:

1. the complete policy and protocol hash;
2. terminal observables and any requested spatial/trace observable;
3. nonlinear residual, all-face current spread and conserved inventories;
4. minimum raw state, negative trial count and non-finite events;
5. RHS/Jacobian/LU evaluations and wall time.

The candidate promotion gate is less than 0.5% change in each pre-registered
observable between the final two positive widths, no worse residual or
conservation error, and convergence toward the zero-width result. Each rung
must also identify whether its comparison reference is the transient hard-cap
path or an explicitly selected steady-state compact-support path; it may not
silently compare against the legacy steady-state logistic default. A speedup
alone is never a pass.

## Executable device studies

`scripts/run_regularization_ladders.py` executes three frozen, real-device
studies rather than a synthetic algebra-only evaluator:

| Study | Active constitutive path | Base width | Fixed trajectory |
|---|---|---:|---|
| `poole-frenkel-device` | absorber Poole-Frenkel mobility | `2e6 V/m` | illuminated, `V=0`, `10 ns` |
| `thermionic-cap-device` | physically normalized heterointerface TE cap | `0.5` relative | illuminated, `V=0.8 V`, `10 ns` |
| `interface-density-device` | dynamic interface-state SRH density projection | `1e-4 m^-3` | dark, `V=0`, `1 ps` |

Each study uses four intervals per electrical layer and records the exact node
coordinates, configuration hash, local source-file hashes, initial-state
source, voltage/light history, and solver tolerances in its study definition.
The applied policy comes back from the solver result; the evaluator does not
copy the request into evidence. Every rung records terminal current and/or
interface-state observables, endpoint/RHS-rate evidence, ion inventory or
interior-current conservation, raw trial/final minima, non-finite events, and
`nfev/njev/nlu` plus wall time.

Run all studies with one BLAS thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/run_regularization_ladders.py all
```

Certificates are written immutably to
`outputs/regularization-ladders/<study>/<certificate_sha256>.json`. A repeated
payload is reused; a different payload is never written over an existing
artifact. The real-device integration contract is locked by
`tests/integration/test_regularization_device_ladders.py`.

These trajectories are deliberately short, controlled kink-sensitivity
probes. They establish wiring and convergence to the hard expression on the
registered grid; they do not replace the broader grid/tolerance matrices or
claim that a `10 ns`/`1 ps` endpoint is a steady operating point.

## Negative states

`interface_density_width_m3` only changes how an opt-in constitutive helper is
evaluated around zero. The diagnostics observe the raw state before that
projection. A negative terminal density, non-finite RHS, or failed inventory
gate remains a failed research run even if the regularized helper returns a
finite rate. Clipping therefore cannot manufacture a certificate.

## Evidence boundary

Passing a regularization ladder establishes numerical insensitivity within the
declared model and protocol. It does not validate Poole-Frenkel mobility,
thermionic-emission parameters, interface state densities, or the underlying
contact model against an external solver or experiment.

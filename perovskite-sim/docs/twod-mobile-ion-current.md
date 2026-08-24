# 2D Mobile-Ion Complete Current

## Scope

The Neumann-x, single-positive-mobile-ion research lane has an instantaneous
terminal-current evaluator containing electron, hole, positive-ion, and
displacement current. It is wired into the public 2D J-V Python/backend
research path only under a matching strict execution protocol.

The historical `compute_terminal_current_2d(snapshot)` API remains the
carrier-conduction-only evaluator and continues to reject snapshots carrying
an ion state. Mobile-ion callers must explicitly use
`compute_mobile_ion_current_components_2d(...)`.

## Discrete Current

At one physical-density state, the evaluator first obtains the same
semidiscrete RHS used by the transient:

```text
(dn/dt, dp/dt, dP/dt) = assemble_rhs_2d(t, y, material, V).
```

The charge rate entering differentiated Poisson is

```text
drho/dt = q * (dp/dt - dn/dt + dP/dt).
```

With `dphi/dt = 0` at the bottom contact and
`dphi/dt = -junction_polarity * dV/dt` at the top contact, the existing
Poisson factorization solves for `dphi/dt`. The vertical displacement current
on every face is then

```text
Jdisp = eps0 * eps_face * dEy/dt.
```

The complete face current is

```text
Jtotal = Jn + Jp + q * FP + Jdisp.
```

`Jn` and `Jp` are reconstructed with the same field-dependent face diffusion
coefficients and the same thermionic interface cap used by the production
continuity RHS. `FP` comes directly from `positive_ion_fluxes_2d`; no separate
post-processing ion law is introduced. Lateral averages use the Neumann
dual-cell widths, and the terminal value is sampled at the top face.

## API

```python
report = compute_mobile_ion_current_components_2d(
    state,
    material,
    V_app=0.1,
    applied_voltage_rate_V_s=0.0,
)
```

For the endpoint of a fixed-voltage dwell, `applied_voltage_rate_V_s=0`. A
nonzero rate is explicit input; displacement current is never inferred from a
secant between two voltage samples. The immutable report retains all vertical
face arrays, lateral averages, terminal components, and the absolute and
relative all-face total-current spread.

## Fail-Closed Boundaries

The evaluator requires an active three-block `(n, p, P)` state, finite
increasing axes, Neumann-x control volumes, blocking single-positive-ion
transport, and exact ohmic carrier reservoir rows. It rejects periodic-x,
frozen-ion states, Robin/selective contacts, incomplete or nonfinite
derivatives, mismatched grids, and nonfinite current components.

Dual mobile ions and mobile-ion Robin-contact current remain outside this
checkpoint. The returned arrays use the native 2D vertical sign convention;
experiment metrics must apply the existing junction-polarity normalization.

## Verification Boundary

Tests pin all four component identities, direct equality with the ion-flux
source, differentiated-Poisson directional derivatives, lateral-uniform 1D/2D
parity, face-uniform Maxwell current, immutable evidence, field-mobility and
thermionic current-source reuse, and every declared fail-closed boundary.

The public protocol, ion-aware state initialization, dwell history, and
per-point diagnostics are documented in `docs/twod-jv-execution-protocol.md`.
This still does not certify a mobile-ion 2D J-V curve. A content-addressed
grid/tolerance certificate is the next required layer.

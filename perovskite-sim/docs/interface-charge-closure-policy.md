# Interface-charge closure policy

Status: `PARKED` (2026-08-23).

SolarLab's active interface-state paths are recombination-only. They do not
currently provide a residual-certified equilibrium occupancy reference or a
self-consistent occupancy-dependent sheet charge in the outer Poisson system.

## Configuration contract

The device schema recognizes two values:

```yaml
device:
  interface_charge_closure: off
```

and the reserved research intent:

```yaml
device:
  interface_charge_closure: equilibrium_referenced
  interface_charge_rebaseline_acknowledged: true
```

`off` is the default and preserves the historical material arrays and RHS.
The acknowledgement records that activating electrostatic trap charge
invalidates the historical SCAPS calibration. It is not an enable switch:
all production material assembly and backend experiment routes still reject
`equilibrium_referenced` with a `PARKED` capability error.

## Charge convention

For electron occupancy `f`, both supported trap characters have the same
equilibrium-referenced increment:

```text
acceptor: Delta sigma = [-q Nt f] - [-q Nt f_eq]
                      = -q Nt (f - f_eq)

donor:    Delta sigma = [q Nt (1-f)] - [q Nt (1-f_eq)]
                      = -q Nt (f - f_eq)
```

`equilibrium_referenced_interface_trap_charge()` returns this signed quantity
directly in C/m2. It exposes no arbitrary `-1/+1` multiplier. Absolute trap
charge is intentionally not implemented because it would additionally require
a neutral-occupancy convention, fixed countercharge, and whole-device charge
neutrality.

The legacy `MaterialArrays.iface_state_charge` scalar is retired. A manually
constructed non-zero value fails before Poisson rather than depositing charge
on the shared interface node.

## Unlock conditions

The research lane remains unavailable until all of these are content-addressed
or executable certificates under one frozen source/config/protocol identity:

- contact-consistent residual-certified dark reference;
- complete charge-off interface steady-state grid/tolerance matrix;
- two-sided Gauss-jump certificate with discontinuous permittivity;
- stored per-interface `f_eq` in the same topology and energy gauge;
- occupancy-dependent sheet charge inside the outer Poisson residual and a
  verified analytic/IFT Jacobian;
- dark reference identity and charge/grid conservation gates.

The pure sign-law tests and existing two-sided electrostatic unit tests are
necessary prerequisites. They are not a device-level interface-charge
certificate.

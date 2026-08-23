# Interface-charge closure policy

Status: research-only steady-state Python lane; production routes remain
`PARKED` (2026-08-23).

SolarLab's production interface-state paths remain recombination-only. The
explicit research API below provides a self-consistent occupancy-dependent
sheet charge in the QF outer Poisson system, but it is not yet backed by the
charged grid/tolerance certificate required for production exposure.

## Configuration contract

The device schema recognizes two values:

```yaml
device:
  interface_charge_closure: "off"
```

and the reserved research intent:

```yaml
device:
  interface_charge_closure: "equilibrium_referenced"
  interface_charge_rebaseline_acknowledged: true
```

`off` is the default and preserves the historical material arrays and RHS.
The acknowledgement records that activating electrostatic trap charge
invalidates the historical SCAPS calibration. A charge-off reference config may
set it while establishing the fresh baseline; doing so changes its semantic
identity without enabling charge. It is not an enable switch: all production
material assembly and backend experiment routes still reject
`equilibrium_referenced` with a `PARKED` capability error. The only activation
path is the explicit two-call research API described below; it internally
builds charge-off material arrays and threads the signed charge through the
separate charged QF system.

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

## Local charged-Gauss primitive

The research-only two-sided element now exposes an explicit
`EquilibriumReferencedSheetCharge` contract and evaluates the coupled local
coordinates
`(phi_L, phi_R, log n_L, log p_L, log n_R, log p_R)`. Its Gauss row includes
the signed incremental sheet charge and an analytic
`d Delta sigma / d log(state)` obtained from the shared SRH occupancy law.
Central-difference tests cover both the occupancy derivative and the complete
2x6 electrostatic tangent. Separate positive/negative sheet-charge cases close
the Gauss law across unequal left/right permittivity to a normalized residual
below `1e-10`.

This primitive is not a device solve and does not unlock the configuration.
The local physics layer now also solves the two electrostatic equations and
four carrier balances jointly. Its analytic Jacobian includes SG half-flux
potential derivatives and the clamp-inactive Fermi-Richardson barrier slice.
It reports the IFT sensitivity of all six eliminated coordinates to the two
adjacent bulk potentials and four bulk log densities, including the resulting
`d Delta sigma / d bulk`. Both the complete local/bulk residual Jacobians and
the sheet-charge sensitivity after independently re-solving perturbed systems
match central differences; the latter uses the roadmap relative threshold
`1e-4`.

## Self-consistent research API

`build_equilibrium_referenced_interface_charge_dark_reference()` first solves
and certifies a charge-off, zero-bias dark state in the same two-sided topology.
It requires one positive-`N_t` `InterfaceDefect` per physical interface and
stores SHA-256 identities for the grid, stack, and dark state together with
`f_eq` and `N_t`. A changed grid, stack, defect density, or uncertified dark
state is rejected before the charged solve.

`solve_equilibrium_referenced_interface_charge_steady_state()` is the only
Python activation API. At each inner Poisson Newton iteration it jointly
eliminates the charged local interface state, distributes `Delta sigma` to the
two adjacent control volumes using the exact left/right dielectric
capacitances, and adds the IFT `d Delta sigma / d phi_bulk` to the banded
Poisson Jacobian. The converged charged local state is reused by continuity;
there is no post-processing-only charge path. The result stores `f_eq`, `f`,
`Delta sigma`, both trace shifts, normalized Gauss residual, and the scaled
local Jacobian condition for every interface.

The dark research result reuses the certified charge-off arrays exactly and
reports zero incremental charge/trace shift. Focused device tests cover dark,
bias, and light; the complete outer banded Jacobian, including a re-solved
local QSS for every perturbation, matches central differences at the roadmap
`1e-4` threshold.

This does not expose the closure through backend, transient, QF impedance, or
2D routes. Those combinations continue to fail through the production
material-assembly capability gate.

## Charge-off reference lane

`configs/interface_charge_reference.yaml` is the uncalibrated Phase-3
reference. It uses `semiconductor_work_function` contacts, explicitly records
the fresh-rebaseline acknowledgement, and removes the historical SCAPS
de-spike, contact-floor, barrier, and interface calibration factors. It makes
no SCAPS-parity claim.

The registered `interface-recombination-charge-off` lane uses the two-sided
trace topology and cancellation-safe quasi-Fermi steady solver. Before
measuring the illuminated J-V arc, it requires a contact thermodynamic
certificate and solves a dark reference in the same topology. Cell artifacts
store the per-interface dark occupancy, dark-state hash, signed capture flux,
carrier-balance defect, local QSS residual, Poisson residual, continuity
bound, and current-spread evidence. The 3x3 grid/tolerance certificate is a
required entry artifact; a partial or failed matrix does not unlock charge.
The absolute interface-state balance uses the same `1e-4 A/m2` conservation
contract as the device continuity bound; the independent normalized local QSS
gate remains `1e-7`.

The source-clean single-threaded matrix at commit `29c94b4` is internally
`certified`: run `d0dc822393290d892e7118bcb7fabd4214b5584815f51ff9ff24f663822687e4`,
certificate
`0a4fdebdf18eb0237eaa1a4bef599872745697d148461f6de25d10a6985a950b`.
All nine cells completed with no failed or missing artifacts. The terminal
grid differences were `0.3311 A/m2` for interface flux, `9.740e-4` for
normalized J-V, and `4.852e-6 V` for Voc; all tolerance differences were at
least four orders below their registered limits (interface flux six orders,
Voc five orders, and normalized J-V four orders).

## Unlock conditions

Production/API unlock remains unavailable until all of these are
content-addressed or executable certificates under one frozen
source/config/protocol identity:

- contact-consistent residual-certified dark reference (completed in the
  charge-off certificate above);
- complete charge-off interface steady-state grid/tolerance matrix (completed
  in the charge-off certificate above);
- local two-sided Gauss jump and analytic sheet-charge tangent with
  discontinuous permittivity (completed); the registered device-level
  outer-coupling certificate remains pending;
- stored per-interface `f_eq` in the same topology and energy gauge (completed
  for the charge-off reference; the charged lane must consume this identity);
- occupancy-dependent sheet charge inside the outer Poisson residual and a
  verified analytic/IFT Jacobian (completed for the research steady-state
  Python lane);
- dark reference identity and charge/grid conservation gates.

The remaining unlock work is a registered charged grid/tolerance certificate,
including barrier-shift convergence and content-addressed dark-reference
identity. Until that artifact is certified, the capability must not be called
production-ready or externally validated.

The pure sign-law tests and existing two-sided electrostatic unit tests are
necessary prerequisites. They are not a device-level interface-charge
certificate.

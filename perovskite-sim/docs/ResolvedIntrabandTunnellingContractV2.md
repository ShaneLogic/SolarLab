# Resolved Intraband Tunnelling Contract V2

## Scope

`wkb-tunnelling-channel-device-v2` describes one isolated conduction-band
material barrier between two adjacent semiconductor reservoirs in the guarded
one-dimensional quasi-Fermi steady-state solver. The registered numerical
reference is `wkb-resolved-intraband-qf-dc-v2`. It uses a dark, ion-free,
300 K, Boltzmann n-type/barrier/n-type stack with a 3 nm undoped barrier.

Only the intraband electron channel has this device contract. Enabling hole,
contact, band-to-band or defect-assisted channels in the device evaluator
raises an explicit capability error. Their local formulas and configuration
schemas do not establish conservative device coupling. Multiple barriers,
mobile ions, interface-plane composition, illumination, AC and transient
tunnelling are outside the registered certificate.

The historical D8 claims and their withdrawal remain in
[WkbTunnellingFamilyContract.md](WkbTunnellingFamilyContract.md). The old
`wkb-tunnelling-channel-qf-dc-v1` definition retains its original thresholds;
its single-face transfer and exact device-equilibrium zero are not claims
about this implementation.

## Physical Energy Reference

Physical band edges and both quasi-Fermi levels use one electron-energy
reference, expressed in eV:

```text
E_C  = -(phi + chi_phys)
E_V  = E_C - Eg_phys
E_Fn = E_C + V_T ln(n / N_C(T))
E_Fp = E_V - V_T ln(p / N_V(T)),   V_T = k_B T / q
```

`physical_quasi_fermi_levels_eV` requires positive, finite carrier densities
and physical effective densities of states on the same grid. Temperature and
physical DOS come from the material arrays, including when thermionic DOS
normalization or DOS folding in the transport potential is disabled.

The solver variable `V_T ln(n) - (phi + chi_transport)` is not an absolute
electron energy. It must not be supplied to a Fermi occupation. The uniform
DOS offset used to diagnose the historical error is not a general correction
for a heterostructure. A future hole particle channel would use `-E_V` and
`-E_Fp` together; the returned `E_Fp` itself keeps the electron-energy sign.

Boltzmann carrier transport is the declared approximation. The synthetic
reference bounds `max(n/N_C)` and reservoir Fermi occupations by 0.05; the
certificate does not establish degenerate carrier transport.

## Barrier and Action

`compile_tunnelling_channels` identifies one interior electrical material
layer with upward conduction-band offsets at both boundaries. It records
the layer name, physical boundaries and adjacent reservoir node ranges.
At least three barrier nodes are required. Missing or ambiguous material
barriers are rejected; the historical `interface_faces` anchor cannot move
the selected material barrier.

The live electrostatic potential sets the barrier profile. For each energy,
the connected forbidden interval containing its core maximum must close
inside the declared reservoir ranges. Disconnected forbidden regions within
the core are rejected. Turning points can move with the state but cannot
silently become a different material barrier.

```text
S(E) = sqrt(2 m* q) / h_bar * integral sqrt(max(U(x) - E, 0)) dx
T(E) = exp(-2 S(E))
```

`piecewise_linear_wkb_segment_actions` integrates every linear band segment
analytically, including partial segments at turning points. The exact
linear-segment result remains finite for a single forbidden node and has a
stable flat-barrier limit. Tests compare rectangular and triangular barriers
with independent expressions and refine a curved barrier. The historical
trapezoidal square-root quadrature order is not the current action rule.

An open Gauss-Legendre rule integrates from the higher reservoir minimum to
the core maximum. Neither endpoint is sampled. A barrier-top node cannot
introduce a transparent one-cell fallback. This is a sub-barrier addition to
the declared classical transport model; it is not a complete quantum
scattering calculation or a validated treatment of all possible parallel
transport processes.

Mesh refinement must preserve material ownership. `_layer_node_masks` uses a
roundoff bound limited by a fraction of the smallest spacing, replacing the
old fixed 1 pm boundary padding that reassigned finely resolved lead nodes
to a different material. Interface nodes retain right-layer ownership.

## Conservative Particle Transfer

Each energy has two turning-point quasi-Fermi levels, interpolated linearly
from the corresponding nodal values. One transmission multiplies both
reservoir occupations:

```text
F(E) = C T(E) [f(E, E_Fn,left) - f(E, E_Fn,right)]
```

`F` is a particle flux per energy and area. The same nonnegative nodal
weights interpolate each reservoir level and remove or add the transferred
particles. If `s_i` is the integrated nodal particle transfer, then

```text
sum_i s_i = 0
sum_i E_Fn,i s_i = integral F(E) [E_Fn,right - E_Fn,left] dE <= 0
```

The second identity concerns the chemical work of the transfer alone. It
does not assert that a biased device has decreasing total free energy.
Every face between the two transfer locations carries the corresponding
particle flux; partial end segments carry their interpolation weights.
Conventional electron current is `J_n = -q F_face`. Its discrete divergence
is the same particle removal and addition used in continuity. The QF
residual and reported current use this same current array, including the
reported interface-boundary override.

Equal, nonsaturated occupations give exactly zero transfer at the local
formula level. A numerically solved equilibrium is accepted with explicit
current and residual tolerances, never by demanding an unexplained exact
zero. Controlled level perturbations verify that the occupations remain
responsive. Tests also exchange reservoirs, shift the common energy zero,
check all transfer faces and compare enabled and disabled device solutions.

## Numerical Acceptance

The input is `tests/fixtures/configs/wkb_resolved_electron_barrier.yaml`.
The complete predeclared limits are in `NumericalRefinementRegistry.yaml`;
its digest is pinned in `ConfigBenchmarkMatrix.yaml`.

| Independent variation | Values |
|---|---|
| Intervals per electrical layer | 24, 48, 96; fixed tanh parameter 5 |
| Newton and Poisson tolerance factor | 1, 0.1, 0.01 |
| Energy integration order inside every cell | 96, 192, 384 |

Every grid/tolerance cell solves all three energy orders, a disabled and
absent-channel pair, equilibrium, and reverse bias. The current-carrying
reference uses 0.02 V. No supply coefficient is adjusted across the matrix.
The terminal effect must exceed 1 percent of the disabled current to make
this particular comparison discriminating. This is an experimental-design
criterion for a synthetic calculation, not a physical law.

Convergence checks include total and disabled current, their difference,
event flux, local tunnelling current, band-window endpoints, actions, and
spatial profiles of potential, log electron density and physical electron
quasi-Fermi level. Raw states, spectra, turning points, currents and transfer
weights are retained in cell measurements with source, input, environment
and protocol identities.

Particle balance and path-current relative errors must be at most `1e-12`.
Steady current spread and carrier continuity bounds must be at most
`1e-4 A m-2`. The equilibrium quantum current bound is `1e-8 A m-2`.
All states must satisfy the contact thermodynamic and nonlinear residual
checks. Refinement of a terminal scalar alone cannot establish acceptance.

## Remaining Physical Limits

The fixed supply coefficient `C = 1e24 m-2 s-1 eV-1` is unfitted and has not
been established from the material's transverse modes or independent
quantum transport. The leading WKB exponent omits a separately validated
matching prefactor. The reference bounds the fraction of gross transmitted
supply with `S < 1` by 1 percent and records the complete action spectrum.
This bound describes the tested approximation range; it is not an error
estimate for the missing prefactor.

Internal conservation and numerical convergence do not establish agreement
with SCAPS, an exact scattering solution or experiment. Further channels or
material predictions require their own physical inputs and validation.

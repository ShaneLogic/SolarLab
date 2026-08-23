# Energy-distributed bulk-trap closure

## Scope

P4.3 adds one explicit, research-only Maxwell-Boltzmann bulk-trap closure. It
does not reinterpret the historical `trap_N_t_bulk`, `trap_N_t_interface`,
`tau_n`, or `tau_p` fields. Those fields still control only the spatial
lifetime profile. The SCAPS compatibility fields `distribution: gaussian`,
`E_char_eV`, and `N_peak_cm3` also remain parked metadata because their
normalization is ambiguous in the imported files.

The new standard-SI schema is independent:

```yaml
bulk_trap_distribution:
  distribution: gaussian
  total_density_m3: 1.0e22
  center_eV_above_vb: 0.562
  energy_sigma_eV: 0.08
  sigma_n_m2: 1.0e-19
  sigma_p_m2: 1.0e-19
  thermal_velocity_m_s: 1.0e5
  charge_transition: acceptor
```

`total_density_m3` is the energy integral, not a peak value. The schema names
the capture kinetics and the neutral charge reference explicitly. An acceptor
is neutral when empty and negative when filled; a donor is positive when empty
and neutral when filled. Unknown keys, a missing Gaussian width, a width on a
single-level defect, non-positive kinetics, and a level outside the band gap
fail closed.

## Shared energy integral

For a trap level `E_t` measured above the valence-band edge,

```text
n1(E_t) = Nc exp[-(Eg - E_t) / Vt]
p1(E_t) = Nv exp[-E_t / Vt]
cn = sigma_n v_th
cp = sigma_p v_th
D(E_t) = cn(n + n1) + cp(p + p1)
f(E_t) = [cn n + cp p1] / D
R(E_t) = Nt(E_t) cn cp [np - ni^2] / D
```

The same finite levels and density weights integrate `f`, `R`, and trap
charge. A Gaussian uses Gauss-Legendre nodes in the probability coordinate of
a normal distribution truncated to `0 <= E_t <= Eg`. The weights therefore
sum to `total_density_m3` exactly, remain inside the physical gap, and resolve
the narrow-width limit without treating a peak density as a total density.
A single-level distribution is represented by one exact node.

The implementation follows the one-electron charge transition in the original
[Shockley-Read analysis](https://doi.org/10.1103/PhysRev.87.835). The explicit
separation of occupancy, continuous trap density, recombination integral, and
donor/acceptor charge follows the equations and charge table in the official
[COMSOL Semiconductor Module trap theory](https://doc.comsol.com/6.3/doc/com.comsol.help.semicond/semicond_ug_semiconductor.6.57.html).

## Electrostatic closure

The restricted contact solve enforces

```text
p - n + ND - NA + Ntrap_charge = 0
```

at one common Fermi level. That state supplies both contact carrier reservoirs
and the semiconductor work function. In the device interior, the Poisson
charge is

```text
rho = q [p - n + ND - NA + Ntrap_charge].
```

`physics/bulk_traps.py` returns analytic derivatives with respect to `n`, `p`,
and electrostatic potential. `solver/bulk_trap_equilibrium.py` puts the exact
`d rho / d phi` on the Newton diagonal. The final certificate separately checks
mass action, face current, the finite-volume Poisson residual, and the
integrated-charge/displacement-flux Gauss identity.

## Capability boundary

The only executable topology is the dedicated
`solve_bulk_trap_pn_equilibrium` research route:

- two homogeneous layers with p-left/n-right doping;
- Maxwell-Boltzmann carriers and fully ionized dopants;
- one identical spatially uniform trap distribution in both layers;
- dark, zero bias, semiconductor-work-function ohmic contacts;
- no BGN, grading, ions, interface recombination, selective contacts, field
  mobility, environment overrides, or production experiment route.

An active distribution passed to the default material builder or production
MoL fails with `BulkTrapChargeCapabilityError`. This prevents a trap-aware
contact potential from being paired with a Poisson equation that silently
omits trap charge.

The first closure assumes steady occupancy, unity level degeneracy, constant
capture cross sections and thermal velocity, and one donor-like or
acceptor-like charge transition. It does not cover transient capture/emission,
Fermi-Dirac trap kinetics, amphoteric or multi-charge defects, field-enhanced
capture, heterojunction band references, or metastable defect conversion.

## Verification contract

`bulk-energy-distributed-trap-equilibrium-v1` freezes:

- spatial grids of 40/80/160 intervals per layer;
- Poisson tolerance factors of 1/0.1/0.01;
- internal energy orders 16/32/64 in every cell;
- a fixed non-equilibrium carrier pair for the recombination quadrature gate;
- `<0.5%` energy-doubling changes for charge, occupancy, and recombination;
- bounded occupancy, positive carriers, trap-aware contact certification,
  mass action, zero face current, Poisson residual, and discrete Gauss balance.

The registered lane is internal numerical evidence only. It is not an
external Sentaurus/COMSOL/SCAPS comparison, experimental validation, a
production J-V/C-V/impedance capability, or evidence that the synthetic trap
parameters describe a particular silicon device.

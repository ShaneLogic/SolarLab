# Foundation Reference Validation Contract V1

## Reference and Meaning

The foundation-stage one-dimensional reference uses the uncalibrated
Boltzmann p-i-n homojunction in
`tests/fixtures/configs/tpv_physical_reference.yaml`. Its physical band gap,
effective DOS, intrinsic density and contacts are specified at 300 K. The
same material input is used for dark equilibrium, illuminated short circuit,
open circuit, maximum power and weak optical perturbations.

The optical source is declared monochromatic at 2 eV with the fixture's
photon flux `Phi = 1e21 m-2 s-1` and absorption coefficient. The incident
power is therefore `P_in = q * 2 * Phi`, approximately `320.435 W m-2`.
The fixed current normalization is the incident photon charge flux
`J_ref = q * Phi`; the volumetric recombination normalization is `Phi/L`.
Neither scale is fitted to the finest calculation. Reported efficiency is
for this synthetic monochromatic input, not AM1.5G performance.

This input isolates classical carrier transport and SRH recombination.
Radiative/Auger losses, optical emission balance, photon recycling and
material-specific chemical kinetics are not validated by its results.

## Steady-State Refinements

Uniform meshes allocate intervals to the three layers in the ratio 3:16:3.
The initial sequence is 44, 88 and 176 total intervals, corresponding to
12.5, 6.25 and 3.125 nm spacing. All physical layer boundaries are nodes.
Equal spacing on both sides of the absorber boundaries also preserves its
control-volume measure for a later uniform ionic population. This choice
replaces the plan's suggested 30/60/120 starting counts while retaining
three independent physical-space resolutions and the stated error limits.

The independent nonlinear/Poisson tolerance factors are 1, 0.1 and 0.01.
Each spatial/tolerance cell retains 20, 10 and 5 mV metric extractions from
one nested set of independently certified steady-state points. A steady
state's certificate must hold independently of its continuation seed.
The first illuminated point explicitly uses the existing density-form
initializer before QF correction; its density seed is not itself accepted
as a QF result. Final QF residual, contact and current limits remain active.
Finite positive reference densities below `1 m-3` are preserved. The old
unit-density floor changed pinned minority-carrier reservoirs and produced
a 0.105 eV equilibrium Fermi-level span on this input despite small currents.
The independent contact-density and mass-action test rejects that behavior.

Open circuit is also solved by an independently bracketed current root.
Maximum power is also located by bounded scalar optimization of solved
steady states. These extra operating points supply the internal profiles
and check the sampled experimental observables.

The J-V convergence observable samples the positive, photovoltaic part of
the current at the same fixed voltages from 0 to 1.6 V, normalized by
`J_ref`. Above the measured zero-current crossing that photovoltaic
contribution is zero. The complete signed measured current is retained in
the raw result; no invalid state is removed to obtain this observable.

## Predeclared Limits

| Quantity | Finest-pair limit |
|---|---|
| Open-circuit voltage | 1 mV absolute |
| Short-circuit current | 0.2 percent relative |
| FF and monochromatic power conversion | 0.5 percent relative |
| Photovoltaic current curve | 0.5 percent of fixed incident photon charge flux |
| Potential and physical quasi-Fermi profiles | 1 mV / 1 meV absolute |
| Log electron and hole density profiles | `ln(1.01)` absolute |
| Net recombination profiles | 1 percent of fixed `Phi/L` |
| TPV waveform and identifiable lifetime | 1 percent relative |
| Blocking-ion number drift, each species | `1e-10` relative |

Profiles are sampled at 17 fixed positions strictly inside each physical
layer; no comparison point lies exactly at a material boundary. Equilibrium,
short-circuit, open-circuit and maximum-power profiles are checked separately.
Near-zero recombination uses the fixed volumetric source scale; it is not
divided by an arbitrary vanishing local value.

Every accepted steady state must have finite positive carrier densities,
consistent contact thermodynamics, bounded nonlinear residual, Poisson
residual at most `1e-12 C m-2`, and integrated carrier-continuity and all-face
current bounds at most `1e-4 A m-2`. Equilibrium additionally requires
coincident flat physical quasi-Fermi levels and zero current within those
numerical limits. Dilute statistics require both `n/N_C` and `p/N_V` below
0.05 on this reference. Photon and output-power budgets are checked.

The starting resolutions are not a convergence assertion. A failed finest
pair requires further resolution or an explicit incomplete result, while
retaining these physical inputs and limits.

## Dynamic and Dimensional Checks

The existing TPV time, integration-tolerance and pulse-amplitude refinements
retain their physical preparation and pulse widths. Spatial refinement adds
an independent axis to that same reference. Frozen-ion checks retain each
species and its compensating background as specified by
`FrozenIonSnapshotContract.md`.

Mobile-ion examples must declare every species density, diffusivity, site
capacity, initial profile, voltage/light history and total observation time.
Changing resolution must preserve each species' total initial number.
Accepted trajectories must preserve inventories, capacities, positivity and
charge/current closure; redistributions must be numerically distinguishable.

The two-dimensional uniform limit uses identical material, contact and
active thermionic definitions in one and two dimensions, with lateral and
vertical refinement varied separately. The existing
`twod-mobile-ion-interface-srh-v1` matrix retains its exact earlier limits
and restricted heterogeneous scope. No general three-dimensional capability
follows from these checks.

All refinement records bind source, physical input, numerical environment,
protocol, raw states and observables. Calibrated SCAPS comparisons and
unfinished Calado reproduction remain separate regression inputs, not
independent physical truth for this reference.

# Degenerate Semiconductor Closure

## Evidence boundary

P4.2 introduces bulk semiconductor statistics in independently reviewable
checkpoints. The repository default remains `maxwell_boltzmann`; an omitted
selector is behaviorally and semantically identical to the historical model.

The material/contact checkpoint provides:

- strict `carrier_statistics: maxwell_boltzmann | fermi_dirac` layer parsing;
- positive `Eg`, `Nc300`, and `Nv300` requirements for every FD layer;
- fully-ionized MB/FD charge-neutrality states at the active temperature;
- contact work functions derived from the same neutrality state;
- an explicit `semiconductor_work_function` requirement for FD stacks; and
- fail-closed default 1D and 2D material assembly unless the restricted
  generalized-FD research transport is selected explicitly.

The bulk-transport checkpoint adds:

- a generalized Einstein factor from the inverse FD logarithmic
  compressibility;
- a diffusion-enhanced generalized Scharfetter-Gummel face flux whose
  logarithmic secant exactly preserves constant quasi-Fermi states;
- statistics-aware contact certification, so FD reservoirs are never
  reinterpreted through an MB logarithm;
- an explicit `research_recombination_off` material policy restricted to a
  dark, uniform-DOS homojunction with ohmic semiconductor-work-function
  contacts; and
- `degenerate-pn-equilibrium-v1`, a 40/80/160 intervals-per-layer by
  1/0.1/0.01 Poisson-tolerance refinement candidate against the abrupt
  depletion-width and peak-field limits.

Default MB transport remains unchanged. The research lane rejects optical
generation, spatial doping profiles, heterojunction fields, interface
recombination/defects, mobile ions, selective contacts, field mobility,
generated material levers, advanced interface closures, and active
`SOLARLAB_*` overrides. It disables recombination because the historical
`np-ni^2` SRH/radiative/Auger laws are MB detailed-balance closures. It is not
a production J-V, C-V, impedance, or transient capability.

## Next gates

Incomplete ionization is the next constitutive gate. It must add donor and
acceptor levels and degeneracy factors to the same neutrality/Jacobian system,
including a temperature refinement contract. Band-gap narrowing follows as a
separate model because it changes band edges, DOS references, intrinsic
products, contacts, and the Poisson source simultaneously.

External solver and experimental validation remain out of scope until frozen
reference inputs and data are registered. Formula implementation and internal
refinement evidence alone can reach only the internal certification tier.

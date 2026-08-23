# Degenerate Semiconductor Closure

## Evidence boundary

P4.2 introduces bulk semiconductor statistics in independently reviewable
checkpoints. The repository default remains `maxwell_boltzmann`; an omitted
selector is behaviorally and semantically identical to the historical model.

The current contact-closure checkpoint provides:

- strict `carrier_statistics: maxwell_boltzmann | fermi_dirac` layer parsing;
- positive `Eg`, `Nc300`, and `Nv300` requirements for every FD layer;
- fully-ionized MB/FD charge-neutrality states at the active temperature;
- contact work functions derived from the same neutrality state;
- an explicit `semiconductor_work_function` requirement for FD stacks; and
- fail-closed 1D and 2D material assembly until generalized FD bulk transport
  is enabled and certified.

This checkpoint therefore certifies the material/configuration and contact
thermodynamic contract. It does not claim that production drift-diffusion,
recombination, J-V, C-V, or impedance paths support Fermi-Dirac statistics.
Silently combining an FD contact potential with Maxwell-Boltzmann bulk fluxes
is rejected.

## Next gates

The bulk transport checkpoint must consume the same density-to-reduced-Fermi
map in a generalized Scharfetter-Gummel flux, expose the generalized Einstein
factor, and demonstrate the dilute MB limit and high-doping p/n refinement.
Incomplete ionization and band-gap narrowing remain separate constitutive
closures because each changes the charge-neutrality and Poisson source terms.

External solver and experimental validation remain out of scope until frozen
reference inputs and data are registered. Formula implementation and internal
refinement evidence alone can reach only the internal certification tier.

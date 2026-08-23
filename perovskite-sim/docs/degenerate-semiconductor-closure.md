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

## Internal refinement certificate

The source-clean commit `d756c76` completed all nine registered cells with
single-threaded BLAS/OpenMP settings:

- run ID: `3c7c98f9d67bbb2ff2864f946183af9db2b008cf0971a76945e6fdaa7a602eb9`;
- certificate SHA-256:
  `968ad3bb67dc696b841a6bb8544c16eba3f9d5748b2fe737e20b8c7e30f8373f`;
- protocol SHA-256:
  `96dd6e56aeb90ad458c5ce86ad31c7aa61eea7e4737573fb7cc30852c5ac91e9`;
- terminal grid differences: depletion-width ratio `1.24453e-3`, peak-field
  ratio `8.10282e-4`, and space-charge-balance error `4.22650e-3`;
- terminal tolerance differences: zero for all three registered observables;
- full-matrix maxima: normalized Poisson residual `1.31861e-12`, normalized
  carrier rate `3.90497e-14`, relative face current `4.02601e-14`, and
  charge-balance error `1.75650e-2`; and
- depletion-width and peak-field analytic errors remained below `3.38%` and
  `2.93%`, respectively.

All cells retained positive carrier densities, statistics-aware contact
certificates, and the declared recombination-off topology. This reaches the
repository's internal numerical certification tier only.

## Next gates

Incomplete ionization is the next constitutive gate. It must add donor and
acceptor levels and degeneracy factors to the same neutrality/Jacobian system,
including a temperature refinement contract. Band-gap narrowing follows as a
separate model because it changes band edges, DOS references, intrinsic
products, contacts, and the Poisson source simultaneously.

External solver and experimental validation remain out of scope until frozen
reference inputs and data are registered. Formula implementation and internal
refinement evidence alone can reach only the internal certification tier.

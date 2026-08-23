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

## Incomplete-ionization checkpoint

The next opt-in layer adds `dopant_ionization_model: discrete_level` with
band-edge binding energies and degeneracy factors. The equilibrium charges are

```text
N_D+ = N_D / [1 + g_D exp(eta_n + (E_C-E_D)/V_T)]
N_A- = N_A / [1 + g_A exp(eta_p + (E_A-E_V)/V_T)]
```

and enter the same nonlinear neutrality and Poisson systems as the MB/FD
carrier densities. The Poisson Newton diagonal includes the exact derivatives
of both occupations. Overflow-safe logistic evaluation keeps the fractions in
`[0, 1]` at extreme reduced Fermi levels. The default `fully_ionized` selector
and its semantic hash remain unchanged; unused binding-level metadata is
rejected rather than silently ignored.

The restricted material path now accepts this closure only under the same
dark, homogeneous, recombination-off research policy as the degenerate PN
solver. Standard transient/J-V/C-V/impedance assembly fails closed. Contact
reservoirs, semiconductor work functions, built-in voltage, local ionized
space charge, and the Poisson tangent all consume the same level parameters.

`incomplete-ionization-temperature-equilibrium-v1` registers a 100, 150,
200, 250, and 300 K scan on the 40/80/160 by 1/0.1/0.01 grid/tolerance matrix.
It checks freeze-out curves, contact thermodynamics, equilibrium currents,
Poisson residuals, integrated charge balance, and bounded ionized fractions.
At low temperature the dopant occupation changes inside the depletion region,
so a fixed-charge abrupt-depletion formula is not a valid oracle and is not
used to certify this lane.

The discrete-level convention follows the standard donor/acceptor occupation
form documented by [nextnano](https://www.nextnano.com/docu/nextnano3/input_syntax/keywords/impurity-parameters.html)
and the incomplete-ionization treatment discussed by
[Xiao et al. (1999)](https://www.sciencedirect.com/science/article/pii/S002627149900027X).
The current checkpoint does not include impurity-band formation, the Mott
transition, or dopant capture/emission kinetics.

The source-clean commit `1090354` completed all nine registered cells with
single-threaded BLAS/OpenMP settings:

- run ID: `ad690a7e5398a6e3829f7f04d470a59fe20144e5c863a896f681a87fa3ac8008`;
- certificate SHA-256:
  `902ae0f91b77cf7403349d4d54553c2d43c3c774b1f2a62530e2a27c9fbc0254`;
- protocol SHA-256:
  `33b421757a1521f214047ae58b3ad0cfd412570b65f6b7103e3fe3ed1c99d779`;
- terminal grid differences: normalized integrated-charge width `2.05168e-4`,
  normalized peak field `1.13830e-3`, and charge-balance curve `4.84260e-4`;
- terminal tolerance differences for those quantities: `2.27381e-7`,
  `4.26088e-8`, and `4.57735e-7`; and
- full-matrix maxima: normalized Poisson residual `9.12692e-9`, normalized
  carrier rate `1.07050e-13`, relative face current `1.08413e-13`,
  charge-balance error `1.96418e-3`, and nine Newton iterations.

The terminal-grid acceptor fraction rises from `0.09027` at 100 K to
`0.68718` at 300 K; the donor fraction rises from `0.19690` to `0.90058`.
Every cell retained bounded ionized fractions, positive carrier densities,
and a statistics/ionization-consistent contact certificate. This reaches the
repository's internal numerical certification tier only.

## Next gates

Band-gap narrowing is the next constitutive gate. It remains a separate model
because it changes band edges, DOS references, intrinsic products, contacts,
and the Poisson source simultaneously. Its combination with incomplete
ionization requires a new certificate rather than inheriting either component
certificate.

External solver and experimental validation remain out of scope until frozen
reference inputs and data are registered. Formula implementation and internal
refinement evidence alone can reach only the internal certification tier.

# Dynamic defect/ion combined device AC

## Status and scope

`run_defect_ion_combined_impedance()` is the research-only D5-E2c adapter. It
supports three independently labelled capabilities:

```text
bulk_defect_plus_ions
interface_defect_plus_ions
bulk_interface_defect_plus_ions
```

The adapter is not connected to the production impedance API or frontend.
Its fixed scope is `research_defect_ion_combined_device_ac_only`. A
defect-only or ion-only certificate is not promoted into a combined
certificate; the complete state is solved and linearized again.

## One nonlinear DC operating point

The DC unknowns are the interior electron/hole quasi-Fermi increments and the
log-density increments of every active positive and negative ion. Carrier
continuity, Poisson electrostatics, ion equilibrium, and blocking inventory
constraints are solved together. For a non-steric ion species,

```text
mu_+ = log(P_+) + phi / V_T
mu_- = log(P_-) - phi / V_T.
```

The DC residual requires `mu` to be constant inside each contiguous mobile
region and replaces one difference by that region's dual-cell inventory
constraint. The physical steric-diffusion mode adds `-log(1-theta)` with the
declared independent or shared site occupancy. Log coordinates enforce
strictly positive active densities without clipping.

For every candidate state, the same eliminated Poisson solve includes

```text
rho = q(p - n + P_+ - P_+0 - P_- + P_-0 + N_D - N_A)
      + rho_bulk_trap,
```

plus every equilibrium-referenced interface sheet charge. Interface stacks
first solve a mobile-ion dark charge-off reference and then bind the charged
measurement state to that exact occupancy reference. A separate ion-free dark
anchor is not used.

The DC certificate fails closed unless all of the following pass on the live
operator:

- contact reservoirs and the Poisson drop have a certified thermodynamic
  reference;
- the nonlinear optimizer and normalized carrier/ion residual pass;
- electron and hole continuity area-rate bounds pass;
- positive/negative electrochemical-potential and blocking-current bounds
  pass;
- every contiguous ion inventory is preserved with dual-cell weights;
- Poisson, DC all-face current spread, bulk trap balance, interface local
  carrier residual, and two-sided Gauss jump pass.

## Coupled AC state

The small-signal state contains

```text
[interior electron QF,
 interior hole QF,
 active positive-ion log density,
 active negative-ion log density,
 bulk-trap logit occupancy,
 interface-trap logit occupancy].
```

Every finite-difference probe reconstructs this physical state and re-solves
the same nonlinear electrostatics. Bulk occupied charge and interface sheet
charge enter Poisson; electron and hole captures enter their own continuity
equations; positive and negative ion fluxes enter separate dynamic rows. No
independently calculated defect and ion spectra are added after the solve.

The frequency-domain equation is

```text
(i omega M - A) delta u = (b - i omega m_V) delta V.
```

The result keeps electron, hole, positive-ion, negative-ion, bulk-trap, and
interface-sheet storage responses separate. These are not blindly summed into
one reported scalar: carrier storage is defined on interior control volumes,
blocking-ion inventory uses all active dual cells, and an interface defect is
an areal zero-thickness population. Mixing those observation domains would not
be a valid discrete Gauss certificate.

## Current observation at a two-sided plane

The two-sided local solve has distinct left and right reservoir currents and a
zero-thickness sheet population. Its placeholder grid face is not an ordinary
volumetric face. The combined adapter reports the interface-face current by a
symmetric reconstruction from the adjacent physical finite-volume faces for
electron, hole, ion, and displacement components. This is an explicit
external-current observation convention, not a replacement for local physics.
The result records this convention as
`interface_current_observation="symmetric_adjacent_physical_faces"`; a
bulk-only result records `ordinary_finite_volume_faces`.

The local four-leg capture balance, fixed-occupancy carrier residual, trap
storage equation, and Gauss jump remain separate certificate gates. In
particular, the symmetric observation cannot make a failed local interface
solve pass.

## Frequency-domain certificate

Certification requires all of these checks at once:

- the combined DC state is freshly certified;
- the dynamic operator at zero occupancy increment embeds the QSS DC operator;
- bulk and interface capture responses close their respective trap-storage
  equations;
- electron, hole, positive-ion, negative-ion, and displacement currents close
  on every reported face and their named conduction sum is exact;
- blocking positive/negative ion inventory response passes;
- componentwise linear backward error and three-level finite-difference
  refinement pass;
- the low-frequency solution approaches the combined QSS-trap reference;
- the high-frequency solution approaches the combined frozen-trap reference;
- the requested grid brackets the actual trap relaxation corners and all ion
  diffusion, blocking-charge, and dielectric scales with the configured margin
  and sampling density.

The frequency grid is evidence, not a display default. Extending far below the
slowest required branch can make the blocking-ion matrix unnecessarily
ill-conditioned, while a conventional `10 Hz` lower bound can omit the ionic
branch entirely. The reference tests therefore use windows derived from the
reported timescales.

## D5-E2c reference evidence

All cases below use the real nonlinear and frequency-domain solvers, three
finite-difference levels, and contact-consistent work-function references.

| Capability | Frequencies | States | DC residual | AC all-face spread | FD change | Ion inventory response | Low / high limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| bulk + positive ion | `1e-3..1e6 Hz` (19) | 14 | `3.24e-18` | `2.72e-12` | `1.74e-8` | `6.70e-13` | `5.00e-13 / 1.38e-13` |
| bulk + dual ions | `1e-3..1e6 Hz` (19) | 19 | `3.24e-18` | `2.72e-12` | `1.74e-8` | `1.24e-12` | `6.08e-13 / 1.38e-13` |
| bulk + negative ion only | `1e-3..1e6 Hz` (19) | 14 | `3.24e-18` | `2.72e-12` | `1.74e-8` | `5.94e-13` | `4.47e-13 / 1.38e-13` |
| interface + positive ion | `1e-3..1e2 Hz` (11) | 21 | `7.11e-15` | `1.16e-8` | `3.84e-8` | `1.05e-12` | `2.25e-7 / 2.41e-6` |
| bulk + interface + positive ion | `1e-3..1e6 Hz` (19) | 24 | `5.65e-14` | `1.49e-5` | `3.60e-5` | `2.09e-12` | `6.54e-5 / 3.05e-4` |

The triple-coupled case also gives bulk/interface trap-balance errors of
`5.89e-9` and `6.44e-16`, a maximum DC ionic current of `3.79e-17 A/m2`, a DC
inventory error of `1.19e-16`, and a Poisson residual of `3.93e-16`.

The integration suite additionally covers an insufficient frequency window,
missing-defect and missing-ion capabilities, an inconsistent contact
thermodynamic reference, a shared-site dual-ion state, and a negative-only
mobile species. These results establish internal D5-E2c numerical/physical
closure only. They are not external SCAPS validation, experimental validation,
or a production API claim.

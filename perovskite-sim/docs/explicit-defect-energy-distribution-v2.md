# Explicit Bulk-Defect Energy Distribution Contract v2

Status: D3-E1 canonical input, carrier-independent quadrature, and pure local
Gaussian/uniform closure. Only `single_level` is enabled in the production
QF/DC material path; distributed production execution remains fail closed
until D3-E3.

## Version and compatibility

The version string is `solarlab-explicit-bulk-defects-v2`. Version 1 remains
the default compatibility constant and its canonical JSON and SHA-256 are
unchanged. A v1 document cannot carry v2 reference/support fields, and a v2
document cannot omit them. This prevents old SCAPS-shaped Gaussian metadata
from acquiring new executable meaning merely because the code was upgraded.

For a v2 distribution:

- `normalization` is always `integrated_total`;
- `total_density_m3` is the finite-support energy integral in `m^-3`;
- `energy_reference` is explicitly `above_valence_band`;
- `center_eV_above_vb` follows the local valence-band edge;
- the complete finite support must lie in `0 <= E_t <= E_g`;
- the implementation never clips a support at a band edge and renormalizes it
  silently.

## Canonical distribution families

```yaml
defect_schema_version: solarlab-explicit-bulk-defects-v2
defect_model: explicit_quasi_steady
bulk_defects:
  - name: absorber_acceptor_distribution
    distribution:
      kind: gaussian
      normalization: integrated_total
      total_density_m3: 1.0e22
      energy_reference: above_valence_band
      center_eV_above_vb: 0.75
      width_eV: 0.08
      width_convention: scaps_characteristic_energy
      support_width_multiplier: 6.0
    charge_transition: acceptor
    neutral_reference: empty
    kinetics:
      sigma_n_m2: 1.0e-19
      sigma_p_m2: 1.0e-19
      thermal_velocity_n_m_s: 1.0e5
      thermal_velocity_p_m_s: 1.0e5
    degeneracy: 1.0
```

The supported shapes follow the SCAPS 3.6.3 definitions while keeping the
SolarLab canonical density integrated:

| `kind` | Shape with unit peak | Finite support |
|---|---|---|
| `single_level` | delta at `E_t` | `E_t` |
| `uniform` | `1` | `E_t +/- width_eV/2` |
| `gaussian`, SCAPS width | `exp[-((E-E_t)/E_c)^2]` | `E_t +/- m E_c/2` |
| `gaussian`, standard deviation | `exp[-0.5((E-E_t)/sigma)^2]` | `E_t +/- m sigma/2` |
| `conduction_band_tail` | `exp[(E-E_t)/E_c]` | `E_t-m E_c <= E <= E_t` |
| `valence_band_tail` | `exp[-(E-E_t)/E_c]` | `E_t <= E <= E_t+m E_c` |

For uniform defects, `width_convention` is `uniform_full_width` and no support
multiplier is accepted. Gaussian widths use either
`gaussian_standard_deviation` or `scaps_characteristic_energy`. Both tails use
`scaps_characteristic_energy`. Gaussian and tail supports require an explicit
`support_width_multiplier`; SolarLab does not insert the SCAPS defaults of 6
and 7 into canonical data.

## Peak and integrated density

Peak density is an importer/display quantity in `m^-3 eV^-1`. For a unit-peak
shape `g(E)`, conversion is

```text
Ntotal = Npeak * integral(g(E) dE over declared support).
```

The exact finite-support shape integrals are:

```text
uniform:                width
Gaussian (sigma):       sigma sqrt(2 pi) erf[m/(2 sqrt(2))]
Gaussian (SCAPS Ec):    Ec sqrt(pi) erf(m/2)
CB/VB exponential tail: Ec [1 - exp(-m)]
```

`physics/defect_distributions.py` exposes both conversion directions. A single
delta level deliberately has no finite `Npeak`; asking for one fails closed.
If a future SCAPS importer receives both `Ntotal` and `Npeak`, it must validate
these equations instead of choosing one silently.

## Carrier-independent quadrature

`build_defect_energy_quadrature` maps the normalized distribution probability
coordinate to finite energy nodes. Density weights are positive and are
corrected only for floating-point summation so that their sum recovers the
declared integrated density. Gaussian nodes use the truncated-normal inverse
CDF; exponential tails use their finite-support inverse CDF. All nodes remain
strictly inside the declared support and local band gap.

`expand_bulk_defect_species_energy` creates auditable single-level node
species while retaining kinetics and charge convention. A source
`single_level` takes a separate exact branch: the expansion returns the
original immutable species object and its exact density, independent of the
requested quadrature order. The existing v1 and v2 single-level production
closures therefore perform identical arithmetic; only their provenance hashes
differ because v2 records the energy reference explicitly.

## D3-E1 local Gaussian and uniform closure

`evaluate_energy_distributed_defect_closure` expands each canonical source
species, evaluates all nodes through the exact D2 monovalent primitive, and
then integrates without introducing a second occupancy model:

```text
R_i       = sum_k R_ik
rho_i     = sum_k rho_ik
fbar_i    = sum_k (N_ik f_ik) / Ntotal_i
dR_i/dx   = sum_k dR_ik/dx
drho_i/dx = sum_k drho_ik/dx
```

The same node occupancy therefore supplies recombination, occupied density,
charge, carrier derivatives, and fixed-quasi-Fermi potential derivatives.
Results retain both levels of evidence: a `source_closures` tuple with the
quadrature and full D2 node closure, plus stable name-sorted totals. Reordering
source species leaves every total array bitwise unchanged while intentionally
changing the provenance identity of the ordered input document.

The result dataclasses recompute every node aggregate, source total, global
total, extrema, and SHA-256 identity during construction. Inconsistent or
non-finite payloads fail closed. A one-node source bypasses multiplication and
summation, preserving the D2 single-level values bit for bit.

`assess_defect_energy_order_refinement` is a separate local validation axis. It
requires an exact 2x order ladder and records, per adjacent pair, the maximum
source occupancy absolute change, charge change normalized by `q Ntotal`,
recombination relative change, and relative change across all analytic
tangents. Its input hash binds the broadcast carrier state, material constants,
source species, order ladder, and threshold. It contains no spatial grid or
solver tolerance field.

For the D3-E1 Gaussian+uniform three-regime fixture, the terminal 16-to-32
changes are `5.72e-8` for occupancy, `5.72e-8` for normalized charge,
`1.16e-7` for recombination, and `3.86e-6` for the worst tangent, all below the
`5e-3` gate. This is internal constitutive convergence, not SCAPS validation.

## Capability boundary after D3-E1

Enabled:

- strict v2 document round-trip and canonical SHA-256;
- all five normalized finite-support distribution contracts;
- analytic `Npeak`/`Ntotal` conversion;
- carrier-independent normalized energy nodes;
- exact v2 single-level use of the existing D2 QF/DC closure;
- pure local Gaussian/uniform occupancy, recombination, charge, and analytic
  tangents with source/node evidence;
- content-addressed local energy-order refinement independent of space and
  solver tolerance.

Still fail closed:

- any distributed closure in the production material/experiment path;
- CB/VB tail local closure before D3-E2;
- distributed contact neutrality, Poisson coupling, J-V, AC, and transient
  execution;
- SCAPS shaped distributed import without explicit, mutually consistent
  `Ntotal`, `Npeak`, characteristic width, and support metadata;
- spatially graded density or energy reference;
- dynamic occupancy, non-unit degeneracy, and multivalent defects.

D3-E2 adds both band tails and frozen SCAPS conversion fixtures. D3-E3 is the
first checkpoint allowed to enable distributed defects in production QF/DC.

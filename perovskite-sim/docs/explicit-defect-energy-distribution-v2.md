# Explicit Bulk-Defect Energy Distribution Contract v2

Status: D3-E0 canonical input and carrier-independent energy quadrature. The
five distribution families are parseable and auditable. Only `single_level`
is enabled in the production QF/DC constitutive closure at this checkpoint;
distributed production execution remains fail closed until D3-E3.

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

## Capability boundary after D3-E0

Enabled:

- strict v2 document round-trip and canonical SHA-256;
- all five normalized finite-support distribution contracts;
- analytic `Npeak`/`Ntotal` conversion;
- carrier-independent normalized energy nodes;
- exact v2 single-level use of the existing D2 QF/DC closure.

Still fail closed:

- distributed recombination, occupancy, charge, and analytic tangents in the
  production closure;
- distributed contact neutrality, Poisson coupling, J-V, AC, and transient
  execution;
- SCAPS shaped distributed import without explicit, mutually consistent
  `Ntotal`, `Npeak`, characteristic width, and support metadata;
- spatially graded density or energy reference;
- dynamic occupancy, non-unit degeneracy, and multivalent defects.

D3-E1 will aggregate Gaussian and uniform node closures and certify energy
order convergence. D3-E2 adds both band tails and frozen SCAPS conversion
fixtures. D3-E3 is the first checkpoint allowed to enable distributed defects
in production QF/DC.

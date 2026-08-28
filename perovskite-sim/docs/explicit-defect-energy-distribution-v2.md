# Explicit Bulk-Defect Energy Distribution Contract v2

Status: D3-E2 canonical input, carrier-independent quadrature, pure local
Gaussian/uniform/CB-tail/VB-tail closure, and strict SCAPS-shaped distributed
conversion. Only `single_level` is enabled in the production QF/DC material
path; distributed production execution remains fail closed until D3-E3.

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
The D3-E2 strict SCAPS adapter validates these equations instead of choosing
one density silently. It is deliberately separate from the legacy loader.

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

## D3-E1/E2 local distributed closure

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

### D3-E2 exponential band tails

The same source/node aggregator now accepts `conduction_band_tail` and
`valence_band_tail`. It does not add a tail-specific occupancy approximation:
the E0 inverse-CDF quadrature supplies positive, exactly normalized density
weights, and every energy node still uses the D2 monovalent closure for
occupancy, recombination, charge, and all analytic tangents.

For the combined CB-tail/VB-tail three-regime fixture, the independent
`16 -> 32 -> 64` energy-order report has input SHA-256
`9a5529321d8dc0848fd27214904b2bf6efb780405d559bb4a191fc432fa4614a`.
The terminal `32 -> 64` maximum changes are `4.62e-9` for occupancy,
`4.62e-9` for charge normalized by `q Ntotal`, `3.93e-8` for recombination,
and `2.54e-7` for the worst tangent. These results are local constitutive
quadrature evidence, not a production DC/J-V or external SCAPS comparison.

## Strict SCAPS-shaped distributed adapter

`scaps_compat.distributed_defects.convert_scaps_distributed_bulk_defect`
accepts only dimensionally explicit distributed inputs. The supported source
fields are:

- one canonical distribution name: `gaussian`, `uniform`,
  `conduction_band_tail`, or `valence_band_tail`;
- exactly one energy field: `E_t_eV_above_vb`, `E_t_eV_below_cb`, or signed
  `E_t_eV_above_intrinsic`;
- `E_char_eV`, plus an explicit `support_width_multiplier` for Gaussian and
  tails; uniform treats `E_char_eV` as its full width and forbids a multiplier;
- at least one of integrated `N_total_cm3` or peak `N_peak_cm3_eV`;
- explicit capture cross sections, charge transition, neutral reference, and
  a layer thermal velocity supplied in `cm s^-1`.

The intrinsic-level conversion is non-degenerate and uses the supplied layer
context:

```text
Ei - Ev = Eg/2 + (VT/2) ln(Nv/Nc)
Et - Ev = (Ei - Ev) + Et_above_intrinsic.
```

The adapter converts `cm^-3 -> m^-3`, `cm^-3 eV^-1 -> m^-3 eV^-1`,
`cm^2 -> m^2`, and `cm s^-1 -> m s^-1` before constructing one canonical v2
species. When both density fields are present, their relative mismatch must be
at most `1e-12`; the integrated total remains canonical and any inconsistency
fails closed. The conversion result retains source cgs fields, resolved SI
values, intrinsic level, shape integral, canonical species, and its own
SHA-256, and recomputes all relationships when constructed.

Four literal fixtures freeze both conversion and v2 document identities:

| Distribution | Conversion SHA-256 | Canonical document SHA-256 |
|---|---|---|
| Gaussian | `ebfd73e6785e77d4caa24f66e6f85ba6056038077bc522a1baf780c4ee0e31c2` | `ca2724f5701bada98e0f0307f43128f2fce55c8ada416c3fd3f726e90a117da6` |
| Uniform | `2be71f3a68525c93fb618866edecd6d62ff5e38f1663818038d7263055133ae3` | `80b880ee09033d012e0dc6f70efc6dc90dcf4a53c8fec81fcc8e42405d96cc74` |
| CB tail | `76c035c065e951aabe86007d81a8797830ba0d91257b3fd67957638f36e34d87` | `c93b2931f5469fc5233a5000e34a68462634148b78a2028513b1043a054cd8c2` |
| VB tail | `1aa4ca63a14d5ae01e4da1051ed71118dd083176c4f98689eb07d4fca78c571a` | `f1ed3a6bdd45a0af807f687e2e7d6de2858e650410942299269e82c9ef2501e9` |

The existing `load_scaps_yaml` and
`bulk_defect_species_from_scaps_mapping` paths are unchanged. Their historical
`N_t_cm3` remains the direct integrated input and legacy `N_peak_cm3` remains
informational; ambiguous old YAML never acquires v2 executable meaning.

## Capability boundary after D3-E2

Enabled:

- strict v2 document round-trip and canonical SHA-256;
- all five normalized finite-support distribution contracts;
- analytic `Npeak`/`Ntotal` conversion;
- carrier-independent normalized energy nodes;
- exact v2 single-level use of the existing D2 QF/DC closure;
- pure local Gaussian/uniform/CB-tail/VB-tail occupancy, recombination, charge,
  and analytic tangents with source/node evidence;
- content-addressed local energy-order refinement independent of space and
  solver tolerance;
- strict SCAPS-shaped distributed conversion with frozen total/peak, energy
  reference, support, unit, and SHA-256 fixtures.

Still fail closed:

- any distributed closure in the production material/experiment path;
- distributed contact neutrality, Poisson coupling, J-V, AC, and transient
  execution;
- legacy or standard YAML activation of distributed inputs; only the explicit
  strict adapter can create a normalized SCAPS-shaped v2 species;
- spatially graded density or energy reference;
- dynamic occupancy, non-unit degeneracy, and multivalent defects.

D3-E3 is the first checkpoint allowed to enable distributed defects in
production QF/DC.

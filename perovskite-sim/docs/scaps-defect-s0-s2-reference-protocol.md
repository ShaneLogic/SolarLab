# SCAPS S0-S2 explicit-defect reference protocol

Status: importer implemented; no external SCAPS reference is bundled.

This protocol creates a reference only from independently executed SCAPS-1D
decks and direct, unmodified profile rows. SolarLab output, fitted curves,
interpolation, resampling, or reconstructed SCAPS values are not accepted as
external evidence.

## Frozen scenarios

The source-of-truth suite is
`reproducibility/scaps_defect_s0_s2_suite.json`. It pins the canonical SolarLab
config and SHA-256 for:

- `S0`: neutral single-level defect in an intrinsic slab;
- `S1`: acceptor defect in an n-type slab;
- `S2`: donor defect in a p-type slab.

The importer re-hashes the suite and all three canonical configs before reading
external data.

## Direct CSV contract

Each scenario needs one CSV with exactly this header and at least three direct
SCAPS rows:

```text
position_um,electron_density_cm3,hole_density_cm3,electrostatic_potential_V,conduction_band_eV,valence_band_eV,defect_occupancy,defect_charge_number_cm3,recombination_rate_cm3_s
```

Rows must be finite and strictly increasing in position. The first and last
positions must match the left contact and declared slab thickness. Carrier
densities must be positive, occupancy must lie in `[0, 1]`, and the band gap
must be positive at every row. The signed defect number density must be zero
for S0, nonpositive and nonzero for S1, and nonnegative and nonzero for S2.

## Parameter manifest

The JSON manifest schema is
`solarlab.scaps_explicit_defect_parameter_manifest`, version `1.0`. Its exact
top-level keys are:

```text
schema, schema_version, solver, numerics, unit_conventions,
sign_conventions, comparison_protocol, scenarios
```

`scenarios` must contain exactly `S0`, `S1`, and `S2`. Each row must record:

```text
canonical_config_sha256, charge_transition, doping_polarity,
source_deck_format, thickness_um, scaps_parameters
```

The canonical hashes and physical labels must match the frozen suite.
`scaps_parameters` must contain the complete values entered into that SCAPS
deck, including material, contact, defect, operating-point, and illumination
settings. `numerics` must record the mesh and convergence settings actually
used. The manifest solver version must equal the CLI value.

The comparison protocol is fixed to dark equilibrium at zero bias, direct
export rows only, with interpolation disabled. The electrostatic-potential
sign/reference convention must be declared explicitly; it is preserved rather
than guessed by the importer.

## Import command

```bash
python scripts/import_scaps_defect_reference.py \
  --project-root . \
  --suite reproducibility/scaps_defect_s0_s2_suite.json \
  --parameter-manifest /path/to/scaps-s0-s2-parameters.json \
  --s0-csv /path/to/S0-profile.csv \
  --s0-source-deck /path/to/S0.def \
  --s1-csv /path/to/S1-profile.csv \
  --s1-source-deck /path/to/S1.def \
  --s2-csv /path/to/S2-profile.csv \
  --s2-source-deck /path/to/S2.def \
  --solver-version 3.3.11 \
  --extracted-at 2026-08-28 \
  --operator "operator identity" \
  --confirm-independent-scaps-export \
  --confirm-direct-unmodified-rows \
  --out /path/to/scaps-defect-s0-s2-reference.json
```

The importer embeds every raw profile, hashes every CSV/deck/manifest/suite,
and adds a canonical `reference_content_sha256`. It refuses to overwrite an
existing output. The resulting artifact supports later cross-code comparison;
its existence alone does not establish parity. Grid-aligned differences and
pre-registered acceptance thresholds must still be evaluated and reported.

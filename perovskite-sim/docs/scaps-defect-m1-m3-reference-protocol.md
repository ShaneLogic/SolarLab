# SCAPS M1-M3 multivalent-defect reference protocol

Status: importer implemented; no external SCAPS reference is bundled.

This protocol creates a reference only from independently executed SCAPS-1D
decks and direct, unmodified profile rows. SolarLab output, fitted curves,
interpolation, resampling, or reconstructed SCAPS values are not accepted as
external evidence. It parallels
`docs/scaps-defect-s0-s2-reference-protocol.md`; the differences below are
exactly the multivalent ones.

## Frozen scenarios

The source-of-truth suite is
`reproducibility/scaps_multivalent_defect_suite.json`. It pins the canonical
SolarLab config and SHA-256 for:

- `M1`: double donor (charge states `+2, +1, 0`) in a p-type slab;
- `M2`: double acceptor (charge states `0, -1, -2`) in an n-type slab;
- `M3`: amphoteric (charge states `+1, 0, -1`) in an intrinsic slab.

The importer re-hashes the suite and all three canonical configs before
reading external data. The suite file is additionally byte-hash-pinned by the
`multivalent-explicit-defect-qf-dc-v1` lane in
`reproducibility/numerical_refinement_registry.yaml`, and its
`external_reference_contract` block is embedded in that lane's frozen protocol
hash — which is why the per-state occupation encoding below is defined here
and in the importer, NOT by editing the suite's `raw_profile_columns`.

## Direct CSV contract

Each scenario needs one CSV with exactly this header and at least three direct
SCAPS rows:

```text
position_um,electron_density_cm3,hole_density_cm3,electrostatic_potential_V,conduction_band_eV,valence_band_eV,defect_charge_number_cm3,recombination_rate_cm3_s,charge_state_occupation_fraction_per_state
```

Rows must be finite and strictly increasing in position. The first and last
positions must match the left contact and declared slab thickness. Carrier
densities must be positive and the band gap positive at every row.

### Per-state occupation encoding (normative)

`charge_state_occupation_fraction_per_state` holds, in ONE cell, the three
charge-state occupation fractions `P_0|P_1|P_2`:

- separator: `|` (pipe; CSV-safe, locale-safe);
- order: most positive state first — the same order as the frozen
  `charge_states_e` (`M1: +2,+1,0`; `M2: 0,-1,-2`; `M3: +1,0,-1`) and the
  same order SCAPS uses on the multiple-level defects properties panel
  ("level 1 the most positive charge", manual §3.6.2.3);
- each fraction in `[0, 1]`; the three must sum to 1 within the manifest's
  `occupation_fraction_sum_tolerance` (at most `1e-3`).

### Net-charge rules (normative)

`defect_charge_number_cm3` is the state-weighted net defect charge
`Nt * sum_s(q_s * P_s)`:

- `M1` must be nonnegative and nonzero; `M2` nonpositive and nonzero;
- `M3` carries NO sign rule — its net charge legitimately changes sign with
  position; that is the scenario's physical content, not an error;
- every row must reproduce `Nt * sum_s(q_s * P_s)` from the same row's
  fractions within
  `net_charge_consistency_tolerance_relative_to_total_density * Nt`
  (tolerance at most 1% of `Nt`). A SCAPS export that disagrees is a finding
  to report back, never data to edit into agreement.

## Parameter manifest

The JSON manifest schema is
`solarlab.scaps_multivalent_defect_parameter_manifest`, version `1.0`. Its
exact top-level keys are:

```text
schema, schema_version, solver, numerics, unit_conventions,
sign_conventions, comparison_protocol, scenarios
```

`sign_conventions.defect_charge` must be `state_weighted_net_charge` (the
monovalent `negative_acceptor_positive_donor` literal does not describe a
three-state defect). `numerics` must additionally record the SCAPS
`recalculate_mesh` setting — the SCAPS manual (§5.1.1) flags the static mesh
as potentially insufficient for multivalent defects, so the setting actually
used must be on record whatever it was.

`comparison_protocol` exact keys:

```text
charge_state_order, interpolation_allowed,
net_charge_consistency_tolerance_relative_to_total_density,
occupation_fraction_separator, occupation_fraction_sum_tolerance,
operating_point, position_tolerance_um, row_policy
```

with `charge_state_order = "most_positive_first"`,
`occupation_fraction_separator = "|"`, and the same fixed operating point as
S0-S2 (dark equilibrium, zero bias, direct export rows only, no
interpolation, position tolerance at most `1e-6` um).

`scenarios` must contain exactly `M1`, `M2`, and `M3`. Each row's exact keys:

```text
canonical_config_sha256, charge_states_e, degeneracy_convention,
doping_polarity, energy_reference, family, scaps_parameters,
source_deck_format, state_degeneracies, thickness_um,
total_defect_density_cm3, transition_capture_cross_sections_cm2,
transition_energies_eV_above_vb
```

The importer cross-checks the defect document against the frozen suite:

- `family`/`doping_polarity`/`canonical_config_sha256` must match the suite;
- `charge_states_e` must be exactly the frozen per-family list;
- `degeneracy_convention` must be `scaps_binomial` for M1/M2 (SCAPS default,
  Eq. 9 of the manual: `g_s = C(H, s)`) and `unity` for M3 (the SCAPS
  "set to one" checkbox); `state_degeneracies` must be `[1, 2, 1]` and
  `[1, 1, 1]` respectively. SCAPS offers exactly these two conventions and
  no free per-state values (manual §3.6.2.3), which is what M3 tests;
- `energy_reference` must be `above_valence_band` (SCAPS option "above EV");
- `transition_energies_eV_above_vb` must hold exactly two strictly
  increasing positive energies;
- `transition_capture_cross_sections_cm2` must hold exactly two
  `{sigma_n_cm2, sigma_p_cm2}` sets and the two must differ — entering one
  set twice is the classic silent operator error the contract exists to
  catch. SCAPS exposes the cross sections per level on the multiple-level
  defects properties panel (manual §3.6.2.3); the thermal velocities are
  layer-level (`layer.vthn`/`layer.vthp`), shared by both transitions.

`scaps_parameters` must contain the complete values entered into that SCAPS
deck; `numerics` the mesh and convergence settings actually used. The
manifest solver version must equal the CLI value.

## Import command

```bash
python scripts/import_scaps_multivalent_defect_reference.py \
  --project-root . \
  --suite reproducibility/scaps_multivalent_defect_suite.json \
  --parameter-manifest /path/to/scaps-m1-m3-parameters.json \
  --m1-csv /path/to/M1-profile.csv \
  --m1-source-deck /path/to/M1.def \
  --m2-csv /path/to/M2-profile.csv \
  --m2-source-deck /path/to/M2.def \
  --m3-csv /path/to/M3-profile.csv \
  --m3-source-deck /path/to/M3.def \
  --solver-version 3.3.11 \
  --extracted-at 2026-09-02 \
  --operator "operator identity" \
  --confirm-independent-scaps-export \
  --confirm-direct-unmodified-rows \
  --out /path/to/scaps-defect-m1-m3-reference.json
```

The importer embeds every raw profile (the fraction cells verbatim plus a
parsed `charge_state_occupation_fractions` list), hashes every
CSV/deck/manifest/suite, and adds a canonical `reference_content_sha256`. It
refuses to overwrite an existing output. The resulting artifact supports
later cross-code comparison; its existence alone does not establish parity.
Grid-aligned differences and pre-registered acceptance thresholds must still
be evaluated and reported.

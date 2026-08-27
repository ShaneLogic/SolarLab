# Explicit Bulk-Defect Input Contract v1

Status: DEF-0 input contract. The schema is executable as configuration data,
but `explicit_quasi_steady` remains solver-gated until the later recombination
and charged-DC checkpoints are complete.

## 1. Compatibility rule

An existing layer that declares none of the new keys continues to use
`tau_n`, `tau_p`, `n1`, and `p1`. Its effective model is
`effective_lifetime`; no defect species is inferred.

The three new layer keys are atomic:

```yaml
defect_schema_version: solarlab-explicit-bulk-defects-v1
defect_model: explicit_quasi_steady
bulk_defects: []
```

If any one is present, all three are required. Unknown fields, incomplete
nested blocks, non-finite values, ambiguous energy references, and unsupported
model names raise before a simulation starts.

`defect_model` selects exactly one recombination representation:

- `effective_lifetime`: production compatibility path; only
  `tau_n/tau_p/n1/p1` execute.
- `explicit_quasi_steady`: reserved production target; microscopic species
  will execute after DEF-1/DEF-3. In DEF-0 it fails closed at material-array
  construction instead of silently falling back to lifetime SRH.
- `explicit_dynamic`: reserved for a future schema and rejected by v1.

Microscopic species retained by the SCAPS adapter while
`defect_model: effective_lifetime` are inactive provenance. They are never
added on top of lifetime SRH. The old `bulk_trap_distribution` research closure
and the new `bulk_defects` inventory cannot coexist on one material.

## 2. Canonical SI schema

The first explicit production target is a named, charge-resolved, single-level
bulk defect with integrated volume density:

```yaml
defect_schema_version: solarlab-explicit-bulk-defects-v1
defect_model: explicit_quasi_steady
bulk_defects:
  - name: absorber_acceptor_1
    distribution:
      kind: single_level
      normalization: integrated_total
      total_density_m3: 1.0e22
      center_eV_above_vb: 0.60
    charge_transition: acceptor
    neutral_reference: empty
    kinetics:
      sigma_n_m2: 1.0e-19
      sigma_p_m2: 1.0e-19
      thermal_velocity_n_m_s: 1.0e5
      thermal_velocity_p_m_s: 1.0e5
    degeneracy: 1.0
```

All physical quantities use canonical SI except energies, which use eV:

| Field | Meaning | Unit / domain |
|---|---|---|
| `total_density_m3` | Energy-integrated defect density | m^-3, positive |
| `center_eV_above_vb` | Defect energy measured upward from the valence band | eV, `0 <= E_t <= E_g` |
| `sigma_n_m2`, `sigma_p_m2` | Electron/hole capture cross sections | m^2, non-negative |
| `thermal_velocity_n_m_s`, `thermal_velocity_p_m_s` | Carrier thermal velocities | m/s, positive |
| `degeneracy` | Occupancy degeneracy factor | dimensionless, positive |

`normalization: integrated_total` is mandatory. A peak density is not accepted
as a substitute because doing so would change the integrated inventory when an
energy width changes.

## 3. Charge and neutral reference

The charge convention is part of the input identity, not an inferred default:

| `charge_transition` | `neutral_reference` | Charge convention |
|---|---|---|
| `neutral` | `all_occupancies` | Defect contributes recombination but no electrostatic charge |
| `acceptor` | `empty` | Empty is neutral; occupied is negative |
| `donor` | `filled` | Filled is neutral; empty is positive |

SCAPS-shaped legacy entries that omit both fields are stored as
`unresolved/unresolved` only under `effective_lifetime`. They cannot activate
`explicit_quasi_steady` until a user supplies the physical transition and
neutral reference. SolarLab does not guess donor/acceptor from trap energy.

## 4. SCAPS-shaped adapter

The adapter retains the existing SCAPS-friendly inputs and converts them to the
canonical object:

```yaml
bulk_defects:
  - name: absorber_acceptor_1
    sigma_n_cm2: 1.0e-15
    sigma_p_cm2: 1.0e-15
    N_t_cm3: 1.0e16
    E_t_eV_above_vb: 0.60
    charge_transition: acceptor
    neutral_reference: empty
    degeneracy: 1.0
```

Conversion rules are:

```text
N_t[m^-3] = N_t[cm^-3] * 1e6
sigma[m^2] = sigma[cm^2] * 1e-4
E_t[above VB] = E_g - E_t[below CB]
```

Exactly one of `E_t_eV_above_vb` and `E_t_eV_below_cb` is required. Decimal
unit scaling is used before conversion to binary float, so physically identical
SI and cgs documents produce the same canonical JSON and SHA-256.

SCAPS `distribution: gaussian` can be retained as inactive compatibility
metadata. Its `E_char_eV` is labelled
`scaps_characteristic_energy`, not silently reinterpreted as a Gaussian
standard deviation. Explicit v1 execution accepts only `single_level`; energy
distributions enter in the later D3 phase with a separately tested quadrature
and normalization contract.

## 5. Canonical identity

`BulkDefectDocument.canonical_json()` recursively serializes the version,
selector, ordered species list, units, charge convention, kinetics, and
degeneracy with sorted keys and finite JSON numbers.
`BulkDefectDocument.sha256` hashes the ASCII canonical JSON.

Species order is retained because result diagnostics will be aligned with the
input list. Names must be unique for explicit execution. An unnamed species is
allowed only as unresolved legacy provenance.

The general `DeviceStack` semantic hash deliberately omits inactive
`effective_lifetime` provenance so frozen numerical baselines do not move.
Selecting `explicit_quasi_steady` includes the full document in semantic
identity. The defect-document SHA-256 remains available in both modes.

## 6. Migration policy

### Existing standard SolarLab YAML

No change is required. Absence of the new three-key block is the historical
`effective_lifetime` path. The loader and backend must not create species from
`tau_n/tau_p/n1/p1`.

### Existing SCAPS-shaped YAML

The adapter continues to derive the same compatibility
`tau_n/tau_p/n1/p1`. It now also retains the microscopic list as inactive,
versioned provenance. Missing charge transitions remain unresolved.

### Opting into the explicit model

Do not switch only the selector. Before changing to
`explicit_quasi_steady`, every species must have:

1. a unique non-empty name;
2. `normalization: integrated_total`;
3. an in-gap single energy level;
4. a declared charge transition and matching neutral reference;
5. finite SI kinetics and positive degeneracy.

During DEF-0 a valid opt-in document raises
`ExplicitDefectCapabilityError` at solver construction. This is intentional:
the configuration contract is frozen before the numerical model is enabled.

## 7. Frontend field mapping (UI-0)

The future editor must be generated from this contract rather than maintaining
a separate list of physics fields:

- per-layer model selector: `effective_lifetime` or, after certification,
  `explicit_quasi_steady`;
- species table: name, transition, total density, energy level, capture
  cross sections, thermal velocities, and degeneracy;
- unit-aware display: SCAPS-style cm^-3/cm^2 may be accepted at the UI edge,
  while requests carry canonical SI;
- transition-controlled neutral reference: visible and serialized, but not an
  independently inconsistent choice;
- strict prevention of lifetime/explicit double counting and unsupported
  Gaussian/dynamic/interface fields;
- read-only document SHA-256 for configuration provenance.

UI-0 freezes this mapping only. UI-1 begins after the public Python execution
contract is stable at DEF-3; UI-2 becomes a normal configuration path only
after DEF-4 numerical and SCAPS-reference gates pass.

## 8. DEF-0 verification boundary

DEF-0 certifies input and compatibility behavior only:

- immutable schema and strict round-trip;
- SI/cgs canonical equivalence;
- energy, normalization, and charge-reference validation;
- SCAPS parsed-species retention;
- backend serialization round-trip;
- inactive metadata is bitwise inert in the RHS;
- explicit execution fails closed.

It does not claim exact multi-species SRH, defect space charge, contact
thermodynamic closure, Poisson coupling, J-V parity with SCAPS, AC response, or
dynamic occupancy. Those claims require the later checkpoints in the roadmap.

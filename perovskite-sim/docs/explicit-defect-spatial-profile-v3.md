# Explicit Bulk-Defect Spatial Profile Contract v3

Status: D3-E4b production implementation and regression verification complete.
The
canonical D3-E4a input contract is now compiled into the guarded QF/DC material,
contact, Poisson, continuity, analytic-tangent, and J-V paths. Independent
three-axis certification remains a D3-E4c requirement.

## Version boundary

The schema version is `solarlab-explicit-bulk-defects-v3`. Versions 1 and 2
retain their prior canonical JSON, document hashes, and device semantic hashes.
A v1 or v2 species cannot carry `spatial_profile`; a v3 document must contain
at least one profiled species. An unprofiled species inside a v3 document has
the exact, explicit meaning of a uniform density.

## Density meaning

`distribution.total_density_m3` remains the layer-average volumetric density
`N_bar`, including for energy-distributed sources. At normalized layer
coordinate `s=x_local/d`,

```text
N_source(s) = N_bar * m(s)
integral from 0 to 1 of m(s) ds = 1
integral from 0 to d of N_source(x) dx = N_bar * d.
```

The multiplier is piecewise linear between declared knots. SolarLab evaluates
the exact trapezoidal integral of the continuous knot function and rejects the
document unless it equals one within an absolute tolerance of `1e-12`. It does
not silently renormalize a user profile. Knot positions must be strictly
increasing, start at exactly 0, end at exactly 1, and all multipliers must be
finite and strictly positive.

```yaml
defect_schema_version: solarlab-explicit-bulk-defects-v3
defect_model: explicit_quasi_steady
bulk_defects:
  - name: graded_acceptor
    distribution:
      kind: gaussian
      normalization: integrated_total
      total_density_m3: 2.0e22
      energy_reference: above_valence_band
      center_eV_above_vb: 0.55
      width_eV: 0.08
      width_convention: scaps_characteristic_energy
      support_width_multiplier: 6.0
    spatial_profile:
      coordinate: normalized_layer_coordinate
      interpolation: piecewise_linear
      density_normalization: layer_average_unity
      knots:
        - {position_fraction: 0.0, density_multiplier: 0.5}
        - {position_fraction: 0.5, density_multiplier: 1.0}
        - {position_fraction: 1.0, density_multiplier: 1.5}
    charge_transition: acceptor
    neutral_reference: empty
    kinetics:
      sigma_n_m2: 1.0e-19
      sigma_p_m2: 2.0e-19
      thermal_velocity_n_m_s: 1.0e5
      thermal_velocity_p_m_s: 1.2e5
    degeneracy: 1.0
```

## Energy reference

Every v3 energy remains measured above the local valence-band edge. The schema
does not introduce a spatial energy-shift field. When band grading is enabled,
the absolute trap energy moves with the local valence band while
`center_eV_above_vb`, width, support, capture kinetics, degeneracy, and charge
convention remain source constants. The complete support must fit inside the
local band gap at every active node.

This separates two concepts that cannot be inferred from each other:

- the spatial population profile changes how much of one source exists at a
  position;
- the energy distribution defines how that local population is apportioned
  across levels relative to the local valence band.

At a contact face, localization resolves the endpoint density
`N_bar*m(0 or 1)` and removes the profile before the scalar contact-neutrality
closure is evaluated. The source object and its canonical v3 document remain
unchanged.

## Production closure

The guarded `qf_dc` material compiler evaluates the profile at each owned grid
node and binds the local band gap and effective densities of states to the same
region. One carrier-independent energy quadrature is retained per physical
source; the local density weights are its normalized weights multiplied by
`m(s)`. Occupancy, SRH recombination, defect charge, carrier derivatives, and
the fixed-quasi-Fermi Poisson tangent are then evaluated from the same local
energy nodes and local band edges.

The front and back reservoirs use the exact endpoint-localized source document.
Thus contact neutrality, built-in potential, interior Poisson charge, continuity
recombination, and J-V diagnostics all share the same source density and neutral
reference. A distributed support that leaves the smallest local band gap is
rejected while building the material cache, before a nonlinear solve begins.

Point diagnostics retain the profile SHA-256 values, nodewise density
multipliers, and per-source multiplier bounds. The J-V/backend summary retains
the hashes and bounds and rejects a sweep whose spatial identity changes across
voltage points. The frontend evidence strip displays how many species are
profiled and the aggregate multiplier range.

## Current capability boundary

Implemented through D3-E4b:

- immutable knot/profile dataclasses and strict mapping parser;
- canonical JSON and SHA-256 for the profile and enclosing v3 document;
- exact piecewise-linear interpolation and layer-average conservation check;
- scalar endpoint/interior localization without changing energy metadata;
- standard YAML/backend round-trip and v3 device semantic hashing;
- frozen v1/v2 document and shipped-config semantic hashes.
- local-band-edge material compilation and endpoint contact localization;
- spatial single-level and distributed-source occupancy, charge,
  recombination, and analytic tangents;
- dark equilibrium and illuminated QF J-V execution on the guarded QF/DC path;
- profile identity and multiplier evidence in point, J-V, backend, and frontend
  results;
- uniform v3 limits for both single-level and Gaussian v2 sources, local-band
  support rejection, and compiled layer-average density checks.

Still outside the D3-E4b capability label:

- a source-clean independent energy x space x solver-tolerance certificate;
- any claim that the graded model has SCAPS parity or experimental validation;
- zero-density knots or discontinuous step profiles;
- spatial changes to energy center, width, support, kinetics, degeneracy, or
  charge transition;
- mobile ions, Fermi-Dirac statistics, AC, dynamic occupancy, interface
  defects, tunnelling, and multivalent charge states;

## Verification

- spatial constitutive/device/QF/backend focused domain: 93 passed;
- default Python suite: 3059 passed, 2 skipped, 264 deselected;
- frontend suite: 428 passed across 32 files;
- TypeScript, Vite production build, compileall, scoped Ruff, critical Ruff,
  and `git diff --check`: passed;
- all 12 Python warnings are pre-existing NumPy `trapz` compatibility/MMS
  warnings, and the Vite build retains its pre-existing large-chunk warning.

D3-E4c will use a new frozen graded configuration and an independent energy x
space x solver-tolerance certificate. The D3-E3 uniform-layer certificate
cannot be reused. D3-E4b establishes production execution and compatibility;
it does not establish three-axis convergence, SCAPS parity, or experimental
validation.

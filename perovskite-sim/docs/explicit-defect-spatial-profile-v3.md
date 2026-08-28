# Explicit Bulk-Defect Spatial Profile Contract v3

Status: D3-E4b production implementation and regression verification complete;
D3-E4c independent certificate contract registered and single-cell validated.
The canonical D3-E4a input contract is compiled into the guarded QF/DC
material, contact, Poisson, continuity, analytic-tangent, and J-V paths. The
source-clean three-axis D3-E4c matrix remains pending.

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

## D3-E4c registered certificate checkpoint

The independent lane is
`spatially-graded-explicit-defect-qf-dc-v1`. Its frozen device is
`configs/graded_distributed_defect_qf_dc_pn.yaml` (SHA-256
`b6db3b87c509bafc7dae5e84640b12a0277668d612b4f9933f848a3ad99a715f`).
The device contains a continuous 0.84 -> 0.80 -> 0.84 eV band-gap notch, four
energy-distributed sources, three conservative spatial profiles, and one
explicitly uniform v3 source.

The outer matrix is 16/32/64 intervals per electrical layer by solver residual
factors 1/0.1/0.01. Every cell independently evaluates defect energy orders
16/32/64. Thirteen observables compare contact state, local bands, carrier and
potential profiles, source multiplier/charge/occupancy profiles, integrated
defect charge, J-V, and Jsc. Thirty-six per-cell quality gates bind topology,
profile conservation and identity, local support, contacts, energy refinement,
state/residual/current closure, positivity, and the three-point illuminated
J-V path.

The first dirty-source runner artifact (`4f9cd563...`) failed closed before the
nonlinear solve because a nearly saturated positive-weight energy distribution
formed `sum(N_i*f_i)/N_total = 1 + epsilon`. The artifact was retained. The
constitutive closure now evaluates the same convex mean as
`sum(w_i*f_i)` below half occupancy and as
`1 - sum(w_i*(1-f_i))` above half occupancy. These expressions are
algebraically identical; the second avoids subtractive loss near saturation
without clipping. Uniform and spatial aggregates use the same helper, and a
near-saturated valence-band-tail regression pins the bound.

A second dirty-source one-cell run (`2a97bb6b...`) completed
`grid=16,tolerance=1` in 44.13 s with all 36 frozen quality gates passing. Its
certificate (`a025f1e463c9d0a040d532775341543cb001f693eeef1b33e9ae3e118cfd7e1a`)
is intentionally `failed` because the other eight cells are missing. It is not
three-axis convergence evidence and will not be reused as the final
certificate.

Adding this graded absorber also exercised a pre-existing temporary V_oc guard.
`thermodynamic_voc_ceiling` now uses the minimum local absorber gap, including
an exact positive-bowing interior minimum. A narrower transport-layer gap no
longer creates an artificially strict thermodynamic ceiling. Stacks without an
absorber role retain the old electrical-layer fallback; stacks with an
undeclared absorber gap return no ceiling instead of borrowing a transport
gap.

Checkpoint verification after these fixes:

- D3-E0 through D3-E4c expanded domain: 182 passed;
- ceiling/grading/reproducibility focus: 44 passed;
- certificate contract focus: 21 passed, 1 deselected;
- default Python suite: 3069 passed, 2 skipped, 264 deselected, 12 pre-existing
  NumPy compatibility/MMS warnings;
- scoped Ruff, compileall, and `git diff --check`: passed.

This checkpoint registers and validates the execution contract. It does not
promote the graded configuration to internally certified until a source-clean
9/9 matrix closes every registered comparison and quality gate. It also makes
no SCAPS parity or experimental-validation claim.

### Immutable v1 partial result and v2 protocol

The source-clean v1 run
`490529820738b29820c70d5fb67a27cb9e62fcde5bc64f680d49cda03ea74ecf`
completed all 9 cells with no failures, missing cells, or reused cells. Its
nine artifact hashes are unique. Of 350 certificate checks, 349 passed,
including all 26 grid/tolerance observable comparisons. The only failure was
`max_dark_current_A_m2` for `grid=64,tolerance=1`: `1.904082e-6 A/m2`
against the frozen `1e-6 A/m2` limit. At the same grid, factors 0.1 and 0.01
gave `3.010e-12` and `9.519e-13 A/m2`, respectively. Thus certificate
`a8c6545c90c380349ccde15354956ef84da6f769686f7c3b072d71c5f91987b6`
is retained as `partial`; neither the artifact nor the gate is changed.

Lane `spatially-graded-explicit-defect-qf-dc-v2` is the versioned follow-up.
It retains the v1 config, three grids, factors 1/0.1/0.01, energy orders,
observables, and quality gates. Its only numerical change is the base Newton
residual tolerance from `1e-8` to `1e-9`, which makes the v2 coarse factor
equal to the already observed passing v1 factor 0.1. The v2 matrix must still
be recomputed from a new source-clean commit before any certification claim.

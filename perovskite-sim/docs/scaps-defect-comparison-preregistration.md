# SCAPS defect-reference comparison: pre-registered protocol

Status: frozen 2026-09-02, BEFORE any external SCAPS export existed (both
suites carry `status: "not_supplied"` at the freezing commit — the git
history is the pre-registration timestamp). Machine-readable half:
`reproducibility/scaps_defect_comparison_thresholds.json`. Executable half:
`scripts/compare_scaps_defect_reference.py`. Changing a threshold after data
arrives requires a new `schema_version` and an explicit note that the change
is post-hoc.

## What is being compared

The dark-equilibrium zero-bias profiles of the six frozen single-layer
scenarios (S0-S2 monovalent, M1-M3 multivalent), as delivered through the
fail-closed importers, against SolarLab's QF/DC solution of the SAME
hash-bound canonical configs. Nothing else: no J-V, no illumination, no
derived p/n device. A PASS is external cross-solver evidence for the
dark-equilibrium bulk defect closure (D7-E2 / DEF-4's external half) — it is
NOT device-level parity and NOT independent validation of any calibrated
lane.

## Alignment policy

- SolarLab solves each scenario with `solve_quasi_fermi_steady_state`
  (dark, 0 V) on the lane-certified grid family (`multilayer_grid`,
  alpha 2.0) at the pre-registered interval counts, with the lane's base
  solve controls.
- The external rows are NEVER interpolated (importer red line). SolarLab's
  own nodal solution is interpolated onto the SCAPS export positions —
  linearly, except carrier densities which interpolate in log10 space.
  In-repo precedent: the refinement lanes' `_fixed_profile` observables.
- **Zero references differ between solvers by convention**, so
  `electrostatic_potential_V`, `conduction_band_eV`, and `valence_band_eV`
  are compared after subtracting each side's own first-row value
  (left-contact anchoring). The band gap is additionally compared rowwise
  and absolutely (`E_C - E_V` carries no zero freedom).
- Net defect charge is compared normalized by the frozen total density
  `Nt` (from the hash-bound config, not from the manifest).
- Net recombination at dark equilibrium is ~0 in both solvers by detailed
  balance; comparing two zeros is meaningless, so the column is a
  **near-zero gate**: max(|R_scaps|, |R_solarlab|) must stay under an
  absolute bound.

## Verdicts and the grid gate

Per column: the comparator also re-solves on a coarser sensitivity grid and
computes the SAME metric between SolarLab's two solutions. If that grid
sensitivity exceeds 25% of the column threshold the verdict is
`INDECISIVE_GRID` (fail closed: the comparison cannot resolve the question
at this discretization), else `PASS`/`FAIL` against the threshold. Overall:
`FAIL` if any column fails, else `INDECISIVE_GRID` if any column is
indecisive, else `PASS`. A report produced with CLI grid overrides is
marked `preregistered_settings: false` and is diagnostic only.

## Thresholds and their basis

Measured SolarLab-side facts backing the choices (probe 2026-09-02, BLAS
pinned; all six scenarios):

- all profiles are spatially uniform at this operating point — the slab is
  at flat band, potential span 0.0000 mV at every tested grid;
- grid doubling (64->128 and 48->96 intervals) moves every extracted
  quantity by ~0 (uniform profiles), so the 25% sensitivity gate is slack
  in practice and exists to catch future protocol drift;
- equilibrium net recombination residual: max over scenarios
  `2.2e8 m^-3 s^-1` = `2.2e2 cm^-3 s^-1` (M3 worst), i.e. ~1e-16 of the
  characteristic defect-exchange scale `sigma*vth*Nt*ni ~ 7e24 m^-3 s^-1`;
- occupancies span the full discriminating range (S1 0.99996, S2 0.00007,
  S0 0.59552; M1 P(+2)=0.986, M2 P(0)=0.000, M3 P(+1)=0.049), so a
  degeneracy-convention or level-mapping error on the SCAPS side cannot
  hide inside the tolerance.

| column metric | threshold | rationale |
|---|---|---|
| electron/hole density, L-inf log10 | 0.02 (~4.7%) | same primitive parameters both sides; generous allowance for export rounding and statistics conventions; a factor-2 degeneracy error is 0.30 in log10 — 15x the bound |
| potential, left-anchored L-inf | 5 mV | profiles are flat (measured 0.0000 mV span); 5 mV is far below kT=25.9 meV yet tolerant of contact-node artifacts |
| E_C, E_V left-anchored L-inf | 5 meV | same basis as potential |
| band gap rowwise L-inf | 2 meV | Eg=0.8 eV is an input on both sides; only export rounding remains |
| defect occupancy (S) / state fractions (M), abs L-inf | 0.02 | a wrong degeneracy convention shifts S-occupancy by O(0.1-0.5) at these operating points; a level-pair swap reorders M fractions by O(1) |
| normalized net charge rho/Nt, abs L-inf | 0.02 | mirrors the internal lane observable (internal bound 5e-4; cross-solver 40x looser) |
| recombination near-zero bound | 1e6 cm^-3 s^-1 | 4 000x above SolarLab's measured residual, ~13 orders below the physical exchange scale; roomy for SCAPS's own convergence residual while still meaning "zero" |

## Running it

```bash
python scripts/compare_scaps_defect_reference.py \
  --project-root . \
  --reference reproducibility/scaps_defect_s0_s2_reference.json \
  --out reproducibility/scaps_defect_s0_s2_comparison_report.json
```

Same command for the multivalent reference (the schema field selects the
mode). The report embeds the reference/thresholds/suite hashes and its own
`comparison_content_sha256`, and refuses to overwrite. Interpretation of an
`INDECISIVE_GRID` or `FAIL`: report back and investigate; do not adjust
thresholds, configs, or data to reach PASS.

Tests: `tests/unit/validation/test_compare_scaps_defect_reference.py`
(self-comparison passes; zero-point shifts are absorbed; density, fraction,
and recombination perturbations fail their own columns; tampered hashes and
suite drift reject; overwrite refused; overrides drop preregistered status).

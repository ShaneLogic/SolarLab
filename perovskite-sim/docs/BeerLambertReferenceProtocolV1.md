# Beer-Lambert Research Baseline

New device simulations and research comparisons use Beer-Lambert by default.
The electrical physics tier does not select the optical model.

- `beer_lambert`: existing scalar absorption, layer `alpha` and device `Phi`.
- `beer_lambert_spectral`: single-pass AM1.5G, 300-1000 nm on 200 samples,
  `alpha(lambda) = 4*pi*k(lambda)/lambda` from the existing optical materials.
  No reflection, interference, or TMM-derived photon recycling is applied.
- `tmm`: explicit opt-in for historical optical regression and dedicated TMM
  development. This is not the new research default.

Historical TMM tests and uniform-generation literature protocols keep their
declared input models. They must not be relabelled as Beer-Lambert results.
A change of optical model or prescribed generation defines a new protocol.
Tandem and EQE/EL routes with their own optical implementation must not be
presented as Beer-Lambert runs unless that route explicitly supports it.

## SCAPS Reproduction

The shipped `scaps_mirror_v2.yaml` entry uses spectral Beer-Lambert and carries
the exact standard staircase defaults: N_grid=30, 40 voltage points, 5 V/s,
Vmax=1.6 V, transient Radau, illuminated, 300 K, forward metrics. The solver
defaults are rtol=1e-4 and atol=1e-6. This is a finite-time comparison, not a
steady-state residual or grid-convergence certificate.

The historical parameter-sweep helper recomputes Vbi from the doped-layer
work functions at each point; the unswept baseline keeps its original 1.30 V.
Both values are explicitly saved, not inferred from a figure.

The frontend's Report parameters loader uses
`GET /api/reference-studies/scaps-beer-lambert?sweep=CHI_ETL&value=0.5`.
It returns the normalized device config and scan hints. Loading a point
restores the report baseline before applying that point. Subsequent edits
remain normal user edits. Run through the ordinary J-V job endpoint.

The runner `scripts/run_scaps_beer_lambert.py` uses the same loader, inline
device parser and J-V dispatch. Each point saves Request.json, Result.json,
the complete forward/reverse curves, status and request SHA-256. The report
is generated from those saved results, not from historical figure images.
The runner freezes a source copy under `Source/` before executing workers;
the manifest records the code fingerprint and Python/NumPy/SciPy versions.

The workbook is extracted read-only with `--prepare-reference` into
`perovskite_sim/data/ScapsBeerLambertReferenceV1.json`. All 93 parameter
values across 11 families are retained, including two CBO rows without
SCAPS metrics. The extraction pins the workbook hash.

The PVK doping extension appends SolarLab-only 1e19 and 1e20 cm^-3 inputs
in the shared catalog without modifying that workbook-derived JSON. Their
SCAPS values remain null and the frontend marks them as high-doping
extrapolation. The original six PVK points and all other computed results
remain unchanged. Use `--extend-pvk --out <existing-report-directory>` to
append only these two points with the report's original frozen solver;
`PvkExtensionManifestV1.json` records their provenance. These densities reach
or exceed the absorber's configured Nc=1e19 cm^-3, so the unchanged
Maxwell-Boltzmann calculation is a model extrapolation, not a validated
degenerate-carrier prediction.
Both added points use N_grid=90 (91 actual nodes), which passes the existing
interface/Debye resolution preflight at 1e20 cm^-3. Their N_grid=30 attempts
are retained under the point's `Attempts/NGrid30/` folder. The frontend
loads N_grid=90 automatically for this extended range; all original points
keep N_grid=30. Passing this guard is not a grid-convergence certificate.

## Evidence Boundaries

The 2026-07-02 report used TMM. The replacement retains its disclosed
electrical calibration (de-spike=0.53; reservoir barrier=0.42 eV), without
refitting to Beer-Lambert. Current code has changed since July, so a
historical-to-current difference cannot be attributed solely to optics.

The historical bulk-defect sweeps use effective lifetimes and n1/p1, not
independent microscopic defect populations. Both bulk-Nt families scale
the combined effective lifetime. The interface sweep retains the historical
areal-density interpretation in cm^-2 despite the workbook's cm^-3 header;
that source ambiguity remains open. Matching optical model names does not
align the two solvers' absorption data, contacts or interface closures.

Preserve the old `FormalVerify.pdf`. Save the new report and raw evidence
under a distinct version; failed or unbracketed points remain visible.

## Running and Replaying

From `perovskite-sim/`, with the optional `report` dependencies installed:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_scaps_beer_lambert.py --out outputs/scaps_bl_new_run --workers 4
python -m pytest tests/integration/backend/test_scaps_beer_lambert_replay.py -m slow
```

Use a new output directory after changing code or inputs. Existing records
are reused only when their request and source contracts match. The three
stored API regressions cover CBO=0.5 eV, ETL doping=1e10 cm^-3 and effective
bulk trap depth=0.6 eV. The normalized frontend fixture separately guards
all active material parameters against formatting-induced precision loss.

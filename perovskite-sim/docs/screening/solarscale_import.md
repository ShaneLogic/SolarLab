# SolarScale MaterialRecord Import

SolarLab can consume SolarScale `material_records.json` exports through a
readiness-gated importer. The importer is a thin screening layer: it maps
DFT/MD-backed absorber properties into an existing SolarLab template and keeps
device-only unknowns as sweep dimensions or diagnostics.

Run from `perovskite-sim/`.

## Dry-run first

Use a dry run before generating configs:

```bash
PYTHONPATH=. python scripts/run_material_screening.py \
  --records ../../5_SolarScale-runs/exports/material_records.json \
  --policy exploratory \
  --base-config configs/nip_MAPbI3.yaml \
  --top-n 10 \
  --out-dir ../../5_SolarScale-runs/solarlab-screening/exploratory \
  --dry-run
```

This writes `screening_plan.json` with selected candidates, skipped records,
ranking scores, mapped fields, missing fields, sweep parameters, and
diagnostics. It does not write generated YAML configs.

## Generate configs

After inspecting the dry-run plan:

```bash
PYTHONPATH=. python scripts/run_material_screening.py \
  --records ../../5_SolarScale-runs/exports/material_records.json \
  --policy production \
  --base-config configs/nip_MAPbI3.yaml \
  --top-n 3 \
  --out-dir ../../5_SolarScale-runs/solarlab-screening/production
```

With `configs/nip_MAPbI3.yaml`, do not pass `--activate-bandgap`. That legacy
template has no complete `chi`/`Eg` band alignment for all electrical layers,
so the importer keeps the HSE band gap as metadata and leaves the template
`Eg` behavior unchanged.

For an activated band-gap process check, use the dedicated SolarScale template:

```bash
PYTHONPATH=. python scripts/run_material_screening.py \
  --records ../../5_SolarScale-runs/exports/material_records.json \
  --policy exploratory \
  --base-config configs/solarscale_nip_band_aligned.yaml \
  --top-n 1 \
  --out-dir ../../5_SolarScale-runs/solarlab-screening/activated-smoke \
  --activate-bandgap \
  --run-smoke \
  --smoke-n-grid 6 \
  --smoke-n-points 2 \
  --smoke-v-max 0.05
```

`configs/solarscale_nip_band_aligned.yaml` is based on the band-aligned
spiro/MAPbI3/TiO2 TMM preset. It lets the importer replace absorber `Eg` only;
it does not infer a new absorber `chi`, `ni`, `alpha`, or material-specific n,k.

The legacy helper remains available:

```bash
PYTHONPATH=. python scripts/generate_solarscale_inputs.py \
  --records ../../5_SolarScale-runs/exports/material_records.json \
  --template configs/nip_MAPbI3.yaml \
  --out-dir ../../5_SolarScale-runs/solarlab-inputs/production \
  --limit 3 \
  --import-policy production
```

## Import policies

`exploratory` accepts records with `screening.readiness` equal to `phonon` or
`promising`, or records explicitly marked `solarlab_provisional_ready`.

`production` accepts only `promising` records, or records explicitly marked
`solarlab_production_ready`.

Blocked, incomplete, and electronic-only records are skipped. A skipped record
still appears in the plan with the rejection reason and missing inputs.

## Ranking policy

Scores are ranking metadata only:

- `final_fom_score` is used first when present.
- `ml_pv_score` is the fallback when `final_fom_score` is missing.
- Neither score is mapped into `MaterialParams`.

## Parameter mapping

Mapped into the absorber only when provenance is `computed` or `derived`:

- `dielectric_static_avg` -> `eps_r`
- `electron_mobility_cm2_v_s` -> `mu_n`, converted to `m^2/V/s`
- `hole_mobility_cm2_v_s` -> `mu_p`, converted to `m^2/V/s`
- `ion_diffusion_coefficient_m2_s` -> `D_ion`
- `ion_activation_energy_ev` -> `E_a_ion`

`band_gap_hse_ev` is required record metadata, but by default it is not mapped
into absorber `Eg`. It is written under `material_metadata["band_gap_hse_ev"]`
in the plan, manifest, and generated config source block. This avoids a
partial band-alignment state where only the absorber band gap changes while
HTL/ETL `chi`/`Eg` are still legacy defaults.

To explicitly activate HSE band gap as absorber `Eg`, pass
`--activate-bandgap`. The importer accepts this only when the base template is
fully band-aligned, meaning every electrical layer has `chi` and a positive
`Eg`. Using `--activate-bandgap` with `configs/nip_MAPbI3.yaml` raises an error.

`swept`, `missing`, `assumed`, and `literature` provenance are not treated as
fixed DFT/MD inputs by this importer. Missing optional mobility or ion fields
fall back to the template values and are listed in diagnostics.

## Sweep boundary

These fields are treated as SolarLab sweep parameters or template assumptions,
not fixed DFT/MD material properties:

- absorber thickness
- SRH lifetimes
- trap density
- surface or interface recombination velocity
- contact work function and transport-layer band alignment

SLME and absorption-edge metadata are not converted into scalar `alpha` or
optical constants. TMM/n,k import is a later workflow that should write
provenance-labeled CSV files under `perovskite_sim/data/nk/` and update the
manifest.

## Smoke JV

For a small process check, generate configs and run a tiny JV sweep on the top
candidate:

```bash
PYTHONPATH=. python scripts/run_material_screening.py \
  --records ../../5_SolarScale-runs/exports/material_records.json \
  --policy exploratory \
  --base-config configs/nip_MAPbI3.yaml \
  --top-n 1 \
  --out-dir ../../5_SolarScale-runs/solarlab-screening/smoke \
  --run-smoke
```

The smoke result only proves that the imported stack reaches SolarLab's JV API.
It is not a publication-grade simulation result.

The activated smoke path is also process validation only. It confirms that a
SolarScale HSE band gap can enter absorber `Eg` with a band-aligned template,
while optical constants and deeper band-alignment policy remain separate
follow-up work.

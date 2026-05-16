# SolarScale MaterialRecord Import

SolarLab can generate first-pass device configs from the `material_records.json` exported by SolarScale.

Run from `perovskite-sim/`:

```bash
PYTHONPATH=. python scripts/generate_solarscale_inputs.py \
  --records ../../5_SolarScale-runs/exports/material_records.json \
  --template configs/nip_MAPbI3.yaml \
  --out-dir ../../5_SolarScale-runs/solarlab-inputs/dry-run \
  --limit 1
```

The importer writes one YAML config per ready material plus `manifest.json`. It skips records whose `dft_result_available` flag is not true.

## Mapping Boundary

Mapped into the absorber layer:

- static dielectric constant -> `eps_r`
- electron and hole mobility, when available, from `cm^2/V/s` to `m^2/V/s`
- first sweep value for absorber thickness, SRH lifetimes, bulk trap density, and interface recombination velocity

Preserved in generated metadata but not activated in the solver for the legacy template:

- DFT/HSE band gap
- effective masses

This is intentional. SolarLab's band-offset model requires calibrated `chi`/`Eg` values for every electrical layer. Setting only the absorber band gap in a legacy template would activate `compute_V_bi()` with uncalibrated contact-layer offsets. Use a fully band-aligned template before enabling DFT band gaps as active solver inputs.

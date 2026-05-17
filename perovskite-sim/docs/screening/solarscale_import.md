# SolarScale MaterialRecord Import

SolarLab can generate first-pass device configs from the `material_records.json` exported by SolarScale.

Run from `perovskite-sim/`:

```bash
PYTHONPATH=. python scripts/generate_solarscale_inputs.py \
  --records ../../5_SolarScale-runs/exports/material_records.json \
  --template configs/nip_MAPbI3.yaml \
  --out-dir ../../5_SolarScale-runs/solarlab-inputs/production-dry-run \
  --limit 1 \
  --import-policy production
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


## Import Policies

`production` is the default policy. It only imports records with `screening.readiness == "promising"`, meaning electronic, phonon, and MD/ion gates have passed.

`exploratory` also imports `phonon` records. Use it to start SolarLab device sweeps before MD/ion migration is complete:

```bash
PYTHONPATH=. python scripts/generate_solarscale_inputs.py \
  --records ../../5_SolarScale-runs/exports/material_records.json \
  --template configs/nip_MAPbI3.yaml \
  --out-dir ../../5_SolarScale-runs/solarlab-inputs/exploratory-dry-run \
  --limit 3 \
  --import-policy exploratory
```

Exploratory outputs are useful for early device sensitivity studies, but they should not be reported as final promising candidates until MD/ion migration data pass the SolarScale gate.

## DFT/MD Parameters Into SolarLab

Safe direct mappings today:

- `dielectric_static_avg` -> absorber `eps_r`
- `electron_mobility_cm2_v_s` and `hole_mobility_cm2_v_s` -> absorber mobilities after converting to `m^2/V/s`
- `ion_diffusion_coefficient_m2_s` -> absorber `D_ion`, when MD/AIMD provides it
- `ion_activation_energy_ev` -> absorber `E_a_ion`, when MD/AIMD provides it

Preserved as metadata until a band-aligned template is available:

- DFT/HSE band gap
- effective masses
- phonon and MD gate evidence

Still treated as SolarLab sweeps or device-template assumptions:

- lifetime
- trap density
- surface/interface recombination velocity
- absorber thickness
- contact work function and HTL/ETL band alignment

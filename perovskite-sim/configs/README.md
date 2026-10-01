# Research Presets

The backend lists three shipped research YAMLs. The frontend keeps two catalog entries, one for SCAPS and one for the internal Calado-based ion-sweep study. The earlier original-paper comparison remains available to scripts and the API.

| Configuration | Mode | Frontend catalog | Purpose |
| --- | --- | --- | --- |
| [scaps_mirror_v2.yaml](scaps_mirror_v2.yaml) | Fast | SCAPS reproduction | Ion-free electronic heterojunction comparison with SCAPS |
| [calado2016_ion_sweep.yaml](calado2016_ion_sweep.yaml) | Full | Calado 2016 - Ion hysteresis | Internal dense ion-sweep figure with explicit continuous voltage history |
| [calado2016_fig1f.yaml](calado2016_fig1f.yaml) | Legacy | Not a separate catalog entry | Original Fig. 1e/1f toy-device comparison; external agreement remains partial |

The current reproducibility matrix covers 55 configurations: these three and 52 fixtures in `tests/fixtures/configs/`, including four 2D fixtures. Fixture-only configurations are not served by the API.

## SCAPS Parity

Start from `scaps_mirror_v2.yaml`. Keep the optical and contact assumptions explicit when comparing with SCAPS. This is a comparison configuration, not a claim that every external parity target has passed.

Relevant drivers are [run_scaps_full_regression.py](../scripts/run_scaps_full_regression.py) and [run_interface_cbo_scan.py](../scripts/run_interface_cbo_scan.py). Use their declared protocols and evidence gates for quantitative comparisons.

## Internal Ion-Sweep Study

`calado2016_ion_sweep.yaml` is the current frontend ion-study entry. Its reference absorber values are `D_ion = 2.585e-18 m^2/s` and `P0 = 1e25 m^-3`. Relative D/D_ref and c0/c_ref sweeps vary one of these at a time.

The preset declares a -1 -> +1.2 -> -1 V history at 0.04 V/s, uniform absorber generation of 2.5e27 m^-3 s^-1, a 120 s dark seed, 30 s dark preparation, 0.5 s branch dwell, and a 3 s dark turnaround. These settings reproduce an internal figure. They are not an original-paper agreement certificate.

Use the declared waveform and preconditioning when comparing scan-rate or ion-population trends. A generic `run_jv_sweep` call does not automatically reproduce every `simulation_hints` protocol field.

## Original Calado Comparison

`calado2016_fig1f.yaml` retains the Legacy toy-device comparison. The strong contact-volume SRH belongs to that model. [plot_calado_fig1f.py](../scripts/plot_calado_fig1f.py) implements the paper-comparison voltage/hold history; [plot_calado_fig1f_scan_rate.py](../scripts/plot_calado_fig1f_scan_rate.py) implements the rate study.

The existing comparison is partial because the forward-scan collapse remains too shallow. See the [package README](../README.md) and [reproducibility registry](../reproducibility/README.md) for recorded metrics and their scope. Do not transfer those metrics to the newer Full-mode internal figure preset.

## Current Checks

From `perovskite-sim/`:

```bash
python -m pytest -q tests/reproducibility/test_research_presets.py tests/unit/backend/test_scaps_inline_config.py tests/unit/experiments/test_plot_calado_fig1f.py
```

These checks cover the exact three-file inventory, loading, API exposure, inline-device semantics, configuration hashes, and Calado protocol helpers. They do not certify full J-V trends or external parity.

New shipped studies require an explicit input, mode, protocol, catalog decision, and acceptance criteria. New test-only devices belong under the fixture directory. Historical numerical expectations remain attached to their original inputs.

# Research Presets

The two research presets below are the ones the frontend catalogue exposes
(`frontend/src/preset-catalog.ts`). The other 50 bundled YAML files, including
the `twod/` and tandem examples, stay in this directory as inputs for the
regression suite and the 52-preset reproducibility matrix; the API lists them,
the frontend does not. They were removed on 2026-09-05 and restored on
2026-09-07 because roughly 200 test modules load them.

| Frontend name | Configuration | Reference mode | Study |
| --- | --- | --- | --- |
| SCAPS parity - Reference v2 | [scaps_mirror_v2.yaml](scaps_mirror_v2.yaml) | Fast | SCAPS partner-device comparison: band offsets, doping, thickness and recombination; no mobile ions. |
| Calado 2016 - Ion hysteresis | [calado2016_fig1f.yaml](calado2016_fig1f.yaml) | Legacy | Ionic redistribution, contact SRH and scan-rate-dependent hysteresis for the Fig. 1f toy device. |

## SCAPS Parity

Start from `scaps_mirror_v2.yaml`. Keep the optical and contact assumptions explicit when
comparing with SCAPS. This is a comparison configuration, not a claim that
every external parity target has passed.

Relevant drivers: [run_scaps_full_regression.py](../scripts/run_scaps_full_regression.py)
and [run_interface_cbo_scan.py](../scripts/run_interface_cbo_scan.py).
Use their declared protocols and evidence gates for quantitative comparisons.

## Calado 2016

The active preset is `calado2016_fig1f.yaml`, with absorber
`D_ion = 2.585e-18 m^2/s` and `P0 = 1e25 m^-3`, displayed as `D_c` and `c0`.
The strong contact-volume SRH is part of the paper's toy model.

For an initial frontend trend study, select the transient solver, set the scan
rate to `0.04 V/s` and the upper bias to `1.2 V`, then vary one parameter at a
time: `D_c`, `c0`, scan rate or contact SRH. A zero-diffusivity control freezes
the ions; zero initial concentration removes that species' initial inventory.
Increase spatial and voltage resolution to check that a trend is numerical-
resolution independent.

The ordinary frontend sweep goes from `0` to `V_max` and back. The paper
comparison uses uniform generation and a `-1 -> +1.2 V`, `3 s` hold, return
protocol, implemented by [plot_calado_fig1f.py](../scripts/plot_calado_fig1f.py).
The rate study is [plot_calado_fig1f_scan_rate.py](../scripts/plot_calado_fig1f_scan_rate.py).
The existing comparison is partial: the forward-scan collapse is still too
shallow. See the [main README](../README.md) for the recorded results.

## Current Checks

From `perovskite-sim/`:

```bash
python -m pytest -q tests/reproducibility/test_research_presets.py tests/unit/backend/test_scaps_inline_config.py tests/unit/experiments/test_plot_calado_fig1f.py
```

These checks cover the exact two-file inventory, loading, API exposure,
inline-device semantics and the Calado protocol helpers. They do not certify
full J-V trends or external parity.

The 52-preset matrix, research scripts and tests keep their original inputs.
Do not repoint them at these two presets while reusing their old numerical
expectations; new studies should define new inputs and acceptance criteria.

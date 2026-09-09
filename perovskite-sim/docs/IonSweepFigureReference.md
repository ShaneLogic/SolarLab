# Ion Sweep Figure Reference

`configs/calado2016_ion_sweep.yaml` is the frontend entry point for the
SolarLab dense ion-sweep figure. It preserves the captured device inputs and
continuous waveform from `DenseIonMatrixV1` (2026-09-07). This is an internal
finite-time reference, not quantitative reproduction of the Calado paper.

Select **Calado 2016 - Ion hysteresis** in Device settings. This is the single
Calado model entry in the frontend; it includes the verified ion-sweep protocol.
The historical `calado2016_fig1f.yaml` remains available to regression scripts.
The J-V pane
loads `simulation_hints.jv_sweep` when this preset or device is selected.
Changing a layer field or a scan setting afterward does not reapply defaults.

## Fixed Protocol

- Full mode, 300 K, manual compatibility V_bi = 1.3 V.
- All four outer contact S values are null: all-carrier Dirichlet contacts.
- N_grid = 60 requested, 61 actual nodes; 111 voltage samples per branch.
- Continuous -1 -> +1.2 -> -1 V ramps at 0.04 V/s.
- Dark 0 V seed: 120 s; dark -1 V preparation: 30 s.
- Illuminated dwell at each branch start: 0.5 s.
- Dark +1.2 V turnaround hold: 3 s.
- Uniform absorber generation: 2.5e27 m^-3 s^-1.
- Radau rtol = 1e-4, density atol = 1 m^-3.

## Parameter Changes

For a diffusivity subcurve, change only the MAPbI3 absorber D_c (stored as
`D_ion`), with D_c = ratio * 2.585e-18 m^2/s. Keep c0 (`P0`) = 1e25 m^-3,
including the ratio-zero frozen-ion control. Transport-layer D_c and c0 stay
zero. Ratios in the figure are 0, 0.1, ..., 2.0.

The corresponding density study changes only absorber c0, with
c0 = ratio * 1e25 m^-3, while holding D_c at 2.585e-18 m^2/s. Changing both
parameters or the history defines a new experiment, not a subcurve of either
one-parameter figure.

Use **Operational range** to inspect the photovoltaic quadrant, or
**Full sweep** to inspect all negative-bias and injection samples. Both
branches determine the operational limits; axis selection never changes data.

## Reference Checks

Compare `HI_P = P_max,reverse / P_max,forward - 1`, not the separate
`hysteresis_index = (P_max,reverse - P_max,forward) / P_max,reverse` field.

| D / D_ref | Archived HI_P |
| --- | ---: |
| 0 | 2.873578e-7 |
| 0.2 | 96.203833 |
| 1 | 0.4794925 |
| 2 | 0.1986760 |

`tests/fixtures/IonDiffusivityFigureReference.json` preserves the original
request and four recorded curve pairs. The integration test compares voltage
samples, both total-current branches (rtol 1e-5, atol 1e-3 A/m^2), and HI_P
(rtol 2e-4, atol 1e-5). The first sample of each branch was omitted by the
archived stored-state current reconstruction. This is a same-grid regression,
not a full spatial or temporal convergence certificate.

```sh
python -m pytest tests/integration/test_ion_diffusivity_figure.py -m slow
```

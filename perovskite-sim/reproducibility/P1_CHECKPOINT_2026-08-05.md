# P1 Continuation Checkpoint (2026-08-05)

This checkpoint continues `P1_CHECKPOINT_2026-08-04.md`. It does not replace
or refresh the frozen P0 baseline. Machine-readable status remains
authoritative in `config_benchmark_matrix.yaml` and `p1_gaps.yaml`.

## Frozen Boundary

- Starting implementation commit: `c31b19d`.
- P0 base commit: `c23e5b9beb3c356250ea32dcb09c78dc45ba28ec`.
- Frozen patch SHA-256:
  `58166a458047984bf85ead3cc5c5b5e29b2c6dc22aa851682a8ca81ef314d82a`.
- P0 reconstruction still verifies all 15 frozen files. No P0 hash or patch
  changed.

## c-Si C-V Internal Closure

The restricted QF frequency-domain result is a distributed carrier-storage
capacitance, not a solver artifact or the whole-wafer geometric capacitance.
The new response certificate exposes the complex state and storage derivatives
for each frequency. The real parts of its electron and hole storage integrals
independently match terminal capacitance within `5e-6` relative; their
quadrature components remain below `5e-3` of capacitance.

An independent check solves two new certified dark DC states at
`V_dc = -0.2 V +/- 1e-4 V` and differentiates the total carrier inventories:

| Quantity at N=200, 100 kHz | Capacitance (F/m2) |
|---|---:|
| Terminal frequency-domain C | 2.886247453e-4 |
| q d integral(n) dx / dV from independent DC states | 2.886247542e-4 |
| q d integral(p) dx / dV from independent DC states | 2.886247548e-4 |

Both independent DC values agree with terminal C within `4e-8` relative. This
closes the internal charge identity without reusing the AC Jacobian.

The Mott-Schottky fit also had the wrong thermal interpretation. A p-n
junction has two thermally diffuse transition-region edges, so the
depletion-model relation is

```text
1/C^2 = 2 (V_bi - V - 2 kT/q) / (q epsilon N)
V_bi,app = -b/a + 2 kT/q
```

The previous `+kT/q` is the one-edge Schottky correction. The API field
`V_bi_fit` remains for compatibility, but documentation and UI now label it
as the apparent p-n depletion-model value `V_bi,app`.

## Registered Numerical Evidence

The registered protocol remains biases `-0.3` through `+0.2 V`, frequencies
10 kHz/100 kHz/1 MHz, and the N=200/300/400 weighted grid ladder.

| N_grid | C(-0.2 V, 100 kHz) (F/m2) | V_bi,app at 100 kHz (V) | N_eff (m-3) |
|---:|---:|---:|---:|
| 200 | 2.886247e-4 | 0.797515 | 9.4928e21 |
| 300 | 2.907073e-4 | 0.781101 | 9.4660e21 |
| 400 | 2.919108e-4 | 0.782066 | 9.5537e21 |

- Grid and frequency convergence, all-face admittance continuity, derivative
  step stability, and linear-solve backward-error gates remain unchanged and
  passing.
- On N=400, the fitted apparent values at 10 kHz/100 kHz/1 MHz are
  `0.782061/0.782066/0.782117 V`.
- The finest apparent intercept is `0.110831 V` below the independently
  configured `0.892896 V` contact-potential magnitude.
- Chang's exact p-n capacitance analysis identifies the differential electron
  or hole inventory as the junction capacitance and warns that the C-V
  intercept is not an accurate barrier estimator.
- Hufschmidt et al. use the p-n `2kT/q` correction and report a typical
  barrier-minus-intercept range of `0.1-0.4 V`. The SolarLab gap is inside that
  range.

The frozen interpretation resource is
`perovskite_sim/data/references/csi_pn_cv_intercept.yaml`, SHA-256
`14e38316eea8c65e1af81e578198f3c07dbacfb86ef239157840d519bf6320d7`.
It records the DOI, source archive, device geometry, measurement protocol,
fit window, extracted values, and limitations. The source archive SHA-256 is
`9a663c632f4f36f38f8e70bfa6b9f67172b294d9fe3b3b842947eb26ac1cd985`.

## Decision Boundary

The internal c-Si C-V interpretation is resolved for the restricted local,
ion-free QF frequency-domain path:

1. The AC terminal response equals independently differentiated electron and
   hole inventories.
2. The p-n thermal correction is now `2kT/q`.
3. The remaining apparent-intercept shift has a published distributed-carrier
   explanation and lies inside the reported range.

`csi-mott-schottky-convergence` remains **open** only because no external c-Si
device with compatible doping, geometry, temperature, frequency, amplitude,
and machine-readable pointwise C-V data is frozen. The Hufschmidt reference
supports the interpretation and provenance, not curve-level agreement with
`configs/cSi_homojunction.yaml`.

The default endpoint-sampled transient impedance path remains unrepaired and
uncertified. Mobile ions, selective contacts, thermionic interfaces, and
non-local photon recycling remain outside this QF small-signal capability.

## Reproducibility Matrix

- Added one content-addressed c-Si p-n C-V interpretation resource.
- The resource verifier now discovers YAML references under
  `perovskite_sim/data/references/` so future sources cannot bypass matrix
  coverage.
- The c-Si QF frequency-domain benchmark now includes eight explicit nodes,
  including the independent DC inventory identity and published intercept
  interpretation.
- Verifier coverage is 28 configs, 19 resources, 17 benchmark contracts, and
  3 schemas.
- Four P1 gaps remain open: Lin tandem current matching, default-driver c-Si
  J-V convergence, compatible external c-Si C-V curve validation, and exact
  external-solver curve provenance.

## Verification

- P0 reconstruction and reproducibility verifier passed; all 15 frozen files
  reconstruct, and every config/resource hash is current.
- Affected small-signal, impedance, Mott-Schottky, and matrix tests:
  `62 passed in 4.76s`.
- c-Si frequency-domain C-V slow regression: `8 passed in 27.20s`.
- Complete default Python suite:
  `1524 passed, 2 skipped, 247 deselected, 12 warnings in 152.39s`.
- Complete frontend suite: `27 files, 377 tests passed`.
- Frontend production build and Python `compileall` passed.

The complete historical slow suite was not repeated. The affected c-Si
frequency-domain slow lane was run explicitly.

## Next Stage

Freeze or digitize a compatible independent c-Si p-n C-V dataset and compare
pointwise capacitance, slope-derived doping, and apparent intercept without
calibrating the SolarLab config to the curve. Record area/edge corrections,
temperature, frequency, AC amplitude, DC sweep direction, doping, geometry,
units, extraction method, source revision, and immutable hashes.

Do not generalize the frequency-domain operator to ions or other model paths
until that external comparison is complete. After the c-Si external gate, the
remaining P1 numerical priority is the default-driver c-Si J-V runtime and
grid-convergence envelope.

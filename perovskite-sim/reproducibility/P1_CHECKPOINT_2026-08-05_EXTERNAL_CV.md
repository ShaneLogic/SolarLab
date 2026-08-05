# P1 External c-Si C-V Checkpoint (2026-08-05)

This checkpoint continues `P1_CHECKPOINT_2026-08-05.md`. It does not alter the
frozen P0 baseline. Machine-readable status remains authoritative in
`config_benchmark_matrix.yaml` and `p1_gaps.yaml`.

## External Source

The selected source is van Nijen et al., *The nature of silicon PN junction
impedance at high frequency*, Solar Energy Materials and Solar Cells 282
(2025) 113383:

- Article DOI: `https://doi.org/10.1016/j.solmat.2024.113383`.
- Dataset version DOI:
  `https://doi.org/10.4121/19445ed8-c8d5-4204-a8c6-adaa7a55ece3.v1`.
- Dataset license: CC BY 4.0.
- Published archive MD5: `44fbe7b6d50b4c8b3d6d2e8b5ecc4c94`.
- Independently verified archive SHA-256:
  `478826a79faad46d705344e34e19a6e18bca1c7b3b8989a5d105baa874d0c67b`.

The source is an independent 2-D Sentaurus calculation, not an experiment. It
uses a dark 200 um by 500 um p+/n silicon device at 298.15 K, 50 um contacts,
uniform donor density `1e21 m-3`, a front Gaussian acceptor peak of `1e25 m-3`,
a 10 um junction depth, and a 10 ms SRH lifetime. The public archive includes
raw admittance data and extraction code but not the proprietary Sentaurus input
deck or exact solver version.

## Frozen Extraction

The reference resource
`perovskite_sim/data/references/csi_vannijen2025_pn_cv.yaml` freezes all eight
0.0-0.7 V rows at 101996.13 Hz, each source CSV SHA-256, and the author-code
extraction:

```text
Z = 1 / (G + iB)
Rs = min(real(Z)) - 0.001 ohm cm2
Cj = imag(1 / (Z - Rs)) / (2 pi f)
```

Only 0.0/0.1/0.2 V are used for comparison because the article identifies
that interval as depletion dominated. The source capacitances are
`6.49537e-5`, `6.55253e-5`, and `6.89984e-5 F/m2`. No SolarLab parameter is
fitted to these values.

## Local Model Extension

`configs/csi_vannijen2025_pn_cv.yaml` maps the published dopant profile through
a new default-off Gaussian profile:

```text
N(d) = N_bulk + (N_edge - N_bulk) exp(-(d/L)^2)
L = 5 um / sqrt(ln(10))
```

This gives `N_A(5 um)=1e24 m-3` and `N_A(10 um)=N_D=1e21 m-3` exactly. Uniform
configs retain their previous material arrays and semantic hashes. The profile
is used consistently by material assembly, contact equilibrium,
`compute_V_bi()`, interface tunnelling doping, and Debye diagnostics. Loader,
backend round-trip, schema, and frontend types carry the optional fields.
The numerical layer partition truncates the Gaussian tail at 15 um, where its
acceptor density is only `1e-5` of the donor density.

## Registered Result

SolarLab uses its residual-certified 1-D QF frequency-domain operator with a
10 mV nominal perturbation. The registered grid ladder is N=400/600/800.

| Bias (V) | Source C (F/m2) | Local N=800 C (F/m2) | Relative difference |
|---:|---:|---:|---:|
| 0.0 | 6.495369e-5 | 7.776555e-5 | 19.72% |
| 0.1 | 6.552533e-5 | 8.424247e-5 | 28.56% |
| 0.2 | 6.899838e-5 | 9.355836e-5 | 35.60% |

The maximum curve change contracts from `2.347e-4` on N=400/600 to
`8.137e-5` on N=600/800. Every retained point has a certified dark DC state;
the worst normalized residual is below `2.8e-9`, continuity and DC current
spread remain below `3.3e-8 A/m2`, AC all-face relative spread remains below
`6.0e-5`, and complex-solve backward error remains below `6.3e-16`.

## Decision Boundary

This is material progress but not closure of
`csi-mott-schottky-convergence`. The local result is numerically converged and
has the correct monotonic depletion trend, yet it is not pointwise equal to the
source. The known structural mismatch is source 2-D partial-contact geometry
versus SolarLab 1-D full-area geometry; the absent source input deck prevents a
strict attribution of the remaining 19.7-35.6 percent difference.
The paper-unspecified Sentaurus transport and recombination settings are also
not guaranteed to match SolarLab's documented c-Si assumptions.

The benchmark is therefore registered as `partial_external_comparison`, the
config remains `partial`, and the P1 gap remains open. A strict next experiment
must either reproduce the 2-D contact geometry with a certified small-signal
operator or use a complete compatible 1-D/full-area external reference.

## Verification

- P0 reconstruction and reproducibility verifier passed: 29 configs,
  20 resources, 18 benchmark contracts, and 3 schemas.
- Focused profile/schema/device/grid tests: `69 passed`.
- Existing c-Si internal frequency-domain C-V lane:
  `8 passed in 27.66s`.
- New van Nijen external-comparison lane: `5 passed in 27.78s`.
- Complete default Python suite:
  `1533 passed, 2 skipped, 252 deselected, 12 warnings in 153.72s`.
- Complete frontend suite: `27 files, 378 tests passed`.
- Python `compileall` and frontend production build passed. The build retained
  the existing large-chunk advisory.

The complete historical slow suite was not repeated. Both affected c-Si slow
lanes were run explicitly.

After this bounded external C-V stage, the next independent P1 numerical lane
is the default-driver c-Si J-V runtime and N=200/300/400 convergence envelope.

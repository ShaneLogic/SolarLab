# Reproducibility & Reference Contracts

This directory holds the **7 frozen machine-readable contracts and reference datasets** used by SolarLab's analytical gates, external benchmark harnesses (`scripts/benchmarks/`), and release-bundle validators (`solarlab.reproducibility.release`).

> [!IMPORTANT]
> Engineering verification, external solver alignment, and formal scientific qualification are tracked separately (`scientific_qualification = False` unless every declared continuum and external reference gate is independently satisfied). Do not relax frozen thresholds or overwrite contract digests without a versioned contract bump.

## Frozen Contracts & Datasets

| File | Role | Primary Consumers |
|:-----|:-----|:------------------|
| [AnalyticGatesV1.json](AnalyticGatesV1.json) | Closed-form analytical tolerances for electrostatics, Bernoulli/SG fluxes, Beer–Lambert/TMM photon conservation, and AC admittance | `tests/unit/`, `tests/refactor/test_release_bundle.py` |
| [ScapsComparisonContractV1.json](ScapsComparisonContractV1.json) | 262-case SCAPS-1D single- and two-parameter sweep definitions, coordinate grids, and reference comparison schema | `scripts/benchmarks/scaps_native_pvk_xy.py`, `scripts/benchmarks/scaps_reference.py`, `scripts/benchmarks/p09_candidate_validation.py` |
| [ScapsParams.xlsx](ScapsParams.xlsx) | Original read-only 13-sheet SCAPS-1D partner parameter & reference metric workbook | `scripts/benchmarks/scaps_reference.py` |
| [ScapsAm15g1Sun.spe](ScapsAm15g1Sun.spe) | Native SCAPS 3.3.10 120-bin AM1.5G 1-sun incident power table (`305–4045 nm`, `1000 W/m²`) | `scripts/benchmarks/scaps_native_pvk_xy.py` |
| [HysteresisReferenceContractV2.json](HysteresisReferenceContractV2.json) | 7-stage dynamic mobile-ion hysteresis protocol, 4-axis independent refinement hierarchy, and `1/3` reference-error / `bounded_failure` rules | `scripts/benchmarks/hysteresis_reference.py`, `tests/refactor/test_hysteresis_reference.py` |
| [PerformanceCasesV1.json](PerformanceCasesV1.json) | Locked 1D/2D runtime & scaling benchmark cases (`TransientJvN80`, `TransientJvN200`, 2D sparse scaling) | `scripts/benchmarks/p09_candidate_validation.py`, `scripts/benchmarks/runtime_timing.py` |
| [RefactorReferenceManifestV1.json](RefactorReferenceManifestV1.json) | Cryptographic provenance manifest pinning historical reference inputs and archival digests | `src/solarlab/reproducibility/`, `tests/refactor/` |

## Benchmark & Qualification Drivers (`../scripts/benchmarks/`)

| Script | Purpose |
|:-------|:--------|
| [../scripts/benchmarks/scaps_native_pvk_xy.py](../scripts/benchmarks/scaps_native_pvk_xy.py) | Native zero-fit SCAPS-1D 262-case driver (`scaps_pvk_xy.yaml` + `ScapsAm15g1Sun.spe`, `89.1%` all-4-gate pass, median $\lvert\Delta V_{\mathrm{oc}}\rvert = 0.60\text{ mV}$) |
| [../scripts/benchmarks/p09_candidate_validation.py](../scripts/benchmarks/p09_candidate_validation.py) | Isolated-wheel candidate validation across historical `scaps_mirror_v2`, Hysteresis 45, and performance cases |
| [../scripts/benchmarks/scaps_reference.py](../scripts/benchmarks/scaps_reference.py) | Standalone historical SCAPS-1D (`scaps_mirror_v2`) 262-case suite runner & scorecard generator |
| [../scripts/benchmarks/hysteresis_reference.py](../scripts/benchmarks/hysteresis_reference.py) | Decoupled 7-stage dynamic ion-hysteresis reference & 4-axis refinement driver |
| [../scripts/benchmarks/hysteresis_current_bounds.py](../scripts/benchmarks/hysteresis_current_bounds.py) | Exact rational/interval terminal current & Gauss enclosure verifier on saved trajectories |
| [../scripts/benchmarks/hysteresis_mobile_bounds.py](../scripts/benchmarks/hysteresis_mobile_bounds.py) | Semidiscrete mobile-ion inventory & flux bound verifier |
| [../scripts/benchmarks/runtime_timing.py](../scripts/benchmarks/runtime_timing.py) | Single-thread BLAS timing helper for `PerformanceCasesV1.json` |

# Physical Trend Validation Harness — Design Spec

**Date:** 2026-05-12
**Status:** approved for implementation
**Scope:** new test file, zero production-code changes

## Purpose

Validate that the perovskite-sim drift-diffusion solver reproduces six well-established physical scaling laws from semiconductor device physics. These are structural correctness checks — they test whether the physics is internally consistent, not whether it matches a specific experimental device.

## Scope boundaries

- **In scope:** six trend tests in one file; each runs a parameter sweep, fits a trend line, and asserts the slope/intercept falls within a literature window
- **Out of scope:** matching experimental data (hero-device configs, statistical DB), modifying production code, new YAML presets, backend/frontend changes

## Architecture

```
tests/validation/
└── test_physical_trends.py    # NEW — six parametrized trend tests
```

- Tagged `@pytest.mark.validation` — excluded from default `pytest` and `pytest -m slow`; invoked explicitly with `pytest -m validation`
- Each test builds its parameter sweep via `dataclasses.replace()` on a baseline preset — no new YAML files required
- All use `nip_MAPbI3_tmm.yaml` as baseline (TMM optics active, FULL physics tier)
- Zero production-code changes

### Shared test utilities

Each test follows the same skeleton:
1. Clone the baseline preset
2. Vary one parameter across N steps (5-7)
3. Run the relevant experiment at each step
4. Extract the trend metric (slope, intercept, ratio)
5. Assert the metric falls within the literature window

A shared helper `_vary_param(stack, field_path, values)` applies a sequence of `dataclasses.replace()` calls — field_path is a dotted string like `"layers[1].Eg"` or `"layers[1].mu_n"`.

## Trend definitions

### 1. V<sub>oc</sub> vs Bandgap

- **Sweep:** absorber Eg: 1.2, 1.4, 1.6, 1.8, 2.0, 2.2 eV (6 steps)
- **Experiment:** J-V sweep (illuminated, forward scan)
- **Metric:** V<sub>oc</sub> loss ΔV = Eg/q − V<sub>oc</sub> at each Eg
- **Assertion:** 0.25 ≤ median(ΔV) ≤ 0.55 V, and |slope(ΔV vs Eg)| ≤ 0.15 V/eV
- **Physics rationale:** in a physically correct simulator, the non-radiative loss is roughly constant with bandgap; V<sub>oc</sub> tracks Eg with slope ≈ 1

### 2. V<sub>oc</sub> vs Thickness

- **Sweep:** absorber thickness: 100, 200, 400, 700, 1000 nm (5 steps)
- **Experiment:** J-V sweep (illuminated, forward scan)
- **Metric:** dV_oc / d(log₁₀(thickness))
- **Assertion:** slope ∈ [30, 90] mV/decade
- **Physics rationale:** thicker absorbers dilute contact recombination; the log-thickness dependence is the SRH bulk-recombination signature

### 3. FF vs Mobility

- **Sweep:** μ_n = μ_p: 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 cm²/Vs → converted to m²/Vs (5 steps, log-spaced)
- **Experiment:** J-V sweep (illuminated, forward scan)
- **Metric:** FF(lowest μ) vs FF(highest μ)
- **Assertion:** FF(μ=1e-6) ≤ FF(μ=1e-2) − 0.05 (absolute FF drop ≥ 5 percentage points)
- **Physics rationale:** below ~1e-4 cm²/Vs transport resistance should measurably degrade FF for a 500 nm absorber

### 4. Ideality Factor

- **Sweep:** none — single dark J-V sweep on the baseline preset
- **Experiment:** dark J-V sweep from V=0 to V=V<sub>oc</sub> (estimated from illuminated V<sub>oc</sub>)
- **Metric:** n_id = (q/kT) × dV / d(ln J) in the low-injection region (J < J<sub>sc</sub>/100)
- **Assertion:** 1.0 ≤ n_id ≤ 2.0
- **Physics rationale:** single-junction recombination ideality falls between 1 (SRH mid-gap) and 2 (radiative or high-injection)

### 5. J<sub>sc</sub> vs Bandgap

- **Sweep:** same Eg sweep as Trend 1 (1.2–2.2 eV, 6 steps)
- **Experiment:** illuminated J-V sweep → extract J<sub>sc</sub>
- **Metric:** monotonicity and magnitude
- **Assertion:** J<sub>sc</sub>(Eg=2.2) < J<sub>sc</sub>(Eg=1.2), and the ratio J<sub>sc</sub>(2.2)/J<sub>sc</sub>(1.2) ≤ 0.7
- **Physics rationale:** wider bandgap → fewer above-gap photons → lower photocurrent; the ratio follows the AM1.5G-integrated photon flux ratio

### 6. V<sub>oc</sub> vs Illumination

- **Sweep:** Suns-V<sub>oc</sub> from 1e-3 to 1 sun, 6 log-spaced points
- **Experiment:** `run_suns_voc` on baseline preset
- **Metric:** slope of V<sub>oc</sub> vs ln(Φ) via linear regression
- **Assertion:** slope ∈ [25, 65] mV/decade (covers n_id from 1.0 to 2.5)
- **Physics rationale:** V<sub>oc</sub> = (n_id · kT/q) · ln(Φ/Φ_0); the slope directly measures the dominant recombination ideality

## Parametrization strategy

Each trend test is `@pytest.mark.parametrize` over the sweep values. The baseline stack is loaded once via a module-scoped fixture (or `@pytest.fixture(scope="module")`). Each parametrized case `dataclasses.replace()`s the relevant field, runs the experiment, and returns the metric.

Tests 1 and 5 share the same Eg sweep — run the J-V once and extract both V<sub>oc</sub> and J<sub>sc</sub> from each result to avoid duplicate computation.

## Test execution

```bash
# Run all validation tests
pytest -m validation

# Run a single trend
pytest -m validation -k "test_voc_vs_bandgap"
```

Expected runtime: ~5-10 minutes for all six trends (dominated by the Radau transient settles). These are longer than unit tests but shorter than the slow TMM regression suite.

## Failure interpretation

| Trend | Failure | Likely culprit |
|-------|---------|---------------|
| V<sub>oc</sub> vs Eg | ΔV grows with Eg | band offsets misapplied; chi/Eg not propagating correctly to V<sub>bi</sub> |
| V<sub>oc</sub> vs thickness | slope ≤ 0 or too steep | contact recombination not modelled; or SRH lifetime too short |
| FF vs mobility | FF doesn't drop | field-dependent mobility hook not engaging; or test μ range too narrow |
| Ideality | n_id < 1 or n_id > 2 | recombination model missing; or measurement in wrong injection regime |
| J<sub>sc</sub> vs Eg | J<sub>sc</sub> flat or inverted | TMM generation not responding to Eg-dependent n,k; or absorbing in transport layers |
| V<sub>oc</sub> vs illum | slope outside [25,65] | Suns-V<sub>oc</sub> baseline subtraction wrong; or dominant recombination mechanism changing with Φ |

## Out of scope (explicitly deferred)

- Matching specific experimental devices (hero-device configs) — separate initiative
- Statistical database comparison — separate initiative
- Any new YAML preset files
- Modifications to production code (`perovskite_sim/`)
- Backend or frontend changes
- CI integration (add to CI pipeline as follow-up after manual validation)

## Dependencies

- Existing experiments: `run_jv_sweep`, `run_suns_voc` (1D)
- Existing preset: `nip_MAPbI3_tmm.yaml` (TMM optics)
- `dataclasses.replace()` for parameter sweeping
- No new Python dependencies beyond what's in `pyproject.toml`

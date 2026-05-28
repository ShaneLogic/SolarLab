# SolarLab Validation Evidence

Date: 2026-05-19

Repository state:

- Repository: SolarLab
- Branch: `main`
- Commit: `43c81d7fefd009ffca598f58f85a40ad4e661e1e`
- Working tree note: the validation was run with documentation-only untracked
  files present; no simulator source files were modified for the manual.

Environment:

- Python: `3.13.5`
- NumPy: `2.1.3`
- SciPy: `1.15.3`
- Pytest: `8.3.4`
- Node.js: `v24.13.0`
- npm: `11.6.2`
- Vite: `8.0.8`
- TeX PDF engine installed for manual generation: `Tectonic 0.16.9`

Validation gates:

| Gate | Command | Result | Time | Notes |
|---|---|---:|---:|---|
| Python default suite | `pytest` | 647 passed, 1 skipped, 72 deselected | 704.90 s | 12 deprecation warnings |
| Python slow suite | `pytest -m slow` | 72 passed, 648 deselected | 2156.96 s | 4 deprecation warnings |
| Python validation suite | `pytest -m validation` | 18 passed, 702 deselected | 217.65 s | no failures |
| Frontend production build | `npm run build` | passed | 0.25 s build step | Vite chunk-size warning only |
| Frontend unit tests | `npm run test:run` | 22 files passed, 320 tests passed | 1.29 s | no failures |

Warnings observed:

- `pytest_asyncio` reported that `asyncio_default_fixture_loop_scope` is unset.
  This is a future-default warning, not a simulator failure.
- Some tests call `np.trapz`, which NumPy now deprecates in favor of
  `numpy.trapezoid` or SciPy integration helpers.
- The frontend build reported that one bundled JavaScript chunk exceeds
  500 kB after minification. This is a packaging/performance warning, not a
  correctness failure.

Evidence interpretation:

- The core Python solver, backend APIs, model schemas, 1D and 2D numerical
  physics tests, validation trend checks, and regression envelopes all passed.
- The frontend TypeScript build and rendering/unit tests passed.
- The validation evidence supports a public technical manual if the manual
  states the remaining limitations: deprecation warnings, 2D frozen-ion scope,
  TMM optical-data requirements, Mott-Schottky caveats, and placeholder/stub
  optical data where documented by the code.

Regression and validation envelopes represented by the passing suites include:

- IonMonger benchmark envelope and heterostack J-V behavior.
- Driftfusion-inspired flat-band and hysteresis behavior.
- TMM Jsc baselines for n-i-p and p-i-n MAPbI3 presets.
- Photon-recycling Voc boost envelope.
- Dark equilibrium charge neutrality and transient ion conservation.
- Physical trends for bandgap, thickness, mobility, dark ideality, and
  Suns-Voc slope.
- 1D/2D lateral-uniform parity and 2D microstructure regression.
- Backend SSE job dispatch and frontend result rendering.

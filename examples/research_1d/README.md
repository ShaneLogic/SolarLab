# New-core 1D research example

This example runs ion-free, classical one-dimensional DC with the actual new
compiler and steady solver. The explicit device/catalog is `device.json`.
It supports Maxwell–Boltzmann carriers, fully ionized dopants, DOS-aware
transport, semiconductor work-function contacts, bulk lifetime SRH/radiative/
Auger recombination and the compiler's bound generation profiles. Unsupported
configured physics is rejected; no legacy engine or fallback is used.

This is a research workflow, not final scientific or release qualification.
Mesh and voltage-grid convergence, full spectral/TMM optics, interface/defect
extensions, ions, transient/hysteresis, 2D and the full registry release remain
outside this example's verified scope. DC voltage continuation is not a
time-dependent voltage scan. Every parameter point is resolved and compiled
cold; states and factorizations are not reused across parameter changes.

## Environment and commands

Run from the repository root. `pyproject.toml` and `uv.lock` provide the pinned
environment; with uv installed, recreate it using:

```sh
uv sync --frozen --extra dev --python 3.13
```

The delivery's existing environment is CPython **3.13.15**, NumPy **2.5.3**,
SciPy **1.18.1**, Pydantic **2.13.5**, and threadpoolctl **3.6.0**. The commands
below use that existing `.venv`; a fresh uv synchronization was not claimed.
Each run saves the actual Python executable digest, installed dependency
versions, loaded numerical-library digests, thread settings, source-file
digests and lock/configuration digests. Uncommitted source bytes are included;
a repository HEAD label is not used as their substitute.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1

# Start with the real dark equilibrium and illuminated DC example.
.venv/bin/python -B examples/research_1d/steady.py

# Two electron-lifetime points, using the existing PreparedSweep runner.
.venv/bin/python -B examples/research_1d/jv_scan.py \
  --store "$HOME/.local/share/solarlab/research_1d/lifetime-demo" \
  --study-id lifetime --parameter tau_n --values "1 us" "2 us"

# Independent read: loads only committed results, with no solver invocation.
.venv/bin/python -I -B examples/research_1d/read_results.py \
  --store "$HOME/.local/share/solarlab/research_1d/lifetime-demo" \
  --study-id lifetime
```

The store path must be fresh, local and outside cloud synchronization; it
contains the active SQLite/WAL and array artifacts. An existing store is
refused rather than overwritten. Choose another store for another calculation;
use the reader for existing results. `--include-arrays` includes decoded arrays
in the reader's JSON, and `read_study(Path(...), "lifetime")` returns NumPy arrays
and all retained attempts to Python callers. The original failed attempts are
not hidden by a later attempt. This example has no automatic retries, crash
resume or process-supervisor integration.

The default uses 12 mean intervals per layer (37 nodes for this device), 51
voltage samples from 0 to 1.25 V at 0.025 V steps, and unchanged
`SteadyOptions` thresholds.
This default is a fast workflow example. Independent refinement found its Jsc
differs from the 145-node result by 1.27e-4 relative, above the final 1e-4 target.
The 73/145-node difference is 1.10e-5, but continuum-error qualification remains
unresolved because the 289-node cold start failed. The default interpolated
Voc is also a voltage-grid estimate; use direct bracketing for precision work.
The independent base-device probe found opposite output-current signs at
1.20 and 1.225 V. The saved sweep must demonstrate its own bracket; changing
parameters can move Voc outside this range. `--voltage-max` and
`--voltage-points` make grid changes explicit; the example bounds declarations
to eight parameter points and 101 voltage samples. Finer research grids require
their own convergence assessment.

`tau_n` changes only the absorber electron lifetime; `tau_p` is retained.
Select `tau_p`, `mu_n` or `mu_p` with `--parameter` and supply explicit values
for mobility. Unqualified numeric lifetimes are seconds and mobilities are
m²/(V·s); quoted unit strings go through the existing configuration resolver.
Use `--layer` to choose a stable layer ID. No partner parameter is silently
changed. `--incident-power` supplies an explicit W/m² denominator; omitted
incident power leaves efficiency null rather than assuming a solar spectrum.

## Saved state and interpretation

The existing `run_prepared_steady_sweep` calls `compile_device_1d` and
`run_steady_jv` for each point. Its returned compiled identity is recorded
directly, without a second compilation. `publish_steady_jv` writes eight
lossless arrays, Newton/initialization evidence and result metadata; the reader
uses `read_steady_jv` plus the exact attempt/terminal bindings and original
numerical-array digests. Array roles, units, grid shapes and artifact integrity
are checked before exposing values.

| Array or metric | Units / interpretation |
| --- | --- |
| `nodes_m` | m; spatial nodes |
| `voltage_V` | V; requested steady operating points |
| `current_density_A_m2` | A/m²; every interior face |
| `contact_current_density_A_m2` | A/m²; exterior left/right contacts |
| `output_current_A_m2` | A/m²; `-voltage_polarity * left_face_current` |
| `potential_V` | V; one node vector per voltage |
| `electron_density_m3`, `hole_density_m3` | m⁻³; one node vector per voltage |
| Jsc, Voc, maximum power | A/m², V, W/m² |
| Fill factor, efficiency | fractions, not percentages |

`succeeded` means the completed curve was persisted after the production
solver's gates; it is not final scientific acceptance. Metrics are
piecewise-linear grid estimates. A missing Voc bracket stays null, together
with the limitation reason, and is reported as `metrics_status=unavailable`.
No failed point, missing bracket or absent irradiance is converted to zero.
The CLI returns 2 for failed/incomplete points or unavailable metrics; its JSON
distinguishes these states. All scientific-qualification flags remain false.

The recorded 2026-10-28 run completed both default curves in 3.91 s on the
environment above. These are grid estimates for this device, not reference
acceptance thresholds:

| Absorber electron lifetime | Jsc (A/m²) | Voc (V) | Maximum power (W/m²) | Fill factor |
| --- | ---: | ---: | ---: | ---: |
| 1 µs | 144.588496 | 1.214658 | 149.510782 | 0.851304 |
| 2 µs | 144.588881 | 1.235411 | 151.290044 | 0.846962 |

An earlier 0.05 V grid ending at 1.3 V exhausted the unchanged iteration
budget; its two failed attempts were retained separately. No solver threshold
was relaxed to obtain the default result.

RunStore attempts represent publication of the returned calculation outcomes.
The actual whole-sweep calculation start/end and elapsed time are separate
metadata; publication claim time is not presented as the solve start.
Research-scoped H_input binds the supplied point and resolved/default/resource
identities, H_physics binds its actual compiled result and generation scale,
and H_execution binds protocol, controls, source and runtime identities.
Failed curves explicitly have no completed compiled-result binding. These
identities are not presented as the pending formal registry-release contract.

## Adding a local model

`steady.py` contains `SaturatingRecombination`, an executable extension example:

```sh
.venv/bin/python -B examples/research_1d/steady.py --additional-recombination
```

It implements `evaluate(n, p) -> RecombinationRate` with a node-wise net pair
rate and analytic density derivatives, and binds coefficients through a stable
`model_id`. The prepared device is compiled with
`compile_device_1d(prepared, additional_recombination=model)`; solver and
equation assembly stay unchanged. The rate is
`B * (n*p - ni²) / (1 + (n+p)/n_sat)`. The flag is an example of the extension
interface, not a scientifically validated additional material mechanism, and
is not silently enabled in `jv_scan.py`.

Focused verification runs one genuine dual-point study, shares its saved
output across readback checks, checks a fresh reader with solver imports
forbidden, preserves failure/history cases and runs the extension command:

```sh
.venv/bin/python -B -m pytest -q tests/integration/test_research_1d_workflow.py
```

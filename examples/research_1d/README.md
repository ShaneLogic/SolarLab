# New-core 1D research example

This example runs ion-free, classical one-dimensional DC with the actual new
compiler and steady solver. The explicit device/catalog is `device.json`.
It supports Maxwell–Boltzmann carriers, fully ionized dopants, DOS-aware
transport, semiconductor work-function contacts, bulk lifetime SRH/radiative/
Auger recombination and the compiler's bound generation profiles. Unsupported
configured physics is rejected; no legacy engine or fallback is used.

This is a research workflow, not final scientific or release qualification.
Mesh and voltage-grid convergence, full spectral/TMM optics, interface/defect
extensions, ions, hysteresis, 2D and the full registry release remain
outside this example's verified scope. A separate fixed-bias generation
transient example is described below. DC voltage continuation is not a
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
below use that existing `.venv`. An independent fresh, offline wheel installation
of commit `d735482` ran the steady, sweep, reader and model-extension examples
outside checkout imports. Later source commit `67aabde` passed the enabled CI
jobs on Linux with CPython 3.11.16 and 3.13.15; those checks are separate from
the earlier installed artifact.
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
The 73/145-node difference is 1.10e-5. The original 289-node cold start failed;
the density initialization correction now passes that start under the same
physical gates, while continuum-error qualification remains unresolved.
The default interpolated Voc underestimates the base device's directly
bracketed result by about 1.289 mV, above the final 0.1 mV target.
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

## Direct open-circuit voltage

For a `compiled` device and the actual `curve` returned by `run_steady_jv`,
refine its solved current bracket with the new-core equations:

```python
from solarlab.experiments.jv.operating_points import solve_direct_voc

voc = solve_direct_voc(compiled, curve, voltage_tolerance_V=1e-6)
print(voc.voltage_V, voc.interval_V, voc.output_current_A_m2)
```

The reported voltage belongs to a solved state. The returned interval is at
most the requested width; the endpoint estimate uses the full width as its
voltage uncertainty control. This replaces the interpolation estimate for Voc
only. It does not certify mesh error, FF, PCE or an exactly zero numerical
current. Failed refinement attempts and accepted endpoints remain available
in the result or error. The API requires a unique actual current bracket and
retains the curve's solver controls.

## Explicit solver controls and process execution

`SteadyOptions()` retains the historical `historical_max_row_v1` stopping
policy. An independently reviewed internal alternative is explicit:

```python
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

options = SteadyOptions(convergence_policy="poisson_contact_balance_v1")
curve = run_steady_jv(compiled, voltages, options=options)
```

The alternative keeps every Poisson and fixed-contact limit and applies one
joint absolute carrier-balance budget. It preserves the residual weights,
Newton merit, work budgets and independent physical gates. Saved results name
the selected policy, and direct Voc checks that policy and recomputes the
physical gates. This internal stopping rule does not qualify mesh accuracy or
change final Jsc, Voc, FF, PCE or hysteresis acceptance targets.

The actual process worker is available in `solarlab.experiments.jv.worker`
through `SteadyJVProtocol`, `prepare_steady_jv_job`,
`steady_jv_worker_binding` and `read_steady_jv_job`. It requires an explicit
prepared device and committed implementation-source binding, then submits to
the existing shared `ProcessSupervisor`. The supervisor owns the attempt,
publishes the eight arrays and execution manifest, and commits one terminal
outcome. Invalid requests and cancellation use the existing runtime controls;
readback never starts a solver. The in-process sweep commands above retain
their documented execution mode.

For a runnable two-point study on that shared process runtime:

```sh
.venv/bin/python -B examples/research_1d/process_sweep.py run \
  --build-source-commit "$(git rev-parse HEAD)" \
  --store "$HOME/.local/share/solarlab/research_1d/process-demo" \
  --work "$HOME/.local/share/solarlab/research_1d/process-demo-work"
```

This uses electron lifetimes of 1 and 2 µs, each with 37 nodes and voltages
0, 0.05 and 0.1 V. Each point is prepared cold and owns a distinct execution
manifest. The existing supervisor admits one point at a time by default;
failed, cancelled and previous attempts remain readable. The JSON prints the
persisted `declaration` path. Read it later without running a solver:

```sh
.venv/bin/python -B examples/research_1d/process_sweep.py read \
  --store "$HOME/.local/share/solarlab/research_1d/process-demo" \
  --declaration /absolute/path/from/the/run-output.json
```

The Python APIs are `prepare_steady_sweep_jobs`, `SteadySweepJobs.submit` and
`read_steady_sweep_jobs` in `solarlab.sweeps.steady_worker`. The short example
does not bracket Voc; its missing metrics stay null. These process checks
verify execution and result integrity, not mesh convergence.

## Local HTTP service

The installed `steady_jv` adapter now uses the same new-core worker through
`/jobs`, `/experiments`, the stored-result endpoints and reconnectable SSE.
Install the server dependencies and start the explicit example from the
repository root, using fresh local store/work directories:

```sh
uv sync --frozen --extra dev --extra server --python 3.13
.venv/bin/python -B examples/research_1d/service.py \
  --build-source-commit "$(git rev-parse HEAD)" \
  --store "$HOME/.local/share/solarlab/research_1d/http-demo" \
  --work "$HOME/.local/share/solarlab/research_1d/http-demo-work" \
  --port 8000
```

The example listens on `127.0.0.1` and installs the explicit `research_1d`
device. `--build-source-commit` identifies actual Git source bytes; the worker
checks them against its installation.
In another terminal, submit the demonstrated short three-point calculation:

```python
import json
from urllib.request import Request, urlopen

base = "http://127.0.0.1:8000"
with urlopen(base + "/experiments") as response:
    headers = {
        "X-Solarlab-Registry": response.getheader("X-Solarlab-Registry"),
        "X-Solarlab-Job-Schema": response.getheader("X-Solarlab-Job-Schema"),
        "Content-Type": "application/json",
    }
job = {
    "schema_version": "solarlab.job-submission.v1",
    "run_id": "my-steady-run", "experiment_id": "steady_jv",
    "version": 1, "engine": "solarlab",
    "input": {
        "device_id": "research_1d", "voltage_V": [0.0, 0.05, 0.1],
        "intervals_per_layer": 12, "generation_scale": 1.0,
        "incident_power_W_m2": None,
    },
    "limits": {
        "wall_seconds": 45.0, "rss_bytes": 1610612736,
        "output_bytes": 100663296, "accepted_steps": None, "calls": None,
    },
}
request = Request(base + "/jobs", data=json.dumps(job).encode(), headers=headers)
with urlopen(request) as response:
    accepted = json.load(response)
print(accepted)
```

A 202 response means the job was accepted. Follow its `events_url`, or poll
`/runs/{run_id}/attempts/{attempt_id}`, until the actual terminal outcome.
`result_url` becomes readable after that outcome; each returned artifact has
its own `content_url`. The arrays are NPY data suitable for
`numpy.load(..., allow_pickle=False)`. Reconnecting SSE with the last event's
sequence in `Last-Event-ID` resumes after that event. Repeating an identical
submission with the same run ID returns the same attempt.

This short protocol does not bracket Voc: its result retains
`open_circuit_not_bracketed`. Use the direct Python research workflow above
for the documented full sweep and direct Voc refinement. Trusted application
code can install additional prepared devices using `SteadyPreparationContext`;
HTTP chooses a registered device and explicit numerical protocol.

## Browser research entry

With the service above listening on port 8000, start the frontend in another
terminal:

```sh
cd web
npm ci --no-audit --no-fund
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://127.0.0.1:5173/?view=steady-jv`. Select an installed device, choose
the voltage samples and generation scale, then click **Run steady DC**. The
default three-point protocol is the short calculation above. The page follows
the actual worker, reads eight committed arrays, and plots J–V, current density,
potential and carrier densities. Select a voltage to inspect its spatial fields.
Voc, fill factor and efficiency remain **Unknown** when unavailable.

**Cancel this attempt** targets the displayed calculation; its final state is
confirmed by the stored terminal event. **Reconnect / refresh run** resumes
reading the same attempt after a connection failure. Run identifiers and raw
events are under **Run diagnostics**. This first entry keeps its selected run
in page memory; preserve its identifier before reloading or closing the page.
Stored results remain available through the service readers.

Vite proxies `/steady-service/` to port 8000 and strips that prefix; `/api`
continues to serve the existing workspace. A deployed static build needs an
equivalent same-origin reverse proxy. This entry exposes steady DC only; the
Python frequency-domain example below is separate. An actual browser run
verified the four Plotly views and spatial selection against the real service;
this does not establish scientific qualification.

## Small-signal response

A separate new-core frequency-domain example linearizes a revalidated DC state:

```sh
.venv/bin/python -B -m examples.research_1d.impedance
```

It runs this device at 1 V DC and requests 0, 1 and 1000 Hz. Carrier contact
channels may be `fixed_reservoir` or finite-rate `robin`; blocking contacts
are explicitly rejected in this slice. A separate 37-node Robin case was
verified at 0, 1, 1000 and 1e6 Hz, including both-terminal DC derivatives and
the contact-cell displacement contribution. The same DC Jacobian and carrier
storage derivatives supply a complex sparse solve. Returned arrays include
potential, electric field, carrier-density perturbations, conduction and
displacement currents, and both terminal currents.

Phasors use `exp(+i*omega*t)`. The default 1 V phasor is a derivative
normalization, not a one-volt finite-amplitude experiment. Admittance density
is in S/m² and impedance-area product in ohm·m²; multiply the former by device
area, or divide the latter by area, for the corresponding total port quantity.
The result retains linear-residual and all-face Maxwell-current diagnostics.
It records the current source and supplied controls used to revalidate the DC
state; it does not reconstruct an unrecorded original DC execution history.
Frequency/mesh convergence and full scientific AC qualification remain open.

## Fixed-bias generation transient

Run the conservative backward-Euler example with the same thread controls:

```sh
.venv/bin/python -B -m examples.research_1d.transient \
  --output "$HOME/.local/share/solarlab/research_1d/transient.json" --refine
```

The output path must be new. This solves the 37-node device at 0 V under
generation scale 1, then applies the declared ramp
`g(t) = 1 + 0.01 * (t / 400 ns)²`. The optional refinement runs 4, 8 and 16
explicit steps with right-endpoint generation values. Every accepted step
solves the difference of physical carrier inventories, evaluated with `expm1`
to avoid subtracting nearly equal densities. It retains algebraic Poisson and
contact equations, and reports both conduction and displacement current.

`TransientProtocol`, `TransientOptions` and `solve_transient` in
`solarlab.experiments.transient` expose the same workflow. Controls and the
accepted initial DC state are explicit. This example selects the independently
reviewed `poisson_contact_balance_v1` policy with unchanged Poisson, contact
and aggregate carrier budgets. There are no hidden substeps or retries.

The JSON retains accepted states, the first failed attempt, physical inventory
changes, raw current and residual diagnostics, and charge-subtraction roundoff.
The command exits nonzero if its requested calculation is incomplete.
Independent checks accepted all 28 example steps; successive final inventory
differences decreased by about a factor of 2.07. This is bounded first-order
evidence, not a time-error bound or mesh qualification. Physical-certificate
status remains `not_evaluated`. Abrupt dark turn-on and very small time steps
have retained Newton/roundoff failures. Variable bias, ions and hysteresis
remain outside this first transient slice.

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

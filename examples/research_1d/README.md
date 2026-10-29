# New-core 1D research example

The baseline example runs ion-free, classical one-dimensional DC with the actual
new compiler and steady solver. The explicit device/catalog is `device.json`.
It supports Maxwell–Boltzmann carriers, fully ionized dopants, DOS-aware
transport, semiconductor work-function contacts, bulk lifetime SRH/radiative/
Auger recombination and the compiler's bound generation profiles. Unsupported
configured physics is rejected; no legacy engine or fallback is used.

This is a research workflow, not final scientific or release qualification.
Interface and broader defect extensions, hysteresis and 2D remain outside these
new-core examples. Coherent spectral/TMM optics has the bounded example below;
graded and incoherent optical extensions remain unsupported. Mesh and voltage-grid convergence
and the full registry release remain unqualified. Separate sections below describe bounded transient,
ionic and neutral-defect examples and their limits. DC voltage continuation is not a
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
channels may be `fixed_reservoir`, finite-rate `robin`, or selective combinations
with a blocking channel, within the solver's checked numerical envelope.
A separate 37-node Robin case was
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

The selective extension has verified a positive-polarity device with electron
exchange only on the right and hole exchange only on the left at 1 V DC,
including both-terminal DC derivatives. Some reversed-polarity and mixed-contact
cases still fail the unchanged Maxwell-current gate, including a mixed
negative-polarity case at 0.95 V. These failures return no accepted response;
general selective-contact robustness remains open. A carrier blocked at both
ends with a requested zero frequency is explicitly outside the validated scope.
Generation and recombination can change that carrier's inventory, so this
boundary does not assume its inventory is universally conserved.

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
have retained Newton/roundoff failures. Ions and qualified hysteresis remain
outside this transient implementation.

An explicit time-dependent voltage example is also available:

```sh
.venv/bin/python -B -m examples.research_1d.transient_voltage \
  --output "$HOME/.local/share/solarlab/research_1d/transient-voltage.json" --refine
```

It uses constant generation scale 1 and
`V(t) = 0.1 + 0.001 * (t / 400 ns)²`, with 4, 8 and 16 declared steps.
Reference changes are included in each physical inventory difference. A recorded
harmonic potential seed imposes the new electrode voltage while preserving the
old carrier populations; Newton solves the full coupled endpoint equations.
Independent checks found approximately first-order inventory and terminal-current
refinement. Reported displacement currents are interval values `ΔD / Δt`, not
instantaneous pulse peaks. This is a driven time history; steady J–V and final
hysteresis qualification remain separate. The fixed-bias branch retains its
previous accepted states and diagnostics exactly in the archived comparison.

## Persisting a transient

`solarlab.experiments.transient_results` provides
`run_transient_for_publication`, `publish_transient_result` and
`read_transient_result`. The computation wrapper records its actual inputs,
selected source-file hashes and runtime during the solve. It does not infer the
history of a supplied compiled device or initial DC state. Publishing a result
without that capture explicitly records its provenance as `not_recorded`.

For an existing compiled device, accepted initial state, protocol and controls:

```python
from solarlab.experiments.transient_results import (
    run_transient_for_publication, publish_transient_result,
)
from solarlab.io.run_store import RunStore

captured = run_transient_for_publication(
    compiled, initial, protocol, options=options, initial_options=initial_options,
)
with RunStore(store_path) as store:
    row = store.enqueue("light-ramp", opaque_identity=captured.identity,
                        input_metadata={"kind": "generation transient"})
    token = store.claim("light-ramp", expected_attempt_id=row["current_attempt_id"],
                        expected_generation=row["generation"], owner_id="research")
    result = publish_transient_result(store, token, captured.result,
                                      nodes_m=captured.nodes_m, capture=captured)
    store.complete(token, state="succeeded" if captured.result.completed else "failed",
                   result=result, artifacts=result["artifact_ids"])
```

Read the committed result in another process:

```sh
.venv/bin/python -B examples/research_1d/read_transient.py \
  --store /absolute/local/store --run-id light-ramp
```

The reader checks fourteen physical arrays plus raw diagnostics and the result
manifest against the committed attempt. Requested times and the accepted prefix
remain separate; an interrupted calculation retains its first failed attempt.
Step currents align with `accepted_times_s[1:]`; no initial displacement current
is invented. A successful read means data were read and verified, including a
readable partial history. It does not change the numerical outcome or the
`not_evaluated` physical-certificate status.

## Transient process jobs

Run the same ion-free time solver in an owned child process with persisted
results and explicit resource limits:

```sh
.venv/bin/python -B examples/research_1d/process_transient.py run \
  --store outputs/transient_jobs/store --work outputs/transient_jobs/work \
  --build-source-commit "$(git rev-parse HEAD)"
.venv/bin/python -B examples/research_1d/process_transient.py read \
  --store outputs/transient_jobs/store --run-id transient-completed
```

The child compiles the device, solves its initial DC state and advances two
37-node time steps. The example binds immutable committed core sources and any
separately captured active adapters; it also records the actual current checkout
HEAD. Readback uses only stored data. A supplied source archive can be selected
with `--source-archive`, keeping its commit separate from the current checkout.

Add `--failure-case --run-id transient-failed-prefix` to the run command to
exercise a deliberately small Newton budget. The failed calculation retains its
accepted prefix and first failed attempt, with one failed terminal and sixteen
artifacts. Run and read exit with status 2 for that numerical failure, even when
readback succeeds. Cancellation retains one cancelled terminal and does not
claim a returned history. Exceptions and resource overrides retain their own
failure meaning. This process interface does not change scientific acceptance
or the transient result's `not_evaluated` status.

## Coupled ionic steady state

The bounded single-ion compiler and steady solver solve potential, electrons,
holes and one positive finite-site ion population together. The runnable example
uses one electrical layer, blocking ion endpoints and a fixed opposite
background, with a conserved physical ion inventory:

```sh
.venv/bin/python -B -m examples.research_1d.ionic_steady \
  --output outputs/ionic_steady.json
```

The independently checked 13-node dark case at 2 mV converged in three Newton
iterations and changed the ion profile by about 3.91%, with zero reported
relative inventory error. The solver checks every original ion balance,
including the row replaced by the inventory constraint, and the electronic
physical controls. The ion balance and face-flux controls are internal numerical
limits; they do not establish scientific qualification.

A three-layer HTL/absorber/ETL example confines the mobile population to an
explicitly declared absorber domain:

```sh
.venv/bin/python -B -m examples.research_1d.ionic_steady --multilayer \
  --output outputs/ionic_multilayer_steady.json
```

This case uses 37 electrical nodes and 13 ion nodes, including both absorber
interface planes. The 124-coordinate cold solve at 2 mV converged in 12 Newton
iterations with zero reported relative inventory error; it solves all four
fields together. Ion charge and storage use physical domain cell volumes,
projected onto the electrical grid. Interface weights come from the actual
geometry. Ion flux is zero outside the declared domain and at its blocking
endpoints.

The compiler records original ionic inputs and the explicit electronic
component projection. Multilayer callers must supply `ion_domain_layer` and
`ion_domain_endpoints=("blocking", "blocking")`; undeclared additional ion domains
are refused. Zero diffusivity retains population, background and storage in
assembly; the connected-domain steady solver rejects partially disconnected
diffusivity. Independent and shared-site dual-ion examples are described below.
The result records `qualification: false`; final mesh, time and scientific
acceptance remain open.

## Driven ionic histories

The physical backward-Euler driver advances potential, carriers and the mobile
ion population together, including the confined absorber domain. Generate an
explicit steady initial artifact, then run a four-step voltage ramp:

```sh
.venv/bin/python -B -m examples.research_1d.ionic_steady --multilayer \
  --output outputs/ionic_initial.json
ionic_initial_sha=$(.venv/bin/python -c 'import hashlib,pathlib; print(hashlib.sha256(pathlib.Path("outputs/ionic_initial.json").read_bytes()).hexdigest())')
.venv/bin/python -B -m examples.research_1d.ionic_transient \
  --initial-result outputs/ionic_initial.json --initial-sha256 "$ionic_initial_sha" \
  --output outputs/ionic_history.json
```

Add `--refine` to compare 4, 8 and 16 time steps from the same initial artifact.
The example drives 2 to 4 mV over one second in the dark on the 37-electrical /
13-ion-node device. The recorded fine history changed the ion distribution by
about 0.29%, with zero reported total-inventory drift. Successive final ion-cell
inventory differences decreased by a factor of about 1.85. This demonstrates a
bounded working history and decreasing timestep differences; it does not
establish continuum accuracy or a qualified hysteresis experiment.

The loader checks the initial file hash, accepted steady evidence, controls and
physical fields, then rechecks initial admission without another steady solve.
Every dynamic ion equation remains present; steady inventory constraints and
zero-flux conditions are not used as transient acceptance criteria. Physical
population differences include stable finite-site increments and voltage-reference
motion. Each step records the full conduction/displacement current and charge
accounting. A failed run preserves its accepted prefix and first failed attempt,
returns exit status 2, and retains `physical_certificate_status: not_evaluated`.

## Independent positive and negative ions

Run a cold coupled dual-ion steady state and four driven time steps:

```sh
.venv/bin/python -B -m examples.research_1d.dual_ionic \
  --output-directory outputs/dual_ionic
```

Use a new output directory: existing evidence is not overwritten. The example
has 37 electrical nodes, 13 nodes per ion species and 137 unknowns. Both species
occupy independent finite sites in the same blocking absorber domain. Their
inventories are constrained separately at steady state; the physical time
equations retain all particle balances and the signed ionic current.

The independently reproduced dark 2 mV steady state converged in 10 Newton
iterations. All four 2-to-4 mV steps over one second converged, with positive
and negative populations changing by about 0.148% and 0.0782%. Per-species
conservation and full conduction/displacement-current checks passed. This is
a bounded working case. The shared-site model has a separate example below;
additional ion domains, mesh/time qualification and hysteresis qualification
remain open.

## Shared-site positive and negative ions

Run the original diffusion-only shared-site model with a common capacity:

```sh
.venv/bin/python -B -m examples.research_1d.shared_site_ionic \
  --output-directory outputs/shared_site_ionic
```

This uses the same 37/13-node blocking absorber with reference densities
`1e24` and `6e23` m⁻³ and a common capacity of `2e24` m⁻³. Both populations
share the vacancy fraction. Their coupled density derivatives, physical
increments and inventory constraints are included in the solve. Accepted
states are not clipped or renormalized.

The dark 2 mV cold solve converged in 12 Newton iterations, followed by all
four requested steps to 4 mV over one second. The largest summed occupancy
was about 98.05%; both populations moved by about 0.36% during the history.
The output preserves each inventory, vacancy fraction, current balance and
failure evidence. Shared sites require equal capacities and the original
diffusion-only model; other shared-site combinations are refused. This
bounded example does not establish mesh/time or final scientific qualification.

## Immobile ionic populations

An exactly zero diffusivity preserves the declared population in every ion
cell. The target comes from the compiled `P0` or `P0_neg` profile; changing a
Newton initial guess does not change that physical target. Shared-site
coordinates may move while the corresponding physical population stays fixed.

```sh
.venv/bin/python -B -m examples.research_1d.immobile_ions \
  --kind shared_mixed --output-directory outputs/immobile_shared
```

The bounded mixed example has 13 electrical nodes and five ion nodes. It
solves a dark 2 mV steady state, four driven time steps, and 0/1 Hz responses.
`--kind single` exercises the single-species fully immobile case. These two
cases have actual run evidence; the `shared_full` variant has assembly checks
but is outside the recorded end-to-end witnesses. Partly disconnected
zero/positive-diffusivity networks remain unsupported. No artificial mobility,
clipping, or redistribution is used to obtain a steady state. Final scientific
qualification remains open.

## Equilibrium ionic J-V

Use the accepted shared-site steady file from the earlier example:

```sh
JV_INITIAL=outputs/shared_site_ionic/Steady.json
JV_INITIAL_SHA=$(shasum -a 256 "$JV_INITIAL" | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_jv solve \
  --kind shared --voltages .001 .002 .003 --generation-scale 0 \
  --initial-policy saved_accepted --initial-result "$JV_INITIAL" \
  --initial-sha256 "$JV_INITIAL_SHA" --output outputs/ionic_jv.json
JV_RESULT_SHA=$(shasum -a 256 outputs/ionic_jv.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_jv read \
  --result outputs/ionic_jv.json --sha256 "$JV_RESULT_SHA"
```

Each point solves the coupled steady equations and preserves each mobile
species' inventory. The preceding accepted state is only a numerical guess;
the first failed point stops the curve and retains its candidate and checks.
This is an equilibrium curve, with no scan-rate or hysteresis interpretation.
The three dark points above are verified. A direct change from the saved dark
state to generation scale 1 at 2 mV stagnated and remains a recorded failure.
The explicit density-initialization route below provides an illuminated state.
Missing short/open circuit brackets leave the corresponding photovoltaic
metrics unavailable.
The current driver requires positive diffusivity in one blocking domain.

## Illuminated ionic initialization

Use the accepted dark `Steady.json` from the shared-site example. Select the
direct policy explicitly to prepare positive carrier densities once and then
solve the original coupled ionic equations:

```sh
LIGHT_INITIAL=outputs/shared_site_ionic/Steady.json
LIGHT_INITIAL_SHA=$(shasum -a 256 "$LIGHT_INITIAL" | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_initialization solve \
  --policy direct_density_seed_v1 \
  --initial-result "$LIGHT_INITIAL" --initial-sha256 "$LIGHT_INITIAL_SHA" \
  --voltage .002 --generation-scale 1 --output outputs/ionic_initialization.json \
  --accepted-steady-output outputs/illuminated_ionic_steady.json
LIGHT_STEADY_SHA=$(shasum -a 256 outputs/illuminated_ionic_steady.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_jv solve \
  --kind shared --voltages .001 .002 .003 --generation-scale 1 \
  --initial-policy saved_accepted --initial-result outputs/illuminated_ionic_steady.json \
  --initial-sha256 "$LIGHT_STEADY_SHA" --output outputs/illuminated_ionic_jv.json
INIT_SHA=$(shasum -a 256 outputs/ionic_initialization.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_initialization read \
  --result outputs/ionic_initialization.json --sha256 "$INIT_SHA"
```

Preparation preserves the initial potential and ionic coordinates. Both ionic
species remain active in the final coupled solve and at each subsequent J–V
point. All original coupled acceptance checks remain in force; a positive
density seed alone is never an accepted solution. Saved seed reuse is optional
and requires both `--density-seed-result` and `--density-seed-sha256`.

The bounded shared-site witness has 37 electrical nodes and 13 nodes per ionic
species. Its three illuminated points pass the original gates, but do not
bracket short or open circuit and do not qualify photovoltaic metrics, mesh
accuracy or hysteresis. The earlier direct dark-to-light stagnation and the
failed auxiliary-electronic strategy remain recorded failures. The successful
direct policy does not run that auxiliary solver. The helper checks wall time
between stages; use a process supervisor when a hard deadline is required.

## Illuminated ionic J–V and sampled photovoltaic metrics

Use the accepted illuminated state produced above to run the shared-site
device from 0 to 1.25 V in uniform 25 mV steps:

```sh
LIGHT_STEADY_SHA=$(shasum -a 256 outputs/illuminated_ionic_steady.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_pv solve \
  --initial-result outputs/illuminated_ionic_steady.json \
  --initial-sha256 "$LIGHT_STEADY_SHA" --output outputs/ionic_pv.json
PV_SHA=$(shasum -a 256 outputs/ionic_pv.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_pv read \
  --result outputs/ionic_pv.json --sha256 "$PV_SHA"
```

The verified 51-point command completed in about 15 seconds on the development
machine. Every point solves the full coupled system with the same physical
checks; reading the result runs no solver. The saved data retain the first
failure and accepted prefix if a calculation stops early.

The bounded example gives sampled Jsc ≈ 144.384 A/m², interpolated Voc ≈
1.21462 V, maximum power ≈ 147.104 W/m² and FF ≈ 0.838815. These use
piecewise-linear voltage interpolation; mesh and voltage-grid accuracy remain
unqualified. PCE is unavailable because incident irradiance is undeclared.
This steady curve has no scan-rate or hysteresis interpretation. Historical
`coarse-probe`, `continue` and `tail` commands retain the earlier failed coarse
attempt and its successful continuations.

## Ionic J–V process jobs

Run the illuminated shared-site curve through the shared process supervisor and
persist its fields, currents, inventories, metrics and initialization evidence:

```sh
.venv/bin/python -B -m examples.research_1d.process_ionic_jv run \
  --build-source-commit "$(git rev-parse HEAD)" \
  --store outputs/ionic_pv_store --work outputs/ionic_pv_work \
  --run-id illuminated-jv
.venv/bin/python -B -m examples.research_1d.process_ionic_jv read \
  --store outputs/ionic_pv_store --run-id illuminated-jv
```

Use a checkout matching the committed source supplied to the worker. The
default initialization records one dark and one illuminated coupled solve,
followed by the 51 curve points. Reading submits no job and imports no solver.
The verified process result contains 15 artifacts; the supervisor alone
publishes them and records the terminal state.

To reuse its exact accepted illuminated initial state:

```sh
.venv/bin/python -B -m examples.research_1d.process_ionic_jv export-initial \
  --store outputs/ionic_pv_store --run-id illuminated-jv \
  --output outputs/ionic_pv_initial.json
```

Pass that file and its printed digest with `run --initial-result ...
--initial-sha256 ...`. Adding `--failure-case` deliberately limits the curve
to one second. A failed run retains its accepted prefix and first failed point,
and its read command returns exit 2. The actual failure witness retained nine
points and failed at 0.225 V; timing can change the prefix on another machine.
Saved reuse performs no initialization solve after admission. The initial
serialization failure remains a separate failed attempt with no curve data.
PCE stays unavailable without declared irradiance, and scientific qualification
remains pending. The HTTP and browser entries below expose this process route.

## Ionic J–V HTTP jobs

Start the installed illuminated shared-site service from a committed checkout:

```sh
.venv/bin/python -B -m examples.research_1d.ionic_jv_service \
  --build-source-commit "$(git rev-parse HEAD)" \
  --current-repository-head "$(git rev-parse HEAD)" \
  --store outputs/ionic_jv_http_store --work outputs/ionic_jv_http_work --port 8000
```

Use the submission and catalog-header code in **Local HTTP service**, with:

```python
from examples.research_1d.ionic_jv_service import example_submission

job = example_submission("my-ionic-jv")
```

The installed protocol uses 51 voltages from 0 to 1.25 V, generation scale one,
12 intervals per layer, two mobile shared-site species and a blocking absorber
domain. The catalog advertises the exact initial-state recipe and numerical
controls. The new core performs the dark initial solve, coupled illuminated
initialization and actual steady curve through the existing process supervisor.

The service preserves the core request bytes and binds the installed model,
source and maintenance context to the stored manifest. GET result/artifact
endpoints and SSE expose completed curves and genuine failed prefixes. Reading
an existing attempt never submits a new calculation. The accepted HTTP witness
contains 51 samples, 15 artifacts and 23 logical arrays; the independent reader
also verified the original failed process prefix. Missing irradiance leaves PCE
unavailable, and sampled/interpolated PV metrics remain scientifically unqualified.
Other grids, protocols, ionic domains and CT transport require separate verified
admission; they are outside this fixed HTTP entry.

## Ionic J–V browser entry

With the ionic J–V HTTP service above running on port 8000, build and preview
the frontend in a second terminal:

```sh
cd web
npm ci --no-audit --no-fund
npm run build
node scripts/ionic-jv-browser.mjs --service http://127.0.0.1:8000 --port 5173
```

Open `http://127.0.0.1:5173/?view=ionic-jv` and select **Run ionic J–V**.
The installed device and full protocol are shown before submission. One click
starts one process job; a lost connection does not automatically resubmit it.
The fixed curve budget is 25 s and the process ceiling is 30 s. Completion on
another machine is not guaranteed within that budget; any accepted prefix and
the first failure remain available.

Eight plots display the accepted curve and spatial fields. **Open a stored
attempt** and the saved readback link retrieve the exact store/run/attempt;
reconnect and reload only read existing data. Failed curves show accepted,
failed and unattempted voltage points separately, with unavailable PV metrics
left empty. The reviewed browser completed the 51-point protocol and reproduced
all 23 logical arrays and 15 artifacts, including an original nine-point failed
prefix. Scientific qualification remains pending. Stop the service and preview
with Ctrl-C when finished; closing the page only closes its reader.

## Multiple blocking ion domains

The additive two-domain compiler and equation assembly can be exercised with:

```sh
.venv/bin/python -B -m examples.research_1d.multiple_ionic_domains
```

It prepares two explicitly separated blocking layer domains on one electronic
mesh, with an ion-free gap. The actual residual, sparse Jacobian and physical
storage retain each domain's species, inventories and original reference
populations. Signed ionic charge enters the common Poisson equation; ionic
particle current is exactly zero at blocking endpoints and through the gap.

Assembly checks cover unequal domain grids, positive and independent dual
populations, shared finite sites and immobile populations, including shared
storage derivatives. Existing one-domain behavior is preserved. This example
evaluates a trial state. For a coupled dark steady solve, run:

```sh
.venv/bin/python -B -m examples.research_1d.multiple_ionic_steady \
  --output outputs/multiple_ionic_steady.json
```

This solves one common electronic system and both blocking ion domains, with
each mobile species constrained to its own domain's reference inventory. The saved
result includes the original balance rows, physical current, source identities
and acceptance checks. The bounded 13-electrical-node, 4+4-ion-node example at
2 mV passed independent review. Connected positive-diffusivity species and
entirely immobile species are supported. Each immobile cell preserves its
physical reference population, including shared-site cross derivatives.
To evolve that same accepted dark state with both mobile domains, run:

```sh
MULTI_SHA=$(shasum -a 256 outputs/multiple_ionic_steady.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.multiple_ionic_transient solve \
  --initial-result outputs/multiple_ionic_steady.json --initial-sha256 "$MULTI_SHA" \
  --output outputs/multiple_ionic_transient.json
MULTI_TIME_SHA=$(shasum -a 256 outputs/multiple_ionic_transient.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.multiple_ionic_transient read \
  --result outputs/multiple_ionic_transient.json --sha256 "$MULTI_TIME_SHA"
```

The four backward-Euler steps ramp the bias from 2 to 4 mV over one second.
The transient retains every physical storage and balance row, separate domain
inventories and signed conduction/displacement currents, including contact
storage. It reads the accepted initial state without another cold solve. A
failed step preserves the accepted prefix and returns exit 2. This bounded
example does not establish time or mesh convergence;
final scientific qualification remains pending.

For a shared-site case with a mobile positive and immobile negative species in
the right domain, and both species mobile in the left domain:

```sh
.venv/bin/python -B -m examples.research_1d.multiple_ionic_steady \
  --mixed-immobile --output outputs/multiple_ionic_d0_steady.json
D0_SHA=$(shasum -a 256 outputs/multiple_ionic_d0_steady.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.multiple_ionic_transient solve \
  --mixed-immobile --initial-result outputs/multiple_ionic_d0_steady.json \
  --initial-sha256 "$D0_SHA" --output outputs/multiple_ionic_d0_transient.json
```

The steady result and four-step history passed independent review under the
original controls. Immobile densities stay fixed within the recorded numerical
error while their shared-site coordinates can change as the partner moves.
The time solver retains the full physical storage rows and records those
changes separately; it does not freeze a coordinate or substitute steady rows.

## Ionic small-signal response

Use the accepted `Steady.json` from the shared-site example above. AC rechecks
that saved state and solves the coupled complex linear system without another
steady solve:

```sh
AC_INITIAL=outputs/shared_site_ionic/Steady.json
AC_INITIAL_SHA=$(shasum -a 256 "$AC_INITIAL" | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_impedance solve \
  --kind shared --initial-result "$AC_INITIAL" --initial-sha256 "$AC_INITIAL_SHA" \
  --frequencies 0 .01 1 1000 --ion-mode mobile --output outputs/ionic_ac.json
AC_RESULT_SHA=$(shasum -a 256 outputs/ionic_ac.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_impedance read \
  --result outputs/ionic_ac.json --sha256 "$AC_RESULT_SHA"
```

Choose a new output path for each calculation. `--ion-mode frozen` holds both
ionic populations fixed at the same operating profile; it retains their DC
charge and reports zero ionic perturbation current. `--kind positive` and
`--kind independent` use the corresponding saved steady examples.

Phasors use `exp(+i omega t)`. The voltage phasor is a linear normalization,
not a finite-amplitude drive. Outputs include complex admittance density
(S/m²), impedance-area product (Ω m²), species perturbations and all-face
conduction/displacement currents. Small solve coordinates and their known
voltage lift are retained separately to avoid losing small currents to
floating-point cancellation. Zero frequency enforces each mobile species'
inventory; finite frequencies retain the full storage matrix.

The bounded examples cover fixed carrier reservoirs and one blocking ionic
domain. They retain the first failed frequency and leave later requested
frequencies unattempted. Result reading validates the saved bytes and does not
solve. These checks do not establish mesh convergence, general contact/domain
support, or final scientific qualification.

## AC response with two blocking ion domains

Use the accepted positive-domain steady result from **Multiple blocking ion
domains** as the operating state. The AC command checks its digest and physical
admission, then solves the coupled complex system without a new DC solve:

```sh
MULTI_SHA=$(shasum -a 256 outputs/multiple_ionic_steady.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.multiple_ionic_impedance solve \
  --initial-result outputs/multiple_ionic_steady.json --initial-sha256 "$MULTI_SHA" \
  --frequencies 0 .01 1 1000 --output outputs/multiple_ionic_ac.json
MULTI_AC_SHA=$(shasum -a 256 outputs/multiple_ionic_ac.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.multiple_ionic_impedance read \
  --result outputs/multiple_ionic_ac.json --sha256 "$MULTI_AC_SHA"
```

The bounded example has 13 electrical nodes and four ion nodes in each of two
separated domains. Each domain preserves its own positive-species inventory;
shared electronic equations appear once. All 47 original rows and all 14 faces
are checked, including conduction and displacement current through ion-free
gaps. Phasors use `exp(+i omega t)`, with admittance density in S/m² and impedance
area product in Ω m². Fixed carrier reservoirs, constant carrier mobility and
connected positive ion diffusivity are required. Other contact, ion and field
transport variants are explicitly refused. The four-frequency example passed
the existing internal gates and independent saved-result review; mesh and final
scientific qualification remain pending.

## Ionic transient process jobs

From a committed checkout, run the shared-site example through the same process
supervisor and durable result store:

```sh
.venv/bin/python -B -m examples.research_1d.process_ionic_transient run \
  --build-source-commit "$(git rev-parse HEAD)" \
  --store outputs/ionic_process_store --work outputs/ionic_process_work \
  --run-id shared-history
.venv/bin/python -B -m examples.research_1d.process_ionic_transient read \
  --store outputs/ionic_process_store --run-id shared-history
```

The child performs an actual coupled cold steady solve followed by four
2-to-4 mV time steps. Add `--failure-case` with a fresh store/work directory
to exercise the deliberately insufficient Newton budget: the failed run
retains its initial state, first accepted step and first failed driven step.
Its reader returns exit status 2 while preserving the available data.

Readback exposes 26 logical arrays with explicit units, including both species,
cell inventories and full-face/terminal currents. They are stored in fixed
same-unit groups within the existing artifact limit. Independent reads compare
the original child and committed payloads without importing a solver. Queued
cancellation creates a cancelled terminal without a fabricated history. This
verified process case uses the existing positive-diffusivity shared-site model;
zero-diffusivity process jobs are not covered here.

## Ionic transient HTTP jobs

Start the explicitly installed dark ionic service from a committed checkout:

```sh
.venv/bin/python -B -m examples.research_1d.ionic_transient_service \
  --build-source-commit "$(git rev-parse HEAD)" \
  --store outputs/ionic_http_store --work outputs/ionic_http_work --port 8000
```

Use the submission and header-reading code in **Local HTTP service**, replacing
the `job` dictionary with:

```python
from examples.research_1d.ionic_transient_service import example_submission

job = example_submission("my-ionic-history")
```

The shared-site example performs a coupled cold steady solve followed by four
time steps from 2 to 4 mV. Passing `failure=True` deliberately limits Newton
work; the failed job retains the accepted samples at 0 and 0.25 s and records
the failed step at 0.5 s. Completed and failed histories use the same existing
job, result, artifact and SSE endpoints. Readback exposes 26 logical arrays
within 16 terminal artifacts and never invokes a solver.

The installed catalog declares positive, independent dual-ion and shared-site
devices. The actual HTTP witness covers the shared-site model with positive
diffusivities in one blocking absorber domain. Its request bytes, fixed
numerical controls, selected device, catalog, source and result bindings are
checked. Illuminated or zero-diffusivity HTTP jobs and multiple domains remain
outside this entry's verified scope.

## Ionic transient browser entry

With the ionic HTTP service above on port 8000, start the frontend as described
in **Browser research entry**, then open
`http://127.0.0.1:5173/?view=ionic-transient`. Choose an installed device, enter
the explicit samples and separate initial-steady/time-step work budgets, then
select **Run ionic transient**.

The page displays electrical and ionic grids, species densities and
inventories, and conduction, displacement and total currents. Requested samples
remain visible alongside accepted, failed and unattempted outcomes. Initial
states have no invented time-step currents; photovoltaic metrics and final
scientific qualification remain unavailable for this dark transient workflow.

The bookmark and **Open stored attempt** form reopen the exact store/run/attempt.
Refresh and reconnect only read its saved data. The actual browser witness
covers the shared-site positive-diffusivity model, one completed submission,
and reopening both completed and failed-prefix HTTP histories. The opt-in
GET-only live test takes an explicit endpoint and saved attempt IDs; ordinary
frontend tests do not submit these numerical jobs.

## Transient HTTP jobs

Start the co-installed steady and ion-free transient service:

```sh
.venv/bin/python -B -m examples.research_1d.transient_service \
  --build-source-commit "$(git rev-parse HEAD)" \
  --store "$HOME/.local/share/solarlab/research_1d/transient-http" \
  --work "$HOME/.local/share/solarlab/research_1d/transient-http-work" \
  --port 8000
```

Use the submission and header-reading code in **Local HTTP service**, replacing
the `job` dictionary with:

```python
from examples.research_1d.transient_service import example_submission

job = example_submission("my-transient-run")
```

The completed example has samples at 0, 100 and 200 ns. Passing `failure=True`
demonstrates a deliberately insufficient Newton budget: the terminal state is
`failed`, with its actual accepted prefix available through the same result and
artifact endpoints. It is not reported as a completed history. Real HTTP tests
cover both outcomes, co-installed steady jobs, SSE replay and queued
cancellation. Stored results can be read without invoking a solver. The HTTP
adapter supports ion-free histories. The separate ionic HTTP service above
uses its own explicit installed catalog.

## Transient browser entry

With the co-installed transient service above on port 8000, start the frontend
as described in **Browser research entry**, then open
`http://127.0.0.1:5173/?view=transient`. Select an installed device and enter the
explicit time, voltage and generation samples before choosing **Run transient**.
The page follows the actual job and plots the accepted temporal and spatial
fields, including conduction, displacement and total terminal currents.

The stored requested protocol labels accepted, failed and unattempted steps.
A failed history displays its actual prefix; missing currents at the initial
time remain unavailable. Temporal axes use the accepted history's time domain.
The displayed bookmark and **Open stored attempt** form reopen the exact saved
attempt after a reload. Reconnect and readback do not submit another calculation.
Cancellation targets the displayed attempt and is confirmed from its terminal
state. This entry retains the same ion-free and scientific-qualification limits
as the transient HTTP adapter.

## Explicit neutral bulk defects

Use the existing validated defect declaration to replace lifetime SRH with one
or more neutral single-level species on the owning layer's nodes:

```sh
.venv/bin/python -B -m examples.research_1d.neutral_defects \
  --output outputs/neutral_defects.json
.venv/bin/python -B -m examples.research_1d.neutral_defects \
  --unequal-dos --output outputs/neutral_defects_unequal_dos.json
```

The default example adds a neutral absorber level to `device.json` and runs a
37-node illuminated steady calculation at 2 mV. `--config <device.json>` accepts
a complete declaration with its own explicit neutral inventory. The saved
result includes the declaration, species binding, individual rates and exact
partials, final fields, original residual and solver outcome.

The explicit inventory replaces lifetime SRH on its owned nodes; radiative and
Auger terms and other layers remain active. Neutral species add no trap charge.
Declared energies remain literal distances above the valence-band edge.
The binding converts the historical DEF1 reference populations using the
physical conduction and valence densities of states, including unequal DOS;
equal-DOS nodes retain their original arithmetic. `--unequal-dos` doubles the
absorber valence DOS and adjusts the declared intrinsic populations consistently.
Its output includes the physical DOS, band gap and thermal voltage for inspection.
Charged, distributed, multivalent and metastable defects remain unsupported.
Both bounded examples passed the original numerical gates; broader material,
mesh and final scientific qualification remain pending.

## Field-dependent carrier transport

Smooth Caughey–Thomas velocity saturation is bound to the actual DC
Scharfetter–Gummel fluxes and full Newton Jacobian:

```sh
.venv/bin/python -B -m examples.research_1d.field_transport \
  --output outputs/field_transport.json
.venv/bin/python -B -m examples.research_1d.field_transport \
  --read outputs/field_transport.json
```

The default 37-node illuminated example enables absorber velocity saturation
and solves at 2 mV. `--config <device.json>` accepts an explicit device with
nonzero saturation velocities. The result includes the effective face
diffusivities, field derivatives, original residual and accepted solver state.
The saved-result reader performs no new solve.

This slice uses the full electrostatic face field, requires `ct_beta_n` or
`ct_beta_p` greater than one on each active carrier face, and keeps Poole–Frenkel
enhancement off. CT-off devices retain the original constant-diffusivity path.
Both AC entry points include the field dependence in the current and voltage
drive tangents. Reuse the accepted DC result for an actual small-signal sweep:

```sh
CT_INITIAL_SHA=$(shasum -a 256 outputs/field_transport.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.field_transport_ac solve \
  --kind ion-free --initial-result outputs/field_transport.json \
  --initial-sha256 "$CT_INITIAL_SHA" --frequencies 0 1 1000 \
  --output outputs/field_transport_ac.json
CT_AC_SHA=$(shasum -a 256 outputs/field_transport_ac.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.field_transport_ac read \
  --result outputs/field_transport_ac.json --sha256 "$CT_AC_SHA"
```

The AC command validates the same-device initial state and performs no new DC
solve. The reader checks and decodes the saved response without recomputation.
Admittance density is in S/m² and current response is normalized per volt.
Independent native witnesses cover this ion-free case and a shared-site ionic
device with fixed-reservoir carrier contacts at 0, 0.01, 1 and 1000 Hz. Ionic
initials must be separately accepted on their actual CT device; an accepted
constant-diffusivity state is not an accepted CT state. Positive-only and
independent-site branches have targeted algebra checks but no separate native
CT witness here. Mesh and final scientific qualification remain pending.

## Supplied spectral optics

An explicit photon spectrum and optical-index resources can drive the new
compiler through the existing coherent transfer-matrix kernels:

```sh
.venv/bin/python -B -m examples.research_1d.optical_generation \
  --output outputs/optical_generation.json
.venv/bin/python -B -m examples.research_1d.optical_generation \
  --read outputs/optical_generation.json
```

The example supplies seven wavelengths from 400 to 700 nm and a constant n-k
table for the absorber. These are illustrative data, not AM1.5 illumination or
calibrated material measurements. The declared spectrum is absolute photon flux
per metre of wavelength; scalar `Phi` does not rescale it. Each optical layer
requires an explicit n-k resource or constant `n_optical` with its absorption
coefficient. The first binding uses coherent front normal incidence and
unit-index surrounding media; it rejects graded/incoherent layers and combined
spectral/fixed-generation declarations.

Generation uses absorbed photons in each physical electrical control volume.
The output records the complete supplied resources, reflected/transmitted and
electrically/parasitically absorbed photons, generation profile, physical fields
and solver checks. The 37-node illuminated 2 mV witness passed the original
numerical gates, and photon accounting closes under its reviewed roundoff check.
Spectral/mesh convergence, calibrated optical data and final scientific
qualification remain unestablished. The reader displays the saved outcome
without another optical or electrical solve.

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

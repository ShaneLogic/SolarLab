# Two-dimensional research examples

This directory exercises the new core with a connected lateral and stack grid.
It assembles the classical Poisson and carrier continuity equations and their
analytic Jacobian, then solves the full two-dimensional system with the unified
Newton solver. The separate algebra example exports arbitrary states, which
are not accepted steady solutions.

## Solve and read steady states

```sh
.venv/bin/python -B -m examples.research_2d.steady \
  --output outputs/2d_steady.json
STEADY_2D_SHA=$(shasum -a 256 outputs/2d_steady.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_2d.steady \
  --read outputs/2d_steady.json --sha256 "$STEADY_2D_SHA"
```

The example requests dark and illuminated operating points at 2 mV on the
13-by-5 grid. Each starts from an explicitly counted one-dimensional numerical
preparation, adds a lateral perturbation, and solves all 195 two-dimensional
coordinates. Illuminated preparation uses the existing cold light-continuation
path. A one-dimensional seed never counts as a two-dimensional solution.

The saved output retains initial lateral fluxes, actual Newton iterations,
original residuals, geometry, source identities, fields and signed face and
terminal currents. Physical checks use integrated carrier balances and
width-normalized cross-section/terminal currents; individual local currents
need not be spatially uniform. A failed target stops the sequence and retains
the failure; the reader performs no solve and returns exit 2 for an incomplete
record. Both bounded operating points have passed their original solver and
physical checks. Mesh and final scientific qualification remain pending.

For an existing failed two-target record whose dark point was accepted,
`--illumination-only --prior-result <file> --prior-sha256 <digest>` preserves and
revalidates that dark result without solving it again, then requests only the
illuminated point. The original failure remains linked in the new output.

## Steady J–V and saved continuation

After the successful steady example above, use its accepted illuminated state
to request 51 steady points from 0 to 1.25 V on the same no-grain geometry:

```sh
.venv/bin/python -B -m examples.research_2d.jv solve \
  --initial-result outputs/2d_steady.json --initial-sha256 "$STEADY_2D_SHA" \
  --output outputs/2d_jv.json
JV_2D_SHA=$(shasum -a 256 outputs/2d_jv.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_2d.jv read \
  --result outputs/2d_jv.json --sha256 "$JV_2D_SHA"
```

Each completed attempt is saved separately beside the result. An incomplete
wave exits 2 and retains accepted states, the first failure and unattempted
voltages. For a wall-budget stop with an accepted prefix, explicit continuation
uses only the last accepted state and the exact remaining voltage slice:

```sh
.venv/bin/python -B -m examples.research_2d.jv continue \
  --prefix-result outputs/2d_jv.json --prefix-sha256 "$JV_2D_SHA" \
  --output outputs/2d_jv_tail.json
```

If that tail stops with only its final endpoint remaining, the same command
accepts the tail file and its digest to finish that one point in a new output.
Other numerical failures, changed models/controls and further continuation
chains are refused. Keep earlier result files at their recorded locations:
linked reads verify those original bytes and preserve their stopped outcomes.

All 51 points have been exercised through a first wave, a tail and a final
endpoint without replaying accepted points. Jsc, Voc, maximum power and FF use
the accepted sampled/interpolated curve; this is not a time scan or hysteresis
calculation. PCE remains unavailable unless `--incident-power-W-m2` is explicitly
supplied. These examples do not establish voltage-grid, mesh or final scientific
accuracy.

## Run in a separate process

From a clean committed checkout, the process example binds the source archive
to the same revision as the running code. Use the locked environment from the
[one-dimensional guide](../research_1d/README.md) and a local, unsynchronized
directory for the active database and work files:

```sh
SOLARLAB_2D_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/solarlab-2d.XXXXXX")
SOLARLAB_2D_REV=$(git rev-parse HEAD)
git archive --format=tar --output "$SOLARLAB_2D_ROOT/source.tar" "$SOLARLAB_2D_REV"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
.venv/bin/python -B -m examples.research_2d.process_steady run \
  --store "$SOLARLAB_2D_ROOT/Store" --work "$SOLARLAB_2D_ROOT/Work" \
  --run-id steady2d \
  --build-source-commit "$SOLARLAB_2D_REV" \
  --source-archive "$SOLARLAB_2D_ROOT/source.tar" \
  --current-repository-head "$SOLARLAB_2D_REV" \
  --receipt-json "$SOLARLAB_2D_ROOT/Receipt.json"
SOLARLAB_2D_ATTEMPT=$(.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["attempt_id"])' \
  "$SOLARLAB_2D_ROOT/Receipt.json")
.venv/bin/python -B -m examples.research_2d.process_steady read \
  --store "$SOLARLAB_2D_ROOT/Store" --run-id steady2d \
  --attempt-id "$SOLARLAB_2D_ATTEMPT"
```

The default requests the same no-grain dark and illuminated 2 mV points on
the 13-by-5 grid. `--targets dark` or `--targets illuminated` selects one target;
`--prepare-only` validates the request without starting a process or solving.
The process uses a 30-second limit and retains 22 logical arrays in 12 NPY
groups, the lossless diagnostic payload and execution manifest. Reads do not
run a solver. Existing run IDs are preserved; use a new ID for a new computation.

For a deliberate failure example, add `--insufficient-budget` with a new run
ID and receipt path. This changes only the full-2D Newton count budget, retains
the failed candidate and leaves later targets unattempted. Both the run and
reader exit 2 for a failed attempt. Copy the closed Store and Work directories
with their contents and empty directories when archiving; open archived stores
only after copying them into a local directory. Thread settings above are
requests recorded with the run, not proof of an observed native thread count.

This process entry has verified successful and failed examples. It does not
provide grain, J–V, HTTP or browser execution; those are separate entry points
with their own implementation and validation status.

## Submit and read through HTTP

The HTTP example installs the same fixed no-grain steady experiment in the
shared job service. From a clean committed checkout, create local paths and
start the service in one terminal:

```sh
SOLARLAB_2D_HTTP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/solarlab-2d-http.XXXXXX")
SOLARLAB_2D_HTTP_REV=$(git rev-parse HEAD)
git archive --format=tar --output "$SOLARLAB_2D_HTTP_ROOT/source.tar" "$SOLARLAB_2D_HTTP_REV"
.venv/bin/python -B -m examples.research_2d.http_steady \
  --store "$SOLARLAB_2D_HTTP_ROOT/Store" --work "$SOLARLAB_2D_HTTP_ROOT/Work" \
  --build-source-commit "$SOLARLAB_2D_HTTP_REV" \
  --source-archive "$SOLARLAB_2D_HTTP_ROOT/source.tar" \
  --current-repository-head "$SOLARLAB_2D_HTTP_REV" --port 8000
```

Starting the service does not solve a device. Add `--prepare-only` to validate
the complete request without opening a server. Choose an unused loopback port;
stop the service with Ctrl-C after its jobs finish.

In a second terminal at the repository root, this client explicitly submits
one dark/illuminated job, waits for its terminal event and reads that exact
attempt. Use a new run ID for each computation:

```sh
.venv/bin/python -B - <<'PY'
import httpx
from examples.research_2d.http_steady import example_submission

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=45) as client:
    catalog = client.get("/experiments")
    catalog.raise_for_status()
    headers = {name: catalog.headers[name] for name in
               ("x-solarlab-registry", "x-solarlab-job-schema")}
    submitted = client.post("/jobs", json=example_submission("research-2d-001"), headers=headers)
    submitted.raise_for_status()
    job = submitted.json()
    target = f"{job['run_id']}/attempts/{job['attempt_id']}"
    events = client.get(f"/runs/{target}/events")
    events.raise_for_status()
    result = client.get(f"/results/{target}")
    result.raise_for_status()
    print(result.json())
PY
```

The result includes accepted-target counts, original failure details, geometry,
array fingerprints and artifact download URLs. Each attempt retains the same
22 logical arrays and 14 artifacts as the process entry. The server also reads
original process attempts under their original identities. An actual HTTP
success and saved historical failure have passed independent readback checks;
HTTP completion does not establish mesh or final scientific accuracy.

## Use the browser

Keep the HTTP service above running on port 8000. In another terminal, start
the frontend with its locked dependencies:

```sh
cd web
npm ci --no-audit --no-fund
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/?view=steady-2d` and select an installed device and
dark, illuminated, or both targets. **Run 2D steady** explicitly submits a new
job. The five plots show accepted potential, carrier densities, directed-face
current density and integrated terminal/cross-section currents. Selecting a
saved target, reopening its link and reconnecting read the recorded attempt
without another solve. Failed and unattempted targets remain distinct.

The browser has been exercised with one actual two-target submission and
independently checked against its saved arrays, original historical failure,
and exact-attempt reload. Stop the frontend and HTTP service with Ctrl-C when
finished. Grain J–V and contact experiments use their separate Python entries;
this browser entry exposes the fixed no-grain steady protocol.

## Neutral grain-region recombination

```sh
.venv/bin/python -B -m examples.research_2d.grains \
  --output outputs/2d_grains.json
GRAIN_2D_SHA=$(shasum -a 256 outputs/2d_grains.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_2d.grains \
  --read outputs/2d_grains.json --sha256 "$GRAIN_2D_SHA"
```

The illustrative absorber region is centered at 0.4 µm, has width 0.3 µm,
and declares electron/hole SRH lifetimes of 0.5/0.25 µs. The compiler uses the
exact overlap with each lateral control cell. It blends the grain and bulk SRH
rates and their density derivatives by that fraction, while retaining bulk
radiative and Auger recombination. This neutral model adds no charge or state
variable. The solved example has nonzero transverse current and passes the
original full-row and physical current-balance checks.

`--algebra <output.json>` exports the binding and arbitrary-state equations
without solving. `--config <device.json>` changes the layered device; the
example's spatial grain declaration stays explicit in `spatial_input()`.
Custom spatial declarations use the existing `JV2DInput.microstructure` field.
The output records layer ownership, fractions, represented widths/areas,
lifetimes and the numerical seed separately from the actual 2D result. The
historical grain-role partition owns interfaces on the left, while physical
material coefficients keep right-layer ownership; both are explicit in the
saved binding. Thin-region resolution and mesh qualification remain unverified.

For the same grain declaration, the shared steady J–V driver also supports a
complete illuminated curve:

```sh
.venv/bin/python -B -m examples.research_2d.grain_jv solve \
  --output outputs/2d_grain_jv.json
GRAIN_JV_SHA=$(shasum -a 256 outputs/2d_grain_jv.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_2d.grain_jv read \
  --result outputs/2d_grain_jv.json --sha256 "$GRAIN_JV_SHA"
```

The default creates and separately records an illuminated numerical preparation
and an actual grain 2D initial solve, then requests 51 points from 0 to 1.25 V.
It grants 80 seconds to the curve. The independently reviewed run additionally
used an enclosing 80-second computation deadline including startup and a
90-second command limit; it completed in 56.186 seconds. These operating limits
do not change solver tolerances or scientific acceptance standards.

An existing initial result can be supplied with `--initial-result` and
`--initial-sha256`; its exact physical source, grain binding, accepted fields
and original gates must match. Older source-mismatched results are refused.
Optional `--baseline-result` and `--baseline-sha256` compare the saved no-grain
curve without solving it again. The comparison retains the common accepted
voltage grid and source identities. It reports sampled Jsc/Voc/Pmax/FF, leaves
PCE unavailable without specified irradiance, and makes no mesh-convergence or
causal claim from the illustrative grain parameters.

## Inspect the equations

From the repository root, using the locked environment described in the
[one-dimensional guide](../research_1d/README.md):

```sh
.venv/bin/python -B -m examples.research_2d.dc_assembly \
  --output outputs/2d_assembly.json
.venv/bin/python -B -m examples.research_2d.dc_assembly \
  --read outputs/2d_assembly.json
```

The example uses five lateral nodes and thirteen stack nodes, with 195 coupled
unknowns, 52 lateral faces and 60 stack faces. It exports coordinates, physical
fields, generation, residuals, the sparse Jacobian, face fluxes and terminal
currents. An explicitly nonuniform lateral state produces lateral carrier and
electric displacement fluxes. A separate uniform-lateral state checks the
reduction to the one-dimensional column equations.

The reader opens existing data without solving. Use a fresh output path for a
new assembly run; existing evidence is preserved.

## Geometry and units

`x` is lateral and `y` follows the layer stack. Flattened fields use C order with
shape `(Ny, Nx)`. Volumes represent control areas per unit out-of-plane depth;
face measures are lengths. Endpoint control widths are physical half widths,
so the geometry must not be equated with the historical one-dimensional
integer-endpoint convention without the documented weight conversion.

| Quantity | Unit |
| --- | --- |
| Control area | m² |
| Face measure | m |
| Particle flux density | m⁻² s⁻¹ |
| Integrated carrier balance per depth | m⁻¹ s⁻¹ |
| Integrated Gauss balance per depth | C/m |
| Cross-section or terminal current per depth | A/m |
| Terminal current divided by lateral width | A/m² |

The initial scope uses fixed carrier reservoirs at the stack ends, zero normal
particle and displacement flux at the lateral boundaries, constant carrier
mobility and the supported Beer–Lambert generation binding. The grain example
adds the declared neutral volumetric SRH regions above. Ions, charged or
distributed defects, field-dependent transport, periodic boundaries and
spectral/coherent/incoherent optical declarations are outside this entry.
The algebra checks do not establish a steady solution, mesh convergence or
final scientific qualification.

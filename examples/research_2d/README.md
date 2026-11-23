# 2D Research Workflows (`examples/research_2d/`)

Runnable two-dimensional research examples exercising `solarlab`'s $(N_y, N_x)$ tensor-product control-volume mesh, sparse analytical Jacobian assembly, grain-boundary area-fraction recombination, selective contacts, field-dependent mobility, spectral optics, and out-of-process / HTTP supervision.

For installation, single-thread BLAS environment variables, and the shared `RunStore` / SHA-256 provenance rules, see the root [README.md](../../README.md) and [1D Workflow Guide](../research_1d/README.md#shared-provenance-storage--qualification-contract).

---

## Module Index (All 10 Workflows)

| Module | Scope & Physical Closure |
|:-------|:-------------------------|
| [dc_assembly.py](dc_assembly.py) | Exports 2D $(13 \times 5)$ coordinates, residuals, sparse Jacobian, lateral/vertical fluxes, and 1D uniform-column reduction checks without solving |
| [steady.py](steady.py) | Solves dark and illuminated 2 mV operating points across all $195$ coupled 2D unknowns from a cold 1D continuation seed |
| [jv.py](jv.py) | Runs a 51-point illuminated 2D steady $J\text{–}V$ sweep ($0 \to 1.25\text{ V}$) with explicit prefix/tail `continue` support |
| [grains.py](grains.py) | Neutral vertical grain-boundary SRH recombination blended by exact lateral control-volume overlap fraction |
| [grain_jv.py](grain_jv.py) | Complete 51-point illuminated 2D $J\text{–}V$ sweep on the grain-boundary stack + optional baseline comparison |
| [contacts.py](contacts.py) | Four-channel 2D carrier contact algebra (blocking, fixed Dirichlet, and finite-rate Robin exchange) + bounded Newton solve |
| [field_transport.py](field_transport.py) | Normal-face 2D Caughey–Thomas velocity saturation ($\beta = 2$) arithmetic checks and dark 2 mV steady solve |
| [optical_generation.py](optical_generation.py) | 2D half-endpoint control-volume coherent spectral optical source construction and photon-budget check |
| [process_steady.py](process_steady.py) | Supervised child-process 2D steady execution bound to an explicit `git archive` source tarball |
| [http_steady.py](http_steady.py) | Local FastAPI `/experiments` + `/jobs` + SSE service backing the `web/?view=steady-2d` browser view |

---

## Representative Command Recipes

### 1. Inspect 2D Assembly, Solve Steady Operating Points & Run 2D $J\text{–}V$

```sh
# 1) Export and verify 2D residuals, sparse Jacobian, and 1D uniform reduction
.venv/bin/python -B -m examples.research_2d.dc_assembly --output outputs/research_2d/assembly.json
.venv/bin/python -B -m examples.research_2d.dc_assembly --read outputs/research_2d/assembly.json

# 2) Solve dark and illuminated 2D steady states at 2 mV (13x5 grid, 195 unknowns)
.venv/bin/python -B -m examples.research_2d.steady --output outputs/research_2d/steady.json
STEADY_2D_SHA=$(shasum -a 256 outputs/research_2d/steady.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_2d.steady \
  --read outputs/research_2d/steady.json --sha256 "$STEADY_2D_SHA"

# 3) 51-point illuminated 2D J-V sweep (0 to 1.25 V) + lossless readback
.venv/bin/python -B -m examples.research_2d.jv solve \
  --initial-result outputs/research_2d/steady.json --initial-sha256 "$STEADY_2D_SHA" \
  --output outputs/research_2d/jv.json
JV_2D_SHA=$(shasum -a 256 outputs/research_2d/jv.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_2d.jv read \
  --result outputs/research_2d/jv.json --sha256 "$JV_2D_SHA"
```

*(If a wall-time budget stops a 2D sweep mid-curve, resume strictly from the last accepted voltage via `python -B -m examples.research_2d.jv continue --prefix-result ... --prefix-sha256 ... --output ...`.)*

### 2. Grain-Boundary Overlap, Contacts, Field Mobility & 2D Spectral Optics

```sh
# Centered absorber grain region (exact control-volume SRH overlap) + full 51-point grain J-V
.venv/bin/python -B -m examples.research_2d.grains --output outputs/research_2d/grains.json
.venv/bin/python -B -m examples.research_2d.grain_jv solve --output outputs/research_2d/grain_jv.json

# Mixed blocking / Dirichlet / Robin 2D carrier contacts
.venv/bin/python -B -m examples.research_2d.contacts --output outputs/research_2d/contacts_algebra.json
.venv/bin/python -B -m examples.research_2d.contacts --solve outputs/research_2d/contacts_steady.json \
  --seed outputs/research_2d/steady.json --seed-sha256 "$STEADY_2D_SHA"

# 2D Caughey-Thomas field-dependent mobility (algebra + dark 2 mV steady solve)
.venv/bin/python -B -m examples.research_2d.field_transport \
  --output outputs/research_2d/ct_algebra.json \
  --steady-output outputs/research_2d/ct_steady.json

# 2D spectral optical generation on half-height endpoint control cells
.venv/bin/python -B -m examples.research_2d.optical_generation \
  --output outputs/research_2d/optical_2d.json
```

### 3. Supervised Process & Local HTTP / Browser Execution

```sh
SOLARLAB_2D_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/solarlab-2d.XXXXXX")
SOLARLAB_2D_REV=$(git rev-parse HEAD)
git archive --format=tar --output "$SOLARLAB_2D_ROOT/source.tar" "$SOLARLAB_2D_REV"

# Out-of-process execution bound to the committed git archive
.venv/bin/python -B -m examples.research_2d.process_steady run \
  --store "$SOLARLAB_2D_ROOT/Store" --work "$SOLARLAB_2D_ROOT/Work" \
  --run-id steady2d --build-source-commit "$SOLARLAB_2D_REV" \
  --source-archive "$SOLARLAB_2D_ROOT/source.tar" \
  --current-repository-head "$SOLARLAB_2D_REV" \
  --receipt-json "$SOLARLAB_2D_ROOT/Receipt.json"

# Local HTTP service for web/?view=steady-2d
.venv/bin/python -B -m examples.research_2d.http_steady \
  --store "$SOLARLAB_2D_ROOT/HttpStore" --work "$SOLARLAB_2D_ROOT/HttpWork" \
  --build-source-commit "$SOLARLAB_2D_REV" \
  --source-archive "$SOLARLAB_2D_ROOT/source.tar" \
  --current-repository-head "$SOLARLAB_2D_REV" --port 8000
```

---

## 2D Geometry & Unit Conventions

`x` is the lateral coordinate ($N_x$ nodes) and `y` follows the vertical layer stack ($N_y$ nodes). Flattened state vectors use row-major `(Ny, Nx)` C order with physical half-width contact control cells (`HALF_ENDPOINT_V1`):

| Quantity | Unit (per unit out-of-plane depth) |
|:---------|:-----------------------------------|
| Cell control area / face measure | $\mathrm{m}^2$ / $\mathrm{m}$ |
| Particle flux density | $\mathrm{m}^{-2}\ \mathrm{s}^{-1}$ |
| Integrated cell carrier balance / Gauss balance | $\mathrm{m}^{-1}\ \mathrm{s}^{-1}$ / $\mathrm{C\ m}^{-1}$ |
| Integrated cross-section current / width-normalized terminal current | $\mathrm{A\ m}^{-1}$ / $\mathrm{A\ m}^{-2}$ |

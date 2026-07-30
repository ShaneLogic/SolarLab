# 2D Lateral Material Heterogeneity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a SolarLab 2D device declare laterally-varying material regions (mesoporous scaffold, second phase, embedded column), and give the resulting in-plane heterointerfaces the same thermionic-emission cap and interface-SRH treatment the vertical interfaces get.

**Architecture:** Two-step "material ID bitmap → parameter field" pipeline, borrowed from ChargeFabrica (`github.com/nsdt-zhaw/ChargeFabrica`). A frozen `LateralGeometry` descriptor rasterises onto the existing `Grid2D` to produce an `(Ny, Nx)` integer region-ID field; `build_material_arrays_2d` then paints per-region `MaterialParams` over the existing `extrude()` baseline. Downstream physics (`flux_2d.sg_fluxes_2d_*` harmonic-mean face D, `continuity_2d` `phi_n = phi + chi`) already handles lateral parameter jumps correctly — it is only ever fed uniform fields today. Interface physics is extended from "y-faces only, column 0 hardcoded" to full `(Ny, Nx)` face masks.

**Tech Stack:** Python 3.13, numpy, scipy (`solve_ivp` Radau, `sparse.linalg.splu`), pytest. No new dependencies.

## Global Constraints

- **Bit-identity is the primary gate.** Every task must leave `LateralGeometry()` (empty) / no-`lateral_geometry`-in-YAML paths producing byte-identical arrays to `main`. Assert with `np.array_equal`, not `np.allclose`.
- **Frozen dataclasses only.** `MaterialParams`, `LayerSpec`, `DeviceStack`, `Grid2D`, `MaterialArrays2D`, `Microstructure` are frozen. Never mutate — use `dataclasses.replace`.
- **SI units everywhere**, except `chi` and `Eg` in eV (numerically equal to volts through `q`).
- **Never hand-roll `MaterialParams` parsing.** Use `perovskite_sim.models.config_loader.material_params_from_dict`. The inline-device parser has drifted from the YAML loader three separate times (see `project_inline_path_parser_drift` — flags, `optical_material`, and 17 fields including `Nc300`, which silently killed the DOS `V_oc` fold).
- **Grid index convention:** `Grid2D` node `(j, i)` → linear index `j * Nx + i` (y-major, C-order). `x` is lateral (`Nx` nodes), `y` is the stack axis (`Ny` nodes). State vector is `[n.flatten(), p.flatten()]`, length `2 * Ny * Nx`.
- **`build_grid_2d(layers, lateral_length, Nx, ...)` returns `Nx + 1` lateral nodes** (`np.linspace(0, L, Nx + 1)`). The parameter is intervals, the grid property `grid.Nx` is nodes. Always read `grid.Nx`, never the constructor argument.
- **Tests that measure wall-clock or touch the near-singular Radau branch must pin BLAS themselves** via a module-scoped autouse fixture — `tests/conftest.py` only pins when the `slow` marker is selected. See `tests/unit/experiments/test_jv_branch_rejection.py` for the pattern.
- **Run `pytest -m slow` before claiming any change to interface or generation defaults is clean.** The slow lane is the only gate covering ion-coupled full sweeps; a default `pytest` run excludes it and will report green on a broken `main` (this happened on 2026-07-28).
- **Commit style:** conventional commits, no attribution trailer. Include `Constraint:` / `Rejected:` / `Not-tested:` git trailers on non-trivial commits.

## Measured Baseline (2026-07-29, this machine)

Establish these numbers before starting; every performance gate references them.

Config `configs/nip_MAPbI3.yaml`, `Ny_per_layer=10` (Ny=31), `lateral_length=500e-9`, `lateral_uniform=True`, cold seed `n=p=1e16`, `V_app=0`, `t_end=1e-9`, `max_nfev=20000`, BLAS pinned to 1 thread:

| `Nx` arg | `grid.Nx` | `N_unk = 2·Ny·Nx` | `run_transient_2d` |
|---:|---:|---:|---:|
| 4 | 5 | 310 | 0.50 s |
| 8 | 9 | 558 | 1.19 s |
| 16 | 17 | 1054 | 4.61 s |
| 32 | 33 | 2046 | 25.79 s |

Cost scales ≈ `N^2.5` (the final doubling costs 5.6×). Cause: `solver_2d.py:860` calls `solve_ivp(..., method="Radau")` with **no `jac` and no `jac_sparsity`**, so Radau builds a dense finite-difference Jacobian — `N` RHS evaluations plus an `O(N³)` dense LU per Jacobian refresh.

**Shipped 2D tests run at `Nx=4`** (`tests/regression/test_twod_validation.py:75`) **and `Nx=10`** (`tests/regression/test_twod_microstructure.py:37`). Four lateral nodes cannot resolve any real morphology. Tasks 1–8 target `Nx ≤ 32` where cost is tolerable; Task 9 attacks the wall.

## Pre-existing Gap This Plan Also Closes

`solver_2d.py:672` computes recombination with `total_recombination(...)` only — bulk SRH + radiative + Auger. **The 1D interface channel (`stack.interfaces` SRVs, `InterfaceDefect`, the interface-plane closure) is entirely absent from the 2D solver, on vertical interfaces too.** Any 2D result on a config with a populated `device.interfaces` block is currently missing that channel silently. Task 6 closes this for y-faces before Task 7 extends it laterally; it is a prerequisite, not a bonus.

## File Structure

**Create:**
- `perovskite_sim/twod/lateral_geometry.py` — `LateralRegion`, `LateralGeometry`, shape rasterisers, `build_region_id_field`. Pure functions on `Grid2D`; no solver imports. Target ~220 lines.
- `perovskite_sim/twod/interface_faces_2d.py` — lateral/vertical heterointerface face detection and interface dual-cell widths. Target ~120 lines.
- `perovskite_sim/twod/interface_recomb_2d.py` — vectorised interface SRH applied on y-face and x-face masks. Target ~160 lines.
- `configs/twod/mesoporous_scaffold_demo.yaml` — capstone demo device.
- `tests/unit/twod/test_lateral_geometry.py`
- `tests/unit/twod/test_interface_faces_2d.py`
- `tests/unit/twod/test_interface_recomb_2d.py`
- `tests/regression/test_twod_lateral_heterogeneity.py`
- `scripts/bench_2d_scaling.py` — reproducible scaling benchmark (Task 9).

**Modify:**
- `perovskite_sim/models/device.py` — add `DeviceStack.lateral_geometry` field (default `None`).
- `perovskite_sim/models/config_loader.py` — parse a top-level `lateral_geometry:` block.
- `perovskite_sim/twod/solver_2d.py:169-171` — `extrude()` → geometry-aware `paint()`; extend `MaterialArrays2D` with x-face masks and interface-SRH caches; wire interface recombination into `assemble_rhs_2d`.
- `perovskite_sim/twod/continuity_2d.py:100-118` — replace the column-0-hardcoded y-face TE loop with full 2D masks; add the x-face cap.
- `perovskite_sim/twod/experiments/jv_sweep_2d.py:230-310` — accept and thread `lateral_geometry`.

---

### Task 1: `LateralGeometry` data model and rasteriser

Pure geometry. No solver, no material parameters yet — this task only answers "which region owns node `(j, i)`".

**Files:**
- Create: `perovskite_sim/twod/lateral_geometry.py`
- Test: `tests/unit/twod/test_lateral_geometry.py`

**Interfaces:**
- Consumes: `perovskite_sim.twod.grid_2d.Grid2D` (fields `x: np.ndarray`, `y: np.ndarray`; properties `Nx`, `Ny`, `n_nodes`).
- Produces:
  - `BoxShape(x_min: float, x_max: float, y_min: float, y_max: float)` — frozen.
  - `SinusoidColumnShape(x_center: float, amplitude: float, wavelength: float, width: float, y_min: float, y_max: float, phase: float = 0.0)` — frozen.
  - `LateralRegion(name: str, material: MaterialParams, shape: BoxShape | SinusoidColumnShape)` — frozen.
  - `LateralGeometry(regions: tuple[LateralRegion, ...] = ())` — frozen.
  - `build_region_id_field(grid: Grid2D, geom: LateralGeometry) -> np.ndarray` — returns `(Ny, Nx)` `int32`; `-1` means "base 1D layer stack", `k` means `geom.regions[k]`. Later regions overwrite earlier ones (painter's algorithm).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/twod/test_lateral_geometry.py`:

```python
import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.lateral_geometry import (
    BoxShape,
    LateralGeometry,
    LateralRegion,
    SinusoidColumnShape,
    build_region_id_field,
)
from perovskite_sim.models.material import MaterialParams


def _grid():
    """31 y-nodes over 300 nm, 21 x-nodes over 100 nm (dx = 5 nm)."""
    layers = [Layer(thickness=100e-9, N=10),
              Layer(thickness=100e-9, N=10),
              Layer(thickness=100e-9, N=10)]
    return build_grid_2d(layers, lateral_length=100e-9, Nx=20,
                         lateral_uniform=True)


def _mat():
    return MaterialParams(chi=4.1, Eg=3.2)


def test_empty_geometry_is_all_base():
    grid = _grid()
    ids = build_region_id_field(grid, LateralGeometry())
    assert ids.shape == (grid.Ny, grid.Nx)
    assert ids.dtype == np.int32
    assert np.all(ids == -1)


def test_box_paints_only_inside():
    grid = _grid()
    geom = LateralGeometry(regions=(
        LateralRegion(name="slab", material=_mat(),
                      shape=BoxShape(x_min=30e-9, x_max=60e-9,
                                     y_min=100e-9, y_max=200e-9)),
    ))
    ids = build_region_id_field(grid, geom)

    inside_x = (grid.x >= 30e-9 - 1e-15) & (grid.x <= 60e-9 + 1e-15)
    inside_y = (grid.y >= 100e-9 - 1e-15) & (grid.y <= 200e-9 + 1e-15)
    expected = np.outer(inside_y, inside_x)

    assert np.array_equal(ids == 0, expected)
    assert np.array_equal(ids == -1, ~expected)


def test_later_region_overwrites_earlier():
    grid = _grid()
    both = BoxShape(x_min=0.0, x_max=100e-9, y_min=0.0, y_max=300e-9)
    geom = LateralGeometry(regions=(
        LateralRegion(name="first", material=_mat(), shape=both),
        LateralRegion(name="second", material=_mat(), shape=both),
    ))
    ids = build_region_id_field(grid, geom)
    assert np.all(ids == 1)


def test_sinusoid_column_center_tracks_y():
    """Column centre must follow x_center + A*sin(2*pi*y/wavelength)."""
    grid = _grid()
    geom = LateralGeometry(regions=(
        LateralRegion(name="scaffold", material=_mat(),
                      shape=SinusoidColumnShape(
                          x_center=50e-9, amplitude=20e-9,
                          wavelength=150e-9, width=20e-9,
                          y_min=0.0, y_max=300e-9)),
    ))
    ids = build_region_id_field(grid, geom)

    for j, y in enumerate(grid.y):
        painted = np.flatnonzero(ids[j, :] == 0)
        assert painted.size > 0, f"row j={j} (y={y:.3e}) painted nothing"
        centre = 50e-9 + 20e-9 * np.sin(2.0 * np.pi * y / 150e-9)
        measured = 0.5 * (grid.x[painted[0]] + grid.x[painted[-1]])
        # one cell of slack: the centre lands between nodes in general
        assert abs(measured - centre) <= (grid.x[1] - grid.x[0])


def test_sinusoid_column_respects_y_window():
    grid = _grid()
    geom = LateralGeometry(regions=(
        LateralRegion(name="scaffold", material=_mat(),
                      shape=SinusoidColumnShape(
                          x_center=50e-9, amplitude=0.0,
                          wavelength=150e-9, width=20e-9,
                          y_min=100e-9, y_max=200e-9)),
    ))
    ids = build_region_id_field(grid, geom)
    outside = (grid.y < 100e-9 - 1e-15) | (grid.y > 200e-9 + 1e-15)
    assert np.all(ids[outside, :] == -1)


def test_zero_width_column_paints_nothing():
    grid = _grid()
    geom = LateralGeometry(regions=(
        LateralRegion(name="degenerate", material=_mat(),
                      shape=SinusoidColumnShape(
                          x_center=50e-9, amplitude=0.0,
                          wavelength=150e-9, width=0.0,
                          y_min=0.0, y_max=300e-9)),
    ))
    ids = build_region_id_field(grid, geom)
    assert np.all(ids == -1)


def test_region_outside_domain_raises():
    grid = _grid()
    geom = LateralGeometry(regions=(
        LateralRegion(name="oops", material=_mat(),
                      shape=BoxShape(x_min=200e-9, x_max=300e-9,
                                     y_min=0.0, y_max=300e-9)),
    ))
    with pytest.raises(ValueError, match="paints no nodes"):
        build_region_id_field(grid, geom)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd perovskite-sim
mkdir -p tests/unit/twod
pytest tests/unit/twod/test_lateral_geometry.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'perovskite_sim.twod.lateral_geometry'`.

- [ ] **Step 3: Write the implementation**

Create `perovskite_sim/twod/lateral_geometry.py`:

```python
"""Lateral (in-plane) material regions for the 2D solver.

The 2D solver's material arrays are built by extruding the 1D per-y layer
stack along x. This module supplies the second ingredient needed for genuine
in-plane heterogeneity: a rasterised region-ID field that says, for every
node ``(j, i)``, whether it belongs to the base 1D stack (``-1``) or to one
of the declared lateral regions (``0 .. len(regions)-1``).

Approach follows ChargeFabrica (Sachsenweger, Torre Cachafeiro & Tress,
Materials Futures, doi:10.1088/2752-5724/ae27e9): build an integer material
map first, then look material parameters up through it. Keeping the two
steps separate means the geometry is testable without a solver.

Everything here is a pure function of ``Grid2D`` plus frozen descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from perovskite_sim.models.material import MaterialParams
from perovskite_sim.twod.grid_2d import Grid2D

# Node-coordinate comparisons are inclusive to this absolute tolerance so a
# boundary declared exactly on a node lands inside the region rather than
# falling through on a float round-trip.
_EDGE_TOL = 1e-15


@dataclass(frozen=True)
class BoxShape:
    """Axis-aligned rectangle in (x, y), metres. Bounds are inclusive."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class SinusoidColumnShape:
    """A column of constant ``width`` whose centre sweeps sinusoidally in x.

    ``x_center(y) = x_center + amplitude * sin(2*pi*y/wavelength + phase)``.
    Painted only for ``y_min <= y <= y_max``. This is the mesoporous-scaffold
    primitive: ``amplitude=0`` degenerates to a straight column.
    """
    x_center: float
    amplitude: float
    wavelength: float
    width: float
    y_min: float
    y_max: float
    phase: float = 0.0


Shape = BoxShape | SinusoidColumnShape


@dataclass(frozen=True)
class LateralRegion:
    """A named region carrying its own full ``MaterialParams``.

    ``material`` is a complete MaterialParams — parsed from YAML by the same
    ``config_loader.material_params_from_dict`` the layer stack uses, so the
    lateral schema can never drift from the layer schema.
    """
    name: str
    material: MaterialParams
    shape: Shape


@dataclass(frozen=True)
class LateralGeometry:
    """Container for the declared lateral regions.

    The default ``()`` is the Stage-A lateral-uniform device: every consumer
    must reduce to plain extrusion when ``regions`` is empty.
    """
    regions: tuple[LateralRegion, ...] = ()

    @property
    def is_empty(self) -> bool:
        return len(self.regions) == 0


def _box_mask(grid: Grid2D, shape: BoxShape) -> np.ndarray:
    in_x = ((grid.x >= shape.x_min - _EDGE_TOL) &
            (grid.x <= shape.x_max + _EDGE_TOL))
    in_y = ((grid.y >= shape.y_min - _EDGE_TOL) &
            (grid.y <= shape.y_max + _EDGE_TOL))
    return np.outer(in_y, in_x)


def _sinusoid_column_mask(grid: Grid2D, shape: SinusoidColumnShape) -> np.ndarray:
    if shape.width <= 0.0 or shape.wavelength <= 0.0:
        # Degenerate descriptors paint nothing rather than raising: a swept
        # study that walks width down to zero should land on the uniform
        # device, not crash.
        return np.zeros((grid.Ny, grid.Nx), dtype=bool)

    centre = shape.x_center + shape.amplitude * np.sin(
        2.0 * np.pi * grid.y / shape.wavelength + shape.phase
    )                                                    # (Ny,)
    half = shape.width / 2.0
    dist = np.abs(grid.x[None, :] - centre[:, None])      # (Ny, Nx)
    in_x = dist <= half + _EDGE_TOL

    in_y = ((grid.y >= shape.y_min - _EDGE_TOL) &
            (grid.y <= shape.y_max + _EDGE_TOL))          # (Ny,)
    return in_x & in_y[:, None]


def shape_mask(grid: Grid2D, shape: Shape) -> np.ndarray:
    """Boolean ``(Ny, Nx)`` mask of nodes owned by ``shape``."""
    if isinstance(shape, BoxShape):
        return _box_mask(grid, shape)
    if isinstance(shape, SinusoidColumnShape):
        return _sinusoid_column_mask(grid, shape)
    raise TypeError(f"unsupported lateral shape: {type(shape).__name__}")


def build_region_id_field(grid: Grid2D, geom: LateralGeometry) -> np.ndarray:
    """Rasterise ``geom`` onto ``grid``.

    Returns an ``(Ny, Nx)`` int32 array: ``-1`` = base 1D layer stack,
    ``k`` = ``geom.regions[k]``. Regions are painted in declaration order,
    so a later region overwrites an earlier one where they overlap.

    A region that paints zero nodes raises ``ValueError`` — silently
    dropping it would make a typoed coordinate look like a physics result.
    A zero-width or zero-wavelength ``SinusoidColumnShape`` is the one
    deliberate exception (see ``_sinusoid_column_mask``); declare
    ``LateralGeometry()`` if you want no regions at all.
    """
    ids = np.full((grid.Ny, grid.Nx), -1, dtype=np.int32)
    for k, region in enumerate(geom.regions):
        mask = shape_mask(grid, region.shape)
        degenerate = (
            isinstance(region.shape, SinusoidColumnShape)
            and (region.shape.width <= 0.0 or region.shape.wavelength <= 0.0)
        )
        if not mask.any() and not degenerate:
            raise ValueError(
                f"lateral region {region.name!r} paints no nodes on this grid "
                f"(x range {grid.x[0]:.3e}..{grid.x[-1]:.3e} m, "
                f"y range {grid.y[0]:.3e}..{grid.y[-1]:.3e} m) — check the "
                f"shape coordinates, or raise Nx/Ny so the feature is resolved"
            )
        ids[mask] = k
    return ids
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_lateral_geometry.py -v
```

Expected: 7 passed.

> **GATE 1 — geometry rasteriser**
> - `pytest tests/unit/twod/test_lateral_geometry.py -v` → 7 passed.
> - `LateralGeometry()` yields an all-`-1` field (nothing painted).
> - A region that paints nothing raises `ValueError`, not a silent no-op.
> - No import of `solver_2d`, `mol`, or `scipy` in `lateral_geometry.py`:
>   `grep -nE "solver_2d|from perovskite_sim.solver|import scipy" perovskite_sim/twod/lateral_geometry.py` → no output.

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/twod/lateral_geometry.py tests/unit/twod/test_lateral_geometry.py
git commit -m "feat(2d): add LateralGeometry region rasteriser

Pure geometry layer for in-plane material heterogeneity: rasterises box and
sinusoid-column regions onto Grid2D into an int32 region-ID field. No solver
coupling yet.

Constraint: must reduce to all-base (-1) for the empty geometry so the
existing extrude() path stays bit-identical
Rejected: per-region MaterialParams subset dicts | full MaterialParams reuses
the shared config_loader parser and cannot drift from the layer schema
Confidence: high
Scope-risk: narrow"
```

---

### Task 2: YAML parsing for `lateral_geometry`

**Files:**
- Modify: `perovskite_sim/models/config_loader.py`
- Modify: `perovskite_sim/models/device.py`
- Test: `tests/unit/twod/test_lateral_geometry.py` (append)

**Interfaces:**
- Consumes: `LateralGeometry`, `LateralRegion`, `BoxShape`, `SinusoidColumnShape` from Task 1; `config_loader.material_params_from_dict(d: dict) -> MaterialParams`.
- Produces:
  - `config_loader.load_lateral_geometry_from_yaml_block(block: Mapping | None) -> LateralGeometry`
  - `DeviceStack.lateral_geometry: LateralGeometry | None = None`

YAML schema:

```yaml
lateral_geometry:
  regions:
    - name: scaffold
      material:           # identical schema to a layer's material fields
        chi: 4.1
        Eg: 3.2
        Nc300: 1.0e27
        Nv300: 1.0e27
        mu_n: 1.0e-6
        mu_p: 1.0e-6
        eps_r: 35.0
        D_ion: 0.0
      shape:
        kind: sinusoid_column
        x_center: 50e-9
        amplitude: 20e-9
        wavelength: 60e-9
        width: 35e-9
        y_min: 100e-9
        y_max: 250e-9
```

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/twod/test_lateral_geometry.py`:

```python
from perovskite_sim.models.config_loader import (
    load_lateral_geometry_from_yaml_block,
)
from perovskite_sim.twod.lateral_geometry import BoxShape, SinusoidColumnShape


def test_yaml_block_none_is_empty_geometry():
    assert load_lateral_geometry_from_yaml_block(None).is_empty
    assert load_lateral_geometry_from_yaml_block({}).is_empty
    assert load_lateral_geometry_from_yaml_block({"regions": []}).is_empty


def test_yaml_block_parses_box_region():
    geom = load_lateral_geometry_from_yaml_block({
        "regions": [{
            "name": "slab",
            "material": {"chi": 4.1, "Eg": 3.2, "mu_n": 1e-6, "mu_p": 1e-6},
            "shape": {"kind": "box", "x_min": 3e-8, "x_max": 6e-8,
                      "y_min": 1e-7, "y_max": 2e-7},
        }]
    })
    assert len(geom.regions) == 1
    r = geom.regions[0]
    assert r.name == "slab"
    assert r.material.chi == pytest.approx(4.1)
    assert r.material.Eg == pytest.approx(3.2)
    assert isinstance(r.shape, BoxShape)
    assert r.shape.x_min == pytest.approx(3e-8)


def test_yaml_block_parses_sinusoid_column():
    geom = load_lateral_geometry_from_yaml_block({
        "regions": [{
            "name": "scaffold",
            "material": {"chi": 2.9, "Eg": 4.5},
            "shape": {"kind": "sinusoid_column", "x_center": 5e-8,
                      "amplitude": 2e-8, "wavelength": 6e-8, "width": 3.5e-8,
                      "y_min": 0.0, "y_max": 1e-6},
        }]
    })
    s = geom.regions[0].shape
    assert isinstance(s, SinusoidColumnShape)
    assert s.wavelength == pytest.approx(6e-8)
    assert s.phase == pytest.approx(0.0)


def test_yaml_unknown_region_key_raises():
    with pytest.raises(ValueError, match="unknown key"):
        load_lateral_geometry_from_yaml_block({
            "regions": [{
                "name": "slab", "material": {"chi": 4.1},
                "shape": {"kind": "box", "x_min": 0.0, "x_max": 1e-8,
                          "y_min": 0.0, "y_max": 1e-8},
                "typo_field": 1.0,
            }]
        })


def test_yaml_unknown_shape_kind_raises():
    with pytest.raises(ValueError, match="unknown shape kind"):
        load_lateral_geometry_from_yaml_block({
            "regions": [{
                "name": "slab", "material": {"chi": 4.1},
                "shape": {"kind": "trapezoid", "x_min": 0.0},
            }]
        })


def test_yaml_scientific_notation_strings_are_coerced():
    """PyYAML 1.1 returns bare 1e-9 as a str; the parser must cope."""
    geom = load_lateral_geometry_from_yaml_block({
        "regions": [{
            "name": "slab", "material": {"chi": 4.1},
            "shape": {"kind": "box", "x_min": "3e-8", "x_max": "6e-8",
                      "y_min": "0", "y_max": "1e-7"},
        }]
    })
    assert geom.regions[0].shape.x_min == pytest.approx(3e-8)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_lateral_geometry.py -k yaml -v
```

Expected: `ImportError: cannot import name 'load_lateral_geometry_from_yaml_block'`.

- [ ] **Step 3: Write the implementation**

Add to `perovskite_sim/models/config_loader.py` (near the existing `load_microstructure_from_yaml_block` import site, so the two lateral-feature parsers sit together):

```python
_LATERAL_REGION_KEYS = frozenset({"name", "material", "shape"})
_BOX_KEYS = frozenset({"kind", "x_min", "x_max", "y_min", "y_max"})
_SINUSOID_KEYS = frozenset({
    "kind", "x_center", "amplitude", "wavelength", "width",
    "y_min", "y_max", "phase",
})


def _f(d: Mapping[str, Any], key: str, default: float | None = None) -> float:
    """Read a float, tolerating PyYAML 1.1 returning bare `1e-9` as a str."""
    if key not in d:
        if default is None:
            raise ValueError(f"lateral_geometry shape missing required key {key!r}")
        return float(default)
    return float(d[key])


def _shape_from_dict(d: Mapping[str, Any]):
    from perovskite_sim.twod.lateral_geometry import BoxShape, SinusoidColumnShape

    kind = str(d.get("kind", "")).strip().lower()
    if kind == "box":
        unknown = set(d.keys()) - _BOX_KEYS
        if unknown:
            raise ValueError(
                f"lateral_geometry box shape unknown key(s): {sorted(unknown)}"
            )
        return BoxShape(
            x_min=_f(d, "x_min"), x_max=_f(d, "x_max"),
            y_min=_f(d, "y_min"), y_max=_f(d, "y_max"),
        )
    if kind == "sinusoid_column":
        unknown = set(d.keys()) - _SINUSOID_KEYS
        if unknown:
            raise ValueError(
                f"lateral_geometry sinusoid_column shape unknown key(s): "
                f"{sorted(unknown)}"
            )
        return SinusoidColumnShape(
            x_center=_f(d, "x_center"), amplitude=_f(d, "amplitude"),
            wavelength=_f(d, "wavelength"), width=_f(d, "width"),
            y_min=_f(d, "y_min"), y_max=_f(d, "y_max"),
            phase=_f(d, "phase", 0.0),
        )
    raise ValueError(
        f"lateral_geometry unknown shape kind {kind!r} "
        f"(expected 'box' or 'sinusoid_column')"
    )


def load_lateral_geometry_from_yaml_block(block: Mapping[str, Any] | None):
    """Parse a YAML ``lateral_geometry:`` block into a ``LateralGeometry``.

    ``None`` / ``{}`` / ``{regions: []}`` all return the empty geometry, so
    configs without the block are byte-identical on the extrusion path.

    Region ``material`` goes through ``material_params_from_dict`` — the same
    parser the layer stack uses — so the two schemas cannot drift (see
    the inline-device parser drift incidents, commit 5c9f2aa).
    """
    from perovskite_sim.twod.lateral_geometry import (
        LateralGeometry, LateralRegion,
    )

    if not block:
        return LateralGeometry()
    raw = block.get("regions") or ()
    regions: list[LateralRegion] = []
    for entry in raw:
        unknown = set(entry.keys()) - _LATERAL_REGION_KEYS
        if unknown:
            raise ValueError(
                f"lateral_geometry region unknown key(s): {sorted(unknown)}"
            )
        regions.append(LateralRegion(
            name=str(entry["name"]),
            material=material_params_from_dict(entry["material"]),
            shape=_shape_from_dict(entry["shape"]),
        ))
    return LateralGeometry(regions=tuple(regions))
```

Wire it into `load_device_from_yaml` beside the existing `microstructure` handling:

```python
    lateral_geometry = load_lateral_geometry_from_yaml_block(
        raw.get("lateral_geometry")
    )
```

and pass `lateral_geometry=lateral_geometry` into the `DeviceStack(...)` construction.

Add the field to `perovskite_sim/models/device.py` on `DeviceStack`, **after every existing field that has a default**, so positional construction elsewhere is unaffected:

```python
    lateral_geometry: "LateralGeometry | None" = None
```

with a `TYPE_CHECKING` import to avoid a circular import:

```python
if TYPE_CHECKING:  # pragma: no cover
    from perovskite_sim.twod.lateral_geometry import LateralGeometry
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_lateral_geometry.py -v
pytest tests/unit/models -v
```

Expected: all pass, including the pre-existing loader tests.

> **GATE 2 — YAML round-trip**
> - `pytest tests/unit/twod/test_lateral_geometry.py -v` → 13 passed.
> - `pytest tests/unit/models -q` → no regressions.
> - Every shipped config still loads and `stack.lateral_geometry` is `LateralGeometry()`:
>   ```bash
>   python3 -c "
>   from pathlib import Path
>   from perovskite_sim.models.config_loader import load_device_from_yaml
>   n=0
>   for p in sorted(Path('configs').rglob('*.yaml')):
>       s = load_device_from_yaml(str(p))
>       g = getattr(s, 'lateral_geometry', None)
>       assert g is None or g.is_empty, p
>       n += 1
>   print(f'{n} configs, all lateral-uniform')
>   "
>   ```
> - Unknown keys raise rather than silently dropping (both region and shape level).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/models/config_loader.py perovskite_sim/models/device.py \
        tests/unit/twod/test_lateral_geometry.py
git commit -m "feat(config): parse a lateral_geometry block into DeviceStack

Region materials go through the shared material_params_from_dict parser, so
the lateral schema cannot drift from the layer schema. Strict-key validation
on both region and shape so typos surface instead of vanishing.

Constraint: absent block must yield LateralGeometry() and load byte-identically
Rejected: a reduced per-region field subset | would reintroduce the parser
drift that killed the Nc300 DOS fold on the inline path (commit 5c9f2aa)
Confidence: high
Scope-risk: narrow"
```

---

### Task 3: Paint material fields in `build_material_arrays_2d`

Replace `extrude()` with a geometry-aware `paint()`. **This is the bit-identity crux of the whole plan.**

**Files:**
- Modify: `perovskite_sim/twod/solver_2d.py:140-175` (signature + `extrude`)
- Test: `tests/unit/twod/test_lateral_geometry.py` (append)

**Interfaces:**
- Consumes: `build_region_id_field` (Task 1); `DeviceStack.lateral_geometry` (Task 2).
- Produces: `build_material_arrays_2d(grid, stack, ustruct, *, lateral_bc="periodic", P_ion_static_1d=None, lateral_geometry=None) -> MaterialArrays2D`. When `lateral_geometry` is `None` it falls back to `stack.lateral_geometry`, then to `LateralGeometry()`.

Painted fields and their `MaterialParams` sources:

| `MaterialArrays2D` field | `MaterialParams` attribute |
|---|---|
| `eps_r` | `eps_r` |
| `chi` | `chi` |
| `Eg` | `Eg` |
| `N_A` | `N_A` |
| `N_D` | `N_D` |
| `D_n` | `mu_n * V_T` |
| `D_p` | `mu_p * V_T` |
| `tau_n`, `tau_p` | `tau_n`, `tau_p` |
| `B_rad`, `C_n`, `C_p` | same names |
| `ni` | `sqrt(Nc300 * Nv300 * exp(-Eg / V_T))` |
| `n1`, `p1` | `n1`, `p1` |
| `G_optical` | forced to `0.0` inside a region (a scaffold does not absorb) |

**Deliberately NOT painted** (documented limitation, restated in the module docstring):
`A_star_n` / `A_star_p` (Richardson constants stay at the extruded 1D values — the lateral TE cap in Task 5 reads them, and a per-region Richardson constant is not calibrated anywhere); `P_ion0_2d` / `P_ion_static` (2D has no mobile ions until the follow-on plan); the DOS band-potential fold (built into the 1D `chi`/`Eg` arrays upstream, so a painted region's `chi` is a *raw* affinity — regions must declare `Nc300`/`Nv300` and get the fold applied in the same pass or not at all; this plan applies **no** fold to painted regions and asserts it in the docstring).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/twod/test_lateral_geometry.py`:

```python
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import build_material_arrays_2d


def _nip_grid_and_stack():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers = [Layer(thickness=L.thickness, N=10)
              for L in electrical_layers(stack)]
    grid = build_grid_2d(layers, lateral_length=100e-9, Nx=20,
                         lateral_uniform=True)
    return grid, stack


_PAINTED_FIELDS = ("eps_r", "chi", "Eg", "N_A", "N_D", "D_n", "D_p",
                   "tau_n", "tau_p", "B_rad", "C_n", "C_p", "ni",
                   "n1", "p1", "G_optical")


def test_empty_geometry_is_bit_identical_to_extrusion():
    """The whole plan rests on this: no geometry -> byte-identical arrays."""
    grid, stack = _nip_grid_and_stack()
    base = build_material_arrays_2d(grid, stack, Microstructure())
    with_geom = build_material_arrays_2d(
        grid, stack, Microstructure(), lateral_geometry=LateralGeometry()
    )
    for field in _PAINTED_FIELDS:
        a, b = getattr(base, field), getattr(with_geom, field)
        assert np.array_equal(a, b), f"{field} not bit-identical"


def test_painted_region_overrides_chi_and_leaves_base_alone():
    grid, stack = _nip_grid_and_stack()
    scaffold = MaterialParams(chi=2.9, Eg=4.5, mu_n=1e-7, mu_p=1e-7,
                              eps_r=35.0, Nc300=1e27, Nv300=1e27)
    geom = LateralGeometry(regions=(
        LateralRegion(name="scaffold", material=scaffold,
                      shape=BoxShape(x_min=40e-9, x_max=60e-9,
                                     y_min=100e-9, y_max=200e-9)),
    ))
    base = build_material_arrays_2d(grid, stack, Microstructure())
    mat = build_material_arrays_2d(grid, stack, Microstructure(),
                                   lateral_geometry=geom)
    ids = build_region_id_field(grid, geom)

    assert np.allclose(mat.chi[ids == 0], 2.9)
    assert np.allclose(mat.Eg[ids == 0], 4.5)
    # Outside the region nothing moved.
    assert np.array_equal(mat.chi[ids == -1], base.chi[ids == -1])
    assert np.array_equal(mat.Eg[ids == -1], base.Eg[ids == -1])


def test_painted_region_does_not_absorb_light():
    grid, stack = _nip_grid_and_stack()
    scaffold = MaterialParams(chi=2.9, Eg=4.5, mu_n=1e-7, mu_p=1e-7,
                              eps_r=35.0, Nc300=1e27, Nv300=1e27)
    geom = LateralGeometry(regions=(
        LateralRegion(name="scaffold", material=scaffold,
                      shape=BoxShape(x_min=40e-9, x_max=60e-9,
                                     y_min=100e-9, y_max=200e-9)),
    ))
    mat = build_material_arrays_2d(grid, stack, Microstructure(),
                                   lateral_geometry=geom)
    ids = build_region_id_field(grid, geom)
    assert np.all(mat.G_optical[ids == 0] == 0.0)


def test_chi_varies_along_x_inside_the_region_rows():
    """The point of the whole exercise: chi is a function of x."""
    grid, stack = _nip_grid_and_stack()
    scaffold = MaterialParams(chi=2.9, Eg=4.5, mu_n=1e-7, mu_p=1e-7,
                              eps_r=35.0, Nc300=1e27, Nv300=1e27)
    geom = LateralGeometry(regions=(
        LateralRegion(name="scaffold", material=scaffold,
                      shape=BoxShape(x_min=40e-9, x_max=60e-9,
                                     y_min=100e-9, y_max=200e-9)),
    ))
    mat = build_material_arrays_2d(grid, stack, Microstructure(),
                                   lateral_geometry=geom)
    ids = build_region_id_field(grid, geom)
    rows = np.flatnonzero((ids == 0).any(axis=1))
    assert rows.size > 0
    for j in rows:
        assert np.unique(mat.chi[j, :]).size >= 2, (
            f"row j={j} still laterally uniform"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_lateral_geometry.py -k "bit_identical or painted or varies" -v
```

Expected: `TypeError: build_material_arrays_2d() got an unexpected keyword argument 'lateral_geometry'`.

- [ ] **Step 3: Write the implementation**

In `perovskite_sim/twod/solver_2d.py`, extend the signature:

```python
def build_material_arrays_2d(
    grid: Grid2D,
    stack: DeviceStack,
    ustruct: Microstructure,
    *,
    lateral_bc: str = "periodic",
    P_ion_static_1d: np.ndarray | None = None,
    lateral_geometry: "LateralGeometry | None" = None,
) -> MaterialArrays2D:
```

Immediately after `mat1d = build_material_arrays_1d(grid.y, stack)` and `Nx, Ny = grid.Nx, grid.Ny`, resolve the geometry and build the ID field:

```python
    from perovskite_sim.twod.lateral_geometry import (
        LateralGeometry, build_region_id_field,
    )
    if lateral_geometry is None:
        lateral_geometry = getattr(stack, "lateral_geometry", None) \
                           or LateralGeometry()
    region_ids = build_region_id_field(grid, lateral_geometry)
    _regions = lateral_geometry.regions
```

Replace `extrude` with a painting version. **Keep the name `extrude` for the fields that must never be painted** so the diff shows intent, and add `paint` alongside:

```python
    def extrude(v_1d: np.ndarray) -> np.ndarray:
        """Broadcast a 1D per-y array to (Ny, Nx), returning a writeable copy.

        Used for fields that are deliberately NOT laterally painted:
        Richardson constants (no per-region calibration exists) and the frozen
        ion background (2D has no mobile ions).
        """
        return np.broadcast_to(v_1d[:, None], (Ny, Nx)).copy()

    def paint(v_1d: np.ndarray, getter) -> np.ndarray:
        """Extrude the 1D field, then overwrite declared lateral regions.

        ``getter(MaterialParams) -> float | None``. Returning ``None`` leaves
        that region at the extruded base value (a region that declines to
        specify a property inherits the underlying layer's).

        With no regions this is exactly ``extrude`` — the returned array is
        built by the same broadcast+copy and never touched again, which is
        what makes the empty-geometry path bit-identical.
        """
        field = np.broadcast_to(v_1d[:, None], (Ny, Nx)).copy()
        for k, region in enumerate(_regions):
            value = getter(region.material)
            if value is None:
                continue
            field[region_ids == k] = float(value)
        return field
```

Then convert the painted fields. `V_T` is needed for `D_n`/`D_p`, so hoist `V_T = float(mat1d.V_T_device)` above this block (it is currently read further down — move it, do not duplicate):

```python
    V_T = float(mat1d.V_T_device)

    eps_r = paint(mat1d.eps_r, lambda m: m.eps_r)
    N_A   = paint(mat1d.N_A,   lambda m: m.N_A)
    N_D   = paint(mat1d.N_D,   lambda m: m.N_D)
    chi   = paint(mat1d.chi,   lambda m: m.chi)
    Eg    = paint(mat1d.Eg,    lambda m: m.Eg)

    def _region_ni(m):
        if not (m.Nc300 and m.Nv300 and m.Eg):
            return None
        return float(np.sqrt(m.Nc300 * m.Nv300 * np.exp(-m.Eg / V_T)))

    ni = paint(np.sqrt(mat1d.ni_sq), _region_ni)

    n1 = paint(mat1d.n1, lambda m: getattr(m, "n1", None))
    p1 = paint(mat1d.p1, lambda m: getattr(m, "p1", None))
    B_rad = paint(mat1d.B_rad, lambda m: m.B_rad)
    C_n_2d = paint(mat1d.C_n, lambda m: m.C_n)
    C_p_2d = paint(mat1d.C_p, lambda m: m.C_p)
```

For `D_n` / `D_p`, keep the existing `_diffusion_per_node` call and paint on top:

```python
    D_n_node_1d, D_p_node_1d = _diffusion_per_node(grid.y, stack, V_T)
    D_n = paint(D_n_node_1d,
                lambda m: (m.mu_n * V_T) if m.mu_n else None)
    D_p = paint(D_p_node_1d,
                lambda m: (m.mu_p * V_T) if m.mu_p else None)
```

For `G_optical`, keep the existing three-branch construction, then zero the regions:

```python
    # A declared lateral region is a scaffold / second phase, not an absorber:
    # it neither generates nor attenuates in this model. Column-resolved
    # optics (each x-column running its own Beer-Lambert / TMM through the
    # painted absorber mask) is deliberately out of scope for this plan.
    for k in range(len(_regions)):
        G_optical[region_ids == k] = 0.0
```

For `tau_n` / `tau_p`, paint **before** `build_tau_field` so grain boundaries still win:

```python
    tau_n_base = paint(tau_n_1d, lambda m: m.tau_n)
    tau_p_base = paint(tau_p_1d, lambda m: m.tau_p)
    tau_n, tau_p = build_tau_field(
        grid, ustruct,
        tau_n_bulk_per_y=tau_n_1d,
        tau_p_bulk_per_y=tau_p_1d,
        layer_role_per_y=layer_role_per_y,
    )
    # build_tau_field extrudes from the 1D bulk; re-apply the lateral paint
    # underneath the GB overrides so a GB inside a painted region still wins.
    gb_touched = np.zeros((Ny, Nx), dtype=bool)
    for gb in ustruct.grain_boundaries:
        mask_x = np.abs(grid.x - gb.x_position) < gb.width / 2.0
        mask_y = np.array([r == gb.layer_role for r in layer_role_per_y])
        gb_touched |= np.outer(mask_y, mask_x)
    tau_n = np.where(gb_touched, tau_n, tau_n_base)
    tau_p = np.where(gb_touched, tau_p, tau_p_base)
```

Finally add `region_ids` to `MaterialArrays2D` so later tasks and diagnostics can read it:

```python
    region_ids: np.ndarray        # (Ny, Nx) int32; -1 = base stack, k = region k
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_lateral_geometry.py -v
pytest tests/regression/test_twod_validation.py -v
pytest tests/regression/test_twod_microstructure.py -v
```

Expected: new tests pass; **both existing 2D regression files pass unchanged**.

> **GATE 3 — painting, with bit-identity as the hard barrier**
> - `test_empty_geometry_is_bit_identical_to_extrusion` passes for all 16 painted fields with `np.array_equal`.
> - `pytest tests/regression/test_twod_validation.py tests/regression/test_twod_microstructure.py -v` → all pass, **no baseline edits**. If any pinned value moved, the paint path is not neutral — stop and fix rather than re-pin.
> - Every shipped config still builds 2D arrays identical to `main`:
>   ```bash
>   git stash && python3 scripts/_dump_2d_arrays.py > /tmp/before.npz.txt
>   git stash pop && python3 scripts/_dump_2d_arrays.py > /tmp/after.npz.txt
>   diff /tmp/before.npz.txt /tmp/after.npz.txt && echo "BIT-IDENTICAL"
>   ```
>   (Write `scripts/_dump_2d_arrays.py` as a throwaway that hashes each painted field for `nip_MAPbI3` and `nip_MAPbI3_tmm`; delete it before commit. **`git stash` skips untracked files** — see `project_solarlab_suite_order_dependence` — so `git add -A` first or the new test file will not be stashed.)
> - A painted region genuinely makes `chi` vary along x (`test_chi_varies_along_x_inside_the_region_rows`).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/twod/solver_2d.py tests/unit/twod/test_lateral_geometry.py
git commit -m "feat(2d): paint lateral region materials over the extruded fields

build_material_arrays_2d gains lateral_geometry=; 16 per-node fields are now
extruded-then-painted. Empty geometry reduces to the original broadcast+copy,
verified bit-identical with np.array_equal on every field.

Constraint: existing 2D regression baselines must not move
Rejected: painting A_star_n/A_star_p | no per-region Richardson constant is
calibrated anywhere, and the TE cap reads them
Directive: painted chi is a RAW affinity — the DOS band-potential fold is NOT
applied to regions. Do not mix a folded base stack with an unfolded region
without reworking the fold to run after painting.
Not-tested: painted regions combined with band_grading or interface_tunneling
Confidence: high
Scope-risk: moderate"
```

---

### Task 4: Detect lateral heterointerface faces

**Files:**
- Create: `perovskite_sim/twod/interface_faces_2d.py`
- Test: `tests/unit/twod/test_interface_faces_2d.py`

**Interfaces:**
- Produces:
  - `detect_interface_x_faces(chi, Eg, *, threshold=0.05) -> tuple[np.ndarray, np.ndarray]` — two boolean `(Ny, Nx-1)` masks `(cb_mask, vb_mask)` marking x-faces whose conduction- / valence-band offset magnitude exceeds `threshold` eV.
  - `detect_interface_y_faces_2d(chi, Eg, *, threshold=0.05) -> tuple[np.ndarray, np.ndarray]` — same for `(Ny-1, Nx)` y-faces. Replaces the column-0 scalar assumption at `continuity_2d.py:105-107`.
  - `dual_cell_widths_2d(x, y, lateral_bc) -> tuple[np.ndarray, np.ndarray]` — `(hx_cell (Nx,), hy_cell (Ny,))`, matching the convention already inlined in `continuity_2d.py` so the two cannot drift.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/twod/test_interface_faces_2d.py`:

```python
import numpy as np

from perovskite_sim.twod.interface_faces_2d import (
    detect_interface_x_faces,
    detect_interface_y_faces_2d,
    dual_cell_widths_2d,
)


def test_uniform_fields_detect_nothing():
    chi = np.full((6, 5), 3.9)
    Eg = np.full((6, 5), 1.6)
    cb_x, vb_x = detect_interface_x_faces(chi, Eg)
    cb_y, vb_y = detect_interface_y_faces_2d(chi, Eg)
    assert cb_x.shape == (6, 4) and vb_x.shape == (6, 4)
    assert cb_y.shape == (5, 5) and vb_y.shape == (5, 5)
    assert not cb_x.any() and not vb_x.any()
    assert not cb_y.any() and not vb_y.any()


def test_x_face_cb_offset_detected_at_the_right_column():
    chi = np.full((6, 5), 3.9)
    chi[:, 3:] = 2.9                       # 1.0 eV step between i=2 and i=3
    Eg = np.full((6, 5), 1.6)
    cb_x, _ = detect_interface_x_faces(chi, Eg)
    expected = np.zeros((6, 4), dtype=bool)
    expected[:, 2] = True                  # face 2 joins node 2 and node 3
    assert np.array_equal(cb_x, expected)


def test_x_face_vb_offset_detected_independently_of_cb():
    """chi constant, Eg stepped -> VB offset only, CB clean."""
    chi = np.full((6, 5), 3.9)
    Eg = np.full((6, 5), 1.6)
    Eg[:, 3:] = 2.6                        # 1.0 eV VB step, zero CB step
    cb_x, vb_x = detect_interface_x_faces(chi, Eg)
    assert not cb_x.any()
    expected = np.zeros((6, 4), dtype=bool)
    expected[:, 2] = True
    assert np.array_equal(vb_x, expected)


def test_offset_below_threshold_is_ignored():
    chi = np.full((6, 5), 3.9)
    chi[:, 3:] = 3.87                      # 0.03 eV < 0.05 eV threshold
    Eg = np.full((6, 5), 1.6)
    cb_x, _ = detect_interface_x_faces(chi, Eg)
    assert not cb_x.any()


def test_y_face_detection_is_per_column_not_column_zero():
    """A step present only in column 4 must be detected only in column 4."""
    chi = np.full((6, 5), 3.9)
    chi[3:, 4] = 2.9
    Eg = np.full((6, 5), 1.6)
    cb_y, _ = detect_interface_y_faces_2d(chi, Eg)
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 4] = True
    assert np.array_equal(cb_y, expected)


def test_dual_cell_widths_neumann_sum_to_domain():
    x = np.linspace(0.0, 1e-7, 6)
    y = np.linspace(0.0, 3e-7, 7)
    hx, hy = dual_cell_widths_2d(x, y, "neumann")
    assert hx.shape == (6,) and hy.shape == (7,)
    assert np.isclose(hx.sum(), x[-1] - x[0])
    assert np.isclose(hy.sum(), y[-1] - y[0])


def test_dual_cell_widths_periodic_matches_continuity_convention():
    """Periodic boundary cells share the wrap face (see poisson_2d)."""
    x = np.linspace(0.0, 1e-7, 6)
    y = np.linspace(0.0, 3e-7, 7)
    hx, _ = dual_cell_widths_2d(x, y, "periodic")
    dx = np.diff(x)
    assert np.isclose(hx[0], dx[0] / 2 + dx[-1] / 2)
    assert np.isclose(hx[-1], dx[-1] / 2 + dx[-2] / 2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_interface_faces_2d.py -v
```

Expected: `ModuleNotFoundError: No module named 'perovskite_sim.twod.interface_faces_2d'`.

- [ ] **Step 3: Write the implementation**

Create `perovskite_sim/twod/interface_faces_2d.py`:

```python
"""Heterointerface face detection on the 2D grid.

Stage A detected interfaces only on y-faces, and read the band offset from
column 0 (``continuity_2d.py:105-107``) because ``chi`` was laterally
uniform by construction. Once ``lateral_geometry`` can paint a region into
part of a row, both assumptions break: offsets exist on x-faces, and a
y-face offset can differ between columns.

These helpers return full 2D boolean face masks so the TE cap and the
interface-SRH channel can be applied without any column-0 shortcut.
"""
from __future__ import annotations

import numpy as np

# Matches the 1D convention in physics/continuity.py: below ~2 kT at 300 K a
# band step is not a barrier worth capping, and treating it as one costs a
# vectorised exp() on every RHS call for no physics.
_DEFAULT_OFFSET_THRESHOLD_EV = 0.05


def _band_edges(chi: np.ndarray, Eg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (E_C-proxy, E_V-proxy) in the sign convention the SG flux uses.

    ``continuity_2d`` drives electrons with ``phi_n = phi + chi`` and holes
    with ``phi_p = phi + chi + Eg``; the offsets that matter are differences
    of exactly those two quantities.
    """
    return chi, chi + Eg


def detect_interface_x_faces(
    chi: np.ndarray,
    Eg: np.ndarray,
    *,
    threshold: float = _DEFAULT_OFFSET_THRESHOLD_EV,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(Ny, Nx-1)`` masks of x-faces with a CB / VB offset.

    Face ``i`` joins node ``i`` and node ``i+1``.
    """
    ec, ev = _band_edges(chi, Eg)
    dEc = ec[:, :-1] - ec[:, 1:]
    dEv = ev[:, :-1] - ev[:, 1:]
    return np.abs(dEc) > threshold, np.abs(dEv) > threshold


def detect_interface_y_faces_2d(
    chi: np.ndarray,
    Eg: np.ndarray,
    *,
    threshold: float = _DEFAULT_OFFSET_THRESHOLD_EV,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(Ny-1, Nx)`` masks of y-faces with a CB / VB offset.

    Face ``j`` joins node ``j`` and node ``j+1``. Unlike the Stage-A loop
    this is evaluated per column, so a region painted into part of a row is
    seen where it actually is.
    """
    ec, ev = _band_edges(chi, Eg)
    dEc = ec[:-1, :] - ec[1:, :]
    dEv = ev[:-1, :] - ev[1:, :]
    return np.abs(dEc) > threshold, np.abs(dEv) > threshold


def dual_cell_widths_2d(
    x: np.ndarray, y: np.ndarray, lateral_bc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node dual-cell widths ``(hx_cell, hy_cell)``.

    Mirrors the construction currently inlined in ``continuity_rhs_2d`` so
    the interface source terms are normalised by exactly the same widths the
    divergence uses. Any change here must be mirrored there (or better,
    ``continuity_rhs_2d`` should be refactored to call this).

    Note the y convention matches ``continuity_2d``: ``hy_cell[0] = dy[0]/2``
    (half-cell at the contact), which deliberately differs from the 1D
    ``dual_cell_faces`` convention ``dx_cell[0] = dx[0]``.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    Nx, Ny = x.size, y.size

    hx_cell = np.empty(Nx)
    if lateral_bc == "periodic":
        hx_cell[0] = dx[0] / 2.0 + dx[-1] / 2.0
        hx_cell[-1] = dx[-1] / 2.0 + dx[-2] / 2.0
        hx_cell[1:-1] = (dx[:-1][1:] + dx[1:][:-1]) / 2.0 \
            if Nx > 3 else dx[0]
    else:
        hx_cell[0] = dx[0] / 2.0
        hx_cell[-1] = dx[-1] / 2.0
        hx_cell[1:-1] = (dx[:-1] + dx[1:])[: Nx - 2] / 2.0

    hy_cell = np.empty(Ny)
    hy_cell[0] = dy[0] / 2.0
    hy_cell[-1] = dy[-1] / 2.0
    hy_cell[1:-1] = (dy[:-1] + dy[1:])[: Ny - 2] / 2.0
    return hx_cell, hy_cell
```

**Before writing the `hx_cell` interior branch, open `continuity_2d.py` around line 128 and copy the exact expression it uses.** The version above is a reconstruction; the gate below checks it against the real one, and if they disagree the file in `continuity_2d.py` is authoritative.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_interface_faces_2d.py -v
```

Expected: 7 passed.

> **GATE 4 — face detection**
> - `pytest tests/unit/twod/test_interface_faces_2d.py -v` → 7 passed.
> - Uniform `chi`/`Eg` detect zero faces in both directions (no spurious interfaces on every existing config).
> - `dual_cell_widths_2d` reproduces the widths `continuity_rhs_2d` computes inline, to machine precision, on a `nip_MAPbI3` grid for both `lateral_bc` values:
>   ```bash
>   pytest tests/unit/twod/test_interface_faces_2d.py::test_dual_cell_widths_periodic_matches_continuity_convention -v
>   ```
>   Plus a direct cross-check against the inline construction — if they differ, fix `interface_faces_2d.py` to match `continuity_2d.py`, not the reverse.
> - CB and VB detection are independent (a pure-`Eg` step marks VB only).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/twod/interface_faces_2d.py tests/unit/twod/test_interface_faces_2d.py
git commit -m "feat(2d): detect heterointerface faces per column in both directions

Replaces the Stage-A assumption that chi varies in y only and can be read
from column 0. Returns full (Ny, Nx-1) / (Ny-1, Nx) boolean masks for CB and
VB offsets, plus the dual-cell widths the interface source terms need.

Constraint: dual_cell_widths_2d must reproduce the widths continuity_rhs_2d
computes inline, or interface sources land on the wrong normalisation
Not-tested: non-uniform (tanh-clustered) lateral grids — every current caller
passes lateral_uniform=True
Confidence: high
Scope-risk: narrow"
```

---

### Task 5: Thermionic-emission cap on lateral faces

**Files:**
- Modify: `perovskite_sim/twod/continuity_2d.py:100-118`
- Modify: `perovskite_sim/twod/solver_2d.py` (cache the four masks on `MaterialArrays2D`)
- Test: `tests/unit/twod/test_interface_faces_2d.py` (append)

**Interfaces:**
- Consumes: `detect_interface_x_faces`, `detect_interface_y_faces_2d` (Task 4).
- Produces: `MaterialArrays2D` gains `iface_cb_x`, `iface_vb_x` (`(Ny, Nx-1)` bool), `iface_cb_y`, `iface_vb_y` (`(Ny-1, Nx)` bool). `continuity_rhs_2d` gains keyword-only `iface_cb_x=None, iface_vb_x=None, iface_cb_y=None, iface_vb_y=None`; when all are `None` the existing `interface_y_faces` loop runs unchanged.

The physics is the same Richardson-Dushman bound already applied on y-faces — `J_TE = A* T² (n_L e^{-max(ΔE,0)/V_T} − n_R e^{-max(−ΔE,0)/V_T})`, and the SG flux is capped to `min(|J_SG|, |J_TE|)` keeping the SG sign.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/twod/test_interface_faces_2d.py`:

```python
import pytest

from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.models.material import MaterialParams
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.lateral_geometry import (
    BoxShape, LateralGeometry, LateralRegion,
)
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import (
    assemble_rhs_2d, build_material_arrays_2d,
)


def _grid_stack(lateral_length=100e-9, Nx=20):
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers = [Layer(thickness=L.thickness, N=10)
              for L in electrical_layers(stack)]
    return build_grid_2d(layers, lateral_length=lateral_length, Nx=Nx,
                         lateral_uniform=True), stack


def _scaffold_geometry():
    scaffold = MaterialParams(chi=2.9, Eg=4.5, mu_n=1e-7, mu_p=1e-7,
                              eps_r=35.0, Nc300=1e27, Nv300=1e27,
                              tau_n=1e-9, tau_p=1e-9)
    return LateralGeometry(regions=(
        LateralRegion(name="scaffold", material=scaffold,
                      shape=BoxShape(x_min=40e-9, x_max=60e-9,
                                     y_min=150e-9, y_max=300e-9)),
    ))


def test_uniform_device_has_no_lateral_interface_faces():
    grid, stack = _grid_stack()
    mat = build_material_arrays_2d(grid, stack, Microstructure())
    assert not mat.iface_cb_x.any()
    assert not mat.iface_vb_x.any()


def test_painted_scaffold_creates_lateral_interface_faces():
    grid, stack = _grid_stack()
    mat = build_material_arrays_2d(grid, stack, Microstructure(),
                                   lateral_geometry=_scaffold_geometry())
    assert mat.iface_cb_x.any(), "1.0 eV lateral CB step went undetected"
    assert mat.iface_vb_x.any()


def test_rhs_is_finite_at_a_lateral_heterojunction():
    """Without the lateral TE cap the SG flux across a 1 eV step in one cell
    overflows; this is the regression that motivates the cap."""
    grid, stack = _grid_stack()
    mat = build_material_arrays_2d(grid, stack, Microstructure(),
                                   lateral_geometry=_scaffold_geometry())
    Ny, Nx = grid.Ny, grid.Nx
    n0 = np.full((Ny, Nx), 1e16)
    p0 = np.full((Ny, Nx), 1e16)
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt = assemble_rhs_2d(0.0, y0, mat, 0.0)
    assert np.all(np.isfinite(dydt))


def test_lateral_te_cap_bounds_the_flux():
    """With the cap on, |dn/dt| at the scaffold edge must be far below the
    uncapped SG magnitude across a 1 eV single-cell step."""
    grid, stack = _grid_stack()
    mat = build_material_arrays_2d(grid, stack, Microstructure(),
                                   lateral_geometry=_scaffold_geometry())
    Ny, Nx = grid.Ny, grid.Nx
    n0 = np.full((Ny, Nx), 1e16)
    p0 = np.full((Ny, Nx), 1e16)
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dn = assemble_rhs_2d(0.0, y0, mat, 0.0)[: Ny * Nx].reshape((Ny, Nx))
    # Uncapped SG across a 1 eV step in ~5 nm reaches ~1e31 1/s (the value the
    # Stage-A y-cap docstring quotes). Capped must be many orders below.
    assert np.max(np.abs(dn)) < 1e28


def test_disabled_path_is_bit_identical():
    """A device with no lateral geometry must produce exactly the Stage-A RHS."""
    grid, stack = _grid_stack()
    mat_a = build_material_arrays_2d(grid, stack, Microstructure())
    mat_b = build_material_arrays_2d(grid, stack, Microstructure(),
                                     lateral_geometry=LateralGeometry())
    Ny, Nx = grid.Ny, grid.Nx
    y0 = np.concatenate([np.full(Ny * Nx, 1e16), np.full(Ny * Nx, 1e16)])
    assert np.array_equal(assemble_rhs_2d(0.0, y0, mat_a, 0.3),
                          assemble_rhs_2d(0.0, y0, mat_b, 0.3))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_interface_faces_2d.py -k "lateral or rhs or disabled" -v
```

Expected: `AttributeError: 'MaterialArrays2D' object has no attribute 'iface_cb_x'`.

- [ ] **Step 3: Write the implementation**

In `solver_2d.py`, after `chi` and `Eg` are painted (Task 3), compute and store the masks:

```python
    from perovskite_sim.twod.interface_faces_2d import (
        detect_interface_x_faces, detect_interface_y_faces_2d,
    )
    iface_cb_x, iface_vb_x = detect_interface_x_faces(chi, Eg)
    iface_cb_y, iface_vb_y = detect_interface_y_faces_2d(chi, Eg)
```

Add the four fields to `MaterialArrays2D` and pass them into the `continuity_rhs_2d` call sites in `assemble_rhs_2d` (there are two — the main path and the field-mobility path; update both).

In `continuity_2d.py`, replace the `interface_y_faces` block with a mask-driven version. Keep the old loop as the fallback so the disabled path is byte-identical:

```python
    _use_masks = iface_cb_x is not None
    if _use_masks and chi is not None and Eg is not None and T is not None:
        T_sq = T * T
        ec = chi
        ev = chi + Eg

        # ---- y-faces (per column; no column-0 shortcut) --------------------
        if iface_cb_y.any():
            Jy_n = Jy_n.copy()
            dEc_y = ec[:-1, :] - ec[1:, :]
            left = n[:-1, :] * np.exp(-np.maximum(dEc_y, 0.0) / V_T)
            right = n[1:, :] * np.exp(-np.maximum(-dEc_y, 0.0) / V_T)
            J_te = A_star_n[:-1, :] * T_sq * (left - right)
            take = iface_cb_y & (np.abs(Jy_n) > np.abs(J_te))
            Jy_n[take] = J_te[take]
        if iface_vb_y.any():
            Jy_p = Jy_p.copy()
            dEv_y = ev[:-1, :] - ev[1:, :]
            left = p[:-1, :] * np.exp(-np.maximum(dEv_y, 0.0) / V_T)
            right = p[1:, :] * np.exp(-np.maximum(-dEv_y, 0.0) / V_T)
            J_te = A_star_p[:-1, :] * T_sq * (left - right)
            take = iface_vb_y & (np.abs(Jy_p) > np.abs(J_te))
            Jy_p[take] = J_te[take]

        # ---- x-faces (new) -------------------------------------------------
        if iface_cb_x.any():
            Jx_n = Jx_n.copy()
            dEc_x = ec[:, :-1] - ec[:, 1:]
            left = n[:, :-1] * np.exp(-np.maximum(dEc_x, 0.0) / V_T)
            right = n[:, 1:] * np.exp(-np.maximum(-dEc_x, 0.0) / V_T)
            J_te = A_star_n[:, :-1] * T_sq * (left - right)
            take = iface_cb_x & (np.abs(Jx_n) > np.abs(J_te))
            Jx_n[take] = J_te[take]
        if iface_vb_x.any():
            Jx_p = Jx_p.copy()
            dEv_x = ev[:, :-1] - ev[:, 1:]
            left = p[:, :-1] * np.exp(-np.maximum(dEv_x, 0.0) / V_T)
            right = p[:, 1:] * np.exp(-np.maximum(-dEv_x, 0.0) / V_T)
            J_te = A_star_p[:, :-1] * T_sq * (left - right)
            take = iface_vb_x & (np.abs(Jx_p) > np.abs(J_te))
            Jx_p[take] = J_te[take]

    elif interface_y_faces and chi is not None and Eg is not None and T is not None:
        # ... existing Stage-A loop, unchanged ...
```

**The periodic wrap face is not covered by `iface_cb_x`** (which has `Nx-1` columns for `Nx` nodes). Add an explicit assertion so the omission is loud rather than silent:

```python
    if _use_masks and lateral_bc == "periodic":
        wrap_dEc = float(np.max(np.abs(ec[:, 0] - ec[:, -1])))
        wrap_dEv = float(np.max(np.abs(ev[:, 0] - ev[:, -1])))
        if max(wrap_dEc, wrap_dEv) > 0.05:
            raise NotImplementedError(
                f"lateral geometry places a heterointerface on the periodic "
                f"wrap face (max |dE| = {max(wrap_dEc, wrap_dEv):.3f} eV). The "
                f"TE cap does not cover the wrap face. Either shift the region "
                f"away from x=0 / x=L_x, or use lateral_bc='neumann'."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_interface_faces_2d.py -v
pytest tests/regression/test_twod_validation.py -v
```

Expected: all pass, existing baselines untouched.

> **GATE 5 — lateral TE cap**
> - `test_disabled_path_is_bit_identical` passes with `np.array_equal` on the full RHS vector at `V=0.3`.
> - `pytest tests/regression/test_twod_validation.py -v` → all pass, **no baseline edits**.
> - `test_rhs_is_finite_at_a_lateral_heterojunction` passes — this is the whole point; without the cap a 1 eV step across one 5 nm cell produces `~1e31` and Radau dies immediately.
> - `test_lateral_te_cap_bounds_the_flux` → `max|dn/dt| < 1e28`.
> - A heterointerface on the periodic wrap face raises `NotImplementedError` with an actionable message, rather than silently skipping the cap.

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/twod/continuity_2d.py perovskite_sim/twod/solver_2d.py \
        tests/unit/twod/test_interface_faces_2d.py
git commit -m "feat(2d): cap thermionic emission on lateral heterointerface faces

Mask-driven TE capping in both directions, replacing the Stage-A y-only loop
that read the band offset from column 0. Disabled path (all masks None) falls
through to the original loop and is bit-identical.

Constraint: without this, SG flux across a 1 eV lateral step in one cell
reaches ~1e31 1/s and Radau fails at the first step
Rejected: silently skipping the periodic wrap face | raises NotImplementedError
with a fix suggestion instead
Not-tested: TE cap interaction with interface_tunneling (1D-scoped) on a
lateral face
Confidence: high
Scope-risk: moderate"
```

---

### Task 6: Port interface SRH to 2D vertical faces

**This closes an existing Stage-A gap** — `solver_2d.py:672` currently applies bulk recombination only, so `stack.interfaces` SRVs and `InterfaceDefect` never reach any 2D run. Do this before Task 7 so the lateral extension has a validated vertical counterpart.

**Files:**
- Create: `perovskite_sim/twod/interface_recomb_2d.py`
- Modify: `perovskite_sim/twod/solver_2d.py` (cache + `assemble_rhs_2d` wiring)
- Test: `tests/unit/twod/test_interface_recomb_2d.py`
- Test: `tests/regression/test_twod_lateral_heterogeneity.py` (parity case)

**Interfaces:**
- Consumes: `perovskite_sim.physics.recombination.interface_recombination(n, p, ni_sq, n1, p1, v_n, v_p) -> float` (scalar; returns `0.0` when `v_n <= 0` or `v_p <= 0`); `dual_cell_widths_2d` (Task 4).
- Produces:
  - `interface_recombination_vec(n, p, ni_sq, n1, p1, v_n, v_p) -> np.ndarray` — vectorised, same zero-velocity guard.
  - `apply_interface_recomb_y_2d(R, n, p, mat) -> np.ndarray` — returns a NEW `(Ny, Nx)` volumetric recombination array with the interface areal rate added at the interface rows, divided by `hy_cell` at that row.

The 1D channel is an **areal** rate `[m⁻² s⁻¹]`; the continuity equation wants a volumetric `[m⁻³ s⁻¹]`, so divide by the dual-cell width — the same normalisation the 1D solver uses, and the reason 1D SRVs are grid-referenced rather than face-physical (see the config gotcha in `perovskite-sim/CLAUDE.md`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/twod/test_interface_recomb_2d.py`:

```python
import numpy as np
import pytest

from perovskite_sim.physics.recombination import interface_recombination
from perovskite_sim.twod.interface_recomb_2d import interface_recombination_vec


def test_vec_matches_scalar_elementwise():
    rng = np.random.default_rng(20260729)
    n = 10.0 ** rng.uniform(14, 24, size=25)
    p = 10.0 ** rng.uniform(14, 24, size=25)
    ni_sq = np.full(25, 1e32)
    n1 = np.full(25, 1e10)
    p1 = np.full(25, 1e10)
    got = interface_recombination_vec(n, p, ni_sq, n1, p1, v_n=1.0, v_p=1.0)
    want = np.array([
        interface_recombination(float(n[k]), float(p[k]), float(ni_sq[k]),
                                float(n1[k]), float(p1[k]), 1.0, 1.0)
        for k in range(25)
    ])
    assert np.allclose(got, want, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("v_n,v_p", [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0),
                                     (-1.0, 1.0)])
def test_blocked_channel_gives_zero(v_n, v_p):
    """Mirrors the 1D F07 fix: one blocked capture channel blocks the cycle."""
    n = np.full(4, 1e20)
    p = np.full(4, 1e20)
    out = interface_recombination_vec(n, p, np.full(4, 1e32),
                                      np.zeros(4), np.zeros(4), v_n, v_p)
    assert np.all(out == 0.0)


def test_equilibrium_gives_zero_net_rate():
    n = np.full(4, 1e16)
    p = np.full(4, 1e16)
    ni_sq = n * p                       # exactly mass action
    out = interface_recombination_vec(n, p, ni_sq, np.zeros(4), np.zeros(4),
                                      v_n=1.0, v_p=1.0)
    assert np.allclose(out, 0.0, atol=1e-6)
```

And in `tests/regression/test_twod_lateral_heterogeneity.py`:

```python
"""2D lateral heterogeneity regression gates.

BLAS is pinned module-scoped: these cases drive the near-singular Radau
branch, whose outcome is thread-count dependent (see the wrong-branch
rejection note in perovskite-sim/CLAUDE.md).
"""
import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def _pin_blas():
    import numpy, scipy.linalg  # noqa: F401  — must load before limiting
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api="blas"):
        yield


def test_interface_srh_parity_2d_vs_1d_on_a_config_with_interfaces():
    """2D must reproduce 1D V_oc on a stack that DECLARES interface SRH.

    Before this task the 2D solver silently omitted the interface channel,
    so this config's 2D V_oc was too high. Gate: |dV_oc| <= 2 mV.
    """
    from perovskite_sim.discretization.grid import Layer
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d

    cfg = "configs/solarscale_nip_band_aligned_iface.yaml"
    stack = load_device_from_yaml(cfg)
    assert stack.interfaces, "test config must declare device.interfaces"

    n_layers = len(electrical_layers(stack))
    res_1d = run_jv_sweep(stack, N_grid=10 * n_layers + 1, V_max=1.2,
                          n_points=25)
    res_2d = run_jv_sweep_2d(stack, lateral_length=500e-9, Nx=4,
                             Ny_per_layer=10, V_max=1.2, V_step=0.05,
                             settle_t=1e-3)

    assert res_2d.metrics.voc_bracketed
    dV = abs(res_2d.metrics.V_oc - res_1d.metrics_fwd.V_oc)
    assert dV <= 2e-3, f"2D-vs-1D V_oc mismatch {dV*1e3:.2f} mV"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_interface_recomb_2d.py -v
pytest tests/regression/test_twod_lateral_heterogeneity.py -v
```

Expected: unit tests fail on missing module; the parity test fails with a `V_oc` mismatch far above 2 mV (that mismatch **is** the gap this task closes — record the measured value in the commit message).

- [ ] **Step 3: Write the implementation**

Create `perovskite_sim/twod/interface_recomb_2d.py`:

```python
"""Interface SRH recombination on the 2D grid.

Stage A computed recombination with ``total_recombination`` only — bulk SRH,
radiative and Auger. The interface channel that ``DeviceStack.interfaces``
declares (and that carries most of the band-offset sensitivity in 1D, see the
config gotcha in perovskite-sim/CLAUDE.md) never reached the 2D solver at
all. This module supplies it, in both directions.

The 1D channel is an AREAL rate [m^-2 s^-1]. Continuity wants a volumetric
source [m^-3 s^-1], so the areal rate is divided by the interface node's dual
cell width. This is why SolarLab SRVs are grid-referenced rather than
face-physical values: the same SRV on a finer mesh gives a different volumetric
source. Preserved deliberately for 1D/2D parity.
"""
from __future__ import annotations

import numpy as np


def interface_recombination_vec(
    n: np.ndarray, p: np.ndarray, ni_sq: np.ndarray,
    n1: np.ndarray, p1: np.ndarray,
    v_n: float, v_p: float,
) -> np.ndarray:
    """Vectorised form of ``physics.recombination.interface_recombination``.

    Returns an array of the same shape as ``n``. A non-positive ``v_n`` or
    ``v_p`` blocks the full SRH cycle and yields exactly zero (mirrors the
    1D F07 fix — a single zero velocity previously raised ZeroDivisionError).
    """
    if v_n <= 0.0 or v_p <= 0.0:
        return np.zeros_like(np.asarray(n, dtype=float))
    denom = (n + n1) / v_p + (p + p1) / v_n
    return (n * p - ni_sq) / denom


def apply_interface_recomb_y_2d(
    R: np.ndarray, n: np.ndarray, p: np.ndarray, mat,
) -> np.ndarray:
    """Add the vertical-interface SRH source to ``R``.

    Returns a NEW ``(Ny, Nx)`` array; ``R`` is not mutated (``assemble_rhs_2d``
    reuses cached arrays across RHS calls).

    ``mat.interface_srh_y`` is a tuple of
    ``(j_node, v_n, v_p, n1_row, p1_row)`` entries built at material-array
    time, where ``j_node`` is the interface's evaluation row.
    """
    entries = getattr(mat, "interface_srh_y", ())
    if not entries:
        return R
    out = R.copy()
    hy_cell = mat.hy_cell                     # (Ny,)
    ni_sq = mat.ni ** 2
    for (j, v_n, v_p, n1_row, p1_row) in entries:
        areal = interface_recombination_vec(
            n[j, :], p[j, :], ni_sq[j, :], n1_row, p1_row, v_n, v_p,
        )
        out[j, :] = out[j, :] + areal / hy_cell[j]
    return out
```

In `solver_2d.py`, build `interface_srh_y` from `electrical_interface_defects(stack)` and `stack.interfaces`. **Use `models/device.py:electrical_interface_defects()`, never index `stack.interface_defects` directly** — the raw tuple is full-layer-aligned and indexing it with the electrical interface number silently shifts every defect on substrate-prefixed stacks (the 2026-06 E10.1 glass regression). Cache `hy_cell` on `MaterialArrays2D` too. Then in `assemble_rhs_2d`, right after the `total_recombination` call:

```python
    from perovskite_sim.twod.interface_recomb_2d import apply_interface_recomb_y_2d
    R = apply_interface_recomb_y_2d(R.reshape((g.Ny, g.Nx)), n, p, mat)
```

(adjust the reshape to wherever `R` is currently un-flattened).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_interface_recomb_2d.py -v
pytest tests/regression/test_twod_lateral_heterogeneity.py::test_interface_srh_parity_2d_vs_1d_on_a_config_with_interfaces -v
pytest tests/regression/test_twod_validation.py -v
```

Expected: all pass. `test_twod_validation.py` uses `nip_MAPbI3`, which declares **no** interfaces, so its baselines must not move.

> **GATE 6 — vertical interface SRH (closes an existing gap)**
> - `pytest tests/unit/twod/test_interface_recomb_2d.py -v` → 6 passed (3 params + 3).
> - 2D-vs-1D `V_oc` parity on `solarscale_nip_band_aligned_iface.yaml` within **2 mV**. Record the pre-fix mismatch in the commit body — it quantifies how wrong every prior 2D run on an interface-carrying config was.
> - `pytest tests/regression/test_twod_validation.py tests/regression/test_twod_microstructure.py -v` → pass with **no baseline edits** (both use interface-free configs, so the new channel must be exactly inert there).
> - Explicit inertness check: `nip_MAPbI3` has `stack.interfaces == ()` ⇒ `mat.interface_srh_y == ()` ⇒ `apply_interface_recomb_y_2d` returns `R` unchanged (same object).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/twod/interface_recomb_2d.py perovskite_sim/twod/solver_2d.py \
        tests/unit/twod/test_interface_recomb_2d.py \
        tests/regression/test_twod_lateral_heterogeneity.py
git commit -m "fix(2d): apply interface SRH recombination in the 2D solver

Stage A computed only bulk recombination, so DeviceStack.interfaces SRVs and
InterfaceDefect never reached any 2D run — silently, on vertical interfaces
too. 2D V_oc on solarscale_nip_band_aligned_iface was <MEASURED> mV above 1D
before this fix; now within 2 mV.

Constraint: interface-free configs must stay bit-identical (interface_srh_y
is empty -> the source function returns R unchanged)
Constraint: defects read via electrical_interface_defects(), never by indexing
stack.interface_defects (substrate-offset shift, E10.1 glass regression)
Directive: the areal rate is divided by the dual-cell width, so 2D SRVs are
grid-referenced exactly as 1D SRVs are. Do not 'fix' this without re-deriving
the 1D normalisation too.
Not-tested: interface_plane_closure / iface_states in 2D — both 1D-scoped
Confidence: high
Scope-risk: moderate"
```

---

### Task 7: Extend interface SRH to lateral faces

**Files:**
- Modify: `perovskite_sim/twod/interface_recomb_2d.py`
- Modify: `perovskite_sim/twod/solver_2d.py`
- Test: `tests/unit/twod/test_interface_recomb_2d.py` (append)

**Interfaces:**
- Produces: `apply_interface_recomb_x_2d(R, n, p, mat) -> np.ndarray`, reading `mat.interface_srh_x`: a tuple of `(mask_2d, v_n, v_p, n1, p1)` where `mask_2d` is a boolean `(Ny, Nx)` node mask and the areal rate is divided by `hx_cell` broadcast across rows.

Which SRV does a lateral interface get? A lateral interface has no `stack.interfaces` slot (that tuple is `len(layers)-1` long and indexed by **vertical** interface). Design decision: **the region carries its own SRV pair**, added to `LateralRegion`:

```python
@dataclass(frozen=True)
class LateralRegion:
    name: str
    material: MaterialParams
    shape: Shape
    v_n: float = 0.0      # m/s, SRV on this region's lateral boundary
    v_p: float = 0.0      # m/s; 0 on either -> channel blocked (F07 semantics)
```

Default `(0.0, 0.0)` means **no lateral interface recombination unless declared** — consistent with SolarLab's existing opt-in-and-silent interface convention, and it keeps Tasks 1–6 bit-identical.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/twod/test_interface_recomb_2d.py`:

```python
from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.models.material import MaterialParams
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.lateral_geometry import (
    BoxShape, LateralGeometry, LateralRegion,
)
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import assemble_rhs_2d, build_material_arrays_2d


def _grid_stack():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers = [Layer(thickness=L.thickness, N=10)
              for L in electrical_layers(stack)]
    return build_grid_2d(layers, lateral_length=100e-9, Nx=20,
                         lateral_uniform=True), stack


def _geom(v_n, v_p):
    scaffold = MaterialParams(chi=2.9, Eg=4.5, mu_n=1e-7, mu_p=1e-7,
                              eps_r=35.0, Nc300=1e27, Nv300=1e27,
                              tau_n=1e-9, tau_p=1e-9)
    return LateralGeometry(regions=(
        LateralRegion(name="scaffold", material=scaffold,
                      shape=BoxShape(x_min=40e-9, x_max=60e-9,
                                     y_min=150e-9, y_max=300e-9),
                      v_n=v_n, v_p=v_p),
    ))


def test_zero_srv_region_is_bit_identical_to_no_srv():
    grid, stack = _grid_stack()
    a = build_material_arrays_2d(grid, stack, Microstructure(),
                                 lateral_geometry=_geom(0.0, 0.0))
    Ny, Nx = grid.Ny, grid.Nx
    y0 = np.concatenate([np.full(Ny * Nx, 1e18), np.full(Ny * Nx, 1e18)])
    dydt_zero = assemble_rhs_2d(0.0, y0, a, 0.3)

    b = build_material_arrays_2d(grid, stack, Microstructure(),
                                 lateral_geometry=_geom(0.0, 1.0))
    # One blocked channel blocks the cycle -> identical to fully blocked.
    assert np.array_equal(dydt_zero, assemble_rhs_2d(0.0, y0, b, 0.3))


def test_lateral_srv_increases_recombination():
    grid, stack = _grid_stack()
    Ny, Nx = grid.Ny, grid.Nx
    y0 = np.concatenate([np.full(Ny * Nx, 1e18), np.full(Ny * Nx, 1e18)])

    off = build_material_arrays_2d(grid, stack, Microstructure(),
                                   lateral_geometry=_geom(0.0, 0.0))
    on = build_material_arrays_2d(grid, stack, Microstructure(),
                                  lateral_geometry=_geom(1.0, 1.0))
    dn_off = assemble_rhs_2d(0.0, y0, off, 0.3)[: Ny * Nx]
    dn_on = assemble_rhs_2d(0.0, y0, on, 0.3)[: Ny * Nx]
    # Extra recombination is a sink: dn/dt must be strictly lower somewhere,
    # and never higher anywhere.
    assert np.any(dn_on < dn_off - 1e-30)
    assert np.all(dn_on <= dn_off + 1e-30)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_interface_recomb_2d.py -k lateral_srv -v
```

Expected: `TypeError: LateralRegion.__init__() got an unexpected keyword argument 'v_n'`.

- [ ] **Step 3: Write the implementation**

Add `v_n`/`v_p` to `LateralRegion` (with defaults, so Task 1's tests still construct it positionally). Extend the YAML region parser to read them (`_LATERAL_REGION_KEYS |= {"v_n", "v_p"}`, `_f(entry, "v_n", 0.0)`).

Add to `interface_recomb_2d.py`:

```python
def region_boundary_mask(region_ids: np.ndarray, k: int) -> np.ndarray:
    """Nodes of region ``k`` that touch a node outside region ``k`` in x.

    The evaluation nodes are taken INSIDE the region (the same convention as
    the 1D interface channel, whose evaluation node sits inside the layer),
    so the areal rate is normalised by the region node's own dual cell.
    """
    inside = region_ids == k
    neighbour_out = np.zeros_like(inside)
    neighbour_out[:, :-1] |= inside[:, :-1] & ~inside[:, 1:]
    neighbour_out[:, 1:] |= inside[:, 1:] & ~inside[:, :-1]
    return neighbour_out


def apply_interface_recomb_x_2d(R, n, p, mat):
    """Add the lateral-interface SRH source to ``R``. Returns a NEW array."""
    entries = getattr(mat, "interface_srh_x", ())
    if not entries:
        return R
    out = R.copy()
    hx_cell = mat.hx_cell                     # (Nx,)
    ni_sq = mat.ni ** 2
    inv_hx = (1.0 / hx_cell)[None, :]         # (1, Nx)
    for (mask, v_n, v_p, n1, p1) in entries:
        if not mask.any():
            continue
        areal = interface_recombination_vec(n, p, ni_sq, n1, p1, v_n, v_p)
        out = out + np.where(mask, areal * inv_hx, 0.0)
    return out
```

Build `interface_srh_x` in `build_material_arrays_2d`:

```python
    interface_srh_x = tuple(
        (region_boundary_mask(region_ids, k), float(r.v_n), float(r.v_p),
         n1, p1)
        for k, r in enumerate(_regions)
        if r.v_n > 0.0 and r.v_p > 0.0
    )
```

and call `apply_interface_recomb_x_2d` in `assemble_rhs_2d` right after the y-face version.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_interface_recomb_2d.py -v
pytest tests/regression/test_twod_validation.py tests/regression/test_twod_microstructure.py -v
```

> **GATE 7 — lateral interface SRH**
> - Region with `v_n=0` **or** `v_p=0` is bit-identical to no lateral SRH (F07 semantics preserved in 2D).
> - A positive SRV pair strictly reduces `dn/dt` somewhere and raises it nowhere — recombination is a sink, and a sign error here would show as a source.
> - Both existing 2D regression files pass with **no baseline edits** (their configs declare no lateral geometry at all).
> - `interface_srh_x` is empty for every shipped config:
>   ```bash
>   python3 -c "
>   from pathlib import Path
>   from perovskite_sim.models.config_loader import load_device_from_yaml
>   for p in sorted(Path('configs').rglob('*.yaml')):
>       s = load_device_from_yaml(str(p))
>       g = getattr(s, 'lateral_geometry', None)
>       assert g is None or g.is_empty or all(
>           r.v_n <= 0 or r.v_p <= 0 for r in g.regions), p
>   print('no shipped config declares lateral SRH')
>   "
>   ```

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/twod/interface_recomb_2d.py perovskite_sim/twod/lateral_geometry.py \
        perovskite_sim/models/config_loader.py tests/unit/twod/test_interface_recomb_2d.py
git commit -m "feat(2d): interface SRH on lateral region boundaries

LateralRegion carries its own (v_n, v_p) — a lateral interface has no
stack.interfaces slot, since that tuple is indexed by vertical interface.
Default (0, 0) keeps every existing path bit-identical.

Constraint: F07 semantics — either velocity non-positive blocks the cycle
Directive: evaluation nodes are taken INSIDE the region so the areal rate is
normalised by the region node's own dual cell, matching the 1D convention.
Changing this changes what a declared SRV means.
Not-tested: overlapping regions with different SRV pairs on a shared boundary
Confidence: medium
Scope-risk: moderate"
```

---

### Task 8: Mesoporous demo config and physics gate

Capstone: a device that only makes sense in 2D, with a falsifiable physical prediction.

**Files:**
- Create: `configs/twod/mesoporous_scaffold_demo.yaml`
- Modify: `perovskite_sim/twod/experiments/jv_sweep_2d.py:230-310`
- Test: `tests/regression/test_twod_lateral_heterogeneity.py` (append)

**Interfaces:**
- Produces: `run_jv_sweep_2d(..., lateral_geometry: LateralGeometry | None = None)`. `None` → `stack.lateral_geometry` → `LateralGeometry()`, mirroring the existing `microstructure` resolution at `jv_sweep_2d.py:307-308`.

**Physical prediction under test:** a wide-gap, electron-blocking scaffold (χ = 2.9 eV vs absorber 3.9 eV → +1.0 eV CB barrier) occupying part of the absorber must **reduce** `J_sc` — it removes absorber volume and lengthens collection paths, and it cannot compensate by collecting laterally. Magnitude bracket rather than a point value, because the reduction depends on grid resolution of the column edge.

- [ ] **Step 1: Write the failing test**

Append to `tests/regression/test_twod_lateral_heterogeneity.py`:

```python
def test_blocking_scaffold_reduces_jsc():
    """A wide-gap electron-blocking column removes absorber and blocks
    lateral collection, so J_sc must fall. Bracketed, not pinned: the exact
    reduction depends on how well the column edge is resolved."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
    from perovskite_sim.twod.lateral_geometry import LateralGeometry

    stack = load_device_from_yaml("configs/twod/mesoporous_scaffold_demo.yaml")
    assert stack.lateral_geometry is not None
    assert not stack.lateral_geometry.is_empty

    kw = dict(lateral_length=100e-9, Nx=20, Ny_per_layer=10,
              V_max=1.2, V_step=0.1, settle_t=1e-3, lateral_bc="neumann")

    uniform = run_jv_sweep_2d(stack, lateral_geometry=LateralGeometry(), **kw)
    with_scaffold = run_jv_sweep_2d(stack, **kw)

    j_uni = abs(uniform.metrics.J_sc)
    j_sca = abs(with_scaffold.metrics.J_sc)
    assert j_uni > 0.0
    frac = (j_uni - j_sca) / j_uni
    assert 0.02 <= frac <= 0.60, (
        f"J_sc reduction {frac:.1%} outside the plausible window "
        f"(uniform {j_uni:.2f}, scaffold {j_sca:.2f} A/m^2)"
    )


def test_scaffold_state_is_laterally_non_uniform():
    """The Stage-A validation gate asserts lateral invariance to 1e-9. With a
    scaffold that invariance MUST break — otherwise the geometry is inert."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d

    stack = load_device_from_yaml("configs/twod/mesoporous_scaffold_demo.yaml")
    res = run_jv_sweep_2d(stack, lateral_length=100e-9, Nx=20,
                          Ny_per_layer=10, V_max=0.6, V_step=0.2,
                          settle_t=1e-3, lateral_bc="neumann",
                          save_snapshots=True)
    snap = res.snapshots[-1]
    spread = (snap.n.max(axis=1) - snap.n.min(axis=1)) / snap.n.mean(axis=1)
    assert spread.max() > 1e-3, (
        "electron density is still laterally uniform — the lateral geometry "
        "is not reaching the solver"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd perovskite-sim
pytest tests/regression/test_twod_lateral_heterogeneity.py -k scaffold -v
```

Expected: `FileNotFoundError` on the config.

- [ ] **Step 3: Write the config and thread the parameter**

Create `configs/twod/mesoporous_scaffold_demo.yaml`. Start from `configs/twod/nip_MAPbI3_uniform.yaml` (or `configs/nip_MAPbI3.yaml`) and append:

```yaml
# ===========================================================================
# Mesoporous scaffold demo — NOT a physical default.
#
# A wide-gap, electron-blocking column painted into the absorber, in the
# spirit of the m-ZrO2 spacer in ChargeFabrica (Sachsenweger, Torre
# Cachafeiro & Tress, Materials Futures, doi:10.1088/2752-5724/ae27e9).
# Parameters are illustrative, not fitted to any measured device. Do not use
# this preset as a calibration baseline.
#
# lateral_bc MUST be "neumann" for this geometry: the column sits at mid-x,
# so with periodic BCs the wrap face is inside the absorber (fine), but any
# variant that moves the column to the domain edge would put a
# heterointerface on the uncapped wrap face and raise NotImplementedError.
# ===========================================================================
lateral_geometry:
  regions:
    - name: scaffold
      v_n: 0.0            # no lateral interface SRH in the demo — the J_sc
      v_p: 0.0            # reduction must come from geometry alone
      material:
        chi: 2.9          # eV — 1.0 eV CB barrier vs the 3.9 eV absorber
        Eg: 4.5           # eV — wide gap, does not absorb
        Nc300: 1.0e27
        Nv300: 1.0e27
        mu_n: 1.0e-7
        mu_p: 1.0e-7
        eps_r: 35.0
        tau_n: 1.0e-9
        tau_p: 1.0e-9
        D_ion: 0.0
      shape:
        kind: sinusoid_column
        x_center: 50.0e-9
        amplitude: 15.0e-9
        wavelength: 200.0e-9
        width: 25.0e-9
        y_min: 60.0e-9
        y_max: 340.0e-9
```

Adjust `y_min`/`y_max` to sit inside the absorber layer of whichever base config you start from — read the layer thicknesses and compute the absorber's y-range rather than copying these numbers blind.

In `jv_sweep_2d.py`, add the parameter and thread it:

```python
def run_jv_sweep_2d(
    stack: DeviceStack,
    *,
    microstructure: Microstructure | None = None,
    lateral_geometry: "LateralGeometry | None" = None,
    ...
):
    ...
    if lateral_geometry is None:
        lateral_geometry = getattr(stack, "lateral_geometry", None) \
                           or LateralGeometry()
    ...
    mat = build_material_arrays_2d(
        grid, stack, microstructure, lateral_bc=lateral_bc,
        P_ion_static_1d=P_ion_static_1d,
        lateral_geometry=lateral_geometry,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/regression/test_twod_lateral_heterogeneity.py -v
pytest tests/regression/test_twod_validation.py tests/regression/test_twod_microstructure.py -v
pytest
```

Expected: all pass. If the `J_sc` reduction lands outside `[2%, 60%]`, **do not widen the bracket** — investigate. Outside that window means either the scaffold is inert (geometry not reaching the solver — check `test_scaffold_state_is_laterally_non_uniform` first) or it is over-blocking (column too wide for `Nx=20`, so it spans the domain).

> **GATE 8 — end-to-end physics**
> - `test_blocking_scaffold_reduces_jsc` → reduction inside `[2%, 60%]`.
> - `test_scaffold_state_is_laterally_non_uniform` → row-wise relative spread of `n` exceeds `1e-3`. **This is the anti-placebo gate**: the Stage-A validation asserts lateral invariance to `1e-9`, so if this fails the whole feature is inert plumbing.
> - Full default suite green: `pytest` → no failures, no baseline edits anywhere.
> - Slow lane green: `pytest -m slow` → no failures. Mandatory before declaring the task done — the slow lane is the only gate covering ion-coupled full sweeps, and a default run reports green on a broken `main`.
> - Wall clock recorded for the demo sweep at `Nx=20`, to size Task 9.

- [ ] **Step 5: Commit**

```bash
git add configs/twod/mesoporous_scaffold_demo.yaml \
        perovskite_sim/twod/experiments/jv_sweep_2d.py \
        tests/regression/test_twod_lateral_heterogeneity.py
git commit -m "feat(2d): mesoporous scaffold demo preset and physics gate

End-to-end: YAML lateral_geometry -> painted material fields -> lateral TE
cap -> J-V. A wide-gap blocking column reduces J_sc by 2-60% and breaks the
lateral invariance the Stage-A gate pins at 1e-9.

Constraint: the demo declares v_n=v_p=0 so the J_sc reduction is attributable
to geometry alone, not to a tuned lateral SRV
Rejected: pinning the J_sc reduction to a point value | it depends on how
well Nx resolves the column edge; a bracket is the honest gate
Directive: banner-labelled NOT a physical default. Do not adopt as a
calibration baseline (same rule as field_mobility_demo / bcx_combined_demo).
Not-tested: scaffold + microstructure GB + Robin contacts + mu(E) together
Confidence: medium
Scope-risk: narrow"
```

---

### Task 9: Jacobian sparsity spike — measurement with a falsifiable outcome

**This task may legitimately conclude "does not work."** Its deliverable is a decision backed by numbers, not a guaranteed speedup. Write the benchmark first so the answer is reproducible either way.

**The risk, stated up front:** `assemble_rhs_2d` solves Poisson *globally* inside the RHS (`solve_poisson_2d` at `solver_2d.py:661`). So `∂(dn[j,i])/∂(n[k,l])` is nonzero for **every** `(k,l)` through `φ` — the true ODE Jacobian is **dense**, and a 5-point `jac_sparsity` pattern is an *approximation*. Radau uses the Jacobian only as its inner Newton iteration matrix and evaluates residuals with the true RHS, so an approximate pattern cannot change the converged answer — but it can slow or destroy Newton convergence. Whether the Poisson coupling is weak enough (screened beyond a Debye length) is an empirical question.

**Files:**
- Create: `scripts/bench_2d_scaling.py`
- Modify: `perovskite_sim/twod/solver_2d.py:846-870` (only if the spike succeeds)
- Test: `tests/regression/test_twod_lateral_heterogeneity.py` (append, only if the spike succeeds)

**Interfaces:**
- Produces (conditional): `build_sg_jac_sparsity_2d(Ny, Nx, lateral_bc, *, halo=1) -> scipy.sparse.csr_matrix` of shape `(2*Ny*Nx, 2*Ny*Nx)`; `run_transient_2d(..., jac_sparsity=None)`.

- [ ] **Step 1: Write the benchmark**

Create `scripts/bench_2d_scaling.py`:

```python
"""Reproducible 2D Radau scaling benchmark.

BLAS is pinned inside this script — tests/conftest.py only pins for pytest,
and an unpinned probe is ~9 cores of thread-thrash on matrices too small to
parallelise (it also corrupts any concurrent timing on the machine).
numpy and scipy.linalg MUST be imported before threadpool_limits, or the
limit is a silent no-op.
"""
import numpy, scipy.linalg  # noqa: F401  — registers the BLAS first
from threadpoolctl import threadpool_limits

import argparse
import time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nip_MAPbI3.yaml")
    ap.add_argument("--nx", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--sparsity", action="store_true",
                    help="pass jac_sparsity to Radau")
    ap.add_argument("--halo", type=int, default=1)
    args = ap.parse_args()

    from perovskite_sim.discretization.grid import Layer
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.twod.grid_2d import build_grid_2d
    from perovskite_sim.twod.microstructure import Microstructure
    from perovskite_sim.twod.solver_2d import (
        build_material_arrays_2d, run_transient_2d,
    )

    stack = load_device_from_yaml(args.config)
    layers = [Layer(thickness=L.thickness, N=10)
              for L in electrical_layers(stack)]

    print(f"{'grid.Nx':>8} {'N_unk':>8} {'settle_s':>10}  note")
    for nx in args.nx:
        grid = build_grid_2d(layers, lateral_length=500e-9, Nx=nx,
                             lateral_uniform=True)
        mat = build_material_arrays_2d(grid, stack, Microstructure(),
                                       lateral_bc="periodic")
        Ny, Nxx = grid.Ny, grid.Nx
        N = 2 * Ny * Nxx
        y0 = np.concatenate([np.full(Ny * Nxx, 1e16),
                             np.full(Ny * Nxx, 1e16)])
        kw = {}
        if args.sparsity:
            from perovskite_sim.twod.solver_2d import build_sg_jac_sparsity_2d
            kw["jac_sparsity"] = build_sg_jac_sparsity_2d(
                Ny, Nxx, "periodic", halo=args.halo)
        t0 = time.time()
        try:
            run_transient_2d(y0, mat, V_app=0.0, t_end=1e-9,
                             max_nfev=20000, **kw)
            print(f"{Nxx:>8} {N:>8} {time.time()-t0:>10.2f}  ok")
        except Exception as exc:
            print(f"{Nxx:>8} {N:>8} {time.time()-t0:>10.2f}  "
                  f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    with threadpool_limits(limits=1, user_api="blas"):
        main()
```

- [ ] **Step 2: Reproduce the dense baseline**

```bash
cd perovskite-sim
python3 scripts/bench_2d_scaling.py --nx 4 8 16 32
```

Expected, within ~20% of the recorded baseline: `0.50 / 1.19 / 4.61 / 25.79 s`. If it differs by more, re-record and update the "Measured Baseline" table before proceeding — every gate below is relative.

- [ ] **Step 3: Implement the sparsity pattern**

Add to `solver_2d.py`:

```python
def build_sg_jac_sparsity_2d(Ny, Nx, lateral_bc, *, halo=1):
    """Approximate sparsity pattern for the 2D (n, p) Radau Jacobian.

    The SG divergence couples each node to its 4 neighbours; recombination
    couples n to p at the same node. Poisson, however, is solved GLOBALLY
    inside assemble_rhs_2d, so the true Jacobian is dense — this pattern
    deliberately truncates that coupling to a ``halo``-cell neighbourhood.

    That truncation is sound only because Radau uses the Jacobian as its
    inner Newton iteration matrix while evaluating residuals with the true
    RHS: a wrong pattern costs Newton iterations, not accuracy. If Newton
    stops converging, widen ``halo`` or abandon the approach — do NOT
    loosen rtol/atol to make it appear to work.
    """
    from scipy import sparse

    n_nodes = Ny * Nx

    def node_neighbours(j, i):
        out = [(j, i)]
        for dj in range(-halo, halo + 1):
            for di in range(-halo, halo + 1):
                jj, ii = j + dj, i + di
                if not (0 <= jj < Ny):
                    continue
                if lateral_bc == "periodic":
                    ii %= Nx
                elif not (0 <= ii < Nx):
                    continue
                out.append((jj, ii))
        return out

    rows, cols = [], []
    for j in range(Ny):
        for i in range(Nx):
            here = j * Nx + i
            for (jj, ii) in node_neighbours(j, i):
                there = jj * Nx + ii
                # n-block, p-block, and both n<->p cross-blocks
                for r_off, c_off in ((0, 0), (0, n_nodes),
                                     (n_nodes, 0), (n_nodes, n_nodes)):
                    rows.append(here + r_off)
                    cols.append(there + c_off)

    data = np.ones(len(rows), dtype=np.int8)
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(2 * n_nodes, 2 * n_nodes)
    )
```

Thread `jac_sparsity` through `run_transient_2d` into `solve_ivp`:

```python
def run_transient_2d(..., jac_sparsity=None) -> np.ndarray:
    ...
        sol = solve_ivp(
            rhs, (0.0, t_end), y0,
            method="Radau",
            rtol=rtol, atol=atol,
            max_step=max_step if max_step is not None else np.inf,
            dense_output=False,
            jac_sparsity=jac_sparsity,
        )
```

`jac_sparsity=None` is `solve_ivp`'s own default, so the disabled path is unchanged.

- [ ] **Step 4: Measure and decide**

```bash
cd perovskite-sim
python3 scripts/bench_2d_scaling.py --nx 4 8 16 32 64 --sparsity --halo 1
python3 scripts/bench_2d_scaling.py --nx 4 8 16 32 64 --sparsity --halo 2
```

Record all three curves (dense, halo=1, halo=2) in the commit message.

Also verify **the answer did not move**, at a size where both paths run:

```bash
python3 - <<'EOF'
import numpy, scipy.linalg  # noqa
from threadpoolctl import threadpool_limits
import numpy as np
with threadpool_limits(limits=1, user_api="blas"):
    from perovskite_sim.discretization.grid import Layer
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.twod.grid_2d import build_grid_2d
    from perovskite_sim.twod.microstructure import Microstructure
    from perovskite_sim.twod.solver_2d import (
        build_material_arrays_2d, run_transient_2d, build_sg_jac_sparsity_2d)
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers = [Layer(thickness=L.thickness, N=10)
              for L in electrical_layers(stack)]
    grid = build_grid_2d(layers, lateral_length=500e-9, Nx=16,
                         lateral_uniform=True)
    mat = build_material_arrays_2d(grid, stack, Microstructure())
    Ny, Nx = grid.Ny, grid.Nx
    y0 = np.concatenate([np.full(Ny*Nx, 1e16), np.full(Ny*Nx, 1e16)])
    a = run_transient_2d(y0, mat, V_app=0.0, t_end=1e-9, max_nfev=200000)
    b = run_transient_2d(y0, mat, V_app=0.0, t_end=1e-9, max_nfev=200000,
                         jac_sparsity=build_sg_jac_sparsity_2d(
                             Ny, Nx, "periodic", halo=1))
    rel = np.max(np.abs(a-b) / np.maximum(np.abs(a), 1e-30))
    print(f"max relative state difference: {rel:.3e}")
EOF
```

> **GATE 9 — sparsity spike (pass/fail/abandon, all three are valid outcomes)**
>
> **PASS** — adopt, and make it the default in `run_jv_sweep_2d`:
> - `Nx=32` settle at least **3× faster** than the 25.79 s dense baseline.
> - `Nx=64` completes without `RuntimeError` (the dense path cannot).
> - Max relative state difference vs dense at `Nx=16` ≤ **1e-6**.
> - `pytest` and `pytest -m slow` green with sparsity on by default.
>
> **PARTIAL** — keep it, opt-in only, off by default:
> - Speedup is real but under 3×, **or** convergence needs `halo=2`.
> - Same correctness bound (≤1e-6) still required.
> - Document the measured curve in `perovskite-sim/CLAUDE.md`.
>
> **ABANDON** — revert the code, keep `scripts/bench_2d_scaling.py` and write up the negative result:
> - Newton fails to converge at any halo, **or** the state difference exceeds 1e-6, **or** there is no speedup.
> - Record in `perovskite-sim/CLAUDE.md` under a "2D scaling — falsified approaches" heading, with the measured numbers, so this is not re-attempted blind. Note the remaining options for a future attempt: (a) promote φ to an explicit unknown so the Jacobian is genuinely sparse; (b) extend the 1D `experiments/steady_state.py` Newton driver to 2D, where the Jacobian is built explicitly and a Schur complement can eliminate φ; (c) accept `Nx ≤ 32` as the working envelope.
>
> **Under no outcome** may `rtol`/`atol` be loosened to manufacture a pass. That converts a convergence failure into a silent accuracy loss.

- [ ] **Step 5: Commit**

On PASS or PARTIAL:

```bash
git add perovskite_sim/twod/solver_2d.py scripts/bench_2d_scaling.py \
        tests/regression/test_twod_lateral_heterogeneity.py
git commit -m "perf(2d): optional sparse Jacobian pattern for the Radau solve

Dense FD Jacobian scales ~N^2.5 (measured 0.50/1.19/4.61/25.79 s at
N=310/558/1054/2046), capping usable Nx at ~32. A truncated 5-point
jac_sparsity pattern gives <MEASURED>.

Constraint: Poisson is solved globally inside the RHS, so the true Jacobian
is dense — this pattern is an approximation. Radau evaluates residuals with
the true RHS, so it costs Newton iterations, not accuracy.
Rejected: loosening rtol/atol to improve the timing | converts a convergence
failure into a silent accuracy loss
Directive: if Newton stalls, widen halo or abandon. Never relax tolerances.
Not-tested: sparsity + lateral geometry + microstructure + mu(E) together
Confidence: medium
Scope-risk: moderate"
```

On ABANDON:

```bash
git checkout -- perovskite_sim/twod/solver_2d.py
git add scripts/bench_2d_scaling.py perovskite-sim/CLAUDE.md
git commit -m "docs(2d): record jac_sparsity as a falsified 2D scaling approach

Measured <NUMBERS>. Poisson's global solve inside the RHS makes the true ODE
Jacobian dense; a truncated pattern <FAILURE MODE>. Benchmark script kept so
the next attempt starts from numbers.

Directive: do not re-attempt truncated jac_sparsity without first addressing
the global Poisson coupling. Remaining options are documented in the CLAUDE.md
entry.
Confidence: high
Scope-risk: narrow"
```

---

## Documentation Task (run after whichever of Tasks 1–9 land)

- [ ] Add a `**2D lateral material heterogeneity (Stage C)**` section to `perovskite-sim/CLAUDE.md`, sibling to the existing Stage A / Stage B blocks. It must state: the YAML schema; that painted `chi` is a raw affinity with **no** DOS band-potential fold; that lateral SRH is opt-in per region and silent when absent; the periodic-wrap-face `NotImplementedError`; the working `Nx` envelope with the measured timings; and that 2D still has no mobile ions.
- [ ] Add the same schema to the README config section (memory `feedback_docs_in_sync`: every new module / physics flag / behavioural change updates README and CLAUDE.md in the same change set).
- [ ] Commit: `docs: describe 2D lateral heterogeneity (Stage C)`.

---

## Out of Scope — follow-on plan required

**2D mobile ions.** ChargeFabrica solves cation and anion continuity in 2D alongside carriers; SolarLab holds ions as a frozen Poisson background (`solver_2d.py:61-63`). Porting `ion_continuity_rhs` doubles the state vector from `2·Ny·Nx` to `4·Ny·Nx`, which at the measured `N^2.5` scaling makes even `Nx=16` cost ~26 s per settle on the dense path.

**Entry gate for that plan: Task 9 concludes PASS or PARTIAL.** If Task 9 concludes ABANDON, 2D ions need the steady-state-Newton route first (option (b) in the Gate 9 abandon notes) and should not be attempted on the transient driver.

That plan will also have to decide what a lateral region does to ion transport. ChargeFabrica sets `cationmob = anionmob = 0` in the scaffold; the harmonic face mean then gives exactly zero, a hard zero-flux wall — and notably those arrays are excluded from its Gaussian smoothing, so the zero cannot leak. SolarLab's `flux_2d` harmonic mean has the same property, so the mechanism ports directly, but `D_ion = 0` in a painted region must be verified to produce a hard wall and not a small leak.

**Also out of scope here:** column-resolved optics (each x-column running its own Beer-Lambert/TMM through the painted absorber mask — currently regions are simply transparent and non-absorbing); the DOS band-potential fold applied to painted regions; per-region Richardson constants; 3D; backend/frontend surfacing of `lateral_geometry`.

---

## Self-Review

**Spec coverage.** The four items from the source analysis map as: sparse Jacobian → Task 9; material ID bitmap → Tasks 1–3; lateral interface physics → Tasks 4–7; 2D ions → explicitly deferred with a stated entry gate. The gap discovered during fact-gathering (2D applies no interface recombination at all) is Task 6. The capstone demo is Task 8.

**Ordering rationale.** The original outline put the sparse Jacobian first. Measurement inverted it: `Nx=32` costs 25.79 s, which is workable, so the physics can be built and validated at modest `Nx` without waiting on a spike that may fail. Task 9 is now last, and nothing depends on it succeeding.

**Type consistency.** `LateralGeometry` / `LateralRegion` / `BoxShape` / `SinusoidColumnShape` / `build_region_id_field` / `shape_mask` are used with identical names and signatures in Tasks 1, 2, 3, 5, 7, 8. `LateralRegion` gains `v_n`/`v_p` in Task 7 with defaults, so Task 1's positional constructions keep working. `interface_recombination_vec` has the same signature in Tasks 6 and 7. `detect_interface_x_faces` / `detect_interface_y_faces_2d` / `dual_cell_widths_2d` are consistent between Tasks 4 and 5. `MaterialArrays2D` accumulates `region_ids` (Task 3), `iface_cb_x`/`iface_vb_x`/`iface_cb_y`/`iface_vb_y` (Task 5), `interface_srh_y`/`hy_cell` (Task 6), `interface_srh_x`/`hx_cell` (Task 7) — additive, no renames.

**Known soft spots, flagged rather than hidden.** (a) `dual_cell_widths_2d`'s interior `hx_cell` expression in Task 4 is reconstructed from reading `continuity_2d.py`; the task says explicitly to copy the real expression and treat `continuity_2d.py` as authoritative. (b) The `J_sc` reduction bracket in Task 8 is `[2%, 60%]` — wide because it depends on how `Nx` resolves the column edge; the paired non-uniformity test is what actually proves the feature is live. (c) Task 9 may fail; its gate says so and prescribes the write-up.

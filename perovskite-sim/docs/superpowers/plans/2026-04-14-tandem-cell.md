# Tandem Cell (2T Monolithic) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a 2-terminal monolithic all-perovskite tandem simulation path (YAML → combined TMM partitioning → two independent sub-cell J-V sweeps → series current-match) that reproduces the Lin 2019 Nature Energy benchmark within ±10% per metric.

**Architecture:** Reuse the existing single-junction drift-diffusion solver (`run_jv_sweep`) verbatim. Add (a) a tandem YAML loader that references two single-junction configs plus a junction block, (b) a tandem optics module that performs one combined TMM run over the full stack and partitions per-layer absorption into `G_top(x)` / `G_bot(x)`, (c) a tandem J-V driver that sweeps each sub-cell independently with its own `G(x)` and then series-matches the two curves through J-axis interpolation. Junction is ideal ohmic in v1.

**Tech Stack:** Python 3.13, numpy, scipy, PyYAML, FastAPI, pytest (existing stack). Frontend: TypeScript / Vite / existing stack-visualizer components.

**Spec reference:** `docs/superpowers/specs/2026-04-14-tandem-cell-design.md`

**Before you start:** The Lin 2019 benchmark task (Task 8) requires exact layer thicknesses and n/k data from the paper's SI. If you do not have access to the SI, stub the benchmark test with `pytest.skip("waiting on Lin 2019 SI data")` and flag it to the spec owner — do NOT invent numbers.

---

## Task 1: Tandem config schema + loader

**Files:**
- Create: `perovskite_sim/models/tandem_config.py`
- Create: `tests/unit/models/test_tandem_config.py`
- Create: `configs/tandem_lin2019.yaml`
- Modify: `perovskite_sim/models/config_loader.py` (dispatch on `device_type`)

**Interface contract** (freeze this before implementing — later tasks depend on it):

```python
from dataclasses import dataclass
from perovskite_sim.models.device import DeviceStack

@dataclass(frozen=True)
class JunctionLayer:
    name: str
    thickness: float           # metres
    optical_material: str      # key into n,k database
    incoherent: bool = False

@dataclass(frozen=True)
class TandemConfig:
    top_cell: DeviceStack          # loaded via load_device_from_yaml
    bottom_cell: DeviceStack       # loaded via load_device_from_yaml
    junction_stack: tuple[JunctionLayer, ...]
    junction_model: str            # "ideal_ohmic" (only value supported in v1)
    light_direction: str           # "top_first" (only value supported in v1)
    benchmark: dict | None         # optional {"reference": str, "target_pce": float, "tolerance_pct": float}
```

- [ ] **Step 1: Write failing test for tandem_config loader happy path**

Create `tests/unit/models/test_tandem_config.py`:

```python
from pathlib import Path
import pytest
import yaml

from perovskite_sim.models.tandem_config import (
    TandemConfig,
    JunctionLayer,
    load_tandem_from_yaml,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _write_minimal_cells(tmp_path: Path) -> tuple[Path, Path]:
    top = tmp_path / "top.yaml"
    bot = tmp_path / "bot.yaml"
    # Copy an existing minimal preset twice so we know the sub-loader works.
    repo_root = Path(__file__).resolve().parents[3]
    src = (repo_root / "configs" / "nip_MAPbI3.yaml").read_text()
    top.write_text(src)
    bot.write_text(src)
    return top, bot


def test_load_tandem_happy_path(tmp_path):
    top, bot = _write_minimal_cells(tmp_path)
    cfg_path = tmp_path / "tandem.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "device_type": "tandem_2T_monolithic",
        "tandem": {
            "top_cell": str(top),
            "bottom_cell": str(bot),
            "junction": {"model": "ideal_ohmic"},
            "light_direction": "top_first",
        },
        "junction_stack": [
            {"name": "recomb", "thickness_nm": 20.0,
             "optical_material": "PEDOT_PSS", "incoherent": False},
        ],
    }))

    cfg = load_tandem_from_yaml(str(cfg_path))

    assert isinstance(cfg, TandemConfig)
    assert cfg.junction_model == "ideal_ohmic"
    assert cfg.light_direction == "top_first"
    assert len(cfg.junction_stack) == 1
    j = cfg.junction_stack[0]
    assert isinstance(j, JunctionLayer)
    assert j.name == "recomb"
    assert j.thickness == pytest.approx(20e-9)
    assert j.optical_material == "PEDOT_PSS"
    assert cfg.top_cell is not cfg.bottom_cell
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/models/test_tandem_config.py::test_load_tandem_happy_path -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perovskite_sim.models.tandem_config'`

- [ ] **Step 3: Implement minimal loader**

Create `perovskite_sim/models/tandem_config.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack


SUPPORTED_JUNCTION_MODELS = frozenset({"ideal_ohmic"})
SUPPORTED_LIGHT_DIRECTIONS = frozenset({"top_first"})


@dataclass(frozen=True)
class JunctionLayer:
    name: str
    thickness: float           # metres
    optical_material: str
    incoherent: bool = False


@dataclass(frozen=True)
class TandemConfig:
    top_cell: DeviceStack
    bottom_cell: DeviceStack
    junction_stack: tuple[JunctionLayer, ...]
    junction_model: str
    light_direction: str
    benchmark: dict | None


def _resolve(base: Path, ref: str) -> str:
    p = Path(ref)
    if not p.is_absolute():
        p = (base.parent / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"tandem sub-cell YAML not found: {p}")
    return str(p)


def load_tandem_from_yaml(path: str) -> TandemConfig:
    cfg_path = Path(path).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("device_type") != "tandem_2T_monolithic":
        raise ValueError(
            f"{path}: expected device_type=tandem_2T_monolithic, got "
            f"{cfg.get('device_type')!r}"
        )

    tandem = cfg["tandem"]
    top_path = _resolve(cfg_path, tandem["top_cell"])
    bot_path = _resolve(cfg_path, tandem["bottom_cell"])
    top_cell = load_device_from_yaml(top_path)
    bottom_cell = load_device_from_yaml(bot_path)

    junction_model = tandem["junction"]["model"]
    if junction_model not in SUPPORTED_JUNCTION_MODELS:
        raise ValueError(
            f"junction.model={junction_model!r} not supported in v1. "
            f"Supported: {sorted(SUPPORTED_JUNCTION_MODELS)}"
        )

    light_direction = tandem.get("light_direction", "top_first")
    if light_direction not in SUPPORTED_LIGHT_DIRECTIONS:
        raise ValueError(
            f"light_direction={light_direction!r} not supported in v1. "
            f"Supported: {sorted(SUPPORTED_LIGHT_DIRECTIONS)}"
        )

    junction_raw = cfg.get("junction_stack", []) or []
    junction_stack = tuple(
        JunctionLayer(
            name=j["name"],
            thickness=float(j["thickness_nm"]) * 1e-9,
            optical_material=j["optical_material"],
            incoherent=bool(j.get("incoherent", False)),
        )
        for j in junction_raw
    )

    return TandemConfig(
        top_cell=top_cell,
        bottom_cell=bottom_cell,
        junction_stack=junction_stack,
        junction_model=junction_model,
        light_direction=light_direction,
        benchmark=cfg.get("benchmark"),
    )
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/models/test_tandem_config.py::test_load_tandem_happy_path -v`
Expected: PASS

- [ ] **Step 5: Write failing validation tests**

Append to `tests/unit/models/test_tandem_config.py`:

```python
def test_rejects_unsupported_junction_model(tmp_path):
    top, bot = _write_minimal_cells(tmp_path)
    cfg_path = tmp_path / "tandem.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "device_type": "tandem_2T_monolithic",
        "tandem": {
            "top_cell": str(top),
            "bottom_cell": str(bot),
            "junction": {"model": "tunnel_diode"},
            "light_direction": "top_first",
        },
        "junction_stack": [],
    }))
    with pytest.raises(ValueError, match="junction.model"):
        load_tandem_from_yaml(str(cfg_path))


def test_rejects_wrong_device_type(tmp_path):
    top, bot = _write_minimal_cells(tmp_path)
    cfg_path = tmp_path / "tandem.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "device_type": "single_junction",
        "tandem": {
            "top_cell": str(top),
            "bottom_cell": str(bot),
            "junction": {"model": "ideal_ohmic"},
            "light_direction": "top_first",
        },
    }))
    with pytest.raises(ValueError, match="device_type"):
        load_tandem_from_yaml(str(cfg_path))
```

- [ ] **Step 6: Run validation tests**

Run: `pytest tests/unit/models/test_tandem_config.py -v`
Expected: 3 PASS (existing validation already implemented in Step 3)

- [ ] **Step 7: Commit**

```bash
git add perovskite_sim/models/tandem_config.py tests/unit/models/test_tandem_config.py
git commit -m "feat(tandem): add TandemConfig loader with junction-model validation"
```

---

## Task 2: Tandem optics — combined TMM + per-sub-cell partitioning

**Files:**
- Create: `perovskite_sim/physics/tandem_optics.py`
- Create: `tests/unit/physics/test_tandem_optics.py`

**Interface contract:**

```python
@dataclass(frozen=True)
class TandemGeneration:
    G_top: np.ndarray          # (N_top,)  [m^-3 s^-1] on top-cell electrical grid
    G_bot: np.ndarray          # (N_bot,)  [m^-3 s^-1] on bottom-cell electrical grid
    parasitic_absorption: float   # fraction of incident photon flux absorbed in junction layers
    top_layer_slice: slice        # indices into the combined stack that belong to the top cell
    bottom_layer_slice: slice     # indices into the combined stack that belong to the bottom cell

def compute_tandem_generation(
    cfg: TandemConfig,
    wavelengths: np.ndarray,       # (n_wl,) in metres
    spectral_flux: np.ndarray,     # (n_wl,)  AM1.5G photon flux [m^-2 s^-1 m^-1]
    nk_database,                   # same type accepted by perovskite_sim.physics.optics.TMMLayer loader
    N_top: int,
    N_bot: int,
) -> TandemGeneration: ...
```

- [ ] **Step 1: Write failing partition test on a synthetic 3-layer stack**

Create `tests/unit/physics/test_tandem_optics.py`:

```python
import numpy as np
import pytest

from perovskite_sim.physics.tandem_optics import (
    TandemGeneration,
    partition_absorption,
)


def test_partition_assigns_layer_ranges_correctly():
    # Synthetic absorption array: 10 grid points, 3 wavelengths.
    # Points 0-3: top-cell, 4-5: junction, 6-9: bottom-cell.
    A = np.ones((10, 3))  # uniform absorption rate
    x = np.linspace(0.0, 1.0, 10)
    spectral_flux = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([400e-9, 500e-9, 600e-9])
    top_slice = slice(0, 4)
    junction_slice = slice(4, 6)
    bot_slice = slice(6, 10)

    G_top, G_bot, parasitic = partition_absorption(
        A, x, wavelengths, spectral_flux,
        top_slice, junction_slice, bot_slice,
    )

    assert G_top.shape == (4,)
    assert G_bot.shape == (4,)
    # Uniform absorption => integrated flux = sum(spectral_flux * d_lam).
    # We only assert positivity + ordering + parasitic fraction bounds here;
    # absolute value validated in the next test with a closed form.
    assert np.all(G_top > 0)
    assert np.all(G_bot > 0)
    assert 0.0 < parasitic < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/physics/test_tandem_optics.py::test_partition_assigns_layer_ranges_correctly -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perovskite_sim.physics.tandem_optics'`

- [ ] **Step 3: Implement partition_absorption**

Create `perovskite_sim/physics/tandem_optics.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TandemGeneration:
    G_top: np.ndarray
    G_bot: np.ndarray
    parasitic_absorption: float
    top_layer_slice: slice
    bottom_layer_slice: slice


def partition_absorption(
    A: np.ndarray,              # (N, n_wl)  per-unit-length absorption rate [m^-1]
    x: np.ndarray,              # (N,)       spatial grid [m]
    wavelengths: np.ndarray,    # (n_wl,)    in metres
    spectral_flux: np.ndarray,  # (n_wl,)    incident photon flux [m^-2 s^-1 m^-1]
    top_slice: slice,
    junction_slice: slice,
    bottom_slice: slice,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split combined-stack absorption into top / bottom generation profiles.

    Returns:
        G_top: shape matches x[top_slice], generation rate [m^-3 s^-1]
        G_bot: shape matches x[bottom_slice]
        parasitic_fraction: photons absorbed in junction layers / total incident
    """
    # Integrate spectral absorption to get G(x) [m^-3 s^-1]
    integrand = A * spectral_flux[None, :]
    G_full = np.trapezoid(integrand, wavelengths, axis=1)  # (N,)

    G_top = G_full[top_slice]
    G_bot = G_full[bottom_slice]

    # Parasitic fraction: photon flux absorbed in junction layers.
    # integral over x and lambda of A * spectral_flux / incident total flux.
    total_incident = float(np.trapezoid(spectral_flux, wavelengths))
    if total_incident <= 0:
        return G_top, G_bot, 0.0

    junction_A = A[junction_slice, :]
    junction_x = x[junction_slice]
    if junction_x.size >= 2:
        # Integrate along x first, then lambda.
        per_lambda = np.trapezoid(junction_A, junction_x, axis=0)  # (n_wl,)
        absorbed = float(np.trapezoid(per_lambda * spectral_flux, wavelengths))
    else:
        absorbed = 0.0

    return G_top, G_bot, absorbed / total_incident
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/physics/test_tandem_optics.py::test_partition_assigns_layer_ranges_correctly -v`
Expected: PASS

- [ ] **Step 5: Add closed-form sanity test**

Append to `tests/unit/physics/test_tandem_optics.py`:

```python
def test_partition_conserves_photon_count():
    """Total photons absorbed across all slices must equal photon count from A."""
    rng = np.random.default_rng(0)
    N = 30
    n_wl = 8
    A = rng.uniform(0.0, 1.0, size=(N, n_wl))
    x = np.linspace(0.0, 500e-9, N)
    wavelengths = np.linspace(400e-9, 800e-9, n_wl)
    spectral_flux = np.full(n_wl, 1e21)

    top = slice(0, 10)
    junc = slice(10, 14)
    bot = slice(14, N)

    G_top, G_bot, parasitic = partition_absorption(
        A, x, wavelengths, spectral_flux, top, junc, bot,
    )

    # Photon count = integral over x of G(x), in the per-cell slices.
    top_photons = float(np.trapezoid(G_top, x[top]))
    bot_photons = float(np.trapezoid(G_bot, x[bot]))
    total_incident = float(np.trapezoid(spectral_flux, wavelengths))
    junc_photons = parasitic * total_incident

    # Reference: same integral across the full stack.
    integrand = A * spectral_flux[None, :]
    G_full = np.trapezoid(integrand, wavelengths, axis=1)
    full_photons = float(np.trapezoid(G_full, x))

    assert top_photons + junc_photons + bot_photons == pytest.approx(
        full_photons, rel=1e-9
    )
```

- [ ] **Step 6: Run all Task 2 tests**

Run: `pytest tests/unit/physics/test_tandem_optics.py -v`
Expected: 2 PASS

- [ ] **Step 7: Add combined-TMM driver `compute_tandem_generation`**

Append to `perovskite_sim/physics/tandem_optics.py`:

```python
from perovskite_sim.physics.optics import TMMLayer, tmm_absorption_profile
from perovskite_sim.models.tandem_config import TandemConfig
from perovskite_sim.models.device import DeviceStack


def _build_tmm_layers_from_stack(
    stack: DeviceStack, nk_database, wavelengths: np.ndarray,
) -> list[TMMLayer]:
    """Adapter: convert a DeviceStack's layer list to TMMLayer objects.

    Follows the same pattern used by the single-junction path in
    perovskite_sim/experiments/jv_sweep.py when TMM is enabled.
    """
    layers = []
    for layer in stack.layers:
        nk = nk_database.lookup(layer.optical_material, wavelengths)
        layers.append(TMMLayer(
            name=layer.name,
            thickness=layer.thickness,
            n=nk.n,
            k=nk.k,
            incoherent=getattr(layer, "incoherent", False),
        ))
    return layers


def compute_tandem_generation(
    cfg: TandemConfig,
    wavelengths: np.ndarray,
    spectral_flux: np.ndarray,
    nk_database,
    N_top: int,
    N_bot: int,
) -> TandemGeneration:
    """One combined TMM run; split result into per-sub-cell generation profiles."""
    top_tmm = _build_tmm_layers_from_stack(cfg.top_cell, nk_database, wavelengths)
    bot_tmm = _build_tmm_layers_from_stack(cfg.bottom_cell, nk_database, wavelengths)
    junc_tmm = [
        TMMLayer(
            name=j.name,
            thickness=j.thickness,
            n=nk_database.lookup(j.optical_material, wavelengths).n,
            k=nk_database.lookup(j.optical_material, wavelengths).k,
            incoherent=j.incoherent,
        )
        for j in cfg.junction_stack
    ]
    combined = top_tmm + junc_tmm + bot_tmm
    n_top, n_junc, n_bot = len(top_tmm), len(junc_tmm), len(bot_tmm)

    # Build boundaries (cumulative layer edges).
    thicknesses = np.array([L.thickness for L in combined])
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))
    total_thickness = float(boundaries[-1])

    # Partition grid: N_top nodes within the top-cell physical extent,
    # a small fixed number in the junction, N_bot in the bottom cell.
    # Use non-overlapping linspaces anchored at the layer boundaries.
    top_end = boundaries[n_top]
    junc_end = boundaries[n_top + n_junc]
    x_top = np.linspace(0.0, top_end, N_top)
    x_junc = np.linspace(top_end, junc_end, max(3, n_junc * 3))
    x_bot = np.linspace(junc_end, total_thickness, N_bot)
    x = np.concatenate([x_top, x_junc[1:-1], x_bot])  # avoid duplicates

    A = tmm_absorption_profile(
        combined, wavelengths, x, boundaries,
        n_ambient=1.0, n_substrate=1.0,
    )

    # Recover slice indices after dedup:
    top_slice = slice(0, N_top)
    # x_junc[1:-1] contributed (len(x_junc) - 2) points
    n_junc_pts = max(3, n_junc * 3) - 2
    junction_slice = slice(N_top, N_top + n_junc_pts)
    bottom_slice = slice(N_top + n_junc_pts, N_top + n_junc_pts + N_bot)

    G_top, G_bot, parasitic = partition_absorption(
        A, x, wavelengths, spectral_flux,
        top_slice, junction_slice, bottom_slice,
    )

    return TandemGeneration(
        G_top=G_top,
        G_bot=G_bot,
        parasitic_absorption=parasitic,
        top_layer_slice=top_slice,
        bottom_layer_slice=bottom_slice,
    )
```

- [ ] **Step 8: Commit**

```bash
git add perovskite_sim/physics/tandem_optics.py tests/unit/physics/test_tandem_optics.py
git commit -m "feat(tandem): add combined-TMM absorption partitioning"
```

---

## Task 3: Tandem J-V driver (series current-match)

**Files:**
- Create: `perovskite_sim/experiments/tandem_jv.py`
- Create: `tests/unit/experiments/test_tandem_jv.py`
- Modify: `perovskite_sim/experiments/__init__.py`

**Interface contract:**

```python
@dataclass(frozen=True)
class TandemJVResult:
    V: np.ndarray              # tandem voltage grid (V)
    J: np.ndarray              # tandem current density (A/m^2)
    V_top: np.ndarray          # top sub-cell voltage at each matched J
    V_bot: np.ndarray          # bottom sub-cell voltage at each matched J
    metrics: JVMetrics         # tandem J_sc, V_oc, FF, PCE
    top_result: JVResult       # raw top-cell forward sweep (for debugging)
    bot_result: JVResult       # raw bottom-cell forward sweep

def run_tandem_jv(
    cfg: TandemConfig,
    wavelengths: np.ndarray,
    spectral_flux: np.ndarray,
    nk_database,
    N_grid: int = 100,
    n_points: int = 50,
) -> TandemJVResult: ...

def series_match_jv(
    top_J: np.ndarray, top_V: np.ndarray,
    bot_J: np.ndarray, bot_V: np.ndarray,
    V_junction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (J_common, V_top, V_bot, V_tandem)."""
```

- [ ] **Step 1: Write failing test for series matching on synthetic curves**

Create `tests/unit/experiments/test_tandem_jv.py`:

```python
import numpy as np
import pytest

from perovskite_sim.experiments.tandem_jv import series_match_jv


def test_series_match_sums_voltages_at_matched_current():
    # Two linear J-V curves with different slopes and intercepts.
    # Sub-cell 1: V = 1.0 - 0.01 * J  (V_oc = 1.0, slope -0.01)
    # Sub-cell 2: V = 0.8 - 0.005 * J (V_oc = 0.8, slope -0.005)
    J_top = np.linspace(-50, 0, 51)
    V_top = 1.0 - 0.01 * J_top
    J_bot = np.linspace(-50, 0, 51)
    V_bot = 0.8 - 0.005 * J_bot

    J_common, V_top_m, V_bot_m, V_tandem = series_match_jv(
        J_top, V_top, J_bot, V_bot, V_junction=0.0,
    )

    # At J=0, V_tandem should equal V_oc_top + V_oc_bot = 1.8 V
    idx_zero = np.argmin(np.abs(J_common))
    assert V_tandem[idx_zero] == pytest.approx(1.8, abs=1e-6)
    # At J=-50, V_tandem should equal (1.0+0.5) + (0.8+0.25) = 2.55 V
    assert V_tandem[0] == pytest.approx(2.55, abs=1e-6)
    # Monotonicity (V_tandem decreasing as J increases)
    assert np.all(np.diff(V_tandem) <= 1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/experiments/test_tandem_jv.py::test_series_match_sums_voltages_at_matched_current -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'perovskite_sim.experiments.tandem_jv'`

- [ ] **Step 3: Implement series_match_jv (minimal)**

Create `perovskite_sim/experiments/tandem_jv.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from perovskite_sim.experiments.jv_sweep import (
    JVResult, JVMetrics, run_jv_sweep, _extract_metrics,
)
from perovskite_sim.models.tandem_config import TandemConfig


def series_match_jv(
    top_J: np.ndarray,
    top_V: np.ndarray,
    bot_J: np.ndarray,
    bot_V: np.ndarray,
    V_junction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Series-add two sub-cell J-V curves at a common current grid.

    Assumes both sub-cells use the same current-density convention. Builds a
    common J grid spanning the intersection of both sub-cells' J ranges, then
    interpolates each sub-cell's V onto that grid and sums.
    """
    # Sort by J ascending (interp requires monotonic x).
    top_order = np.argsort(top_J)
    bot_order = np.argsort(bot_J)
    tJ, tV = top_J[top_order], top_V[top_order]
    bJ, bV = bot_J[bot_order], bot_V[bot_order]

    j_lo = max(tJ[0], bJ[0])
    j_hi = min(tJ[-1], bJ[-1])
    if j_lo >= j_hi:
        raise ValueError(
            f"Sub-cell J ranges do not overlap: top=[{tJ[0]},{tJ[-1]}], "
            f"bottom=[{bJ[0]},{bJ[-1]}]"
        )

    n = max(len(tJ), len(bJ))
    J_common = np.linspace(j_lo, j_hi, n)
    V_top_m = np.interp(J_common, tJ, tV)
    V_bot_m = np.interp(J_common, bJ, bV)
    V_tandem = V_top_m + V_bot_m + V_junction
    return J_common, V_top_m, V_bot_m, V_tandem
```

- [ ] **Step 4: Run the series-match test**

Run: `pytest tests/unit/experiments/test_tandem_jv.py::test_series_match_sums_voltages_at_matched_current -v`
Expected: PASS

- [ ] **Step 5: Write failing test for current-limited J_sc**

Append to `tests/unit/experiments/test_tandem_jv.py`:

```python
def test_series_match_is_limited_by_smaller_jsc():
    # Top cell current-limited (J_sc = -20), bottom cell J_sc = -30.
    # Series tandem J_sc must equal the smaller in magnitude → -20.
    J_top = np.linspace(-20, 0, 41)
    V_top = 1.0 + 0.02 * J_top       # V_oc=1.0 at J=0, drops as J becomes more negative
    J_bot = np.linspace(-30, 0, 61)
    V_bot = 0.8 + 0.015 * J_bot

    J_common, _, _, V_tandem = series_match_jv(J_top, V_top, J_bot, V_bot)

    # The shared J range can only go as negative as -20 (the top-cell limit).
    assert J_common[0] == pytest.approx(-20.0, abs=1e-9)
    # V_tandem at J=0 is sum of V_oc
    idx_zero = np.argmin(np.abs(J_common))
    assert V_tandem[idx_zero] == pytest.approx(1.8, abs=1e-6)
```

- [ ] **Step 6: Run the current-limited test**

Run: `pytest tests/unit/experiments/test_tandem_jv.py::test_series_match_is_limited_by_smaller_jsc -v`
Expected: PASS

- [ ] **Step 7: Add `run_tandem_jv` glue**

Append to `perovskite_sim/experiments/tandem_jv.py`:

```python
@dataclass(frozen=True)
class TandemJVResult:
    V: np.ndarray
    J: np.ndarray
    V_top: np.ndarray
    V_bot: np.ndarray
    metrics: JVMetrics
    top_result: JVResult
    bot_result: JVResult


def run_tandem_jv(
    cfg: TandemConfig,
    wavelengths: np.ndarray,
    spectral_flux: np.ndarray,
    nk_database,
    N_grid: int = 100,
    n_points: int = 50,
) -> TandemJVResult:
    """Full tandem J-V: combined TMM → two sub-cell sweeps → series match."""
    from perovskite_sim.physics.tandem_optics import compute_tandem_generation

    gen = compute_tandem_generation(
        cfg, wavelengths, spectral_flux, nk_database,
        N_top=N_grid, N_bot=N_grid,
    )

    # Inject externally-computed G(x) into the single-junction sweep via a
    # DeviceStack override. jv_sweep's generation source is read from
    # stack.layers; we use the existing `run_jv_sweep` with a stack whose
    # material arrays have been pre-loaded with our G profile. See the
    # `fixed_generation` argument planned for jv_sweep (Task 4 prerequisite).
    top_result = run_jv_sweep(
        cfg.top_cell, N_grid=N_grid, n_points=n_points,
        fixed_generation=gen.G_top,
    )
    bot_result = run_jv_sweep(
        cfg.bottom_cell, N_grid=N_grid, n_points=n_points,
        fixed_generation=gen.G_bot,
    )

    # Use forward sweep for the matching (v1 ignores hysteresis at tandem level).
    J_common, V_top_m, V_bot_m, V_tandem = series_match_jv(
        top_result.J_fwd, top_result.V_fwd,
        bot_result.J_fwd, bot_result.V_fwd,
        V_junction=0.0,
    )

    metrics = _extract_metrics(V_tandem, J_common)

    return TandemJVResult(
        V=V_tandem,
        J=J_common,
        V_top=V_top_m,
        V_bot=V_bot_m,
        metrics=metrics,
        top_result=top_result,
        bot_result=bot_result,
    )
```

- [ ] **Step 8: Export from `perovskite_sim/experiments/__init__.py`**

Open `perovskite_sim/experiments/__init__.py` and append:

```python
from perovskite_sim.experiments.tandem_jv import (
    TandemJVResult,
    run_tandem_jv,
    series_match_jv,
)

__all__ = [
    *globals().get("__all__", []),
    "TandemJVResult",
    "run_tandem_jv",
    "series_match_jv",
]
```

- [ ] **Step 9: Run unit tests**

Run: `pytest tests/unit/experiments/test_tandem_jv.py -v`
Expected: 2 PASS

Run: `pytest tests/unit/ -x --no-header -q`
Expected: all existing unit tests still green.

- [ ] **Step 10: Commit**

```bash
git add perovskite_sim/experiments/tandem_jv.py perovskite_sim/experiments/__init__.py tests/unit/experiments/test_tandem_jv.py
git commit -m "feat(tandem): add series current-match J-V driver"
```

---

## Task 4: Teach `run_jv_sweep` to accept a pre-computed `G(x)`

**Files:**
- Modify: `perovskite_sim/experiments/jv_sweep.py`
- Modify: `tests/unit/experiments/test_jv_sweep.py` (or create if missing)

**Why this task exists:** Task 3 calls `run_jv_sweep(..., fixed_generation=gen.G_top)`. The existing signature doesn't expose that hook; the tandem path needs it so each sub-cell runs drift-diffusion with a caller-supplied generation profile instead of recomputing from optics.

- [ ] **Step 1: Locate current generation source in jv_sweep**

Run: `grep -n "generation\|G_x\|tmm_generation\|beer_lambert" perovskite_sim/experiments/jv_sweep.py`

Expected: one or two call sites that build `G(x)` before passing it to `solve_illuminated_ss`. Note the exact variable name and line range — you will replace it with a branch that uses `fixed_generation` when provided.

- [ ] **Step 2: Write failing test for fixed_generation override**

Append to `tests/unit/experiments/test_jv_sweep.py` (create the file if missing with the standard `from perovskite_sim.experiments.jv_sweep import run_jv_sweep` import):

```python
import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_fixed_generation_override_is_honored(tmp_path):
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    # A zero generation profile should kill J_sc (no photocurrent).
    N_grid = 60
    G_zero = np.zeros(N_grid)
    result = run_jv_sweep(
        stack, N_grid=N_grid, n_points=20, fixed_generation=G_zero,
    )
    assert abs(result.metrics_fwd.J_sc) < 1.0  # A/m^2; effectively zero
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/experiments/test_jv_sweep.py::test_fixed_generation_override_is_honored -v`
Expected: FAIL with `TypeError: run_jv_sweep() got an unexpected keyword argument 'fixed_generation'`

- [ ] **Step 4: Add `fixed_generation` parameter**

Edit `perovskite_sim/experiments/jv_sweep.py` — in `run_jv_sweep`:

```python
def run_jv_sweep(
    stack: DeviceStack,
    N_grid: int = 100,
    v_rate: float = 0.1,
    n_points: int = 50,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    V_max: float | None = None,
    progress: ProgressCallback | None = None,
    fixed_generation: np.ndarray | None = None,
) -> JVResult:
```

Inside the function, locate the block that computes `G_x` (the generation profile on the electrical grid) and wrap it:

```python
if fixed_generation is not None:
    if fixed_generation.shape != (N_grid,):
        raise ValueError(
            f"fixed_generation shape {fixed_generation.shape} != (N_grid={N_grid},)"
        )
    G_x = fixed_generation.astype(float, copy=True)
else:
    G_x = <existing generation computation, unchanged>
```

Do not remove the existing computation — keep it as the `else` branch so the single-junction path is unchanged.

- [ ] **Step 5: Run the new test**

Run: `pytest tests/unit/experiments/test_jv_sweep.py::test_fixed_generation_override_is_honored -v`
Expected: PASS

- [ ] **Step 6: Run the full unit suite**

Run: `pytest tests/unit/ -x -q`
Expected: all green (regression check — single-junction path unchanged).

- [ ] **Step 7: Commit**

```bash
git add perovskite_sim/experiments/jv_sweep.py tests/unit/experiments/test_jv_sweep.py
git commit -m "feat(jv_sweep): accept fixed_generation override for tandem callers"
```

---

## Task 5: Wide-gap and Sn-Pb sub-cell preset YAMLs

**Files:**
- Create: `configs/nip_wideGap_FACs_1p77.yaml`
- Create: `configs/nip_SnPb_1p22.yaml`
- Create (if missing): `perovskite_sim/data/nk/FA_Cs_1p77.csv`, `perovskite_sim/data/nk/SnPb_1p22.csv`
- Modify: `perovskite_sim/data/nk/manifest.yaml` (register new materials)
- Create: `tests/unit/models/test_tandem_presets.py`

**Important:** These presets need realistic doping, lifetimes, and thicknesses. **Copy the closest existing single-junction preset** (`configs/nip_MAPbI3.yaml`) and overwrite only the values you can justify from literature (bandgap, thickness, n/k material key). **Do not invent numbers.** Where you cannot find a value, use the MAPbI3 preset's value unchanged and add a YAML comment `# TODO(benchmark): refine after Lin 2019 SI data is loaded`.

- [ ] **Step 1: Read the MAPbI3 preset to understand the field shape**

Run: `cat configs/nip_MAPbI3.yaml`

Note every field that will change for the wide-gap / narrow-gap variants: `Eg`, `chi`, layer thicknesses, absorber `optical_material`, and the `device.V_max` upper bound (needs to fit the tandem sub-cell V_oc, not the tandem total).

- [ ] **Step 2: Create wide-gap preset**

Create `configs/nip_wideGap_FACs_1p77.yaml` by copying `configs/nip_MAPbI3.yaml` and editing:
- absorber `Eg: 1.77`
- absorber `chi: 3.9`  (comment: `# FA-Cs wide-gap, Saliba et al. JPCL 2016`)
- absorber `optical_material: FA_Cs_1p77`
- absorber `thickness: 400e-9`  (Lin 2019 nominal; update in Task 8 if SI differs)
- all other fields unchanged
- add top-level comment block explaining this is a tandem sub-cell preset and should not be used standalone without re-tuning contacts

- [ ] **Step 3: Create narrow-gap preset**

Create `configs/nip_SnPb_1p22.yaml` analogously:
- absorber `Eg: 1.22`
- absorber `chi: 4.15`
- absorber `optical_material: SnPb_1p22`
- absorber `thickness: 850e-9`
- same top-level comment block

- [ ] **Step 4: Add placeholder n/k CSVs (DATA GATHERING STEP)**

If `perovskite_sim/data/nk/FA_Cs_1p77.csv` and `SnPb_1p22.csv` do not already exist, you must obtain them before the benchmark test in Task 8 can pass. **Options:**
  1. Extract from the Lin 2019 SI (preferred).
  2. Use Aguiar 2016 / Leijtens 2017 digitized n/k data for FA-Cs wide-gap.
  3. Use Rajagopal 2017 / Hao 2014 Sn-Pb n/k data for the narrow-gap.

Until real data is in place, create stub CSVs that linearly interpolate between MAPbI3 and a shifted bandgap edge, and mark the benchmark test `xfail` in Task 8.

CSV format must match existing files — inspect `perovskite_sim/data/nk/MAPbI3.csv` to confirm column order (`wavelength_nm, n, k`).

- [ ] **Step 5: Register the new materials in the nk manifest**

Edit `perovskite_sim/data/nk/manifest.yaml` and add entries mirroring the existing `MAPbI3` entry shape:

```yaml
FA_Cs_1p77:
  file: FA_Cs_1p77.csv
  source: "TODO(benchmark): Lin 2019 SI / Saliba 2016"
  eg_eV: 1.77

SnPb_1p22:
  file: SnPb_1p22.csv
  source: "TODO(benchmark): Lin 2019 SI / Hao 2014"
  eg_eV: 1.22
```

- [ ] **Step 6: Write a preset-loads-without-error test**

Create `tests/unit/models/test_tandem_presets.py`:

```python
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_wideGap_preset_loads():
    stack = load_device_from_yaml("configs/nip_wideGap_FACs_1p77.yaml")
    absorber = next(L for L in stack.layers if getattr(L, "role", None) == "absorber")
    assert absorber.Eg == 1.77


def test_SnPb_preset_loads():
    stack = load_device_from_yaml("configs/nip_SnPb_1p22.yaml")
    absorber = next(L for L in stack.layers if getattr(L, "role", None) == "absorber")
    assert absorber.Eg == 1.22
```

- [ ] **Step 7: Run the preset tests**

Run: `pytest tests/unit/models/test_tandem_presets.py -v`
Expected: 2 PASS. If the `role` attribute doesn't exist, swap the iteration for the known absorber layer index from the YAML ordering.

- [ ] **Step 8: Commit**

```bash
git add configs/nip_wideGap_FACs_1p77.yaml configs/nip_SnPb_1p22.yaml \
        perovskite_sim/data/nk/FA_Cs_1p77.csv perovskite_sim/data/nk/SnPb_1p22.csv \
        perovskite_sim/data/nk/manifest.yaml \
        tests/unit/models/test_tandem_presets.py
git commit -m "feat(tandem): add wide-gap + Sn-Pb sub-cell presets"
```

---

## Task 6: Tandem YAML benchmark config

**Files:**
- Create: `configs/tandem_lin2019.yaml`

- [ ] **Step 1: Write the config**

Create `configs/tandem_lin2019.yaml`:

```yaml
# Lin et al., Nature Energy 4, 864-873 (2019)
# https://www.nature.com/articles/s41560-019-0466-3
# Monolithic all-perovskite tandem, 24.8% PCE on 0.049 cm^2
schema_version: 1
device_type: tandem_2T_monolithic

tandem:
  top_cell: nip_wideGap_FACs_1p77.yaml
  bottom_cell: nip_SnPb_1p22.yaml
  junction:
    model: ideal_ohmic
  light_direction: top_first

junction_stack:
  - name: Au_nanoparticles
    thickness_nm: 1.0
    optical_material: Au
    incoherent: false
  - name: PEDOT_PSS_recomb
    thickness_nm: 20.0
    optical_material: PEDOT_PSS
    incoherent: false

benchmark:
  reference: lin2019_nature_energy
  target_pce: 24.8
  target_jsc_ma_cm2: 15.6
  target_voc_v: 1.965
  target_ff: 0.79
  tolerance_pct: 10.0
```

Note: `target_*` numbers are from Lin 2019 Table 1 — confirm against the paper before running Task 8.

- [ ] **Step 2: Verify it loads**

Run:
```bash
python -c "from perovskite_sim.models.tandem_config import load_tandem_from_yaml; \
cfg = load_tandem_from_yaml('configs/tandem_lin2019.yaml'); \
print(cfg.junction_model, cfg.benchmark['target_pce'])"
```

Expected output: `ideal_ohmic 24.8`

- [ ] **Step 3: Commit**

```bash
git add configs/tandem_lin2019.yaml
git commit -m "feat(tandem): add Lin 2019 benchmark tandem config"
```

---

## Task 7: Backend `/simulate/tandem` endpoint

**Files:**
- Create: `backend/routes/tandem.py`
- Modify: `backend/main.py` (register router)
- Create: `tests/integration/test_backend_tandem.py`

- [ ] **Step 1: Check existing backend route patterns**

Run: `grep -n "APIRouter\|include_router\|@app.post\|/simulate" backend/main.py backend/routes/*.py 2>/dev/null | head -40`

Observe the existing pattern (either `@app.post` directly in `main.py` or `APIRouter` modules in `backend/routes/`). Match whichever pattern is already used.

- [ ] **Step 2: Write failing integration test**

Create `tests/integration/test_backend_tandem.py`:

```python
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_tandem_endpoint_returns_metrics(client):
    payload = {"config_path": "configs/tandem_lin2019.yaml", "N_grid": 40, "n_points": 15}
    r = client.post("/simulate/tandem", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "metrics" in data
    assert "V" in data and "J" in data
    assert len(data["V"]) == len(data["J"]) > 0
    for key in ("V_oc", "J_sc", "FF", "PCE"):
        assert key in data["metrics"]
```

- [ ] **Step 3: Run test to verify failure**

Run: `pytest tests/integration/test_backend_tandem.py -v`
Expected: FAIL with 404 or NotFound for `/simulate/tandem`.

- [ ] **Step 4: Add the route handler**

Create `backend/routes/tandem.py`:

```python
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

from perovskite_sim.models.tandem_config import load_tandem_from_yaml
from perovskite_sim.experiments.tandem_jv import run_tandem_jv
from perovskite_sim.data import load_am15g, load_nk_database

router = APIRouter()


class TandemRequest(BaseModel):
    config_path: str
    N_grid: int = 80
    n_points: int = 40


@router.post("/simulate/tandem")
def simulate_tandem(req: TandemRequest):
    try:
        cfg = load_tandem_from_yaml(req.config_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    wavelengths, spectral_flux = load_am15g()
    nk_db = load_nk_database()

    result = run_tandem_jv(
        cfg, wavelengths, spectral_flux, nk_db,
        N_grid=req.N_grid, n_points=req.n_points,
    )

    return {
        "V": result.V.tolist(),
        "J": result.J.tolist(),
        "V_top": result.V_top.tolist(),
        "V_bot": result.V_bot.tolist(),
        "metrics": {
            "V_oc": result.metrics.V_oc,
            "J_sc": result.metrics.J_sc,
            "FF": result.metrics.FF,
            "PCE": result.metrics.PCE,
        },
        "benchmark": cfg.benchmark,
    }
```

If `load_am15g` / `load_nk_database` don't exist in `perovskite_sim.data`, check the existing single-junction backend route (`grep -n "am15g\|nk_database" backend/`) and import the same names it uses.

- [ ] **Step 5: Register the router**

Edit `backend/main.py` — find where existing routers are included (or where existing `@app.post` routes live) and add:

```python
from backend.routes.tandem import router as tandem_router
app.include_router(tandem_router)
```

If the file uses `@app.post` directly instead of `APIRouter`, promote this task's handler to an `@app.post("/simulate/tandem")` function in `main.py` with the same body.

- [ ] **Step 6: Run the test**

Run: `pytest tests/integration/test_backend_tandem.py -v`
Expected: PASS (assuming Lin 2019 preset loads; if the n,k CSVs are stubs, J/V values will be unrealistic but the endpoint should still return 200).

- [ ] **Step 7: Commit**

```bash
git add backend/routes/tandem.py backend/main.py tests/integration/test_backend_tandem.py
git commit -m "feat(backend): add /simulate/tandem endpoint"
```

---

## Task 8: Lin 2019 benchmark regression test

**Files:**
- Create: `tests/integration/test_tandem_lin2019.py`

**Blocker check:** Before writing this test, verify that `perovskite_sim/data/nk/FA_Cs_1p77.csv` and `SnPb_1p22.csv` contain real (not stub) data. If they are stubs, mark the test `@pytest.mark.xfail(reason="waiting on Lin 2019 SI n,k data")` and proceed — do NOT invent numbers to make it pass.

- [ ] **Step 1: Write the benchmark test**

Create `tests/integration/test_tandem_lin2019.py`:

```python
import pytest

from perovskite_sim.models.tandem_config import load_tandem_from_yaml
from perovskite_sim.experiments.tandem_jv import run_tandem_jv
from perovskite_sim.data import load_am15g, load_nk_database


@pytest.mark.slow
def test_lin2019_benchmark_within_tolerance():
    cfg = load_tandem_from_yaml("configs/tandem_lin2019.yaml")
    assert cfg.benchmark is not None
    tol = cfg.benchmark["tolerance_pct"] / 100.0

    wavelengths, spectral_flux = load_am15g()
    nk_db = load_nk_database()

    result = run_tandem_jv(
        cfg, wavelengths, spectral_flux, nk_db,
        N_grid=120, n_points=60,
    )
    m = result.metrics

    target_pce = cfg.benchmark["target_pce"]
    target_jsc = cfg.benchmark["target_jsc_ma_cm2"] * 10.0   # mA/cm^2 -> A/m^2
    target_voc = cfg.benchmark["target_voc_v"]
    target_ff = cfg.benchmark["target_ff"]

    assert m.PCE == pytest.approx(target_pce, rel=tol), (
        f"PCE {m.PCE:.2f} outside ±{tol*100:.0f}% of target {target_pce:.2f}"
    )
    assert abs(m.J_sc) == pytest.approx(target_jsc, rel=tol)
    assert m.V_oc == pytest.approx(target_voc, rel=tol)
    assert m.FF == pytest.approx(target_ff, rel=tol)
```

- [ ] **Step 2: Run the slow benchmark test**

Run: `pytest tests/integration/test_tandem_lin2019.py -v -m slow`
Expected: PASS if real n/k data + Lin 2019 thicknesses are loaded; otherwise XFAIL with the reason message from the blocker check.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_tandem_lin2019.py
git commit -m "test(tandem): add Lin 2019 benchmark regression test"
```

---

## Task 9: Frontend — Tandem tab scaffold

**Files:**
- Create: `frontend/src/panels/tandem.ts`
- Create: `frontend/src/stack/tandem-stack-visualizer.ts`
- Modify: `frontend/src/types.ts` (add tandem types)
- Modify: `frontend/src/App.ts` or router (register Tandem tab)

- [ ] **Step 1: Inspect frontend structure**

Run: `ls frontend/src/panels frontend/src/stack 2>/dev/null` and `grep -n "tab\|Tab\|route\|panels/" frontend/src/App.ts frontend/src/main.ts 2>/dev/null | head -30`

Note the existing tab registration pattern.

- [ ] **Step 2: Add tandem types**

Edit `frontend/src/types.ts` and append:

```typescript
export interface TandemJunctionLayer {
  name: string;
  thicknessNm: number;
  opticalMaterial: string;
  incoherent: boolean;
}

export interface TandemBenchmark {
  reference: string;
  targetPce: number;
  targetJscMaCm2?: number;
  targetVocV?: number;
  targetFf?: number;
  tolerancePct: number;
}

export interface TandemConfigView {
  topCellPath: string;
  bottomCellPath: string;
  junctionStack: TandemJunctionLayer[];
  junctionModel: "ideal_ohmic";
  lightDirection: "top_first";
  benchmark: TandemBenchmark | null;
}

export interface TandemJVPayload {
  V: number[];
  J: number[];
  V_top: number[];
  V_bot: number[];
  metrics: { V_oc: number; J_sc: number; FF: number; PCE: number };
  benchmark: TandemBenchmark | null;
}
```

- [ ] **Step 3: Create `TandemStackVisualizer` wrapper**

Create `frontend/src/stack/tandem-stack-visualizer.ts`:

```typescript
import { StackVisualizer } from "./stack-visualizer";

export class TandemStackVisualizer {
  private readonly container: HTMLElement;
  private readonly top: StackVisualizer;
  private readonly bot: StackVisualizer;

  constructor(container: HTMLElement) {
    this.container = container;
    this.container.innerHTML = `
      <div class="tandem-stack-layout">
        <div class="tandem-col" data-role="top">
          <h3>Top sub-cell (wide-gap)</h3>
          <div class="tandem-stack-host" id="tandem-top-host"></div>
        </div>
        <div class="tandem-junction">
          <h3>Junction</h3>
          <div id="tandem-junction-host"></div>
        </div>
        <div class="tandem-col" data-role="bottom">
          <h3>Bottom sub-cell (narrow-gap)</h3>
          <div class="tandem-stack-host" id="tandem-bot-host"></div>
        </div>
      </div>
    `;
    this.top = new StackVisualizer(
      this.container.querySelector("#tandem-top-host") as HTMLElement,
    );
    this.bot = new StackVisualizer(
      this.container.querySelector("#tandem-bot-host") as HTMLElement,
    );
  }
}
```

- [ ] **Step 4: Create Tandem panel with run button and results area**

Create `frontend/src/panels/tandem.ts`:

```typescript
import { TandemJVPayload } from "../types";
import { TandemStackVisualizer } from "../stack/tandem-stack-visualizer";

export class TandemPanel {
  private readonly root: HTMLElement;
  private visualizer: TandemStackVisualizer | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
  }

  mount(): void {
    this.root.innerHTML = `
      <section class="tandem-panel">
        <header>
          <h2>Tandem Simulation (2T monolithic)</h2>
          <div class="tandem-controls">
            <input type="text" id="tandem-config-path"
                   value="configs/tandem_lin2019.yaml" />
            <button id="tandem-run-btn">Run tandem J-V</button>
          </div>
        </header>
        <div id="tandem-stack-visualizer"></div>
        <div id="tandem-results"></div>
      </section>
    `;
    this.visualizer = new TandemStackVisualizer(
      this.root.querySelector("#tandem-stack-visualizer") as HTMLElement,
    );
    const runBtn = this.root.querySelector("#tandem-run-btn") as HTMLButtonElement;
    runBtn.addEventListener("click", () => this.runSimulation());
  }

  private async runSimulation(): Promise<void> {
    const pathInput = this.root.querySelector("#tandem-config-path") as HTMLInputElement;
    const resultsDiv = this.root.querySelector("#tandem-results") as HTMLElement;
    resultsDiv.textContent = "Running…";

    const resp = await fetch("/simulate/tandem", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        config_path: pathInput.value,
        N_grid: 80,
        n_points: 40,
      }),
    });
    if (!resp.ok) {
      resultsDiv.textContent = `Error: ${resp.status} ${await resp.text()}`;
      return;
    }
    const data: TandemJVPayload = await resp.json();
    this.renderResults(data, resultsDiv);
  }

  private renderResults(data: TandemJVPayload, host: HTMLElement): void {
    const m = data.metrics;
    const lines = [
      `V_oc = ${m.V_oc.toFixed(3)} V`,
      `J_sc = ${(m.J_sc / 10).toFixed(2)} mA/cm²`,
      `FF   = ${m.FF.toFixed(3)}`,
      `PCE  = ${m.PCE.toFixed(2)} %`,
    ];
    if (data.benchmark) {
      lines.push("");
      lines.push(`Benchmark: ${data.benchmark.reference}`);
      lines.push(`  target PCE = ${data.benchmark.targetPce} %`);
      lines.push(`  tolerance  = ±${data.benchmark.tolerancePct}%`);
    }
    host.innerHTML = `<pre>${lines.join("\n")}</pre>`;
  }
}
```

- [ ] **Step 5: Register Tandem tab in the app shell**

Edit `frontend/src/App.ts` (or the router wherever tabs are registered) following the pattern used by the existing Layer builder tab — import `TandemPanel`, add a new tab entry with label "Tandem" and a mount point, and call `panel.mount()` when selected.

- [ ] **Step 6: Smoke-test in the browser**

Start the dev server (`cd frontend && npm run dev` or the project's equivalent) and backend (`uvicorn backend.main:app --reload`). Open the Tandem tab, click "Run tandem J-V", verify that the results area shows metrics (real or stubbed) without throwing a console error.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/panels/tandem.ts \
        frontend/src/stack/tandem-stack-visualizer.ts \
        frontend/src/types.ts \
        frontend/src/App.ts
git commit -m "feat(frontend): add Tandem tab scaffold"
```

---

## Task 10: Full suite regression + final review

- [ ] **Step 1: Run the full unit suite**

Run: `pytest tests/unit -q`
Expected: all green.

- [ ] **Step 2: Run the fast integration suite**

Run: `pytest tests/integration -q -m "not slow"`
Expected: all green (tandem backend test passes; Lin 2019 benchmark skipped since it's `slow`).

- [ ] **Step 3: Run the slow suite**

Run: `pytest tests/integration -q -m slow`
Expected: Lin 2019 benchmark PASS or XFAIL (xfail only if n/k stub data is still in place).

- [ ] **Step 4: Manual UI smoke test**

Load the Tandem tab, run the default Lin 2019 config, confirm: two sub-stacks render, junction block visible, metrics panel populates, no console errors. If n/k data is stubbed, document that the metrics are not expected to match Lin 2019 yet.

- [ ] **Step 5: Mark v1 complete**

```bash
git log --oneline origin/main..HEAD
```

Expected: commits from Tasks 1–9 in order. Push feature branch: `git push -u origin feat/tandem-cell`. Open PR with spec reference in description.

---

## Spec coverage check (self-review)

| Spec section | Covered by |
|---|---|
| §1 Goal, v1 scope | Tasks 1–9 |
| §2 Lin 2019 benchmark anchor | Tasks 5, 6, 8 |
| §3 Independent sub-cells architecture | Tasks 3, 4 |
| §4.1 Combined TMM + partitioning | Task 2 |
| §4.2 Series match via interpolation | Task 3 |
| §5 YAML schema (referenced files + junction block) | Tasks 1, 6 |
| §6 File structure | Tasks 1–9 (maps 1:1) |
| §7 Data flow | Tasks 2, 3, 7 |
| §8.1 Unit tests — optics partitioning | Task 2 |
| §8.1 Unit tests — series match | Task 3 |
| §8.1 Unit tests — loader validation | Task 1 |
| §8.2 Slow benchmark integration test | Task 8 |
| §8.3 Backend test | Task 7 |
| §8.4 Frontend manual test | Tasks 9, 10 |
| §9 Out of scope | Explicitly not touched |

All spec requirements have an implementing task.

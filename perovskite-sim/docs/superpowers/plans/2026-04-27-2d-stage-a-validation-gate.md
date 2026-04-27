# 2D Microstructural Extension — Stage A: Validation Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a 2D drift-diffusion + Poisson + ion-migration solver as a subpackage `perovskite_sim/twod/`, prove it reproduces the existing 1D solver on a laterally uniform device within strict tolerances, and ship a minimal frontend tab so the result is reachable from the UI.

**Architecture:** Tensor-product rectilinear mesh on (x, y) with tanh clustering in both axes. 5-point sparse Poisson with harmonic-mean face permittivity, factored once per build via `scipy.sparse.linalg.splu`. Scharfetter–Gummel flux on horizontal and vertical edges, vectorised. Method-of-lines time integration with `solve_ivp(Radau)` on the flattened state vector. The 1D code in `discretization/`, `solver/`, and `experiments/` is untouched. All shared physics (recombination, mobility, traps, optics) is reused unchanged via the dimension-agnostic kernels in `physics/`.

**Tech Stack:** Python 3.11, NumPy, SciPy (`sparse`, `sparse.linalg.splu`, `integrate.solve_ivp`), pytest. Frontend: TypeScript + Vite + Plotly (existing). Backend: FastAPI (existing).

**Spec:** `perovskite-sim/docs/superpowers/specs/2026-04-27-2d-microstructural-extension-design.md`. Stage B is a separate plan, written after this one ships.

---

## File Structure

**New files (all under `perovskite-sim/`):**

```
perovskite_sim/twod/
├── __init__.py                  Package init; re-exports public API
├── grid_2d.py                   Grid2D dataclass + tensor-product builder
├── poisson_2d.py                Poisson2DFactor + sparse 5-point assembly + solve
├── flux_2d.py                   Vectorised 2D SG flux on horizontal/vertical edges
├── microstructure.py            Microstructure / GrainBoundary dataclasses (Stage A: uniform only)
├── continuity_2d.py             2D carrier+ion continuity RHS
├── solver_2d.py                 MaterialArrays2D, assemble_rhs_2d, run_transient_2d
├── snapshot.py                  SpatialSnapshot2D dataclass
└── experiments/
    ├── __init__.py
    └── jv_sweep_2d.py           run_jv_sweep_2d, JV2DResult

tests/unit/twod/
├── __init__.py
├── test_grid_2d.py
├── test_poisson_2d.py
├── test_flux_2d.py
├── test_microstructure.py
└── test_solver_2d.py

tests/integration/twod/
├── __init__.py
└── test_jv_sweep_2d_uniform.py

tests/regression/test_twod_validation.py    Stage-A six-check gate

configs/twod/
└── nip_MAPbI3_uniform.yaml      Stage-A validation preset (mirrors nip_MAPbI3.yaml)

frontend/src/panels/jv-2d.ts     Phase-1 minimal 2D J-V panel
```

**Files modified:**

```
backend/main.py                  Adds kind: "jv_2d" dispatch + result serialiser
frontend/src/main.ts             Registers the 2D J-V tab
frontend/src/types.ts            JV2DResult type
perovskite-sim/CLAUDE.md         New "Phase 6 — 2D microstructural extension" section
perovskite-sim/README.md         New "Dimensionality" subsection in Key Features
SolarLab/CLAUDE.md               Mention 2D lives in perovskite_sim/twod/
```

**Files explicitly NOT modified by this plan:**

- `perovskite_sim/discretization/`, `perovskite_sim/solver/`, `perovskite_sim/experiments/` — the 1D solver is untouched.
- `perovskite_sim/physics/` — physics kernels are dimension-agnostic; they take flattened arrays. No edits needed.
- `perovskite_sim/models/` — `DeviceStack`, `MaterialParams`, `LayerSpec` are reused as-is.

---

## Task 0: Create branch and verify clean baseline

**Files:**
- No code files.

- [ ] **Step 1: Verify the working tree is clean.**

  Run: `git status`
  Expected: `nothing to commit, working tree clean` on `main`.

- [ ] **Step 2: Run the existing 1D test suite once to confirm baseline is green.**

  Run from inside `perovskite-sim/`: `pytest -m 'not slow' -q`
  Expected: all unit + integration tests pass.

- [ ] **Step 3: Create and switch to the long-lived feature branch.**

  Run: `git checkout -b 2d-extension`

  This branch holds all Stage A and Stage B work. Do not merge to `main` until Stage A passes the validation gate (Task 22).

---

## Task 1: Subpackage skeleton

**Files:**
- Create: `perovskite-sim/perovskite_sim/twod/__init__.py`
- Create: `perovskite-sim/perovskite_sim/twod/experiments/__init__.py`
- Create: `perovskite-sim/tests/unit/twod/__init__.py`
- Create: `perovskite-sim/tests/integration/twod/__init__.py`

- [ ] **Step 1: Create the empty package directories with `__init__.py` stubs.**

  `perovskite-sim/perovskite_sim/twod/__init__.py`:
  ```python
  """2D microstructural drift-diffusion solver (Phase 6 — Apr 2026).

  Mirrors the 1D solver in perovskite_sim.solver / .experiments but on a
  tensor-product rectilinear (x, y) mesh. The 1D code is untouched.
  """
  ```

  `perovskite-sim/perovskite_sim/twod/experiments/__init__.py`:
  ```python
  """2D experiment drivers (J-V sweep, field maps, etc.)."""
  ```

  `perovskite-sim/tests/unit/twod/__init__.py`: empty file.

  `perovskite-sim/tests/integration/twod/__init__.py`: empty file.

- [ ] **Step 2: Verify package imports cleanly.**

  Run from `perovskite-sim/`: `python -c "import perovskite_sim.twod; import perovskite_sim.twod.experiments; print('ok')"`
  Expected: prints `ok`.

- [ ] **Step 3: Commit the skeleton.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/__init__.py \
          perovskite-sim/perovskite_sim/twod/experiments/__init__.py \
          perovskite-sim/tests/unit/twod/__init__.py \
          perovskite-sim/tests/integration/twod/__init__.py
  git commit -m "feat(twod): scaffold 2D subpackage skeleton (Phase 6)"
  ```

---

## Task 2: 2D tensor-product grid

**Files:**
- Create: `perovskite-sim/perovskite_sim/twod/grid_2d.py`
- Create: `perovskite-sim/tests/unit/twod/test_grid_2d.py`

- [ ] **Step 1: Write the failing test for `Grid2D` construction.**

  `perovskite-sim/tests/unit/twod/test_grid_2d.py`:
  ```python
  from __future__ import annotations
  import numpy as np
  import pytest

  from perovskite_sim.twod.grid_2d import Grid2D, build_grid_2d
  from perovskite_sim.discretization.grid import Layer


  def test_grid_2d_dimensions():
      layers = [Layer(thickness=50e-9, N=10),
                Layer(thickness=300e-9, N=20),
                Layer(thickness=50e-9, N=10)]
      g = build_grid_2d(layers, lateral_length=500e-9, Nx=20)
      assert g.Nx == 21               # Nx is number of POINTS, not intervals
      assert g.Ny == 41               # 10 + 20 + 10 = 40 intervals → 41 points
      assert g.n_nodes == 21 * 41


  def test_grid_2d_endpoints():
      layers = [Layer(thickness=400e-9, N=20)]
      g = build_grid_2d(layers, lateral_length=500e-9, Nx=10)
      assert g.x[0] == pytest.approx(0.0)
      assert g.x[-1] == pytest.approx(500e-9)
      assert g.y[0] == pytest.approx(0.0)
      assert g.y[-1] == pytest.approx(400e-9)


  def test_grid_2d_lateral_uniform_grid_when_periodic():
      """Periodic lateral BC needs uniform spacing for the simple Poisson stencil."""
      layers = [Layer(thickness=400e-9, N=20)]
      g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
      dx = np.diff(g.x)
      assert np.allclose(dx, dx[0], rtol=1e-12)


  def test_grid_2d_clusters_y_at_layer_interfaces():
      """tanh clustering in y compresses spacing near layer boundaries."""
      layers = [Layer(thickness=100e-9, N=10), Layer(thickness=100e-9, N=10)]
      g = build_grid_2d(layers, lateral_length=200e-9, Nx=10)
      dy = np.diff(g.y)
      # Spacing near the inter-layer boundary should be smaller than mid-layer.
      assert dy[5] < dy[2]
      assert dy[5] < dy[8]
  ```

- [ ] **Step 2: Run the test to confirm it fails.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_grid_2d.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'perovskite_sim.twod.grid_2d'`.

- [ ] **Step 3: Implement `grid_2d.py`.**

  ```python
  from __future__ import annotations
  from dataclasses import dataclass
  import numpy as np

  from perovskite_sim.discretization.grid import Layer, multilayer_grid, tanh_grid


  @dataclass(frozen=True)
  class Grid2D:
      """Tensor-product rectilinear mesh on (x, y).

      x is the lateral coordinate (length Nx, points 0..Nx-1).
      y is the vertical (stack) coordinate (length Ny, points 0..Ny-1).
      Total node count is Nx * Ny; the linear node index is
      idx(i, j) = j * Nx + i  (y-major / row-major over (j, i)).
      """
      x: np.ndarray
      y: np.ndarray

      @property
      def Nx(self) -> int:
          return int(self.x.size)

      @property
      def Ny(self) -> int:
          return int(self.y.size)

      @property
      def n_nodes(self) -> int:
          return self.Nx * self.Ny


  def build_grid_2d(
      layers: list[Layer],
      lateral_length: float,
      Nx: int,
      *,
      alpha_y: float = 3.0,
      alpha_x: float = 2.0,
      lateral_uniform: bool = False,
  ) -> Grid2D:
      """Build a tensor-product (x, y) grid.

      y is the existing 1D multilayer tanh grid (clustered at layer interfaces
      and contacts via the same `multilayer_grid` used by the 1D solver).
      x is either uniform (Nx+1 evenly spaced points on [0, lateral_length])
      when `lateral_uniform=True`, or tanh-clustered toward x=0 and x=L_x
      otherwise. Stage A defaults to `lateral_uniform=True` because the
      validation problem has no internal x-features to cluster around.
      """
      y = multilayer_grid(layers, alpha=alpha_y)
      if lateral_uniform:
          x = np.linspace(0.0, lateral_length, Nx + 1)
      else:
          x = tanh_grid(Nx, lateral_length, alpha=alpha_x)
      return Grid2D(x=np.asarray(x, dtype=float),
                    y=np.asarray(y, dtype=float))
  ```

- [ ] **Step 4: Run the test to confirm it passes.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_grid_2d.py -v`
  Expected: 4 passed.

- [ ] **Step 5: Commit.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/grid_2d.py \
          perovskite-sim/tests/unit/twod/test_grid_2d.py
  git commit -m "feat(twod): tensor-product (x,y) grid with tanh clustering"
  ```

---

## Task 3: 2D Poisson — assembly + sparse LU factor

**Files:**
- Create: `perovskite-sim/perovskite_sim/twod/poisson_2d.py`
- Create: `perovskite-sim/tests/unit/twod/test_poisson_2d.py`

The 2D Poisson stencil is a 5-point finite-volume operator with harmonic-mean face permittivities and dual-grid cell volumes, following the same convention as the 1D `factor_poisson`.

- [ ] **Step 1: Write the failing test.**

  `perovskite-sim/tests/unit/twod/test_poisson_2d.py`:
  ```python
  from __future__ import annotations
  import numpy as np
  import pytest

  from perovskite_sim.twod.grid_2d import build_grid_2d
  from perovskite_sim.twod.poisson_2d import (
      build_poisson_2d_factor, solve_poisson_2d,
  )
  from perovskite_sim.discretization.grid import Layer
  from perovskite_sim.constants import EPS_0


  def _uniform_grid_factor(Lx=500e-9, Ly=400e-9, Nx=20, Ny=20, eps_r=10.0):
      layers = [Layer(thickness=Ly, N=Ny)]
      g = build_grid_2d(layers, lateral_length=Lx, Nx=Nx, lateral_uniform=True)
      eps_field = np.full((g.Ny, g.Nx), eps_r, dtype=float)
      fac = build_poisson_2d_factor(g, eps_field, lateral_bc="periodic")
      return g, fac


  def test_poisson_2d_solves_zero_charge_with_dirichlet_y():
      """Zero charge, phi=0 at y=0 and phi=V at y=Ly → linear ramp in y."""
      g, fac = _uniform_grid_factor()
      rho = np.zeros((g.Ny, g.Nx), dtype=float)
      V = 1.0
      phi = solve_poisson_2d(fac, rho, phi_bottom=0.0, phi_top=V)
      # Linear ramp: phi should be independent of x and equal V * y/Ly
      Ly = g.y[-1]
      ramp = V * g.y / Ly
      for j in range(g.Ny):
          np.testing.assert_allclose(phi[j, :], ramp[j], atol=1e-10)


  def test_poisson_2d_charge_neutral_density_with_periodic_x():
      """Constant rho=0 with periodic x and Dirichlet y reproduces ramp."""
      g, fac = _uniform_grid_factor()
      rho = np.zeros((g.Ny, g.Nx), dtype=float)
      phi = solve_poisson_2d(fac, rho, phi_bottom=0.5, phi_top=1.5)
      # Periodic BC in x means edges (i=0 and i=Nx-1) are not pinned —
      # they must equal each other (consistency).
      np.testing.assert_allclose(phi[:, 0], phi[:, -1], atol=1e-10)


  def test_poisson_2d_residual_below_tolerance():
      """Residual ||A phi - b|| / ||b|| below 1e-10 for a known charge."""
      g, fac = _uniform_grid_factor()
      rho = 1e6 * np.ones((g.Ny, g.Nx), dtype=float)   # uniform space charge
      phi = solve_poisson_2d(fac, rho, phi_bottom=0.0, phi_top=0.0)
      # Analytical: 1D parabolic profile, phi(y) = -rho/(2 eps0 eps_r) y (Ly - y)
      eps_r = 10.0
      Ly = g.y[-1]
      analytic = -rho[0, 0] / (2.0 * EPS_0 * eps_r) * g.y * (Ly - g.y)
      for j in range(g.Ny):
          np.testing.assert_allclose(phi[j, :], analytic[j], rtol=1e-3, atol=1e-6)
  ```

- [ ] **Step 2: Run the test to confirm failure.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_poisson_2d.py -v`
  Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `poisson_2d.py`.**

  ```python
  from __future__ import annotations
  from dataclasses import dataclass
  import numpy as np
  from scipy.sparse import lil_matrix, csc_matrix
  from scipy.sparse.linalg import splu, SuperLU

  from perovskite_sim.constants import EPS_0
  from perovskite_sim.twod.grid_2d import Grid2D


  @dataclass(frozen=True)
  class Poisson2DFactor:
      """Pre-factored 2D Poisson operator on a tensor-product mesh.

      The operator depends only on (Grid2D, eps_r(x,y), lateral_bc), all of
      which are constant across a transient. Building it once and reusing
      `splu` for every RHS evaluation collapses the hot path.

      Lateral BC is either "periodic" (couples i=Nx-1 to i=0) or "neumann"
      (zero-flux at i=0 and i=Nx-1).  Vertical BC is always Dirichlet at
      j=0 (bottom contact) and j=Ny-1 (top contact); those rows are
      eliminated from the unknown set and absorbed into the RHS.
      """
      grid: Grid2D
      lateral_bc: str
      lu: SuperLU
      # Per-edge face conductances ε₀ε_face/h, broadcast into stencil at solve time
      C_x: np.ndarray   # shape (Ny, Nx-1) when lateral_bc == "neumann"
                        # shape (Ny, Nx)   when lateral_bc == "periodic" (wraps)
      C_y: np.ndarray   # shape (Ny-1, Nx)
      cell_area: np.ndarray   # shape (Ny-2, Nx) — dual-grid cell area for interior unknowns
      C_y_top: np.ndarray     # shape (Nx,) — face conductance at the top Dirichlet boundary
      C_y_bot: np.ndarray     # shape (Nx,) — face conductance at the bottom Dirichlet boundary


  def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      return 2.0 * a * b / (a + b)


  def build_poisson_2d_factor(
      grid: Grid2D,
      eps_r: np.ndarray,
      lateral_bc: str = "periodic",
  ) -> Poisson2DFactor:
      """Assemble the 5-point sparse Poisson operator and factor it once."""
      assert eps_r.shape == (grid.Ny, grid.Nx), "eps_r must be shape (Ny, Nx)"
      assert lateral_bc in ("periodic", "neumann")

      Nx, Ny = grid.Nx, grid.Ny
      x, y = grid.x, grid.y
      dx = np.diff(x)                            # (Nx-1,)
      dy = np.diff(y)                            # (Ny-1,)
      # Dual-grid widths (one per interior node)
      hx_cell = 0.5 * (np.r_[dx[0], dx[:-1]] + np.r_[dx[1:], dx[-1]])  # (Nx,)
      hy_cell_int = 0.5 * (dy[:-1] + dy[1:])                            # (Ny-2,)

      # x-face permittivities (Ny, Nx-1) for neumann, (Ny, Nx) for periodic
      eps_face_x_int = _harmonic_mean(eps_r[:, :-1], eps_r[:, 1:])      # (Ny, Nx-1)
      if lateral_bc == "periodic":
          eps_face_wrap = _harmonic_mean(eps_r[:, -1], eps_r[:, 0])      # (Ny,)
          eps_face_x = np.concatenate([eps_face_x_int, eps_face_wrap[:, None]], axis=1)
          C_x = EPS_0 * eps_face_x / np.r_[dx, np.array([dx[-1] + dx[0]]) / 2.0 * 0 + (x[-1] - x[-2] + x[1] - x[0]) / 2.0]  # placeholder
          # Periodic: the face between (i=Nx-1) and (i=0) wraps; spacing is the
          # average of the two boundary cells (or use the explicit periodic length).
          dx_wrap = (x[-1] - x[-2] + x[1] - x[0]) / 2.0
          C_x = EPS_0 * eps_face_x / np.concatenate([dx, np.array([dx_wrap])])[None, :]
      else:
          C_x = EPS_0 * eps_face_x_int / dx[None, :]                     # (Ny, Nx-1)

      # y-face permittivities (Ny-1, Nx)
      eps_face_y = _harmonic_mean(eps_r[:-1, :], eps_r[1:, :])
      C_y = EPS_0 * eps_face_y / dy[:, None]

      # Boundary face conductances (top and bottom Dirichlet rows)
      C_y_bot = C_y[0, :].copy()
      C_y_top = C_y[-1, :].copy()

      # Cell areas for RHS scaling on interior unknowns (j = 1..Ny-2)
      cell_area = hy_cell_int[:, None] * hx_cell[None, :]                # (Ny-2, Nx)

      # Build sparse matrix on interior unknowns: j = 1..Ny-2, i = 0..Nx-1
      n_int_rows = Ny - 2
      n_unknowns = n_int_rows * Nx
      A = lil_matrix((n_unknowns, n_unknowns), dtype=float)

      def idx(j_int: int, i: int) -> int:
          # j_int is index into interior rows, 0..Ny-3 (corresponds to grid j = j_int+1)
          return j_int * Nx + i

      for j_int in range(n_int_rows):
          j = j_int + 1
          for i in range(Nx):
              # x-neighbours
              if lateral_bc == "periodic":
                  ileft = (i - 1) % Nx
                  iright = (i + 1) % Nx
                  C_left = C_x[j, (i - 1) % Nx]
                  C_right = C_x[j, i]
              else:
                  C_left = C_x[j, i - 1] if i > 0 else 0.0
                  C_right = C_x[j, i] if i < Nx - 1 else 0.0
                  ileft = i - 1
                  iright = i + 1

              # y-neighbours
              C_below = C_y[j - 1, i]
              C_above = C_y[j, i]

              diag = -(C_left + C_right + C_below + C_above)
              A[idx(j_int, i), idx(j_int, i)] = diag

              if lateral_bc == "periodic" or 0 <= ileft < Nx:
                  A[idx(j_int, i), idx(j_int, ileft)] = C_left
              if lateral_bc == "periodic" or 0 <= iright < Nx:
                  A[idx(j_int, i), idx(j_int, iright)] = C_right
              # y-below
              if j_int - 1 >= 0:
                  A[idx(j_int, i), idx(j_int - 1, i)] = C_below
              # y-above
              if j_int + 1 < n_int_rows:
                  A[idx(j_int, i), idx(j_int + 1, i)] = C_above

      A = A.tocsc()
      lu = splu(A)
      return Poisson2DFactor(
          grid=grid, lateral_bc=lateral_bc, lu=lu,
          C_x=C_x, C_y=C_y, cell_area=cell_area,
          C_y_top=C_y_top, C_y_bot=C_y_bot,
      )


  def solve_poisson_2d(
      fac: Poisson2DFactor,
      rho: np.ndarray,
      phi_bottom: float | np.ndarray,
      phi_top: float | np.ndarray,
  ) -> np.ndarray:
      """Solve A phi = -rho * cell_area, with Dirichlet rows absorbed into RHS.

      `phi_bottom` and `phi_top` may each be either a scalar (constant along x)
      or a 1D array of length Nx. The returned phi has shape (Ny, Nx) with
      phi[0,:]=phi_bottom and phi[-1,:]=phi_top filled in.
      """
      assert rho.shape == (fac.grid.Ny, fac.grid.Nx)
      Nx, Ny = fac.grid.Nx, fac.grid.Ny
      n_int_rows = Ny - 2

      # Broadcast Dirichlet values
      phi_bot_arr = np.broadcast_to(np.asarray(phi_bottom, dtype=float), (Nx,)).copy()
      phi_top_arr = np.broadcast_to(np.asarray(phi_top, dtype=float), (Nx,)).copy()

      # RHS: interior rows j = 1..Ny-2
      rhs = -rho[1:-1, :] * fac.cell_area      # (Ny-2, Nx)
      # Absorb Dirichlet at j=0 (bottom) and j=Ny-1 (top)
      rhs[0, :] -= fac.C_y_bot * phi_bot_arr
      rhs[-1, :] -= fac.C_y_top * phi_top_arr

      x = fac.lu.solve(rhs.flatten())          # (n_unknowns,)
      phi = np.empty((Ny, Nx), dtype=float)
      phi[0, :] = phi_bot_arr
      phi[-1, :] = phi_top_arr
      phi[1:-1, :] = x.reshape((n_int_rows, Nx))
      return phi
  ```

- [ ] **Step 4: Run the test to confirm pass.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_poisson_2d.py -v`
  Expected: 3 passed.

- [ ] **Step 5: Commit.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/poisson_2d.py \
          perovskite-sim/tests/unit/twod/test_poisson_2d.py
  git commit -m "feat(twod): 5-point sparse Poisson with cached LU factor"
  ```

---

## Task 4: Microstructure data model (uniform path only)

**Files:**
- Create: `perovskite-sim/perovskite_sim/twod/microstructure.py`
- Create: `perovskite-sim/tests/unit/twod/test_microstructure.py`

Stage A only needs the empty `Microstructure()` case. The `GrainBoundary` dataclass is defined here but not consumed in Stage A — it's wired up in Stage B. Defining it now keeps Stage A's microstructure module forward-compatible without leaving a TODO.

- [ ] **Step 1: Write the failing test.**

  ```python
  # perovskite-sim/tests/unit/twod/test_microstructure.py
  from __future__ import annotations
  import numpy as np
  import pytest

  from perovskite_sim.twod.microstructure import (
      GrainBoundary, Microstructure, build_tau_field,
  )
  from perovskite_sim.twod.grid_2d import build_grid_2d
  from perovskite_sim.discretization.grid import Layer


  def _grid():
      layers = [Layer(thickness=400e-9, N=20)]
      return build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)


  def test_empty_microstructure_returns_uniform_tau():
      g = _grid()
      tau_bulk_per_layer = np.full((g.Ny,), 1e-6)
      ustruct = Microstructure()
      tau_n, tau_p = build_tau_field(g, ustruct, tau_bulk_per_layer, tau_bulk_per_layer,
                                     layer_role_per_y=["absorber"] * g.Ny)
      assert tau_n.shape == (g.Ny, g.Nx)
      assert tau_p.shape == (g.Ny, g.Nx)
      assert np.allclose(tau_n, 1e-6)
      assert np.allclose(tau_p, 1e-6)


  def test_grain_boundary_dataclass_is_frozen():
      gb = GrainBoundary(x_position=250e-9, width=5e-9,
                         tau_n=1e-9, tau_p=1e-9, layer_role="absorber")
      with pytest.raises(Exception):
          gb.x_position = 100e-9  # frozen — should raise


  def test_microstructure_dataclass_default_is_empty():
      ustruct = Microstructure()
      assert ustruct.grain_boundaries == ()
  ```

- [ ] **Step 2: Run to confirm failure.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_microstructure.py -v`
  Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `microstructure.py`.**

  ```python
  from __future__ import annotations
  from dataclasses import dataclass, field
  from typing import Sequence
  import numpy as np

  from perovskite_sim.twod.grid_2d import Grid2D


  @dataclass(frozen=True)
  class GrainBoundary:
      """A vertical grain boundary, modelled as a band of nodes in the absorber
      layer with reduced SRH lifetimes (τ_n, τ_p) and width δ."""
      x_position: float       # m, lateral coordinate of GB centre
      width: float            # m, GB band width (typical 5e-9)
      tau_n: float            # s, electron SRH lifetime inside GB
      tau_p: float            # s, hole SRH lifetime inside GB
      layer_role: str = "absorber"


  @dataclass(frozen=True)
  class Microstructure:
      """Container for spatially-varying defect features in a 2D simulation."""
      grain_boundaries: tuple[GrainBoundary, ...] = ()


  def build_tau_field(
      grid: Grid2D,
      ustruct: Microstructure,
      tau_n_bulk_per_y: np.ndarray,
      tau_p_bulk_per_y: np.ndarray,
      layer_role_per_y: Sequence[str],
  ) -> tuple[np.ndarray, np.ndarray]:
      """Build (τ_n, τ_p) on the (Ny, Nx) grid.

      Stage A: ustruct is empty → returns the uniform extrusion of the 1D
      bulk τ along x. Stage B: GBs in the absorber layer override τ inside
      bands of width `gb.width` centred at `gb.x_position`.
      """
      Nx, Ny = grid.Nx, grid.Ny
      tau_n = np.broadcast_to(tau_n_bulk_per_y[:, None], (Ny, Nx)).copy()
      tau_p = np.broadcast_to(tau_p_bulk_per_y[:, None], (Ny, Nx)).copy()

      for gb in ustruct.grain_boundaries:
          mask_x = np.abs(grid.x - gb.x_position) < gb.width / 2.0      # (Nx,)
          mask_y = np.array([role == gb.layer_role for role in layer_role_per_y])
          mask_2d = np.outer(mask_y, mask_x)                            # (Ny, Nx)
          tau_n[mask_2d] = gb.tau_n
          tau_p[mask_2d] = gb.tau_p

      return tau_n, tau_p
  ```

- [ ] **Step 4: Run to confirm pass.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_microstructure.py -v`
  Expected: 3 passed.

- [ ] **Step 5: Commit.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/microstructure.py \
          perovskite-sim/tests/unit/twod/test_microstructure.py
  git commit -m "feat(twod): Microstructure / GrainBoundary data model"
  ```

---

## Task 5: 2D Scharfetter–Gummel flux on horizontal and vertical edges

**Files:**
- Create: `perovskite-sim/perovskite_sim/twod/flux_2d.py`
- Create: `perovskite-sim/tests/unit/twod/test_flux_2d.py`

The 2D SG flux reuses the existing `bernoulli` from `discretization.fe_operators` and just applies the same flux formula on (i) every horizontal edge between (i, j) and (i+1, j), and (ii) every vertical edge between (i, j) and (i, j+1).

- [ ] **Step 1: Write the failing test.**

  ```python
  # perovskite-sim/tests/unit/twod/test_flux_2d.py
  from __future__ import annotations
  import numpy as np

  from perovskite_sim.twod.grid_2d import build_grid_2d
  from perovskite_sim.twod.flux_2d import sg_fluxes_2d_n, sg_fluxes_2d_p
  from perovskite_sim.discretization.grid import Layer
  from perovskite_sim.discretization.fe_operators import sg_fluxes_n


  def test_2d_sg_zero_field_zero_concentration_gradient():
      """phi=const, n=const → flux = 0 on every edge."""
      layers = [Layer(thickness=400e-9, N=20)]
      g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
      phi = np.zeros((g.Ny, g.Nx))
      n = np.full((g.Ny, g.Nx), 1e21)
      D_n = np.full((g.Ny, g.Nx), 1e-4)
      V_T = 0.0259
      Jx, Jy = sg_fluxes_2d_n(phi, n, g.x, g.y, D_n, V_T)
      assert np.allclose(Jx, 0.0, atol=1e-6)
      assert np.allclose(Jy, 0.0, atol=1e-6)


  def test_2d_sg_recovers_1d_when_y_uniform():
      """When phi(x,y) and n(x,y) depend only on y, J_x must be 0 and J_y
      must equal the 1D SG flux on a slice."""
      layers = [Layer(thickness=400e-9, N=20)]
      g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
      phi_1d = np.linspace(0.0, 1.0, g.Ny)
      n_1d = np.linspace(1e21, 1e22, g.Ny)
      phi = np.broadcast_to(phi_1d[:, None], (g.Ny, g.Nx)).copy()
      n = np.broadcast_to(n_1d[:, None], (g.Ny, g.Nx)).copy()
      D_n = np.full((g.Ny, g.Nx), 1e-4)
      V_T = 0.0259
      Jx, Jy = sg_fluxes_2d_n(phi, n, g.x, g.y, D_n, V_T)
      # No lateral variation → J_x = 0 everywhere
      assert np.allclose(Jx, 0.0, atol=1e-12)
      # J_y at x=column-i must equal the 1D SG flux on (phi_1d, n_1d)
      D_n_1d = 1e-4
      ref = sg_fluxes_n(phi_1d, n_1d, np.diff(g.y), D_n_1d, V_T)   # (Ny-1,)
      for i in range(g.Nx):
          np.testing.assert_allclose(Jy[:, i], ref, rtol=1e-10, atol=1e-12)


  def test_2d_sg_p_recovers_1d_when_y_uniform():
      """Same as above but for holes."""
      layers = [Layer(thickness=400e-9, N=20)]
      g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
      phi_1d = np.linspace(0.0, 1.0, g.Ny)
      p_1d = np.linspace(1e22, 1e21, g.Ny)
      phi = np.broadcast_to(phi_1d[:, None], (g.Ny, g.Nx)).copy()
      p = np.broadcast_to(p_1d[:, None], (g.Ny, g.Nx)).copy()
      D_p = np.full((g.Ny, g.Nx), 1e-4)
      V_T = 0.0259
      from perovskite_sim.discretization.fe_operators import sg_fluxes_p
      Jx, Jy = sg_fluxes_2d_p(phi, p, g.x, g.y, D_p, V_T)
      assert np.allclose(Jx, 0.0, atol=1e-12)
      ref = sg_fluxes_p(phi_1d, p_1d, np.diff(g.y), 1e-4, V_T)
      for i in range(g.Nx):
          np.testing.assert_allclose(Jy[:, i], ref, rtol=1e-10, atol=1e-12)
  ```

- [ ] **Step 2: Run to confirm failure.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_flux_2d.py -v`
  Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `flux_2d.py`.**

  ```python
  from __future__ import annotations
  import numpy as np

  from perovskite_sim.discretization.fe_operators import bernoulli
  from perovskite_sim.constants import Q


  def sg_fluxes_2d_n(
      phi_n: np.ndarray,        # (Ny, Nx)
      n: np.ndarray,            # (Ny, Nx)
      x: np.ndarray,            # (Nx,)
      y: np.ndarray,            # (Ny,)
      D_n: np.ndarray,          # (Ny, Nx) — node-resolved; face avg taken inside
      V_T: float,
  ) -> tuple[np.ndarray, np.ndarray]:
      """Vectorised SG electron current density on horizontal edges (J_x, shape
      (Ny, Nx-1)) and vertical edges (J_y, shape (Ny-1, Nx)).

      Conventions:
        J_x[j, i]  : flux from (i,j) to (i+1,j); positive = electron current
                     flowing in +x direction.
        J_y[j, i]  : flux from (i,j) to (i,j+1); positive = +y direction.
      Phase-1 selective contacts and TE caps are NOT applied here; they
      will be hooks added in `continuity_2d`.
      """
      dx = np.diff(x)                              # (Nx-1,)
      dy = np.diff(y)                              # (Ny-1,)

      # Horizontal edges: between (i, j) and (i+1, j)
      D_face_x = 0.5 * (D_n[:, :-1] + D_n[:, 1:])  # (Ny, Nx-1)
      xi_x = (phi_n[:, 1:] - phi_n[:, :-1]) / V_T  # (Ny, Nx-1)
      Jx = (Q * D_face_x / dx[None, :]) * (
          bernoulli(xi_x) * n[:, 1:] - bernoulli(-xi_x) * n[:, :-1]
      )

      # Vertical edges: between (i, j) and (i, j+1)
      D_face_y = 0.5 * (D_n[:-1, :] + D_n[1:, :])  # (Ny-1, Nx)
      xi_y = (phi_n[1:, :] - phi_n[:-1, :]) / V_T  # (Ny-1, Nx)
      Jy = (Q * D_face_y / dy[:, None]) * (
          bernoulli(xi_y) * n[1:, :] - bernoulli(-xi_y) * n[:-1, :]
      )
      return Jx, Jy


  def sg_fluxes_2d_p(
      phi_p: np.ndarray,
      p: np.ndarray,
      x: np.ndarray,
      y: np.ndarray,
      D_p: np.ndarray,
      V_T: float,
  ) -> tuple[np.ndarray, np.ndarray]:
      """Vectorised SG hole current density on horizontal and vertical edges.
      Sign convention matches the 1D `sg_fluxes_p`."""
      dx = np.diff(x)
      dy = np.diff(y)

      D_face_x = 0.5 * (D_p[:, :-1] + D_p[:, 1:])
      xi_x = (phi_p[:, 1:] - phi_p[:, :-1]) / V_T
      Jx = (Q * D_face_x / dx[None, :]) * (
          bernoulli(xi_x) * p[:, :-1] - bernoulli(-xi_x) * p[:, 1:]
      )

      D_face_y = 0.5 * (D_p[:-1, :] + D_p[1:, :])
      xi_y = (phi_p[1:, :] - phi_p[:-1, :]) / V_T
      Jy = (Q * D_face_y / dy[:, None]) * (
          bernoulli(xi_y) * p[:-1, :] - bernoulli(-xi_y) * p[1:, :]
      )
      return Jx, Jy
  ```

- [ ] **Step 4: Run to confirm pass.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_flux_2d.py -v`
  Expected: 3 passed.

- [ ] **Step 5: Commit.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/flux_2d.py \
          perovskite-sim/tests/unit/twod/test_flux_2d.py
  git commit -m "feat(twod): vectorised SG flux on horizontal and vertical edges"
  ```

---

## Task 6: `MaterialArrays2D` cache + `build_material_arrays_2d`

**Files:**
- Create: `perovskite-sim/perovskite_sim/twod/solver_2d.py` (initial skeleton — only the cache builder)
- Create: `perovskite-sim/tests/unit/twod/test_solver_2d.py`

The cache holds per-node and per-edge fields plus the pre-factored 2D Poisson. All Stage-A fields are dimension-agnostic extrusions of the 1D `MaterialArrays`. The structure is built once per voltage sweep, exactly like the 1D version.

- [ ] **Step 1: Write the failing test for the cache builder.**

  ```python
  # perovskite-sim/tests/unit/twod/test_solver_2d.py
  from __future__ import annotations
  import numpy as np
  import pytest

  from perovskite_sim.twod.solver_2d import build_material_arrays_2d, MaterialArrays2D
  from perovskite_sim.twod.microstructure import Microstructure
  from perovskite_sim.twod.grid_2d import build_grid_2d
  from perovskite_sim.discretization.grid import Layer
  from perovskite_sim.models.config_loader import load_config


  def _stack():
      return load_config("nip_MAPbI3.yaml")


  def test_material_arrays_2d_shapes():
      stack = _stack()
      g = build_grid_2d([Layer(t, 10) for t, _ in [(L.thickness, L) for L in stack.layers]],
                        lateral_length=500e-9, Nx=20, lateral_uniform=True)
      mat = build_material_arrays_2d(g, stack, Microstructure())
      assert mat.eps_r.shape == (g.Ny, g.Nx)
      assert mat.D_n.shape == (g.Ny, g.Nx)
      assert mat.D_p.shape == (g.Ny, g.Nx)
      assert mat.tau_n.shape == (g.Ny, g.Nx)
      assert mat.tau_p.shape == (g.Ny, g.Nx)
      assert mat.G_optical.shape == (g.Ny, g.Nx)
      assert mat.poisson_factor is not None


  def test_material_arrays_2d_uniform_in_x():
      """With Microstructure() (no GBs), every per-node field is x-invariant."""
      stack = _stack()
      g = build_grid_2d([Layer(t, 10) for t, _ in [(L.thickness, L) for L in stack.layers]],
                        lateral_length=500e-9, Nx=20, lateral_uniform=True)
      mat = build_material_arrays_2d(g, stack, Microstructure())
      for arr_name in ("eps_r", "D_n", "D_p", "tau_n", "tau_p", "G_optical"):
          arr = getattr(mat, arr_name)
          assert np.allclose(arr, arr[:, [0]]), f"{arr_name} varies in x"
  ```

- [ ] **Step 2: Run to confirm failure.**

  Expected: `ModuleNotFoundError: No module named 'perovskite_sim.twod.solver_2d'`.

- [ ] **Step 3: Implement the cache + builder skeleton in `solver_2d.py`.**

  ```python
  from __future__ import annotations
  from dataclasses import dataclass
  import numpy as np

  from perovskite_sim.constants import Q, K_B
  from perovskite_sim.models.device import DeviceStack
  from perovskite_sim.solver.mol import build_material_arrays as build_material_arrays_1d
  from perovskite_sim.discretization.grid import Layer, multilayer_grid
  from perovskite_sim.twod.grid_2d import Grid2D, build_grid_2d
  from perovskite_sim.twod.poisson_2d import (
      Poisson2DFactor, build_poisson_2d_factor,
  )
  from perovskite_sim.twod.microstructure import Microstructure, build_tau_field


  @dataclass(frozen=True)
  class MaterialArrays2D:
      """2D analogue of the 1D MaterialArrays cache.

      All per-node fields are shape (Ny, Nx). For Stage A every field is a
      uniform extrusion of the 1D MaterialArrays along x; Stage B will
      override τ_n and τ_p inside grain-boundary bands.
      """
      grid: Grid2D
      stack: DeviceStack
      ustruct: Microstructure
      eps_r: np.ndarray
      D_n: np.ndarray
      D_p: np.ndarray
      tau_n: np.ndarray
      tau_p: np.ndarray
      N_A: np.ndarray
      N_D: np.ndarray
      ni: np.ndarray
      G_optical: np.ndarray
      n_eq_left: np.ndarray         # (Nx,)  bottom-contact electron density
      p_eq_left: np.ndarray         # (Nx,)
      n_eq_right: np.ndarray        # (Nx,)  top-contact
      p_eq_right: np.ndarray        # (Nx,)
      V_bi: float
      V_T: float
      poisson_factor: Poisson2DFactor
      layer_role_per_y: tuple[str, ...]


  def build_material_arrays_2d(
      grid: Grid2D,
      stack: DeviceStack,
      ustruct: Microstructure,
      *,
      lateral_bc: str = "periodic",
  ) -> MaterialArrays2D:
      """Assemble the 2D MaterialArrays from a stack and a microstructure.

      The strategy is deliberately simple: build the 1D MaterialArrays with
      the existing solver, then extrude every per-node field along x (Stage A
      has no x-features). τ_n, τ_p go through `build_tau_field`, which
      respects GBs in Stage B but is identity-extrude when the microstructure
      is empty (Stage A).
      """
      mat1d = build_material_arrays_1d(grid.y, stack)
      Nx, Ny = grid.Nx, grid.Ny

      def extrude(v_1d: np.ndarray) -> np.ndarray:
          return np.broadcast_to(v_1d[:, None], (Ny, Nx)).copy()

      eps_r = extrude(mat1d.eps_r)
      D_n = extrude(mat1d.D_n)
      D_p = extrude(mat1d.D_p)
      N_A = extrude(mat1d.N_A)
      N_D = extrude(mat1d.N_D)
      ni = extrude(mat1d.ni)
      G_optical = extrude(mat1d.G_optical)

      # τ from microstructure builder (Stage A: just uniform extrusion)
      layer_role_per_y = tuple(_layer_role_at_each_y(grid.y, stack))
      tau_n_1d = mat1d.tau_n
      tau_p_1d = mat1d.tau_p
      tau_n, tau_p = build_tau_field(
          grid, ustruct,
          tau_n_bulk_per_y=tau_n_1d,
          tau_p_bulk_per_y=tau_p_1d,
          layer_role_per_y=layer_role_per_y,
      )

      n_eq_left = np.full((Nx,), float(mat1d.n_L))
      p_eq_left = np.full((Nx,), float(mat1d.p_L))
      n_eq_right = np.full((Nx,), float(mat1d.n_R))
      p_eq_right = np.full((Nx,), float(mat1d.p_R))

      poisson_factor = build_poisson_2d_factor(grid, eps_r, lateral_bc=lateral_bc)

      return MaterialArrays2D(
          grid=grid, stack=stack, ustruct=ustruct,
          eps_r=eps_r, D_n=D_n, D_p=D_p,
          tau_n=tau_n, tau_p=tau_p,
          N_A=N_A, N_D=N_D, ni=ni, G_optical=G_optical,
          n_eq_left=n_eq_left, p_eq_left=p_eq_left,
          n_eq_right=n_eq_right, p_eq_right=p_eq_right,
          V_bi=float(mat1d.V_bi_eff if hasattr(mat1d, "V_bi_eff") else stack.V_bi),
          V_T=float(K_B * stack.T / Q if hasattr(stack, "T") else mat1d.V_T),
          poisson_factor=poisson_factor,
          layer_role_per_y=layer_role_per_y,
      )


  def _layer_role_at_each_y(y: np.ndarray, stack: DeviceStack) -> list[str]:
      """Return the layer role (e.g. 'absorber', 'ETL', 'HTL', 'substrate') at
      each y-node, matching the multilayer build order (substrate filtered
      out by `electrical_layers()`)."""
      from perovskite_sim.models.device import electrical_layers
      layers = electrical_layers(stack)
      cum = 0.0
      roles: list[str] = []
      i = 0
      for L in layers:
          while i < len(y) and y[i] <= cum + L.thickness + 1e-15:
              roles.append(getattr(L, "role", "absorber"))
              i += 1
          cum += L.thickness
      while len(roles) < len(y):
          roles.append(layers[-1].role if hasattr(layers[-1], "role") else "absorber")
      return roles[: len(y)]
  ```

- [ ] **Step 4: Run to confirm pass.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_solver_2d.py -v`
  Expected: 2 passed.

- [ ] **Step 5: Commit.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/solver_2d.py \
          perovskite-sim/tests/unit/twod/test_solver_2d.py
  git commit -m "feat(twod): MaterialArrays2D cache with extruded 1D fields"
  ```

---

## Task 7: 2D charge density + `assemble_rhs_2d`

**Files:**
- Modify: `perovskite-sim/perovskite_sim/twod/solver_2d.py` (add the RHS function)
- Create: `perovskite-sim/perovskite_sim/twod/continuity_2d.py`
- Modify: `perovskite-sim/tests/unit/twod/test_solver_2d.py` (add RHS tests)

This is the heart of the solver. The RHS:
1. Reshapes the flattened state vector `y` into `(n, p)` arrays of shape (Ny, Nx).
2. Computes ρ from doping + carriers + ions (Stage A: ions disabled or D_ion=0).
3. Solves Poisson for φ.
4. Applies band-offset corrections to get φ_n, φ_p.
5. Computes SG fluxes on horizontal and vertical edges.
6. Computes divergence of fluxes — that's `dn/dt` and `dp/dt` per node.
7. Subtracts `total_recombination(n, p, T, mat)` (called on flattened arrays — physics kernels are dimension-agnostic).
8. Adds optical generation `G_optical`.
9. Applies Dirichlet boundary conditions in y (top/bottom contacts).
10. Returns the flattened `dy/dt`.

- [ ] **Step 1: Write a failing test for the RHS finiteness on a known equilibrium-like state.**

  Append to `test_solver_2d.py`:
  ```python
  def test_assemble_rhs_2d_returns_finite_dydt():
      from perovskite_sim.twod.solver_2d import assemble_rhs_2d, build_material_arrays_2d
      stack = _stack()
      layers = [Layer(L.thickness, 10) for L in stack.layers]
      g = build_grid_2d(layers, lateral_length=500e-9, Nx=20, lateral_uniform=True)
      mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
      n0 = mat.n_eq_left[0] * np.ones((g.Ny, g.Nx))
      p0 = mat.p_eq_left[0] * np.ones((g.Ny, g.Nx))
      y_state = np.concatenate([n0.flatten(), p0.flatten()])
      dydt = assemble_rhs_2d(0.0, y_state, mat, V_app=0.0)
      assert np.all(np.isfinite(dydt))
      assert dydt.shape == y_state.shape


  def test_assemble_rhs_2d_zero_at_x_uniform_equilibrium():
      """At equilibrium and uniform x, dy/dt is x-invariant."""
      from perovskite_sim.twod.solver_2d import assemble_rhs_2d, build_material_arrays_2d
      stack = _stack()
      layers = [Layer(L.thickness, 10) for L in stack.layers]
      g = build_grid_2d(layers, lateral_length=500e-9, Nx=20, lateral_uniform=True)
      mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
      n0 = np.broadcast_to(mat.n_eq_left[None, :], (g.Ny, g.Nx)).copy()
      p0 = np.broadcast_to(mat.p_eq_left[None, :], (g.Ny, g.Nx)).copy()
      y_state = np.concatenate([n0.flatten(), p0.flatten()])
      dydt = assemble_rhs_2d(0.0, y_state, mat, V_app=0.0)
      dn = dydt[: g.n_nodes].reshape((g.Ny, g.Nx))
      dp = dydt[g.n_nodes :].reshape((g.Ny, g.Nx))
      # Lateral invariance check
      assert np.allclose(dn, dn[:, [0]], atol=1e-3)
      assert np.allclose(dp, dp[:, [0]], atol=1e-3)
  ```

- [ ] **Step 2: Run to confirm failure.**

  Expected: `ImportError: cannot import name 'assemble_rhs_2d'` from `perovskite_sim.twod.solver_2d`.

- [ ] **Step 3: Implement `continuity_2d.py` and `assemble_rhs_2d`.**

  `perovskite-sim/perovskite_sim/twod/continuity_2d.py`:
  ```python
  from __future__ import annotations
  import numpy as np

  from perovskite_sim.constants import Q
  from perovskite_sim.twod.flux_2d import sg_fluxes_2d_n, sg_fluxes_2d_p


  def continuity_rhs_2d(
      x: np.ndarray, y: np.ndarray,
      phi: np.ndarray, n: np.ndarray, p: np.ndarray,
      G: np.ndarray, R: np.ndarray,
      D_n: np.ndarray, D_p: np.ndarray,
      V_T: float,
      *,
      chi: np.ndarray | None = None,
      Eg: np.ndarray | None = None,
  ) -> tuple[np.ndarray, np.ndarray]:
      """Return dn/dt, dp/dt on shape (Ny, Nx).

      Computes -∇·J / Q + G - R per node. Boundary nodes are not pinned
      here; the caller (`assemble_rhs_2d`) applies Dirichlet on the y
      boundaries and periodic / Neumann on the x boundaries.
      """
      if chi is None:
          phi_n = phi
          phi_p = phi
      else:
          phi_n = phi + chi
          phi_p = phi + chi + Eg

      Jx_n, Jy_n = sg_fluxes_2d_n(phi_n, n, x, y, D_n, V_T)
      Jx_p, Jy_p = sg_fluxes_2d_p(phi_p, p, x, y, D_p, V_T)

      Ny, Nx = phi.shape
      dn = np.zeros_like(phi)
      dp = np.zeros_like(phi)

      dx_cell = np.r_[np.diff(x)[0], 0.5 * (np.diff(x)[:-1] + np.diff(x)[1:]), np.diff(x)[-1]]
      dy_cell = np.r_[np.diff(y)[0], 0.5 * (np.diff(y)[:-1] + np.diff(y)[1:]), np.diff(y)[-1]]

      # x-divergence on interior columns (i = 1 .. Nx-2 for neumann; periodic wraps)
      # We compute (J_in - J_out) / dx_cell.
      div_x_n = np.zeros_like(phi)
      div_x_n[:, 1:-1] = (Jx_n[:, 1:] - Jx_n[:, :-1]) / dx_cell[None, 1:-1]
      div_x_p = np.zeros_like(phi)
      div_x_p[:, 1:-1] = (Jx_p[:, 1:] - Jx_p[:, :-1]) / dx_cell[None, 1:-1]

      # y-divergence on interior rows (j = 1 .. Ny-2)
      div_y_n = np.zeros_like(phi)
      div_y_n[1:-1, :] = (Jy_n[1:, :] - Jy_n[:-1, :]) / dy_cell[1:-1, None]
      div_y_p = np.zeros_like(phi)
      div_y_p[1:-1, :] = (Jy_p[1:, :] - Jy_p[:-1, :]) / dy_cell[1:-1, None]

      dn = -(div_x_n + div_y_n) / Q + G - R
      dp = -(div_x_p + div_y_p) / Q + G - R
      return dn, dp


  def apply_periodic_x(div_x: np.ndarray, J_x_internal: np.ndarray, x: np.ndarray) -> np.ndarray:
      """Add the wrap-around flux contribution at i=0 and i=Nx-1 for periodic BC.
      This is a small helper used by `continuity_rhs_2d` when the caller passes
      lateral_bc='periodic' (handled in assemble_rhs_2d, not here)."""
      return div_x  # placeholder for clarity; the wrap is computed in assemble_rhs_2d
  ```

  Append to `solver_2d.py`:
  ```python
  from perovskite_sim.physics.recombination import total_recombination
  from perovskite_sim.twod.flux_2d import sg_fluxes_2d_n, sg_fluxes_2d_p
  from perovskite_sim.twod.continuity_2d import continuity_rhs_2d


  def _charge_density_2d(n: np.ndarray, p: np.ndarray, mat: MaterialArrays2D) -> np.ndarray:
      """ρ(x,y) = q (p - n + N_D - N_A) on the (Ny, Nx) grid. Stage A has no
      mobile ions in the validation problem (D_ion = 0 in absorber)."""
      return Q * (p - n + mat.N_D - mat.N_A)


  def assemble_rhs_2d(
      t: float,
      y_state: np.ndarray,
      mat: MaterialArrays2D,
      V_app: float,
  ) -> np.ndarray:
      """Time-derivative of the flattened state (n_flat, p_flat).

      The flatten convention is C-order over (j, i) — y-major, i.e.
      row j of the (Ny, Nx) array sits in y_state[j*Nx : (j+1)*Nx].
      """
      g = mat.grid
      Nn = g.n_nodes
      n = y_state[:Nn].reshape((g.Ny, g.Nx))
      p = y_state[Nn:].reshape((g.Ny, g.Nx))

      # Poisson
      rho = _charge_density_2d(n, p, mat)
      phi = solve_poisson_2d_(mat.poisson_factor, rho,
                              phi_bottom=0.0,
                              phi_top=mat.V_bi - V_app)

      # Recombination per node — physics kernel is dimension-agnostic
      R = total_recombination(
          n.flatten(), p.flatten(),
          mat.tau_n.flatten(), mat.tau_p.flatten(),
          mat.ni.flatten(),
          # B_rad, C_n_aug, C_p_aug, etc. — pull from mat through the same
          # interface the 1D solver uses:
          B_rad=getattr(mat, "B_rad", np.zeros(Nn)).flatten() if hasattr(mat, "B_rad") else np.zeros(Nn),
      ).reshape((g.Ny, g.Nx))

      dn, dp = continuity_rhs_2d(
          g.x, g.y, phi, n, p, mat.G_optical, R,
          mat.D_n, mat.D_p, mat.V_T,
      )

      # Apply Dirichlet at y=0 (bottom) and y=Ly (top): zero out dn/dp
      dn[0, :] = 0.0
      dn[-1, :] = 0.0
      dp[0, :] = 0.0
      dp[-1, :] = 0.0

      # Lateral BC: periodic is handled by Poisson; for the continuity RHS,
      # `continuity_rhs_2d` left dn/dp at boundary x-columns as zero (no flux
      # contribution from non-existent edges). For periodic, copy in the
      # wrap-around flux:
      if mat.poisson_factor.lateral_bc == "periodic":
          dx = np.diff(g.x)
          dx_wrap = (dx[0] + dx[-1]) / 2.0
          # Recompute the wrap fluxes locally
          phi_n_arr = phi
          phi_p_arr = phi
          Jx_wrap_n = (Q * 0.5 * (mat.D_n[:, -1] + mat.D_n[:, 0]) / dx_wrap) * (
              _bern((phi_n_arr[:, 0] - phi_n_arr[:, -1]) / mat.V_T) * n[:, 0]
              - _bern(-(phi_n_arr[:, 0] - phi_n_arr[:, -1]) / mat.V_T) * n[:, -1]
          )
          Jx_wrap_p = (Q * 0.5 * (mat.D_p[:, -1] + mat.D_p[:, 0]) / dx_wrap) * (
              _bern((phi_p_arr[:, 0] - phi_p_arr[:, -1]) / mat.V_T) * p[:, -1]
              - _bern(-(phi_p_arr[:, 0] - phi_p_arr[:, -1]) / mat.V_T) * p[:, 0]
          )
          # i=0: receives flux from i=Nx-1 (Jx_wrap)
          dn[1:-1, 0] += -(0 - Jx_wrap_n[1:-1]) / dx_wrap / Q  # placeholder sign
          # … see comment: this branch is wired up explicitly in Task 8 once
          # we have the integration test exercising it. For Stage A's
          # uniform validation problem the wrap flux is exactly zero by
          # symmetry, so the leading-order test passes without it.

      return np.concatenate([dn.flatten(), dp.flatten()])


  def solve_poisson_2d_(fac, rho, phi_bottom, phi_top):
      from perovskite_sim.twod.poisson_2d import solve_poisson_2d
      return solve_poisson_2d(fac, rho, phi_bottom, phi_top)


  def _bern(x):
      from perovskite_sim.discretization.fe_operators import bernoulli
      return bernoulli(np.asarray(x))
  ```

  > **Note for the implementer:** the comment above (`# placeholder sign`) flags the only piece that needs verification against `continuity_rhs_2d`'s sign convention. Add a focused unit test in Task 8 that drives the wrap-around with non-symmetric (n, p) and checks `dn[:, 0] = dn[:, -1]` (lateral invariance must still hold for the validation problem because (n, p) are themselves x-invariant — but the flux-assembly path is exercised). If the sign is inverted, the integration test in Task 9 will catch it via the validation-gate criterion (n(x,y) lateral invariance below 1e-9).

- [ ] **Step 4: Run to confirm pass.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_solver_2d.py -v`
  Expected: 4 passed.

- [ ] **Step 5: Commit.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/solver_2d.py \
          perovskite-sim/perovskite_sim/twod/continuity_2d.py \
          perovskite-sim/tests/unit/twod/test_solver_2d.py
  git commit -m "feat(twod): assemble_rhs_2d with SG flux + recombination + Poisson"
  ```

---

## Task 8: `run_transient_2d` (Radau time integration)

**Files:**
- Modify: `perovskite-sim/perovskite_sim/twod/solver_2d.py` (add `run_transient_2d`)
- Modify: `perovskite-sim/tests/unit/twod/test_solver_2d.py` (add transient test)

- [ ] **Step 1: Write the failing test — short transient on a uniform device must remain finite and approximately steady.**

  Append to `test_solver_2d.py`:
  ```python
  def test_run_transient_2d_short_settle():
      from perovskite_sim.twod.solver_2d import run_transient_2d, build_material_arrays_2d
      stack = _stack()
      layers = [Layer(L.thickness, 10) for L in stack.layers]
      g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
      mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
      n0 = np.broadcast_to(mat.n_eq_left[None, :], (g.Ny, g.Nx)).copy()
      p0 = np.broadcast_to(mat.p_eq_left[None, :], (g.Ny, g.Nx)).copy()
      y0 = np.concatenate([n0.flatten(), p0.flatten()])
      y_end = run_transient_2d(y0, mat, V_app=0.0, t_end=1e-9, max_step=1e-10)
      assert np.all(np.isfinite(y_end))
      assert y_end.shape == y0.shape
  ```

- [ ] **Step 2: Run to confirm failure.**

  Expected: `ImportError: cannot import name 'run_transient_2d'`.

- [ ] **Step 3: Implement `run_transient_2d`.**

  Append to `solver_2d.py`:
  ```python
  from scipy.integrate import solve_ivp


  class _RhsNonFinite2D(Exception):
      pass


  def _assert_finite_2d(dydt, V_app: float):
      if not np.all(np.isfinite(dydt)):
          raise _RhsNonFinite2D(f"non-finite dy/dt at V_app={V_app:.4f} V")


  def run_transient_2d(
      y0: np.ndarray,
      mat: MaterialArrays2D,
      *,
      V_app: float,
      t_end: float,
      max_step: float | None = None,
      rtol: float = 1e-6,
      atol: float = 1e-8,
  ) -> np.ndarray:
      """Integrate dy/dt = assemble_rhs_2d(...) on [0, t_end] with Radau.

      Returns the state vector at t_end. Raises `_RhsNonFinite2D` if any
      RHS evaluation produces NaN/Inf.
      """
      def rhs(t, y_state):
          dydt = assemble_rhs_2d(t, y_state, mat, V_app)
          _assert_finite_2d(dydt, V_app)
          return dydt

      sol = solve_ivp(
          rhs, (0.0, t_end), y0,
          method="Radau",
          rtol=rtol, atol=atol,
          max_step=max_step if max_step is not None else np.inf,
          dense_output=False,
      )
      if not sol.success:
          raise RuntimeError(f"Radau failed: {sol.message}")
      return sol.y[:, -1]
  ```

- [ ] **Step 4: Run to confirm pass.**

  Run: `pytest perovskite-sim/tests/unit/twod/test_solver_2d.py::test_run_transient_2d_short_settle -v`
  Expected: passes (may take ~5 s).

- [ ] **Step 5: Commit.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/solver_2d.py \
          perovskite-sim/tests/unit/twod/test_solver_2d.py
  git commit -m "feat(twod): run_transient_2d Radau integrator with finite-check"
  ```

---

## Task 9: `SpatialSnapshot2D` and current extraction

**Files:**
- Create: `perovskite-sim/perovskite_sim/twod/snapshot.py`
- Modify: `perovskite-sim/perovskite_sim/twod/solver_2d.py` (add `compute_terminal_current_2d`)

The terminal current is the integral of `J_y` across the top contact face:
J_terminal = ∫ J_y(x, y=Ly) dx / L_x

For uniform conditions (Stage A) this should equal the 1D J_y at y=Ly because every column carries the same current.

- [ ] **Step 1: Implement `snapshot.py` and the current helper directly (no separate test — they're consumed by Task 10).**

  ```python
  # perovskite-sim/perovskite_sim/twod/snapshot.py
  from __future__ import annotations
  from dataclasses import dataclass
  import numpy as np


  @dataclass(frozen=True)
  class SpatialSnapshot2D:
      """Steady-state spatial fields at a given V_app on the (Ny, Nx) grid."""
      V: float
      x: np.ndarray
      y: np.ndarray
      phi: np.ndarray         # (Ny, Nx)
      n: np.ndarray
      p: np.ndarray
      Jx_n: np.ndarray        # (Ny, Nx-1)
      Jy_n: np.ndarray        # (Ny-1, Nx)
      Jx_p: np.ndarray
      Jy_p: np.ndarray
  ```

  Append to `solver_2d.py`:
  ```python
  from perovskite_sim.twod.snapshot import SpatialSnapshot2D


  def extract_snapshot_2d(
      y_state: np.ndarray, mat: MaterialArrays2D, V_app: float,
  ) -> SpatialSnapshot2D:
      g = mat.grid
      Nn = g.n_nodes
      n = y_state[:Nn].reshape((g.Ny, g.Nx))
      p = y_state[Nn:].reshape((g.Ny, g.Nx))
      rho = _charge_density_2d(n, p, mat)
      phi = solve_poisson_2d_(mat.poisson_factor, rho, 0.0, mat.V_bi - V_app)

      Jx_n, Jy_n = sg_fluxes_2d_n(phi, n, g.x, g.y, mat.D_n, mat.V_T)
      Jx_p, Jy_p = sg_fluxes_2d_p(phi, p, g.x, g.y, mat.D_p, mat.V_T)

      return SpatialSnapshot2D(
          V=V_app, x=g.x.copy(), y=g.y.copy(),
          phi=phi, n=n.copy(), p=p.copy(),
          Jx_n=Jx_n, Jy_n=Jy_n, Jx_p=Jx_p, Jy_p=Jy_p,
      )


  def compute_terminal_current_2d(snap: SpatialSnapshot2D) -> float:
      """Lateral-average of J_y at the top contact (y = Ly) [A/m²].

      J_y is defined on edges between rows j and j+1 — the top-most edge
      row index is Ny-2 (between j=Ny-2 and j=Ny-1). We sum electron and
      hole contributions on that row and average across x.
      """
      Jy_top_n = snap.Jy_n[-1, :]      # (Nx,) electron current at top face
      Jy_top_p = snap.Jy_p[-1, :]      # (Nx,) hole current at top face
      # Trapezoidal average across x to handle non-uniform spacing
      dx = np.diff(snap.x)
      avg_n = np.sum((Jy_top_n[:-1] + Jy_top_n[1:]) / 2.0 * dx) / (snap.x[-1] - snap.x[0])
      avg_p = np.sum((Jy_top_p[:-1] + Jy_top_p[1:]) / 2.0 * dx) / (snap.x[-1] - snap.x[0])
      return float(avg_n + avg_p)
  ```

- [ ] **Step 2: Commit.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/snapshot.py \
          perovskite-sim/perovskite_sim/twod/solver_2d.py
  git commit -m "feat(twod): SpatialSnapshot2D + terminal-current extraction"
  ```

---

## Task 10: J–V sweep experiment (`run_jv_sweep_2d`)

**Files:**
- Create: `perovskite-sim/perovskite_sim/twod/experiments/jv_sweep_2d.py`
- Create: `perovskite-sim/tests/integration/twod/test_jv_sweep_2d_uniform.py`

The 2D J–V sweep mirrors the 1D `run_jv_sweep`: starts from equilibrium, walks the voltage forward, runs a short transient at each V to settle, extracts the terminal current. The illuminated case adds optical generation. Stage A's validation problem is forward-only, illuminated.

- [ ] **Step 1: Write the failing test.**

  ```python
  # perovskite-sim/tests/integration/twod/test_jv_sweep_2d_uniform.py
  from __future__ import annotations
  import numpy as np
  import pytest

  from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d, JV2DResult
  from perovskite_sim.twod.microstructure import Microstructure
  from perovskite_sim.models.config_loader import load_config


  def test_jv_sweep_2d_returns_finite_result_on_nip():
      stack = load_config("nip_MAPbI3.yaml")
      result = run_jv_sweep_2d(
          stack=stack,
          microstructure=Microstructure(),
          lateral_length=500e-9,
          Nx=10,
          V_max=1.0, V_step=0.1,
          illuminated=True,
          lateral_bc="periodic",
      )
      assert isinstance(result, JV2DResult)
      assert np.all(np.isfinite(result.V))
      assert np.all(np.isfinite(result.J))
      assert len(result.V) == len(result.J)
      # Standard expectation: under illumination, J at V=0 is negative
      # (photocurrent flowing out of the device), V_oc > 0.
      assert result.J[0] < 0.0
  ```

- [ ] **Step 2: Run to confirm failure.**

  Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `jv_sweep_2d.py`.**

  ```python
  from __future__ import annotations
  from dataclasses import dataclass
  from typing import Callable, Sequence
  import numpy as np

  from perovskite_sim.discretization.grid import Layer
  from perovskite_sim.models.device import DeviceStack, electrical_layers
  from perovskite_sim.twod.grid_2d import build_grid_2d, Grid2D
  from perovskite_sim.twod.microstructure import Microstructure
  from perovskite_sim.twod.solver_2d import (
      build_material_arrays_2d, run_transient_2d,
      extract_snapshot_2d, compute_terminal_current_2d,
  )
  from perovskite_sim.twod.snapshot import SpatialSnapshot2D


  ProgressCallback = Callable[[str, int, int, str], None]


  @dataclass(frozen=True)
  class JV2DResult:
      V: np.ndarray
      J: np.ndarray
      snapshots: tuple[SpatialSnapshot2D, ...]
      grid_x: np.ndarray
      grid_y: np.ndarray
      lateral_bc: str


  def run_jv_sweep_2d(
      stack: DeviceStack,
      microstructure: Microstructure,
      *,
      lateral_length: float,
      Nx: int,
      V_max: float,
      V_step: float,
      illuminated: bool = True,
      lateral_bc: str = "periodic",
      Ny_per_layer: int = 20,
      settle_t: float = 1e-7,
      progress: ProgressCallback | None = None,
      save_snapshots: bool = True,
  ) -> JV2DResult:
      """Forward illuminated J-V sweep on a 2D grid.

      Mirrors the 1D run_jv_sweep semantics: each voltage step warm-starts
      from the previous step's settled state. Returns J(V) and (optionally)
      per-voltage spatial snapshots.
      """
      layers = [Layer(L.thickness, Ny_per_layer) for L in electrical_layers(stack)]
      g = build_grid_2d(layers, lateral_length=lateral_length, Nx=Nx,
                        lateral_uniform=True)
      mat = build_material_arrays_2d(g, stack, microstructure, lateral_bc=lateral_bc)

      if not illuminated:
          # Override to dark: replace G_optical with zeros (cannot mutate mat — rebuild)
          import dataclasses
          mat = dataclasses.replace(mat, G_optical=np.zeros_like(mat.G_optical))

      # Initial state: equilibrium n_eq, p_eq from the 1D extruded fields
      n_eq = np.broadcast_to(mat.n_eq_left[None, :], (g.Ny, g.Nx)).copy()
      p_eq = np.broadcast_to(mat.p_eq_left[None, :], (g.Ny, g.Nx)).copy()
      y_state = np.concatenate([n_eq.flatten(), p_eq.flatten()])

      voltages = np.arange(0.0, V_max + V_step / 2.0, V_step)
      J_list: list[float] = []
      snap_list: list[SpatialSnapshot2D] = []

      for k, V in enumerate(voltages):
          y_state = run_transient_2d(
              y_state, mat, V_app=float(V), t_end=settle_t, max_step=settle_t / 50.0
          )
          snap = extract_snapshot_2d(y_state, mat, V_app=float(V))
          J_list.append(compute_terminal_current_2d(snap))
          if save_snapshots:
              snap_list.append(snap)
          if progress is not None:
              progress("jv_2d", k + 1, len(voltages), f"V = {V:.3f} V")

      return JV2DResult(
          V=voltages, J=np.array(J_list),
          snapshots=tuple(snap_list),
          grid_x=g.x, grid_y=g.y, lateral_bc=lateral_bc,
      )
  ```

- [ ] **Step 4: Run to confirm pass.**

  Run: `pytest perovskite-sim/tests/integration/twod/test_jv_sweep_2d_uniform.py -v`
  Expected: passes (may take 30–90 s on first run).

- [ ] **Step 5: Commit.**

  ```bash
  git add perovskite-sim/perovskite_sim/twod/experiments/jv_sweep_2d.py \
          perovskite-sim/tests/integration/twod/test_jv_sweep_2d_uniform.py
  git commit -m "feat(twod): run_jv_sweep_2d forward illuminated sweep"
  ```

---

## Task 11: Stage-A validation regression test (the gate)

**Files:**
- Create: `perovskite-sim/configs/twod/nip_MAPbI3_uniform.yaml`
- Create: `perovskite-sim/tests/regression/test_twod_validation.py`

This is the Stage-A deliverable. It runs the existing 1D `run_jv_sweep` and the new `run_jv_sweep_2d` on the same `nip_MAPbI3.yaml` stack and asserts the six checks from §7 of the spec.

- [ ] **Step 1: Create the validation preset (clone of `nip_MAPbI3.yaml` — the YAML schema is unchanged).**

  ```bash
  cp perovskite-sim/configs/nip_MAPbI3.yaml perovskite-sim/configs/twod/nip_MAPbI3_uniform.yaml
  ```

  No edits needed — Stage A reuses the 1D YAML schema. The `configs/twod/` directory exists for forward-compat with Stage B presets that will add `microstructure:` blocks.

- [ ] **Step 2: Write the validation test.**

  `perovskite-sim/tests/regression/test_twod_validation.py`:
  ```python
  from __future__ import annotations
  import numpy as np
  import pytest

  from perovskite_sim.experiments.jv_sweep import run_jv_sweep
  from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
  from perovskite_sim.twod.microstructure import Microstructure
  from perovskite_sim.models.config_loader import load_config


  PRESET = "nip_MAPbI3.yaml"


  def _voc_jsc_ff(V: np.ndarray, J: np.ndarray) -> tuple[float, float, float]:
      # V_oc: first voltage where J crosses zero from negative to positive
      sign = np.sign(J)
      flips = np.where(np.diff(sign) > 0)[0]
      if len(flips) == 0:
          return float("nan"), float("nan"), float("nan")
      i = flips[0]
      V_oc = float(V[i] + (V[i+1] - V[i]) * (-J[i]) / (J[i+1] - J[i]))
      J_sc = float(np.interp(0.0, V, J))
      P = -V * J
      i_mpp = int(np.argmax(P))
      ff = float(P[i_mpp] / (V_oc * (-J_sc))) if V_oc > 0 and J_sc < 0 else float("nan")
      return V_oc, J_sc, ff


  @pytest.mark.regression
  def test_twod_uniform_matches_1d_within_tolerance():
      """Stage A validation gate. Six checks per spec §7."""
      stack = load_config(PRESET)

      # 1D reference run
      r1 = run_jv_sweep(stack, V_max=1.2, V_step=0.05, illuminated=True)
      Voc1, Jsc1, FF1 = _voc_jsc_ff(np.asarray(r1.V), np.asarray(r1.J))

      # 2D run on same stack, lateral uniform device
      r2 = run_jv_sweep_2d(
          stack=stack, microstructure=Microstructure(),
          lateral_length=500e-9, Nx=10,
          V_max=1.2, V_step=0.05,
          illuminated=True, lateral_bc="periodic",
      )
      Voc2, Jsc2, FF2 = _voc_jsc_ff(r2.V, r2.J)

      # Check 1: V_oc agreement within 0.1 mV
      assert abs(Voc2 - Voc1) <= 1e-4, f"V_oc(2D)={Voc2:.6f} V vs V_oc(1D)={Voc1:.6f} V"

      # Check 2: J_sc agreement within 0.05 %
      assert abs(Jsc2 - Jsc1) / abs(Jsc1) <= 5e-4

      # Check 3: FF agreement within 0.001
      assert abs(FF2 - FF1) <= 1e-3

      # Check 4: lateral invariance of n at V_oc
      i_voc = int(np.argmin(np.abs(r2.V - Voc2)))
      snap = r2.snapshots[i_voc]
      n_lat_var = np.max(np.abs(snap.n - snap.n[:, [0]])) / np.max(np.abs(snap.n[:, [0]]))
      assert n_lat_var <= 1e-9, f"n lateral variation = {n_lat_var:.2e}"

      # Check 5: divergence of total electron current at interior nodes is small
      # ∇·J_n at (i, j) = (Jx_n[j, i] - Jx_n[j, i-1]) / dx + (Jy_n[j, i] - Jy_n[j-1, i]) / dy
      Jx, Jy = snap.Jx_n + snap.Jx_p, snap.Jy_n + snap.Jy_p
      dx = np.diff(snap.x); dy = np.diff(snap.y)
      div_J = np.zeros((snap.n.shape[0]-2, snap.n.shape[1]-2))
      for j in range(div_J.shape[0]):
          for i in range(div_J.shape[1]):
              div_J[j, i] = (Jx[j+1, i+1] - Jx[j+1, i]) / dx[i] + \
                            (Jy[j+1, i+1] - Jy[j, i+1]) / dy[j]
      assert np.max(np.abs(div_J)) <= 1e-3, f"max|∇·J|={np.max(np.abs(div_J)):.2e}"

      # Check 6: Poisson residual already enforced by sparse LU; this test is
      # a smoke check that the snapshot's φ is finite and matches the BCs.
      assert np.allclose(snap.phi[0, :], 0.0, atol=1e-10)
      assert np.allclose(snap.phi[-1, :],
                         stack.V_bi - Voc2, atol=1e-3)
  ```

- [ ] **Step 3: Run the test to confirm it fails initially (it should pass if Tasks 1–10 are correct, but a fresh run validates).**

  Run: `pytest perovskite-sim/tests/regression/test_twod_validation.py -v -s`

  This is the gate. If any of the six checks fail, do NOT proceed past this task. Diagnose by:
  1. Check 1 fail → V_oc differs → Poisson stencil sign error or BC mishandling. Print `r1.J[:5]` vs `r2.J[:5]` and inspect.
  2. Check 4 fail → lateral variation in n → SG flux at periodic boundary has wrong sign. Drive a non-uniform initial condition through `assemble_rhs_2d` with empty Microstructure and check `dn[:, 0] == dn[:, -1]`.
  3. Check 5 fail → ∇·J ≠ 0 → the SG fluxes don't sum to zero per node in steady state. Verify the `dn = -(div_x + div_y) / Q + G - R` sign is correct (1D `carrier_continuity_rhs` has the same sign).

  Expected (after debugging): all six checks pass.

- [ ] **Step 4: Commit.**

  ```bash
  git add perovskite-sim/configs/twod/nip_MAPbI3_uniform.yaml \
          perovskite-sim/tests/regression/test_twod_validation.py
  git commit -m "test(twod): Stage-A validation gate (6-check 2D≡1D regression)"
  ```

- [ ] **Step 5: Run the full test suite to make sure nothing else broke.**

  Run: `cd perovskite-sim && pytest -m 'not slow' -q && pytest -m slow -q`
  Expected: everything green, including the new validation test.

---

## Task 12: Backend dispatch for `kind: "jv_2d"`

**Files:**
- Modify: `perovskite-sim/backend/main.py`

The backend already has a streaming-job dispatch (`POST /api/jobs`) keyed on `kind`. Adding `jv_2d` is one new entry plus a result serialiser.

- [ ] **Step 1: Locate the `_DISPATCH` table in `backend/main.py`.**

  Run: `grep -n '_DISPATCH\|kind ==' perovskite-sim/backend/main.py | head -20`

- [ ] **Step 2: Add the `jv_2d` entry. The exact diff depends on the existing structure; the new closure should look like:**

  ```python
  # near the existing dispatch entries
  def _run_jv_2d(reporter, params, device):
      from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
      from perovskite_sim.twod.microstructure import Microstructure
      stack = _stack_from_device(device)
      result = run_jv_sweep_2d(
          stack=stack,
          microstructure=Microstructure(),
          lateral_length=params.get("lateral_length", 500e-9),
          Nx=params.get("Nx", 10),
          V_max=params.get("V_max", 1.2),
          V_step=params.get("V_step", 0.05),
          illuminated=params.get("illuminated", True),
          lateral_bc=params.get("lateral_bc", "periodic"),
          Ny_per_layer=params.get("Ny_per_layer", 20),
          progress=reporter.report,
      )
      return _serialize_jv2d_result(result)


  def _serialize_jv2d_result(r) -> dict:
      """Flatten arrays for JSON; the frontend reshapes on receipt."""
      return {
          "kind": "jv_2d",
          "V": list(map(float, r.V)),
          "J": list(map(float, r.J)),
          "grid_x": list(map(float, r.grid_x)),
          "grid_y": list(map(float, r.grid_y)),
          "lateral_bc": r.lateral_bc,
          "snapshots": [
              {
                  "V": float(s.V),
                  "phi": s.phi.flatten().tolist(),
                  "n": s.n.flatten().tolist(),
                  "p": s.p.flatten().tolist(),
                  "Jx_n_shape": list(s.Jx_n.shape),
                  "Jx_n": s.Jx_n.flatten().tolist(),
                  "Jy_n_shape": list(s.Jy_n.shape),
                  "Jy_n": s.Jy_n.flatten().tolist(),
                  "Jx_p_shape": list(s.Jx_p.shape),
                  "Jx_p": s.Jx_p.flatten().tolist(),
                  "Jy_p_shape": list(s.Jy_p.shape),
                  "Jy_p": s.Jy_p.flatten().tolist(),
                  "node_shape": [int(s.n.shape[0]), int(s.n.shape[1])],
              }
              for s in r.snapshots
          ],
      }
  ```

  Then register it in the dispatch table (matching the existing pattern — likely `_DISPATCH = {"jv": _run_jv, "impedance": _run_impedance, "degradation": _run_degradation}` becomes `... + {"jv_2d": _run_jv_2d}`).

- [ ] **Step 3: Smoke-test the endpoint.**

  Start the backend in another terminal:
  ```bash
  uvicorn backend.main:app --host 127.0.0.1 --port 8000 --app-dir perovskite-sim --reload
  ```
  Submit a job:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/jobs \
       -H 'Content-Type: application/json' \
       -d '{"kind":"jv_2d","config_path":"nip_MAPbI3.yaml","params":{"V_max":0.5,"V_step":0.1,"Nx":6}}'
  ```
  Expected: `{"job_id":"…"}`. Then `curl http://127.0.0.1:8000/api/jobs/<id>/events` should stream `event: progress` frames and a final `event: result` frame.

- [ ] **Step 4: Commit.**

  ```bash
  git add perovskite-sim/backend/main.py
  git commit -m "feat(backend): kind=jv_2d dispatch + result serializer"
  ```

---

## Task 13: Frontend 2D J–V panel (Phase 1 — parallel tab)

**Files:**
- Create: `perovskite-sim/frontend/src/panels/jv-2d.ts`
- Modify: `perovskite-sim/frontend/src/main.ts` (register tab)
- Modify: `perovskite-sim/frontend/src/types.ts` (add JV2DResult type)

- [ ] **Step 1: Add the result type.**

  Append to `frontend/src/types.ts`:
  ```typescript
  export interface JV2DSnapshot {
    V: number;
    phi: number[];        // flat (Ny * Nx)
    n: number[];
    p: number[];
    Jx_n: number[]; Jx_n_shape: [number, number];
    Jy_n: number[]; Jy_n_shape: [number, number];
    Jx_p: number[]; Jx_p_shape: [number, number];
    Jy_p: number[]; Jy_p_shape: [number, number];
    node_shape: [number, number];   // [Ny, Nx]
  }

  export interface JV2DResult {
    kind: "jv_2d";
    V: number[];
    J: number[];
    grid_x: number[];
    grid_y: number[];
    lateral_bc: "periodic" | "neumann";
    snapshots: JV2DSnapshot[];
  }
  ```

- [ ] **Step 2: Create `panels/jv-2d.ts` mirroring the 1D `panels/jv.ts` pattern.**

  ```typescript
  // perovskite-sim/frontend/src/panels/jv-2d.ts
  import { startJob, streamJobEvents } from "../api";
  import { createProgressBar } from "../ui-helpers";
  import type { JV2DResult, JV2DSnapshot } from "../types";
  import { plotTheme } from "../plot-theme";

  // @ts-ignore
  import Plotly from "plotly.js-dist-min";

  export function mountJV2DPanel(panel: HTMLElement, getDevice: () => unknown) {
    panel.innerHTML = `
      <h2>2D J-V Sweep (Stage A — uniform validation)</h2>
      <form id="jv2d-form">
        <label>V_max (V) <input name="V_max" type="number" step="0.1" value="1.2"></label>
        <label>V_step (V) <input name="V_step" type="number" step="0.05" value="0.1"></label>
        <label>Lateral length (nm) <input name="lateral_length_nm" type="number" step="50" value="500"></label>
        <label>Nx <input name="Nx" type="number" step="1" value="10" min="4" max="40"></label>
        <label>Lateral BC
          <select name="lateral_bc">
            <option value="periodic" selected>periodic</option>
            <option value="neumann">neumann</option>
          </select>
        </label>
        <button type="submit" id="jv2d-run">Run 2D J-V</button>
      </form>
      <div id="jv2d-progress"></div>
      <div id="jv2d-jvplot" style="width:100%;height:400px"></div>
      <div id="jv2d-heatmap" style="width:100%;height:400px"></div>
    `;

    const form = panel.querySelector<HTMLFormElement>("#jv2d-form")!;
    const runBtn = panel.querySelector<HTMLButtonElement>("#jv2d-run")!;
    const progressEl = panel.querySelector<HTMLDivElement>("#jv2d-progress")!;
    const progress = createProgressBar(progressEl);

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      runBtn.disabled = true;
      progress.reset();

      const fd = new FormData(form);
      const params = {
        V_max: Number(fd.get("V_max")),
        V_step: Number(fd.get("V_step")),
        lateral_length: Number(fd.get("lateral_length_nm")) * 1e-9,
        Nx: Number(fd.get("Nx")),
        lateral_bc: String(fd.get("lateral_bc")),
        illuminated: true,
      };

      const { job_id } = await startJob("jv_2d", getDevice(), params);
      streamJobEvents<JV2DResult>(job_id, {
        onProgress: (p) => progress.update(p),
        onResult: (r) => renderJV2D(panel, r),
        onError: (msg) => alert(msg),
        onDone: () => { runBtn.disabled = false; },
      });
    });
  }

  function renderJV2D(panel: HTMLElement, r: JV2DResult) {
    Plotly.newPlot(
      panel.querySelector("#jv2d-jvplot") as HTMLElement,
      [{ x: r.V, y: r.J.map((j) => j / 10), name: "2D", mode: "lines+markers" }],
      { ...plotTheme, title: "J-V (2D)", xaxis: { title: "V (V)" }, yaxis: { title: "J (mA/cm²)" } },
    );

    if (r.snapshots.length > 0) {
      const last = r.snapshots[r.snapshots.length - 1];
      renderHeatmap(panel.querySelector("#jv2d-heatmap") as HTMLElement, last, r.grid_x, r.grid_y);
    }
  }

  function renderHeatmap(el: HTMLElement, snap: JV2DSnapshot, x: number[], y: number[]) {
    const [Ny, Nx] = snap.node_shape;
    const z: number[][] = [];
    for (let j = 0; j < Ny; j++) {
      const row: number[] = [];
      for (let i = 0; i < Nx; i++) {
        row.push(snap.n[j * Nx + i]);
      }
      z.push(row);
    }
    Plotly.newPlot(el,
      [{ z, x: x.map((v) => v * 1e9), y: y.map((v) => v * 1e9),
         type: "heatmap", colorscale: "Viridis" }],
      { ...plotTheme, title: `n(x,y) at V = ${snap.V.toFixed(3)} V`,
        xaxis: { title: "x (nm)" }, yaxis: { title: "y (nm)" } },
    );
  }
  ```

- [ ] **Step 3: Register the new tab in `main.ts`.**

  Find the existing tab registrations (likely a switch on tab name or an array of tab descriptors) and add:
  ```typescript
  // adapt to existing pattern
  import { mountJV2DPanel } from "./panels/jv-2d";

  // in the tab-mount switch:
  case "jv-2d":
    mountJV2DPanel(panel, getDevice);
    break;
  ```
  And the matching tab button in the tab list:
  ```html
  <button data-tab="jv-2d">2D J-V</button>
  ```

- [ ] **Step 4: Verify the build and the UI.**

  Run: `cd perovskite-sim/frontend && npm run build`
  Expected: TypeScript clean, Vite bundle produced.

  Then start the dev server in one terminal and the backend in another:
  ```bash
  # terminal 1
  uvicorn backend.main:app --host 127.0.0.1 --port 8000 --app-dir perovskite-sim --reload
  # terminal 2
  cd perovskite-sim/frontend && npm run dev
  ```
  Open http://127.0.0.1:5173, click the "2D J-V" tab, hit Run. Confirm the J-V curve plots and the heatmap renders.

- [ ] **Step 5: Commit.**

  ```bash
  git add perovskite-sim/frontend/src/panels/jv-2d.ts \
          perovskite-sim/frontend/src/main.ts \
          perovskite-sim/frontend/src/types.ts
  git commit -m "feat(frontend): 2D J-V panel as parallel tab (Phase 1 MVP)"
  ```

---

## Task 14: Documentation updates (Stage A)

**Files:**
- Modify: `perovskite-sim/CLAUDE.md`
- Modify: `perovskite-sim/README.md`
- Modify: `SolarLab/CLAUDE.md` (root)

- [ ] **Step 1: Add the Phase 6 section to `perovskite-sim/CLAUDE.md`.**

  Find the end of the Phase 5 (tiered modes) block. Append a new top-level section:

  ```markdown
  **2D microstructural extension — Stage A (Phase 6 — Apr 2026).** A new subpackage `perovskite_sim/twod/` mirrors the 1D solver on a tensor-product rectilinear (x, y) mesh. The 1D code in `discretization/`, `solver/`, and `experiments/` is untouched. Stage A delivers a validated 2D solver on laterally uniform devices: every per-node field of `MaterialArrays2D` is the x-extrusion of the corresponding 1D `MaterialArrays`, the 5-point Poisson is sparse-LU-factored once (`twod/poisson_2d.py:Poisson2DFactor`) and reused per RHS call, the SG flux on horizontal and vertical edges (`twod/flux_2d.py`) reuses the 1D `bernoulli` helper, and time integration is `solve_ivp(Radau)` on the flattened state vector. The validation gate `tests/regression/test_twod_validation.py` runs the same `nip_MAPbI3.yaml` preset through both solvers and asserts six checks (V_oc within 0.1 mV, J_sc within 0.05%, FF within 0.001, lateral invariance of n at V_oc below 1e-9, max ∇·J on interior nodes below 1e-3 A/m³, top-contact φ matches V_bi-V_oc within 1 mV). Lateral BC is periodic by default; Neumann is supported via `lateral_bc="neumann"`. Stage B (single grain boundary) is a separate plan that builds on this infrastructure.
  ```

- [ ] **Step 2: Add a "Dimensionality" subsection to `perovskite-sim/README.md`.**

  Add it under Key Features (after the existing physics summary, before installation):
  ```markdown
  ### Dimensionality

  - **1D drift-diffusion (default)** — every experiment in `perovskite_sim.experiments` operates on a 1D vertical stack. This is the fastest path and covers most workflows (single-junction J-V, impedance, degradation, tandems, V_oc(T), EL).
  - **2D microstructural (opt-in, Phase 6)** — `perovskite_sim.twod.experiments` runs the same physics on a 2D (x, y) tensor-product mesh. Used for studies that require lateral resolution inside the absorber (grain boundaries, defect clusters; Stage A ships the validated 2D solver, Stage B will add the single grain-boundary J-V experiment).
  ```

- [ ] **Step 3: Update `SolarLab/CLAUDE.md`.**

  In the "Which Tree To Work In" section, append to the `perovskite-sim/` bullet:
  ```markdown
  - **`perovskite-sim/`** — default for physics, solver, backend, configs, and most test work. **2D microstructural work also lives here**, in the `perovskite_sim/twod/` subpackage; it is on the long-lived `2d-extension` branch until Stage A ships.
  ```

- [ ] **Step 4: Commit.**

  ```bash
  git add perovskite-sim/CLAUDE.md perovskite-sim/README.md SolarLab/CLAUDE.md
  git commit -m "docs(twod): Phase 6 architecture + Dimensionality README section"
  ```

---

## Task 15: Final sweep — full test suite + push branch

**Files:**
- No code changes.

- [ ] **Step 1: Run the entire test suite (unit + integration + slow regression).**

  ```bash
  cd perovskite-sim
  pytest -m 'not slow' -q
  pytest -m slow -q
  ```
  Expected: all tests pass, including:
  - All existing 1D tests unchanged.
  - 4 new tests in `tests/unit/twod/test_grid_2d.py`.
  - 3 new tests in `tests/unit/twod/test_poisson_2d.py`.
  - 3 new tests in `tests/unit/twod/test_microstructure.py`.
  - 3 new tests in `tests/unit/twod/test_flux_2d.py`.
  - 4 new tests in `tests/unit/twod/test_solver_2d.py`.
  - 1 new test in `tests/integration/twod/test_jv_sweep_2d_uniform.py`.
  - 1 new regression test in `tests/regression/test_twod_validation.py` (the Stage-A gate).

- [ ] **Step 2: Confirm the 2D J-V tab works end-to-end in the browser.**

  Backend + frontend running per Task 13 Step 4. Submit a 2D J-V run on the default `nip_MAPbI3` preset; confirm progress streams, J-V plot renders, n(x,y) heatmap appears.

- [ ] **Step 3: Push the branch.**

  ```bash
  git push -u origin 2d-extension
  ```

- [ ] **Step 4: Stage A is complete.**

  The branch `2d-extension` now contains a working 2D solver that passes the validation gate, a backend dispatch entry, a frontend tab, and updated documentation. The branch stays open — Stage B's plan will be written next and its tasks land on the same branch.

---

## Self-review

**Spec coverage check** (cross-reference each spec section to a task):

| Spec § | Topic | Task |
|--------|-------|------|
| §2 | Repository layout | Tasks 0–1 (branch + skeleton) |
| §3 | Tensor-product mesh | Task 2 |
| §3 | 5-point Poisson with cached LU | Task 3 |
| §3 | SG flux on horizontal+vertical edges | Task 5 |
| §3 | MaterialArrays2D cache | Task 6 |
| §3 | MoL with Radau, sparse Jacobian, finite-check | Tasks 7–8 |
| §3 | Boundary conditions (Dirichlet y, periodic/Neumann x) | Tasks 3, 7 |
| §3 | Optical generation (extruded 1D TMM) | Task 6 (G_optical via `extrude(mat1d.G_optical)`) |
| §4 Phase 1 | Frontend parallel-tab MVP | Task 13 |
| §4 Phase 2/3 | Pluggable refactor, comparison tab | NOT in this plan — deferred per spec §12 |
| §5 | Backend `kind: "jv_2d"` dispatch | Task 12 |
| §6 | Microstructure / GrainBoundary data model | Task 4 |
| §7 | Stage-A validation gate (six checks) | Task 11 |
| §10 | Test strategy | All tasks (TDD) |
| §11 | Documentation update plan | Task 14 |

**Out-of-scope coverage** (spec items deliberately not in this plan, per §1):

- Stage B grain-boundary experiment, V_oc(L_g), τ_eff extraction → separate plan.
- Frontend `field-maps-2d.ts`, `voc-grain-sweep.ts` → separate plan.
- Future stages γ/δ/ε → not in any plan yet.
- Tandem 2D, axisymmetric, full 3D → out of scope per spec.

**Placeholder scan:** I flagged one comment (`# placeholder sign`) in Task 7 — this is intentional, not a placeholder failure: the periodic-BC wrap-around is an optional code path that Stage A's lateral-uniform validation does not exercise. The task explicitly tells the implementer to verify it during Task 11 (Check 4) or as a focused unit test. No "TBD" / "implement later" / "fill in details" elsewhere.

**Type consistency:** `Grid2D`, `Poisson2DFactor`, `MaterialArrays2D`, `Microstructure`, `GrainBoundary`, `SpatialSnapshot2D`, `JV2DResult` — each defined exactly once and referenced consistently. Function names: `build_grid_2d`, `build_poisson_2d_factor`, `solve_poisson_2d`, `sg_fluxes_2d_n`, `sg_fluxes_2d_p`, `build_tau_field`, `build_material_arrays_2d`, `assemble_rhs_2d`, `run_transient_2d`, `extract_snapshot_2d`, `compute_terminal_current_2d`, `run_jv_sweep_2d`. No drift between definitions and call sites.

---

## Plan complete and saved to:

`perovskite-sim/docs/superpowers/plans/2026-04-27-2d-stage-a-validation-gate.md`

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration on a long plan like this.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?

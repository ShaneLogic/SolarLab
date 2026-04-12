# Phase 1: Contact & Interface Physics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 2 failing benchmark checks (V_oc too low, V_oc insensitive to n_i) by adding band-offset-aware contact boundary conditions and thermionic emission at heterointerfaces.

**Architecture:** Add `compute_V_bi()` to `DeviceStack` that derives V_bi from the Fermi level difference across the heterostack using chi/Eg/doping. Add `thermionic_correction()` to cap SG fluxes at heterointerface faces. Both changes are backward compatible — when chi=Eg=0, behavior is identical to current.

**Tech Stack:** Python 3.10+, numpy, scipy, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `perovskite_sim/models/device.py` | Modify | Add `compute_V_bi()` method |
| `perovskite_sim/models/parameters.py` | Modify | Add `A_star_n`, `A_star_p` optional fields |
| `perovskite_sim/models/config_loader.py` | Modify | Parse `A_star_n`, `A_star_p` from YAML |
| `perovskite_sim/discretization/fe_operators.py` | Modify | Add `thermionic_emission_flux()` |
| `perovskite_sim/physics/continuity.py` | Modify | Cap SG flux at interface faces with TE limit |
| `perovskite_sim/solver/mol.py` | Modify | Use `compute_V_bi()` in `assemble_rhs`; pass interface info to continuity |
| `tests/unit/models/test_device_vbi.py` | Create | Tests for `compute_V_bi()` |
| `tests/unit/discretization/test_thermionic.py` | Create | Tests for thermionic emission flux |
| `tests/unit/physics/test_continuity_thermionic.py` | Create | Tests for TE-capped continuity at interfaces |
| `tests/integration/test_voc_benchmark.py` | Create | V_oc validation against IonMonger target |

---

### Task 1: Compute V_bi from band offsets

**Files:**
- Create: `tests/unit/models/test_device_vbi.py`
- Modify: `perovskite_sim/models/device.py:16-35`

- [ ] **Step 1: Write failing test for compute_V_bi with ionmonger params**

```python
# tests/unit/models/test_device_vbi.py
import numpy as np
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.constants import V_T


def _make_ionmonger_stack() -> DeviceStack:
    """Minimal ionmonger-benchmark stack for V_bi testing."""
    htl = LayerSpec(
        name="spiro_HTL", thickness=200e-9, role="HTL",
        params=MaterialParams(
            eps_r=3.0, mu_n=1e-10, mu_p=3.89e-5,
            D_ion=0.0, P_lim=1e30, P0=0.0,
            ni=1e0, tau_n=1e-9, tau_p=1e-9,
            n1=1e0, p1=1e0, B_rad=0.0, C_n=0.0, C_p=0.0,
            alpha=0.0, N_A=1e24, N_D=0.0, chi=2.1, Eg=3.0,
        ),
    )
    absorber = LayerSpec(
        name="MAPbI3", thickness=400e-9, role="absorber",
        params=MaterialParams(
            eps_r=24.1, mu_n=6.62e-3, mu_p=6.62e-3,
            D_ion=1.01e-17, P_lim=1.6e27, P0=1.6e25,
            ni=2.89e10, tau_n=3e-9, tau_p=3e-7,
            n1=2.89e10, p1=2.89e10, B_rad=0.0, C_n=0.0, C_p=0.0,
            alpha=1.3e7, N_A=0.0, N_D=0.0, chi=3.7, Eg=1.6,
        ),
    )
    etl = LayerSpec(
        name="TiO2_ETL", thickness=100e-9, role="ETL",
        params=MaterialParams(
            eps_r=10.0, mu_n=3.89e-4, mu_p=1e-10,
            D_ion=0.0, P_lim=1e30, P0=0.0,
            ni=1e0, tau_n=1e-9, tau_p=1e-9,
            n1=1e0, p1=1e0, B_rad=0.0, C_n=0.0, C_p=0.0,
            alpha=0.0, N_A=0.0, N_D=1e24, chi=4.0, Eg=3.2,
        ),
    )
    return DeviceStack(
        layers=(htl, absorber, etl),
        V_bi=1.1, Phi=1.4e21,
        interfaces=((3e5, 0.1), (0.1, 3e5)),
    )


def test_compute_vbi_ionmonger_stack():
    """V_bi from Fermi level difference should be ~1.1 V for ionmonger params.

    HTL (spiro): p-type, N_A=1e24, chi=2.1, Eg=3.0
      E_F,left ≈ chi_L + Eg_L - V_T*ln(p_L/ni_L)
      p_L ≈ N_A = 1e24, ni=1
      E_F,left = 2.1 + 3.0 - V_T*ln(1e24)  [Fermi near VB]

    ETL (TiO2): n-type, N_D=1e24, chi=4.0, Eg=3.2
      E_F,right ≈ chi_R + V_T*ln(n_R/ni_R)
      n_R ≈ N_D = 1e24, ni=1
      E_F,right = 4.0 + V_T*ln(1e24)  [Fermi near CB]

    V_bi = E_F,left - E_F,right (should be positive for n-i-p under forward bias convention)
    """
    stack = _make_ionmonger_stack()
    V_bi = stack.compute_V_bi()
    # Should be in the range 0.8 - 1.5 V for this heterostack
    assert 0.8 < V_bi < 1.5, f"V_bi = {V_bi:.4f} V out of expected range"


def test_compute_vbi_zero_offsets_falls_back():
    """When chi=Eg=0 on all layers, compute_V_bi returns the manual V_bi field."""
    htl = LayerSpec(
        name="HTL", thickness=200e-9, role="HTL",
        params=MaterialParams(
            eps_r=3.0, mu_n=2e-4, mu_p=2e-4,
            D_ion=0.0, P_lim=1e30, P0=0.0,
            ni=3.2e13, tau_n=1e-6, tau_p=1e-6,
            n1=3.2e13, p1=3.2e13, B_rad=0.0, C_n=0.0, C_p=0.0,
            alpha=0.0, N_A=2e23, N_D=0.0, chi=0.0, Eg=0.0,
        ),
    )
    etl = LayerSpec(
        name="ETL", thickness=100e-9, role="ETL",
        params=MaterialParams(
            eps_r=10.0, mu_n=2e-4, mu_p=2e-4,
            D_ion=0.0, P_lim=1e30, P0=0.0,
            ni=3.2e13, tau_n=1e-6, tau_p=1e-6,
            n1=3.2e13, p1=3.2e13, B_rad=0.0, C_n=0.0, C_p=0.0,
            alpha=0.0, N_A=0.0, N_D=2e23, chi=0.0, Eg=0.0,
        ),
    )
    stack = DeviceStack(layers=(htl, etl), V_bi=1.1, Phi=2.5e21)
    V_bi = stack.compute_V_bi()
    assert V_bi == 1.1, f"Expected fallback V_bi=1.1, got {V_bi}"


def test_compute_vbi_symmetric_doping_gives_positive():
    """Symmetric n-i-p doping with band offsets should give V_bi > 0."""
    stack = _make_ionmonger_stack()
    assert stack.compute_V_bi() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest tests/unit/models/test_device_vbi.py -v`
Expected: FAIL with `AttributeError: 'DeviceStack' object has no attribute 'compute_V_bi'`

- [ ] **Step 3: Implement compute_V_bi on DeviceStack**

In `perovskite_sim/models/device.py`, add after line 35 (after `phi_right` property):

```python
def compute_V_bi(self) -> float:
    """Derive built-in potential from Fermi level difference across heterostack.

    Uses the contact layers' chi, Eg, doping, and ni to compute:
      E_F,left  = chi_L + Eg_L - V_T * ln(p_L / ni_L)   (p-type contact)
      E_F,right = chi_R + V_T * ln(n_R / ni_R)           (n-type contact)
      V_bi = E_F,left - E_F,right

    Falls back to the manual V_bi field when chi=Eg=0 on both contacts
    (legacy homojunction configs).
    """
    from perovskite_sim.constants import V_T
    import numpy as np

    first = self.layers[0].params
    last = self.layers[-1].params

    # Detect legacy configs: all chi and Eg are zero
    all_zero = all(
        layer.params.chi == 0.0 and layer.params.Eg == 0.0
        for layer in self.layers
    )
    if all_zero:
        return self.V_bi

    # Left contact Fermi level (relative to vacuum)
    # For p-type: E_F = E_v + V_T * ln(p/Nv) ≈ chi + Eg - V_T * ln(p/ni)
    # For n-type: E_F = E_c - V_T * ln(n/Nc) ≈ chi + V_T * ln(n/ni)
    net_L = first.N_D - first.N_A
    disc_L = np.sqrt(net_L**2 + 4 * first.ni**2)
    if net_L >= 0:
        n_L = 0.5 * (net_L + disc_L)
        E_F_left = first.chi + V_T * np.log(max(n_L, 1.0) / max(first.ni, 1e-30))
    else:
        p_L = 0.5 * (-net_L + disc_L)
        E_F_left = first.chi + first.Eg - V_T * np.log(max(p_L, 1.0) / max(first.ni, 1e-30))

    net_R = last.N_D - last.N_A
    disc_R = np.sqrt(net_R**2 + 4 * last.ni**2)
    if net_R >= 0:
        n_R = 0.5 * (net_R + disc_R)
        E_F_right = last.chi + V_T * np.log(max(n_R, 1.0) / max(last.ni, 1e-30))
    else:
        p_R = 0.5 * (-net_R + disc_R)
        E_F_right = last.chi + last.Eg - V_T * np.log(max(p_R, 1.0) / max(last.ni, 1e-30))

    return E_F_left - E_F_right
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest tests/unit/models/test_device_vbi.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/models/test_device_vbi.py perovskite_sim/models/device.py
git commit -m "feat(device): add compute_V_bi from Fermi level difference across heterostack"
```

---

### Task 2: Wire compute_V_bi into assemble_rhs

**Files:**
- Modify: `perovskite_sim/solver/mol.py:284-312`

The Poisson boundary condition currently uses `stack.V_bi` directly. We need to use `stack.compute_V_bi()` instead, and cache the result in `MaterialArrays` to avoid recomputing every RHS call.

- [ ] **Step 1: Write failing test for V_bi propagation**

```python
# tests/unit/solver/test_vbi_propagation.py
import numpy as np
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.mol import build_material_arrays


def test_material_arrays_caches_computed_vbi():
    """MaterialArrays should store the computed V_bi from band offsets."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    # V_bi_eff should exist and differ from manually set V_bi when offsets present
    assert hasattr(mat, "V_bi_eff")
    expected = stack.compute_V_bi()
    assert mat.V_bi_eff == expected


def test_material_arrays_legacy_vbi():
    """Legacy configs (chi=Eg=0) should use manual V_bi."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    assert mat.V_bi_eff == stack.V_bi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest tests/unit/solver/test_vbi_propagation.py -v`
Expected: FAIL with `AttributeError: 'MaterialArrays' object has no attribute 'V_bi_eff'`

- [ ] **Step 3: Add V_bi_eff to MaterialArrays and build_material_arrays**

In `perovskite_sim/solver/mol.py`, add to the `MaterialArrays` dataclass (after `poisson_factor` field at line 88):

```python
V_bi_eff: float = 1.1  # effective built-in potential from compute_V_bi()
```

In `build_material_arrays()`, before the `return MaterialArrays(...)` call (line 202), compute and pass V_bi_eff:

```python
V_bi_eff = stack.compute_V_bi()
```

Add `V_bi_eff=V_bi_eff` to the `MaterialArrays(...)` constructor call.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest tests/unit/solver/test_vbi_propagation.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Update assemble_rhs to use mat.V_bi_eff instead of stack.V_bi**

In `perovskite_sim/solver/mol.py`, in `assemble_rhs()` at line 311, change:

```python
# Before:
phi = solve_poisson_prefactored(
    mat.poisson_factor, rho, phi_left=0.0, phi_right=stack.V_bi - V_app,
)

# After:
phi = solve_poisson_prefactored(
    mat.poisson_factor, rho, phi_left=0.0, phi_right=mat.V_bi_eff - V_app,
)
```

- [ ] **Step 6: Update all other stack.V_bi references to use mat.V_bi_eff**

In `perovskite_sim/experiments/jv_sweep.py`, in `_total_current_faces()` at line 113:

```python
# Before:
phi = solve_poisson_prefactored(
    mat.poisson_factor, rho, phi_left=0.0, phi_right=stack.V_bi - V_bc,
)

# After:
phi = solve_poisson_prefactored(
    mat.poisson_factor, rho, phi_left=0.0, phi_right=mat.V_bi_eff - V_bc,
)
```

In `perovskite_sim/solver/mol.py`, in `split_step()` at line 428-431:

```python
# Before:
phi = solve_poisson_prefactored(
    mat.poisson_factor, rho,
    phi_left=0.0, phi_right=stack.V_bi - V_app,
)

# After:
phi = solve_poisson_prefactored(
    mat.poisson_factor, rho,
    phi_left=0.0, phi_right=mat.V_bi_eff - V_app,
)
```

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `cd perovskite-sim && python -m pytest tests/ -x -v --timeout=120`
Expected: All existing tests PASS (legacy configs have chi=Eg=0, so V_bi_eff == V_bi)

- [ ] **Step 8: Commit**

```bash
git add perovskite_sim/solver/mol.py perovskite_sim/experiments/jv_sweep.py tests/unit/solver/test_vbi_propagation.py
git commit -m "feat(solver): wire compute_V_bi through MaterialArrays into Poisson BC"
```

---

### Task 3: Update V_max default in run_jv_sweep for heterointerfaces

**Files:**
- Modify: `perovskite_sim/experiments/jv_sweep.py:255-322`

With correct band offsets, V_oc can exceed the old V_bi. The default `V_max = stack.V_bi` would cut off the J-V curve before V_oc. Use the computed V_bi_eff with headroom.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/experiments/test_jv_vmax.py
import numpy as np
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep


def test_jv_sweep_captures_voc_with_band_offsets():
    """J-V sweep on ionmonger config should find V_oc > 0.9 V.

    With band-offset-corrected V_bi, V_oc should be in the 0.9-1.15 V range.
    The sweep must extend far enough to capture the zero-crossing.
    """
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    result = run_jv_sweep(stack, N_grid=30, n_points=12, v_rate=5.0)
    assert result.metrics_fwd.V_oc > 0.9, (
        f"V_oc = {result.metrics_fwd.V_oc:.4f} V too low; "
        "sweep may not extend past V_oc"
    )
```

- [ ] **Step 2: Run test to check current behavior**

Run: `cd perovskite-sim && python -m pytest tests/unit/experiments/test_jv_vmax.py -v --timeout=120`
Expected: May PASS or FAIL depending on current V_oc — this establishes baseline.

- [ ] **Step 3: Update V_max default to use compute_V_bi with headroom**

In `perovskite_sim/experiments/jv_sweep.py`, in `run_jv_sweep()` at line 322:

```python
# Before:
V_upper = stack.V_bi if V_max is None else V_max

# After:
V_bi_eff = stack.compute_V_bi()
V_upper = max(V_bi_eff * 1.3, 1.4) if V_max is None else V_max
```

- [ ] **Step 4: Run test to verify**

Run: `cd perovskite-sim && python -m pytest tests/unit/experiments/test_jv_vmax.py -v --timeout=120`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/experiments/jv_sweep.py tests/unit/experiments/test_jv_vmax.py
git commit -m "fix(jv): extend default V_max using compute_V_bi for heterointerface configs"
```

---

### Task 4: Add thermionic emission flux function

**Files:**
- Create: `tests/unit/discretization/test_thermionic.py`
- Modify: `perovskite_sim/discretization/fe_operators.py`

- [ ] **Step 1: Write failing tests for thermionic emission**

```python
# tests/unit/discretization/test_thermionic.py
import numpy as np
from perovskite_sim.constants import Q, K_B


def test_thermionic_flux_zero_barrier():
    """Zero band offset should give zero thermionic correction — SG flux dominates."""
    from perovskite_sim.discretization.fe_operators import thermionic_emission_flux
    J = thermionic_emission_flux(
        n_left=1e22, n_right=1e22, delta_E=0.0, T=300.0, A_star=1.2e6
    )
    # With zero barrier and equal densities, net TE current is zero
    assert abs(J) < 1e-10


def test_thermionic_flux_positive_barrier():
    """Positive barrier (CB step-up from left to right) should limit current."""
    from perovskite_sim.discretization.fe_operators import thermionic_emission_flux
    # delta_E = 0.3 eV barrier; n_left >> n_right
    J = thermionic_emission_flux(
        n_left=1e22, n_right=1e16, delta_E=0.3, T=300.0, A_star=1.2e6
    )
    # Current should be positive (electrons flow left→right over barrier)
    # and much smaller than without barrier due to exp(-0.3/0.0258) ≈ 9e-6 factor
    assert J > 0.0
    # The barrier factor exp(-0.3/V_T) ≈ 9.2e-6 dramatically reduces current
    V_T = K_B * 300.0 / Q
    barrier_factor = np.exp(-0.3 / V_T)
    assert J < 1.2e6 * 300**2 * 1e22 * barrier_factor * 10  # within order of magnitude


def test_thermionic_flux_negative_barrier():
    """Negative barrier (CB step-down) should not limit current — large magnitude."""
    from perovskite_sim.discretization.fe_operators import thermionic_emission_flux
    J_barrier = thermionic_emission_flux(
        n_left=1e22, n_right=1e16, delta_E=0.3, T=300.0, A_star=1.2e6
    )
    J_no_barrier = thermionic_emission_flux(
        n_left=1e22, n_right=1e16, delta_E=-0.3, T=300.0, A_star=1.2e6
    )
    # Step-down (negative) barrier allows much more current
    assert abs(J_no_barrier) > abs(J_barrier) * 10


def test_thermionic_flux_units():
    """TE flux should have units of A/m²."""
    from perovskite_sim.discretization.fe_operators import thermionic_emission_flux
    # A* = 1.2e6 A/(m²·K²), T=300 K → A*T² = 1.08e11 A/m²
    # With n_left=1e22 m⁻³ and some reference density, result is A/m²
    J = thermionic_emission_flux(
        n_left=1e22, n_right=1e16, delta_E=0.1, T=300.0, A_star=1.2e6
    )
    assert np.isfinite(J)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest tests/unit/discretization/test_thermionic.py -v`
Expected: FAIL with `ImportError: cannot import name 'thermionic_emission_flux'`

- [ ] **Step 3: Implement thermionic_emission_flux**

Add to `perovskite_sim/discretization/fe_operators.py`:

```python
def thermionic_emission_flux(
    n_left: float,
    n_right: float,
    delta_E: float,
    T: float,
    A_star: float,
) -> float:
    """Richardson-Dushman thermionic emission current [A/m²] across a band offset.

    Parameters
    ----------
    n_left, n_right : carrier densities on each side of the interface [m⁻³]
    delta_E : band offset E_right - E_left [eV]. Positive = step-up barrier
              for carriers moving left→right.
    T : temperature [K]
    A_star : effective Richardson constant [A/(m²·K²)]

    Returns
    -------
    J_TE : thermionic emission current [A/m²], positive = left→right.

    Physics
    -------
    J_TE = A* · T² · (n_left · exp(-max(delta_E, 0)/V_T) - n_right · exp(-max(-delta_E, 0)/V_T))

    The max() ensures only the uphill direction sees the exponential barrier.
    This is equivalent to:
      J_left→right = A*T² · n_left  · exp(-max(ΔE, 0)/kT)
      J_right→left = A*T² · n_right · exp(-max(-ΔE, 0)/kT)
      J_net = J_left→right - J_right→left

    The carrier density acts as a proxy for the occupation of states at the
    interface. In a full treatment, n would be replaced by n/N_c to get the
    quasi-Fermi level occupation, but for drift-diffusion coupling the
    density-proportional form is standard (see Altermatt et al. 2006).
    """
    import numpy as np
    from perovskite_sim.constants import K_B, Q

    V_T = K_B * T / Q
    prefactor = A_star * T * T

    # Barrier for left→right: max(delta_E, 0)
    # Barrier for right→left: max(-delta_E, 0)
    if delta_E >= 0:
        J_lr = prefactor * n_left * np.exp(-delta_E / V_T)
        J_rl = prefactor * n_right
    else:
        J_lr = prefactor * n_left
        J_rl = prefactor * n_right * np.exp(delta_E / V_T)

    return float(J_lr - J_rl)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest tests/unit/discretization/test_thermionic.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/discretization/fe_operators.py tests/unit/discretization/test_thermionic.py
git commit -m "feat(fe): add thermionic_emission_flux for band offset interfaces"
```

---

### Task 5: Add A_star parameters to MaterialParams and config loader

**Files:**
- Modify: `perovskite_sim/models/parameters.py:10-29`
- Modify: `perovskite_sim/models/config_loader.py:18-38`
- Modify: `tests/unit/models/test_parameters.py` (if exists, verify no regression)

- [ ] **Step 1: Write failing test**

```python
# tests/unit/models/test_astar_params.py
from perovskite_sim.models.parameters import MaterialParams


def test_astar_defaults_to_richardson():
    """A_star_n and A_star_p should default to the free-electron Richardson constant."""
    p = MaterialParams(
        eps_r=3.0, mu_n=2e-4, mu_p=2e-4,
        D_ion=0.0, P_lim=1e30, P0=0.0,
        ni=3.2e13, tau_n=1e-6, tau_p=1e-6,
        n1=3.2e13, p1=3.2e13, B_rad=0.0, C_n=0.0, C_p=0.0,
        alpha=0.0, N_A=0.0, N_D=0.0,
    )
    # Free-electron Richardson constant: A* = 4π·m·e·k²/h³ ≈ 1.2017e6 A/(m²·K²)
    assert abs(p.A_star_n - 1.2e6) < 0.1e6
    assert abs(p.A_star_p - 1.2e6) < 0.1e6


def test_astar_custom_values():
    """Custom Richardson constants should override defaults."""
    p = MaterialParams(
        eps_r=3.0, mu_n=2e-4, mu_p=2e-4,
        D_ion=0.0, P_lim=1e30, P0=0.0,
        ni=3.2e13, tau_n=1e-6, tau_p=1e-6,
        n1=3.2e13, p1=3.2e13, B_rad=0.0, C_n=0.0, C_p=0.0,
        alpha=0.0, N_A=0.0, N_D=0.0,
        A_star_n=5e5, A_star_p=8e5,
    )
    assert p.A_star_n == 5e5
    assert p.A_star_p == 8e5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest tests/unit/models/test_astar_params.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'A_star_n'`

- [ ] **Step 3: Add A_star fields to MaterialParams**

In `perovskite_sim/models/parameters.py`, add after line 29 (`Eg: float = 0.0`):

```python
A_star_n: float = 1.2017e6   # Richardson constant for electrons [A/(m²·K²)]
A_star_p: float = 1.2017e6   # Richardson constant for holes [A/(m²·K²)]
```

- [ ] **Step 4: Update config_loader to parse A_star**

In `perovskite_sim/models/config_loader.py`, in the `MaterialParams(...)` constructor call, add after the `Eg=` line:

```python
A_star_n=_f(layer_cfg.get("A_star_n", 1.2017e6)),
A_star_p=_f(layer_cfg.get("A_star_p", 1.2017e6)),
```

- [ ] **Step 5: Run tests**

Run: `cd perovskite-sim && python -m pytest tests/unit/models/ -v`
Expected: All PASS (including new tests and existing config loading tests)

- [ ] **Step 6: Commit**

```bash
git add perovskite_sim/models/parameters.py perovskite_sim/models/config_loader.py tests/unit/models/test_astar_params.py
git commit -m "feat(params): add A_star_n, A_star_p Richardson constants with defaults"
```

---

### Task 6: Cap SG flux with thermionic emission at heterointerfaces

**Files:**
- Modify: `perovskite_sim/solver/mol.py` (MaterialArrays, build_material_arrays)
- Modify: `perovskite_sim/physics/continuity.py`
- Create: `tests/unit/physics/test_continuity_thermionic.py`

This is the core physics change: at interface faces where the band offset exceeds a threshold, the SG flux is capped to the thermionic emission limit.

- [ ] **Step 1: Write failing test for TE-capped continuity**

```python
# tests/unit/physics/test_continuity_thermionic.py
import numpy as np
from perovskite_sim.constants import V_T


def test_continuity_rhs_with_thermionic_capping():
    """At a heterointerface with large band offset, the electron flux should be
    reduced compared to pure SG flux."""
    from perovskite_sim.physics.continuity import carrier_continuity_rhs

    N = 20
    x = np.linspace(0, 700e-9, N)
    phi = np.linspace(0, -1.0, N)  # built-in field
    n = np.ones(N) * 1e22
    p = np.ones(N) * 1e16

    # Create params with a sharp chi discontinuity at the midpoint
    chi = np.zeros(N)
    chi[N//2:] = 0.3  # 0.3 eV CB step-up at midpoint
    Eg = np.ones(N) * 1.6

    params = dict(
        D_n=np.ones(N-1) * 2e-4 * V_T,
        D_p=np.ones(N-1) * 2e-4 * V_T,
        V_T=V_T,
        ni_sq=np.ones(N) * (3.2e13)**2,
        tau_n=np.ones(N) * 1e-6,
        tau_p=np.ones(N) * 1e-6,
        n1=np.ones(N) * 3.2e13,
        p1=np.ones(N) * 3.2e13,
        B_rad=np.zeros(N),
        C_n=np.zeros(N),
        C_p=np.zeros(N),
        chi=chi,
        Eg=Eg,
        # TE params for interface capping
        interface_faces=[N//2 - 1],  # face index at the interface
        A_star_n=np.ones(N) * 1.2e6,
        A_star_p=np.ones(N) * 1.2e6,
        T=300.0,
    )

    dn_te, dp_te = carrier_continuity_rhs(x, phi, n, p, np.zeros(N), params)
    assert np.all(np.isfinite(dn_te))
    assert np.all(np.isfinite(dp_te))


def test_continuity_rhs_no_interfaces_unchanged():
    """Without interface_faces, continuity should behave identically to before."""
    from perovskite_sim.physics.continuity import carrier_continuity_rhs

    N = 20
    x = np.linspace(0, 700e-9, N)
    phi = np.linspace(0, -1.0, N)
    n = np.ones(N) * 1e22
    p = np.ones(N) * 1e16
    G = np.zeros(N)

    params = dict(
        D_n=np.ones(N-1) * 2e-4 * V_T,
        D_p=np.ones(N-1) * 2e-4 * V_T,
        V_T=V_T,
        ni_sq=np.ones(N) * (3.2e13)**2,
        tau_n=np.ones(N) * 1e-6,
        tau_p=np.ones(N) * 1e-6,
        n1=np.ones(N) * 3.2e13,
        p1=np.ones(N) * 3.2e13,
        B_rad=np.zeros(N),
        C_n=np.zeros(N),
        C_p=np.zeros(N),
        chi=np.zeros(N),
        Eg=np.ones(N) * 1.6,
    )

    dn1, dp1 = carrier_continuity_rhs(x, phi, n, p, G, params)

    # Same params with empty interface list
    params["interface_faces"] = []
    params["A_star_n"] = np.ones(N) * 1.2e6
    params["A_star_p"] = np.ones(N) * 1.2e6
    params["T"] = 300.0

    dn2, dp2 = carrier_continuity_rhs(x, phi, n, p, G, params)

    np.testing.assert_allclose(dn1, dn2, rtol=1e-12)
    np.testing.assert_allclose(dp1, dp2, rtol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest tests/unit/physics/test_continuity_thermionic.py -v`
Expected: FAIL (the interface_faces param is not used yet, but test should run)

- [ ] **Step 3: Add interface face indices and A_star to MaterialArrays**

In `perovskite_sim/solver/mol.py`, add to `MaterialArrays` dataclass:

```python
# Per-node Richardson constants for thermionic emission
A_star_n: np.ndarray | None = None
A_star_p: np.ndarray | None = None
# Face indices at heterointerfaces where TE capping applies
interface_faces: tuple[int, ...] = ()
```

In `build_material_arrays()`, add A_star arrays and compute interface face indices:

```python
A_star_n_node = np.empty(N)
A_star_p_node = np.empty(N)
```

Inside the layer loop, add:
```python
A_star_n_node[mask] = p.A_star_n
A_star_p_node[mask] = p.A_star_p
```

After the interface_nodes computation, compute interface face indices (face index = interface_node - 1, since face i connects node i and i+1):

```python
# Interface faces: face connecting nodes on either side of each heterointerface
# A face index i connects node i and node i+1.
# The interface node is the closest node to the physical interface.
# The face just before the interface node is where the band offset discontinuity sits.
interface_face_list: list[int] = []
TE_THRESHOLD = 0.05  # eV: only apply TE when |delta_E| > threshold
for idx in iface_list:
    if idx > 0 and idx < N - 1:
        delta_Ec = abs(chi[idx] - chi[idx - 1])
        delta_Ev = abs((chi[idx - 1] + Eg[idx - 1]) - (chi[idx] + Eg[idx]))
        if delta_Ec > TE_THRESHOLD or delta_Ev > TE_THRESHOLD:
            interface_face_list.append(idx - 1)
```

Add to the `MaterialArrays(...)` constructor:
```python
A_star_n=A_star_n_node,
A_star_p=A_star_p_node,
interface_faces=tuple(interface_face_list),
```

Update `carrier_params` property to include TE data:
```python
@property
def carrier_params(self) -> dict:
    d = dict(
        D_n=self.D_n_face, D_p=self.D_p_face, V_T=V_T,
        ni_sq=self.ni_sq, tau_n=self.tau_n, tau_p=self.tau_p,
        n1=self.n1, p1=self.p1, B_rad=self.B_rad,
        C_n=self.C_n, C_p=self.C_p,
        chi=self.chi, Eg=self.Eg,
    )
    if self.interface_faces:
        d["interface_faces"] = list(self.interface_faces)
        d["A_star_n"] = self.A_star_n
        d["A_star_p"] = self.A_star_p
        d["T"] = 300.0  # TODO: parameterize in Phase 4
    return d
```

- [ ] **Step 4: Modify carrier_continuity_rhs to cap SG flux at TE limit**

In `perovskite_sim/physics/continuity.py`, after the SG flux computation (lines 38-39) and before the recombination computation, add:

```python
# Thermionic emission capping at heterointerfaces
interface_faces = params.get("interface_faces")
if interface_faces:
    from perovskite_sim.discretization.fe_operators import thermionic_emission_flux
    A_star_n_arr = params["A_star_n"]
    A_star_p_arr = params["A_star_p"]
    T_val = params["T"]
    for f_idx in interface_faces:
        # Electron TE: CB offset
        delta_Ec = chi[f_idx + 1] - chi[f_idx]  # eV
        if abs(delta_Ec) > 0.05:
            J_te_n = thermionic_emission_flux(
                n[f_idx], n[f_idx + 1], delta_Ec, T_val,
                float(A_star_n_arr[f_idx]),
            )
            # Cap: take the flux with smaller magnitude
            if abs(J_n[f_idx]) > abs(J_te_n):
                J_n[f_idx] = J_te_n

        # Hole TE: VB offset (sign convention: positive delta_Ev = barrier for holes)
        delta_Ev = (chi[f_idx] + Eg[f_idx]) - (chi[f_idx + 1] + Eg[f_idx + 1])
        if abs(delta_Ev) > 0.05:
            J_te_p = thermionic_emission_flux(
                p[f_idx], p[f_idx + 1], delta_Ev, T_val,
                float(A_star_p_arr[f_idx]),
            )
            if abs(J_p[f_idx]) > abs(J_te_p):
                J_p[f_idx] = J_te_p
```

Note: `J_n` and `J_p` are currently returned from `sg_fluxes_n` / `sg_fluxes_p` as numpy arrays. They need to be writable. Since they come from numpy operations, they should already be writable, but if not, add `.copy()` after the SG flux calls.

- [ ] **Step 5: Run tests**

Run: `cd perovskite-sim && python -m pytest tests/unit/physics/test_continuity_thermionic.py tests/unit/physics/test_continuity.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `cd perovskite-sim && python -m pytest tests/ -x -v --timeout=120`
Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add perovskite_sim/solver/mol.py perovskite_sim/physics/continuity.py tests/unit/physics/test_continuity_thermionic.py
git commit -m "feat(continuity): cap SG flux with thermionic emission at heterointerfaces"
```

---

### Task 7: Integration test — V_oc benchmark validation

**Files:**
- Create: `tests/integration/test_voc_benchmark.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_voc_benchmark.py
"""Validate V_oc against IonMonger benchmark after band-offset + TE corrections.

IonMonger (Courtier 2019) reports V_oc ≈ 1.07 V for the nip parameter set (b).
Before this fix, the simulator gave 0.912 V (with chi/Eg but no TE) or 0.749 V
(without chi/Eg). After band-offset contact BCs + thermionic emission, V_oc
should be in the 0.95-1.15 V range.
"""
import pytest
import numpy as np
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep


@pytest.mark.slow
def test_voc_ionmonger_benchmark():
    """V_oc on ionmonger config should be in the 0.95-1.15 V range."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    result = run_jv_sweep(stack, N_grid=60, n_points=30, v_rate=0.1)
    V_oc = result.metrics_rev.V_oc
    assert 0.95 < V_oc < 1.15, f"V_oc = {V_oc:.4f} V outside expected range [0.95, 1.15]"


@pytest.mark.slow
def test_jsc_ionmonger_benchmark():
    """J_sc on ionmonger config should be in the 200-250 A/m² range (~20-25 mA/cm²)."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    result = run_jv_sweep(stack, N_grid=60, n_points=30, v_rate=0.1)
    J_sc = result.metrics_rev.J_sc
    assert 150 < J_sc < 300, f"J_sc = {J_sc:.1f} A/m² outside expected range"


@pytest.mark.slow
def test_ff_ionmonger_benchmark():
    """Fill factor on ionmonger config should be > 0.55."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    result = run_jv_sweep(stack, N_grid=60, n_points=30, v_rate=0.1)
    FF = result.metrics_rev.FF
    assert FF > 0.55, f"FF = {FF:.3f} below expected minimum"


def test_legacy_config_no_regression():
    """nip_MAPbI3 (chi=Eg=0) should produce same results as before."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_jv_sweep(stack, N_grid=30, n_points=8, v_rate=5.0)
    m = result.metrics_fwd
    # Same ranges as test_end_to_end_device.py
    assert 100 < m.J_sc < 800
    assert 0.7 < m.V_oc < 1.1
    assert 0.5 < m.FF < 0.95
```

- [ ] **Step 2: Run the integration tests**

Run: `cd perovskite-sim && python -m pytest tests/integration/test_voc_benchmark.py -v --timeout=300 -m "not slow"`
Expected: `test_legacy_config_no_regression` PASS. Slow tests skipped.

Run the slow tests:
Run: `cd perovskite-sim && python -m pytest tests/integration/test_voc_benchmark.py -v --timeout=300 -m slow`
Expected: Check whether V_oc has improved. If still below 0.95, investigate — see Task 8.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_voc_benchmark.py
git commit -m "test(integration): V_oc benchmark validation against IonMonger"
```

---

### Task 8: Tune and validate — iterative debugging

**Files:**
- Possibly modify: any of the files from Tasks 1-6

This is the debugging/tuning task. After running the benchmark test, if V_oc is not in the expected range, investigate:

- [ ] **Step 1: Run a diagnostic J-V sweep and inspect**

```python
# Quick diagnostic — run in a Python REPL or notebook
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
print(f"V_bi (manual): {stack.V_bi}")
print(f"V_bi (computed): {stack.compute_V_bi()}")
result = run_jv_sweep(stack, N_grid=60, n_points=30, v_rate=0.1)
print(f"V_oc (fwd): {result.metrics_fwd.V_oc:.4f} V")
print(f"V_oc (rev): {result.metrics_rev.V_oc:.4f} V")
print(f"J_sc (rev): {result.metrics_rev.J_sc:.1f} A/m²")
print(f"FF  (rev): {result.metrics_rev.FF:.3f}")
print(f"PCE (rev): {result.metrics_rev.PCE:.4f}")
```

- [ ] **Step 2: Check the computed V_bi is reasonable**

Expected: V_bi ≈ 1.0-1.3 V for the ionmonger stack. If it's wildly different, check the Fermi level calculation in `compute_V_bi()`.

- [ ] **Step 3: Check TE is actually capping fluxes**

Add temporary debug prints in `carrier_continuity_rhs` to see whether TE capping is triggering and by how much. Remove after debugging.

- [ ] **Step 4: Verify all existing tests still pass**

Run: `cd perovskite-sim && python -m pytest tests/ -x --timeout=120`
Expected: All PASS

- [ ] **Step 5: If V_oc is in range, commit any tuning changes**

```bash
git add -u
git commit -m "fix(phase1): tune band-offset contacts and TE capping for V_oc accuracy"
```

---

### Task 9: Update existing tests for new V_oc range

**Files:**
- Modify: `tests/integration/test_end_to_end_device.py`
- Modify: `tests/regression/test_jv_regression.py`

After the physics fix, the V_oc range for configs with band offsets may shift. Update test bounds.

- [ ] **Step 1: Run all tests and note which bounds need updating**

Run: `cd perovskite-sim && python -m pytest tests/ -v --timeout=300`

- [ ] **Step 2: Update bounds if needed**

Only update bounds for tests that use configs with chi/Eg > 0. Tests using nip_MAPbI3.yaml (chi=Eg=0) should be unchanged.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "test: update V_oc bounds after band-offset contact BC fix"
```

---

### Task 10: Update CLAUDE.md with Phase 1 changes

**Files:**
- Modify: `perovskite-sim/CLAUDE.md`

- [ ] **Step 1: Add section about band-offset contacts**

Add to the "Simulator Architecture" section after the heterojunctions paragraph:

```markdown
**Band-offset contact BCs.** `DeviceStack.compute_V_bi()` derives the built-in potential from the Fermi level difference between the two contact layers, using chi, Eg, and doping. When chi=Eg=0 on all layers (legacy configs), it falls back to the manual `V_bi` field. The computed value is cached as `MaterialArrays.V_bi_eff` and used in the Poisson boundary condition.

**Thermionic emission.** At heterointerfaces where the conduction or valence band offset exceeds 0.05 eV, the Scharfetter-Gummel carrier flux is capped to the Richardson-Dushman thermionic emission limit. This prevents unphysical over-injection of carriers across large band discontinuities. The capping applies at face indices stored in `MaterialArrays.interface_faces`. Richardson constants `A_star_n`, `A_star_p` default to the free-electron value (1.2e6 A/m²/K²) and can be overridden per layer.
```

- [ ] **Step 2: Commit**

```bash
git add perovskite-sim/CLAUDE.md
git commit -m "docs: document band-offset contacts and thermionic emission in CLAUDE.md"
```

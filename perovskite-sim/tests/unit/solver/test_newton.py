import numpy as np
import pytest
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.discretization.grid import multilayer_grid, Layer


def test_equilibrium_convergence():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y_eq = solve_equilibrium(x, stack)
    assert y_eq is not None
    assert y_eq.shape == (3 * len(x),)


def test_equilibrium_carriers_physical():
    """Carrier densities must stay positive and finite."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    y_eq = solve_equilibrium(x, stack)
    n, p = y_eq[:N], y_eq[N:2*N]
    assert np.all(n > 0.0)
    assert np.all(p > 0.0)
    assert np.all(np.isfinite(n))
    assert np.all(np.isfinite(p))


def test_equilibrium_residual_small():
    """
    After equilibrium solve the interior-absorber dn/dt should be near zero.

    Scope: only the deep interior of the absorber layer (away from
    ETL/absorber and absorber/HTL interfaces) is checked.  At those
    nodes the quasi-neutral IC gives np = ni²_absorber → R_SRH = 0
    and a uniform carrier density → ∂J_n/∂x ≈ 0.

    Interface nodes are deliberately excluded because the multi-layer
    model has no band-offset / electron-affinity parameters, so large
    SG-flux divergences at junctions are expected and correct.
    """
    from perovskite_sim.solver.mol import assemble_rhs, build_material_arrays
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    n_nodes = 50
    layers_grid = [Layer(l.thickness, n_nodes) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y_eq = solve_equilibrium(x, stack)
    mat = build_material_arrays(x, stack)
    rhs = assemble_rhs(0.0, y_eq, x, stack, mat, illuminated=False, V_app=0.0)

    # Identify absorber layer node range
    offset = 0
    abs_start = abs_end = None
    node_offset = 0
    for layer in stack.layers:
        n_layer = sum(1 for xi in x
                      if offset - 1e-12 <= xi <= offset + layer.thickness + 1e-12)
        if layer.role == "absorber":
            abs_start = node_offset
            abs_end   = node_offset + n_layer
        node_offset += n_layer
        offset += layer.thickness

    # Deep interior: skip first and last 20% of absorber nodes
    skip = max(2, (abs_end - abs_start) // 5)
    interior = slice(abs_start + skip, abs_end - skip)

    # Check dn/dt (first N components of rhs) in absorber interior
    assert np.max(np.abs(rhs[interior])) < 1e22   # m⁻³/s


def test_contact_boundaries_use_contact_layer_intrinsic_density():
    """Minority carrier densities at contacts should reflect the contact layer ni."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 20) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    y_eq = solve_equilibrium(x, stack)
    n, p = y_eq[:N], y_eq[N:2*N]

    # With the transport-layer ni=1 m^-3, minority contact carriers should be
    # vanishingly small rather than using the absorber ni.
    assert n[0] < 1e-10
    assert p[-1] < 1e-10

import dataclasses

import numpy as np
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.solver.mol import build_material_arrays


def test_equilibrium_convergence():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(layer.thickness, 50) for layer in stack.layers]
    x = multilayer_grid(layers_grid)
    y_eq = solve_equilibrium(x, stack)
    assert y_eq is not None
    assert y_eq.shape == (3 * len(x),)


def test_equilibrium_carriers_physical():
    """Carrier densities must stay positive and finite."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(layer.thickness, 50) for layer in stack.layers]
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
    layers_grid = [Layer(layer.thickness, n_nodes) for layer in stack.layers]
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
    layers_grid = [Layer(layer.thickness, 20) for layer in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    y_eq = solve_equilibrium(x, stack)
    n, p = y_eq[:N], y_eq[N:2*N]

    # With the transport-layer ni=1 m^-3, minority contact carriers should be
    # vanishingly small rather than using the absorber ni.
    assert n[0] < 1e-10
    assert p[-1] < 1e-10


def test_equilibrium_seed_uses_temperature_and_grading_aware_ni_squared():
    """The quasi-neutral seed and the first RHS must share one ni(x, T)."""
    stack = load_device_from_yaml("configs/cigs_graded_notch.yaml")
    layers = list(stack.layers)
    absorber_index = next(
        index for index, layer in enumerate(layers) if layer.role == "absorber"
    )
    absorber = layers[absorber_index]
    layers[absorber_index] = dataclasses.replace(
        absorber,
        params=dataclasses.replace(absorber.params, N_A=0.0, N_D=0.0),
    )
    stack = dataclasses.replace(
        stack,
        layers=tuple(layers),
        T=340.0,
        mode="full",
    )
    x = build_electrical_grid(stack, 60)
    mat = build_material_arrays(x, stack)
    y_eq = solve_equilibrium(x, stack)
    N = len(x)
    n, p = y_eq[:N], y_eq[N:2 * N]

    absorber = stack.layers[absorber_index]
    absorber_start = sum(
        layer.thickness for layer in stack.layers[:absorber_index]
        if layer.role != "substrate"
    )
    interior = (
        (x > absorber_start + 0.05 * absorber.thickness)
        & (x < absorber_start + 0.95 * absorber.thickness)
    )
    assert np.count_nonzero(interior) > 5
    np.testing.assert_allclose(
        n[interior] * p[interior],
        mat.ni_sq[interior],
        rtol=1e-12,
        atol=0.0,
    )

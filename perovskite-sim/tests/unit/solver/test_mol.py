import numpy as np
import pytest
from perovskite_sim.solver.mol import assemble_rhs, StateVec, split_step

NI = 3.2e13


def test_state_vec_roundtrip():
    N = 50
    n = NI * np.ones(N); p = NI * np.ones(N); P = 1e24 * np.ones(N)
    y = StateVec.pack(n, p, P)
    sv = StateVec.unpack(y, N)
    np.testing.assert_allclose(sv.n, n)
    np.testing.assert_allclose(sv.p, p)
    np.testing.assert_allclose(sv.P, P)


def test_assemble_rhs_shape():
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    n = NI * np.ones(N); p = NI * np.ones(N); P = 1e24 * np.ones(N)
    y0 = StateVec.pack(n, p, P)
    dydt = assemble_rhs(0.0, y0, x, stack, illuminated=False)
    assert dydt.shape == y0.shape


def test_split_step_shape_and_success():
    """split_step must return same-shaped state and succeed for a short step."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y0 = solve_illuminated_ss(x, stack, V_app=0.0)
    y_new, ok = split_step(x, y0, dt=0.05, stack=stack, V_app=0.0)
    assert ok
    assert y_new.shape == y0.shape
    assert np.all(np.isfinite(y_new))


def test_split_step_warns_on_negative_ions():
    """split_step must emit RuntimeWarning when significant ion density is clipped."""
    import warnings
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.solver.mol import StateVec, split_step
    from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    y0 = solve_illuminated_ss(x, stack, V_app=0.0)
    # Inject significantly negative ion density to force clipping path
    sv = StateVec.unpack(y0, N)
    P_negative = sv.P.copy()
    P_negative[N // 2] = -1e20   # significant negative value
    y_bad = StateVec.pack(sv.n, sv.p, P_negative)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        split_step(x, y_bad, dt=0.01, stack=stack, V_app=0.0)
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warnings) > 0, "Expected RuntimeWarning about ion clipping"
    assert "clip" in runtime_warnings[0].message.args[0].lower()


def test_interface_d_harmonic_mean():
    """D_face at a layer interface must equal 2·D_a·D_b/(D_a+D_b)."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.solver.mol import _build_carrier_params
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 20) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    params = _build_carrier_params(x, stack)
    D_n_face = params["D_n"]
    # At the HTL/absorber interface, D_n_face should be harmonic mean
    htl_thickness = stack.layers[0].thickness
    x_face = 0.5 * (x[:-1] + x[1:])
    iface_idx = int(np.argmin(np.abs(x_face - htl_thickness)))
    D_htl = stack.layers[0].params.D_n
    D_abs = stack.layers[1].params.D_n
    expected_harmonic = 2.0 * D_htl * D_abs / (D_htl + D_abs)
    assert abs(D_n_face[iface_idx] - expected_harmonic) < 1e-4 * expected_harmonic + 1e-40, (
        f"D_n_face at interface={D_n_face[iface_idx]:.3e}, "
        f"expected harmonic mean={expected_harmonic:.3e}"
    )


def test_split_step_advances_ions():
    """After a 0.05 s split step the absorber ion distribution must change."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    y0 = solve_illuminated_ss(x, stack, V_app=0.0)
    y_new, ok = split_step(x, y0, dt=0.05, stack=stack, V_app=0.0)
    assert ok
    offset = stack.layers[0].thickness
    abs_mask = (x > offset) & (x < offset + stack.layers[1].thickness)
    P0_abs = y0[2*N:][abs_mask]
    P1_abs = y_new[2*N:][abs_mask]
    assert not np.allclose(P0_abs, P1_abs)


def test_short_dark_transient_keeps_ions_in_absorber():
    """Ion-blocking transport layers should not accumulate mobile ions."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.solver.newton import solve_equilibrium
    from perovskite_sim.solver.mol import StateVec, run_transient

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y0 = solve_equilibrium(x, stack)
    N = len(x)
    sol = run_transient(
        x, y0, (0.0, 1e-9), np.array([1e-9]), stack, illuminated=False, V_app=0.0
    )
    sv = StateVec.unpack(sol.y[:, -1], N)

    abs_lo = stack.layers[0].thickness
    abs_hi = abs_lo + stack.layers[1].thickness
    htl = x < abs_lo
    etl = x > abs_hi

    assert np.allclose(sv.P[htl], 0.0, atol=1e-24)
    assert np.allclose(sv.P[etl], 0.0, atol=1e-24)
    assert np.all(sv.P >= 0.0)


def test_split_step_preserves_ion_inventory():
    """Zero-flux ion evolution should conserve total mobile-ion inventory."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
    from perovskite_sim.solver.mol import StateVec, split_step

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y0 = solve_illuminated_ss(x, stack, V_app=0.0)
    N = len(x)
    sv0 = StateVec.unpack(y0, N)

    y1, ok = split_step(x, y0, dt=0.05, stack=stack, V_app=0.0)
    assert ok
    sv1 = StateVec.unpack(y1, N)

    m0 = np.trapezoid(sv0.P, x)
    m1 = np.trapezoid(sv1.P, x)
    np.testing.assert_allclose(m1, m0, rtol=1e-10, atol=1e-6)
    assert np.all(sv1.P >= 0.0)

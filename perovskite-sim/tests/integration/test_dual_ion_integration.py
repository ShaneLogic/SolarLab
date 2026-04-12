"""Integration tests for dual-species ion migration."""
import numpy as np
import pytest
from dataclasses import replace

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays, StateVec
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.discretization.grid import multilayer_grid, Layer

pytestmark = pytest.mark.slow


def _make_dual_stack():
    """IonMonger benchmark with negative ions in the absorber."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    new_layers = []
    for layer in stack.layers:
        if layer.role == "absorber":
            new_params = replace(
                layer.params,
                D_ion_neg=3.2e-18,
                P0_neg=1.6e25,
                P_lim_neg=1.6e27,
            )
            new_layers.append(replace(layer, params=new_params))
        else:
            new_layers.append(layer)
    return replace(stack, layers=new_layers)


def _build_grid(stack, N=30):
    layers = [Layer(thickness=l.thickness, N=N) for l in stack.layers]
    return multilayer_grid(layers, alpha=3.0)


class TestDualIonEquilibrium:

    def test_state_vector_4n(self):
        """Dual-species equilibrium should produce a 4N state vector."""
        stack = _make_dual_stack()
        x = _build_grid(stack)
        N = len(x)
        y0 = solve_equilibrium(x, stack)
        assert len(y0) == 4 * N

    def test_ion_conservation_at_equilibrium(self):
        """Both species should be at their equilibrium densities."""
        stack = _make_dual_stack()
        x = _build_grid(stack)
        N = len(x)
        mat = build_material_arrays(x, stack)
        y0 = solve_equilibrium(x, stack)
        sv = StateVec.unpack(y0, N)

        P_pos_total = np.trapz(sv.P, x)
        P0_pos_total = np.trapz(mat.P_ion0, x)
        assert abs(P_pos_total - P0_pos_total) / P0_pos_total < 1e-10

        P_neg_total = np.trapz(sv.P_neg, x)
        P0_neg_total = np.trapz(mat.P_ion0_neg, x)
        assert abs(P_neg_total - P0_neg_total) / P0_neg_total < 1e-10

    def test_single_species_still_3n(self):
        """Without D_ion_neg, state vector should remain 3N."""
        stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
        x = _build_grid(stack)
        N = len(x)
        y0 = solve_equilibrium(x, stack)
        assert len(y0) == 3 * N


class TestDualIonJV:

    def test_jv_dual_species(self):
        """Dual-species J-V should produce physically reasonable metrics."""
        from perovskite_sim.experiments.jv_sweep import run_jv_sweep
        stack = _make_dual_stack()
        result = run_jv_sweep(stack, N_grid=25, n_points=8, v_rate=5.0)
        V_oc = result.metrics_rev.V_oc
        J_sc = abs(result.metrics_rev.J_sc)
        assert 0.8 < V_oc < 1.3, f"V_oc={V_oc:.3f} out of range"
        assert 100 < J_sc < 400, f"J_sc={J_sc:.1f} out of range"

    def test_jv_dual_vs_single_similar(self):
        """At fast scan rate, dual and single species should give similar results."""
        from perovskite_sim.experiments.jv_sweep import run_jv_sweep
        stack_single = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
        stack_dual = _make_dual_stack()
        r_single = run_jv_sweep(stack_single, N_grid=25, n_points=8, v_rate=5.0)
        r_dual = run_jv_sweep(stack_dual, N_grid=25, n_points=8, v_rate=5.0)
        # At 5 V/s, the slow negative species barely moves
        ratio = abs(r_dual.metrics_rev.J_sc / r_single.metrics_rev.J_sc)
        assert 0.8 < ratio < 1.2, f"J_sc ratio={ratio:.2f}"

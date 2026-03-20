import pytest
from perovskite_sim.models.parameters import MaterialParams, SolverConfig, load_config
from perovskite_sim.models.device import DeviceStack


def test_material_params_immutable():
    p = MaterialParams(
        eps_r=24.1, mu_n=2e-4, mu_p=2e-4,
        D_ion=1e-16, P_lim=1e27, P0=1e24,
        ni=3.2e13, tau_n=1e-6, tau_p=1e-6,
        n1=3.2e13, p1=3.2e13, B_rad=5e-22,
        C_n=1e-42, C_p=1e-42, alpha=1e7,
        N_A=0.0, N_D=0.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        p.eps_r = 10.0   # frozen dataclass


def test_solver_config_defaults():
    cfg = SolverConfig()
    assert cfg.rtol == 1e-4
    assert cfg.atol == 1e-6
    assert cfg.N == 200


def test_device_stack_total_thickness():
    from perovskite_sim.models.device import LayerSpec
    stack = DeviceStack(layers=[
        LayerSpec(name="ETL",       thickness=100e-9, params=None, role="ETL"),
        LayerSpec(name="perovskite",thickness=400e-9, params=None, role="absorber"),
        LayerSpec(name="HTL",       thickness=200e-9, params=None, role="HTL"),
    ])
    assert abs(stack.total_thickness - 700e-9) < 1e-12

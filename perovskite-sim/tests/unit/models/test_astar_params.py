"""Tests for Richardson constant (A_star) parameters on MaterialParams."""
from perovskite_sim.models.parameters import MaterialParams


def _make_params(**overrides) -> MaterialParams:
    """Create a MaterialParams with sensible defaults, applying overrides."""
    defaults = dict(
        eps_r=24.1,
        mu_n=2e-4,
        mu_p=2e-4,
        D_ion=1.6e-18,
        P_lim=1.6e25,
        P0=1.6e25,
        ni=5.76e10,
        tau_n=3e-9,
        tau_p=3e-9,
        n1=5.76e10,
        p1=5.76e10,
        B_rad=2.3e-17,
        C_n=0.0,
        C_p=0.0,
        alpha=1.3e7,
        N_A=0.0,
        N_D=0.0,
        chi=3.9,
        Eg=1.6,
    )
    defaults.update(overrides)
    return MaterialParams(**defaults)


def test_astar_defaults_to_richardson() -> None:
    """MaterialParams without A_star args should default to ~1.2e6 A/(m^2*K^2)."""
    params = _make_params()
    expected = 1.2017e6
    assert params.A_star_n == expected
    assert params.A_star_p == expected


def test_astar_custom_values() -> None:
    """Passing custom A_star_n and A_star_p should override defaults."""
    params = _make_params(A_star_n=5e5, A_star_p=8e5)
    assert params.A_star_n == 5e5
    assert params.A_star_p == 8e5

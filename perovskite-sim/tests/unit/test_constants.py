from perovskite_sim import constants


def test_constants_values():
    assert abs(constants.Q   - 1.602176634e-19) < 1e-30
    assert abs(constants.K_B - 1.380649e-23)    < 1e-34
    assert abs(constants.T   - 300.0)           < 1e-10
    assert abs(constants.EPS_0 - 8.854187817e-12) < 1e-23
    # V_T at 300 K ≈ 0.025852 V
    assert abs(constants.V_T - 0.025852) < 1e-5


def test_constants_consistent():
    """V_T must equal K_B*T/Q."""
    assert abs(constants.V_T - constants.K_B * constants.T / constants.Q) < 1e-15


def test_all_symbols_exported():
    for name in ("Q", "K_B", "T", "V_T", "EPS_0"):
        assert hasattr(constants, name), f"constants.{name} missing"

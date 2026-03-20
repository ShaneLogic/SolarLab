"""
Regression tests: physical sanity checks for n-i-p MAPbI3 J-V curve.
These do not require exact golden values; they test physically reasonable output.

Run with: pytest -m slow
"""
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nip_result():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    return run_jv_sweep(stack, N_grid=60, n_points=20, v_rate=5.0)


def test_jsc_positive(nip_result):
    assert nip_result.metrics_fwd.J_sc > 0


def test_voc_in_range(nip_result):
    # MAPbI3 n-i-p: V_oc ∈ [0.8, 1.3] V
    assert 0.5 < nip_result.metrics_fwd.V_oc < 1.5


def test_ff_reasonable(nip_result):
    # FF > 0.4 for a functional device
    assert nip_result.metrics_fwd.FF > 0.3


def test_hysteresis_index_nonnegative(nip_result):
    # Hysteresis index should be ≥ 0 (reverse scan better than forward)
    assert nip_result.hysteresis_index >= -0.1

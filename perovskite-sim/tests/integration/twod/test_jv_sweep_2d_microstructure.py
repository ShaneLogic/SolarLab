from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.mark.regression
@pytest.mark.slow
def test_jv_sweep_2d_singleGB_runs_to_completion():
    """End-to-end Stage-B smoke test: run_jv_sweep_2d on the singleGB preset
    must accept microstructure=None (auto-pickup from stack.microstructure),
    finish without solver blow-up, and exhibit measurable τ heterogeneity at
    the GB column relative to the bulk column at V=0."""
    stack = load_device_from_yaml("configs/twod/nip_MAPbI3_singleGB.yaml")
    res = run_jv_sweep_2d(
        stack=stack,
        microstructure=None,           # picks up stack.microstructure
        lateral_length=500e-9, Nx=8,
        V_max=0.6, V_step=0.2,
        Ny_per_layer=8, settle_t=1e-3,
    )
    assert res.V.shape == (4,)
    assert res.J.shape == (4,)
    assert np.all(np.isfinite(res.J))

    # τ heterogeneity must produce measurable carrier suppression at the
    # GB column at V=0. Two robustness considerations shape the threshold:
    # (a) .max() over the whole y-axis is dominated by the ohmic-Dirichlet
    # contact pin (~1e24), so we compare *interior* absorber nodes only —
    # those where n is bounded away from the boundary clamps in both
    # columns. (b) The shipped singleGB preset uses a moderately passivated
    # film (τ_GB=50 ns, width=5 nm), so the suppression at V=0 (far from
    # V_oc, where injection is small) is only fractions of a percent. The
    # threshold 0.999 catches a "GB has zero effect" regression while
    # accepting the modest physically-tuned drop.
    snap0 = res.snapshots[0]
    i_gb = int(np.argmin(np.abs(snap0.x - 250e-9)))
    n_gb = snap0.n[:, i_gb]
    n_bulk = snap0.n[:, 0]
    interior = (n_bulk > 1e15) & (n_bulk < 1e23)
    assert interior.sum() >= 3, \
        f"too few non-pinned interior nodes: {interior.sum()}"
    ratio_min = float((n_gb[interior] / n_bulk[interior]).min())
    assert ratio_min < 0.999, \
        f"GB column did not show carrier suppression at V=0; min n_gb/n_bulk={ratio_min:.4f}"

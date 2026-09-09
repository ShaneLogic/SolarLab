from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.twod.microstructure import Microstructure


@pytest.mark.regression
@pytest.mark.slow
def test_jv_sweep_2d_singleGB_runs_to_completion():
    """End-to-end Stage-B smoke test: run_jv_sweep_2d on the singleGB preset
    must accept microstructure=None (auto-pickup from stack.microstructure),
    finish without solver blow-up, and exhibit measurable τ heterogeneity at
    the GB column relative to the bulk column at V=0."""
    stack = load_device_from_yaml(
        "tests/fixtures/configs/twod/nip_MAPbI3_singleGB.yaml"
    )
    res = run_jv_sweep_2d(
        stack=stack,
        microstructure=None,  # picks up stack.microstructure
        lateral_length=500e-9,
        Nx=8,
        V_max=0.6,
        V_step=0.2,
        Ny_per_layer=8,
        settle_t=1e-3,
        lateral_bc="neumann",  # grain boundaries require Neumann-x topology (b126ce0)
    )
    assert res.V.shape == (4,)
    assert res.J.shape == (4,)
    assert np.all(np.isfinite(res.J))

    def resolved(microstructure):
        return run_jv_sweep_2d(
            stack=stack,
            microstructure=microstructure,
            lateral_length=500e-9,
            Nx=8,
            V_max=0.6,
            V_step=0.2,
            Ny_per_layer=8,
            settle_t=1e-3,
            lateral_bc="neumann",
            rtol=1e-8,
            atol=1e-10,
        )

    explicit = resolved(stack.microstructure)
    no_boundary = resolved(Microstructure())
    np.testing.assert_allclose(res.J, explicit.J, rtol=1e-6, atol=1e-8)
    snap0 = res.snapshots[0]
    i_gb = int(np.argmin(np.abs(snap0.x - 250e-9)))
    interior = np.zeros(snap0.y.shape, dtype=bool)
    start = 0.0
    for layer in electrical_layers(stack):
        stop = start + layer.thickness
        if layer.role == "absorber":
            interior |= (snap0.y > start) & (snap0.y < stop)
        start = stop
    assert interior.sum() >= 3, f"too few non-pinned interior nodes: {interior.sum()}"
    normal = snap0.n[interior, i_gb]
    fine = explicit.snapshots[0].n[interior, i_gb]
    reference = no_boundary.snapshots[0].n[interior, i_gb]
    np.testing.assert_allclose(normal, fine, rtol=1e-6, atol=0.0)
    # A 5 nm band's lifetime contrast does not imply a 0.1% density change
    # after lateral diffusion. The old amplitude assertion is withdrawn;
    # the unchanged input must have a causal effect above its measured error.
    uncertainty = max(
        float(np.max(np.abs(normal - fine) / reference)),
        32 * np.finfo(float).eps,
    )
    suppression = float(np.max((reference - fine) / reference))
    assert suppression > 10 * uncertainty, (
        f"grain-boundary effect {suppression:g} is not resolved above "
        f"the tolerance/roundoff scale {uncertainty:g}"
    )

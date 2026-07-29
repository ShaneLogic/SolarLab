"""The interface SRV in the shipped screening config is a grid-referenced
calibration, not a face-value surface-recombination velocity.

``configs/solarscale_nip_band_aligned_iface.yaml`` declares
``sigma * v_th * N_t`` = 0.1 m/s at both hetero-interfaces. That number was
not read off a measurement of the interface; it was tuned so the device-level
penalty lands where perovskite devices actually sit (50-150 mV), at one
specific mesh. The solver samples the cross-carrier interface SRH rate on
bulk-interior nodes and normalises it by the interface dual-cell width, so
both the sampling point and the normalisation move with the mesh — refine the
grid and the same declared SRV buys a different penalty.

CLAUDE.md and the config header both say so in prose. These tests make it
checkable, and they are written to fail in two different, useful ways:

* ``test_calibrated_penalty_holds`` pins the penalty at the mesh the config
  was calibrated on. It fails if a change to the interface machinery silently
  invalidates that calibration — which prose cannot catch.
* ``test_only_the_interface_channel_is_grid_referenced`` asserts the
  behaviour the caveat describes: refining the mesh leaves the interface-free
  device where it was and moves the interface-active one. If someone makes
  the interface channel mesh-independent, this test fails and says so — the
  caveat is then stale and CLAUDE.md needs updating. A guard that reports its
  own obsolescence beats one that quietly keeps passing.

The voltage resolution is deliberately held fixed (dV = V_max/(n_points-1) =
0.025 V) across both meshes. Scaling n_points with N_grid would vary the
SPATIAL grid and the voltage grid together, and V_oc is interpolated between
the bracketing voltage samples, so the two effects would be inseparable.
"""
from __future__ import annotations

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml

pytestmark = pytest.mark.slow

PLAIN = "configs/solarscale_nip_band_aligned.yaml"
IFACE = "configs/solarscale_nip_band_aligned_iface.yaml"

#: The mesh the SRV was calibrated on, and the one the config header names.
N_GRID_CALIBRATED = 60
N_GRID_REFINED = 120
#: Fixed across meshes: V_max / (n_points - 1) = 1.5 / 60 = 0.025 V.
V_MAX = 1.5
N_POINTS = 61


def _voc(path: str, N_grid: int) -> float:
    """Forward-sweep V_oc for a shipped config, run exactly as shipped."""
    stack = load_device_from_yaml(path)
    result = run_jv_sweep(stack, N_grid=N_grid, n_points=N_POINTS,
                          v_rate=0.5, V_max=V_MAX, v_max_max_attempts=2)
    assert result.metrics_fwd.voc_bracketed, (
        f"{path} at N_grid={N_grid} did not bracket V_oc; the envelopes below "
        "describe nothing if the sweep never crossed zero"
    )
    return result.metrics_fwd.V_oc


@pytest.fixture(scope="module")
def voc_table() -> dict[tuple[str, int], float]:
    """Four sweeps, shared by both tests (each is ~30-130 s)."""
    return {
        (cfg, n): _voc(cfg, n)
        for cfg in (PLAIN, IFACE)
        for n in (N_GRID_CALIBRATED, N_GRID_REFINED)
    }


def test_calibrated_penalty_holds(voc_table):
    """At its own mesh the declared interface costs the calibrated penalty.

    Measured 2026-07-29: plain 1.0882 V, iface 0.9002 V, penalty -188.0 mV.
    The envelope is +-10 mV — wide enough not to flap on unrelated solver
    work, tight enough that a change to the interface sampling or its
    normalisation shows up here rather than in someone's screening result.
    """
    plain = voc_table[(PLAIN, N_GRID_CALIBRATED)]
    iface = voc_table[(IFACE, N_GRID_CALIBRATED)]
    penalty_mV = 1000.0 * (iface - plain)
    assert -198.0 < penalty_mV < -178.0, (
        f"interface penalty {penalty_mV:.1f} mV at N_grid="
        f"{N_GRID_CALIBRATED}, calibrated at -188.0 mV. Either the interface "
        "channel changed, or the config did — the SRV in "
        "solarscale_nip_band_aligned_iface.yaml needs re-calibrating and the "
        "note in CLAUDE.md needs the new number."
    )


def test_only_the_interface_channel_is_grid_referenced(voc_table):
    """Refining the mesh moves the interface-active device and not the other.

    This is the caveat itself, executable. Measured 2026-07-29 going from
    N_grid 60 to 120 at fixed voltage resolution:

        plain   1.0882 -> 1.0880   (-0.2 mV — converged)
        iface   0.9002 -> 0.8884   (-11.8 mV)

    So essentially all of the mesh sensitivity lives in the interface channel;
    the same device without it is already converged at N_grid = 60. That is
    what "the SRV is grid-referenced, not a face value" means, stated as
    numbers rather than as a warning.
    """
    plain_shift_mV = 1000.0 * abs(
        voc_table[(PLAIN, N_GRID_REFINED)] - voc_table[(PLAIN, N_GRID_CALIBRATED)]
    )
    iface_shift_mV = 1000.0 * abs(
        voc_table[(IFACE, N_GRID_REFINED)] - voc_table[(IFACE, N_GRID_CALIBRATED)]
    )

    assert plain_shift_mV < 3.0, (
        f"the interface-FREE config moved {plain_shift_mV:.1f} mV between "
        f"N_grid {N_GRID_CALIBRATED} and {N_GRID_REFINED} (was 0.2 mV). It is "
        "the control here: if it is no longer converged at the coarse mesh, "
        "the contrast this test draws is not about the interface channel."
    )
    assert iface_shift_mV > 5.0, (
        f"the interface-active config moved only {iface_shift_mV:.1f} mV "
        f"between N_grid {N_GRID_CALIBRATED} and {N_GRID_REFINED} (was "
        "11.8 mV). If the interface channel has become mesh-independent that "
        "is an improvement — but the grid-referenced-SRV caveat in CLAUDE.md "
        "and in the config header is then stale and should be removed, and "
        "this test with it."
    )

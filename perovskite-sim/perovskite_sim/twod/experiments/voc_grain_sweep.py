"""V_oc(L_g) headline experiment.

For each grain size L_g in the input sequence, runs a 2D J-V sweep on a
device with one centred vertical grain boundary (GB) and periodic lateral
BCs. The GB band has reduced SRH lifetime ``tau_gb`` and width ``gb_width``;
its position is fixed at ``x = L_g / 2`` so each L_g sees one GB inside the
unit cell. Returns V_oc, J_sc, FF as functions of L_g.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

from perovskite_sim.models.device import DeviceStack
from perovskite_sim.twod.microstructure import GrainBoundary, Microstructure
from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
from perovskite_sim.experiments.jv_sweep import compute_metrics


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class VocGrainSweepResult:
    """Output of ``run_voc_grain_sweep``.

    Arrays are aligned: ``V_oc_V[k]``, ``J_sc_Am2[k]``, ``FF[k]`` correspond
    to ``grain_sizes_m[k]``. All quantities are SI: m, V, A/m².
    """
    grain_sizes_m: np.ndarray
    V_oc_V: np.ndarray
    J_sc_Am2: np.ndarray
    FF: np.ndarray


def run_voc_grain_sweep(
    stack: DeviceStack,
    grain_sizes: Sequence[float],
    *,
    tau_gb: tuple[float, float] = (1e-9, 1e-9),
    gb_width: float = 10e-9,
    Nx: int = 10,
    Ny_per_layer: int = 10,
    V_max: float = 1.2,
    V_step: float = 0.05,
    illuminated: bool = True,
    settle_t: float = 1e-3,
    progress: ProgressCallback | None = None,
) -> VocGrainSweepResult:
    """Sweep lateral grain size with one centred absorber GB.

    Each grain size ``L_g`` produces a 2D run with ``lateral_length = L_g``,
    ``Nx`` lateral intervals, periodic lateral BCs, and a single GB at
    ``x_position = L_g / 2``. Returns V_oc, J_sc, FF for each grain size.

    Parameters
    ----------
    stack
        Device configuration. ``stack.microstructure`` is ignored — this
        sweep paints a single GB per L_g.
    grain_sizes
        Sequence of grain sizes (m). Non-positive entries are skipped.
    tau_gb
        ``(tau_n, tau_p)`` inside the GB band (s). Default 1 ns / 1 ns.
    gb_width
        GB band width (m). Default 10 nm.
    Nx, Ny_per_layer
        2D mesh density passed through to ``run_jv_sweep_2d``.
    V_max, V_step
        Voltage sweep range (V).
    illuminated
        Whether the sweep is under illumination (default True).
    settle_t
        Settle time at each voltage step (s).
    progress
        Optional ``progress(stage, current, total, message)`` callback.
    """
    L_arr = np.asarray([g for g in grain_sizes if g > 0.0], dtype=float)
    n = L_arr.size
    V_oc = np.zeros(n)
    J_sc = np.zeros(n)
    FF = np.zeros(n)
    for k, L_g in enumerate(L_arr):
        gb = GrainBoundary(
            x_position=float(L_g) / 2.0,
            width=gb_width,
            tau_n=tau_gb[0],
            tau_p=tau_gb[1],
            layer_role="absorber",
        )
        ms = Microstructure(grain_boundaries=(gb,))
        r = run_jv_sweep_2d(
            stack=stack, microstructure=ms,
            lateral_length=float(L_g), Nx=Nx,
            V_max=V_max, V_step=V_step,
            illuminated=illuminated, lateral_bc="periodic",
            Ny_per_layer=Ny_per_layer, settle_t=settle_t,
        )
        V = np.asarray(r.V)
        J = np.asarray(r.J)
        if len(V) >= 2 and J[0] < 0:
            J = -J
        m = compute_metrics(V, J)
        V_oc[k] = m.V_oc
        J_sc[k] = m.J_sc
        FF[k] = m.FF
        if progress is not None:
            progress(
                "voc_grain_sweep", k + 1, n,
                f"L_g={L_g * 1e9:.0f} nm  V_oc={m.V_oc * 1e3:.1f} mV",
            )
    return VocGrainSweepResult(
        grain_sizes_m=L_arr,
        V_oc_V=V_oc,
        J_sc_Am2=J_sc,
        FF=FF,
    )

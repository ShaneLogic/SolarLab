"""4-axis independent 2D refinement harness and sparse structural/resource metrics (P06-03).

Exercises 1-axis-at-a-time refinement on ``solarlab``'s unified sparse 2D
``Assembler`` + conservative Backward Euler stepper:
1. ``Nx`` (lateral mesh intervals only),
2. ``Ny_per_layer`` (vertical mesh intervals per electrical layer only),
3. ``dt_s`` (time step size only across a fixed physical ``total_time_s`` window),
4. ``nonlinear_tolerance`` (Newton residual tolerance scale only),
while holding the other three axes strictly fixed at ``baseline``.

Records per-point ``dof``, ``jacobian_nnz``, SuperLU ``lu_nnz`` fill
(``scipy.sparse.linalg.splu(J).nnz``), ``rss_bytes``, ``wall_seconds``,
acceptance, and observable deltas from the finest point along each axis, with
``scientific_qualification=False`` and ``performance_gate_evaluated=False``
(exclusive-CPU 3-repeat timing gates are never claimed here).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.compile_ionic_2d import compile_ionic_device_2d
from solarlab.device.inputs import GrainBoundaryInput
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.two_dimensional.inputs import JV2DInput, MicrostructureInput
from solarlab.experiments.two_dimensional.jv_sweep_2d import (
    decode_json_payload,
    default_ionic_steady_options_1d,
    device_has_ions,
    embed_frozen_ion_excess_2d,
    encode_json_payload,
    project_prepared_for_electronic_2d,
    write_atomic_json,
)
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.nonlinear.ionic_2d import solve_ionic_steady_state_2d
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.time.backward_euler_2d import (
    AssembledStorage2D,
    BackwardEuler2DOptions,
    BackwardEuler2DStep,
    run_backward_euler_2d,
)
from solarlab.solvers.time.ionic_backward_euler_2d import (
    AssembledIonic2DProblem,
    IonicBackwardEuler2DOptions,
    IonicBackwardEuler2DStep,
    compute_mobile_ion_current_components_2d,
    run_ionic_backward_euler_2d,
)

__all__ = [
    "FourAxisRefinementReport2D",
    "REFINEMENT_2D_SCHEMA",
    "RefinementAxisName",
    "RefinementAxisReport2D",
    "RefinementBaseline2D",
    "RefinementPointMetrics2D",
    "current_max_rss_bytes",
    "read_refinement_report_2d",
    "run_four_axis_refinement_2d",
    "write_refinement_report_2d",
]

RefinementAxisName: TypeAlias = Literal["Nx", "Ny_per_layer", "dt_s", "nonlinear_tolerance"]
REFINEMENT_2D_SCHEMA = "solarlab.refinement_2d.report.v1"


def current_max_rss_bytes() -> int:
    """Return process peak RSS in bytes (macOS reports bytes; Linux reports KiB)."""
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


@dataclass(frozen=True, slots=True)
class RefinementBaseline2D:
    """Fixed baseline controls held constant on the three non-varied axes."""

    lateral_length_m: float = 2.0e-7
    Nx: int = 4
    Ny_per_layer: int = 6
    dt_s: float = 2.5e-7
    total_time_s: float = 1.0e-6
    nonlinear_tol_scale: float = 1.0
    voltage_V: float = 0.4
    generation_scale: float = 1.0
    lateral_bc: Literal["neumann", "periodic"] = "neumann"
    ion_dynamics: Literal["frozen", "single_mobile"] = "frozen"
    gb_width_m: float | None = 2.0e-8
    tau_gb_s: tuple[float, float] = (1.0e-9, 1.0e-9)

    def __post_init__(self) -> None:
        if not math.isfinite(self.lateral_length_m) or self.lateral_length_m <= 0.0:
            raise ValueError("lateral_length_m must be finite and positive")
        if type(self.Nx) is not int or self.Nx < 1:
            raise ValueError("Nx must be a positive integer")
        if type(self.Ny_per_layer) is not int or self.Ny_per_layer < 2:
            raise ValueError("Ny_per_layer must be at least 2")
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if not math.isfinite(self.total_time_s) or self.total_time_s < self.dt_s:
            raise ValueError("total_time_s must be >= dt_s")
        if not math.isfinite(self.nonlinear_tol_scale) or self.nonlinear_tol_scale <= 0.0:
            raise ValueError("nonlinear_tol_scale must be finite and positive")
        if not math.isfinite(self.voltage_V) or not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("voltage_V must be finite and generation_scale >= 0")
        if self.gb_width_m is not None:
            if not math.isfinite(self.gb_width_m) or self.gb_width_m <= 0.0 or self.gb_width_m > self.lateral_length_m:
                raise ValueError("gb_width_m must be in (0, lateral_length_m]")
            if self.lateral_bc != "neumann":
                raise ValueError("centered grain-boundary refinement requires lateral_bc='neumann'")
        if self.ion_dynamics == "single_mobile" and self.lateral_bc != "neumann":
            raise ValueError("single_mobile refinement requires lateral_bc='neumann'")


@dataclass(frozen=True, slots=True)
class RefinementPointMetrics2D:
    """Structural sparsity, SuperLU fill, RSS, timing, and observable record for one point."""

    axis: RefinementAxisName
    control_value: float
    nx_intervals: int
    ny_per_layer: int
    dt_s: float
    nonlinear_tol_scale: float
    dof: int
    jacobian_nnz: int
    lu_nnz: int
    rss_bytes: int
    wall_seconds: float
    newton_iterations: int
    accepted: bool
    termination: str
    terminal_current_A_m2: float
    ion_inventory_rel_drift: float | None = None
    delta_from_finest_current_A_m2: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RefinementAxisReport2D:
    """Independent 1-axis refinement series holding the other 3 axes fixed."""

    axis: RefinementAxisName
    held_fixed: dict[str, float | int | str]
    points: tuple[RefinementPointMetrics2D, ...]
    all_accepted: bool
    finest_control_value: float
    finest_terminal_current_A_m2: float | None


@dataclass(frozen=True, slots=True)
class FourAxisRefinementReport2D:
    """Complete 4-axis independent 2D refinement report (P06-03 clauses 1-2)."""

    baseline: RefinementBaseline2D
    nx_axis: RefinementAxisReport2D
    ny_axis: RefinementAxisReport2D
    dt_axis: RefinementAxisReport2D
    nonlinear_tolerance_axis: RefinementAxisReport2D
    scientific_qualification: bool = False
    performance_gate_evaluated: bool = False
    performance_gate_note: str = (
        "single_run_structural_observation_only; exclusive_cpu_3_repeat_gate_deferred"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REFINEMENT_2D_SCHEMA,
            "scientific_qualification": False,
            "performance_gate_evaluated": False,
            "performance_gate_note": self.performance_gate_note,
            "baseline": asdict(self.baseline),
            "nx_axis": asdict(self.nx_axis),
            "ny_axis": asdict(self.ny_axis),
            "dt_axis": asdict(self.dt_axis),
            "nonlinear_tolerance_axis": asdict(self.nonlinear_tolerance_axis),
        }


def _scaled_steady_options(scale: float) -> SteadyOptions:
    base = BackwardEuler2DOptions().electronic
    return replace(
        base,
        poisson_tolerance=base.poisson_tolerance * scale,
        continuity_tolerance_A_m2=base.continuity_tolerance_A_m2 * scale,
        contact_log_tolerance=base.contact_log_tolerance * scale,
    )


def _microstructure_for_baseline(
    prepared: PreparedDevice,
    baseline: RefinementBaseline2D,
) -> MicrostructureInput:
    if baseline.gb_width_m is None:
        return MicrostructureInput(grain_boundaries=())
    abs_ids = [lyr.id for lyr in prepared.layers if lyr.role != "substrate" and lyr.role == "absorber"]
    if not abs_ids:
        return MicrostructureInput(grain_boundaries=())
    gb = GrainBoundaryInput.model_validate(
        {
            "id": "refinement_gb",
            "layer_ids": abs_ids,
            "x_position": 0.5 * baseline.lateral_length_m,
            "width": baseline.gb_width_m,
            "tau_n": baseline.tau_gb_s[0],
            "tau_p": baseline.tau_gb_s[1],
        }
    )
    return MicrostructureInput(grain_boundaries=(gb,))


def _build_time_grid(total_time_s: float, dt_s: float) -> np.ndarray:
    n_steps = max(1, int(round(total_time_s / dt_s)))
    return np.linspace(0.0, total_time_s, n_steps + 1, dtype=np.float64)


def _evaluate_one_point(
    prepared: PreparedDevice,
    baseline: RefinementBaseline2D,
    *,
    axis: RefinementAxisName,
    control_value: float,
    nx_val: int,
    ny_val: int,
    dt_val: float,
    tol_scale: float,
) -> RefinementPointMetrics2D:
    t0 = time.monotonic()
    ms_input = _microstructure_for_baseline(prepared, baseline)
    t_grid = _build_time_grid(baseline.total_time_s, dt_val)
    actual_dt = float(t_grid[1] - t_grid[0])
    elec_tol = _scaled_steady_options(tol_scale)

    if baseline.ion_dynamics == "single_mobile":
        _, ion_layer_id = device_has_ions(prepared)
        exec_spatial = JV2DInput(
            kind="jv_2d",
            lateral_length=baseline.lateral_length_m,
            Nx=nx_val,
            Ny_per_layer=ny_val,
            lateral_bc=baseline.lateral_bc,
            microstructure=ms_input,
            ion_dynamics="single_mobile",
            interface_srh="off",
        )
        ionic_dev = compile_ionic_device_2d(
            prepared,
            exec_spatial,
            ion_domain_layer=ion_layer_id,
            ion_domain_endpoints=("blocking", "blocking"),
        )
        # Seed from 1D dark equilibrium on grain-free Nx=1 column
        aux_prep = project_prepared_for_electronic_2d(
            prepared, strip_ions=False, strip_interfaces=False, strip_grains=True
        )
        aux_sp = JV2DInput(
            kind="jv_2d",
            lateral_length=baseline.lateral_length_m,
            Nx=1,
            Ny_per_layer=ny_val,
            lateral_bc="neumann",
            microstructure=MicrostructureInput(grain_boundaries=()),
            ion_dynamics="single_mobile",
            interface_srh="off",
        )
        aux_dev = compile_ionic_device_2d(
            aux_prep,
            aux_sp,
            ion_domain_layer=ionic_dev.domain_layer_id,
            ion_domain_endpoints=("blocking", "blocking"),
        )
        dark_aux = solve_ionic_steady_state_2d(aux_dev, voltage_V=0.0, generation_scale=0.0)
        if not dark_aux.accepted or dark_aux.state is None:
            raise RuntimeError(f"refinement 1D ionic seed failed: {dark_aux.error}")
        ny_e = ionic_dev.electronic_component.geometry.ny
        ny_i = ionic_dev.ion_geometry.ny
        aux_nx = aux_dev.electronic_component.geometry.nx
        u_e_1d = dark_aux.state.u[:3 * aux_dev.node_count].reshape(3, ny_e, aux_nx)[:, :, 0].ravel()
        y_i_1d = dark_aux.state.u[3 * aux_dev.node_count:].reshape(ny_i, aux_nx)[:, 0]
        s0_ion = ionic_dev.state(
            np.r_[
                ionic_dev.electronic_component.extrude_coordinates(u_e_1d),
                np.repeat(y_i_1d, ionic_dev.ion_geometry.nx),
            ],
            voltage_V=0.0,
        )

        # Measure sparse Jacobian & SuperLU fill on actual BE step matrix J_BE = J + M / dt
        p_curr = AssembledIonic2DProblem(ionic_dev, baseline.voltage_V, baseline.generation_scale)
        p_prev = AssembledIonic2DProblem(ionic_dev, 0.0, 0.0)
        be_step_ion = IonicBackwardEuler2DStep(p_curr, p_prev, s0_ion.u, actual_dt)
        j_be_ion = be_step_ion.jacobian(s0_ion.u)
        if not isinstance(j_be_ion, csc_matrix):
            raise TypeError("2D ionic BE Jacobian must be a preallocated sparse csc_matrix")
        jac_nnz = int(j_be_ion.nnz)
        lu_nnz = int(splu(j_be_ion).nnz)

        ion_opts = IonicBackwardEuler2DOptions(
            electronic=elec_tol,
            ion_step_rtol=max(1.0e-12 * tol_scale, 1.0e-14),
            ion_inventory_rtol=1.0e-9,
        )
        dv_dt = baseline.voltage_V / baseline.total_time_s
        run_i = run_ionic_backward_euler_2d(
            ionic_dev,
            s0_ion,
            t_grid,
            voltage_fn=lambda t: dv_dt * t,
            generation_fn=lambda _t: baseline.generation_scale,
            options=ion_opts,
        )
        wall = time.monotonic() - t0
        rss = current_max_rss_bytes()
        if not run_i.accepted:
            return RefinementPointMetrics2D(
                axis=axis, control_value=control_value, nx_intervals=nx_val, ny_per_layer=ny_val,
                dt_s=actual_dt, nonlinear_tol_scale=tol_scale, dof=ionic_dev.state_size,
                jacobian_nnz=jac_nnz, lu_nnz=lu_nnz, rss_bytes=rss, wall_seconds=wall,
                newton_iterations=run_i.total_newton_iterations, accepted=False, termination=run_i.termination,
                terminal_current_A_m2=math.nan, ion_inventory_rel_drift=None, error=run_i.error,
            )
        curr_c = compute_mobile_ion_current_components_2d(
            ionic_dev,
            run_i.states[-1],
            generation_scale=baseline.generation_scale,
            applied_voltage_rate_V_s=dv_dt,
        )
        return RefinementPointMetrics2D(
            axis=axis, control_value=control_value, nx_intervals=nx_val, ny_per_layer=ny_val,
            dt_s=actual_dt, nonlinear_tol_scale=tol_scale, dof=ionic_dev.state_size,
            jacobian_nnz=jac_nnz, lu_nnz=lu_nnz, rss_bytes=rss, wall_seconds=wall,
            newton_iterations=run_i.total_newton_iterations, accepted=True, termination=run_i.termination,
            terminal_current_A_m2=curr_c.terminal_total_A_m2,
            ion_inventory_rel_drift=run_i.ion_diagnostics.relative_inventory_drift,
        )

    # Frozen-ion / classical 2D refinement lane
    has_ions, ion_layer_id = device_has_ions(prepared)
    clean_prep = project_prepared_for_electronic_2d(prepared, strip_ions=has_ions, strip_interfaces=False)
    geom_sp = JV2DInput(
        kind="jv_2d",
        lateral_length=baseline.lateral_length_m,
        Nx=nx_val,
        Ny_per_layer=ny_val,
        lateral_bc=baseline.lateral_bc,
        microstructure=ms_input,
    )
    elec_dev = compile_device_2d(clean_prep, geom_sp)
    if has_ions and ion_layer_id is not None:
        n_layers = len([lyr for lyr in prepared.layers if lyr.role != "substrate"])
        prep_1d_ion = project_prepared_for_electronic_2d(
            prepared, strip_ions=False, strip_interfaces=False, strip_grains=True
        )
        from solarlab.device.compile_ionic_1d import compile_ionic_device_1d

        ion_1d = compile_ionic_device_1d(
            prep_1d_ion,
            intervals_per_layer=(ny_val,) * n_layers,
            ion_domain_layer=ion_layer_id,
            ion_domain_endpoints=("blocking", "blocking"),
        )
        dark_1d_i = solve_ionic_steady_state(
            ion_1d,
            voltage_V=0.0,
            generation_scale=0.0,
            options=default_ionic_steady_options_1d(ion_1d.ion_node_count),
        )
        if not dark_1d_i.accepted or dark_1d_i.state is None:
            raise RuntimeError(f"refinement 1D frozen-ion seed failed: {dark_1d_i.error}")
        exc: np.ndarray = np.zeros(ion_1d.node_count, dtype=np.float64)
        exc[ion_1d.ion_node_indices] = (
            dark_1d_i.state.positive_ion_density_m3 - ion_1d.positive_ion_reference_m3
        ) * ion_1d.ion_charge_weight_ratio
        elec_dev = embed_frozen_ion_excess_2d(elec_dev, exc)
        u_1d = dark_1d_i.state.u[:3 * ion_1d.node_count]
    else:
        dark_1d = solve_steady_state(elec_dev.column, voltage_V=0.0, generation_scale=0.0)
        if dark_1d.state is None:
            raise RuntimeError("refinement 1D dark seed failed")
        u_1d = dark_1d.state.u

    s0_elec = elec_dev.state(elec_dev.extrude_coordinates(u_1d), voltage_V=0.0)
    st_curr = AssembledStorage2D(elec_dev, baseline.voltage_V, baseline.generation_scale)
    st_prev = AssembledStorage2D(elec_dev, 0.0, 0.0)
    be_step_e = BackwardEuler2DStep(st_curr, st_prev, s0_elec.u, actual_dt)
    j_be_e = be_step_e.jacobian(s0_elec.u)
    if not isinstance(j_be_e, csc_matrix):
        raise TypeError("2D BE Jacobian must be a preallocated sparse csc_matrix")
    jac_nnz = int(j_be_e.nnz)
    lu_nnz = int(splu(j_be_e).nnz)

    be_opts = BackwardEuler2DOptions(electronic=elec_tol)
    dv_dt = baseline.voltage_V / baseline.total_time_s
    run_e = run_backward_euler_2d(
        elec_dev,
        s0_elec,
        t_grid,
        voltage_fn=lambda t: dv_dt * t,
        generation_fn=lambda _t: baseline.generation_scale,
        options=be_opts,
    )
    wall = time.monotonic() - t0
    rss = current_max_rss_bytes()
    if not run_e.accepted:
        return RefinementPointMetrics2D(
            axis=axis, control_value=control_value, nx_intervals=nx_val, ny_per_layer=ny_val,
            dt_s=actual_dt, nonlinear_tol_scale=tol_scale, dof=3 * elec_dev.node_count,
            jacobian_nnz=jac_nnz, lu_nnz=lu_nnz, rss_bytes=rss, wall_seconds=wall,
            newton_iterations=run_e.total_newton_iterations, accepted=False, termination=run_e.termination,
            terminal_current_A_m2=math.nan, error=run_e.error,
        )
    j_term = float(run_e.accountings[-1].terminal_total_A_m2[1])
    return RefinementPointMetrics2D(
        axis=axis, control_value=control_value, nx_intervals=nx_val, ny_per_layer=ny_val,
        dt_s=actual_dt, nonlinear_tol_scale=tol_scale, dof=3 * elec_dev.node_count,
        jacobian_nnz=jac_nnz, lu_nnz=lu_nnz, rss_bytes=rss, wall_seconds=wall,
        newton_iterations=run_e.total_newton_iterations, accepted=True, termination=run_e.termination,
        terminal_current_A_m2=j_term,
    )


def _finalize_axis(
    axis: RefinementAxisName,
    held_fixed: dict[str, float | int | str],
    raw_points: list[RefinementPointMetrics2D],
    *,
    finest_is_largest: bool,
) -> RefinementAxisReport2D:
    ordered = sorted(raw_points, key=lambda p: p.control_value, reverse=not finest_is_largest)
    # After sorting: ordered[-1] is always the finest point (largest Nx/Ny or smallest dt/tol)
    finest = ordered[-1]
    finest_j = finest.terminal_current_A_m2 if finest.accepted else None
    enriched = tuple(
        replace(
            pt,
            delta_from_finest_current_A_m2=(
                abs(pt.terminal_current_A_m2 - finest_j)
                if pt.accepted and finest_j is not None
                else None
            ),
        )
        for pt in ordered
    )
    return RefinementAxisReport2D(
        axis=axis,
        held_fixed=held_fixed,
        points=enriched,
        all_accepted=all(pt.accepted for pt in enriched),
        finest_control_value=finest.control_value,
        finest_terminal_current_A_m2=finest_j,
    )


def run_four_axis_refinement_2d(
    prepared: PreparedDevice,
    *,
    baseline: RefinementBaseline2D | None = None,
    nx_values: Sequence[int] = (4, 8, 12),
    ny_per_layer_values: Sequence[int] = (4, 8, 12),
    dt_s_values: Sequence[float] = (5.0e-7, 2.5e-7, 1.25e-7),
    nonlinear_tol_scales: Sequence[float] = (10.0, 1.0, 0.1),
) -> FourAxisRefinementReport2D:
    """Execute independent 1-axis-at-a-time 2D refinements across Nx, Ny, dt, and nonlinear tol."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("PreparedDevice is required")
    base = RefinementBaseline2D() if baseline is None else baseline

    nx_seq = [int(v) for v in nx_values]
    ny_seq = [int(v) for v in ny_per_layer_values]
    dt_seq = [float(v) for v in dt_s_values]
    tol_seq = [float(v) for v in nonlinear_tol_scales]
    if len(set(nx_seq)) < 2 or any(v < 1 for v in nx_seq):
        raise ValueError("nx_values requires at least 2 distinct positive integers")
    if len(set(ny_seq)) < 2 or any(v < 2 for v in ny_seq):
        raise ValueError("ny_per_layer_values requires at least 2 distinct integers >= 2")
    if len(set(dt_seq)) < 2 or any(not math.isfinite(v) or v <= 0.0 or v > base.total_time_s for v in dt_seq):
        raise ValueError("dt_s_values requires at least 2 distinct positive steps <= baseline.total_time_s")
    if len(set(tol_seq)) < 2 or any(not math.isfinite(v) or v <= 0.0 for v in tol_seq):
        raise ValueError("nonlinear_tol_scales requires at least 2 distinct positive scales")

    # Axis 1: Lateral Nx only
    nx_pts = [
        _evaluate_one_point(
            prepared, base, axis="Nx", control_value=float(nx_v),
            nx_val=nx_v, ny_val=base.Ny_per_layer, dt_val=base.dt_s, tol_scale=base.nonlinear_tol_scale,
        )
        for nx_v in nx_seq
    ]
    nx_rep = _finalize_axis(
        "Nx",
        {
            "Ny_per_layer": base.Ny_per_layer,
            "dt_s": base.dt_s,
            "nonlinear_tol_scale": base.nonlinear_tol_scale,
        },
        nx_pts,
        finest_is_largest=True,
    )

    # Axis 2: Vertical Ny_per_layer only
    ny_pts = [
        _evaluate_one_point(
            prepared, base, axis="Ny_per_layer", control_value=float(ny_v),
            nx_val=base.Nx, ny_val=ny_v, dt_val=base.dt_s, tol_scale=base.nonlinear_tol_scale,
        )
        for ny_v in ny_seq
    ]
    ny_rep = _finalize_axis(
        "Ny_per_layer",
        {
            "Nx": base.Nx,
            "dt_s": base.dt_s,
            "nonlinear_tol_scale": base.nonlinear_tol_scale,
        },
        ny_pts,
        finest_is_largest=True,
    )

    # Axis 3: Temporal dt_s only
    dt_pts = [
        _evaluate_one_point(
            prepared, base, axis="dt_s", control_value=dt_v,
            nx_val=base.Nx, ny_val=base.Ny_per_layer, dt_val=dt_v, tol_scale=base.nonlinear_tol_scale,
        )
        for dt_v in dt_seq
    ]
    dt_rep = _finalize_axis(
        "dt_s",
        {
            "Nx": base.Nx,
            "Ny_per_layer": base.Ny_per_layer,
            "nonlinear_tol_scale": base.nonlinear_tol_scale,
        },
        dt_pts,
        finest_is_largest=False,
    )

    # Axis 4: Nonlinear tolerance scale only
    tol_pts = [
        _evaluate_one_point(
            prepared, base, axis="nonlinear_tolerance", control_value=tol_v,
            nx_val=base.Nx, ny_val=base.Ny_per_layer, dt_val=base.dt_s, tol_scale=tol_v,
        )
        for tol_v in tol_seq
    ]
    tol_rep = _finalize_axis(
        "nonlinear_tolerance",
        {
            "Nx": base.Nx,
            "Ny_per_layer": base.Ny_per_layer,
            "dt_s": base.dt_s,
        },
        tol_pts,
        finest_is_largest=False,
    )

    return FourAxisRefinementReport2D(
        baseline=base,
        nx_axis=nx_rep,
        ny_axis=ny_rep,
        dt_axis=dt_rep,
        nonlinear_tolerance_axis=tol_rep,
        scientific_qualification=False,
        performance_gate_evaluated=False,
    )


def write_refinement_report_2d(path: str | Path, report: FourAxisRefinementReport2D) -> str:
    """Write a lossless JSON artifact for ``FourAxisRefinementReport2D`` and return SHA-256."""
    target = Path(path)
    write_atomic_json(target, encode_json_payload(report.to_dict()))
    return hashlib.sha256(target.read_bytes()).hexdigest()


def read_refinement_report_2d(
    path: str | Path, *, expected_sha256: str | None = None
) -> FourAxisRefinementReport2D:
    """Read and verify a lossless JSON ``FourAxisRefinementReport2D`` artifact."""
    raw = Path(path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError("FourAxisRefinementReport2D SHA-256 mismatch")
    doc = decode_json_payload(json.loads(raw))
    if (
        doc.get("schema") != REFINEMENT_2D_SCHEMA
        or doc.get("scientific_qualification") is not False
        or doc.get("performance_gate_evaluated") is not False
    ):
        raise ValueError("invalid FourAxisRefinementReport2D schema or qualification flags")
    base_d = dict(doc["baseline"])
    base_d["tau_gb_s"] = tuple(base_d["tau_gb_s"])

    def load_axis(axis_doc: dict[str, Any]) -> RefinementAxisReport2D:
        pts = tuple(RefinementPointMetrics2D(**p) for p in axis_doc["points"])
        return RefinementAxisReport2D(
            axis=cast(RefinementAxisName, axis_doc["axis"]),
            held_fixed=dict(axis_doc["held_fixed"]),
            points=pts,
            all_accepted=bool(axis_doc["all_accepted"]),
            finest_control_value=float(axis_doc["finest_control_value"]),
            finest_terminal_current_A_m2=(
                None if axis_doc["finest_terminal_current_A_m2"] is None
                else float(axis_doc["finest_terminal_current_A_m2"])
            ),
        )

    return FourAxisRefinementReport2D(
        baseline=RefinementBaseline2D(**base_d),
        nx_axis=load_axis(doc["nx_axis"]),
        ny_axis=load_axis(doc["ny_axis"]),
        dt_axis=load_axis(doc["dt_axis"]),
        nonlinear_tolerance_axis=load_axis(doc["nonlinear_tolerance_axis"]),
        scientific_qualification=False,
        performance_gate_evaluated=False,
        performance_gate_note=str(doc["performance_gate_note"]),
    )

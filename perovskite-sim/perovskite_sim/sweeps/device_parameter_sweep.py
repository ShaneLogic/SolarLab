"""Controlled parameter sweeps for band alignment, doping, and defects.

The helpers in this module keep the SolarLab solver path unchanged. Each
``SweepPoint`` starts from a baseline ``DeviceStack``, applies a small set of
material/interface parameter updates with explicit unit conversions, then runs
the public ``run_jv_sweep`` experiment API.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from perovskite_sim.constants import V_T
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec, electrical_layers
from perovskite_sim.models.parameters import MaterialParams


@dataclass(frozen=True)
class SweepPoint:
    """One parameter point in a controlled device sweep."""

    point_id: str
    axis: str
    label: str
    updates: Mapping[str, float | str] = field(default_factory=dict)


def cm3_to_m3(value_cm3: float) -> float:
    """Convert a volumetric density from cm^-3 to m^-3."""

    return float(value_cm3) * 1.0e6


def cms_to_ms(value_cm_s: float) -> float:
    """Convert a velocity from cm/s to m/s."""

    return float(value_cm_s) * 1.0e-2


def srh_n1_p1_from_trap_depth(
    ni_m3: float,
    Eg_eV: float,
    trap_depth_eV: float,
    *,
    reference: str = "below_cb",
    thermal_voltage: float = V_T,
) -> tuple[float, float]:
    """Map a trap energy to SRH ``n1``/``p1`` densities.

    ``n1`` and ``p1`` follow the standard SRH relation, referenced to the
    intrinsic level ``E_i``:

    ``n1 = ni * exp((E_t - E_i) / V_T)``
    ``p1 = ni * exp((E_i - E_t) / V_T)``

    Supported ``reference`` values:

    - ``below_cb``: ``trap_depth_eV`` is ``E_c - E_t``.
    - ``above_vb``: ``trap_depth_eV`` is ``E_t - E_v``.
    - ``from_midgap``: ``trap_depth_eV`` is signed ``E_t - E_i``.
    - ``midgap``: force ``E_t = E_i`` and ignore ``trap_depth_eV``.
    """

    if ni_m3 <= 0.0:
        raise ValueError(f"ni_m3 must be positive, got {ni_m3}")
    if Eg_eV <= 0.0:
        raise ValueError(f"Eg_eV must be positive, got {Eg_eV}")
    if thermal_voltage <= 0.0:
        raise ValueError(f"thermal_voltage must be positive, got {thermal_voltage}")

    reference_key = reference.lower()
    if reference_key == "below_cb":
        et_minus_ei = Eg_eV / 2.0 - float(trap_depth_eV)
    elif reference_key == "above_vb":
        et_minus_ei = float(trap_depth_eV) - Eg_eV / 2.0
    elif reference_key == "from_midgap":
        et_minus_ei = float(trap_depth_eV)
    elif reference_key == "midgap":
        et_minus_ei = 0.0
    else:
        raise ValueError(
            "trap depth reference must be one of "
            "'below_cb', 'above_vb', 'from_midgap', or 'midgap'"
        )

    ratio = math.exp(et_minus_ei / thermal_voltage)
    return float(ni_m3 * ratio), float(ni_m3 / ratio)


def make_pilot_points() -> list[SweepPoint]:
    """Return a notebook-safe one-factor pilot set.

    The set is intentionally small enough for interactive development while
    touching every requested parameter family.
    """

    points = [SweepPoint("baseline", "baseline", "baseline", {})]
    points.extend(
        SweepPoint(f"etl_delta_ec_{_slug(v)}", "etl_delta_ec", f"{v:g} eV", {"etl_delta_ec_eV": v})
        for v in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    )
    points.extend(
        SweepPoint(f"htl_delta_ev_{_slug(v)}", "htl_delta_ev", f"{v:g} eV", {"htl_delta_ev_eV": v})
        for v in [-0.5, -0.25, 0.0, 0.25, 0.5]
    )
    points.extend(
        SweepPoint(
            f"etl_doping_{_sci_slug(v)}",
            "etl_doping",
            f"{v:.0e} cm^-3",
            {"etl_doping_cm3": v},
        )
        for v in [1e10, 1e12, 1e14, 1e16, 1e18, 1e20]
    )
    points.extend(
        SweepPoint(
            f"absorber_doping_{_sci_slug(v)}",
            "absorber_doping",
            f"{v:.0e} cm^-3",
            {"absorber_doping_cm3": v, "absorber_doping_type": "acceptor"},
        )
        for v in [1e8, 1e10, 1e12, 1e14, 1e16, 1e18]
    )
    points.extend(
        SweepPoint(
            f"absorber_defect_depth_{_slug(v)}",
            "absorber_defect_depth",
            f"{v:g} eV below CB",
            {"absorber_defect_depth_eV": v, "trap_depth_reference": "below_cb"},
        )
        for v in [0.1, 0.3, 0.6]
    )
    points.extend(
        SweepPoint(
            f"absorber_defect_density_{_sci_slug(v)}",
            "absorber_defect_density",
            f"{v:.0e} cm^-3",
            {"absorber_defect_density_cm3": v},
        )
        for v in [1e10, 1e14, 1e18]
    )
    points.extend(
        SweepPoint(
            f"interface_trap_density_{_sci_slug(v)}",
            "interface_trap_density",
            f"{v:.0e} cm^-3",
            {"interface_trap_density_cm3": v, "interface_trap_decay_nm": 10.0},
        )
        for v in [1e8, 1e13, 1e18]
    )
    points.extend(
        SweepPoint(
            f"interface_srv_{_sci_slug(v)}",
            "interface_srv",
            f"{v:.0e} cm/s",
            {"interface_srv_cm_s": v},
        )
        for v in [1e1, 1e4, 1e7]
    )
    return points


def make_coupled_points() -> list[SweepPoint]:
    """Return a small 3 x 3 x 3 coupled matrix."""

    points: list[SweepPoint] = []
    for delta_ec in [-0.5, 0.0, 0.5]:
        for etl_doping in [1e12, 1e16, 1e20]:
            for interface_srv in [1e1, 1e4, 1e7]:
                point_id = (
                    f"coupled_ec_{_slug(delta_ec)}"
                    f"__nd_{_sci_slug(etl_doping)}"
                    f"__srv_{_sci_slug(interface_srv)}"
                )
                points.append(
                    SweepPoint(
                        point_id,
                        "coupled_delta_ec_etl_doping_srv",
                        f"DeltaEc={delta_ec:g} eV, ND={etl_doping:.0e} cm^-3, SRV={interface_srv:.0e} cm/s",
                        {
                            "etl_delta_ec_eV": delta_ec,
                            "etl_doping_cm3": etl_doping,
                            "interface_srv_cm_s": interface_srv,
                        },
                    )
                )
    return points


def make_defect_matrix_points() -> list[SweepPoint]:
    """Return the absorber defect depth x density matrix requested by the user."""

    points: list[SweepPoint] = []
    for depth in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for density in _decades(10, 18):
            points.append(
                SweepPoint(
                    f"defect_depth_{_slug(depth)}__density_{_sci_slug(density)}",
                    "absorber_defect_depth_density",
                    f"{depth:g} eV below CB, {density:.0e} cm^-3",
                    {
                        "absorber_defect_depth_eV": depth,
                        "trap_depth_reference": "below_cb",
                        "absorber_defect_density_cm3": density,
                    },
                )
            )
    return points


def make_full_one_factor_points(*, etl_delta_step: float = 0.1) -> list[SweepPoint]:
    """Return one-factor sweeps over the full requested ranges."""

    points = [SweepPoint("baseline", "baseline", "baseline", {})]
    points.extend(
        SweepPoint(f"etl_delta_ec_{_slug(v)}", "etl_delta_ec", f"{v:g} eV", {"etl_delta_ec_eV": v})
        for v in _float_range(-1.0, 1.0, etl_delta_step)
    )
    points.extend(
        SweepPoint(
            f"etl_doping_{_sci_slug(v)}",
            "etl_doping",
            f"{v:.0e} cm^-3",
            {"etl_doping_cm3": v},
        )
        for v in _decades(10, 20)
    )
    points.extend(
        SweepPoint(
            f"absorber_doping_{_sci_slug(v)}",
            "absorber_doping",
            f"{v:.0e} cm^-3",
            {"absorber_doping_cm3": v, "absorber_doping_type": "acceptor"},
        )
        for v in _decades(8, 18)
    )
    points.extend(make_defect_matrix_points())
    points.extend(
        SweepPoint(
            f"interface_trap_density_{_sci_slug(v)}",
            "interface_trap_density",
            f"{v:.0e} cm^-3",
            {"interface_trap_density_cm3": v, "interface_trap_decay_nm": 10.0},
        )
        for v in _decades(8, 18)
    )
    points.extend(
        SweepPoint(
            f"interface_srv_{_sci_slug(v)}",
            "interface_srv",
            f"{v:.0e} cm/s",
            {"interface_srv_cm_s": v},
        )
        for v in _decades(1, 7)
    )
    return points


def apply_sweep_point(
    stack: DeviceStack,
    point: SweepPoint,
    *,
    sync_vbi: bool = True,
    trap_tau_reference_density_m3: float = 1.0e22,
    default_interface_trap_decay_m: float = 10.0e-9,
) -> DeviceStack:
    """Return ``stack`` with ``point`` updates applied."""

    updated = stack
    updates = dict(point.updates)

    if "etl_delta_ec_eV" in updates:
        updated = _apply_etl_delta_ec(updated, float(updates["etl_delta_ec_eV"]))
    if "htl_delta_ev_eV" in updates:
        updated = _apply_htl_delta_ev(updated, float(updates["htl_delta_ev_eV"]))
    if "etl_doping_cm3" in updates:
        updated = _replace_layer_params_by_role(
            updated,
            "ETL",
            N_D=cm3_to_m3(float(updates["etl_doping_cm3"])),
            N_A=0.0,
        )
    if "absorber_doping_cm3" in updates:
        density_m3 = cm3_to_m3(float(updates["absorber_doping_cm3"]))
        doping_type = str(updates.get("absorber_doping_type", "acceptor")).lower()
        if doping_type in {"acceptor", "p", "p-type"}:
            updated = _replace_layer_params_by_role(updated, "absorber", N_A=density_m3, N_D=0.0)
        elif doping_type in {"donor", "n", "n-type"}:
            updated = _replace_layer_params_by_role(updated, "absorber", N_D=density_m3, N_A=0.0)
        else:
            raise ValueError(f"unknown absorber_doping_type {doping_type!r}")
    if "absorber_defect_depth_eV" in updates:
        updated = _apply_absorber_defect_depth(
            updated,
            float(updates["absorber_defect_depth_eV"]),
            reference=str(updates.get("trap_depth_reference", "below_cb")),
        )
    if "absorber_defect_density_cm3" in updates:
        updated = _apply_absorber_defect_density(
            updated,
            cm3_to_m3(float(updates["absorber_defect_density_cm3"])),
            trap_tau_reference_density_m3=trap_tau_reference_density_m3,
        )
    if "interface_trap_density_cm3" in updates:
        decay_m = float(updates.get("interface_trap_decay_nm", default_interface_trap_decay_m * 1e9)) * 1.0e-9
        updated = _apply_absorber_interface_trap_density(
            updated,
            cm3_to_m3(float(updates["interface_trap_density_cm3"])),
            trap_tau_reference_density_m3=trap_tau_reference_density_m3,
            trap_decay_length_m=decay_m,
        )
    if "interface_srv_cm_s" in updates:
        updated = _apply_internal_interface_srv(
            updated,
            cms_to_ms(float(updates["interface_srv_cm_s"])),
            target=str(updates.get("interface_target", "all")),
        )
    if "interface_defect_N_t_cm2" in updates:
        updated = _apply_interface_defect_N_t_cm2(
            updated,
            float(updates["interface_defect_N_t_cm2"]),
            target=str(updates.get("interface_defect_target", "pvk/etl")),
        )
    if "interface_defect_E_t_eV" in updates:
        updated = _apply_interface_defect_E_t_eV(
            updated,
            float(updates["interface_defect_E_t_eV"]),
            target=str(updates.get("interface_defect_target", "pvk/etl")),
        )

    if sync_vbi:
        updated = dataclasses.replace(updated, V_bi=updated.compute_V_bi())
    return updated


def run_sweep(
    config_path: str | Path,
    points: Sequence[SweepPoint],
    *,
    N_grid: int = 30,
    n_points: int = 8,
    v_rate: float = 5.0,
    V_max: float | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_points: int | None = None,
    sync_vbi: bool = True,
) -> dict[str, Any]:
    """Run a controlled JV sweep matrix and return JSON-serialisable results."""

    resolved_config_path = _resolve_existing_path(config_path)
    baseline = load_device_from_yaml(str(resolved_config_path))
    selected_points = list(points if max_points is None else points[:max_points])
    records = []
    started = time.perf_counter()
    for index, point in enumerate(selected_points, start=1):
        records.append(
            run_sweep_point(
                baseline,
                point,
                index=index,
                N_grid=N_grid,
                n_points=n_points,
                v_rate=v_rate,
                V_max=V_max,
                rtol=rtol,
                atol=atol,
                sync_vbi=sync_vbi,
            )
        )
    elapsed = time.perf_counter() - started
    return {
        "schema": "solarlab.device_parameter_sweep",
        "schema_version": "0.1",
        "config_path": str(resolved_config_path),
        "settings": {
            "N_grid": N_grid,
            "n_points": n_points,
            "v_rate": v_rate,
            "V_max": V_max,
            "rtol": rtol,
            "atol": atol,
            "sync_vbi": sync_vbi,
            "requested_points": len(points),
            "executed_points": len(selected_points),
            "max_points": max_points,
        },
        "summary": _summarise_records(records, elapsed),
        "records": records,
    }


def run_sweep_point(
    baseline: DeviceStack,
    point: SweepPoint,
    *,
    index: int = 1,
    N_grid: int = 30,
    n_points: int = 8,
    v_rate: float = 5.0,
    V_max: float | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    sync_vbi: bool = True,
) -> dict[str, Any]:
    """Run one point and capture success or failure without aborting the sweep."""

    started = time.perf_counter()
    record: dict[str, Any] = {
        "index": index,
        "point_id": point.point_id,
        "axis": point.axis,
        "label": point.label,
        "updates": dict(point.updates),
    }
    try:
        stack = apply_sweep_point(baseline, point, sync_vbi=sync_vbi)
        record["derived"] = describe_stack(stack)
        result = run_jv_sweep(
            stack,
            N_grid=N_grid,
            n_points=n_points,
            v_rate=v_rate,
            V_max=V_max,
            rtol=rtol,
            atol=atol,
        )
        record.update(
            {
                "simulation_status": "succeeded",
                "metrics_fwd": dataclasses.asdict(result.metrics_fwd),
                "metrics_rev": dataclasses.asdict(result.metrics_rev),
                "hysteresis_index": float(result.hysteresis_index),
            }
        )
    except Exception as exc:  # noqa: BLE001 - matrix runs should keep going
        record.update(
            {
                "simulation_status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    record["elapsed_s"] = time.perf_counter() - started
    return record


def describe_stack(stack: DeviceStack) -> dict[str, Any]:
    """Return compact derived quantities useful for plots and audits."""

    absorber = _params_by_role(stack, "absorber")
    etl = _params_by_role(stack, "ETL")
    htl = _params_by_role(stack, "HTL")
    out: dict[str, Any] = {
        "configured_V_bi": stack.V_bi,
        "computed_V_bi": stack.compute_V_bi(),
        "interfaces_m_s": [list(pair) for pair in stack.interfaces],
    }
    if absorber is not None:
        out.update(
            {
                "absorber_chi_eV": absorber.chi,
                "absorber_Eg_eV": absorber.Eg,
                "absorber_N_A_cm3": absorber.N_A / 1.0e6,
                "absorber_N_D_cm3": absorber.N_D / 1.0e6,
                "absorber_tau_n_s": absorber.tau_n,
                "absorber_tau_p_s": absorber.tau_p,
                "absorber_n1_m3": absorber.n1,
                "absorber_p1_m3": absorber.p1,
                "absorber_trap_N_t_bulk_cm3": None
                if absorber.trap_N_t_bulk is None
                else absorber.trap_N_t_bulk / 1.0e6,
                "absorber_trap_N_t_interface_cm3": None
                if absorber.trap_N_t_interface is None
                else absorber.trap_N_t_interface / 1.0e6,
            }
        )
    if absorber is not None and etl is not None:
        out["etl_delta_ec_eV"] = absorber.chi - etl.chi
        out["etl_N_D_cm3"] = etl.N_D / 1.0e6
    if absorber is not None and htl is not None:
        out["htl_delta_ev_eV"] = absorber.chi + absorber.Eg - htl.chi - htl.Eg
        out["htl_N_A_cm3"] = htl.N_A / 1.0e6
    return out


def write_results_json(results: Mapping[str, Any], path: str | Path) -> None:
    """Write sweep results to JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


def write_results_csv(results: Mapping[str, Any], path: str | Path) -> None:
    """Write a flat CSV summary of sweep records."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "point_id",
        "axis",
        "label",
        "simulation_status",
        "Voc_fwd",
        "Jsc_fwd",
        "FF_fwd",
        "PCE_fwd",
        "Voc_rev",
        "Jsc_rev",
        "FF_rev",
        "PCE_rev",
        "hysteresis_index",
        "configured_V_bi",
        "computed_V_bi",
        "etl_delta_ec_eV",
        "htl_delta_ev_eV",
        "etl_N_D_cm3",
        "absorber_N_A_cm3",
        "absorber_N_D_cm3",
        "absorber_tau_n_s",
        "absorber_tau_p_s",
        "elapsed_s",
        "updates_json",
        "error_type",
        "error_message",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in results.get("records", []) or []:
            writer.writerow(_csv_row(record))


def write_summary_plots(
    results: Mapping[str, Any],
    out_dir: str | Path,
    *,
    metrics: Sequence[str] = ("PCE", "V_oc"),
) -> list[str]:
    """Write simple one-factor trend plots and return generated paths."""

    import matplotlib.pyplot as plt

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = [r for r in results.get("records", []) or [] if r.get("simulation_status") == "succeeded"]
    axes = sorted({r.get("axis") for r in records if r.get("axis") not in {None, "baseline"} and not str(r.get("axis")).startswith("coupled")})
    paths: list[str] = []
    for axis in axes:
        axis_records = [r for r in records if r.get("axis") == axis]
        x_values = [_axis_x_value(axis, r) for r in axis_records]
        if any(x is None for x in x_values):
            continue
        order = sorted(range(len(axis_records)), key=lambda i: float(x_values[i]))
        x_sorted = [float(x_values[i]) for i in order]
        for metric in metrics:
            y_sorted = [_metric_value(axis_records[i], metric) for i in order]
            if all(y is None for y in y_sorted):
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            ax.plot(x_sorted, y_sorted, marker="o", linewidth=1.8)
            if _is_log_axis(axis) and all(x > 0 for x in x_sorted):
                ax.set_xscale("log")
            ax.set_xlabel(_axis_label(axis))
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs {_axis_label(axis)}")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            path = root / f"{axis}_{_metric_slug(metric)}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))
    return paths


def _apply_etl_delta_ec(stack: DeviceStack, delta_ec_eV: float) -> DeviceStack:
    absorber = _require_params_by_role(stack, "absorber")
    etl = _require_params_by_role(stack, "ETL")
    return _replace_layer_params_by_role(stack, "ETL", chi=absorber.chi - delta_ec_eV, Eg=etl.Eg)


def _apply_htl_delta_ev(stack: DeviceStack, delta_ev_eV: float) -> DeviceStack:
    absorber = _require_params_by_role(stack, "absorber")
    htl = _require_params_by_role(stack, "HTL")
    chi_htl = absorber.chi + absorber.Eg - htl.Eg - delta_ev_eV
    return _replace_layer_params_by_role(stack, "HTL", chi=chi_htl, Eg=htl.Eg)


def _apply_absorber_defect_depth(
    stack: DeviceStack,
    trap_depth_eV: float,
    *,
    reference: str,
) -> DeviceStack:
    absorber = _require_params_by_role(stack, "absorber")
    n1, p1 = srh_n1_p1_from_trap_depth(
        absorber.ni,
        absorber.Eg,
        trap_depth_eV,
        reference=reference,
    )
    return _replace_layer_params_by_role(stack, "absorber", n1=n1, p1=p1)


def _apply_absorber_defect_density(
    stack: DeviceStack,
    density_m3: float,
    *,
    trap_tau_reference_density_m3: float,
) -> DeviceStack:
    absorber = _require_params_by_role(stack, "absorber")
    scale = _trap_tau_scale(density_m3, trap_tau_reference_density_m3)
    return _replace_layer_params_by_role(
        stack,
        "absorber",
        trap_N_t_bulk=density_m3,
        tau_n=absorber.tau_n * scale,
        tau_p=absorber.tau_p * scale,
    )


def _apply_absorber_interface_trap_density(
    stack: DeviceStack,
    density_m3: float,
    *,
    trap_tau_reference_density_m3: float,
    trap_decay_length_m: float,
) -> DeviceStack:
    absorber = _require_params_by_role(stack, "absorber")
    bulk_density = absorber.trap_N_t_bulk or trap_tau_reference_density_m3
    return _replace_layer_params_by_role(
        stack,
        "absorber",
        trap_N_t_interface=density_m3,
        trap_N_t_bulk=bulk_density,
        trap_decay_length=trap_decay_length_m,
        trap_profile_shape="exponential",
    )


def _apply_internal_interface_srv(
    stack: DeviceStack,
    srv_m_s: float,
    *,
    target: str = "all",
) -> DeviceStack:
    interfaces = [tuple(pair) for pair in stack.interfaces]
    required_len = max(0, len(stack.layers) - 1)
    while len(interfaces) < required_len:
        interfaces.append((0.0, 0.0))
    target_key = target.lower()
    for i in range(required_len):
        left = stack.layers[i]
        right = stack.layers[i + 1]
        if left.role == "substrate" or right.role == "substrate":
            interfaces[i] = (0.0, 0.0)
            continue
        pair_name = frozenset({left.role.lower(), right.role.lower()})
        should_set = (
            target_key == "all"
            or (target_key == "htl_absorber" and pair_name == frozenset({"htl", "absorber"}))
            or (target_key == "absorber_etl" and pair_name == frozenset({"absorber", "etl"}))
        )
        if should_set:
            interfaces[i] = (srv_m_s, srv_m_s)
    return dataclasses.replace(stack, interfaces=tuple(interfaces))


def _apply_interface_defect_N_t_cm2(
    stack: DeviceStack,
    N_t_cm2_areal: float,
    *,
    target: str = "pvk/etl",
) -> DeviceStack:
    """Set the SRV on ``DeviceStack.interfaces[k]`` from a SCAPS areal
    trap density ``N_t [cm^-2]``, preserving ``InterfaceDefect.E_t_eV``.

    Translates SCAPS kinetic identity using fixed σ=1e-15 cm² and
    v_th=1e7 cm/s — the standard SCAPS PVK/ETL Gaussian-defect cross-
    section and thermal velocity. SRV [m/s] = σ_cm2 · v_th_cm_s ·
    N_t_cm2 · 1e-2. The sweep handler does NOT touch
    ``MaterialParams.trap_N_t_interface`` (the Phase 4a layer-trap-
    profile knob driven by the separate ``interface_trap_density_cm3``
    sweep key).

    Raises ``ValueError`` if the named target has no existing
    ``InterfaceDefect`` entry — the YAML must declare the defect first
    (with its ``E_t_eV_below_cb``) before the sweep can drive its
    density. Otherwise sweeping a defect-free interface would silently
    invent a defect on it, masking config bugs.

    Target alias resolves the heterointerface by adjacent layer roles
    (case-insensitive, ``pvk`` ≡ ``absorber``): ``pvk/etl``,
    ``htl/pvk``, ``left`` (first interior interface), ``right`` (last
    interior interface).
    """
    SIGMA_CM2 = 1.0e-15
    V_TH_CM_S = 1.0e7
    srv_m_s = SIGMA_CM2 * V_TH_CM_S * float(N_t_cm2_areal) * 1.0e-2

    n_interfaces = max(0, len(stack.layers) - 1)
    if n_interfaces == 0:
        raise ValueError("DeviceStack has no interior interfaces")
    k = _resolve_interface_sweep_target(target, stack.layers, n_interfaces)

    defects = list(stack.interface_defects) if stack.interface_defects else [None] * n_interfaces
    if defects[k] is None:
        raise ValueError(
            f"interface_defect_N_t_cm2 sweep target {target!r} has no "
            "InterfaceDefect entry in DeviceStack.interface_defects — "
            "declare one in the YAML interfaces: block before sweeping"
        )
    interfaces = list(stack.interfaces) if stack.interfaces else [(0.0, 0.0)] * n_interfaces
    interfaces[k] = (srv_m_s, srv_m_s)
    return dataclasses.replace(
        stack,
        interfaces=tuple(interfaces),
        interface_defects=tuple(defects),
    )


def _apply_interface_defect_E_t_eV(
    stack: DeviceStack,
    E_t_eV: float,
    *,
    target: str = "pvk/etl",
) -> DeviceStack:
    """Set the trap depth ``E_t_eV`` on the targeted ``InterfaceDefect``.

    Mirrors ``_apply_interface_defect_N_t_cm2`` but drives the trap level
    rather than the density. The build path recomputes the interface SRH
    ``(n1, p1)`` from ``defect.E_t_eV`` (``solver/mol.py:build_material_arrays``
    via ``srh_n1_p1_from_trap_depth(reference="below_cb")``), so replacing
    the field is sufficient. Surface velocities on ``stack.interfaces`` are
    left untouched. Raises ``ValueError`` if the target has no declared
    ``InterfaceDefect`` — the YAML must declare the defect first.
    """
    n_interfaces = max(0, len(stack.layers) - 1)
    if n_interfaces == 0:
        raise ValueError("DeviceStack has no interior interfaces")
    k = _resolve_interface_sweep_target(target, stack.layers, n_interfaces)
    defects = list(stack.interface_defects) if stack.interface_defects else [None] * n_interfaces
    if defects[k] is None:
        raise ValueError(
            f"interface_defect_E_t_eV sweep target {target!r} has no "
            "InterfaceDefect entry in DeviceStack.interface_defects — "
            "declare one in the YAML interfaces: block before sweeping"
        )
    defects[k] = dataclasses.replace(defects[k], E_t_eV=float(E_t_eV))
    return dataclasses.replace(stack, interface_defects=tuple(defects))


# Adjacent-layer-role alias resolver for the interface-defect sweep
# handler. Mirrors the resolver in ``perovskite_sim.scaps_compat.loader``
# but kept local here so the sweeps package stays decoupled from the
# SCAPS compatibility module.
_INTERFACE_SWEEP_ROLE_ALIAS = {
    "pvk": "absorber",
    "perovskite": "absorber",
    "absorber": "absorber",
    "htl": "htl",
    "etl": "etl",
}


def _resolve_interface_sweep_target(
    target: str, layers: tuple[LayerSpec, ...], n_interfaces: int,
) -> int:
    """Resolve a heterointerface target alias to a stack.layers index.

    Accepts ``A/B`` role pairs (case-insensitive, ``pvk`` ≡ ``absorber``)
    or the positional aliases ``left`` (first interior interface) /
    ``right`` (last interior interface).
    """
    key = target.strip().lower()
    if key in ("left", "first"):
        return 0
    if key in ("right", "last"):
        return n_interfaces - 1
    parts = [p.strip() for p in key.split("/")]
    if len(parts) != 2 or any(not p for p in parts):
        raise ValueError(
            f"unknown interface target {target!r} "
            "(expected 'A/B' role pair or 'left'/'right')"
        )
    left_alias = _INTERFACE_SWEEP_ROLE_ALIAS.get(parts[0], parts[0])
    right_alias = _INTERFACE_SWEEP_ROLE_ALIAS.get(parts[1], parts[1])
    for k in range(n_interfaces):
        left_role = layers[k].role.lower()
        right_role = layers[k + 1].role.lower()
        if left_role == left_alias and right_role == right_alias:
            return k
    raise ValueError(
        f"unknown interface target {target!r} — no adjacent layer pair "
        f"matches roles ({left_alias!r}, {right_alias!r}) in the stack "
        + str([l.role for l in layers])
    )


def _replace_layer_params_by_role(stack: DeviceStack, role: str, **updates: Any) -> DeviceStack:
    role_key = role.lower()
    new_layers: list[LayerSpec] = []
    replaced = False
    for layer in stack.layers:
        if not replaced and layer.role.lower() == role_key:
            if layer.params is None:
                raise ValueError(f"layer {layer.name!r} has no MaterialParams")
            new_params = dataclasses.replace(layer.params, **updates)
            new_layers.append(dataclasses.replace(layer, params=new_params))
            replaced = True
        else:
            new_layers.append(layer)
    if not replaced:
        raise ValueError(f"no layer with role {role!r}")
    return dataclasses.replace(stack, layers=tuple(new_layers))


def _require_params_by_role(stack: DeviceStack, role: str) -> MaterialParams:
    params = _params_by_role(stack, role)
    if params is None:
        raise ValueError(f"no material params found for role {role!r}")
    return params


def _params_by_role(stack: DeviceStack, role: str) -> MaterialParams | None:
    role_key = role.lower()
    for layer in stack.layers:
        if layer.role.lower() == role_key:
            return layer.params
    return None


def _trap_tau_scale(density_m3: float, reference_density_m3: float) -> float:
    if density_m3 <= 0.0:
        raise ValueError(f"trap density must be positive, got {density_m3}")
    if reference_density_m3 <= 0.0:
        raise ValueError(f"reference trap density must be positive, got {reference_density_m3}")
    return reference_density_m3 / density_m3


def _summarise_records(records: Sequence[Mapping[str, Any]], elapsed_s: float) -> dict[str, Any]:
    succeeded = sum(1 for r in records if r.get("simulation_status") == "succeeded")
    failed = sum(1 for r in records if r.get("simulation_status") == "failed")
    return {
        "succeeded": succeeded,
        "failed": failed,
        "elapsed_s": elapsed_s,
    }


def _csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    derived = record.get("derived", {}) or {}
    fwd = record.get("metrics_fwd", {}) or {}
    rev = record.get("metrics_rev", {}) or {}
    return {
        "index": record.get("index"),
        "point_id": record.get("point_id"),
        "axis": record.get("axis"),
        "label": record.get("label"),
        "simulation_status": record.get("simulation_status"),
        "Voc_fwd": fwd.get("V_oc"),
        "Jsc_fwd": fwd.get("J_sc"),
        "FF_fwd": fwd.get("FF"),
        "PCE_fwd": fwd.get("PCE"),
        "Voc_rev": rev.get("V_oc"),
        "Jsc_rev": rev.get("J_sc"),
        "FF_rev": rev.get("FF"),
        "PCE_rev": rev.get("PCE"),
        "hysteresis_index": record.get("hysteresis_index"),
        "configured_V_bi": derived.get("configured_V_bi"),
        "computed_V_bi": derived.get("computed_V_bi"),
        "etl_delta_ec_eV": derived.get("etl_delta_ec_eV"),
        "htl_delta_ev_eV": derived.get("htl_delta_ev_eV"),
        "etl_N_D_cm3": derived.get("etl_N_D_cm3"),
        "absorber_N_A_cm3": derived.get("absorber_N_A_cm3"),
        "absorber_N_D_cm3": derived.get("absorber_N_D_cm3"),
        "absorber_tau_n_s": derived.get("absorber_tau_n_s"),
        "absorber_tau_p_s": derived.get("absorber_tau_p_s"),
        "elapsed_s": record.get("elapsed_s"),
        "updates_json": json.dumps(record.get("updates", {}), sort_keys=True),
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
    }


def _axis_x_value(axis: str, record: Mapping[str, Any]) -> float | None:
    updates = record.get("updates", {}) or {}
    if axis == "etl_delta_ec":
        return _optional_float(updates.get("etl_delta_ec_eV"))
    if axis == "htl_delta_ev":
        return _optional_float(updates.get("htl_delta_ev_eV"))
    if axis == "etl_doping":
        return _optional_float(updates.get("etl_doping_cm3"))
    if axis == "absorber_doping":
        return _optional_float(updates.get("absorber_doping_cm3"))
    if axis == "absorber_defect_depth":
        return _optional_float(updates.get("absorber_defect_depth_eV"))
    if axis == "absorber_defect_density":
        return _optional_float(updates.get("absorber_defect_density_cm3"))
    if axis == "interface_trap_density":
        return _optional_float(updates.get("interface_trap_density_cm3"))
    if axis == "interface_srv":
        return _optional_float(updates.get("interface_srv_cm_s"))
    return None


def _metric_value(record: Mapping[str, Any], metric: str) -> float | None:
    fwd = record.get("metrics_fwd", {}) or {}
    key = {"V_oc": "V_oc", "J_sc": "J_sc", "FF": "FF", "PCE": "PCE"}.get(metric)
    if key is None:
        return None
    value = fwd.get(key)
    return None if value is None else float(value)


def _axis_label(axis: str) -> str:
    return {
        "etl_delta_ec": "ETL Delta Ec (eV)",
        "htl_delta_ev": "HTL Delta Ev (eV)",
        "etl_doping": "ETL donor density (cm^-3)",
        "absorber_doping": "Absorber acceptor density (cm^-3)",
        "absorber_defect_depth": "Absorber defect depth (eV)",
        "absorber_defect_density": "Absorber defect density (cm^-3)",
        "interface_trap_density": "Interface trap density (cm^-3)",
        "interface_srv": "Interface SRV (cm/s)",
    }.get(axis, axis)


def _is_log_axis(axis: str) -> bool:
    return any(token in axis for token in ("doping", "density", "srv"))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _decades(start_exp: int, end_exp: int) -> list[float]:
    return [10.0 ** exp for exp in range(start_exp, end_exp + 1)]


def _float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    count = int(round((stop - start) / step))
    return [round(start + i * step, 12) for i in range(count + 1)]


def _slug(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _sci_slug(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def _metric_slug(metric: str) -> str:
    return metric.lower().replace("_", "")


def _resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.is_absolute():
        return candidate

    search_roots = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent]
    for root in list(search_roots):
        search_roots.extend(root.parents)
    seen: set[Path] = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return candidate

"""Executable device studies for the Phase-1 RHS regularization contract.

The generic certificate schema cannot establish that a transition width is
actually wired through a device solve.  This module provides three fixed,
small-grid studies that exercise the Poole-Frenkel, thermionic-cap, and
interface-state density paths with the same four-rung width ladder.  Each
study uses a real Poisson/drift-diffusion/Radau trajectory and reports the
policy returned by the solve, rather than echoing the requested policy.

These studies certify numerical width sensitivity only.  They do not validate
the underlying constitutive models or material parameters.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import time
from typing import Callable, Iterator, Literal

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import (
    _compute_current,
    _compute_current_ss_with_spread,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.physics.regularization import RHSRegularization
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import (
    StateVec,
    assemble_rhs,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.validation.regularization_certificate import (
    AppliedRunContext,
    MetricSpec,
    ObservableSpec,
    QualityGateSpec,
    RegularizationCertificate,
    RegularizationMeasurement,
    RegularizationRungRequest,
    RegularizationStudy,
    run_regularization_ladder,
)


DeviceRegularizationStudyId = Literal[
    "poole-frenkel-device",
    "thermionic-cap-device",
    "interface-density-device",
]

DEVICE_REGULARIZATION_STUDY_IDS: tuple[DeviceRegularizationStudyId, ...] = (
    "poole-frenkel-device",
    "thermionic-cap-device",
    "interface-density-device",
)

_INTERFACE_STATE_ENV = "SOLARLAB_INTERFACE_PLANE_STATE"
_SOURCE_FILES = (
    "perovskite_sim/constants.py",
    "perovskite_sim/discretization/grid.py",
    "perovskite_sim/experiments/jv_sweep.py",
    "perovskite_sim/models/config_loader.py",
    "perovskite_sim/models/device.py",
    "perovskite_sim/physics/continuity.py",
    "perovskite_sim/physics/field_mobility.py",
    "perovskite_sim/physics/generation.py",
    "perovskite_sim/physics/interface_plane.py",
    "perovskite_sim/physics/ion_migration.py",
    "perovskite_sim/physics/poisson.py",
    "perovskite_sim/physics/recombination.py",
    "perovskite_sim/physics/regularization.py",
    "perovskite_sim/scaps_compat/loader.py",
    "perovskite_sim/solver/mol.py",
    "perovskite_sim/solver/newton.py",
    "perovskite_sim/solver/numerical_diagnostics.py",
    "perovskite_sim/validation/regularization_certificate.py",
    "perovskite_sim/validation/regularization_executors.py",
)


@dataclass(frozen=True)
class PreparedDeviceRegularizationStudy:
    """A fixed study and its real-device evaluator."""

    study_id: DeviceRegularizationStudyId
    study: RegularizationStudy
    evaluator: Callable[[RegularizationRungRequest], RegularizationMeasurement]


@dataclass(frozen=True)
class _TransientProtocol:
    illuminated: bool
    voltage_V: float
    duration_s: float
    rtol: float
    atol_density_m3: float


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest(project_root: Path) -> dict[str, str]:
    return {
        relative: _file_sha256(project_root / relative) for relative in _SOURCE_FILES
    }


@contextmanager
def _temporary_environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _device_grid(stack: DeviceStack) -> np.ndarray:
    return multilayer_grid(
        [Layer(layer.thickness, 4) for layer in electrical_layers(stack)]
    )


def _with_poole_frenkel(stack: DeviceStack) -> DeviceStack:
    layers = tuple(
        replace(
            layer,
            params=replace(
                layer.params,
                pf_gamma_n=2.0e-3,
                pf_gamma_p=2.0e-3,
            ),
        )
        if layer.role == "absorber"
        else layer
        for layer in stack.layers
    )
    return replace(stack, layers=layers, mode="full")


def _state_minima(
    report, state_blocks: tuple[str, ...], *, final: bool
) -> dict[str, float]:
    source = (
        report.final_minimum_density_m3 if final else report.minimum_trial_density_m3
    )
    if source is None:
        raise RuntimeError("regularization solve did not report terminal minima")
    attributes = {
        "n": "n",
        "p": "p",
        "P": "positive_ion_active",
        "P_neg": "negative_ion_active",
        "interface_state": "interface_state",
    }
    result: dict[str, float] = {}
    for block in state_blocks:
        value = getattr(source, attributes[block])
        if value is None or not np.isfinite(value):
            raise RuntimeError(f"state block {block!r} has no finite minimum")
        result[block] = float(value)
    return result


def _terminal_rhs_rate(
    terminal: np.ndarray,
    rhs: np.ndarray,
    *,
    n_nodes: int,
    n_interface_states: int,
    state_blocks: tuple[str, ...],
) -> float:
    state = StateVec.unpack(terminal, n_nodes, n_interface_states)
    derivative = StateVec.unpack(rhs, n_nodes, n_interface_states)
    values = {
        "n": (state.n, derivative.n),
        "p": (state.p, derivative.p),
        "P": (state.P, derivative.P),
        "P_neg": (state.P_neg, derivative.P_neg),
        "interface_state": (state.iface_state, derivative.iface_state),
    }
    rates: list[float] = []
    for block in state_blocks:
        block_state, block_rhs = values[block]
        if block_state is None or block_rhs is None:
            raise RuntimeError(f"state block {block!r} is absent")
        scale = max(float(np.max(np.abs(block_state))), np.finfo(float).tiny)
        rates.append(float(np.max(np.abs(block_rhs))) / scale)
    return max(rates)


def _prepare_stack(
    study_id: DeviceRegularizationStudyId,
    project_root: Path,
) -> tuple[DeviceStack, str, dict[str, object]]:
    if study_id == "poole-frenkel-device":
        relative = "configs/nip_MAPbI3.yaml"
        stack = _with_poole_frenkel(load_device_from_yaml(project_root / relative))
        overrides: dict[str, object] = {
            "absorber_pf_gamma_n_sqrt_m_V": 2.0e-3,
            "absorber_pf_gamma_p_sqrt_m_V": 2.0e-3,
            "mode": "full",
        }
    elif study_id in {"thermionic-cap-device", "interface-density-device"}:
        relative = "configs/scaps_mirror_v2.yaml"
        stack = load_scaps_yaml(project_root / relative)
        overrides = {}
        if study_id == "thermionic-cap-device":
            stack = replace(stack, te_physical_norm=True)
            overrides["te_physical_norm"] = True
        else:
            overrides["interface_plane_state"] = True
    else:
        raise KeyError(f"unknown regularization study {study_id!r}")
    return stack, relative, overrides


def _study_parameters(
    study_id: DeviceRegularizationStudyId,
) -> tuple[
    RHSRegularization,
    _TransientProtocol,
    tuple[str, ...],
    tuple[ObservableSpec, ...],
    tuple[MetricSpec, ...],
    tuple[MetricSpec, ...],
    tuple[QualityGateSpec, ...],
]:
    common_residuals = (MetricSpec("endpoint_time_error_s", "s", 1.0e-30),)
    common_health = (
        QualityGateSpec("endpoint_reached", "1", "eq", 1.0),
        QualityGateSpec("strict_zero_floor_pass", "1", "eq", 1.0),
    )
    if study_id == "poole-frenkel-device":
        return (
            RHSRegularization(poole_frenkel_field_width_V_m=2.0e6),
            _TransientProtocol(True, 0.0, 1.0e-8, 1.0e-5, 1.0e-3),
            ("n", "p", "P"),
            (ObservableSpec("terminal_current_A_m2", "A m^-2", 1.0),),
            common_residuals
            + (
                MetricSpec(
                    "terminal_rhs_rate_per_s",
                    "s^-1",
                    6.0e6,
                    non_worsening_atol=2.5e3,
                ),
            ),
            (
                MetricSpec(
                    "positive_ion_inventory_relative_drift",
                    "1",
                    1.0e-10,
                    non_worsening_atol=1.0e-14,
                ),
            ),
            common_health
            + (
                QualityGateSpec(
                    "positive_ion_inventory_relative_drift",
                    "1",
                    "le",
                    1.0e-10,
                ),
            ),
        )
    if study_id == "thermionic-cap-device":
        return (
            RHSRegularization(te_cap_relative_width=0.5),
            _TransientProtocol(True, 0.8, 1.0e-8, 1.0e-7, 1.0e-3),
            ("n", "p"),
            (ObservableSpec("terminal_current_A_m2", "A m^-2", 1.0),),
            common_residuals
            + (
                MetricSpec(
                    "terminal_rhs_rate_per_s",
                    "s^-1",
                    1.3e7,
                    non_worsening_atol=50.0,
                ),
            ),
            (
                MetricSpec(
                    "interior_current_spread_A_m2",
                    "A m^-2",
                    2.0e4,
                    non_worsening_atol=0.1,
                ),
            ),
            common_health,
        )
    if study_id == "interface-density-device":
        return (
            RHSRegularization(interface_density_width_m3=1.0e-4),
            _TransientProtocol(False, 0.0, 1.0e-12, 1.0e-7, 1.0e-12),
            ("n", "p", "interface_state"),
            (
                ObservableSpec("interface_state_m3", "m^-3", 1.0e-30),
                ObservableSpec("terminal_current_A_m2", "A m^-2", 1.0),
            ),
            common_residuals
            + (
                MetricSpec(
                    "terminal_rhs_rate_per_s",
                    "s^-1",
                    4.5e9,
                    non_worsening_atol=1.0,
                ),
            ),
            (
                MetricSpec(
                    "interior_current_spread_A_m2",
                    "A m^-2",
                    5.0e6,
                    non_worsening_atol=1.0e-6,
                ),
            ),
            common_health,
        )
    raise KeyError(f"unknown regularization study {study_id!r}")


def prepare_device_regularization_study(
    study_id: DeviceRegularizationStudyId,
    *,
    project_root: Path | str,
) -> PreparedDeviceRegularizationStudy:
    """Build one frozen real-device ladder and evaluator."""

    root = Path(project_root).resolve()
    if study_id not in DEVICE_REGULARIZATION_STUDY_IDS:
        raise KeyError(f"unknown regularization study {study_id!r}")
    (
        base_policy,
        protocol,
        state_blocks,
        observables,
        residuals,
        conservation_errors,
        health_gates,
    ) = _study_parameters(study_id)
    stack, config_relative, overrides = _prepare_stack(study_id, root)
    if study_id == "interface-density-device":
        with _temporary_environment(_INTERFACE_STATE_ENV, "1"):
            x = _device_grid(stack)
            material = build_material_arrays(x, stack)
            initial_state = solve_equilibrium(x, stack)
    else:
        x = _device_grid(stack)
        material = build_material_arrays(x, stack)
        initial_state = solve_equilibrium(x, stack)

    if study_id == "poole-frenkel-device" and not material.has_field_mobility:
        raise RuntimeError("Poole-Frenkel device study did not activate field mobility")
    if study_id == "thermionic-cap-device" and not material.te_physical_norm:
        raise RuntimeError("thermionic device study did not activate physical TE")
    if study_id == "interface-density-device" and material.N_iface_state <= 0:
        raise RuntimeError("interface device study did not activate state blocks")

    config_path = root / config_relative
    study = RegularizationStudy.from_values(
        base_policy=base_policy,
        protocol={
            "atol_density_m3": protocol.atol_density_m3,
            "duration_s": protocol.duration_s,
            "illuminated": protocol.illuminated,
            "initial_state": "dark_equilibrium",
            "environment_overrides": (
                {_INTERFACE_STATE_ENV: "1"}
                if study_id == "interface-density-device"
                else {}
            ),
            "rtol": protocol.rtol,
            "schema": "device-regularization-transient-v1",
            "voltage_V": protocol.voltage_V,
        },
        config={
            "config_path": config_relative,
            "config_sha256": _file_sha256(config_path),
            "overrides": overrides,
            "source_sha256": _source_manifest(root),
        },
        grid={
            "construction": "four-intervals-per-electrical-layer",
            "node_count": len(x),
            "x_m": x.tolist(),
        },
        tolerances={
            "atol_density_m3": protocol.atol_density_m3,
            "method": "Radau",
            "rtol": protocol.rtol,
        },
        observables=observables,
        residuals=residuals,
        conservation_errors=conservation_errors,
        physical_health_gates=health_gates,
        state_blocks=state_blocks,
    )

    n_nodes = len(x)
    initial_vector = np.asarray(initial_state, dtype=float).copy()
    initial_sv = StateVec.unpack(initial_vector, n_nodes, material.N_iface_state)
    initial_inventory = (
        dual_cell_integral(x, initial_sv.P) if "P" in state_blocks else None
    )

    def evaluate(request: RegularizationRungRequest) -> RegularizationMeasurement:
        if request.study.definition_sha256 != study.definition_sha256:
            raise RuntimeError("regularization request changed the frozen study")
        started = time.perf_counter()
        solution = run_transient(
            x,
            initial_vector.copy(),
            (0.0, protocol.duration_s),
            np.array([protocol.duration_s]),
            stack,
            illuminated=protocol.illuminated,
            V_app=protocol.voltage_V,
            rtol=protocol.rtol,
            atol=protocol.atol_density_m3,
            mat=material,
            regularization=request.policy,
        )
        wall_time = time.perf_counter() - started
        values = np.asarray(getattr(solution, "y", np.empty((0, 0))))
        if values.ndim != 2 or values.shape[1] != 1:
            raise RuntimeError("regularization solve returned no terminal state")
        terminal = values[:, -1]
        applied_policy = getattr(solution, "rhs_regularization", None)
        if not isinstance(applied_policy, RHSRegularization):
            raise RuntimeError("regularization solve did not report applied policy")
        rhs = assemble_rhs(
            protocol.duration_s,
            terminal,
            x,
            stack,
            material,
            protocol.illuminated,
            protocol.voltage_V,
            regularization=applied_policy,
        )
        report = solution.numerical_diagnostics
        endpoint_error = abs(float(solution.t[-1]) - protocol.duration_s)
        rhs_rate = _terminal_rhs_rate(
            terminal,
            rhs,
            n_nodes=n_nodes,
            n_interface_states=material.N_iface_state,
            state_blocks=state_blocks,
        )
        terminal_current = _compute_current(
            x,
            terminal,
            stack,
            protocol.voltage_V,
            y_prev=initial_vector,
            dt=protocol.duration_s,
            mat=material,
            V_app_prev=protocol.voltage_V,
        )
        _, current_spread = _compute_current_ss_with_spread(
            x,
            terminal,
            stack,
            protocol.voltage_V,
            mat=material,
        )
        observable_values: dict[str, object] = {
            "terminal_current_A_m2": terminal_current,
        }
        if "interface_state" in state_blocks:
            interface_state = StateVec.unpack(
                terminal, n_nodes, material.N_iface_state
            ).iface_state
            if interface_state is None:
                raise RuntimeError("interface state observable is absent")
            observable_values["interface_state_m3"] = interface_state

        if initial_inventory is not None:
            terminal_inventory = dual_cell_integral(
                x,
                StateVec.unpack(terminal, n_nodes, material.N_iface_state).P,
            )
            inventory_drift = abs(terminal_inventory - initial_inventory) / max(
                abs(initial_inventory), np.finfo(float).tiny
            )
            conservation_values = {
                "positive_ion_inventory_relative_drift": inventory_drift,
            }
        else:
            inventory_drift = None
            conservation_values = {
                "interior_current_spread_A_m2": current_spread,
            }

        physical_health = {
            "endpoint_reached": float(endpoint_error == 0.0),
            "strict_zero_floor_pass": float(report.would_pass_strict),
        }
        if inventory_drift is not None:
            physical_health["positive_ion_inventory_relative_drift"] = inventory_drift

        return RegularizationMeasurement.from_values(
            study,
            applied=AppliedRunContext(
                policy=applied_policy,
                protocol_sha256=study.protocol.sha256,
                config_sha256=study.config.sha256,
                grid_sha256=study.grid.sha256,
                tolerances_sha256=study.tolerances.sha256,
            ),
            observables=observable_values,
            residuals={
                "endpoint_time_error_s": endpoint_error,
                "terminal_rhs_rate_per_s": rhs_rate,
            },
            conservation_errors=conservation_values,
            minimum_trial_state_m3=_state_minima(report, state_blocks, final=False),
            terminal_minimum_state_m3=_state_minima(report, state_blocks, final=True),
            physical_health=physical_health,
            solver_accepted=bool(solution.success and endpoint_error == 0.0),
            negative_trial_count=report.negative_trial_evaluations,
            nonfinite_event_count=(
                report.nonfinite_trial_evaluations + report.nonfinite_rhs_evaluations
            ),
            nfev=int(solution.nfev),
            njev=int(solution.njev),
            nlu=int(solution.nlu),
            wall_time_s=wall_time,
        )

    return PreparedDeviceRegularizationStudy(study_id, study, evaluate)


def run_device_regularization_study(
    study_id: DeviceRegularizationStudyId,
    *,
    project_root: Path | str,
) -> RegularizationCertificate:
    """Execute and certify all four rungs of one real-device study."""

    prepared = prepare_device_regularization_study(study_id, project_root=project_root)
    return run_regularization_ladder(prepared.study, prepared.evaluator)


__all__ = [
    "DEVICE_REGULARIZATION_STUDY_IDS",
    "DeviceRegularizationStudyId",
    "PreparedDeviceRegularizationStudy",
    "prepare_device_regularization_study",
    "run_device_regularization_study",
]

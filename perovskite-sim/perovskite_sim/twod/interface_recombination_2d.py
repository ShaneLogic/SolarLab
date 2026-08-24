from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from perovskite_sim.models.device import (
    DeviceStack,
    electrical_interface_defects,
    electrical_interfaces,
)
from perovskite_sim.physics.recombination import interface_recombination


_REQUIRED_INTERFACE_METADATA = (
    "interface_eval_node_n",
    "interface_eval_node_p",
    "interface_calibration_factor",
    "interface_n1",
    "interface_p1",
    "interface_ni_sq_eff",
    "interface_n_L_eq",
    "interface_p_R_eq",
)


@dataclass(frozen=True, slots=True)
class TwoSidedInterfaceSRHCoupling2D:
    """One validated horizontal two-sided interface-SRH sheet."""

    interface_index: int
    interface_row: int
    left_sample_row: int
    right_sample_row: int
    electron_capture_velocity_m_s: float
    hole_capture_velocity_m_s: float
    n1_m3: float
    p1_m3: float
    pair_a_equilibrium_product_m6: float
    pair_b_equilibrium_product_m6: float


@dataclass(frozen=True, slots=True)
class TwoSidedInterfaceSRHReport2D:
    """Resolved surface rates and their area-conservative volume sink."""

    interface_rows: tuple[int, ...]
    pair_a_surface_rate_m2_s: np.ndarray
    pair_b_surface_rate_m2_s: np.ndarray
    total_surface_rate_m2_s: np.ndarray
    pair_a_clamped: np.ndarray
    pair_b_clamped: np.ndarray
    volumetric_sink_m3_s: np.ndarray


def _finite_nonnegative(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _finite_positive(name: str, value: float) -> float:
    number = _finite_nonnegative(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _unsupported_interface_modes(stack: DeviceStack, material_1d) -> tuple[str, ...]:
    unsupported: list[str] = []
    if material_1d.iface_plane_projection:
        unsupported.append("interface-plane projection")
    if material_1d.iface_shared_occ:
        unsupported.append("shared occupancy")
    if material_1d.iface_plane_closure:
        unsupported.append("interface-plane closure")
    if material_1d.iface_plane_generation:
        unsupported.append("interface-plane generation")
    if float(material_1d.het_recomb_despike) != 0.0:
        unsupported.append("heterojunction de-spike")
    if int(material_1d.N_iface_state) != 0:
        unsupported.append("dynamic interface-plane state")
    if getattr(stack, "interface_charge_closure", "off") != "off":
        unsupported.append("interface charge")
    if os.environ.get("SOLARLAB_IFACE_QSS") == "1":
        unsupported.append("legacy QSS environment path")
    if os.environ.get("SOLARLAB_IFACE_ALLOW_GEN") == "1":
        unsupported.append("interface-generation escape hatch")
    return tuple(unsupported)


def build_two_sided_interface_srh_couplings_2d(
    stack: DeviceStack,
    material_1d,
) -> tuple[TwoSidedInterfaceSRHCoupling2D, ...]:
    """Build the clamp-passive cross-node slice from the 1D material cache."""
    if not getattr(stack, "interface_two_sided", False):
        raise ValueError(
            "2D two-sided interface SRH requires explicit "
            "DeviceStack.interface_two_sided=True"
        )
    if not material_1d.iface_two_sided:
        raise ValueError("the effective 1D material did not enable two-sided SRH")
    unsupported = _unsupported_interface_modes(stack, material_1d)
    if unsupported:
        raise ValueError(
            "2D two-sided interface SRH does not support: "
            + ", ".join(unsupported)
        )

    interfaces = electrical_interfaces(stack)
    defects = electrical_interface_defects(stack)
    nodes = tuple(material_1d.interface_nodes)
    if not interfaces or len(interfaces) != len(nodes):
        raise ValueError(
            "2D two-sided interface SRH requires one interface velocity pair "
            "per electrical interface"
        )
    for name in _REQUIRED_INTERFACE_METADATA:
        values = tuple(getattr(material_1d, name, ()))
        if len(values) != len(nodes):
            raise ValueError(
                f"2D two-sided interface SRH requires {name} metadata "
                "for every electrical interface"
            )

    couplings: list[TwoSidedInterfaceSRHCoupling2D] = []
    state_size = np.asarray(material_1d.ni_sq).size
    for index, ((v_n_raw, v_p_raw), row) in enumerate(zip(interfaces, nodes)):
        v_n_base = _finite_nonnegative("electron capture velocity", v_n_raw)
        v_p_base = _finite_nonnegative("hole capture velocity", v_p_raw)
        if v_n_base == 0.0 or v_p_base == 0.0:
            continue
        defect = defects[index] if index < len(defects) else None
        if defect is None:
            raise ValueError(
                "each active 2D two-sided interface requires an InterfaceDefect"
            )
        right_row = int(material_1d.interface_eval_node_n[index])
        left_row = int(material_1d.interface_eval_node_p[index])
        if (
            row <= 0
            or row >= state_size - 1
            or left_row != row - 1
            or right_row != row + 1
        ):
            raise ValueError(
                "2D two-sided interface SRH requires interior idx-1/idx+1 "
                "cross-node sampling"
            )

        calibration = _finite_positive(
            "interface calibration factor",
            material_1d.interface_calibration_factor[index],
        )
        v_n = _finite_positive(
            "effective electron capture velocity",
            v_n_base * calibration,
        )
        v_p = _finite_positive(
            "effective hole capture velocity",
            v_p_base * calibration,
        )
        n1 = _finite_nonnegative("interface n1", material_1d.interface_n1[index])
        p1 = _finite_nonnegative("interface p1", material_1d.interface_p1[index])
        pair_a_reference = _finite_nonnegative(
            "pair-A equilibrium product",
            material_1d.interface_ni_sq_eff[index],
        )
        pair_b_reference = _finite_nonnegative(
            "pair-B equilibrium product",
            material_1d.interface_n_L_eq[index]
            * material_1d.interface_p_R_eq[index],
        )
        couplings.append(
            TwoSidedInterfaceSRHCoupling2D(
                interface_index=index,
                interface_row=int(row),
                left_sample_row=left_row,
                right_sample_row=right_row,
                electron_capture_velocity_m_s=v_n,
                hole_capture_velocity_m_s=v_p,
                n1_m3=n1,
                p1_m3=p1,
                pair_a_equilibrium_product_m6=pair_a_reference,
                pair_b_equilibrium_product_m6=pair_b_reference,
            )
        )
    if not couplings:
        raise ValueError("2D two-sided interface SRH has no active capture sheet")
    return tuple(couplings)


def _normal_control_volume_widths(y: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(y, dtype=float)
    if (
        coordinates.ndim != 1
        or coordinates.size < 3
        or not np.all(np.isfinite(coordinates))
        or np.any(np.diff(coordinates) <= 0.0)
    ):
        raise ValueError("y must be finite, 1-D, and strictly increasing")
    spacing = np.diff(coordinates)
    widths = np.empty_like(coordinates)
    widths[0] = 0.5 * spacing[0]
    widths[-1] = 0.5 * spacing[-1]
    widths[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return widths


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def evaluate_two_sided_interface_srh_2d(
    n: np.ndarray,
    p: np.ndarray,
    y: np.ndarray,
    couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...],
) -> TwoSidedInterfaceSRHReport2D:
    """Evaluate two capture directions and construct a sheet-conservative sink."""
    electrons = np.asarray(n, dtype=float)
    holes = np.asarray(p, dtype=float)
    if (
        electrons.ndim != 2
        or electrons.shape != holes.shape
        or electrons.shape[0] != np.asarray(y).size
        or not np.all(np.isfinite(electrons))
        or not np.all(np.isfinite(holes))
    ):
        raise ValueError("2D interface-SRH carrier arrays must be finite and shape matched")
    if not couplings:
        raise ValueError("at least one two-sided interface-SRH coupling is required")
    interface_rows = tuple(coupling.interface_row for coupling in couplings)
    interface_indices = tuple(coupling.interface_index for coupling in couplings)
    if len(set(interface_rows)) != len(interface_rows):
        raise ValueError("2D interface-SRH coupling rows must be unique")
    if len(set(interface_indices)) != len(interface_indices):
        raise ValueError("2D interface-SRH interface indices must be unique")

    normal_widths = _normal_control_volume_widths(y)
    pair_a = np.empty((len(couplings), electrons.shape[1]), dtype=float)
    pair_b = np.empty_like(pair_a)
    pair_a_clamped = np.empty(pair_a.shape, dtype=bool)
    pair_b_clamped = np.empty(pair_a.shape, dtype=bool)
    sink = np.zeros_like(electrons)

    for output_row, coupling in enumerate(couplings):
        row = coupling.interface_row
        if (
            row <= 0
            or row >= electrons.shape[0] - 1
            or coupling.left_sample_row != row - 1
            or coupling.right_sample_row != row + 1
        ):
            raise ValueError(
                "interface-SRH sheet must use interior idx-1/idx+1 sampling"
            )
        v_n = _finite_positive(
            "effective electron capture velocity",
            coupling.electron_capture_velocity_m_s,
        )
        v_p = _finite_positive(
            "effective hole capture velocity",
            coupling.hole_capture_velocity_m_s,
        )
        n1 = _finite_nonnegative("interface n1", coupling.n1_m3)
        p1 = _finite_nonnegative("interface p1", coupling.p1_m3)
        pair_a_reference = _finite_nonnegative(
            "pair-A equilibrium product",
            coupling.pair_a_equilibrium_product_m6,
        )
        pair_b_reference = _finite_nonnegative(
            "pair-B equilibrium product",
            coupling.pair_b_equilibrium_product_m6,
        )
        raw_a = np.asarray(
            interface_recombination(
                electrons[coupling.right_sample_row, :],
                holes[coupling.left_sample_row, :],
                pair_a_reference,
                n1,
                p1,
                v_n,
                v_p,
            ),
            dtype=float,
        )
        raw_b = np.asarray(
            interface_recombination(
                np.maximum(electrons[coupling.left_sample_row, :], 0.0),
                np.maximum(holes[coupling.right_sample_row, :], 0.0),
                pair_b_reference,
                n1,
                p1,
                v_n,
                v_p,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(raw_a)) or not np.all(np.isfinite(raw_b)):
            raise ValueError("2D interface SRH produced a non-finite surface rate")
        if raw_a.shape != (electrons.shape[1],) or raw_b.shape != raw_a.shape:
            raise ValueError("2D interface SRH produced a malformed surface-rate row")
        pair_a_clamped[output_row, :] = raw_a <= 0.0
        pair_b_clamped[output_row, :] = raw_b <= 0.0
        pair_a[output_row, :] = np.maximum(raw_a, 0.0)
        pair_b[output_row, :] = np.maximum(raw_b, 0.0)
        sink[row, :] += (
            pair_a[output_row, :] + pair_b[output_row, :]
        ) / normal_widths[row]

    total = pair_a + pair_b
    return TwoSidedInterfaceSRHReport2D(
        interface_rows=interface_rows,
        pair_a_surface_rate_m2_s=_read_only(pair_a),
        pair_b_surface_rate_m2_s=_read_only(pair_b),
        total_surface_rate_m2_s=_read_only(total),
        pair_a_clamped=_read_only(pair_a_clamped),
        pair_b_clamped=_read_only(pair_b_clamped),
        volumetric_sink_m3_s=_read_only(sink),
    )


__all__ = [
    "TwoSidedInterfaceSRHCoupling2D",
    "TwoSidedInterfaceSRHReport2D",
    "build_two_sided_interface_srh_couplings_2d",
    "evaluate_two_sided_interface_srh_2d",
]

"""Bound 1D iodine reaction-diffusion-electrostatics device and state contracts.

Separates immutable compiled data (:class:`CompiledIodineDevice1D` and
:class:`IodineState1D`) from device-input resolution (:mod:`solarlab.device`)
and nonlinear/time solvers (:mod:`solarlab.solvers`) in strict compliance with
SolarLab's ``import-linter`` layer hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import ContractError, Vector, frozen_array, integer_indices
from solarlab.equations.dc_1d import CompiledDevice1D, State1D
from solarlab.mesh.geometry import ControlVolumeGeometry1D
from solarlab.physics.chemistry import (
    IodineKineticsParameters,
    SpeciesCoordinateMap,
)
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger

__all__ = [
    "ContactReactionMode",
    "CompiledIodineDevice1D",
    "IODINE_BLOCKS",
    "IodineBoundaryPortSpec",
    "IodineDomainMode",
    "IodineState1D",
]

IODINE_BLOCKS: tuple[str, str, str] = ("iodide", "iodine_atom", "iodine_molecule")

IodineDomainMode = Literal["absorber_blocking", "full_stack_permeable"]
ContactReactionMode = Literal[
    "none",
    "hole_oxidation_to_I0",
    "immobilized_product_sheet",
]

_DOMAIN_MODES: frozenset[str] = frozenset(get_args(IodineDomainMode))
_CONTACT_MODES: frozenset[str] = frozenset(get_args(ContactReactionMode))


@dataclass(frozen=True, slots=True)
class IodineBoundaryPortSpec:
    """Boundary port closure for outer ``left`` or ``right`` chemical boundary."""

    side: Literal["left", "right"]
    chem_node: int
    elec_node: int
    outward_normal: int
    mol_escape_velocity_m_s: float = 0.0
    contact_reaction_mode: ContactReactionMode = "none"
    contact_oxidation_rate_m4_s: float = 0.0
    p1_iodine_m3: float = 0.0
    reversible_contact: bool = False

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ContractError("boundary_side_must_be_left_or_right")
        if self.outward_normal not in (-1, 1):
            raise ContractError("boundary_outward_normal_must_be_unit_sign")
        if self.contact_reaction_mode not in _CONTACT_MODES:
            raise ContractError("unknown_contact_reaction_mode")
        for label, val in (
            ("mol_escape_velocity_m_s", self.mol_escape_velocity_m_s),
            ("contact_oxidation_rate_m4_s", self.contact_oxidation_rate_m4_s),
            ("p1_iodine_m3", self.p1_iodine_m3),
        ):
            fval = float(val)
            if not math.isfinite(fval) or fval < 0.0:
                raise ContractError(f"{label}_must_be_finite_nonnegative")
            object.__setattr__(self, label, fval)
        if (
            self.contact_reaction_mode == "none"
            and self.contact_oxidation_rate_m4_s > 0.0
        ):
            raise ContractError(
                "positive_contact_oxidation_rate_requires_conserving_reaction_mode"
            )

    @property
    def removes_device_iodine(self) -> bool:
        """True if this boundary allows net iodine loss from mobile inventory."""
        return self.mol_escape_velocity_m_s > 0.0 or (
            self.contact_reaction_mode == "immobilized_product_sheet"
            and self.contact_oxidation_rate_m4_s > 0.0
        )


@dataclass(frozen=True, slots=True)
class IodineState1D:
    """Physical trial/accepted state of a 1D iodine-chemistry device."""

    u: Vector
    electronic_state: State1D
    iodide_density_m3: Vector
    iodine_atom_density_m3: Vector
    iodine_molecule_density_m3: Vector
    total_iodine_inventory_m2: float
    species_inventories_m2: dict[str, float]
    device_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "u", frozen_array(self.u))
        object.__setattr__(
            self, "iodide_density_m3", frozen_array(self.iodide_density_m3)
        )
        object.__setattr__(
            self, "iodine_atom_density_m3", frozen_array(self.iodine_atom_density_m3)
        )
        object.__setattr__(
            self,
            "iodine_molecule_density_m3",
            frozen_array(self.iodine_molecule_density_m3),
        )
        n_elec = self.electronic_state.potential_V.size
        m_chem = self.iodide_density_m3.size
        if self.u.shape != (3 * n_elec + 3 * m_chem,):
            raise ValueError("iodine state vector shape mismatch")
        if not np.array_equal(self.u[: 3 * n_elec], self.electronic_state.u):
            raise ValueError("iodine state must match its electronic component state")
        for arr in (
            self.iodide_density_m3,
            self.iodine_atom_density_m3,
            self.iodine_molecule_density_m3,
        ):
            if arr.shape != (m_chem,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                raise ValueError("iodine species densities must be finite and positive")

    @property
    def voltage_V(self) -> float:
        return self.electronic_state.voltage_V


@dataclass(frozen=True)
class CompiledIodineDevice1D:
    """Immutable compiled 1D device with coupled ``I^- / I^0 / I_2`` chemistry."""

    device_id: str
    sha256: str
    original_prepared_sha256: str
    electronic_component: CompiledDevice1D
    domain_mode: IodineDomainMode
    domain_layer_ids: tuple[str, ...]
    chem_geometry: ControlVolumeGeometry1D
    chem_node_indices: NDArray[np.intp]
    compensating_donor_m3: Vector
    initial_iodide_m3: Vector
    initial_atom_m3: Vector
    initial_mol_m3: Vector
    iodide_map: SpeciesCoordinateMap
    atom_map: SpeciesCoordinateMap
    mol_map: SpeciesCoordinateMap
    iodide_diffusivity_faces_m2_s: Vector
    atom_diffusivity_faces_m2_s: Vector
    mol_diffusivity_faces_m2_s: Vector
    log_partition_minus_faces: Vector
    log_partition_atom_faces: Vector
    log_partition_mol_faces: Vector
    kinetics: IodineKineticsParameters
    left_boundary: IodineBoundaryPortSpec
    right_boundary: IodineBoundaryPortSpec
    binding_manifest_json: str
    steric_site_limit_m3: Vector | None = None
    steric_diffusion_only: bool = True
    capabilities: tuple[str, ...] = field(
        default=(
            "classical_1d_electronic_component",
            "iodine_reaction_diffusion_1d",
            "physical_iodine_storage",
            "small_signal_iodine_ac",
        )
    )

    def __post_init__(self) -> None:
        if self.domain_mode not in _DOMAIN_MODES:
            raise ContractError("unknown_iodine_domain_mode")
        indices = integer_indices(self.chem_node_indices)
        m_count = self.chem_geometry.node_count
        n_count = self.electronic_component.node_count
        if (
            indices.shape != (m_count,)
            or np.any(np.diff(indices) != 1)
            or indices[0] < 0
            or indices[-1] >= n_count
        ):
            raise ContractError("chem_node_indices_must_be_contiguous_slice_of_nodes")
        idx_frozen: NDArray[np.intp] = np.frombuffer(indices.tobytes(), dtype=np.intp)
        object.__setattr__(self, "chem_node_indices", idx_frozen)

        for name in (
            "compensating_donor_m3",
            "initial_iodide_m3",
            "initial_atom_m3",
            "initial_mol_m3",
        ):
            vec = frozen_array(getattr(self, name))
            if vec.shape != (m_count,) or not np.all(np.isfinite(vec)) or np.any(vec < 0.0):
                raise ContractError(f"{name}_must_be_finite_nonnegative_chem_vector")
            object.__setattr__(self, name, vec)

        faces = m_count - 1
        for name in (
            "iodide_diffusivity_faces_m2_s",
            "atom_diffusivity_faces_m2_s",
            "mol_diffusivity_faces_m2_s",
        ):
            fvec = frozen_array(getattr(self, name))
            if fvec.shape != (faces,) or not np.all(np.isfinite(fvec)) or np.any(fvec < 0.0):
                raise ContractError(f"{name}_must_be_finite_nonnegative_face_vector")
            object.__setattr__(self, name, fvec)

        for name in (
            "log_partition_minus_faces",
            "log_partition_atom_faces",
            "log_partition_mol_faces",
        ):
            pvec = frozen_array(getattr(self, name))
            if pvec.shape != (faces,) or not np.all(np.isfinite(pvec)):
                raise ContractError(f"{name}_must_be_finite_face_vector")
            object.__setattr__(self, name, pvec)

        if self.steric_site_limit_m3 is not None:
            slim = frozen_array(self.steric_site_limit_m3)
            if slim.shape != (m_count,) or np.any(slim <= 0.0):
                raise ContractError("steric_site_limit_m3_must_be_positive_chem_vector")
            object.__setattr__(self, "steric_site_limit_m3", slim)

        if (
            self.iodide_map.size != m_count
            or self.atom_map.size != m_count
            or self.mol_map.size != m_count
            or self.kinetics.size != m_count
        ):
            raise ContractError("chem_maps_and_kinetics_must_match_chem_node_count")

    @property
    def identity(self) -> str:
        return self.sha256

    @property
    def node_count(self) -> int:
        return self.electronic_component.node_count

    @property
    def chem_node_count(self) -> int:
        return self.chem_geometry.node_count

    @property
    def state_size(self) -> int:
        return 3 * self.node_count + 3 * self.chem_node_count

    @property
    def chem_slices(self) -> dict[str, slice]:
        base = 3 * self.node_count
        m = self.chem_node_count
        return {
            "iodide": slice(base, base + m),
            "iodine_atom": slice(base + m, base + 2 * m),
            "iodine_molecule": slice(base + 2 * m, base + 3 * m),
        }

    @property
    def chem_weight_ratio(self) -> Vector:
        """Cell-volume embedding ratio ``w_chem / w_elec[chem_node_indices]``."""
        return frozen_array(
            self.chem_geometry.cell_measures_m
            / self.electronic_component.geometry.cell_measures_m[self.chem_node_indices]
        )

    @property
    def initial_total_iodine_atoms_m3(self) -> Vector:
        """Node-wise initial elemental iodine concentration ``c_- + c_0 + 2 c_2``."""
        return frozen_array(
            self.initial_iodide_m3 + self.initial_atom_m3 + 2.0 * self.initial_mol_m3
        )

    @property
    def reference_iodine_inventory_m2(self) -> float:
        """Initial represented total iodine sheet inventory on the coordinate maps."""
        u_init = self.initial_chem_coordinates()
        m = self.chem_node_count
        cm = self.iodide_map.density_m3(u_init[:m])
        c0 = self.atom_map.density_m3(u_init[m : 2 * m])
        c2 = self.mol_map.density_m3(u_init[2 * m : 3 * m])
        weights = self.chem_geometry.cell_measures_m
        return math.fsum((weights * (cm + c0 + 2.0 * c2)).tolist())

    @property
    def reference_species_inventories_m2(self) -> dict[str, float]:
        u_init = self.initial_chem_coordinates()
        m = self.chem_node_count
        weights = self.chem_geometry.cell_measures_m
        return {
            "iodide": math.fsum(
                (weights * self.iodide_map.density_m3(u_init[:m])).tolist()
            ),
            "iodine_atom": math.fsum(
                (weights * self.atom_map.density_m3(u_init[m : 2 * m])).tolist()
            ),
            "iodine_molecule": math.fsum(
                (weights * self.mol_map.density_m3(u_init[2 * m : 3 * m])).tolist()
            ),
        }

    @property
    def has_open_iodine_loss(self) -> bool:
        return (
            self.left_boundary.removes_device_iodine
            or self.right_boundary.removes_device_iodine
        )

    @property
    def is_unreactive_limit(self) -> bool:
        k = self.kinetics
        assert k.k_e_3body_m6_s is not None
        bulk_zero = bool(
            np.all(k.k_a_m3_s == 0.0)
            and np.all(k.k_b_s_1 == 0.0)
            and np.all(k.k_c_m3_s == 0.0)
            and np.all(k.k_d_m3_s == 0.0)
            and np.all(k.k_e_m3_s == 0.0)
            and np.all(k.k_e_3body_m6_s == 0.0)
        )
        contact_zero = (
            self.left_boundary.contact_oxidation_rate_m4_s == 0.0
            and self.right_boundary.contact_oxidation_rate_m4_s == 0.0
        )
        return bulk_zero and contact_zero

    @property
    def binding_manifest(self) -> dict[str, object]:
        return dict(json.loads(self.binding_manifest_json))

    def project_chem_density(self, chem_density_m3: ArrayLike) -> Vector:
        """Embed a ``chem`` cell density onto ``nodes`` scaled by ``w_chem / w_nodes``."""
        arr = np.asarray(chem_density_m3, dtype=np.float64)
        if arr.shape != (self.chem_node_count,) or not np.all(np.isfinite(arr)):
            raise ValueError("chem density must be finite with shape (chem_node_count,)")
        out: NDArray[np.float64] = np.zeros(self.node_count, dtype=np.float64)
        out[self.chem_node_indices] = self.chem_weight_ratio * arr
        return frozen_array(out)

    def initial_chem_coordinates(self) -> Vector:
        u_minus = self.iodide_map.coordinate_from_density(self.initial_iodide_m3)
        u_atom = self.atom_map.coordinate_from_density(self.initial_atom_m3)
        u_mol = self.mol_map.coordinate_from_density(self.initial_mol_m3)
        return frozen_array(np.concatenate((u_minus, u_atom, u_mol)))

    def initial_coordinates(
        self,
        *,
        voltage_V: float = 0.0,
        electronic_u: ArrayLike | None = None,
    ) -> Vector:
        if electronic_u is None:
            left, right = self.electronic_component.contact_potentials(voltage_V)
            pos = (
                self.electronic_component.nodes_m
                - self.electronic_component.nodes_m[0]
            ) / self.electronic_component.geometry.extent_m
            phi = left + (right - left) * pos
            elec = np.concatenate((phi, np.zeros(2 * self.node_count)))
        else:
            elec = np.asarray(electronic_u, dtype=np.float64)
            if elec.shape != (3 * self.node_count,):
                raise ValueError("electronic_u must have shape (3 * node_count,)")
        return frozen_array(np.concatenate((elec, self.initial_chem_coordinates())))

    def fields_and_tangents(
        self,
        u: ArrayLike,
        *,
        voltage_V: float = 0.0,
    ) -> tuple[
        Vector,
        Vector,
        Vector,
        Vector,
        Vector,
        Vector,
        Vector,
        Vector,
        Vector,
    ]:
        """Return ``(phi, n, p, c_-, c_0, c_2, dc_-/du_-, dc_0/du_0, dc_2/du_2)``."""
        vec = np.asarray(u, dtype=np.float64)
        if vec.shape != (self.state_size,) or not np.all(np.isfinite(vec)):
            raise ValueError("iodine coordinates must be finite of shape (state_size,)")
        n_elec = 3 * self.node_count
        phi, n, p = self.electronic_component.fields(vec[:n_elec], voltage_V=voltage_V)
        slices = self.chem_slices
        cm, mm = self.iodide_map.density_and_tangent_m3(vec[slices["iodide"]])
        c0, m0 = self.atom_map.density_and_tangent_m3(vec[slices["iodine_atom"]])
        c2, m2 = self.mol_map.density_and_tangent_m3(vec[slices["iodine_molecule"]])
        if any(
            not np.all(np.isfinite(arr)) or np.any(arr <= 0.0)
            for arr in (n, p, cm, c0, c2)
        ):
            raise FloatingPointError("trial densities outside positive finite domain")
        return phi, n, p, cm, c0, c2, mm, m0, m2

    def charge_ledger(self, n: Vector, p: Vector, c_minus_m3: Vector) -> ChargeLedger:
        component = self.electronic_component.charge_ledger(n, p)
        iodide_entry = ChargeEntry(
            "iodide",
            "iodine_chemistry",
            -1,
            self.project_chem_density(c_minus_m3),
            self.project_chem_density(self.compensating_donor_m3),
            group="iodine_space_charge",
        )
        return component.with_entry(iodide_entry)

    def state(self, u: ArrayLike, *, voltage_V: float = 0.0) -> IodineState1D:
        vec = np.asarray(u, dtype=np.float64)
        _, _, _, cm, c0, c2, _, _, _ = self.fields_and_tangents(vec, voltage_V=voltage_V)
        elec_state = self.electronic_component.state(
            vec[: 3 * self.node_count], voltage_V=voltage_V
        )
        w = self.chem_geometry.cell_measures_m
        inv_minus = math.fsum((w * cm).tolist())
        inv_atom = math.fsum((w * c0).tolist())
        inv_mol = math.fsum((w * c2).tolist())
        total_inv = math.fsum((w * (cm + c0 + 2.0 * c2)).tolist())
        return IodineState1D(
            u=vec,
            electronic_state=elec_state,
            iodide_density_m3=cm,
            iodine_atom_density_m3=c0,
            iodine_molecule_density_m3=c2,
            total_iodine_inventory_m2=total_inv,
            species_inventories_m2={
                "iodide": inv_minus,
                "iodine_atom": inv_atom,
                "iodine_molecule": inv_mol,
            },
            device_sha256=self.sha256,
        )

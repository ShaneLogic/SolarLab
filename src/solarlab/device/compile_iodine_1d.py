"""Compile a PreparedDevice and IodineChemistryDeclaration into CompiledIodineDevice1D.

Preserves existing :class:`~solarlab.device.inputs.DeviceInput` and frontend
configuration schemas untouched while binding multi-layer or absorber-confined
``I^- / I^0 / I_2`` chemistry onto SolarLab's shared 1D geometry and electronic
component projection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Literal, get_args

import numpy as np
from numpy.typing import NDArray

from solarlab.config.resolve_device import resolve_device
from solarlab.contracts import ContractError, frozen_array
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError, TERegularizationName
from solarlab.equations.iodine_1d import (
    CompiledIodineDevice1D,
    ContactReactionMode,
    IodineBoundaryPortSpec,
    IodineDomainMode,
)
from solarlab.mesh.geometry import (
    PHYSICAL_HALF_ENDPOINT_V2,
    half_endpoint_geometry,
    physical_half_endpoint_geometry,
)
from solarlab.physics.band_structure.temperature import ion_diffusivity_at_temperature
from solarlab.physics.chemistry import (
    BoundedSpeciesMap,
    IodineKineticsParameters,
    IodineThermodynamicMode,
    ReactionEOrder,
    SoftplusSpeciesMap,
    SpeciesCoordinateMap,
    equilibrium_iodine_populations_m3,
    harmonic_face_diffusivity_m2_s,
    smoluchowski_rate_m3_s,
)
from solarlab.physics.transport.electrochemical_flux import LEGACY_SUMMED_POTENTIAL_V1
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_equilibrium

__all__ = [
    "IodineBoundaryDeclaration",
    "IodineChemistryDeclaration",
    "IodineLayerSpec",
    "IodineRateParameterization",
    "bitton_tessler_2023_table_s1_chemistry",
    "compile_iodine_device_1d",
]

IODINE_COMPILER_SCHEMA = "solarlab.compiled_iodine_device_1d.v1"

IodineRateParameterization = Literal[
    "explicit_rates",
    "smoluchowski_capture_radius",
]
_RATE_MODES: frozenset[str] = frozenset(get_args(IodineRateParameterization))


@dataclass(frozen=True, slots=True)
class IodineLayerSpec:
    """Per-layer SI transport, reaction, partition, and initial-population inputs."""

    layer_id: str
    D_minus_m2_s: float
    D_atom_m2_s: float
    D_mol_m2_s: float
    initial_iodide_m3: float = 0.0
    initial_atom_m3: float = 0.0
    initial_mol_m3: float = 0.0
    compensating_donor_m3: float | None = None
    E_a_minus_eV: float = 0.0
    E_a_atom_eV: float = 0.0
    E_a_mol_eV: float = 0.0
    rate_mode: IodineRateParameterization = "explicit_rates"
    k_a_m3_s: float = 0.0
    k_b_s_1: float = 0.0
    k_c_m3_s: float = 0.0
    k_d_m3_s: float = 0.0
    k_e_m3_s: float = 0.0
    k_e_3body_m6_s: float = 0.0
    capture_radius_m: float = 0.2e-9
    K_diss_m3: float = 1.0e14
    n_e_star_m3: float = 1.0e14
    partition_minus: float = 1.0
    partition_atom: float = 1.0
    partition_mol: float = 1.0
    iodide_site_limit_m3: float | None = None

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise ContractError("layer_id_must_be_nonempty")
        if self.rate_mode not in _RATE_MODES:
            raise ContractError("unknown_iodine_rate_parameterization")
        for name in (
            "D_minus_m2_s",
            "D_atom_m2_s",
            "D_mol_m2_s",
            "initial_iodide_m3",
            "initial_atom_m3",
            "initial_mol_m3",
            "E_a_minus_eV",
            "E_a_atom_eV",
            "E_a_mol_eV",
            "k_a_m3_s",
            "k_b_s_1",
            "k_c_m3_s",
            "k_d_m3_s",
            "k_e_m3_s",
            "k_e_3body_m6_s",
            "capture_radius_m",
            "n_e_star_m3",
        ):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val < 0.0:
                raise ContractError(f"{name}_must_be_finite_nonnegative")
            object.__setattr__(self, name, val)

        for pname in ("K_diss_m3", "partition_minus", "partition_atom", "partition_mol"):
            pval = float(getattr(self, pname))
            if not math.isfinite(pval) or pval <= 0.0:
                raise ContractError(f"{pname}_must_be_strictly_positive")
            object.__setattr__(self, pname, pval)

        comp_donor = (
            self.initial_iodide_m3
            if self.compensating_donor_m3 is None
            else float(self.compensating_donor_m3)
        )
        if not math.isfinite(comp_donor) or comp_donor < 0.0:
            raise ContractError("compensating_donor_m3_must_be_finite_nonnegative")
        object.__setattr__(self, "compensating_donor_m3", comp_donor)

        if self.iodide_site_limit_m3 is not None:
            slim = float(self.iodide_site_limit_m3)
            if not math.isfinite(slim) or slim <= 0.0 or self.initial_iodide_m3 >= slim:
                raise ContractError("initial_iodide_must_be_below_positive_site_limit")
            object.__setattr__(self, "iodide_site_limit_m3", slim)

    @classmethod
    def from_cgs(
        cls,
        layer_id: str,
        *,
        D_minus_cm2_s: float,
        D_atom_cm2_s: float,
        D_mol_cm2_s: float,
        initial_iodide_cm3: float = 0.0,
        initial_atom_cm3: float = 0.0,
        initial_mol_cm3: float = 0.0,
        compensating_donor_cm3: float | None = None,
        rate_mode: IodineRateParameterization = "explicit_rates",
        k_a_cm3_s: float = 0.0,
        k_b_s_1: float = 0.0,
        k_c_cm3_s: float = 0.0,
        k_d_cm3_s: float = 0.0,
        k_e_cm3_s: float = 0.0,
        k_e_3body_cm6_s: float = 0.0,
        capture_radius_nm: float = 0.2,
        K_diss_cm3: float = 1.0e8,
        n_e_star_cm3: float = 1.0e8,
        partition_minus: float = 1.0,
        partition_atom: float = 1.0,
        partition_mol: float = 1.0,
        iodide_site_limit_cm3: float | None = None,
    ) -> IodineLayerSpec:
        """Construct SI layer specification directly from literature CGS units."""
        return cls(
            layer_id=layer_id,
            D_minus_m2_s=D_minus_cm2_s * 1.0e-4,
            D_atom_m2_s=D_atom_cm2_s * 1.0e-4,
            D_mol_m2_s=D_mol_cm2_s * 1.0e-4,
            initial_iodide_m3=initial_iodide_cm3 * 1.0e6,
            initial_atom_m3=initial_atom_cm3 * 1.0e6,
            initial_mol_m3=initial_mol_cm3 * 1.0e6,
            compensating_donor_m3=(
                None
                if compensating_donor_cm3 is None
                else compensating_donor_cm3 * 1.0e6
            ),
            rate_mode=rate_mode,
            k_a_m3_s=k_a_cm3_s * 1.0e-6,
            k_b_s_1=k_b_s_1,
            k_c_m3_s=k_c_cm3_s * 1.0e-6,
            k_d_m3_s=k_d_cm3_s * 1.0e-6,
            k_e_m3_s=k_e_cm3_s * 1.0e-6,
            k_e_3body_m6_s=k_e_3body_cm6_s * 1.0e-12,
            capture_radius_m=capture_radius_nm * 1.0e-9,
            K_diss_m3=K_diss_cm3 * 1.0e6,
            n_e_star_m3=n_e_star_cm3 * 1.0e6,
            partition_minus=partition_minus,
            partition_atom=partition_atom,
            partition_mol=partition_mol,
            iodide_site_limit_m3=(
                None
                if iodide_site_limit_cm3 is None
                else iodide_site_limit_cm3 * 1.0e6
            ),
        )


@dataclass(frozen=True, slots=True)
class IodineBoundaryDeclaration:
    """User-level boundary closure for volatile ``I_2`` escape and contact oxidation."""

    mol_escape_velocity_m_s: float = 0.0
    contact_reaction_mode: ContactReactionMode = "none"
    contact_oxidation_rate_m4_s: float = 0.0


@dataclass(frozen=True, slots=True)
class IodineChemistryDeclaration:
    """Complete iodine chemistry configuration bound to a PreparedDevice."""

    layers: tuple[IodineLayerSpec, ...]
    domain_mode: IodineDomainMode = "absorber_blocking"
    thermodynamic_mode: IodineThermodynamicMode = "wegscheider_detailed_balance"
    reaction_e_order: ReactionEOrder = "bimolecular_n"
    left_boundary: IodineBoundaryDeclaration = field(
        default_factory=IodineBoundaryDeclaration
    )
    right_boundary: IodineBoundaryDeclaration = field(
        default_factory=IodineBoundaryDeclaration
    )
    initialize_equilibrium_split: bool = False
    enforce_spatial_wegscheider: bool = True
    steric_diffusion_only: bool = True

    def __post_init__(self) -> None:
        items = tuple(self.layers)
        if not items:
            raise ContractError("iodine_chemistry_requires_at_least_one_layer_spec")
        ids = [item.layer_id for item in items]
        if len(set(ids)) != len(ids):
            raise ContractError("duplicate_iodine_layer_id")
        object.__setattr__(self, "layers", items)


def bitton_tessler_2023_table_s1_chemistry(
    *,
    hbl_layer_id: str = "hbl",
    absorber_layer_id: str = "absorber",
    ebl_layer_id: str = "ebl",
    domain_mode: IodineDomainMode = "full_stack_permeable",
    thermodynamic_mode: IodineThermodynamicMode = "wegscheider_detailed_balance",
    reaction_e_order: ReactionEOrder = "bimolecular_n",
    rate_mode: IodineRateParameterization = "explicit_rates",
    perovskite_ka_cm3_s: float = 1.0e-11,
    left_escape_m_s: float = 0.0,
    right_escape_m_s: float = 0.0,
    left_boundary: IodineBoundaryDeclaration | None = None,
    right_boundary: IodineBoundaryDeclaration | None = None,
    initialize_equilibrium_split: bool = True,
) -> IodineChemistryDeclaration:
    """Return Bitton & Tessler (2023) SI Table S1 iodine parameters."""
    unreactive = float(perovskite_ka_cm3_s) == 0.0
    eff_rate_mode: IodineRateParameterization = (
        "explicit_rates" if unreactive else rate_mode
    )
    pvk = IodineLayerSpec.from_cgs(
        absorber_layer_id,
        D_minus_cm2_s=1.0e-8,
        D_atom_cm2_s=1.0e-8,
        D_mol_cm2_s=1.0e-9,
        initial_iodide_cm3=1.0e18,
        compensating_donor_cm3=1.0e18,
        rate_mode=eff_rate_mode,
        k_a_cm3_s=0.0 if unreactive else perovskite_ka_cm3_s,
        k_b_s_1=0.0 if unreactive else 3.0e5,
        k_c_cm3_s=0.0 if unreactive else 5.0e-7,
        k_d_cm3_s=0.0 if unreactive else 5.0e-15,
        k_e_cm3_s=0.0 if unreactive else 1.3e-8,
        capture_radius_nm=0.2,
        K_diss_cm3=1.0e8,
        n_e_star_cm3=1.0e8,
    )
    if domain_mode == "absorber_blocking":
        layers: tuple[IodineLayerSpec, ...] = (pvk,)
    else:
        hbl = IodineLayerSpec.from_cgs(
            hbl_layer_id,
            D_minus_cm2_s=1.0e-12,
            D_atom_cm2_s=1.0e-12,
            D_mol_cm2_s=1.0e-13,
            initial_iodide_cm3=0.0,
            compensating_donor_cm3=0.0,
            rate_mode=eff_rate_mode,
            k_a_cm3_s=0.0 if unreactive else 6.5e-13,
            k_b_s_1=0.0 if unreactive else 6.0e3,
            k_c_cm3_s=0.0 if unreactive else 2.5e-11,
            k_d_cm3_s=0.0 if unreactive else 5.0e-19,
            k_e_cm3_s=0.0 if unreactive else 1.3e-10,
            capture_radius_nm=0.2,
            K_diss_cm3=1.0e8,
            n_e_star_cm3=1.0e6,
        )
        ebl = IodineLayerSpec.from_cgs(
            ebl_layer_id,
            D_minus_cm2_s=1.0e-12,
            D_atom_cm2_s=1.0e-12,
            D_mol_cm2_s=1.0e-13,
            initial_iodide_cm3=0.0,
            compensating_donor_cm3=0.0,
            rate_mode=eff_rate_mode,
            k_a_cm3_s=0.0 if unreactive else 6.5e-13,
            k_b_s_1=0.0 if unreactive else 6.0e-12,
            k_c_cm3_s=0.0 if unreactive else 2.5e-11,
            k_d_cm3_s=0.0 if unreactive else 5.0e-19,
            k_e_cm3_s=0.0 if unreactive else 1.3e-10,
            capture_radius_nm=0.2,
            K_diss_cm3=1.0e8,
            n_e_star_cm3=1.0e6,
        )
        layers = (hbl, pvk, ebl)
    lb = (
        IodineBoundaryDeclaration(mol_escape_velocity_m_s=left_escape_m_s)
        if left_boundary is None
        else left_boundary
    )
    rb = (
        IodineBoundaryDeclaration(mol_escape_velocity_m_s=right_escape_m_s)
        if right_boundary is None
        else right_boundary
    )
    return IodineChemistryDeclaration(
        layers=layers,
        domain_mode=domain_mode,
        thermodynamic_mode=thermodynamic_mode,
        reaction_e_order=reaction_e_order,
        left_boundary=lb,
        right_boundary=rb,
        initialize_equilibrium_split=initialize_equilibrium_split and not unreactive,
    )


def _default_positive_maps(
    m_count: int,
    initial_iodide_m3: np.ndarray,
    site_limit_m3: np.ndarray | None,
) -> tuple[SpeciesCoordinateMap, SpeciesCoordinateMap, SpeciesCoordinateMap]:
    max_iodide = max(float(np.max(initial_iodide_m3)), 1.0e20)
    if site_limit_m3 is not None:
        eff_init = np.maximum(initial_iodide_m3, 1.0e6)
        theta = np.clip(eff_init / site_limit_m3, 1.0e-18, 0.999999)
        logit_ref = np.log(theta) - np.log1p(-theta)
        iodide_map: SpeciesCoordinateMap = BoundedSpeciesMap(
            site_limit_m3=site_limit_m3,
            logit_reference=logit_ref,
            vacuum_floor_m3=1.0,
        )
    else:
        iodide_map = SoftplusSpeciesMap(
            scale_m3=np.full(m_count, max_iodide * 0.1),
            vacuum_floor_m3=1.0,
        )
    s_0 = max(max_iodide * 1.0e-5, 1.0e17)
    atom_map = SoftplusSpeciesMap(
        scale_m3=np.full(m_count, s_0),
        vacuum_floor_m3=1.0,
    )
    s_2 = max(max_iodide * 1.0e-8, 1.0e14)
    mol_map = SoftplusSpeciesMap(
        scale_m3=np.full(m_count, s_2),
        vacuum_floor_m3=1.0,
    )
    return iodide_map, atom_map, mol_map


def compile_iodine_device_1d(
    prepared: PreparedDevice,
    chemistry: IodineChemistryDeclaration,
    *,
    intervals_per_layer: int | Sequence[int] = 40,
    te_softness: float = 0.02,
    te_regularization_mode: TERegularizationName = "legacy_logistic",
    optical_model: Literal["beer_lambert", "beer_lambert_spectral", "tmm"] | None = None,
    dielectric_rule: str | None = None,
    geometry_name: str | None = None,
    band_step_mode: str = LEGACY_SUMMED_POTENTIAL_V1,
) -> CompiledIodineDevice1D:
    """Compile a PreparedDevice and IodineChemistryDeclaration into a 1D system."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("compile_iodine_device_1d requires a PreparedDevice")
    if not isinstance(chemistry, IodineChemistryDeclaration):
        raise TypeError(
            "compile_iodine_device_1d requires an IodineChemistryDeclaration"
        )

    elec_layers = tuple(
        layer for layer in prepared.layers if layer.role != "substrate"
    )
    layer_ids = tuple(layer.id for layer in elec_layers)
    spec_by_id = {spec.layer_id: spec for spec in chemistry.layers}

    if chemistry.domain_mode == "absorber_blocking":
        if len(chemistry.layers) != 1:
            raise CapabilityError(
                "absorber_blocking mode requires exactly one IodineLayerSpec"
            )
        target_id = chemistry.layers[0].layer_id
        if target_id not in layer_ids:
            raise CapabilityError(
                f"absorber_blocking layer {target_id!r} not in electrical stack"
            )
        domain_layer_ids = (target_id,)
    else:
        if set(spec_by_id) != set(layer_ids):
            raise CapabilityError(
                "full_stack_permeable mode requires an IodineLayerSpec for every "
                f"non-substrate electrical layer {layer_ids}"
            )
        domain_layer_ids = layer_ids

    # Project out legacy mobile-ion parameters from the electronic component
    projection = prepared.to_input().editing_data()
    for layer_dict in projection["layers"]:
        for key in ("P0", "P0_neg", "D_ion", "D_ion_neg"):
            layer_dict.setdefault("parameters", {})[key] = 0.0
    for key in ("ion_steric_diffusion_only", "ion_steric_shared_site"):
        projection.setdefault("settings", {})[key] = False
    component_prepared = resolve_device(
        DeviceInput.model_validate(projection),
        prepared.defaults,
        prepared.resources,
        sources=prepared.sources,
    )
    component = compile_device_1d(
        component_prepared,
        intervals_per_layer=intervals_per_layer,
        te_softness=te_softness,
        te_regularization_mode=te_regularization_mode,
        optical_model=optical_model,
        dielectric_rule=dielectric_rule,
        geometry_name=geometry_name,
        band_step_mode=band_step_mode,
    )

    temp_K = component.parameters.temperature_K
    n_count = component.node_count
    planes = component.geometry.interfaces

    # Resolve layer node spans [left_l, right_l] on the shared-node electrical grid
    layer_spans: list[tuple[str, int, int]] = []
    for l_idx, l_id in enumerate(layer_ids):
        left = 0 if l_idx == 0 else planes[l_idx - 1].shared_node
        right = n_count - 1 if l_idx == len(layer_ids) - 1 else planes[l_idx].shared_node
        if left is None or right is None:
            raise CapabilityError(
                "iodine 1D compiler requires shared electrostatic interface nodes"
            )
        layer_spans.append((l_id, int(left), int(right)))

    if chemistry.domain_mode == "absorber_blocking":
        target_id = domain_layer_ids[0]
        _, left_idx, right_idx = next(
            item for item in layer_spans if item[0] == target_id
        )
        chem_indices: NDArray[np.intp] = np.arange(left_idx, right_idx + 1, dtype=np.intp)
        active_spans = [(target_id, 0, int(chem_indices.size) - 1)]
    else:
        chem_indices = np.arange(n_count, dtype=np.intp)
        active_spans = layer_spans

    chem_geometry = (
        physical_half_endpoint_geometry(component.nodes_m[chem_indices])
        if component.geometry.name == PHYSICAL_HALF_ENDPOINT_V2
        else half_endpoint_geometry(component.nodes_m[chem_indices])
    )

    m_count = chem_geometry.node_count
    m_faces = m_count - 1
    chem_nodes_m = chem_geometry.nodes_m
    chem_dx = np.diff(chem_nodes_m)
    chem_volumes = chem_geometry.cell_measures_m

    # Accumulate cell-measure-weighted quantities per layer so shared interface
    # nodes conserve exact layer sheet inventories and reaction integrals!
    comp_donor = np.zeros(m_count, dtype=np.float64)
    init_cm = np.zeros(m_count, dtype=np.float64)
    init_c0 = np.zeros(m_count, dtype=np.float64)
    init_c2 = np.zeros(m_count, dtype=np.float64)
    k_a = np.zeros(m_count, dtype=np.float64)
    k_b = np.zeros(m_count, dtype=np.float64)
    k_c = np.zeros(m_count, dtype=np.float64)
    k_d = np.zeros(m_count, dtype=np.float64)
    k_e = np.zeros(m_count, dtype=np.float64)
    k_e3 = np.zeros(m_count, dtype=np.float64)
    k_diss = np.zeros(m_count, dtype=np.float64)
    n_star = np.zeros(m_count, dtype=np.float64)
    w_accum = np.zeros(m_count, dtype=np.float64)

    d_minus_faces = np.zeros(m_faces, dtype=np.float64)
    d_atom_faces = np.zeros(m_faces, dtype=np.float64)
    d_mol_faces = np.zeros(m_faces, dtype=np.float64)
    part_minus_nodes = np.ones(m_count, dtype=np.float64)
    part_atom_nodes = np.ones(m_count, dtype=np.float64)
    part_mol_nodes = np.ones(m_count, dtype=np.float64)

    has_steric = any(spec.iodide_site_limit_m3 is not None for spec in chemistry.layers)
    if has_steric and not all(
        spec.iodide_site_limit_m3 is not None for spec in chemistry.layers
    ):
        raise CapabilityError(
            "iodide_site_limit_m3 must be set on all active layers or none"
        )
    site_limit = np.zeros(m_count, dtype=np.float64) if has_steric else None

    d_elec_chem = component.parameters.electron_diffusivity_m2_s[chem_indices]
    d_hole_chem = component.parameters.hole_diffusivity_m2_s[chem_indices]

    for l_id, c_left, c_right in active_spans:
        spec = spec_by_id[l_id]
        # Sub-cell half-interval weights inside layer [c_left, c_right]
        sub_w: NDArray[np.float64] = np.zeros(c_right - c_left + 1, dtype=np.float64)
        layer_dx = chem_dx[c_left:c_right]
        sub_w[:-1] += 0.5 * layer_dx
        sub_w[1:] += 0.5 * layer_dx
        sl = slice(c_left, c_right + 1)

        dm_T = ion_diffusivity_at_temperature(
            spec.D_minus_m2_s, temp_K, spec.E_a_minus_eV
        )
        d0_T = ion_diffusivity_at_temperature(
            spec.D_atom_m2_s, temp_K, spec.E_a_atom_eV
        )
        d2_T = ion_diffusivity_at_temperature(
            spec.D_mol_m2_s, temp_K, spec.E_a_mol_eV
        )
        d_minus_faces[c_left:c_right] = dm_T
        d_atom_faces[c_left:c_right] = d0_T
        d_mol_faces[c_left:c_right] = d2_T

        if spec.rate_mode == "smoluchowski_capture_radius":
            ka_nodes = (
                np.full(sub_w.size, spec.k_a_m3_s)
                if spec.k_a_m3_s > 0.0
                else smoluchowski_rate_m3_s(
                    dm_T, d_hole_chem[sl], spec.capture_radius_m
                )
            )
            kc_nodes = smoluchowski_rate_m3_s(
                d0_T, d_elec_chem[sl], spec.capture_radius_m
            )
            kd_nodes = smoluchowski_rate_m3_s(d0_T, d0_T, spec.capture_radius_m)
            ke_nodes = smoluchowski_rate_m3_s(
                d2_T, d_elec_chem[sl], spec.capture_radius_m
            )
        else:
            ka_nodes = np.full(sub_w.size, spec.k_a_m3_s)
            kc_nodes = np.full(sub_w.size, spec.k_c_m3_s)
            kd_nodes = np.full(sub_w.size, spec.k_d_m3_s)
            ke_nodes = np.full(sub_w.size, spec.k_e_m3_s)

        assert spec.compensating_donor_m3 is not None
        w_accum[sl] += sub_w
        comp_donor[sl] += sub_w * spec.compensating_donor_m3
        init_cm[sl] += sub_w * spec.initial_iodide_m3
        init_c0[sl] += sub_w * spec.initial_atom_m3
        init_c2[sl] += sub_w * spec.initial_mol_m3
        k_a[sl] += sub_w * ka_nodes
        k_b[sl] += sub_w * spec.k_b_s_1
        k_c[sl] += sub_w * kc_nodes
        k_d[sl] += sub_w * kd_nodes
        k_e[sl] += sub_w * ke_nodes
        k_e3[sl] += sub_w * spec.k_e_3body_m6_s
        k_diss[sl] += sub_w * spec.K_diss_m3
        n_star[sl] += sub_w * spec.n_e_star_m3
        if site_limit is not None:
            assert spec.iodide_site_limit_m3 is not None
            site_limit[sl] += sub_w * spec.iodide_site_limit_m3

        # Interior nodes take exact layer partition; shared heterointerface node
        # takes geometric mean so total chemical step across both adjacent faces
        # equals ln(K_part_right / K_part_left) identically!
        part_minus_nodes[c_left + 1 : c_right] = spec.partition_minus
        part_atom_nodes[c_left + 1 : c_right] = spec.partition_atom
        part_mol_nodes[c_left + 1 : c_right] = spec.partition_mol

    # Endpoint & shared interface partition geometric means
    first_spec = spec_by_id[active_spans[0][0]]
    last_spec = spec_by_id[active_spans[-1][0]]
    part_minus_nodes[0] = first_spec.partition_minus
    part_atom_nodes[0] = first_spec.partition_atom
    part_mol_nodes[0] = first_spec.partition_mol
    part_minus_nodes[-1] = last_spec.partition_minus
    part_atom_nodes[-1] = last_spec.partition_atom
    part_mol_nodes[-1] = last_spec.partition_mol
    for span_i in range(len(active_spans) - 1):
        shared_c = active_spans[span_i][2]
        sp_l = spec_by_id[active_spans[span_i][0]]
        sp_r = spec_by_id[active_spans[span_i + 1][0]]
        part_minus_nodes[shared_c] = math.sqrt(
            sp_l.partition_minus * sp_r.partition_minus
        )
        part_atom_nodes[shared_c] = math.sqrt(sp_l.partition_atom * sp_r.partition_atom)
        part_mol_nodes[shared_c] = math.sqrt(sp_l.partition_mol * sp_r.partition_mol)

    # Normalize accumulated cell integrals by chem_volumes
    # (note: on physical_half_endpoint_v2 vs half_endpoint_v1 w_accum matches chem_volumes)
    inv_w = 1.0 / w_accum
    comp_donor *= inv_w
    init_cm *= inv_w
    init_c0 *= inv_w
    init_c2 *= inv_w
    k_a *= inv_w
    k_b *= inv_w
    k_c *= inv_w
    k_d *= inv_w
    k_e *= inv_w
    k_e3 *= inv_w
    k_diss *= inv_w
    n_star *= inv_w
    if site_limit is not None:
        site_limit *= inv_w

    # Harmonic smoothing at faces adjacent to layer transitions if needed
    # (each face already lies inside one layer with uniform layer diffusivity;
    # validate via harmonic_face_diffusivity_m2_s for strict finiteness).
    d_minus_faces = harmonic_face_diffusivity_m2_s(d_minus_faces, d_minus_faces)
    d_atom_faces = harmonic_face_diffusivity_m2_s(d_atom_faces, d_atom_faces)
    d_mol_faces = harmonic_face_diffusivity_m2_s(d_mol_faces, d_mol_faces)

    log_part_minus = np.log(part_minus_nodes[1:] / part_minus_nodes[:-1])
    log_part_atom = np.log(part_atom_nodes[1:] / part_atom_nodes[:-1])
    log_part_mol = np.log(part_mol_nodes[1:] / part_mol_nodes[:-1])

    ni_sq_chem = component.parameters.intrinsic_product_m6[chem_indices]
    n1_override: np.ndarray | None = None
    if (
        chemistry.thermodynamic_mode == "wegscheider_detailed_balance"
        and chemistry.enforce_spatial_wegscheider
        and np.any((k_a > 0.0) & (k_b > 0.0))
    ):
        # Anchor vacuum I^0/I^- redox level and I_2 dissociation free energy to
        # the primary absorber (or highest initial-iodine layer) so dark thermal
        # equilibrium satisfies zero net flux AND zero net reaction simultaneously
        # across all interior and shared heterointerface nodes.
        ref_span = max(
            active_spans,
            key=lambda item: spec_by_id[item[0]].initial_iodide_m3,
        )
        ref_mid = (ref_span[1] + ref_span[2]) // 2
        n1_ref = float((k_a[ref_mid] * ni_sq_chem[ref_mid]) / k_b[ref_mid])
        kdiss_ref = float(k_diss[ref_mid])
        n_ref_chem = component.reference_electron_density_m3[chem_indices]
        phi_ref_chem = component.reference_potential_V[chem_indices]
        vt_V = component.thermal_voltage_V
        rel_pm = part_minus_nodes / part_minus_nodes[ref_mid]
        rel_p0 = part_atom_nodes / part_atom_nodes[ref_mid]
        rel_p2 = part_mol_nodes / part_mol_nodes[ref_mid]
        n1_spatial = (
            n1_ref
            * (rel_pm / rel_p0)
            * (n_ref_chem / n_ref_chem[ref_mid])
            * np.exp(-(phi_ref_chem - phi_ref_chem[ref_mid]) / vt_V)
        )
        active_mask = k_a > 0.0
        k_b = np.where(active_mask, (k_a * ni_sq_chem) / n1_spatial, 0.0)
        n1_override = np.where(active_mask, n1_spatial, 0.0)
        k_diss = kdiss_ref * ((rel_p0 * rel_p0) / rel_p2)

    kinetics = IodineKineticsParameters(
        k_a_m3_s=k_a,
        k_b_s_1=k_b,
        k_c_m3_s=k_c,
        k_d_m3_s=k_d,
        k_e_m3_s=k_e,
        k_e_3body_m6_s=k_e3,
        n_i_sq_m6=ni_sq_chem,
        K_diss_m3=k_diss,
        n_e_star_m3=n_star,
        n_1_iodine_m3=n1_override,
        thermodynamic_mode=chemistry.thermodynamic_mode,
        reaction_e_order=chemistry.reaction_e_order,
    )

    if (
        chemistry.initialize_equilibrium_split
        and chemistry.thermodynamic_mode == "wegscheider_detailed_balance"
    ):
        eq_res = solve_equilibrium(component, options=SteadyOptions())
        n_eq = eq_res.state.electron_density_m3[chem_indices]
        tot_i = init_cm + init_c0 + 2.0 * init_c2
        init_cm, init_c0, init_c2 = equilibrium_iodine_populations_m3(
            tot_i, n_eq, kinetics
        )

    iodide_map, atom_map, mol_map = _default_positive_maps(
        m_count, np.asarray(init_cm, dtype=np.float64), site_limit
    )

    reversible_ct = chemistry.thermodynamic_mode == "wegscheider_detailed_balance"
    p1_left = (
        float(k_b[0] / k_a[0]) if k_a[0] > 0.0 and k_b[0] > 0.0 else 0.0
    )
    p1_right = (
        float(k_b[-1] / k_a[-1]) if k_a[-1] > 0.0 and k_b[-1] > 0.0 else 0.0
    )
    left_port = IodineBoundaryPortSpec(
        side="left",
        chem_node=0,
        elec_node=int(chem_indices[0]),
        outward_normal=-1,
        mol_escape_velocity_m_s=chemistry.left_boundary.mol_escape_velocity_m_s,
        contact_reaction_mode=chemistry.left_boundary.contact_reaction_mode,
        contact_oxidation_rate_m4_s=chemistry.left_boundary.contact_oxidation_rate_m4_s,
        p1_iodine_m3=p1_left,
        reversible_contact=reversible_ct
        and chemistry.left_boundary.contact_reaction_mode == "hole_oxidation_to_I0",
    )
    right_port = IodineBoundaryPortSpec(
        side="right",
        chem_node=m_count - 1,
        elec_node=int(chem_indices[-1]),
        outward_normal=1,
        mol_escape_velocity_m_s=chemistry.right_boundary.mol_escape_velocity_m_s,
        contact_reaction_mode=chemistry.right_boundary.contact_reaction_mode,
        contact_oxidation_rate_m4_s=chemistry.right_boundary.contact_oxidation_rate_m4_s,
        p1_iodine_m3=p1_right,
        reversible_contact=reversible_ct
        and chemistry.right_boundary.contact_reaction_mode == "hole_oxidation_to_I0",
    )

    manifest = {
        "schema": IODINE_COMPILER_SCHEMA,
        "original_prepared_sha256": prepared.content_sha256,
        "electronic_component_sha256": component.identity,
        "domain_mode": chemistry.domain_mode,
        "domain_layer_ids": list(domain_layer_ids),
        "chem_node_indices": chem_indices.tolist(),
        "chem_volumes_m": chem_volumes.tolist(),
        "kinetics_identity": kinetics.identity,
        "maps": [iodide_map.identity, atom_map.identity, mol_map.identity],
        "left_boundary": {
            "escape": left_port.mol_escape_velocity_m_s,
            "mode": left_port.contact_reaction_mode,
            "rate": left_port.contact_oxidation_rate_m4_s,
        },
        "right_boundary": {
            "escape": right_port.mol_escape_velocity_m_s,
            "mode": right_port.contact_reaction_mode,
            "rate": right_port.contact_oxidation_rate_m4_s,
        },
        "scientific_qualification": False,
    }
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(manifest_json.encode()).hexdigest()

    return CompiledIodineDevice1D(
        device_id=component.device_id,
        sha256=digest,
        original_prepared_sha256=prepared.content_sha256,
        electronic_component=component,
        domain_mode=chemistry.domain_mode,
        domain_layer_ids=domain_layer_ids,
        chem_geometry=chem_geometry,
        chem_node_indices=chem_indices,
        compensating_donor_m3=frozen_array(comp_donor),
        initial_iodide_m3=frozen_array(init_cm),
        initial_atom_m3=frozen_array(init_c0),
        initial_mol_m3=frozen_array(init_c2),
        iodide_map=iodide_map,
        atom_map=atom_map,
        mol_map=mol_map,
        iodide_diffusivity_faces_m2_s=d_minus_faces,
        atom_diffusivity_faces_m2_s=d_atom_faces,
        mol_diffusivity_faces_m2_s=d_mol_faces,
        log_partition_minus_faces=frozen_array(log_part_minus),
        log_partition_atom_faces=frozen_array(log_part_atom),
        log_partition_mol_faces=frozen_array(log_part_mol),
        kinetics=kinetics,
        left_boundary=left_port,
        right_boundary=right_port,
        binding_manifest_json=manifest_json,
        steric_site_limit_m3=None if site_limit is None else frozen_array(site_limit),
        steric_diffusion_only=chemistry.steric_diffusion_only,
    )

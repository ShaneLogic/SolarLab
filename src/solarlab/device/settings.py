"""Declared device scalar settings, independent of configuration file formats."""

from __future__ import annotations

from typing import Literal
from pydantic import ValidationInfo, field_validator
from solarlab.materials.parameters import QuantityInput
from solarlab.materials.full_parameters import ParameterFieldsInput, quantity_field as q


class DeviceSettingsInput(ParameterFieldsInput):
    @classmethod
    def _allows_null(cls, name: str) -> bool:
        return name in {"built_in_potential_mode", "work_function_left_eV", "work_function_right_eV"}

    phi_left: QuantityInput | None = q("V", signed=True)
    V_bi: QuantityInput | None = q("V", signed=True)
    built_in_potential_mode: Literal["legacy_manual", "semiconductor_work_function", "metal_work_function"] | None = None
    work_function_left_eV: QuantityInput | None = q("eV", positive=True, nullable=True)
    work_function_right_eV: QuantityInput | None = q("eV", positive=True, nullable=True)
    Phi: QuantityInput | None = q("m^-2/s")
    T: QuantityInput | None = q("K", positive=True)
    mode: Literal["legacy", "fast", "full"] | None = None
    interface_plane_projection: bool | None = None
    dos_band_potentials: bool | None = None
    te_physical_norm: bool | None = None
    ion_steric_diffusion_only: bool | None = None
    ion_steric_shared_site: bool | None = None
    flat_band_contacts: bool | None = None
    flat_band_metal_contacts: bool | None = None
    contact_phi_B_eV: QuantityInput | None = q("eV")
    interface_two_sided: bool | None = None
    interface_shared_occupancy: bool | None = None
    interface_plane_closure: bool | None = None
    interface_plane_generation: bool | None = None
    het_recomb_despike: QuantityInput | None = q("1")
    band_grading: bool | None = None
    graded_optics: bool | None = None
    interface_tunneling: bool | None = None
    tunnel_mass_eff: QuantityInput | None = q("1", positive=True)
    jv_solver_policy: Literal["general", "cancellation_safe_qf_required"] | None = None
    interface_charge_closure: Literal["off", "equilibrium_referenced"] | None = None
    interface_charge_rebaseline_acknowledged: bool | None = None

    @field_validator("*")
    @classmethod
    def _null_flags(cls, value: object, info: ValidationInfo) -> object:
        if value is None and not cls._allows_null(str(info.field_name)):
            raise ValueError("omit to inherit; explicit null is not allowed for this setting")
        return value

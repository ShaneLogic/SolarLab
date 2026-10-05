"""Injected immutable default data; no Python-source or legacy dependency."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from solarlab.device.settings import DeviceSettingsInput
from solarlab.materials.full_parameters import full_parameter_items
from solarlab.materials.parameters import Scalar
from solarlab.units import normalize_quantity

__all__ = ["DefaultCatalog"]


def _pairs(values: tuple[tuple[str, Scalar], ...]) -> dict[str, Scalar]:
    pairs = tuple(values)
    if any(not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str) for pair in pairs):
        raise ValueError("catalog values must be named scalar pairs")
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("duplicate catalog field")
    return dict(pairs)


@dataclass(frozen=True, slots=True)
class DefaultCatalog:
    material: tuple[tuple[str, Scalar], ...]
    device: tuple[tuple[str, Scalar], ...]
    scaps: tuple[tuple[str, Scalar], ...]
    interface: tuple[tuple[str, Scalar], ...]
    constants: tuple[tuple[str, Scalar], ...]
    contacts: tuple[tuple[str, Scalar], ...]
    structural: tuple[tuple[str, Scalar], ...]
    defect_model: str
    defect_schema_version: str | None
    defect_degeneracy: float
    distribution_width_convention: str
    evidence: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "material", full_parameter_items(tuple(_pairs(self.material).items())))
        device = DeviceSettingsInput.model_validate(_pairs(self.device))
        if set(device.model_fields_set) != set(DeviceSettingsInput.model_fields):
            raise ValueError("device default catalog is incomplete")
        object.__setattr__(self, "device", device.normalized_items())
        scaps = full_parameter_items(tuple(_pairs(self.scaps).items()))
        if {name for name, _ in scaps} != {"D_ion", "P_lim", "P0", "B_rad", "C_n", "C_p", "alpha", "tau_n", "tau_p"}:
            raise ValueError("SCAPS adapter default catalog is incomplete")
        object.__setattr__(self, "scaps", scaps)
        interface = _pairs(self.interface)
        if set(interface) != {"calibration_factor", "iface_state_calibration_factor"}:
            raise ValueError("interface default catalog is incomplete")
        factors = tuple((name, normalize_quantity(value, "1")) for name, value in sorted(interface.items()))
        if any(value < 0 for _, value in factors):
            raise ValueError("interface factors must be nonnegative")
        object.__setattr__(self, "interface", factors)
        constants = _pairs(self.constants)
        if set(constants) != {"Q", "K_B", "T"}:
            raise ValueError("SCAPS conversion constants are incomplete")
        normalized_constants = {name: normalize_quantity(value, "1") for name, value in constants.items()}
        if any(value <= 0 for value in normalized_constants.values()):
            raise ValueError("conversion constants must be positive")
        object.__setattr__(self, "constants", tuple(sorted(normalized_constants.items())))
        contacts = _pairs(self.contacts)
        if set(contacts) != {"S_n_left", "S_p_left", "S_n_right", "S_p_right"}:
            raise ValueError("contact default catalog is incomplete")
        normalized_contacts = {name: None if value is None else normalize_quantity(value, "m/s") for name, value in contacts.items()}
        if any(value is not None and value < 0 for value in normalized_contacts.values()):
            raise ValueError("contact velocities must be nonnegative or null")
        object.__setattr__(self, "contacts", tuple(sorted(normalized_contacts.items())))
        structural = _pairs(self.structural)
        if set(structural) != {"grain_boundary_layer_role", "tandem_light_direction", "junction_incoherent"}:
            raise ValueError("structural input default catalog is incomplete")
        if not isinstance(structural["grain_boundary_layer_role"], str) or not structural["grain_boundary_layer_role"].strip():
            raise ValueError("grain-boundary default requires an explicit layer role")
        if structural["tandem_light_direction"] != "top_first" or type(structural["junction_incoherent"]) is not bool:
            raise ValueError("unsupported tandem structural defaults")
        object.__setattr__(self, "structural", tuple(sorted(structural.items())))
        if self.defect_model != "effective_lifetime" or self.defect_schema_version is not None:
            raise ValueError("undeclared microscopic defects cannot be default-enabled")
        degeneracy = normalize_quantity(self.defect_degeneracy, "1")
        if degeneracy <= 0 or self.distribution_width_convention != "not_applicable":
            raise ValueError("invalid single-level default metadata")
        object.__setattr__(self, "defect_degeneracy", degeneracy)
        evidence = tuple(self.evidence)
        if not evidence or any(not isinstance(pair, tuple) or len(pair) != 2
                               or not isinstance(pair[0], str) or not pair[0]
                               or not isinstance(pair[1], str) or re.fullmatch(r"[0-9a-f]{64}", pair[1]) is None for pair in evidence):
            raise ValueError("default catalog requires source evidence identities")
        if len({name for name, _ in evidence}) != len(evidence):
            raise ValueError("duplicate default source identity")
        object.__setattr__(self, "evidence", tuple(sorted(evidence)))

    def to_mapping(self) -> dict[str, Any]:
        return {"schema": "solarlab.default-catalog.v1",
                **{name: dict(getattr(self, name)) for name in ("material", "device", "scaps", "interface", "constants", "contacts", "structural")},
                "defect_model": self.defect_model, "defect_schema_version": self.defect_schema_version,
                "defect_degeneracy": self.defect_degeneracy,
                "distribution_width_convention": self.distribution_width_convention,
                "evidence": [list(pair) for pair in self.evidence]}

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> DefaultCatalog:
        expected = {"schema", "material", "device", "scaps", "interface", "constants", "contacts", "structural", "defect_model",
                    "defect_schema_version", "defect_degeneracy", "distribution_width_convention", "evidence"}
        if not isinstance(data, dict) or set(data) != expected or data["schema"] != "solarlab.default-catalog.v1":
            raise ValueError("invalid default catalog schema/keys")
        if any(not isinstance(data[name], dict) for name in ("material", "device", "scaps", "interface", "constants", "contacts", "structural")):
            raise ValueError("catalog sections require explicit mappings")
        return cls(material=tuple(data["material"].items()), device=tuple(data["device"].items()),
                   scaps=tuple(data["scaps"].items()), interface=tuple(data["interface"].items()),
                   constants=tuple(data["constants"].items()), contacts=tuple(data["contacts"].items()),
                   structural=tuple(data["structural"].items()), defect_model=data["defect_model"],
                   defect_schema_version=data["defect_schema_version"], defect_degeneracy=data["defect_degeneracy"],
                   distribution_width_convention=data["distribution_width_convention"],
                   evidence=tuple(tuple(pair) for pair in data["evidence"]))

    @property
    def content_sha256(self) -> str:
        # Effective values are part of identity, not merely their input hash.
        return hashlib.sha256(json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

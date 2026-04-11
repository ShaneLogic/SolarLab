from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from perovskite_sim.models.parameters import MaterialParams


@dataclass(frozen=True)
class LayerSpec:
    name: str
    thickness: float         # metres
    params: Optional[MaterialParams]
    role: str               # "ETL", "absorber", "HTL"


@dataclass(frozen=True)
class DeviceStack:
    layers: tuple[LayerSpec, ...]
    phi_left: float = 0.0   # V
    V_bi: float = 1.1       # built-in voltage [V]
    Phi: float = 2.5e21     # photon flux [m⁻² s⁻¹] (AM1.5G)
    # Interface recombination: (v_n, v_p) per internal interface [m/s].
    # interfaces[0] = interface between layers[0] and layers[1], etc.
    # Empty tuple means no interface recombination.
    interfaces: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    @property
    def phi_right(self) -> float:
        return self.phi_left + self.V_bi

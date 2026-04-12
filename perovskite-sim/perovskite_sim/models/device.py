from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
from perovskite_sim.constants import V_T
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

    def compute_V_bi(self) -> float:
        """Derive the built-in potential from the Fermi-level difference
        across the heterostack.

        Uses the numerically stable two-branch formula for majority carrier
        density: net = 0.5*(N_D - N_A), disc = sqrt(net**2 + ni**2), then
        n or p = net + disc  (or  -net + disc for minority branch).

        Falls back to the manual ``self.V_bi`` field when all layers have
        chi = Eg = 0 (backward compatibility with legacy configs).
        """
        all_zero = all(
            layer.params.chi == 0.0 and layer.params.Eg == 0.0
            for layer in self.layers
            if layer.params is not None
        )
        if all_zero:
            return self.V_bi

        left = self.layers[0].params
        right = self.layers[-1].params

        e_f_left = _fermi_level(left)
        e_f_right = _fermi_level(right)
        return e_f_left - e_f_right


def _fermi_level(p: MaterialParams) -> float:
    """Compute the Fermi level (in eV, referenced to vacuum) for a layer.

    Convention: E_F is measured as a positive energy below the vacuum level,
    so a deeper Fermi level has a larger numerical value.

    The intrinsic level sits at E_i = chi + Eg/2. Then:
    - n-type (N_D > N_A):  E_F = E_i - V_T * ln(n / ni)  (moves toward Ec)
    - p-type (N_A > N_D):  E_F = E_i + V_T * ln(p / ni)  (moves toward Ev)
    - intrinsic:           E_F = E_i

    The majority carrier density uses the numerically stable two-branch
    formula: net = 0.5*(N_D - N_A), disc = sqrt(net**2 + ni**2).
    """
    ni = p.ni
    e_i = p.chi + p.Eg / 2.0
    net = 0.5 * (p.N_D - p.N_A)
    disc = math.sqrt(net * net + ni * ni)

    if p.N_D > p.N_A:
        # n-type: majority electrons; n = net + disc
        n = net + disc
        return e_i - V_T * math.log(n / ni)
    elif p.N_A > p.N_D:
        # p-type: majority holes; p = -net + disc
        hole = -net + disc
        return e_i + V_T * math.log(hole / ni)
    else:
        # intrinsic
        return e_i

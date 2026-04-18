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
    # Device temperature [K]. Default 300 K (isothermal).
    T: float = 300.0
    # Simulation mode name; resolved to a SimulationMode by resolve_mode().
    # "full" (default) enables every physics upgrade the config supports;
    # "legacy" reproduces pre-upgrade behaviour for benchmarking.
    mode: str = "full"
    # Selective / Schottky outer contact surface recombination velocities
    # (Phase 3.3 — Apr 2026). When all four are None the contacts are
    # ohmic Dirichlet (current behaviour, bit-identical numerics). When
    # any is set the corresponding boundary uses a Robin-type flux
    # ``J = ±q · S · (n − n_eq)`` and the carrier density at the
    # boundary node is allowed to evolve. Units: m/s. ``S = 0`` is a
    # perfectly blocking contact; ``S → ∞`` recovers the ohmic limit.
    S_n_left: Optional[float] = None
    S_p_left: Optional[float] = None
    S_n_right: Optional[float] = None
    S_p_right: Optional[float] = None

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
        # Compute V_bi from the electrical-only contacts. Substrate layers
        # have no Fermi level for the drift-diffusion problem and must be
        # excluded, or a glass layer at index 0 (chi=Eg=0) would drag the
        # left contact potential to zero.
        elec = tuple(l for l in self.layers if l.role != "substrate")
        if not elec:
            return self.V_bi
        all_zero = all(
            layer.params.chi == 0.0 and layer.params.Eg == 0.0
            for layer in elec
            if layer.params is not None
        )
        if all_zero:
            return self.V_bi

        left = elec[0].params
        right = elec[-1].params

        e_f_left = _fermi_level(left)
        e_f_right = _fermi_level(right)
        return e_f_left - e_f_right


def electrical_layers(stack: "DeviceStack") -> tuple["LayerSpec", ...]:
    """Return layers that participate in the drift-diffusion solve.

    Layers with role == "substrate" are optical-only and skipped. The TMM
    optical path still walks stack.layers (full list); only the electrical
    path uses this filtered view.

    Substrate layers must form a contiguous prefix of stack.layers (or be
    entirely absent). Any substrate layer after a non-substrate layer is
    unsupported and raises ValueError, because the grid/interface indexing
    below assumes the post-filter layer order is a prefix of the full list.
    """
    seen_non_substrate = False
    for layer in stack.layers:
        if layer.role == "substrate":
            if seen_non_substrate:
                raise ValueError(
                    "substrate layers must form a contiguous prefix of "
                    "stack.layers (mid-stack or trailing substrate layers "
                    "are not supported)"
                )
        else:
            seen_non_substrate = True
    return tuple(l for l in stack.layers if l.role != "substrate")


def electrical_interfaces(
    stack: "DeviceStack",
) -> tuple[tuple[float, float], ...]:
    """Return interfaces aligned to electrical_layers (substrate excluded).

    ``stack.interfaces`` still has length ``len(stack.layers) - 1`` and is
    indexed against the *full* layer list. After filtering out a substrate
    prefix, ``electrical_layers`` drops the first ``substrate_prefix``
    layers; the first ``substrate_prefix`` interfaces therefore describe
    substrate↔substrate or substrate↔first-electrical-layer boundaries and
    have no electrical counterpart, so they must be dropped as well. All
    subsequent interfaces are kept in order.

    Assumes the contiguous-substrate-at-edge layout enforced by
    ``electrical_layers``; multi-substrate-prefix is fine, mid-stack or
    trailing substrate is rejected upstream.
    """
    # Count how many contiguous leading layers are substrate.
    substrate_prefix = 0
    for layer in stack.layers:
        if layer.role == "substrate":
            substrate_prefix += 1
        else:
            break
    elec_n = sum(1 for l in stack.layers if l.role != "substrate")
    desired = max(0, elec_n - 1)
    start = substrate_prefix
    return tuple(stack.interfaces[start : start + desired])


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

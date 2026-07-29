"""Physical constants used throughout perovskite-sim.

All values are exact SI 2019 redefinition values except EPS_0.
"""
from __future__ import annotations

Q     = 1.602176634e-19   # elementary charge [C]
K_B   = 1.380649e-23      # Boltzmann constant [J/K]
T     = 300.0             # reference temperature [K]
V_T   = K_B * T / Q      # thermal voltage at 300 K [V]  ≈ 0.025852 V
EPS_0 = 8.854187817e-12   # vacuum permittivity [F/m]
H_PLANCK = 6.62607015e-34  # Planck constant [J s]
M_E   = 9.1093837015e-31  # electron rest mass [kg]

# Richardson constant for a free electron, A* = 4 pi q m_e k_B^2 / h^3.
# Historically the per-layer default; see physics/thermionic_constants.py
# for why a layer that declares an effective density of states should
# derive its own instead.
A_STAR_FREE_ELECTRON = (
    4.0 * 3.141592653589793 * Q * M_E * K_B * K_B / (H_PLANCK ** 3)
)

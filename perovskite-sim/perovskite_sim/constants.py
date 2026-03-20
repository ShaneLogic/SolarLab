"""Physical constants used throughout perovskite-sim.

All values are exact SI 2019 redefinition values except EPS_0.
"""
from __future__ import annotations

Q     = 1.602176634e-19   # elementary charge [C]
K_B   = 1.380649e-23      # Boltzmann constant [J/K]
T     = 300.0             # reference temperature [K]
V_T   = K_B * T / Q      # thermal voltage at 300 K [V]  ≈ 0.025852 V
EPS_0 = 8.854187817e-12   # vacuum permittivity [F/m]

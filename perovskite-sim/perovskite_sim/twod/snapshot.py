from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpatialSnapshot2D:
    """Steady-state spatial fields at a given V_app on the (Ny, Nx) grid."""
    V: float
    x: np.ndarray            # (Nx,)
    y: np.ndarray            # (Ny,)
    phi: np.ndarray          # (Ny, Nx)
    n: np.ndarray            # (Ny, Nx)
    p: np.ndarray            # (Ny, Nx)
    Jx_n: np.ndarray         # (Ny, Nx-1)
    Jy_n: np.ndarray         # (Ny-1, Nx)
    Jx_p: np.ndarray         # (Ny, Nx-1)
    Jy_p: np.ndarray         # (Ny-1, Nx)

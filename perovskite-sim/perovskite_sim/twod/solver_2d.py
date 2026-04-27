from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.solver.mol import build_material_arrays as build_material_arrays_1d
from perovskite_sim.twod.grid_2d import Grid2D
from perovskite_sim.twod.poisson_2d import (
    Poisson2DFactor, build_poisson_2d_factor,
)
from perovskite_sim.twod.microstructure import Microstructure, build_tau_field


@dataclass(frozen=True)
class MaterialArrays2D:
    """2D analogue of the 1D MaterialArrays cache.

    All per-node fields are shape (Ny, Nx). For Stage A every field is a
    uniform extrusion of the 1D MaterialArrays along x; Stage B will
    override τ_n and τ_p inside grain-boundary bands.

    Field-name notes vs. 1D MaterialArrays:
    - ``D_n`` / ``D_p``: per-node (Ny, Nx) diffusion coefficients. The 1D
      cache stores only per-face values (``D_n_face``, ``D_p_face``); here
      we re-derive the per-node values from layer mu * V_T so that the 2D
      Scharfetter–Gummel scheme can form its own face averages in any
      direction.
    - ``ni``: per-node intrinsic carrier density. The 1D cache stores
      ``ni_sq`` (squared); we take the square root here.
    - ``G_optical``: always a (Ny, Nx) ndarray. When the 1D stack uses
      Beer-Lambert (``MaterialArrays.G_optical is None``), we fill zeros
      and expect the 2D RHS to generate G from Beer-Lambert itself.
    """
    grid: Grid2D
    stack: DeviceStack
    ustruct: Microstructure
    eps_r: np.ndarray
    D_n: np.ndarray
    D_p: np.ndarray
    tau_n: np.ndarray
    tau_p: np.ndarray
    N_A: np.ndarray
    N_D: np.ndarray
    ni: np.ndarray
    G_optical: np.ndarray
    n_eq_left: np.ndarray         # (Nx,)  bottom-contact electron density
    p_eq_left: np.ndarray         # (Nx,)
    n_eq_right: np.ndarray        # (Nx,)  top-contact
    p_eq_right: np.ndarray        # (Nx,)
    V_bi: float
    V_T: float
    poisson_factor: Poisson2DFactor
    layer_role_per_y: tuple[str, ...]


def build_material_arrays_2d(
    grid: Grid2D,
    stack: DeviceStack,
    ustruct: Microstructure,
    *,
    lateral_bc: str = "periodic",
) -> MaterialArrays2D:
    """Assemble the 2D MaterialArrays from a stack and a microstructure.

    Strategy: build the 1D MaterialArrays with the existing solver, then
    extrude every per-node field along x (Stage A has no x-features).
    τ_n, τ_p go through ``build_tau_field``, which respects GBs in Stage B
    but identity-extrudes when the microstructure is empty (Stage A).

    Field-name adaptations from the prescribed spec:
    - ``D_n`` / ``D_p`` are reconstructed per-node from layer mu * V_T,
      because the 1D ``MaterialArrays`` only caches per-face values
      (``D_n_face`` / ``D_p_face``).
    - ``ni`` is derived from ``sqrt(mat1d.ni_sq)``.
    - ``G_optical`` is zeroed when ``mat1d.G_optical is None`` (Beer-Lambert
      configs); the 2D RHS will supply Beer-Lambert generation instead.
    - ``V_T`` is read from ``mat1d.V_T_device`` (actual field name on 1D cache).
    """
    mat1d = build_material_arrays_1d(grid.y, stack)
    Nx, Ny = grid.Nx, grid.Ny

    def extrude(v_1d: np.ndarray) -> np.ndarray:
        """Broadcast a 1D per-y array to (Ny, Nx), returning a writeable copy."""
        return np.broadcast_to(v_1d[:, None], (Ny, Nx)).copy()

    eps_r = extrude(mat1d.eps_r)
    N_A = extrude(mat1d.N_A)
    N_D = extrude(mat1d.N_D)
    ni = extrude(np.sqrt(mat1d.ni_sq))

    # G_optical: 1D may be None for Beer-Lambert stacks; always return an array
    if mat1d.G_optical is not None:
        G_optical = extrude(mat1d.G_optical)
    else:
        G_optical = np.zeros((Ny, Nx), dtype=float)

    # D_n / D_p per-node: reconstruct from layer mu * V_T because the 1D
    # MaterialArrays only stores per-face harmonic means (D_n_face, D_p_face).
    # We compute per-node values here so the 2D SG flux can form its own
    # directional face averages.
    V_T = float(mat1d.V_T_device)
    D_n_node_1d, D_p_node_1d = _diffusion_per_node(grid.y, stack, V_T)
    D_n = extrude(D_n_node_1d)
    D_p = extrude(D_p_node_1d)

    # tau: may be a 1D array or a scalar — normalise to (Ny,) then pass to
    # build_tau_field so Stage B grain-boundary overrides work correctly.
    tau_n_1d = np.atleast_1d(mat1d.tau_n)
    tau_p_1d = np.atleast_1d(mat1d.tau_p)
    if tau_n_1d.size == 1:
        tau_n_1d = np.full(Ny, float(tau_n_1d[0]))
    if tau_p_1d.size == 1:
        tau_p_1d = np.full(Ny, float(tau_p_1d[0]))

    layer_role_per_y = tuple(_layer_role_at_each_y(grid.y, stack))
    tau_n, tau_p = build_tau_field(
        grid, ustruct,
        tau_n_bulk_per_y=tau_n_1d,
        tau_p_bulk_per_y=tau_p_1d,
        layer_role_per_y=layer_role_per_y,
    )

    # Boundary equilibrium concentrations — scalars on the 1D side.
    # Broadcast to length Nx for x-uniform contacts.
    n_eq_left = np.full((Nx,), float(mat1d.n_L))
    p_eq_left = np.full((Nx,), float(mat1d.p_L))
    n_eq_right = np.full((Nx,), float(mat1d.n_R))
    p_eq_right = np.full((Nx,), float(mat1d.p_R))

    poisson_factor = build_poisson_2d_factor(grid, eps_r, lateral_bc=lateral_bc)

    # V_bi: use the computed V_bi_eff (from band offsets) stored on the 1D cache
    V_bi = float(mat1d.V_bi_eff)

    return MaterialArrays2D(
        grid=grid, stack=stack, ustruct=ustruct,
        eps_r=eps_r, D_n=D_n, D_p=D_p,
        tau_n=tau_n, tau_p=tau_p,
        N_A=N_A, N_D=N_D, ni=ni, G_optical=G_optical,
        n_eq_left=n_eq_left, p_eq_left=p_eq_left,
        n_eq_right=n_eq_right, p_eq_right=p_eq_right,
        V_bi=V_bi, V_T=V_T,
        poisson_factor=poisson_factor,
        layer_role_per_y=layer_role_per_y,
    )


def _diffusion_per_node(
    y: np.ndarray,
    stack: DeviceStack,
    V_T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild per-node D_n, D_p from layer mobility × thermal voltage.

    The 1D MaterialArrays only stores per-face harmonic-mean diffusion
    coefficients (D_n_face, D_p_face, length Ny-1). The 2D solver needs
    per-node values (length Ny) to form face averages in both x and y.
    We replicate the same mu * V_T construction used inside
    build_material_arrays, without temperature scaling for now (Stage A).
    """
    from perovskite_sim.physics.temperature import mu_at_T, thermal_voltage
    from perovskite_sim.models.mode import resolve_mode

    sim_mode = resolve_mode(getattr(stack, "mode", "full"))
    if sim_mode.use_temperature_scaling:
        T_dev = stack.T
        V_T_actual = thermal_voltage(T_dev)
    else:
        T_dev = 300.0
        V_T_actual = V_T  # already computed by caller from mat1d.V_T_device

    Ny = len(y)
    D_n_node = np.empty(Ny)
    D_p_node = np.empty(Ny)

    elec = electrical_layers(stack)
    offset = 0.0
    for layer in elec:
        mask = (y >= offset - 1e-12) & (y <= offset + layer.thickness + 1e-12)
        p = layer.params
        mu_n = mu_at_T(p.mu_n, T_dev, p.mu_T_gamma)
        mu_p = mu_at_T(p.mu_p, T_dev, p.mu_T_gamma)
        D_n_node[mask] = mu_n * V_T_actual
        D_p_node[mask] = mu_p * V_T_actual
        offset += layer.thickness

    return D_n_node, D_p_node


def _layer_role_at_each_y(y: np.ndarray, stack: DeviceStack) -> list[str]:
    """Return the layer role string at each y-node.

    Uses ``electrical_layers()`` (substrate filtered) so the cumulative
    thickness boundaries match the drift-diffusion grid built by
    ``multilayer_grid``.
    """
    layers = electrical_layers(stack)
    boundaries = [0.0]
    for L in layers:
        boundaries.append(boundaries[-1] + L.thickness)

    roles: list[str] = []
    for y_node in y:
        for k, L in enumerate(layers):
            if y_node <= boundaries[k + 1] + 1e-15:
                roles.append(getattr(L, "role", "absorber"))
                break
        else:
            roles.append(getattr(layers[-1], "role", "absorber"))
    return roles

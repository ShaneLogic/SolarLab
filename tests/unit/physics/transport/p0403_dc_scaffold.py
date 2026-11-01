"""Test-local DC drift-diffusion scaffold for the P04-03 acceptance tests.

This is NOT a production solver (the equilibrium/steady Newton belongs to
P04-04). It only composes the P04-03 kernels into a 1D DC system so the V1
acceptance can be exercised end to end:

* bands: :func:`solarlab.physics.statistics.layered_band_structure`
  (DOS folded, right-layer ownership);
* carrier balance: :class:`solarlab.contracts.TermSink` over the
  ``INTEGER_ENDPOINT_V1`` control volumes with the N-class
  :func:`~solarlab.physics.transport.electrochemical_flux.carrier_particle_fluxes`;
* contacts: fixed reservoirs (Dirichlet rows) or blocking/Robin channels with
  :func:`~solarlab.physics.contacts.boundary_flux.boundary_particle_flux`;
* Poisson: :class:`solarlab.physics.electrostatics.PoissonOperator1D` with the
  semiconductor work-function contact potential.

The unknowns are ``(phi, ln n, ln p)``; the Jacobian is assembled from the
analytic face-flux derivatives, so quadratic convergence is itself a check of
those derivatives. Recombination is a test-local SRH law (P04-07 owns the
production one).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from solarlab.contracts import (
    AREA, LENGTH, PARTICLE, SECOND, VOLUME, CellSource, EquationSpec, FaceFlux, Layout,
    Support, TermSink,
)
from solarlab.mesh import INTEGER_ENDPOINT_V1, RIGHT_LAYER_WINS, build_geometry, multilayer_grid
from solarlab.mesh.grid import layer_node_masks
from solarlab.physics.contacts import (
    SEMICONDUCTOR_WORK_FUNCTION, BuiltInPotentialSources, CarrierExchangeChannel,
    ContactMaterial, doping_equilibrium_densities,
)
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.physics.electrostatics import PoissonOperator1D
from solarlab.physics.statistics import (
    BandStructure, LayerBandInput, layered_band_structure, thermal_voltage,
)
from solarlab.physics.transport import legacy_face_diffusivity
from solarlab.physics.transport.electrochemical_flux import carrier_particle_fluxes

Q = 1.602176634e-19
FLUX_UNIT = PARTICLE / (AREA * SECOND)
SOURCE_UNIT = PARTICLE / (VOLUME * SECOND)
assert SOURCE_UNIT * LENGTH == FLUX_UNIT


@dataclass(frozen=True)
class ScaffoldLayer:
    thickness_m: float
    intervals: int
    chi_eV: float
    Eg_eV: float
    Nc_m3: float
    Nv_m3: float
    N_D_m3: float
    N_A_m3: float
    D_n_m2_s: float
    D_p_m2_s: float
    eps_r: float
    generation_m3_s: float = 0.0
    tau_s: float = 1.0e-6
    role: str = ""


# p-HTL / intrinsic absorber / n-ETL heterojunction with DOS contrasts.
PIN_LAYERS = (
    ScaffoldLayer(8.0e-8, 16, 3.6, 1.8, 2.0e25, 3.0e25, 0.0, 1.0e24, 1.0e-6, 1.0e-6, 3.0,
                  role="HTL"),
    ScaffoldLayer(3.0e-7, 40, 3.9, 1.6, 1.0e25, 1.0e25, 0.0, 0.0, 5.0e-5, 5.0e-5, 25.0,
                  generation_m3_s=1.0e27, tau_s=1.0e-6, role="absorber"),
    ScaffoldLayer(8.0e-8, 16, 4.0, 2.0, 4.0e25, 2.0e25, 1.0e24, 0.0, 1.0e-6, 1.0e-6, 9.0,
                  role="ETL"),
)


@dataclass
class ScaffoldDevice:
    layers: tuple[ScaffoldLayer, ...]
    x: np.ndarray
    bands: BandStructure
    V_T: float
    Nc: np.ndarray
    Nv: np.ndarray
    N_D: np.ndarray
    N_A: np.ndarray
    ni_sq: np.ndarray
    generation: np.ndarray
    tau: np.ndarray
    D_n_face: np.ndarray
    D_p_face: np.ndarray
    poisson: PoissonOperator1D
    widths: np.ndarray
    V_bi_bc: float
    fermi_level_eV: float
    n_eq: tuple[float, float]
    p_eq: tuple[float, float]
    channels: dict[str, CarrierExchangeChannel] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.x.size)

    @property
    def generation_limit_A_m2(self) -> float:
        """``q * sum(w G)``: the photocurrent if every generated pair is collected."""
        return float(Q * np.sum(self.widths * self.generation))


def build_device(layers: Sequence[ScaffoldLayer] = PIN_LAYERS, *, fold_dos: bool = True,
                 temperature_K: float = 300.0,
                 velocities: dict[str, float | None] | None = None) -> ScaffoldDevice:
    layers = tuple(layers)
    x = multilayer_grid([(layer.thickness_m, layer.intervals) for layer in layers])
    thicknesses = [layer.thickness_m for layer in layers]
    masks = layer_node_masks(x, thicknesses, RIGHT_LAYER_WINS)
    V_T = thermal_voltage(temperature_K)

    def paint(name: str) -> np.ndarray:
        values = np.zeros(x.shape)
        for layer, mask in zip(layers, masks):
            values[mask] = getattr(layer, name)
        return values

    bands = layered_band_structure(
        x, [LayerBandInput(layer.thickness_m, layer.chi_eV, layer.Eg_eV, layer.Nc_m3,
                           layer.Nv_m3, layer.role) for layer in layers],
        thermal_voltage_V=V_T, fold_dos=fold_dos)
    Nc, Nv, Eg = paint("Nc_m3"), paint("Nv_m3"), bands.physical_band_gap_eV
    ni_sq = Nc * Nv * np.exp(-Eg / V_T)
    geometry = build_geometry(INTEGER_ENDPOINT_V1, x, np.cumsum(thicknesses)[:-1])
    poisson = PoissonOperator1D.from_harmonic_nodes(x, paint("eps_r"),
                                                    geometry_name=INTEGER_ENDPOINT_V1)

    def material(layer: ScaffoldLayer) -> ContactMaterial:
        return ContactMaterial(temperature_K, layer.chi_eV, layer.Eg_eV, layer.Nc_m3,
                               layer.Nv_m3, layer.N_D_m3, layer.N_A_m3)

    sources = BuiltInPotentialSources(mode=SEMICONDUCTOR_WORK_FUNCTION, manual_V_bi=0.0,
                                      left_material=material(layers[0]),
                                      right_material=material(layers[-1]))
    from solarlab.physics.contacts import semiconductor_work_function
    W_left = semiconductor_work_function(material(layers[0]))
    N_D, N_A = paint("N_D_m3"), paint("N_A_m3")
    n_left, p_left = doping_equilibrium_densities(N_D[0], N_A[0], ni_sq[0])
    n_right, p_right = doping_equilibrium_densities(N_D[-1], N_A[-1], ni_sq[-1])
    device = ScaffoldDevice(
        layers=layers, x=x, bands=bands, V_T=V_T, Nc=Nc, Nv=Nv, N_D=N_D, N_A=N_A,
        ni_sq=ni_sq, generation=paint("generation_m3_s"), tau=paint("tau_s"),
        D_n_face=legacy_face_diffusivity(paint("D_n_m2_s")),
        D_p_face=legacy_face_diffusivity(paint("D_p_m2_s")),
        poisson=poisson, widths=geometry.cell_measures_m.copy(),
        V_bi_bc=sources.poisson_potential(), fermi_level_eV=-W_left,
        n_eq=(n_left, n_right), p_eq=(p_left, p_right),
    )
    velocities = velocities or {}
    for carrier, side in (("n", "left"), ("p", "left"), ("n", "right"), ("p", "right")):
        index = 0 if side == "left" else 1
        reservoir = (device.n_eq if carrier == "n" else device.p_eq)[index]
        device.channels[carrier + "_" + side] = CarrierExchangeChannel(
            carrier, side, velocities.get(carrier + "_" + side), reservoir)
    return device


def equilibrium_densities(device: ScaffoldDevice, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boltzmann densities of one common Fermi level from the physical band edges."""
    chi = device.bands.physical_electron_affinity_eV
    gap = device.bands.physical_band_gap_eV
    E_F = device.fermi_level_eV
    n = device.Nc * np.exp((E_F + phi + chi) / device.V_T)
    p = device.Nv * np.exp(-(E_F + phi + chi + gap) / device.V_T)
    return n, p


def _charge(device: ScaffoldDevice, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    return Q * (p - n + device.N_D - device.N_A)


def solve_equilibrium(device: ScaffoldDevice, *, iterations: int = 200) -> np.ndarray:
    """Poisson-Boltzmann equilibrium potential (dense damped Newton on phi)."""
    x = device.x
    phi = device.V_bi_bc * (x - x[0]) / (x[-1] - x[0])
    C = device.poisson.face_capacitance_F_m2
    widths = device.poisson.interior_widths_m
    for _ in range(iterations):
        n, p = equilibrium_densities(device, phi)
        residual = device.poisson.residual(phi, _charge(device, n, p), 0.0, device.V_bi_bc)
        jacobian = device.poisson.jacobian().toarray()
        drho = -Q * (n + p) / device.V_T
        jacobian[np.arange(1, device.size - 1), np.arange(1, device.size - 1)] += widths * drho[1:-1]
        step = np.linalg.solve(jacobian, -residual)
        largest = np.max(np.abs(step))
        phi = phi + step * min(1.0, 0.1 / largest) if largest > 0 else phi
        if largest < 1e-14:
            return phi
    raise AssertionError("scaffold Poisson-Boltzmann did not converge; C=%r" % (C[:1],))


@dataclass
class DcState:
    phi: np.ndarray
    log_n: np.ndarray
    log_p: np.ndarray

    @property
    def n(self) -> np.ndarray:
        return np.exp(self.log_n)

    @property
    def p(self) -> np.ndarray:
        return np.exp(self.log_p)

    def vector(self) -> np.ndarray:
        return np.concatenate((self.phi, self.log_n, self.log_p))

    @classmethod
    def from_vector(cls, values: np.ndarray, size: int) -> DcState:
        return cls(values[:size].copy(), values[size:2 * size].copy(), values[2 * size:].copy())


def equilibrium_state(device: ScaffoldDevice) -> DcState:
    phi = solve_equilibrium(device)
    n, p = equilibrium_densities(device, phi)
    return DcState(phi, np.log(n), np.log(p))


def _srh(device: ScaffoldDevice, n: np.ndarray, p: np.ndarray):
    ni = np.sqrt(device.ni_sq)
    denominator = device.tau * (n + ni) + device.tau * (p + ni)
    excess = n * p - device.ni_sq
    rate = excess / denominator
    d_n = (p * denominator - excess * device.tau) / denominator**2
    d_p = (n * denominator - excess * device.tau) / denominator**2
    return rate, d_n, d_p


def _layout(size: int) -> Layout:
    supports = (Support("cells", "cell", (size,)), Support("faces", "face", (size - 1,)))
    equations = (
        EquationSpec("electron", "electron", "cells", (size,), FLUX_UNIT),
        EquationSpec("hole", "hole", "cells", (size,), FLUX_UNIT),
    )
    return Layout(supports, (), equations)


def residual_and_jacobian(device: ScaffoldDevice, state: DcState, *, V_app: float,
                          generation_scale: float):
    """Scaled-free DC residual ``[Poisson, electron, hole]`` and its Jacobian."""
    size = device.size
    n, p = state.n, state.p
    fluxes = carrier_particle_fluxes(device.x, state.phi, n, p, device.D_n_face,
                                     device.D_p_face, device.V_T, device.bands)
    G = generation_scale * device.generation
    R, dR_dn, dR_dp = _srh(device, n, p)
    geometry = build_geometry(INTEGER_ENDPOINT_V1, device.x)
    contract = geometry.to_contract_geometry()
    layout = _layout(size)
    rows = {}
    for carrier, flux in (("electron", fluxes.electron), ("hole", fluxes.hole)):
        sink = TermSink(layout, carrier, contract)
        sink.add(FaceFlux(carrier + ".sg", "faces", flux.flux, FLUX_UNIT))
        sink.add(CellSource(carrier + ".net_generation", "cells", G - R, SOURCE_UNIT))
        rows[carrier] = np.array(sink.value(), copy=True)

    residual = np.zeros(3 * size)
    jacobian = np.zeros((3 * size, 3 * size))
    phi_cols = np.arange(size)
    n_cols = size + np.arange(size)
    p_cols = 2 * size + np.arange(size)

    # Poisson rows.
    rho = _charge(device, n, p)
    phi_right = device.V_bi_bc - (-1.0 if device.V_bi_bc < 0.0 else 1.0) * V_app
    residual[:size] = device.poisson.residual(state.phi, rho, 0.0, phi_right)
    jacobian[:size, :size] = device.poisson.jacobian().toarray()
    interior = np.arange(1, size - 1)
    widths = device.poisson.interior_widths_m
    jacobian[interior, n_cols[interior]] = -Q * widths * n[1:-1]
    jacobian[interior, p_cols[interior]] = Q * widths * p[1:-1]

    faces = np.arange(size - 1)
    for offset, carrier, flux, density, columns, d_rate, other_columns, d_rate_other in (
        (size, "electron", fluxes.electron, n, n_cols, dR_dn, p_cols, dR_dp),
        (2 * size, "hole", fluxes.hole, p, p_cols, dR_dp, n_cols, dR_dn),
    ):
        block = rows[carrier]
        rows_index = offset + np.arange(size)
        # Face flux removes from the left cell and adds to the right cell.
        for sign, cell in ((-1.0, faces), (1.0, faces + 1)):
            np.add.at(jacobian, (rows_index[cell], columns[faces]),
                      sign * flux.left_density_derivative * density[:-1])
            np.add.at(jacobian, (rows_index[cell], columns[faces + 1]),
                      sign * flux.right_density_derivative * density[1:])
            np.add.at(jacobian, (rows_index[cell], phi_cols[faces]),
                      sign * flux.left_potential_derivative)
            np.add.at(jacobian, (rows_index[cell], phi_cols[faces + 1]),
                      sign * flux.right_potential_derivative)
        jacobian[rows_index, columns] -= device.widths * d_rate * density
        jacobian[rows_index, other_columns] -= device.widths * d_rate_other * (
            p if carrier == "electron" else n)
        for side, node in (("left", 0), ("right", size - 1)):
            channel = device.channels[("n" if carrier == "electron" else "p") + "_" + side]
            if channel.kind == "fixed_reservoir":
                jacobian[rows_index[node], :] = 0.0
                jacobian[rows_index[node], columns[node]] = 1.0
                block[node] = np.log(density[node]) - np.log(channel.reservoir_density_m3)
            else:
                exchange = boundary_particle_flux(channel, density[node])
                # Inflow through the contact face: +F at the left, -F at the right.
                inflow_sign = 1.0 if side == "left" else -1.0
                block[node] += inflow_sign * exchange.flux
                jacobian[rows_index[node], columns[node]] += (
                    inflow_sign * exchange.density_derivative * density[node])
        residual[offset:offset + size] = block
    return residual, jacobian


def solve_dc(device: ScaffoldDevice, state: DcState, *, V_app: float = 0.0,
             generation_scale: float = 1.0, iterations: int = 80) -> tuple[DcState, int]:
    """Damped Newton on ``(phi, ln n, ln p)``; returns the state and iteration count."""
    vector = state.vector()
    size = device.size
    for iteration in range(1, iterations + 1):
        residual, jacobian = residual_and_jacobian(
            device, DcState.from_vector(vector, size), V_app=V_app,
            generation_scale=generation_scale)
        scale = 1.0 / np.max(np.abs(jacobian), axis=1)
        step = np.linalg.solve(jacobian * scale[:, None], -residual * scale)
        largest_phi = np.max(np.abs(step[:size]))
        largest_log = np.max(np.abs(step[size:]))
        damping = min(1.0, 0.2 / max(largest_phi, 1e-300), 2.0 / max(largest_log, 1e-300))
        vector = vector + damping * step
        if damping == 1.0 and largest_phi < 1e-13 and largest_log < 1e-11:
            return DcState.from_vector(vector, size), iteration
    raise AssertionError("scaffold DC Newton did not converge")


def illuminated_state(device: ScaffoldDevice, *, V_app: float = 0.0) -> DcState:
    """Continuation in the generation scale from the dark equilibrium."""
    state = equilibrium_state(device)
    for scale in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        state, _ = solve_dc(device, state, V_app=V_app, generation_scale=scale)
    return state


def terminal_current_density(device: ScaffoldDevice, state: DcState) -> np.ndarray:
    """Total carrier current ``J_n + J_p`` [A/m^2] on every face, along +x."""
    fluxes = carrier_particle_fluxes(device.x, state.phi, state.n, state.p, device.D_n_face,
                                     device.D_p_face, device.V_T, device.bands)
    return fluxes.electron_current() + fluxes.hole_current()


def contact_currents(device: ScaffoldDevice, state: DcState) -> tuple[float, float]:
    """Current through the left and right contact faces (Robin/blocking channels)."""
    totals = []
    for side, node in (("left", 0), ("right", device.size - 1)):
        total = 0.0
        for carrier, density, charge in (("n", state.n, -Q), ("p", state.p, Q)):
            channel = device.channels[carrier + "_" + side]
            if channel.kind == "fixed_reservoir":
                return (float("nan"), float("nan"))
            total += charge * boundary_particle_flux(channel, density[node]).flux
        totals.append(total)
    return totals[0], totals[1]

"""Ion-free classical 1D Poisson and carrier continuity.

Coordinates are u = (phi, z_n, z_p), with dimensionless quasi-Fermi offsets
z_n and z_p are measured from each carrier's majority electrode Fermi level.
At zero bias both references equal E_F0. Physical densities
come from one common equilibrium reference and the physical bands. SG faces
consume the differences of z directly, so a constant Fermi level gives an
exactly zero flux even across DOS and band offsets.

The implicit residual is F(u, udot) = transport + source - d(storage)/dt.
Only steady operation is admitted by the production driver. The explicit
storage partials retain that architecture; they are not transient qualification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, diags, lil_matrix

from solarlab.contracts import (
    AREA,
    LENGTH,
    PARTICLE,
    SECOND,
    VOLUME,
    CellSource,
    EquationSpec,
    FaceFlux,
    Geometry,
    Layout,
    Support,
    TermSink,
    frozen_array,
)
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.stable_bernoulli import bernoulli_pair
from solarlab.equations.field_transport_1d import FieldTransport1D
from solarlab.mesh.geometry import ControlVolumeGeometry1D
from solarlab.physics.band_structure.binding import ParameterBinding
from solarlab.physics.contacts import CarrierExchangeChannel, junction_polarity
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.physics.electrostatics import PoissonOperator1D
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger
from solarlab.physics.generation.profile import GenerationProfile
from solarlab.physics.recombination.bulk import (
    RecombinationRate,
    sum_recombination_rates,
)
from solarlab.physics.recombination.reaction import BulkRecombinationReaction
from solarlab.physics.transport.electrochemical_flux import (
    CarrierFaceFlux,
    CarrierFaceFluxes,
    carrier_particle_fluxes,
)

__all__ = [
    "CapabilityError",
    "CompiledDevice1D",
    "DC1DProblem",
    "LocalRecombinationModel",
    "State1D",
]

Vector: TypeAlias = NDArray[np.float64]
Q = ELEMENTARY_CHARGE_C
FLUX_UNIT = PARTICLE / (AREA * SECOND)
SOURCE_UNIT = PARTICLE / (VOLUME * SECOND)
assert SOURCE_UNIT * LENGTH == FLUX_UNIT


class CapabilityError(ValueError):
    """An enabled mechanism has no binding in this first 1D DC slice."""


class LocalRecombinationModel(Protocol):
    """An injected local net pair rate and its exact density partials.

    The model owns its bound coefficients and a stable, parameter-specific
    model_id. Its evaluate(n, p) returns production RecombinationRate arrays
    of shape (N,); no file loader, registry, or solver dependency is required.
    """

    model_id: str

    def evaluate(self, n: Vector, p: Vector) -> RecombinationRate: ...


@dataclass(frozen=True)
class State1D:
    """Immutable coordinates and physical fields on one compiled device."""

    u: Vector
    potential_V: Vector
    electron_density_m3: Vector
    hole_density_m3: Vector
    device_sha256: str
    voltage_V: float = 0.0

    def __post_init__(self) -> None:
        for name in ("u", "potential_V", "electron_density_m3", "hole_density_m3"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        n = self.potential_V.size
        if self.potential_V.ndim != 1:
            raise ValueError("state potential must be a vector on the nodes")
        if self.u.shape != (3 * n,):
            raise ValueError("state coordinates must have shape (3 * node_count,)")
        for density in (self.electron_density_m3, self.hole_density_m3):
            if density.shape != (n,) or np.any(density <= 0.0):
                raise ValueError("state densities must be positive on every node")
        if not self.device_sha256:
            raise ValueError("a state must identify its compiled device")
        if not math.isfinite(self.voltage_V):
            raise ValueError("the state coordinate reference voltage must be finite")


@dataclass(frozen=True)
class CompiledDevice1D:
    """Cold-compiled geometry, physical models, and contact reservoirs."""

    device_id: str
    sha256: str
    geometry: ControlVolumeGeometry1D
    parameters: ParameterBinding
    poisson: PoissonOperator1D
    electron_diffusivity_faces_m2_s: Vector
    hole_diffusivity_faces_m2_s: Vector
    generation: GenerationProfile
    recombination: BulkRecombinationReaction
    channels: tuple[CarrierExchangeChannel, ...]
    phi_left_V: float
    built_in_potential_V: float
    fermi_level_eV: float
    reference_potential_V: Vector
    reference_electron_density_m3: Vector
    reference_hole_density_m3: Vector
    additional_recombination: LocalRecombinationModel | None = None
    capabilities: tuple[str, ...] = (
        "ion_free",
        "classical_1d_dc",
        "maxwell_boltzmann",
        "semiconductor_work_function_contacts",
        "bulk_srh_radiative_auger",
    )
    field_transport: FieldTransport1D | None = None
    _carrier_layout: Layout = field(init=False, repr=False)
    _carrier_geometry: Geometry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "electron_diffusivity_faces_m2_s",
            "hole_diffusivity_faces_m2_s",
            "reference_potential_V",
            "reference_electron_density_m3",
            "reference_hole_density_m3",
        ):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        count = self.node_count
        for name in (
            "reference_potential_V",
            "reference_electron_density_m3",
            "reference_hole_density_m3",
        ):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} does not match the geometry")
        for name in ("electron_diffusivity_faces_m2_s", "hole_diffusivity_faces_m2_s"):
            if getattr(self, name).shape != (count - 1,):
                raise ValueError(f"{name} does not match the geometry faces")
        transport = self.field_transport
        if transport is not None and (
            not np.array_equal(transport.nodes_m, self.nodes_m)
            or transport.thermal_voltage_V != self.thermal_voltage_V
            or not np.array_equal(transport.base_electron_diffusivity_m2_s, self.electron_diffusivity_faces_m2_s)
            or not np.array_equal(transport.base_hole_diffusivity_m2_s, self.hole_diffusivity_faces_m2_s)
        ):
            raise ValueError("field transport must bind this device's grid, temperature and low-field diffusivities")
        for name in ("reference_electron_density_m3", "reference_hole_density_m3"):
            if np.any(getattr(self, name) <= 0.0):
                raise ValueError("all reference densities must be finite and positive")
        if self.poisson.node_count != count or self.recombination.node_count != count:
            raise ValueError("all compiled operators must share the node count")
        expected = {(carrier, side) for carrier in ("n", "p") for side in ("left", "right")}
        if {(channel.carrier, channel.side) for channel in self.channels} != expected:
            raise ValueError("the compiled device needs all four contact channels")
        if len(self.channels) != 4:
            raise ValueError("duplicate contact channel")
        self.generation.require_inputs(self.generation.binding_sha256, self.geometry)
        if self.additional_recombination is not None:
            if not isinstance(self.additional_recombination.model_id, str) or not self.additional_recombination.model_id:
                raise ValueError("an additional recombination law needs a model_id")
        layout = Layout(
            (Support("cells", "cell", (count,)), Support("faces", "face", (count - 1,))),
            (),
            (
                EquationSpec("electron", "electron", "cells", (count,), FLUX_UNIT),
                EquationSpec("hole", "hole", "cells", (count,), FLUX_UNIT),
            ),
        )
        object.__setattr__(self, "_carrier_layout", layout)
        object.__setattr__(self, "_carrier_geometry", self.geometry.to_contract_geometry())

    @property
    def node_count(self) -> int:
        return self.geometry.node_count

    @property
    def identity(self) -> str:
        """Identity of the actual geometry and physical/model bindings."""
        return self.sha256

    @property
    def nodes_m(self) -> Vector:
        return self.geometry.nodes_m

    @property
    def thermal_voltage_V(self) -> float:
        return self.parameters.thermal_voltage_V

    @property
    def voltage_polarity(self) -> float:
        return junction_polarity(self.built_in_potential_V)

    @property
    def generation_limit_A_m2(self) -> float:
        return Q * self.generation.areal_generation_m2_s

    def contact_potentials(self, voltage_V: float) -> tuple[float, float]:
        voltage = float(voltage_V)
        if not math.isfinite(voltage):
            raise ValueError("applied voltage must be finite")
        return (
            self.phi_left_V,
            self.phi_left_V + self.built_in_potential_V - self.voltage_polarity * voltage,
        )

    def channel(self, carrier: str, side: str) -> CarrierExchangeChannel:
        return next(c for c in self.channels if c.carrier == carrier and c.side == side)

    def carrier_reference_energies(self, voltage_V: float) -> tuple[float, float]:
        """Majority-electrode Fermi offsets from the common dark reference."""
        bias = self.voltage_polarity * float(voltage_V)
        return (bias, 0.0) if self.voltage_polarity > 0.0 else (0.0, bias)

    def carrier_reference_potentials(self, voltage_V: float) -> tuple[Vector, Vector]:
        """Shift before subtraction so the majority-contact exponent is exact zero."""
        electron, hole = self.carrier_reference_energies(voltage_V)
        return self.reference_potential_V - electron, self.reference_potential_V - hole

    def rebase_coordinates(self, u: ArrayLike, old_voltage_V: float, new_voltage_V: float) -> Vector:
        """Preserve the physical state when its electrode reference changes."""
        values = np.array(u, dtype=float, copy=True)
        old = self.carrier_reference_energies(old_voltage_V)
        new = self.carrier_reference_energies(new_voltage_V)
        count = self.node_count
        for offset, before, after in zip((count, 2 * count), old, new):
            values[offset:offset + count] += (before - after) / self.thermal_voltage_V
        return values

    def fields(self, u: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector]:
        values = np.asarray(u, dtype=float)
        count = self.node_count
        if values.shape != (3 * count,) or not np.all(np.isfinite(values)):
            raise ValueError("coordinates must be finite with shape (3 * node_count,)")
        phi = values[:count]
        reference_n, reference_p = self.carrier_reference_potentials(voltage_V)
        shift_n = (phi - reference_n) / self.thermal_voltage_V
        shift_p = (phi - reference_p) / self.thermal_voltage_V
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            n = self.reference_electron_density_m3 * np.exp(shift_n + values[count:2 * count])
            p = self.reference_hole_density_m3 * np.exp(-shift_p - values[2 * count:])
        return phi, n, p

    def state(self, u: ArrayLike, *, voltage_V: float = 0.0) -> State1D:
        phi, n, p = self.fields(u, voltage_V=voltage_V)
        return State1D(np.asarray(u, dtype=float), phi, n, p, self.sha256, voltage_V)

    def charge_ledger(self, n: Vector, p: Vector) -> ChargeLedger:
        """Poisson reads the same explicit population ledger as its tangents."""
        return ChargeLedger(
            self.node_count,
            len(self.geometry.interfaces),
            (
                ChargeEntry("hole", "hole", 1, p),
                ChargeEntry("electron", "electron", -1, n),
                ChargeEntry("acceptor", "dopant", -1, self.parameters.acceptor_density_m3),
                ChargeEntry("donor", "dopant", 1, self.parameters.donor_density_m3),
            ),
        )


def _quasi_fermi_potential_tangents(
    flux: CarrierFaceFlux, drive: Vector, carrier_sign: float, thermal: float
) -> tuple[Vector, Vector]:
    """Chain the SG kernel through density(phi, z) without equilibrium cancellation."""
    pair = bernoulli_pair(flux.energy_step)
    forward = np.divide(
        pair.forward_derivative, pair.forward,
        out=np.zeros_like(flux.flux), where=pair.forward != 0.0,
    )
    backward = np.divide(
        pair.backward_derivative, pair.backward,
        out=np.zeros_like(flux.flux), where=pair.backward != 0.0,
    )
    # An underflowed B(x), x >> 1, still has d(log B)/dx = 1/x - 1.
    tiny_forward = pair.forward == 0.0
    tiny_backward = pair.backward == 0.0
    forward[tiny_forward] = 1.0 / flux.energy_step[tiny_forward] - 1.0
    backward[tiny_backward] = -1.0 / flux.energy_step[tiny_backward] - 1.0
    from_left = drive < 0.0
    left_factor = np.where(from_left, 1.0 + forward, -backward)
    right_factor = np.where(from_left, -forward, 1.0 + backward)
    factor = carrier_sign * flux.flux / thermal
    return factor * left_factor, factor * right_factor


@dataclass(frozen=True)
class DC1DProblem:
    """Production F(u, udot), Fy, and Fudot for one operating point."""

    device: CompiledDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0

    def __post_init__(self) -> None:
        self.device.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and non-negative")

    def _derivative(self, udot: ArrayLike | None) -> Vector:
        if udot is None:
            return np.zeros(3 * self.device.node_count)
        derivative = np.asarray(udot, dtype=float)
        if derivative.shape != (3 * self.device.node_count,) or not np.all(np.isfinite(derivative)):
            raise ValueError("udot must be finite and match the coordinate vector")
        return derivative

    def fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        device = self.device
        count = device.node_count
        values = np.asarray(u, dtype=float)
        phi, n, p = device.fields(values, voltage_V=self.voltage_V)
        electron_diffusivity = device.electron_diffusivity_faces_m2_s
        hole_diffusivity = device.hole_diffusivity_faces_m2_s
        if device.field_transport is not None:
            transport = device.field_transport.evaluate(phi)
            electron_diffusivity = transport.electron_diffusivity_m2_s
            hole_diffusivity = transport.hole_diffusivity_m2_s
        return carrier_particle_fluxes(
            device.nodes_m, phi, n, p,
            electron_diffusivity, hole_diffusivity,
            device.thermal_voltage_V,
            device.parameters.band_structure,
            electron_electrochemical_step=np.diff(values[count:2 * count]),
            hole_electrochemical_step=-np.diff(values[2 * count:]),
        )

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        evaluation = self.device.recombination.evaluate(n, p, derivatives=derivatives)
        evaluation.require_domain()
        total = evaluation.total
        model = self.device.additional_recombination
        if model is not None:
            extra = model.evaluate(n, p)
            if not isinstance(extra, RecombinationRate) or not extra.has_derivatives:
                raise ValueError("a local recombination model must return rate and both exact partials")
            for value in (extra.rate, extra.d_rate_dn, extra.d_rate_dp):
                array = np.asarray(value)
                if array.shape != n.shape or not np.all(np.isfinite(array)):
                    raise ValueError("local recombination outputs must be finite on every node")
            total = sum_recombination_rates(total, extra)
        return total

    def _contact_exchange(self, u: Vector, channel: CarrierExchangeChannel, node: int) -> tuple[float, float]:
        """Evaluate the production contact slope on an accurate density excess."""
        count = self.device.node_count
        references = self.device.carrier_reference_potentials(self.voltage_V)
        reference = references[0] if channel.carrier == "n" else references[1]
        shift = (u[node] - reference[node]) / self.device.thermal_voltage_V
        exponent = shift + u[count + node] if channel.carrier == "n" else -shift - u[2 * count + node]
        boundary = boundary_particle_flux(channel, channel.reservoir_density_m3)
        excess = channel.reservoir_density_m3 * np.expm1(exponent)
        # The boundary kernel supplies n_out * S, including exactly zero for blocking.
        return boundary.density_derivative * excess, boundary.density_derivative

    def residual(self, u: ArrayLike, udot: ArrayLike | None = None) -> Vector:
        device = self.device
        count = device.node_count
        values = np.asarray(u, dtype=float)
        phi, n, p = device.fields(values, voltage_V=self.voltage_V)
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)) or np.any(n <= 0.0) or np.any(p <= 0.0):
            return np.full(3 * count, np.nan)
        derivative = self._derivative(udot)
        fluxes = self.fluxes(values)
        rate = self.recombination_rate(n, p, derivatives=False)
        source = self.generation_scale * device.generation.source_m3_s - rate.rate
        result = np.empty(3 * count)
        left, right = device.contact_potentials(self.voltage_V)
        rho = device.charge_ledger(n, p).space_charge_C_m3()
        result[:count] = device.poisson.residual(phi, rho, left, right)
        geometry = device._carrier_geometry
        for carrier, offset, density, flux, sign in (
            ("n", count, n, fluxes.electron, 1.0),
            ("p", 2 * count, p, fluxes.hole, -1.0),
        ):
            name = "electron" if carrier == "n" else "hole"
            sink = TermSink(device._carrier_layout, name, geometry)
            sink.add(FaceFlux(name + ".sg", "faces", flux.flux, FLUX_UNIT))
            sink.add(CellSource(name + ".net_pairs", "cells", source, SOURCE_UNIT))
            rows = np.array(sink.value(), copy=True)
            rate_of_density = sign * density * (
                derivative[:count] / device.thermal_voltage_V + derivative[offset:offset + count]
            )
            rows -= device.geometry.cell_measures_m * rate_of_density
            for side, node in (("left", 0), ("right", count - 1)):
                channel = device.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    references = device.carrier_reference_potentials(self.voltage_V)
                    reference = references[0] if carrier == "n" else references[1]
                    shift = (phi[node] - reference[node]) / device.thermal_voltage_V
                    rows[node] = sign * (shift + values[offset + node])
                else:
                    exchange, _ = self._contact_exchange(values, channel, node)
                    rows[node] += (1.0 if side == "left" else -1.0) * exchange
            result[offset:offset + count] = rows
        return result

    def storage_jacobian(self, u: ArrayLike) -> csc_matrix:
        """Exact Fudot; contact constraints and Poisson rows have no storage."""
        device = self.device
        count = device.node_count
        _, n, p = device.fields(u, voltage_V=self.voltage_V)
        wn = device.geometry.cell_measures_m * n
        wp = device.geometry.cell_measures_m * p
        for carrier, weights in (("n", wn), ("p", wp)):
            for side, node in (("left", 0), ("right", count - 1)):
                if device.channel(carrier, side).kind == "fixed_reservoir":
                    weights[node] = 0.0
        matrix = lil_matrix((3 * count, 3 * count), dtype=float)
        matrix[count:2 * count, :count] = diags(-wn / device.thermal_voltage_V)
        matrix[count:2 * count, count:2 * count] = diags(-wn)
        matrix[2 * count:, :count] = diags(wp / device.thermal_voltage_V)
        matrix[2 * count:, 2 * count:] = diags(wp)
        return matrix.tocsc()

    def jacobian(self, u: ArrayLike, udot: ArrayLike | None = None, *, alpha: float = 0.0) -> csc_matrix:
        """Analytic Fy + alpha Fudot in the production quasi-Fermi coordinates."""
        device = self.device
        count = device.node_count
        thermal = device.thermal_voltage_V
        values = np.asarray(u, dtype=float)
        _, n, p = device.fields(values, voltage_V=self.voltage_V)
        derivative = self._derivative(udot)
        fluxes = self.fluxes(values)
        rate = self.recombination_rate(n, p)
        transport = None if device.field_transport is None else device.field_transport.evaluate(values[:count])
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        widths = device.geometry.cell_measures_m
        matrix = lil_matrix((3 * count, 3 * count), dtype=float)
        matrix[:count, :count] = device.poisson.jacobian()
        interior = range(1, count - 1)
        charge_widths = device.poisson.interior_widths_m
        for local, node in enumerate(interior):
            matrix[node, node] += -Q * charge_widths[local] * (n[node] + p[node]) / thermal
            matrix[node, count + node] = -Q * charge_widths[local] * n[node]
            matrix[node, 2 * count + node] = -Q * charge_widths[local] * p[node]
        source_n = -np.asarray(rate.d_rate_dn) * n
        source_p = np.asarray(rate.d_rate_dp) * p
        source_phi = (source_n + source_p) / thermal
        for carrier, offset, density, flux, sign in (
            ("n", count, n, fluxes.electron, 1.0),
            ("p", 2 * count, p, fluxes.hole, -1.0),
        ):
            drive = sign * np.diff(values[offset:offset + count])
            left_phi, right_phi = _quasi_fermi_potential_tangents(flux, drive, sign, thermal)
            if transport is not None:
                diffusivity = (transport.electron_diffusivity_m2_s if carrier == "n"
                               else transport.hole_diffusivity_m2_s)
                d_diffusivity = (transport.electron_d_diffusivity_d_field if carrier == "n"
                                 else transport.hole_d_diffusivity_d_field)
                # SG is linear in positive D. E = -(phi_R-phi_L)/dx, so
                # this is the additional (dGamma/dD)(dD/dE)(dE/dphi) term.
                field_chain = (flux.flux / diffusivity) * d_diffusivity / np.diff(device.nodes_m)
                left_phi = left_phi + field_chain
                right_phi = right_phi - field_chain
            left_z = sign * flux.left_density_derivative * density[:-1]
            right_z = sign * flux.right_density_derivative * density[1:]
            for face in range(count - 1):
                for orientation, node in ((-1.0, face), (1.0, face + 1)):
                    row = offset + node
                    matrix[row, face] += orientation * left_phi[face]
                    matrix[row, face + 1] += orientation * right_phi[face]
                    matrix[row, offset + face] += orientation * left_z[face]
                    matrix[row, offset + face + 1] += orientation * right_z[face]
            density_rate = sign * density * (
                derivative[:count] / thermal + derivative[offset:offset + count]
            )
            for node in range(count):
                row = offset + node
                matrix[row, node] += widths[node] * (source_phi[node] - sign * density_rate[node] / thermal)
                matrix[row, count + node] += widths[node] * source_n[node]
                matrix[row, 2 * count + node] += widths[node] * source_p[node]
                matrix[row, offset + node] -= widths[node] * sign * density_rate[node]
            for side, node in (("left", 0), ("right", count - 1)):
                row = offset + node
                channel = device.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    matrix.rows[row] = [node, offset + node]
                    matrix.data[row] = [sign / thermal, sign]
                else:
                    _, slope = self._contact_exchange(values, channel, node)
                    inflow = 1.0 if side == "left" else -1.0
                    matrix[row, node] += inflow * slope * sign * density[node] / thermal
                    matrix[row, offset + node] += inflow * slope * sign * density[node]
        result = matrix.tocsc()
        if alpha != 0.0:
            if not math.isfinite(alpha):
                raise ValueError("alpha must be finite")
            result = result + alpha * self.storage_jacobian(values)
        return result

    def current_density(self, u: ArrayLike) -> Vector:
        fluxes = self.fluxes(u)
        return fluxes.electron_current() + fluxes.hole_current()

    def contact_current_density(self, u: ArrayLike) -> Vector:
        values = np.asarray(u, dtype=float)
        fluxes = self.fluxes(values)
        _, n, p = self.device.fields(values, voltage_V=self.voltage_V)
        rate = self.recombination_rate(n, p, derivatives=False)
        pairs = self.device.geometry.cell_measures_m * (
            self.generation_scale * self.device.generation.source_m3_s - rate.rate
        )
        currents = []
        for side, node, face in (("left", 0, 0), ("right", self.device.node_count - 1, -1)):
            total = 0.0
            for carrier, charge, flux in (("n", -Q, fluxes.electron), ("p", Q, fluxes.hole)):
                channel = self.device.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    exchange = float(flux.flux[face] + (-1.0 if side == "left" else 1.0) * pairs[node])
                else:
                    exchange, _ = self._contact_exchange(values, channel, node)
                total += charge * exchange
            currents.append(total)
        return np.asarray(currents)

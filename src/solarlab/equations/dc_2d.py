"""Classical 2D DC on one connected lateral/stack face graph.

Unknowns are (phi, z_n, z_p) using the accepted majority-electrode references.
Gauss and carrier balances use physical areas and face lengths per unit depth;
fixed-reservoir contact rows are unweighted potential/log-density constraints.
Finite exchange and blocking contacts retain conservative carrier rows.
There is no independent-column residual, nonlinear solver or transient route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, lil_matrix

from solarlab.contracts import (
    AREA, COULOMB, LENGTH, PARTICLE, SECOND, VOLUME, CellSource, EquationSpec,
    FaceFlux, Layout, Support, TermSink, frozen_array,
)
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.face_average import VACUUM_PERMITTIVITY_F_M
from solarlab.equations.dc_1d import CompiledDevice1D, _quasi_fermi_potential_tangents
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.physics.contacts.exchange import CarrierExchangeChannel
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger
from solarlab.physics.generation.profile import GenerationProfile
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.physics.recombination.reaction import (
    BulkRecombinationCoefficients, BulkRecombinationEvaluation, BulkRecombinationReaction,
)
from solarlab.physics.statistics.band_potentials import BandStructure
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFlux, CarrierFaceFluxes, stable_sg_particle_flux

Vector: TypeAlias = NDArray[np.float64]
Q = ELEMENTARY_CHARGE_C


class GrainRecombinationModel2D(Protocol):
    """A geometry-bound rate correction supplied by device assembly."""

    @property
    def geometry_sha256(self) -> str: ...

    def evaluate(self, evaluation: BulkRecombinationEvaluation,
                 coefficients: BulkRecombinationCoefficients) -> RecombinationRate: ...


@dataclass(frozen=True)
class State2D:
    u: Vector
    potential_V: Vector
    electron_density_m3: Vector
    hole_density_m3: Vector
    device_sha256: str
    voltage_V: float = 0.0

    def __post_init__(self) -> None:
        for name in ("u", "potential_V", "electron_density_m3", "hole_density_m3"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        shape = self.potential_V.shape
        if len(shape) != 2 or self.u.shape != (3 * self.potential_V.size,):
            raise ValueError("2D fields require (Ny,Nx) and a matching coordinate vector")
        for value in (self.electron_density_m3, self.hole_density_m3):
            if value.shape != shape or np.any(value <= 0.0):
                raise ValueError("2D carrier densities must be positive on every node")
        if not self.device_sha256 or not math.isfinite(self.voltage_V):
            raise ValueError("state requires device identity and finite reference voltage")


@dataclass(frozen=True)
class CompiledDevice2D:
    """Extruded physical coefficients; equations act on the full 2D face graph.

    column supplies only accepted physical coefficients/contact references.
    Its integer-endpoint geometry is not used for 2D conservation weights.
    generation_y is explicitly rebound to physical half-endpoint cells.
    """

    sha256: str
    column: CompiledDevice1D
    geometry: RectangularGeometry2D
    generation_y: GenerationProfile
    relative_permittivity_y: Vector
    spatial_declaration_json: bytes
    grain_recombination: GrainRecombinationModel2D | None = None
    column_projection_json: bytes | None = None
    band_structure: BandStructure = field(init=False)
    recombination: BulkRecombinationReaction = field(init=False)
    generation_source_m3_s: Vector = field(init=False)
    electron_diffusivity_faces_m2_s: Vector = field(init=False)
    hole_diffusivity_faces_m2_s: Vector = field(init=False)
    face_capacitance_F_m2: Vector = field(init=False)
    layout: Layout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        column, geometry = self.column, self.geometry
        if self.grain_recombination is not None and self.grain_recombination.geometry_sha256 != geometry.sha256:
            raise ValueError("grain recombination must be bound to this exact 2D geometry")
        if not self.sha256 or not np.array_equal(column.nodes_m, geometry.y_geometry.nodes_m):
            raise ValueError("column coefficients must match the 2D stack grid and identity")
        if column.field_transport is not None or column.recombination.explicit_defects is not None:
            raise ValueError("2D DC currently requires constant mobility and lifetime SRH")
        if column.additional_recombination is not None:
            raise ValueError("2D DC requires the bound bulk reaction")
        if all(c.kind == "blocking" for c in column.channels):
            raise ValueError("fully_blocking_contacts: a closed carrier inventory constraint is not bound in 2D DC")
        eps = frozen_array(self.relative_permittivity_y)
        if eps.shape != (geometry.ny,) or np.any(eps <= 0.0):
            raise ValueError("positive physical permittivity is required on every stack node")
        object.__setattr__(self, "relative_permittivity_y", eps)
        self.generation_y.require_inputs(self.generation_y.binding_sha256, geometry.y_geometry)
        object.__setattr__(self, "generation_source_m3_s", frozen_array(self.extrude(self.generation_y.source_m3_s)))
        bands = column.parameters.band_structure
        object.__setattr__(self, "band_structure", BandStructure(
            self.extrude(bands.physical_electron_affinity_eV), self.extrude(bands.physical_band_gap_eV),
            self.extrude(bands.transport_electron_affinity_eV), self.extrude(bands.transport_band_gap_eV),
            bands.dos_folded, bands.reference_layer,
        ))
        coefficients = column.recombination.coefficients
        names = ("ni_sq_m6", "tau_n_s", "tau_p_s", "n1_m3", "p1_m3", "radiative_m3_s", "auger_n_m6_s", "auger_p_m6_s")
        object.__setattr__(self, "recombination", BulkRecombinationReaction(
            BulkRecombinationCoefficients(self.node_count, *(self.extrude(getattr(coefficients, name)) for name in names))))
        for carrier in ("electron", "hole"):
            nodal = getattr(column.parameters, carrier + "_diffusivity_m2_s")
            vertical = getattr(column, carrier + "_diffusivity_faces_m2_s")
            values = np.r_[np.repeat(nodal, geometry.nx - 1), np.repeat(vertical, geometry.nx)]
            if np.any(values <= 0.0):
                raise ValueError("2D DC requires positive finite carrier diffusivities")
            object.__setattr__(self, carrier + "_diffusivity_faces_m2_s", frozen_array(values))
        horizontal_capacitance = (VACUUM_PERMITTIVITY_F_M * eps[:, None] / np.diff(geometry.x_m)[None, :]).ravel()
        object.__setattr__(self, "face_capacitance_F_m2", frozen_array(np.r_[
            horizontal_capacitance, np.repeat(column.poisson.face_capacitance_F_m2, geometry.nx)]))
        faces = geometry.face_spacing_m.size
        object.__setattr__(self, "layout", Layout(
            (Support("cells", "cell", (self.node_count,)), Support("faces", "face", (faces,))), (),
            (EquationSpec("gauss", "electrostatics", "cells", (self.node_count,), COULOMB / LENGTH, role="constraint"),
             EquationSpec("electron", "electron", "cells", (self.node_count,), PARTICLE / (LENGTH * SECOND)),
             EquationSpec("hole", "hole", "cells", (self.node_count,), PARTICLE / (LENGTH * SECOND))),
        ))

    @property
    def identity(self) -> str:
        return self.sha256

    @property
    def node_count(self) -> int:
        return self.geometry.node_count

    @property
    def thermal_voltage_V(self) -> float:
        return self.column.thermal_voltage_V

    @property
    def fixed_carrier_mask(self) -> NDArray[np.bool_]:
        """Density-constraint rows, indexed by (electron/hole, node)."""
        mask: NDArray[np.bool_] = np.zeros((2, self.node_count), dtype=bool)
        sides = self.geometry.contact_nodes.reshape(2, self.geometry.nx)
        for carrier_index, carrier in enumerate(("n", "p")):
            for side, nodes in zip(("left", "right"), sides):
                if self.column.channel(carrier, side).kind == "fixed_reservoir":
                    mask[carrier_index, nodes] = True
        mask.flags.writeable = False
        return mask

    @property
    def generation_limit_A_m(self) -> float:
        return Q * float(np.sum(self.generation_source_m3_s * self.geometry.contract_geometry.volumes))

    def extrude(self, values_y: ArrayLike) -> Vector:
        values = np.asarray(values_y, dtype=float)
        if values.shape != (self.geometry.ny,):
            raise ValueError("extruded nodal coefficients must match the stack grid")
        return np.repeat(values, self.geometry.nx)

    def extrude_coordinates(self, u_1d: ArrayLike) -> Vector:
        values = np.asarray(u_1d, dtype=float)
        if values.shape != (3 * self.geometry.ny,) or not np.all(np.isfinite(values)):
            raise ValueError("reference coordinates must match the compiled column")
        return np.repeat(values.reshape(3, self.geometry.ny), self.geometry.nx, axis=1).ravel()

    def reference_potentials(self, voltage_V: float) -> tuple[Vector, Vector]:
        left, right = self.column.carrier_reference_potentials(voltage_V)
        return self.extrude(left), self.extrude(right)

    def fields(self, u: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector]:
        values = np.asarray(u, dtype=float)
        count = self.node_count
        if values.shape != (3 * count,) or not np.all(np.isfinite(values)):
            raise ValueError("coordinates must be finite and match the 2D state layout")
        phi = values[:count]
        ref_n, ref_p = self.reference_potentials(voltage_V)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            n = self.extrude(self.column.reference_electron_density_m3) * np.exp(
                (phi - ref_n) / self.thermal_voltage_V + values[count:2 * count])
            p = self.extrude(self.column.reference_hole_density_m3) * np.exp(
                -(phi - ref_p) / self.thermal_voltage_V - values[2 * count:])
        return phi, n, p

    def state(self, u: ArrayLike, *, voltage_V: float = 0.0) -> State2D:
        phi, n, p = self.fields(u, voltage_V=voltage_V)
        shape = self.geometry.shape
        return State2D(np.asarray(u, dtype=float), phi.reshape(shape), n.reshape(shape), p.reshape(shape), self.sha256, voltage_V)

    def charge_ledger(self, n: Vector, p: Vector) -> ChargeLedger:
        return ChargeLedger(self.node_count, 0, (
            ChargeEntry("hole", "hole", 1, p), ChargeEntry("electron", "electron", -1, n),
            ChargeEntry("acceptor", "dopant", -1, self.extrude(self.column.parameters.acceptor_density_m3)),
            ChargeEntry("donor", "dopant", 1, self.extrude(self.column.parameters.donor_density_m3)),
        ))


@dataclass(frozen=True)
class DC2DProblem:
    device: CompiledDevice2D
    voltage_V: float = 0.0
    generation_scale: float = 0.0

    def __post_init__(self) -> None:
        self.device.column.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and nonnegative")

    def fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        d = self.device
        values = np.asarray(u, dtype=float)
        phi, n, p = d.fields(values, voltage_V=self.voltage_V)
        left, right = d.geometry.contract_geometry.pairs.T
        potentials = d.band_structure.transport_potentials(phi)
        fluxes = []
        for density, potential, diffusivity, offset, sign in (
            (n, potentials[0], d.electron_diffusivity_faces_m2_s, d.node_count, 1.0),
            (p, potentials[1], d.hole_diffusivity_faces_m2_s, 2 * d.node_count, -1.0),
        ):
            step = -sign * (potential[right] - potential[left]) / d.thermal_voltage_V
            coefficient = diffusivity / d.geometry.face_spacing_m
            drive = sign * (values[offset + right] - values[offset + left])
            evaluated = stable_sg_particle_flux(density[left], density[right], step, coefficient, electrochemical_step=drive)
            per_volt = -sign / d.thermal_voltage_V * evaluated.energy_step_derivative
            fluxes.append(CarrierFaceFlux(evaluated.flux, evaluated.left_density_derivative, evaluated.right_density_derivative,
                                         -per_volt, per_volt, step, coefficient))
        return CarrierFaceFluxes(fluxes[0], fluxes[1])

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        evaluation = self.device.recombination.evaluate(n, p, derivatives=derivatives)
        evaluation.require_domain()
        if self.device.grain_recombination is not None:
            return self.device.grain_recombination.evaluate(evaluation, self.device.recombination.coefficients)
        return evaluation.total

    def displacement(self, u: ArrayLike) -> Vector:
        phi, _, _ = self.device.fields(u, voltage_V=self.voltage_V)
        left, right = self.device.geometry.contract_geometry.pairs.T
        return -self.device.face_capacitance_F_m2 * (phi[right] - phi[left])

    def _contact_exchange(self, u: Vector, channel: CarrierExchangeChannel,
                          nodes: NDArray[np.int64]) -> tuple[Vector, float]:
        """The 1D contact law/expm1 construction on a whole y contact face.

        Left/right means y-min/y-max here. Gamma is directed along +y,
        in particles/(m2*s); its exact density slope is n_out*S.
        """
        d, count = self.device, self.device.node_count
        references = d.reference_potentials(self.voltage_V)
        reference = references[0] if channel.carrier == "n" else references[1]
        shift = (u[nodes] - reference[nodes]) / d.thermal_voltage_V
        exponent = shift + u[count + nodes] if channel.carrier == "n" else -shift - u[2 * count + nodes]
        boundary = boundary_particle_flux(channel, channel.reservoir_density_m3)
        excess = channel.reservoir_density_m3 * np.expm1(exponent)
        return boundary.density_derivative * excess, boundary.density_derivative

    def residual(self, u: ArrayLike) -> Vector:
        d, count = self.device, self.device.node_count
        values = np.asarray(u, dtype=float)
        phi, n, p = d.fields(values, voltage_V=self.voltage_V)
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)) or np.any(n <= 0.0) or np.any(p <= 0.0):
            return np.full(3 * count, np.nan)
        fluxes = self.fluxes(values)
        source = self.generation_scale * d.generation_source_m3_s - self.recombination_rate(n, p, derivatives=False).rate
        outputs = []
        for name, flux, flux_unit, cell_source, source_unit in (
            ("gauss", self.displacement(values), COULOMB / AREA, d.charge_ledger(n, p).space_charge_C_m3(), COULOMB / VOLUME),
            ("electron", fluxes.electron.flux, PARTICLE / (AREA * SECOND), source, PARTICLE / (VOLUME * SECOND)),
            ("hole", fluxes.hole.flux, PARTICLE / (AREA * SECOND), source, PARTICLE / (VOLUME * SECOND)),
        ):
            sink = TermSink(d.layout, name, d.geometry.contract_geometry)
            sink.add(FaceFlux(name + ".transport", "faces", flux, flux_unit))
            sink.add(CellSource(name + ".source", "cells", cell_source, source_unit))
            outputs.append(np.array(sink.value(), copy=True))
        bottom, top = d.column.contact_potentials(self.voltage_V)
        outputs[0][:d.geometry.nx] = phi[:d.geometry.nx] - bottom
        outputs[0][-d.geometry.nx:] = phi[-d.geometry.nx:] - top
        sides = d.geometry.contact_nodes.reshape(2, d.geometry.nx)
        for carrier, rows, offset, sign, reference in zip(("n", "p"), outputs[1:], (count, 2 * count),
                                                          (1.0, -1.0), d.reference_potentials(self.voltage_V)):
            for side, nodes, inflow in zip(("left", "right"), sides, (1.0, -1.0)):
                channel = d.column.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    rows[nodes] = sign * ((phi[nodes] - reference[nodes]) / d.thermal_voltage_V + values[offset + nodes])
                else:
                    exchange, _ = self._contact_exchange(values, channel, nodes)
                    rows[nodes] += inflow * exchange * d.geometry.x_widths_m
        return np.concatenate(outputs)

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        """Full analytic Newton Jacobian, including both spatial directions."""
        d, count = self.device, self.device.node_count
        values = np.asarray(u, dtype=float)
        _, n, p = d.fields(values, voltage_V=self.voltage_V)
        thermal = d.thermal_voltage_V
        geometry = d.geometry.contract_geometry
        left, right = geometry.pairs.T
        matrix = lil_matrix((3 * count, 3 * count), dtype=float)
        for a, b, capacitance in zip(left, right, d.face_capacitance_F_m2 * geometry.face_measures):
            matrix[a, a] -= capacitance
            matrix[a, b] += capacitance
            matrix[b, a] += capacitance
            matrix[b, b] -= capacitance
        for node, area in enumerate(geometry.volumes):
            matrix[node, node] -= Q * area * (n[node] + p[node]) / thermal
            matrix[node, count + node] = -Q * area * n[node]
            matrix[node, 2 * count + node] = -Q * area * p[node]
        fluxes = self.fluxes(values)
        rate = self.recombination_rate(n, p)
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        source_n = -np.asarray(rate.d_rate_dn) * n
        source_p = np.asarray(rate.d_rate_dp) * p
        source_phi = (source_n + source_p) / thermal
        for offset, sign, density, flux in ((count, 1.0, n, fluxes.electron), (2 * count, -1.0, p, fluxes.hole)):
            drive = sign * (values[offset + right] - values[offset + left])
            # Reuse the accepted cancellation-safe quasi-Fermi chain, without altering its source.
            phi_left, phi_right = _quasi_fermi_potential_tangents(flux, drive, sign, thermal)
            z_left = sign * flux.left_density_derivative * density[left]
            z_right = sign * flux.right_density_derivative * density[right]
            for face, (a, b) in enumerate(geometry.pairs):
                for orientation, node in ((-1.0, a), (1.0, b)):
                    weight = orientation * geometry.face_measures[face]
                    row = offset + node
                    matrix[row, a] += weight * phi_left[face]
                    matrix[row, b] += weight * phi_right[face]
                    matrix[row, offset + a] += weight * z_left[face]
                    matrix[row, offset + b] += weight * z_right[face]
            for node, area in enumerate(geometry.volumes):
                row = offset + node
                matrix[row, node] += area * source_phi[node]
                matrix[row, count + node] += area * source_n[node]
                matrix[row, 2 * count + node] += area * source_p[node]
        for node in d.geometry.contact_nodes:
            matrix.rows[node], matrix.data[node] = [int(node)], [1.0]
        sides = d.geometry.contact_nodes.reshape(2, d.geometry.nx)
        for carrier, offset, sign, density in (("n", count, 1.0, n), ("p", 2 * count, -1.0, p)):
            for side, nodes, inflow in zip(("left", "right"), sides, (1.0, -1.0)):
                channel = d.column.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    for node in nodes:
                        matrix.rows[offset + node] = [int(node), int(offset + node)]
                        matrix.data[offset + node] = [sign / thermal, sign]
                else:
                    _, slope = self._contact_exchange(values, channel, nodes)
                    for node, width in zip(nodes, d.geometry.x_widths_m):
                        matrix[offset + node, node] += inflow * slope * sign * density[node] / thermal * width
                        matrix[offset + node, offset + node] += inflow * slope * sign * density[node] * width
        return matrix.tocsc()

    def current_density(self, u: ArrayLike) -> Vector:
        """Internal directed face currents [A/m2], x faces followed by y faces."""
        fluxes = self.fluxes(u)
        return fluxes.electron_current() + fluxes.hole_current()

    def cross_section_current_A_m(self, u: ArrayLike) -> Vector:
        g = self.device.geometry
        current_y = self.current_density(u)[g.x_face_count:].reshape(g.ny - 1, g.nx)
        return np.sum(current_y * g.x_widths_m[None, :], axis=1)

    def contact_particle_fluxes(self, u: ArrayLike) -> tuple[Vector, Vector]:
        """Y-min/y-max flux densities along +y, using each channel's own law.

        Fixed channels close the contact-cell balance, including lateral
        inflow and half-cell generation/recombination. Finite/blocking
        channels use their actual exchange; a failed row is never closed by
        substituting an inferred terminal flux.
        """
        d, g = self.device, self.device.geometry
        _, n, p = d.fields(u, voltage_V=self.voltage_V)
        source = self.generation_scale * d.generation_source_m3_s - self.recombination_rate(n, p, derivatives=False).rate
        pairs = (source * g.contract_geometry.volumes).reshape(g.shape)
        fluxes = self.fluxes(u)
        result = []
        horizontal_pairs = g.contract_geometry.pairs[:g.x_face_count]
        for carrier, flux in zip(("n", "p"), (fluxes.electron.flux, fluxes.hole.flux)):
            x_inflow = np.zeros(d.node_count)
            weighted = flux[:g.x_face_count] * g.contract_geometry.face_measures[:g.x_face_count]
            np.add.at(x_inflow, horizontal_pairs[:, 0], -weighted)
            np.add.at(x_inflow, horizontal_pairs[:, 1], weighted)
            x_inflow = x_inflow.reshape(g.shape)
            y_flux = flux[g.x_face_count:].reshape(g.ny - 1, g.nx)
            contact = np.stack((y_flux[0] - (x_inflow[0] + pairs[0]) / g.x_widths_m,
                                y_flux[-1] + (x_inflow[-1] + pairs[-1]) / g.x_widths_m))
            for side_index, (side, nodes) in enumerate(zip(("left", "right"), g.contact_nodes.reshape(2, g.nx))):
                channel = d.column.channel(carrier, side)
                if channel.kind != "fixed_reservoir":
                    contact[side_index], _ = self._contact_exchange(np.asarray(u, dtype=float), channel, nodes)
            result.append(contact)
        return result[0], result[1]

    def terminal_current_A_m(self, u: ArrayLike) -> Vector:
        """Integrated bottom/top currents along +y, per metre of out-of-plane depth."""
        electron, hole = self.contact_particle_fluxes(u)
        return np.sum((-Q * electron + Q * hole) * self.device.geometry.x_widths_m[None, :], axis=1)

    def terminal_current_density_A_m2(self, u: ArrayLike) -> Vector:
        return self.terminal_current_A_m(u) / self.device.geometry.lateral_length_m

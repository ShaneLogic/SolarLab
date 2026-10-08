"""Coupled physical-SI semiconductor prototype for two frozen one-slab cases.

No production solver is imported. All rates, sparse derivatives and exterior
ports are evaluated by this kernel. The dense common-contract bridge is
explicitly limited to the first N8 diagnostic; scaling belongs to the adapter.
"""

from __future__ import annotations

from scripts.benchmarks.native_history import HistoryLimitError, emit_record
from scripts.benchmarks.runtime_timing import RuntimeTiming

from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, diags, hstack, vstack
import yaml

from perovskite_sim.constants import EPS_0, K_B, Q
from scripts.benchmarks.contract_prototype import (
    AREA, COULOMB, LENGTH, ONE, PARTICLE, SECOND, VOLT, VOLUME,
    BalanceTerms, BoundLinearSource, ContractError,
    EquationSpec, FloatArray, Geometry, ImmutableArrays, ImplicitSystem,
    Layout, LinearCoordinates, LinearFactor, LinearSourceSpec, LinearStorage,
    LinearTerm, PhysicalLinearForm, Point, RateView, StateIncrement, Support,
    TerminalPort, VariableSpec, frozen_array, _linear_arithmetic_identity,
)
from scripts.benchmarks.port_prototype import PortSample
from scripts.benchmarks.sparse_prototype import SparseGraph, TermSupport, factorize


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SlabDefinition:
    id: str
    source_path: str
    source_sha256: str
    dynamic: bool
    length: float
    area: float
    temperature: float
    epsilon: float
    mu_n: float
    mu_p: float
    diffusion_ion: float
    ion_initial: float
    ion_capacity: float
    trap_density: float
    capture_n: float
    capture_p: float
    n1: float
    p1: float
    intrinsic_product: float
    ni: float
    n_eq: float
    p_eq: float
    f_eq: float
    alpha: float
    photon_reference: float

    @property
    def vt(self) -> float:
        return K_B * self.temperature / Q

    @property
    def identity(self) -> str:
        return digest(asdict(self))

    @classmethod
    def from_plan(cls, plan_path: Path, repo: Path, case_id: str,
                  *, area: float | None = None) -> SlabDefinition:
        plan = json.loads(plan_path.read_text())
        case = next((c for c in plan["cases"] if c["id"] == case_id), None)
        if case is None:
            raise ContractError("unregistered_device_case")
        path = repo / case["source"]
        source = path.read_bytes()
        receipt = next(x for x in plan["sources"] if x["path"] == case["source"])
        if sha256(source).hexdigest() != receipt["sha256"]:
            raise ContractError("device_source_changed")
        cfg = yaml.safe_load(source)
        if len(cfg["layers"]) != 1:
            raise ContractError("prototype_requires_one_slab")
        layer, dev = cfg["layers"][0], cfg["device"]
        defect = layer["bulk_defects"][0]
        dynamic = case_id == "DynamicAcceptorIonPublicDeviceV1"
        if (case_id not in {"S0NeutralPublicDeviceV1", "DynamicAcceptorIonPublicDeviceV1"}
                or defect["degeneracy"] != 1.0 or len(layer["bulk_defects"]) != 1
                or defect["distribution"]["kind"] != "single_level"
                or layer["defect_model"] != "explicit_quasi_steady"
                or defect["charge_transition"] != ("acceptor" if dynamic else "neutral")
                or dev["built_in_potential_mode"] != "semiconductor_work_function"):
            raise ContractError("unsupported_device_definition")
        get = lambda key: float(layer[key])
        T = float(dev["T"])
        if T != 300.0 or any(get(k) != 0 for k in ["N_A", "N_D", "B_rad", "C_n", "C_p"]):
            raise ContractError("prototype_material_scope")
        vt = K_B * T / Q
        nc, nv, gap = get("Nc300"), get("Nv300"), get("Eg")
        energy = float(defect["distribution"]["center_eV_above_vb"])
        nt = float(defect["distribution"]["total_density_m3"])
        kinetics = defect["kinetics"]
        cn = float(kinetics["sigma_n_m2"]) * float(kinetics["thermal_velocity_n_m_s"])
        cp = float(kinetics["sigma_p_m2"]) * float(kinetics["thermal_velocity_p_m_s"])
        n1, p1 = nc * math.exp(-(gap - energy) / vt), nv * math.exp(-energy / vt)
        ni2 = nc * nv * math.exp(-gap / vt)
        ni = math.sqrt(ni2)

        def occupancy(n: float, p: float) -> float:
            return (cn * n + cp * p1) / (cn * (n + n1) + cp * (p + p1))

        def neutrality(u: float) -> float:
            n, p = ni * math.exp(u), ni * math.exp(-u)
            return (p - n - nt * occupancy(n, p)) / nt

        u = brentq(neutrality, -100.0, 100.0, xtol=2e-14,
                   rtol=4 * np.finfo(float).eps, maxiter=256) if dynamic else 0.0
        ne, pe = ni * math.exp(u), ni * math.exp(-u)
        a = float(case["model"]["area_m2"] if area is None else area)
        if not np.isfinite(a) or a <= 0:
            raise ContractError("invalid_device_area")
        return cls(case_id, str(path), receipt["sha256"], dynamic, get("thickness"), a, T,
                   EPS_0 * get("eps_r"), get("mu_n"), get("mu_p"), get("D_ion"),
                   get("P0"), get("P_lim"), nt, cn, cp, n1, p1, ni2, ni, ne, pe,
                   occupancy(ne, pe), get("alpha"), float(case["photon_flux_m2_s"]))


@dataclass(frozen=True)
class ProtocolSegment:
    id: str
    start: float
    end: float
    voltage: tuple[float, float]
    photons: tuple[float, float]

    def inputs(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(time) or time < self.start or time > self.end:
            raise ContractError("time_outside_frozen_segment")
        start = np.array([self.voltage[0], self.photons[0]])
        end = np.array([self.voltage[1], self.photons[1]])
        rate = (end - start) / (self.end - self.start)
        if time == self.start:
            return frozen_array(start), frozen_array(rate)
        if time == self.end:
            return frozen_array(end), frozen_array(rate)
        return frozen_array(start + rate * (time - self.start)), frozen_array(rate)


def protocol_from_plan(plan_path: Path, case_id: str) -> tuple[ProtocolSegment, ...]:
    case = next(c for c in json.loads(plan_path.read_text())["cases"] if c["id"] == case_id)
    result = tuple(ProtocolSegment(s["id"], float(s["start_s"]), float(s["end_s"]),
                                   tuple(s["voltage_V"]), tuple(s["Phi_m2_s"]))
                   for s in case["protocol"]["segments"])
    if (not result or result[0].start != 0 or any(s.end <= s.start for s in result)
            or any(a.end != b.start or a.voltage[1] != b.voltage[0]
                   or a.photons[1] != b.photons[0] for a, b in zip(result, result[1:]))):
        raise ContractError("invalid_frozen_protocol")
    return result


def bernoulli_with_derivative(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ContractError("nonfinite_bernoulli_argument")
    z = np.abs(x)
    b, derivative = np.empty_like(z), np.empty_like(z)
    small = z <= 0.05
    u = z[small]
    b[small] = 1 - u/2 + u**2/12 - u**4/720 + u**6/30240 - u**8/1209600 + u**10/47900160
    derivative[small] = -0.5 + u/6 - u**3/180 + u**5/5040 - u**7/151200 + u**9/4790016
    u = z[~small]
    exponential, denominator = np.exp(-u), -np.expm1(-u)
    b[~small] = u * exponential / denominator
    derivative[~small] = exponential * (denominator - u) / denominator**2
    negative = x < 0
    b[negative] += z[negative]
    derivative[negative] = -derivative[negative] - 1
    return b, derivative


@dataclass(frozen=True)
class DeviceEvaluation(ImmutableArrays):
    rate: np.ndarray
    algebraic: np.ndarray
    nodal_rates: Mapping[str, np.ndarray]
    nodal_derivatives: Mapping[tuple[str, str], csr_matrix]
    nodal_input: Mapping[str, np.ndarray]
    rate_jacobian: csr_matrix | None
    input_jacobian: np.ndarray
    charge_density: np.ndarray
    displacement: np.ndarray
    electron_current: np.ndarray
    hole_current: np.ndarray
    ion_flux: np.ndarray
    reaction_n: np.ndarray
    reaction_p: np.ndarray


@dataclass(frozen=True)
class DeviceObservation(ImmutableArrays):
    point: Point
    derivative: np.ndarray
    input_rate: np.ndarray
    origin: str
    ports: PortSample
    body_charge: float
    body_charge_rate: float
    gauss_defect: float
    tangent_residual: np.ndarray
    interior_total_current: np.ndarray

    def __post_init__(self) -> None:
        if self.origin not in {"native", "interpolant", "physical_tangent", "algebraic_probe",
                               "stop_output", "endpoint_restore", "segment_initial"}:
            raise ContractError("unknown_derivative_origin")
        for name in ["derivative", "input_rate", "tangent_residual", "interior_total_current"]:
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        if self.derivative.shape != self.point.y.shape or self.input_rate.shape != self.point.inputs.shape:
            raise ContractError("observation_rate_shape")


class CoupledSlab:
    """One source-defined model, sparse physical operator and exterior ledger."""

    def __init__(self, definition: SlabDefinition, intervals: int = 8):
        if type(intervals) is not int or intervals < 4 or intervals > 256:
            raise ContractError("prototype_grid_scope")
        self.definition, self.intervals, self.count = definition, intervals, intervals + 1
        m, N = definition, self.count
        x = m.length * (1 + np.tanh(3*np.linspace(-1, 1, N))/np.tanh(3)) / 2
        edges = np.r_[0.0, (x[1:] + x[:-1])/2, m.length]
        self.x, self.edges, self.dx = map(frozen_array, [x, edges, np.diff(x)])
        self.geometry = Geometry(m.area*np.diff(edges), np.column_stack((np.arange(N-1), np.arange(1, N))),
                                 np.full(N-1, m.area))
        self.absorption = frozen_array(m.area*np.exp(-m.alpha*edges[:-1]) * (-np.expm1(-m.alpha*np.diff(edges))))
        self.interior = np.arange(1, N-1)
        density = PARTICLE/VOLUME
        supports = (Support("nodes", "cell", (N,)), Support("interior", "cell", (N-2,)),
                    Support("faces", "face", (N-1,)), Support("contacts", "global", (2,)))
        variables = [VariableSpec("n_m3", "carriers", "nodes", (N,), density, lower=0),
                     VariableSpec("p_m3", "carriers", "nodes", (N,), density, lower=0),
                     VariableSpec("phi_V", "electrostatics", "nodes", (N,), VOLT, role="constraint")]
        equations = [EquationSpec("a_n", "carriers", "interior", (N-2,), PARTICLE/SECOND,
                                  derivative_support=("n_m3", "phi_V", "f" if m.dynamic else "p_m3")),
                     EquationSpec("b_p", "carriers", "interior", (N-2,), PARTICLE/SECOND,
                                  derivative_support=("p_m3", "phi_V", "f" if m.dynamic else "n_m3"))]
        self.rate_fields = [("a_n", "n_m3", self.interior), ("b_p", "p_m3", self.interior)]
        if m.dynamic:
            variables += [VariableSpec("c_m3", "ions", "nodes", (N,), density, lower=0, upper=m.ion_capacity),
                          VariableSpec("f", "traps", "nodes", (N,), ONE, lower=0, upper=1)]
            equations += [EquationSpec("c_ion", "ions", "nodes", (N,), PARTICLE/SECOND, derivative_support=("c_m3", "phi_V")),
                          EquationSpec("d_trap", "traps", "nodes", (N,), PARTICLE/SECOND, derivative_support=("n_m3", "p_m3", "f"))]
            self.rate_fields += [("c_ion", "c_m3", np.arange(N)), ("d_trap", "f", np.arange(N))]
        equations += [EquationSpec("x_poisson", "electrostatics", "interior", (N-2,), COULOMB, role="constraint",
                                   derivative_support=tuple(v.id for v in variables)),
                      EquationSpec("y_phi_contact", "contacts", "contacts", (2,), VOLT, role="constraint", derivative_support=("phi_V",)),
                      EquationSpec("z_n_contact", "contacts", "contacts", (2,), density, role="constraint", derivative_support=("n_m3",)),
                      EquationSpec("zz_p_contact", "contacts", "contacts", (2,), density, role="constraint", derivative_support=("p_m3",))]
        self.layout = Layout(supports, tuple(variables), tuple(equations))
        self.coordinates = LinearCoordinates(self.layout, "physical-si-v1:"+digest(
            {"definition": m.identity, "grid_hex": [float(v).hex() for v in x]}))
        self.ports = (TerminalPort("left", -1, m.area), TerminalPort("right", 1, m.area))
        self.dynamic_count = sum(len(nodes) for _, _, nodes in self.rate_fields)
        self.topology = digest({"definition": m.identity, "x_hex": [float(v).hex() for v in x],
                                "area_hex": m.area.hex(), "layout": self.layout.identity})
        self.source_identity = digest({"definition": m.identity, "topology": self.topology,
                                       "kernel_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                                       "linear_arithmetic": _linear_arithmetic_identity()})
        self.supports = self._supports()
        self.graph = SparseGraph(self.layout, self.supports, topology_identity=self.topology)
        mass_rows, mass_cols, mass_values = [], [], []
        for eq, var, nodes in self.rate_fields:
            rows = np.arange(len(nodes)) + self.graph.row_offsets[eq].start
            mass_rows.extend(rows); mass_cols.extend(self.layout.offsets[var].start + nodes)
            mass_values.extend(self.geometry.volumes[nodes] * (m.trap_density if var == "f" else 1))
        self.mass = coo_matrix((mass_values, (mass_rows, mass_cols)), shape=self.graph.shape).tocsc()
        self.dynamic_mass = self.mass[:self.dynamic_count].tocsr()
        self.G, self.Ga, self.g_constant = self._constraints()
        phi_columns = np.arange(self.layout.offsets["phi_V"].start, self.layout.offsets["phi_V"].stop)
        # Interior Poisson + two Dirichlet rows; the same constant factor is
        # used for finite-g estimates and tangent rates, never state projection.
        self.phi_constraint_count = N
        self.phi_factor = factorize(self.G[:N, phi_columns].tocsc())
        self.factorizations = 1
        self.S, self.Drow = self._scales()

    def _supports(self) -> tuple[TermSupport, ...]:
        N, m = self.count, self.definition
        eqs = {e.id: e for e in self.layout.equations}
        varspec = {v.id: v for v in self.layout.variables}
        terms = []
        for eq in self.layout.equations:
            nodes = self.interior if eq.id in {"a_n", "b_p", "x_poisson"} else np.arange(N)
            if eq.id in {"y_phi_contact", "z_n_contact", "zz_p_contact"}:
                nodes = np.array([0, N-1])
            for var in eq.derivative_support:
                adjacent = ((eq.id == "a_n" and var in {"n_m3", "phi_V"})
                            or (eq.id == "b_p" and var in {"p_m3", "phi_V"})
                            or eq.id == "c_ion" or (eq.id == "x_poisson" and var == "phi_V"))
                pairs = [(row, col) for row, node in enumerate(nodes)
                         for col in (range(max(0, node-1), min(N, node+2)) if adjacent else [node])]
                terms.append(TermSupport(eq.id+":"+var, eq.owner, eq.id, var,
                                         np.asarray(pairs, dtype=int),
                                         kind="port" if "contact" in eq.id else "local",
                                         unit=eq.unit/varspec[var].unit))
        return tuple(terms)

    def _constraints(self) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
        m, N, k = self.definition, self.count, self.dynamic_count
        rows, columns, values = [], [], []
        def add(row, var, node, value):
            rows.append(row); columns.append(self.layout.offsets[var].start+node); values.append(value)
        for row, i in enumerate(self.interior):
            a, b = m.area*m.epsilon/self.dx[i-1], m.area*m.epsilon/self.dx[i]
            for node, value in [(i-1, -a), (i, a+b), (i+1, -b)]: add(row, "phi_V", node, value)
            add(row, "n_m3", i, Q*self.geometry.volumes[i])
            add(row, "p_m3", i, -Q*self.geometry.volumes[i])
            if m.dynamic:
                add(row, "c_m3", i, -Q*self.geometry.volumes[i])
                add(row, "f", i, Q*self.geometry.volumes[i]*m.trap_density)
        q = N-2
        for var, offset in [("phi_V", q), ("n_m3", q+2), ("p_m3", q+4)]:
            add(offset, var, 0, 1.0); add(offset+1, var, N-1, 1.0)
        matrix = coo_matrix((values, (rows, columns)), shape=(self.layout.size-k, self.layout.size)).tocsr()
        a = np.zeros((matrix.shape[0], 2)); a[q+1, 0] = 1
        constant = np.zeros(matrix.shape[0]); constant[q+2:q+4] = -m.n_eq; constant[q+4:q+6] = -m.p_eq
        if m.dynamic: constant[:q] = Q*self.geometry.volumes[self.interior]*m.ion_initial
        return matrix, frozen_array(a), frozen_array(constant)

    def _scales(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.definition
        columns = np.ones(self.layout.size)
        for name, scale in [("n_m3", m.ni), ("p_m3", m.ni), ("phi_V", m.vt)]: columns[self.layout.offsets[name]] = scale
        if m.dynamic: columns[self.layout.offsets["c_m3"]] = m.ion_initial
        rows = np.ones(self.layout.size)
        for eq, var, nodes in self.rate_fields:
            if var in {"n_m3", "p_m3"}:
                tau = m.length**2/((m.mu_n if var == "n_m3" else m.mu_p)*m.vt)
                reference = self.geometry.volumes[nodes]*m.ni/tau
            elif var == "c_m3": reference = self.geometry.volumes[nodes]*m.ion_initial/(m.length**2/m.diffusion_ion)
            else:
                relaxation = m.capture_n*(m.n_eq+m.n1)+m.capture_p*(m.p_eq+m.p1)
                reference = self.geometry.volumes[nodes]*m.trap_density*relaxation
            rows[self.graph.row_offsets[eq]] = 1/reference
        rows[self.graph.row_offsets["x_poisson"]] = 1/(m.area*m.epsilon*m.vt/m.length)
        rows[self.graph.row_offsets["y_phi_contact"]] = 1/m.vt
        rows[self.graph.row_offsets["z_n_contact"]] = 1/m.n_eq
        rows[self.graph.row_offsets["zz_p_contact"]] = 1/m.p_eq
        return frozen_array(columns), frozen_array(rows)

    def point(self, fields: Mapping[str, np.ndarray], time: float = 0.0,
              inputs=(0.0, 0.0)) -> Point:
        if set(fields) != set(self.layout.offsets): raise ContractError("device_state_fields_mismatch")
        values = np.concatenate([np.asarray(fields[v.id], float) for v in self.layout.variables])
        point = self.coordinates.point(values, time, inputs)
        self.validate(point)
        return point

    def initial(self) -> Point:
        m, N = self.definition, self.count
        fields = {"n_m3": np.full(N, m.n_eq), "p_m3": np.full(N, m.p_eq), "phi_V": np.zeros(N)}
        if m.dynamic: fields.update(c_m3=np.full(N, m.ion_initial), f=np.full(N, m.f_eq))
        return self.point(fields)

    def field(self, point: Point, name: str) -> np.ndarray:
        value = point.state.field(name)
        if not isinstance(value, FloatArray): raise ContractError("product_kernel_requires_explicit_float_physical_state")
        return value.values

    def validate(self, point: Point) -> None:
        if point.state.layout.identity != self.layout.identity or point.coordinate_reference != self.coordinates.reference or point.inputs.shape != (2,):
            raise ContractError("device_point_binding_mismatch")
        if point.inputs[1] < 0: raise ContractError("negative_photon_flux")
        if any(not np.array_equal(self.field(point, v.id), point.y[self.layout.offsets[v.id]])
               for v in self.layout.variables):
            raise ContractError("physical_point_coordinate_state_mismatch")
        for name in ["n_m3", "p_m3"]:
            if np.any(self.field(point, name) <= 0): raise ContractError("nonpositive_active_carrier")
        if self.definition.dynamic:
            c, f = self.field(point, "c_m3"), self.field(point, "f")
            if np.any(c < 0) or np.any(c >= self.definition.ion_capacity): raise ContractError("ion_outside_smooth_physical_domain")
            if np.any(f < 0) or np.any(f > 1): raise ContractError("trap_outside_physical_domain")

    def evaluate(self, point: Point, *, derivatives: bool = False) -> DeviceEvaluation:
        self.validate(point)
        m, N, volume = self.definition, self.count, self.geometry.volumes
        n, p, phi = (self.field(point, v) for v in ["n_m3", "p_m3", "phi_V"])
        xi = np.diff(phi)/m.vt
        b, bp = bernoulli_with_derivative(xi)
        pref_n, pref_p = m.mu_n*m.vt/self.dx, m.mu_p*m.vt/self.dx
        jn = Q*pref_n*(b*np.diff(n)-xi*n[:-1])
        jp = Q*pref_p*(-b*np.diff(p)-xi*p[1:])
        flows = {"n_m3": -jn/Q, "p_m3": jp/Q}
        flow_derivatives = {
            "n_m3": {"n_m3": (pref_n*(b+xi), -pref_n*b),
                     "phi_V": (pref_n*(bp*np.diff(n)-n[:-1])/m.vt, -pref_n*(bp*np.diff(n)-n[:-1])/m.vt)},
            "p_m3": {"p_m3": (pref_p*b, -pref_p*(b+xi)),
                     "phi_V": (-pref_p*(-bp*np.diff(p)-p[1:])/m.vt, pref_p*(-bp*np.diff(p)-p[1:])/m.vt)},
        }
        if m.dynamic:
            c, f = self.field(point, "c_m3"), self.field(point, "f")
            chemical = -np.log1p(-c/m.ion_capacity)
            xi_ion = xi + np.diff(chemical)
            bi, bip = bernoulli_with_derivative(xi_ion)
            pref = m.diffusion_ion/self.dx
            fi = pref*(bi*(c[:-1]-c[1:])-xi_ion*c[1:])
            drive_derivative = pref*(bip*(c[:-1]-c[1:])-c[1:])
            flows["c_m3"] = fi
            flow_derivatives["c_m3"] = {
                "c_m3": (pref*bi-drive_derivative/(m.ion_capacity-c[:-1]),
                          -pref*(bi+xi_ion)+drive_derivative/(m.ion_capacity-c[1:])),
                "phi_V": (-drive_derivative/m.vt, drive_derivative/m.vt)}
            rn = m.capture_n*(n*(1-f)-m.n1*f)
            rp = m.capture_p*(p*f-m.p1*(1-f))
            derivatives_n = {"n_m3": m.capture_n*(1-f), "f": -m.capture_n*(n+m.n1)}
            derivatives_p = {"p_m3": m.capture_p*f, "f": m.capture_p*(p+m.p1)}
        else:
            c, fi = np.zeros(N), np.zeros(N-1)
            relaxation = m.capture_n*(n+m.n1)+m.capture_p*(p+m.p1)
            f = (m.capture_n*n+m.capture_p*m.p1)/relaxation
            per_trap = m.capture_n*m.capture_p*(n*p-m.n1*m.p1)/relaxation
            rn, rp = per_trap, per_trap
            derivatives_n = {"n_m3": m.capture_n*m.capture_p*p/relaxation-per_trap*m.capture_n/relaxation,
                             "p_m3": m.capture_n*m.capture_p*n/relaxation-per_trap*m.capture_p/relaxation}
            derivatives_p = derivatives_n
        rates = {var: np.zeros(N) for _, var, _ in self.rate_fields}
        derivative_entries: dict[tuple[str, str], list] = {}
        def add(var, against, row, col, value):
            if not derivatives:
                return
            derivative_entries.setdefault((var, against), [[], [], []])
            storage = derivative_entries[(var, against)]
            storage[0].append(row); storage[1].append(col); storage[2].append(value)
        for var, flux in flows.items():
            rates[var][:-1] -= m.area*flux; rates[var][1:] += m.area*flux
            for against, (left, right) in (flow_derivatives[var].items() if derivatives else ()):
                for face in range(N-1):
                    for node, sign in [(face, -1), (face+1, 1)]:
                        add(var, against, node, face, sign*m.area*left[face])
                        add(var, against, node, face+1, sign*m.area*right[face])
        generation = point.inputs[1]*self.absorption
        rates["n_m3"] += generation-volume*m.trap_density*rn
        rates["p_m3"] += generation-volume*m.trap_density*rp
        for var, partials in ([("n_m3", derivatives_n), ("p_m3", derivatives_p)] if derivatives else ()):
            for against, value in partials.items():
                for i in range(N): add(var, against, i, i, -volume[i]*m.trap_density*value[i])
        if m.dynamic:
            rates["f"] = volume*m.trap_density*(rn-rp)
            for against in ({**derivatives_n, **derivatives_p} if derivatives else ()):
                value = np.asarray(derivatives_n.get(against, np.zeros(N)))-np.asarray(derivatives_p.get(against, np.zeros(N)))
                for i in range(N): add("f", against, i, i, volume[i]*m.trap_density*value[i])
        nodal_jac = {key: coo_matrix((v[2], (v[0], v[1])), shape=(N, N)).tocsr()
                     for key, v in derivative_entries.items()}
        ra_nodal = {var: np.column_stack((np.zeros(N), self.absorption if var in {"n_m3", "p_m3"} else np.zeros(N)))
                    for var in rates}
        blocks = []
        for eq, var, nodes in (self.rate_fields if derivatives else ()):
            blocks.append(hstack([nodal_jac.get((var, v.id), csr_matrix((N, N)))[nodes]
                                  for v in self.layout.variables], format="csr"))
        rate = np.concatenate([rates[var][nodes] for _, var, nodes in self.rate_fields])
        ra = np.vstack([ra_nodal[var][nodes] for _, var, nodes in self.rate_fields])
        charge = Q*(p-n+(c-m.ion_initial-m.trap_density*f if m.dynamic else 0))
        displacement = -m.epsilon*np.diff(phi)/self.dx
        g = self.G @ point.y + self.Ga @ point.inputs + self.g_constant
        return DeviceEvaluation(frozen_array(rate), frozen_array(g),
                                MappingProxyType({k: frozen_array(v) for k, v in rates.items()}),
                                MappingProxyType(nodal_jac), MappingProxyType({k: frozen_array(v) for k, v in ra_nodal.items()}),
                                vstack(blocks, format="csr") if derivatives else None, frozen_array(ra), frozen_array(charge),
                                frozen_array(displacement), frozen_array(jn), frozen_array(jp), frozen_array(fi),
                                frozen_array(rn), frozen_array(rp))

    def sparse_jacobian(self, point: Point, cj: float) -> csc_matrix:
        if not np.isfinite(cj): raise ContractError("nonfinite_cj")
        evaluation = self.evaluate(point, derivatives=True)
        full = vstack((-evaluation.rate_jacobian, self.G), format="csr") + cj*self.mass
        return self._assemble_declared(full)

    def _assemble_declared(self, full) -> csc_matrix:
        """Fill the public term graph, including every declared zero slot."""
        values = {}
        for term in self.supports:
            rows = term.pairs[:, 0]+self.graph.row_offsets[term.equation].start
            cols = term.pairs[:, 1]+self.layout.offsets[term.variable].start
            values[term.id] = np.asarray(full[rows, cols]).ravel()
        return self.graph.assemble(values, supports=self.supports, topology_identity=self.topology)

    def residual(self, point: Point, derivative: np.ndarray, input_rate: np.ndarray) -> np.ndarray:
        if np.asarray(derivative).shape != point.y.shape or np.asarray(input_rate).shape != (2,):
            raise ContractError("device_rate_shape")
        derivative, input_rate = frozen_array(derivative), frozen_array(input_rate)
        e = self.evaluate(point)
        return np.r_[self.dynamic_mass @ derivative-e.rate, e.algebraic]

    def common_system(self) -> ImplicitSystem:
        if self.intervals != 8 or self.layout.size > 45:
            raise ContractError("dense_common_bridge_limited_to_N8")
        def terms(point):
            e = self.evaluate(point, derivatives=True)
            return BalanceTerms(e.rate, e.algebraic, e.rate_jacobian.toarray(), e.input_jacobian,
                                np.zeros(self.dynamic_count), self.G.toarray(), self.Ga,
                                np.zeros(self.G.shape[0]))
        return ImplicitSystem(LinearStorage(self.dynamic_mass.toarray(), self.coordinates.reference), terms)

    def public_problem(self):
        """Consumer of the shared cheap-residual/analytic-sparse contract.

        This adapter requires the reviewed shared API. It does not synthesize
        a fallback protocol or call the N8 dense convenience implementation.
        """
        from scripts.benchmarks.contract_prototype import (
            PhysicalStorage, SparseLinearization, SparseStructure, ValidatedProblem,
        )
        structure = SparseStructure.from_graph(self.graph)
        units = tuple(e.unit*SECOND for e in self.layout.equations if e.role == "storage"
                      for _ in range(int(np.prod(e.shape))))
        storage = PhysicalStorage(self.storage_value, self.storage_delta, units, self.source_identity)
        def linearize(point, ydot, adot):
            # Q is linear in the physical SI coordinates for these two cases.
            # Consequently Fy has no ydot/adot dependence, and Fadot/Ft=0.
            e = self.evaluate(point, derivatives=True)
            fy = self._assemble_declared(vstack((-e.rate_jacobian, self.G), format="csr"))
            fydot = self._assemble_declared(self.mass.tocsr())
            fa = np.vstack((-e.input_jacobian, self.Ga))
            return SparseLinearization(fy, fydot, fa, np.zeros_like(fa),
                                       np.zeros(self.layout.size), structure, self.source_identity)
        return ValidatedProblem(self.layout, 2, self.dynamic_count, storage, structure,
                                self.source_identity, self.residual, linearize,
                                self.conservative_residual, self.conservative_jacobian,
                                point_validator=self.validate)

    def storage_value(self, point: Point) -> np.ndarray:
        self.validate(point)
        return frozen_array(self.dynamic_mass @ point.y)

    def storage_delta(self, left: Point, right: Point, increment: StateIncrement) -> np.ndarray:
        increment.validate(left, right)
        self.validate(left); self.validate(right)
        fields = [increment.field(v.id) for v in self.layout.variables]
        if any(not isinstance(v, FloatArray) for v in fields):
            raise ContractError("product_storage_requires_explicit_float_increment")
        return frozen_array(self.dynamic_mass @ np.concatenate([v.values for v in fields]))

    def conservative_residual(self, left: Point, right: Point, increment: StateIncrement) -> np.ndarray:
        if right.time <= left.time: raise ContractError("nonpositive_physical_interval")
        e = self.evaluate(right)
        return frozen_array(np.r_[self.storage_delta(left, right, increment)-(right.time-left.time)*e.rate, e.algebraic])

    def conservative_jacobian(self, left: Point, right: Point) -> csc_matrix:
        h = right.time-left.time
        if h <= 0: raise ContractError("nonpositive_physical_interval")
        matrix = self.sparse_jacobian(right, 1/h)
        factors = np.r_[np.full(self.dynamic_count, h), np.ones(self.G.shape[0])]
        return csc_matrix((matrix.data*factors[matrix.indices], matrix.indices.copy(), matrix.indptr.copy()),
                          shape=matrix.shape)

    def tangent_rate(self, point: Point, input_rate: np.ndarray) -> np.ndarray:
        """Separately named estimate at unchanged state; never a native rate."""
        rate = frozen_array(input_rate)
        if rate.shape != (2,): raise ContractError("device_input_rate_shape")
        e, result = self.evaluate(point), np.zeros(self.layout.size)
        for eq, var, nodes in self.rate_fields:
            coefficient = self.geometry.volumes[nodes]*(self.definition.trap_density if var == "f" else 1)
            result[self.layout.offsets[var].start+nodes] = e.rate[self.graph.row_offsets[eq]]/coefficient
        forcing = -(self.G[:self.count] @ result+self.Ga[:self.count] @ rate)
        result[self.layout.offsets["phi_V"]] = self.phi_factor.solve(forcing)
        return frozen_array(result)

    def observe(self, point: Point, derivative: np.ndarray, input_rate: np.ndarray,
                origin: str = "algebraic_probe", side: str = "continuous") -> DeviceObservation:
        e, m, volume = self.evaluate(point), self.definition, self.geometry.volumes
        derivative, input_rate = frozen_array(derivative), frozen_array(input_rate)
        if derivative.shape != point.y.shape or input_rate.shape != (2,): raise ContractError("device_rate_shape")
        rates = {v.id: derivative[self.layout.offsets[v.id]] for v in self.layout.variables}
        rho_dot = Q*(rates["p_m3"]-rates["n_m3"]
                     + (rates["c_m3"]-m.trap_density*rates["f"] if m.dynamic else 0))
        d_dot = -m.epsilon*np.diff(rates["phi_V"])/self.dx
        end = np.array([0, self.count-1]); widths = volume[end]/m.area
        d_outer = np.array([e.displacement[0]-e.charge_density[0]*widths[0],
                            e.displacement[-1]+e.charge_density[-1]*widths[1]])
        dd_outer = np.array([d_dot[0]-rho_dot[0]*widths[0], d_dot[-1]+rho_dot[-1]*widths[1]])
        # Endpoint reservoir exchange is a particle balance on the half cell.
        exchange_n = volume[end]*rates["n_m3"][end]-e.nodal_rates["n_m3"][end]
        exchange_p = volume[end]*rates["p_m3"][end]-e.nodal_rates["p_m3"][end]
        inward_conduction = Q*(exchange_p-exchange_n)
        jc = np.array([inward_conduction[0]/m.area, -inward_conduction[1]/m.area])
        charges = np.array([float(p.charge(d_outer[i])) for i, p in enumerate(self.ports)])
        total = np.array([float(p.current(jc[i], dd_outer[i])) for i, p in enumerate(self.ports)])
        sample = PortSample(point.time, point.inputs[0], input_rate[0], charges, inward_conduction, total, side)
        body = float(volume @ e.charge_density)
        return DeviceObservation(point, derivative, input_rate, origin, sample, body,
                                 float(volume @ rho_dot), float(body+sum(charges)),
                                 self.G @ derivative+self.Ga @ input_rate,
                                 e.electron_current+e.hole_current+Q*e.ion_flux+d_dot)

    def affine_constraint_error(self, point: Point) -> dict:
        """Exact linear defect estimator; returns evidence, never changes state."""
        e = self.evaluate(point)
        delta_phi = self.phi_factor.solve(-e.algebraic[:self.count])
        induced_D = -self.definition.epsilon*np.diff(delta_phi)/self.dx
        # Ideal carrier constraints involve only the two endpoint populations.
        # Their correction does not enter an interior Poisson row, but it does
        # enter the endpoint half-cell Gauss law and the total body charge.
        contact_delta = -e.algebraic[self.count:]
        end = np.array([0, self.count-1])
        rho_delta = Q*(contact_delta[2:]-contact_delta[:2])
        widths = self.geometry.volumes[end]/self.definition.area
        charges = self.definition.area*np.array([
            induced_D[0]-rho_delta[0]*widths[0],
            -induced_D[-1]-rho_delta[1]*widths[1]])
        body_delta = float(self.geometry.volumes[end] @ rho_delta)
        return {"point_identity": point.identity, "state_changed": False,
                "potential_error_bound_V": float(np.max(np.abs(delta_phi))),
                "metal_charge_error_bound_C": np.abs(charges).tolist(),
                "body_charge_error_bound_C": abs(body_delta),
                "potential_correction_V": delta_phi.tolist(),
                "contact_density_correction_m3": contact_delta.tolist(),
                "metal_charge_correction_C": charges.tolist(),
                "body_charge_correction_C": body_delta,
                "preserves_differential_storage": True,
                "linear_solve_residual": float(np.max(np.abs(self.G[:self.count, self.layout.offsets['phi_V']] @ delta_phi+e.algebraic[:self.count]))),
                "contact_density_residual_m3": e.algebraic[self.count:].tolist(),
                "Poisson_constraint_charge_C": e.algebraic[:self.count-2].tolist(),
                "factorizations": self.factorizations, "nonlinear_remainder": 0.0,
                "acceptance_awarded": False}

    def port_partials(self, point: Point) -> dict[str, csr_matrix | np.ndarray]:
        """Actual exterior charge/current derivatives, including half cells."""
        m, N, size = self.definition, self.count, self.layout.size
        e = self.evaluate(point, derivatives=True)
        end = np.array([0, N-1])
        derivative_rows = []
        for var in ["n_m3", "p_m3"]:
            derivative_rows.append(hstack([e.nodal_derivatives.get((var, v.id), csr_matrix((N, N)))[end]
                                           for v in self.layout.variables], format="csr"))
        conduction_y = Q*(derivative_rows[0]-derivative_rows[1])
        rows, cols, vals = [], [], []
        for row, node in enumerate(end):
            for var, sign in [("n_m3", -1), ("p_m3", 1)]:
                rows.append(row); cols.append(self.layout.offsets[var].start+node)
                vals.append(Q*sign*self.geometry.volumes[node])
        conduction_rate = coo_matrix((vals, (rows, cols)), shape=(2, size)).tocsr()
        rows, cols, vals = [], [], []
        for row, (i, j) in enumerate([(0, 1), (N-2, N-1)]):
            for node, sign in [(i, 1), (j, -1)]:
                rows.append(row); cols.append(self.layout.offsets["phi_V"].start+node)
                vals.append(sign*m.epsilon/self.dx[i])
            node, half_sign = (0, -1) if row == 0 else (N-1, 1)
            width = self.geometry.volumes[node]/m.area
            for var, sign in [("n_m3", -1), ("p_m3", 1)]+([("c_m3", 1), ("f", -m.trap_density)] if m.dynamic else []):
                rows.append(row); cols.append(self.layout.offsets[var].start+node)
                vals.append(half_sign*width*Q*sign)
        displacement_y = coo_matrix((vals, (rows, cols)), shape=(2, size)).tocsr()
        charge_y = diags([m.area, -m.area]) @ displacement_y
        body_y = np.zeros(size)
        for var, sign in [("n_m3", -1), ("p_m3", 1)]+([("c_m3", 1), ("f", -m.trap_density)] if m.dynamic else []):
            body_y[self.layout.offsets[var]] = Q*sign*self.geometry.volumes
        return {"charge_y": charge_y.tocsr(), "conduction_y": conduction_y,
                "conduction_ydot": conduction_rate, "current_y": conduction_y,
                "current_ydot": (conduction_rate+charge_y).tocsr(),
                "body_y": frozen_array(body_y), "current_inputs": np.zeros((2, 2)),
                "current_input_rates": np.zeros((2, 2))}

    def numeric_packet(self) -> dict:
        initial = self.initial()
        return {"definition": asdict(self.definition), "definition_identity": self.definition.identity,
                "intervals": self.intervals, "nodes": self.count, "x_m": self.x.tolist(),
                "cell_edges_m": self.edges.tolist(), "volumes_m3": self.geometry.volumes.tolist(),
                "face_areas_m2": self.geometry.face_measures.tolist(), "layout_identity": self.layout.identity,
                "variable_offsets": {k: [v.start, v.stop] for k, v in self.layout.offsets.items()},
                "equation_offsets": {k: [v.start, v.stop] for k, v in self.graph.row_offsets.items()},
                "initial_y_physical": initial.y.tolist(), "initial_identity": initial.identity,
                "column_scaling": self.S.tolist(), "row_scaling": self.Drow.tolist(),
                "sparse_graph_identity": self.graph.identity, "sparse_nnz": self.graph.nnz,
                "sparse_edges": self.graph.edges().tolist(), "mass_diagonal_entries": self.mass.data.tolist(),
                "mass_rows": self.mass.tocoo().row.tolist(), "mass_columns": self.mass.tocoo().col.tolist()}


@dataclass(frozen=True)
class AffineDeviceEvaluation:
    """Values retain explicit precision; sparse derivatives use binary64."""

    rate: object
    algebraic: object
    nodal_rates: Mapping[str, object]
    nodal_derivatives: Mapping[tuple[str, str], csr_matrix]
    nodal_input: Mapping[str, np.ndarray]
    rate_jacobian: csr_matrix | None
    input_jacobian: np.ndarray
    charge_density: object
    displacement: object
    electron_current: object
    hole_current: object
    ion_flux: object
    ion_exterior_flux: object
    reaction_n: object
    reaction_p: object
    reference_reaction_n: object
    reference_reaction_p: object
    reaction_change_n: object
    reaction_change_p: object
    linear_actions: Mapping[str, object]


@dataclass(frozen=True)
class AffineDeviceObservation:
    """Physical rates and port values, before an explicit float output."""

    point: Point
    derivative: object
    input_rate: np.ndarray
    origin: str
    event_side: str
    metal_charge: object
    conduction_inward: object
    total_inward: object
    body_charge: object
    body_charge_rate: object
    gauss_defect: object
    tangent_residual: np.ndarray
    interior_total_current: object
    charge_integrands: object
    linear_actions: Mapping[str, object]


@dataclass(frozen=True)
class AffinePointEvaluation:
    """One read-only value evaluation bound to its exact immutable Point."""

    point: Point
    source_identity: str
    values: AffineDeviceEvaluation

    def __post_init__(self):
        # The value-only path has frozen DD/NumPy arrays and mapping proxies;
        # no mutable sparse derivative object is shared between consumers.
        if self.values.rate_jacobian is not None or self.values.nodal_derivatives:
            raise ContractError("observation_evaluation_must_be_value_only")
        if self.values.input_jacobian.flags.writeable or any(
                value.flags.writeable for value in self.values.nodal_input.values()):
            raise ContractError("observation_evaluation_must_be_readonly")


class AffineCoupledSlab(CoupledSlab):
    """The same two physical models, with one fixed original reference.

    Point.y contains physical SI remainders; StateView contains the physical
    populations generated from the immutable reference and named primitives.
    The mass matrix is unchanged. No evaluated state becomes a new reference,
    and finite reference charge/capture residuals are never zeroed.

    The public fixed-reference trial builder must be provided by the shared
    precision contract. There is deliberately no private trial-state fallback.
    """

    def __init__(self, definition: SlabDefinition, intervals: int = 8):
        from scripts.benchmarks.contract_prototype import StateView
        from scripts.benchmarks.precision_prototype import DoubleArithmetic, DoubleArray, RelativeCoordinates

        super().__init__(definition, intervals)
        self.arithmetic = DoubleArithmetic()
        m, N = self.definition, self.count
        fields = {"n_m3": DoubleArray(np.full(N, m.n_eq)),
                  "p_m3": DoubleArray(np.full(N, m.p_eq)),
                  "phi_V": DoubleArray(np.zeros(N))}
        if m.dynamic:
            fields.update(c_m3=DoubleArray(np.full(N, m.ion_initial)),
                          f=DoubleArray(np.full(N, m.f_eq)))
        self.coordinates = RelativeCoordinates(self.layout, {v.id: "linear" for v in self.layout.variables})
        self.reference = self.coordinates.initial(StateView(self.layout, fields.items()),
                                                  inputs=(0.0, 0.0))
        self.source_identity = digest({
            "physical_kernel": self.source_identity,
            "representation": "fixed-original-physical-reference-v1",
            "reference": self.reference.identity,
            "physical_mass_unchanged": True,
        })
        self.face_pairs = self.geometry.pairs
        self.linear_forms = MappingProxyType({
            kind: self._physical_forms(kind) for kind in ("state", "increment", "rate")
        })

    def _physical_forms(self, kind):
        """Declare the affine physical coefficients, never read rounded J data.

        These small COO descriptions share the public authority-aware action.
        Volumes already contain area; only displacement faces receive A.
        Every quotient keeps its actual dx as a declared divisor.
        """
        m, N, source = self.definition, self.count, self.source_identity
        factor = lambda name, value, unit: LinearFactor(name, float(value), unit, source)
        q = factor("elementary_charge", Q, COULOMB/PARTICLE)
        nt = factor("trap_density", m.trap_density, PARTICLE/VOLUME)
        c0 = factor("fixed_ion_background", m.ion_initial, PARTICLE/VOLUME)
        eps = factor("permittivity", m.epsilon, COULOMB/VOLT/LENGTH)
        area = factor("cross_section", m.area, AREA)
        volumes = tuple(factor(f"volume_{i}", v, VOLUME) for i, v in enumerate(self.geometry.volumes))
        spacings = tuple(factor(f"spacing_{i}", d, LENGTH) for i, d in enumerate(self.dx))
        forms = {}

        def form(name, units, terms, sources=()):
            units, terms = tuple(units), tuple(terms)
            forms[name] = PhysicalLinearForm(
                self.layout, tuple(f"{name}_{i}" for i in range(len(units))),
                units, terms, source, kind, len(terms), tuple(sources))

        def charge(row, node, factors=(), sign=1):
            terms = [LinearTerm(row, "p_m3", node, (q, *factors), sign),
                     LinearTerm(row, "n_m3", node, (q, *factors), -sign)]
            if m.dynamic:
                terms += [LinearTerm(row, "c_m3", node, (q, *factors), sign),
                          LinearTerm(row, "f", node, (q, *factors, nt), -sign)]
                if kind == "state":
                    terms.append(LinearTerm(row, None, 0, (q, *factors, c0), -sign))
            return terms

        def displacement(row, face, factors=(), sign=1):
            return [LinearTerm(row, "phi_V", face, (*factors, eps), sign, spacings[face]),
                    LinearTerm(row, "phi_V", face+1, (*factors, eps), -sign, spacings[face])]

        def metal(row, side):
            node, face, sign = (0, 0, 1) if side == 0 else (N-1, N-2, -1)
            return displacement(row, face, (area,), sign)+charge(row, node, (volumes[node],), -1)

        per_time = SECOND if kind == "rate" else ONE
        mass = []
        for equation, variable, nodes in self.rate_fields:
            for local, node in enumerate(nodes):
                row = self.graph.row_offsets[equation].start+local
                mass.append(LinearTerm(row, variable, int(node),
                                       (volumes[node], nt) if variable == "f" else (volumes[node],)))
        form("storage", [PARTICLE/per_time]*self.dynamic_count, mass)
        form("charge_density", [COULOMB/VOLUME/per_time]*N,
             [term for i in range(N) for term in charge(i, i)])
        body = [term for i in range(N) for term in charge(0, i, (volumes[i],))]
        form("body_charge", [COULOMB/per_time], body)
        form("displacement", [COULOMB/AREA/per_time]*(N-1),
             [term for i in range(N-1) for term in displacement(i, i)])
        form("metal_charge", [COULOMB/per_time]*2, metal(0, 0)+metal(1, 1))
        form("gauss_defect", [COULOMB/per_time], body+metal(0, 0)+metal(0, 1))
        if m.dynamic:
            inventory = [LinearTerm(0, "c_m3", i, (volumes[i],)) for i in range(N)]
            form("ion_inventory", [PARTICLE/per_time], inventory)
            if kind == "state":
                form("ion_inventory_change", [PARTICLE], inventory+[
                    LinearTerm(0, None, 0, (volumes[i], c0), -1) for i in range(N)])

        # The same physical affine constraints apply to states and tangents.
        # Inputs are actual Point-bound sources, not an inferred Jacobian row.
        if kind in {"state", "rate"}:
            contact_source = LinearSourceSpec("applied_contact", "contacts", (2,), VOLT/per_time, source)
            terms = []
            for row, i in enumerate(self.interior):
                terms += displacement(row, int(i), (area,))
                terms += displacement(row, int(i)-1, (area,), -1)
                terms += charge(row, int(i), (volumes[i],), -1)
            offset = N-2
            for block, variable in enumerate(("phi_V", "n_m3", "p_m3")):
                for side, node in enumerate((0, N-1)):
                    row = offset+2*block+side
                    terms.append(LinearTerm(row, variable, node))
                    if block == 0:
                        terms.append(LinearTerm(row, contact_source.id, side))
                    elif kind == "state":
                        eq = factor(variable+"_reservoir", m.n_eq if block == 1 else m.p_eq, PARTICLE/VOLUME)
                        terms.append(LinearTerm(row, None, 0, (eq,), -1))
            units = [equation.unit/per_time for equation in self.layout.equations
                     if equation.role == "constraint" for _ in range(int(np.prod(equation.shape)))]
            form("constraints", units, terms, (contact_source,))

        if kind == "rate":
            nodal = {variable: LinearSourceSpec("nodal_"+variable, "nodes", (N,), PARTICLE/SECOND, source)
                     for _, variable, _ in self.rate_fields}
            balance = list(mass)
            for equation, variable, nodes in self.rate_fields:
                balance += [LinearTerm(self.graph.row_offsets[equation].start+j,
                                       nodal[variable].id, int(node), sign=-1)
                            for j, node in enumerate(nodes)]
            form("balance", [PARTICLE/SECOND]*self.dynamic_count, balance, nodal.values())

            def conduction(row, node):
                return [LinearTerm(row, "p_m3", node, (q, volumes[node])),
                        LinearTerm(row, "n_m3", node, (q, volumes[node]), -1),
                        LinearTerm(row, nodal["n_m3"].id, node, (q,)),
                        LinearTerm(row, nodal["p_m3"].id, node, (q,), -1)]

            carrier_sources = (nodal["n_m3"], nodal["p_m3"])
            currents = conduction(0, 0)+conduction(1, N-1)
            form("conduction", [COULOMB/SECOND]*2, currents, carrier_sources)
            form("total_current", [COULOMB/SECOND]*2,
                 currents+metal(0, 0)+metal(1, 1), carrier_sources)
            form("charge_integrands", [COULOMB/SECOND]*3,
                 conduction(0, 0)+conduction(0, N-1)+metal(1, 0)+metal(2, 1), carrier_sources)
            face_sources = (
                LinearSourceSpec("electron_current", "faces", (N-1,), COULOMB/AREA/SECOND, source),
                LinearSourceSpec("hole_current", "faces", (N-1,), COULOMB/AREA/SECOND, source),
                LinearSourceSpec("ion_flux", "faces", (N-1,), PARTICLE/AREA/SECOND, source))
            terms = [term for i in range(N-1) for term in (
                *displacement(i, i), LinearTerm(i, "electron_current", i),
                LinearTerm(i, "hole_current", i), LinearTerm(i, "ion_flux", i, (q,)))]
            form("interior_total_current", [COULOMB/AREA/SECOND]*(N-1), terms, face_sources)
        return MappingProxyType(forms)

    def physical_rate_view(self, point, derivative, input_rate):
        """Admit full public rates unchanged; retain the old physical-vector path."""
        if isinstance(derivative, RateView):
            derivative.validate(point, input_rate, self.source_identity)
            return derivative
        derivative, input_rate = frozen_array(derivative), frozen_array(input_rate)
        if derivative.shape != point.y.shape or input_rate.shape != (2,):
            raise ContractError("device_rate_shape")
        return RateView(point, FloatArray(derivative), input_rate,
                        source_identity=self.source_identity,
                        mapping_identity="physical-si-rate-v1:"+self.source_identity,
                        origin="physical-rate", raw_coordinates=point.y, raw_rate=derivative)

    def linear_action(self, name, point, *, rate=None, increment=None, left=None,
                      evaluation=None):
        """Dispatch metadata to the public action; no local word arithmetic."""
        from scripts.benchmarks.precision_prototype import DoubleArray

        if rate is not None and increment is not None:
            raise ContractError("ambiguous_physical_linear_operand")
        kind = "increment" if increment is not None else "rate" if rate is not None else "state"
        form = self.linear_forms[kind][name]
        entries = []
        values = self.evaluated_values(point, evaluation) if any(
            source.id != "applied_contact" for source in form.sources) else None
        for spec in form.sources:
            if spec.id == "applied_contact":
                voltage = rate.input_rate[0] if kind == "rate" else point.inputs[0]
                value = DoubleArray([0.0, voltage])
            elif spec.id.startswith("nodal_"):
                value = values.nodal_rates[spec.id.removeprefix("nodal_")]
            else:
                value = getattr(values, spec.id)
            entries.append((spec, value))
        sources = BoundLinearSource.bind(point, entries)
        if kind == "increment":
            return increment.linear_form(left, point, form, arithmetic=self.arithmetic, sources=sources)
        if kind == "rate":
            rate.validate(point, rate.input_rate, self.source_identity)
            return rate.linear_form(form, arithmetic=self.arithmetic, point=point, sources=sources)
        return point.state.linear_form(form, arithmetic=self.arithmetic, point=point, sources=sources)

    def initial(self) -> Point:
        return self.reference

    def point(self, fields, time=0.0, inputs=(0.0, 0.0)) -> Point:
        raise ContractError("affine_device_requires_explicit_reference_trial")

    def trial(self, cumulative_increment, time=0.0, inputs=(0.0, 0.0), *, predecessor=None,
              transition_representation="four-word-v1"):
        # This original remainder prototype also restores v1affine histories.
        # The versioned voltage map explicitly selects paired transitions.
        builder = getattr(self.coordinates, "trial", None)
        if builder is None:
            raise ContractError("public_fixed_reference_trial_unavailable")
        return builder(self.reference, cumulative_increment, time, inputs,
                       predecessor=self.reference if predecessor is None else predecessor,
                       transition_representation=transition_representation)

    def field(self, point: Point, name: str):
        from scripts.benchmarks.precision_prototype import DoubleArray

        value = point.state.field(name)
        if not isinstance(value, DoubleArray):
            raise ContractError("affine_device_requires_explicit_precision_state")
        return value

    def remainders(self, point: Point) -> Mapping[str, object]:
        # Public mapped-field differences contract primitives before projection.
        return MappingProxyType({
            v.id: self.field(point, v.id).difference(self.field(self.reference, v.id))
            for v in self.layout.variables
        })

    def validate(self, point: Point) -> None:
        if point.state.layout.identity != self.layout.identity or point.inputs.shape != (2,):
            raise ContractError("affine_device_point_binding_mismatch")
        if point.inputs[1] < 0:
            raise ContractError("negative_photon_flux")
        for variable in self.layout.variables:
            # Calling the bound root field's public method rejects an unrelated
            # resolved array that merely happens to have the same values.
            self.field(self.reference, variable.id).difference(self.field(point, variable.id))
        for name in ("n_m3", "p_m3"):
            if np.any(self.field(point, name).as_dd() <= 0):
                raise ContractError("nonpositive_active_carrier")
        if self.definition.dynamic:
            c, f = (self.field(point, name).as_dd() for name in ("c_m3", "f"))
            if np.any(c < 0) or np.any(c >= self.definition.ion_capacity):
                raise ContractError("ion_outside_smooth_physical_domain")
            if np.any(f < 0) or np.any(f > 1):
                raise ContractError("trap_outside_physical_domain")

    @staticmethod
    def project_output(value) -> np.ndarray:
        """Named last-stage output projection, never a source of state fields."""
        from scripts.benchmarks.precision_prototype import DoubleArray

        if not isinstance(value, DoubleArray):
            raise ContractError("explicit_double_output_required")
        return frozen_array(value.high + value.low)

    def _capture(self, point, *, derivatives=True):
        """Reference plus finite bilinear changes; original imbalance retained."""
        from scripts.benchmarks.precision_prototype import DD

        m = self.definition
        partial_n = partial_p = None
        delta = self.remainders(point)
        n0, p0 = (self.field(self.reference, name).as_dd() for name in ("n_m3", "p_m3"))
        dn, dp = (delta[name].as_dd() for name in ("n_m3", "p_m3"))
        n, p = n0+dn, p0+dp
        cn, cp, n1, p1 = map(DD, (m.capture_n, m.capture_p, m.n1, m.p1))
        if m.dynamic:
            f0, df = self.field(self.reference, "f").as_dd(), delta["f"].as_dd()
            f = f0+df
            rn0 = cn*(n0*(1-f0)-n1*f0)
            rp0 = cp*(p0*f0-p1*(1-f0))
            drn = cn*((1-f0)*dn-(n0+n1)*df-dn*df)
            drp = cp*(f0*dp+(p0+p1)*df+dp*df)
            rn, rp = rn0+drn, rp0+drp
            if derivatives:
                partial_n = {"n_m3": cn*(1-f), "f": -cn*(n+n1)}
                partial_p = {"p_m3": cp*f, "f": cp*(p+p1)}
        else:
            lam0 = cn*(n0+n1)+cp*(p0+p1)
            dlambda = cn*dn+cp*dp
            lam = lam0+dlambda
            numerator0 = n0*p0-n1*p1
            dnumerator = p0*dn+n0*dp+dn*dp
            rn0 = rp0 = cn*cp*numerator0/lam0
            drn = drp = (cn*cp*dnumerator-rn0*dlambda)/lam
            rn = rp = rn0+drn
            if derivatives:
                partial_n = {"n_m3": cn*cp*p/lam-rn*cn/lam,
                             "p_m3": cn*cp*n/lam-rn*cp/lam}
                partial_p = partial_n
        return rn, rp, partial_n, partial_p, rn0, rp0, drn, drp

    def evaluate(self, point: Point, *, derivatives: bool = False) -> AffineDeviceEvaluation:
        from scripts.benchmarks.precision_prototype import DD, DoubleArray, bernoulli

        self.validate(point)
        m, N, volume = self.definition, self.count, self.geometry.volumes
        a = self.arithmetic
        pack = DoubleArray.from_dd
        n, p = (self.field(point, name).as_dd() for name in ("n_m3", "p_m3"))
        dphi = point.state.face_difference("phi_V", self.face_pairs).as_dd()
        dn = point.state.face_difference("n_m3", self.face_pairs).as_dd()
        dp = point.state.face_difference("p_m3", self.face_pairs).as_dd()
        xi = dphi/m.vt
        b = bernoulli(pack(xi)).as_dd()
        # The analytic derivative is nonsingular and is projected only for
        # the sparse binary64 Jacobian. Values use retained physical deltas.
        if derivatives:
            bp = DD(bernoulli_with_derivative(self.project_output(pack(xi)))[1])
        pref_n, pref_p = DD(m.mu_n)*m.vt/DD(self.dx), DD(m.mu_p)*m.vt/DD(self.dx)
        jn = Q*pref_n*(b*dn-xi*n[:-1])
        jp = Q*pref_p*(-b*dp-xi*p[1:])
        flows = {"n_m3": -jn/Q, "p_m3": jp/Q}
        if derivatives:
            flow_derivatives = {
                "n_m3": {"n_m3": (pref_n*(b+xi), -pref_n*b),
                          "phi_V": (pref_n*(bp*dn-n[:-1])/m.vt, -pref_n*(bp*dn-n[:-1])/m.vt)},
                "p_m3": {"p_m3": (pref_p*b, -pref_p*(b+xi)),
                          "phi_V": (-pref_p*(-bp*dp-p[1:])/m.vt, pref_p*(-bp*dp-p[1:])/m.vt)},
            }
        if m.dynamic:
            c = self.field(point, "c_m3").as_dd()
            dc = point.state.face_difference("c_m3", self.face_pairs).as_dd()
            # log(vacancy_L/vacancy_R), with the named spatial density
            # difference retained before subtracting near-equal logarithms.
            chemical_change = -(-dc/(DD(m.ion_capacity)-c[:-1])).log1p()
            xi_ion = xi+chemical_change
            bi = bernoulli(pack(xi_ion)).as_dd()
            if derivatives:
                bip = DD(bernoulli_with_derivative(self.project_output(pack(xi_ion)))[1])
            pref = DD(m.diffusion_ion)/DD(self.dx)
            fi = pref*(-bi*dc-xi_ion*c[1:])
            if derivatives:
                drive = pref*(-bip*dc-c[1:])
            flows["c_m3"] = fi
            if derivatives:
                flow_derivatives["c_m3"] = {
                    "c_m3": (pref*bi-drive/(DD(m.ion_capacity)-c[:-1]),
                               -pref*(bi+xi_ion)+drive/(DD(m.ion_capacity)-c[1:])),
                    "phi_V": (-drive/m.vt, drive/m.vt),
                }
        else:
            fi = DD(np.zeros(N-1))
        rn, rp, partial_n, partial_p, rn0, rp0, drn, drp = self._capture(point, derivatives=derivatives)
        rates = {}
        derivative_entries = {}

        def add(var, against, row, col, value):
            if derivatives:
                rows, cols, vals = derivative_entries.setdefault((var, against), [[], [], []])
                rows.append(row); cols.append(col); vals.append(float(value))

        for var, flux in flows.items():
            rates[var] = m.area*a.concatenate((-flux[:1], flux[:-1]-flux[1:], flux[-1:]))
            if derivatives:
                for against, (left, right) in flow_derivatives[var].items():
                    left, right = self.project_output(pack(left)), self.project_output(pack(right))
                    for face in range(N-1):
                        for node, sign in ((face, -1), (face+1, 1)):
                            add(var, against, node, face, sign*m.area*left[face])
                            add(var, against, node, face+1, sign*m.area*right[face])
        generation = DD(self.absorption)*point.inputs[1]
        rates["n_m3"] = rates["n_m3"]+generation-DD(volume)*m.trap_density*rn
        rates["p_m3"] = rates["p_m3"]+generation-DD(volume)*m.trap_density*rp
        for var, partials in ((("n_m3", partial_n), ("p_m3", partial_p)) if derivatives else ()):
            for against, value in partials.items():
                values = self.project_output(pack(-DD(volume)*m.trap_density*value))
                for i in range(N):
                    add(var, against, i, i, values[i])
        if m.dynamic:
            rates["f"] = DD(volume)*m.trap_density*(rn-rp)
            if derivatives:
                for against in {**partial_n, **partial_p}:
                    partial = partial_n.get(against, DD(np.zeros(N)))-partial_p.get(against, DD(np.zeros(N)))
                    values = self.project_output(pack(DD(volume)*m.trap_density*partial))
                    for i in range(N):
                        add("f", against, i, i, values[i])
        nodal_jac = {key: coo_matrix((v[2], (v[0], v[1])), shape=(N, N)).tocsr()
                     for key, v in derivative_entries.items()}
        ra_nodal = {var: np.column_stack((np.zeros(N), self.absorption if var in {"n_m3", "p_m3"} else np.zeros(N)))
                    for var in rates}
        blocks = [hstack([nodal_jac.get((var, v.id), csr_matrix((N, N)))[nodes]
                          for v in self.layout.variables], format="csr")
                  for _, var, nodes in (self.rate_fields if derivatives else ())]
        rate = a.concatenate([rates[var][nodes] for _, var, nodes in self.rate_fields])
        ra = np.vstack([ra_nodal[var][nodes] for _, var, nodes in self.rate_fields])
        linear = {name: self.linear_action(name, point)
                  for name in ("charge_density", "displacement", "constraints")}
        return AffineDeviceEvaluation(
            pack(rate), linear["constraints"].value, MappingProxyType({k: pack(v) for k, v in rates.items()}),
            MappingProxyType(nodal_jac), MappingProxyType({k: frozen_array(v) for k, v in ra_nodal.items()}),
            vstack(blocks, format="csr") if derivatives else None, frozen_array(ra),
            linear["charge_density"].value, linear["displacement"].value, pack(jn), pack(jp), pack(fi),
            pack(DD(np.zeros(2))), pack(rn), pack(rp),
            pack(rn0), pack(rp0), pack(drn), pack(drp), MappingProxyType(linear),
        )

    def storage_value(self, point):
        self.validate(point)
        return self.linear_action("storage", point).value

    def storage_delta(self, left, right, increment):
        increment.validate(left, right)
        self.validate(left); self.validate(right)
        return self.linear_action("storage", right, increment=increment, left=left).value

    def residual(self, point, derivative, input_rate):
        rate = self.physical_rate_view(point, derivative, input_rate)
        evaluation = self.observation_evaluation(point)
        balance = self.linear_action("balance", point, rate=rate, evaluation=evaluation)
        return self.arithmetic.freeze(self.arithmetic.concatenate(
            (balance.value.as_dd(), evaluation.values.algebraic.as_dd())))

    def conservative_residual(self, left, right, increment):
        if right.time <= left.time:
            raise ContractError("nonpositive_physical_interval")
        e = self.evaluate(right)
        finite = self.storage_delta(left, right, increment).as_dd()-(right.time-left.time)*e.rate.as_dd()
        return self.arithmetic.freeze(self.arithmetic.concatenate((finite, e.algebraic.as_dd())))

    def common_system(self):
        raise ContractError("affine_device_requires_public_sparse_problem")

    def public_problem(self, *, accepted_rate_mapping_identity=None):
        from scripts.benchmarks.contract_prototype import PhysicalStorage, SparseLinearization, SparseStructure, ValidatedProblem

        structure = SparseStructure.from_graph(self.graph)
        units = tuple(e.unit*SECOND for e in self.layout.equations if e.role == "storage"
                      for _ in range(int(np.prod(e.shape))))
        storage = PhysicalStorage(self.storage_value, self.storage_delta, units,
                                  self.source_identity, self.arithmetic)

        def linearize(point, ydot, adot):
            e = self.evaluate(point, derivatives=True)
            # This physical storage and the external lift are affine. Thus
            # Qyy*ydot, Qya*adot, Qyt and their input/time counterparts vanish
            # here only. Nonlinear storage keeps the full public contractions.
            fy = self._assemble_declared(vstack((-e.rate_jacobian, self.G), format="csr"))
            fydot = self._assemble_declared(self.mass.tocsr())
            fa = np.vstack((-e.input_jacobian, self.Ga))
            return SparseLinearization(fy, fydot, fa, np.zeros_like(fa),
                                       np.zeros(self.layout.size), structure, self.source_identity)

        return ValidatedProblem(self.layout, 2, self.dynamic_count, storage, structure,
                                self.source_identity, self.residual, linearize,
                                self.conservative_residual, self.conservative_jacobian,
                                point_validator=self.validate, arithmetic=self.arithmetic,
                                accepted_rate_mapping_identity=accepted_rate_mapping_identity)

    def observation_evaluation(self, point, *, state_actions=False):
        """Carry one point-bound value; optionally prepare shared observation actions.

        Residual-only callers keep the value-only path. Raw/tangent observation
        pairs may share the three state actions; rate actions remain separate.
        """
        values = self.evaluate(point)
        if state_actions:
            linear = dict(values.linear_actions)
            linear.update({name: self.linear_action(name, point)
                           for name in ("metal_charge", "body_charge", "gauss_defect")})
            values = replace(values, linear_actions=MappingProxyType(linear))
        return AffinePointEvaluation(point, self.source_identity, values)

    def evaluated_values(self, point, evaluation=None):
        if evaluation is None:
            return self.evaluate(point)
        if (not isinstance(evaluation, AffinePointEvaluation)
                or evaluation.source_identity != self.source_identity
                or evaluation.point.identity != point.identity):
            raise ContractError("foreign_or_stale_observation_evaluation")
        return evaluation.values

    def tangent_rate(self, point, input_rate, *, evaluation=None):
        rate = frozen_array(input_rate)
        if rate.shape != (2,):
            raise ContractError("device_input_rate_shape")
        e, result = self.evaluated_values(point, evaluation), np.zeros(self.layout.size)
        for eq, var, nodes in self.rate_fields:
            coefficient = self.geometry.volumes[nodes]*(self.definition.trap_density if var == "f" else 1)
            value = e.rate.take(np.arange(self.graph.row_offsets[eq].start, self.graph.row_offsets[eq].stop))
            result[self.layout.offsets[var].start+nodes] = self.project_output(value)/coefficient
        result[self.layout.offsets["phi_V"]] = self.phi_factor.solve(
            -(self.G[:self.count] @ result+self.Ga[:self.count] @ rate))
        return frozen_array(result)

    def observe(self, point, derivative, input_rate, origin="algebraic_probe", side="continuous", *, evaluation=None):
        if origin not in {"native", "interpolant", "physical_tangent", "algebraic_probe",
                           "stop_output", "endpoint_restore", "segment_initial", "declared_polynomial"}:
            raise ContractError("unknown_derivative_origin")
        if side not in {"continuous", "left", "right"}:
            raise ContractError("unknown_event_side")
        rate = self.physical_rate_view(point, derivative, input_rate)
        input_rate = rate.input_rate
        evaluation = self.observation_evaluation(point, state_actions=True) if evaluation is None else evaluation
        e = self.evaluated_values(point, evaluation)
        linear = dict(e.linear_actions)
        linear.update({name: self.linear_action(name, point)
                       for name in ("metal_charge", "body_charge", "gauss_defect")
                       if name not in linear})
        linear.update({name+"_rate": self.linear_action(name, point, rate=rate, evaluation=evaluation)
                       for name in ("body_charge", "metal_charge", "constraints", "conduction",
                                    "total_current", "charge_integrands", "interior_total_current")})
        scalar = lambda name: linear[name].value.take(np.asarray(0))
        kept_derivative = derivative if isinstance(derivative, RateView) else frozen_array(derivative)
        return AffineDeviceObservation(
            point, kept_derivative, input_rate, origin, side,
            linear["metal_charge"].value, linear["conduction_rate"].value,
            linear["total_current_rate"].value, scalar("body_charge"), scalar("body_charge_rate"),
            scalar("gauss_defect"), self.project_output(linear["constraints_rate"].value),
            linear["interior_total_current_rate"].value, linear["charge_integrands_rate"].value,
            MappingProxyType(linear),
        )

    def finite_physical_changes(self, left, right, increment):
        """Exact linear charge/storage and finite capture changes, no stepping."""
        from scripts.benchmarks.precision_prototype import DD, DoubleArray

        increment.validate(left, right)
        self.validate(left); self.validate(right)
        pack, m = DoubleArray.from_dd, self.definition
        dn, dp = (increment.field(name).as_dd() for name in ("n_m3", "p_m3"))
        n0, p0 = (self.field(left, name).as_dd() for name in ("n_m3", "p_m3"))
        n1, p1 = (self.field(right, name).as_dd() for name in ("n_m3", "p_m3"))
        if m.dynamic:
            df = increment.field("f").as_dd()
            f0, f1 = self.field(left, "f").as_dd(), self.field(right, "f").as_dd()
            drn = m.capture_n*((1-f1)*dn-(n0+m.n1)*df)
            drp = m.capture_p*(f1*dp+(p0+m.p1)*df)
        else:
            rn0 = self._capture(left)[0]
            lam1 = m.capture_n*(n1+m.n1)+m.capture_p*(p1+m.p1)
            drn = drp = (m.capture_n*m.capture_p*(n0*dp+p1*dn)
                         -rn0*(m.capture_n*dn+m.capture_p*dp))/lam1
        linear = {name: self.linear_action(name, right, increment=increment, left=left)
                  for name in ("storage", "charge_density", "displacement", "body_charge", "metal_charge")}
        return MappingProxyType({
            "storage": linear["storage"].value,
            "capture_n_per_s": pack(drn), "capture_p_per_s": pack(drp),
            "charge_density_C_m3": linear["charge_density"].value,
            "displacement_C_m2": linear["displacement"].value,
            "body_charge_C": linear["body_charge"].value.take(np.asarray(0)),
            "metal_charge_C": linear["metal_charge"].value,
            "linear_actions": MappingProxyType(linear),
        })

    def affine_constraint_error(self, point, *, evaluation=None):
        """A finite affine defect estimate at unchanged physical populations."""
        g = self.project_output(self.evaluated_values(point, evaluation).algebraic)
        delta_phi = self.phi_factor.solve(-g[:self.count])
        induced_D = -self.definition.epsilon*np.diff(delta_phi)/self.dx
        contact_delta = -g[self.count:]
        end = np.array([0,self.count-1])
        rho_delta = Q*(contact_delta[2:]-contact_delta[:2])
        width = self.geometry.volumes[end]/self.definition.area
        charges = self.definition.area*np.array([
            induced_D[0]-rho_delta[0]*width[0],
            -induced_D[-1]-rho_delta[1]*width[1]])
        return {
            "point_identity":point.identity,"state_changed":False,
            "potential_correction_V":delta_phi.tolist(),
            "potential_error_bound_V":float(np.max(np.abs(delta_phi))),
            "metal_charge_correction_C":charges.tolist(),
            "metal_charge_error_bound_C":np.abs(charges).tolist(),
            "body_charge_correction_C":float(self.geometry.volumes[end] @ rho_delta),
            "contact_density_correction_m3":contact_delta.tolist(),
            "preserves_differential_storage":True,"factorizations":self.factorizations,
            "nonlinear_remainder":0.0,"acceptance_awarded":False,
            "linear_solve_residual":float(np.max(np.abs(
                self.G[:self.count,self.layout.offsets["phi_V"]] @ delta_phi+g[:self.count]))),
            "residual_projection":"explicit DD-to-binary64 for the affine estimator only",
        }

    def numeric_packet(self):
        from scripts.benchmarks.precision_prototype import encode_point

        packet = super().numeric_packet()
        packet.pop("initial_y_physical")
        packet.update({
            "representation": "fixed-original-reference-physical-SI-remainders",
            "initial_reference": encode_point(self.reference),
            "initial_remainders_SI": self.reference.y.tolist(),
            "physical_reference_fields": {
                v.id: {"high":self.field(self.reference,v.id).high.tolist(),
                       "low":self.field(self.reference,v.id).low.tolist()}
                for v in self.layout.variables},
            "reference_residual_zeroed": False,
            "physical_mass_and_derivative_units_unchanged": True,
            "native_run_admitted": False,
        })
        return packet


@dataclass(frozen=True)
class AffineScaledIDAAdapter:
    """Algebra only. IDA would integrate S*z remainders from the fixed root."""

    model: AffineCoupledSlab
    segment: ProtocolSegment
    problem: object = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "problem", self.model.public_problem())

    def point(self, time, z):
        inputs, _ = self.segment.inputs(time)
        return self.model.trial(self.model.S*np.asarray(z), time, inputs)[0]

    def residual(self, time, z, zdot):
        point = self.point(time, z)
        _, adot = self.segment.inputs(time)
        value = self.problem.residual(point, self.model.S*np.asarray(zdot), adot)
        return self.model.Drow*self.model.project_output(value)

    def jacobian(self, time, z, cj, zdot=None):
        point = self.point(time, z)
        ydot = np.zeros_like(z) if zdot is None else self.model.S*np.asarray(zdot)
        _, adot = self.segment.inputs(time)
        physical = self.problem.linearize(point, ydot, adot).ida_matrix(cj)
        columns = np.repeat(np.arange(physical.shape[1]), np.diff(physical.indptr))
        values = physical.data*self.model.Drow[physical.indices]*self.model.S[columns]
        return csc_matrix((values,physical.indices.copy(),physical.indptr.copy()),shape=physical.shape)


class AffineDeviceHistory:
    """A shared reference once, then replayable public trial inputs and rates.

    Restoration uses decode_point for the reference and the public trial
    builder for samples. It never edits a generic authority payload or infers
    hidden solver values from projected physical fields.
    """

    def __init__(self, model: AffineCoupledSlab):
        from scripts.benchmarks.precision_prototype import encode_point

        self.model = model
        self.reference_record = {
            "schema": "solarlab.affine-device-reference.v1",
            "source_identity": model.source_identity,
            "layout_identity": model.layout.identity,
            "reference": encode_point(model.reference),
            "reference_point_identity": model.reference.identity,
            "column_scaling_hex": [float(x).hex() for x in model.S],
            "primitive_units": "physical SI cumulative remainders; no transformed storage",
        }
        self.reference_digest = digest(self.reference_record)

    def sample(self, point, cumulative_increment, predecessor, physical_ydot, input_rate, *,
               origin="algebraic_probe", event_side="continuous", raw_z=None, raw_zdot=None):
        from scripts.benchmarks.precision_prototype import PrimitiveExpansion

        self.model.validate(point)
        primitive = PrimitiveExpansion.from_value(cumulative_increment)
        if primitive.shape != (self.model.layout.size,):
            raise ContractError("history_primitive_shape")
        rebuilt, _ = self.model.trial(primitive, point.time, point.inputs, predecessor=predecessor)
        if rebuilt.identity != point.identity:
            raise ContractError("history_primitive_point_mismatch")
        return self._sample_record(point, primitive, predecessor, physical_ydot, input_rate,
                                   origin=origin, event_side=event_side,
                                   raw_z=raw_z, raw_zdot=raw_zdot)

    def build_sample(self, cumulative_increment, time, inputs, predecessor, physical_ydot,
                     input_rate, *, origin="algebraic_probe", event_side="continuous",
                     raw_z=None, raw_zdot=None):
        """Construct and serialize one public trial without accepting a supplied Point."""
        from scripts.benchmarks.precision_prototype import PrimitiveExpansion

        primitive = PrimitiveExpansion.from_value(cumulative_increment)
        if primitive.shape != (self.model.layout.size,):
            raise ContractError("history_primitive_shape")
        point, increment = self.model.trial(primitive, time, inputs, predecessor=predecessor)
        self.model.validate(point)
        record = self._sample_record(point, primitive, predecessor, physical_ydot, input_rate,
                                     origin=origin, event_side=event_side,
                                     raw_z=raw_z, raw_zdot=raw_zdot)
        return point, increment, record

    def _sample_record(self, point, primitive, predecessor, physical_ydot, input_rate, *,
                       origin, event_side, raw_z, raw_zdot):
        """Serialize only after sample verified, or build_sample constructed, the Point."""
        physical_ydot, input_rate = frozen_array(physical_ydot), frozen_array(input_rate)
        if physical_ydot.shape != point.y.shape or input_rate.shape != (2,):
            raise ContractError("history_rate_shape")
        if origin not in {"native", "interpolant", "physical_tangent", "algebraic_probe",
                           "stop_output", "endpoint_restore", "segment_initial"}:
            raise ContractError("unknown_derivative_origin")
        if event_side not in {"continuous", "left", "right"}:
            raise ContractError("unknown_event_side")
        if (raw_z is None) != (raw_zdot is None):
            raise ContractError("incomplete_raw_solver_pair")
        actual_native_origin = origin in {"native", "interpolant", "stop_output", "endpoint_restore", "segment_initial"}
        if actual_native_origin and raw_z is None:
            raise ContractError("raw_solver_values_required")
        solver = None
        if raw_z is not None:
            raw_z, raw_zdot = frozen_array(raw_z), frozen_array(raw_zdot)
            if raw_z.shape != point.y.shape or raw_zdot.shape != point.y.shape:
                raise ContractError("raw_solver_shape")
            if (not np.array_equal(self.model.S*raw_z, point.y)
                    or not np.array_equal(self.model.S*raw_zdot, physical_ydot)
                    or any(np.any(word != 0) for word in primitive.words[1:])):
                raise ContractError("raw_solver_physical_map_mismatch")
            solver = {"z_hex":[float(x).hex() for x in raw_z],
                      "zdot_hex":[float(x).hex() for x in raw_zdot],
                      "mapping":"u=S*z; physical_ydot=S*zdot, fixed original reference"}
        return {
            "schema":"solarlab.affine-device-sample.v1",
            "source_identity":self.model.source_identity,
            "reference_digest":self.reference_digest,
            "reference_point_identity":self.model.reference.identity,
            "point_identity":point.identity, "predecessor_identity":predecessor.identity,
            "time_hex":float(point.time).hex(), "input_hex":[float(x).hex() for x in point.inputs],
            "cumulative_primitive_words_hex":[[float(x).hex() for x in word] for word in primitive.words],
            "physical_ydot_hex":[float(x).hex() for x in physical_ydot],
            "input_rate_hex":[float(x).hex() for x in input_rate],
            "origin":origin, "event_side":event_side, "raw_solver":solver,
            "reference_embedded":False,
        }

    def restore(self, reference_record, sample, predecessor):
        from scripts.benchmarks.precision_prototype import PrimitiveExpansion, decode_point

        if (digest(reference_record) != self.reference_digest
                or sample.get("reference_digest") != self.reference_digest
                or sample.get("source_identity") != self.model.source_identity
                or sample.get("reference_point_identity") != self.model.reference.identity
                or sample.get("predecessor_identity") != predecessor.identity):
            raise ContractError("history_reference_or_predecessor_mismatch")
        reference = decode_point(reference_record["reference"], self.model.layout)
        if reference.identity != self.model.reference.identity:
            raise ContractError("history_reference_decode_mismatch")
        primitive = PrimitiveExpansion([
            np.array([float.fromhex(x) for x in word], dtype=float)
            for word in sample["cumulative_primitive_words_hex"]])
        point, increment = self.model.coordinates.trial(
            reference, primitive, float.fromhex(sample["time_hex"]),
            [float.fromhex(x) for x in sample["input_hex"]], predecessor=predecessor,
            transition_representation="four-word-v1")
        if point.identity != sample.get("point_identity"):
            raise ContractError("history_restored_identity_mismatch")
        rate = np.array([float.fromhex(x) for x in sample["physical_ydot_hex"]])
        input_rate = np.array([float.fromhex(x) for x in sample["input_rate_hex"]])
        solver = sample.get("raw_solver")
        regenerated = self.sample(
            point, primitive, predecessor, rate, input_rate,
            origin=sample["origin"], event_side=sample["event_side"],
            raw_z=None if solver is None else [float.fromhex(x) for x in solver["z_hex"]],
            raw_zdot=None if solver is None else [float.fromhex(x) for x in solver["zdot_hex"]])
        if digest(regenerated) != digest(sample):
            raise ContractError("history_sample_noncanonical")
        return point, increment, frozen_array(rate), frozen_array(input_rate)


@dataclass(frozen=True)
class ScaledIDAAdapter:
    """Callback algebra only; construction never initializes or steps IDA."""

    model: CoupledSlab
    segment: ProtocolSegment
    problem: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem", self.model.public_problem())

    def point(self, time, z) -> Point:
        inputs, _ = self.segment.inputs(time)
        return self.model.coordinates.point(self.model.S*np.asarray(z), time, inputs)

    def residual(self, time, z, zdot) -> np.ndarray:
        point = self.point(time, z)
        _, adot = self.segment.inputs(time)
        return self.model.Drow*self.problem.residual(point, self.model.S*np.asarray(zdot), adot)

    def jacobian(self, time, z, cj, zdot=None) -> csc_matrix:
        point = self.point(time, z)
        ydot = np.zeros_like(z) if zdot is None else self.model.S*np.asarray(zdot)
        _, adot = self.segment.inputs(time)
        physical = self.problem.linearize(point, ydot, adot).ida_matrix(cj)
        # Preserve every declared CSC slot, including an instantaneous zero.
        columns = np.repeat(np.arange(physical.shape[1]), np.diff(physical.indptr))
        values = physical.data*self.model.Drow[physical.indices]*self.model.S[columns]
        return csc_matrix((values, physical.indices.copy(), physical.indptr.copy()), shape=physical.shape)

    def finite_storage_increment(self, left: Point, right: Point, increment: StateIncrement) -> np.ndarray:
        # No S here: the public increment is already in physical SI units.
        return self.problem.storage.delta(left, right, increment)


def _qualification_scope(model, segments, policy):
    """The explicit P02 extension; the default N8 route never calls this."""
    from scripts.benchmarks.qualification_policy import validate_policy, validate_layout
    try:
        policy = validate_policy(policy)
        if type(model) not in (CoupledSlab, AffineCoupledSlab):
            raise ValueError("qualification_model_type")
        validate_layout(model.numeric_packet(), policy)
        if digest([asdict(s) for s in segments]) != digest(policy["origin"]["segments"]):
            raise ValueError("qualification_protocol_changed")
        return policy
    except (ValueError, KeyError, TypeError) as error:
        raise ContractError(str(error)) from error


def prepare_native_request(model: CoupledSlab, segments: tuple[ProtocolSegment, ...], *,
                           qualification_policy: Mapping | None = None) -> dict:
    """Construct exact proposed inputs; no backend import or native execution."""
    if isinstance(model, AffineCoupledSlab):
        raise ContractError("affine_native_request_requires_separate_domain_and_weight_policy")
    if qualification_policy is not None:
        qualification_policy = _qualification_scope(model, segments, qualification_policy)
    if (qualification_policy is None and model.intervals != 8) or len(segments) != 3:
        raise ContractError("first_native_pilot_scope")
    m = model.definition
    initial = model.initial()
    z0 = initial.y/model.S
    decoded = ScaledIDAAdapter(model, segments[0]).point(segments[0].start, z0)
    adot = segments[0].inputs(segments[0].start)[1]
    physical_rate = model.tangent_rate(decoded, adot)
    qref = Q*m.area*m.length*max(m.ni, m.ion_initial, m.trap_density)
    jref = Q*m.vt*(m.mu_n+m.mu_p)*m.ni/m.length
    current_budget = m.area*0.005*(Q*m.photon_reference if m.photon_reference else jref)
    quadrature = {}
    if qualification_policy is None:
        for order in [8, 16, 32]:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            quadrature[str(order)] = {"nodes": nodes.tolist(), "weights": weights.tolist()}
        times = {s.id: np.linspace(s.start, s.end, 33) for s in segments}
    else:
        quadrature = qualification_policy["origin"]["quadrature"]
        times = {key: np.asarray(value) for key, value in
                 qualification_policy["origin"]["observation_times"].items()}
    if m.dynamic and qualification_policy is None:
        hold = segments[-1]
        elapsed = np.geomspace(1e-7, 9.0, 33)
        elapsed[-1] = hold.end-hold.start
        times[hold.id] = np.unique(np.r_[times[hold.id], hold.start+elapsed])
    indices, types = [], []
    for field in ["n_m3", "p_m3"]+(["c_m3", "f"] if m.dynamic else []):
        section = model.layout.offsets[field]
        indices.extend(range(section.start, section.stop))
        types.extend([2 if field in {"n_m3", "p_m3"} else 1]*model.count)
    request = {"schema": "solarlab.real-device-native-request.v1", "case_id": m.id,
            "numeric_packet": model.numeric_packet(), "segments": [asdict(s) for s in segments],
            "z0": z0.tolist(), "zdot0": (physical_rate/model.S).tolist(),
            "actual_initial_y": decoded.y.tolist(), "initial_scaling_quantization": (decoded.y-initial.y).tolist(),
            "actual_initial_identity": decoded.identity,
            "controls": {"rtol": 1e-4, "atol": [1e-8]*model.layout.size,
                         "max_step": 0.02 if m.dynamic else 2e-8, "max_order": 5,
                         "max_num_steps": 200000, "max_nonlin_iters": 4,
                         "max_conv_fails": 10, "first_step": 0.0,
                         "calc_initcond": None, "linsolver": "sparse", "nthreads": 1,
                         "constraints_idx": indices, "constraints_type": types},
            "budgets": {"wall_s": 180, "rss_bytes": 2147483648, "native_steps": 200000,
                        "residual_calls": 2000000, "total_output_bytes": 268435456, "charge_C": 1e-10*qref,
                        "current_A": current_budget, "constraint_charge_C": 1e-10*qref/3,
                        "constraint_potential_V": 0.001/3, "contact_relative": 0.01/3,
                        "ion_inventory_relative": 1e-10, "maximum_site_fraction": 0.999},
            "observation_times": {key: values.tolist() for key, values in times.items()},
            "quadrature": quadrature,
            "observation_policy": "retain_raw_and_separately_named_physical_tangent",
            "mandatory": {"state_and_affine_constraints": True, "raw_tangent_current_agreement": True,
                          "both_interval_and_prefix_charge_ledgers": True,
                          "interval_and_prefix_defect_plus_uncertainty": True,
                          "interval_and_prefix_reference_share": True},
            "quadrature_uncertainty_scope": "absolute16/32 discrepancy is a declared diagnostic estimate, not a rigorous continuum certificate",
            "refinement_or_full_device_qualification": False,
            "uncertified_by_first_pilot": ["time and mesh convergence", "independent trajectory state error",
                                           "full real-device current accuracy", "R1 research precision"],
            "admission": "requires a separate coordinator message bound to this exact request digest"}
    if qualification_policy is not None:
        from scripts.benchmarks.qualification_policy import raw_controls
        request.update(qualification_policy=qualification_policy,
                       controls=raw_controls(qualification_policy, request["numeric_packet"]),
                       budgets=dict(qualification_policy["origin"]["budgets"]),
                       mandatory=dict(qualification_policy["origin"]["mandatory"]))
    return request


def snapshot_solver_result(result) -> dict:
    """Copy before interpolation can mutate a binding's borrowed buffers."""
    return {"time": float(result.t), "z": frozen_array(result.y), "zdot": frozen_array(result.yp),
            "success": bool(result.success), "status": int(result.status), "message": str(result.message)}


def charge_interval_bound(delta_charge, integral_estimates, previous_defects,
                          previous_uncertainties, charge_budget: float) -> dict:
    """Componentwise combined interval/prefix accounting in physical coulombs.

    The 16/32 difference is an explicitly limited reference estimate. Passing
    this allocation neither proves quadrature convergence nor bounds the
    continuum trajectory. Each body/metal component retains its own ledger.
    """
    delta = frozen_array(delta_charge)
    estimates = frozen_array(integral_estimates)
    prior_d, prior_u = frozen_array(previous_defects), frozen_array(previous_uncertainties)
    if (delta.ndim != 1 or estimates.shape != (3, delta.size) or prior_d.shape != delta.shape
            or prior_u.shape != delta.shape or np.any(prior_d < 0) or np.any(prior_u < 0)
            or not np.isfinite(charge_budget) or charge_budget <= 0):
        raise ContractError("invalid_charge_error_accounting")
    defect = np.abs(delta-estimates[-1])
    uncertainty = np.abs(estimates[-1]-estimates[-2])
    cumulative_d, cumulative_u = prior_d+defect, prior_u+uncertainty
    interval_total, prefix_total = defect+uncertainty, cumulative_d+cumulative_u
    checks = {"interval_total": bool(np.all(interval_total <= charge_budget)),
              "prefix_total": bool(np.all(prefix_total <= charge_budget)),
              "interval_reference_share": bool(np.all(uncertainty <= charge_budget/3)),
              "prefix_reference_share": bool(np.all(cumulative_u <= charge_budget/3))}
    return {"delta_charge_C": delta.tolist(), "integrals_C": estimates.tolist(),
            "absolute_defect_C": defect.tolist(), "quadrature_uncertainty_estimate_C": uncertainty.tolist(),
            "cumulative_absolute_defect_C": cumulative_d.tolist(),
            "cumulative_quadrature_uncertainty_estimate_C": cumulative_u.tolist(),
            "interval_total_bound_C": interval_total.tolist(), "prefix_total_bound_C": prefix_total.tolist(),
            "reference_estimate": "absolute16/32 difference; not a rigorous continuum certificate",
            "charge_budget_C": charge_budget, "reference_allocation_C": charge_budget/3,
            "checks": checks, "passed": all(checks.values())}


def state_quality_evidence(model: CoupledSlab, point: Point, budgets: Mapping) -> dict:
    """Physical finite-g eligibility at one unchanged state, not LTE proof."""
    model.validate(point)
    m = model.definition
    bound = model.affine_constraint_error(point)
    contacts = np.asarray(bound["contact_density_residual_m3"])/np.array([m.n_eq, m.n_eq, m.p_eq, m.p_eq])
    charge_error = max(sum(abs(v) for v in bound["Poisson_constraint_charge_C"]),
                       max(bound["metal_charge_error_bound_C"]), bound["body_charge_error_bound_C"])
    metrics = {"minimum_n_m3": float(np.min(model.field(point, "n_m3"))),
               "minimum_p_m3": float(np.min(model.field(point, "p_m3"))),
               "constraint_potential_bound_V": bound["potential_error_bound_V"],
               "constraint_charge_bound_C": charge_error,
               "constraint_details": bound,
               "contact_max_relative": float(np.max(np.abs(contacts))), "state_identity": point.identity,
               "trajectory_error_certified": False}
    passed = (bound["potential_error_bound_V"] <= budgets["constraint_potential_V"]
              and charge_error <= budgets["constraint_charge_C"]
              and np.max(np.abs(contacts)) <= budgets["contact_relative"])
    if m.dynamic:
        c, f = model.field(point, "c_m3"), model.field(point, "f")
        inventory = float(model.geometry.volumes @ c)
        reference = m.area*m.length*m.ion_initial
        metrics.update(ion_inventory_relative=abs(inventory/reference-1),
                       maximum_site_fraction=float(np.max(c/m.ion_capacity)),
                       minimum_f=float(np.min(f)), maximum_f=float(np.max(f)))
        passed = passed and metrics["ion_inventory_relative"] <= budgets["ion_inventory_relative"] and metrics["maximum_site_fraction"] <= budgets["maximum_site_fraction"]
    metrics["passed"] = bool(passed)
    return metrics


def observation_payload(reading: DeviceObservation) -> dict:
    return {"time_s": reading.point.time, "point_identity": reading.point.identity,
            "coordinate_reference": reading.point.coordinate_reference,
            "physical_y": reading.point.y.tolist(), "inputs": reading.point.inputs.tolist(),
            "physical_ydot": reading.derivative.tolist(), "input_rates": reading.input_rate.tolist(),
            "origin": reading.origin, "event_side": reading.ports.side,
            "metal_charge_C": reading.ports.charge.tolist(), "conduction_inward_A": reading.ports.conduction.tolist(),
            "total_inward_A": reading.ports.current.tolist(), "body_charge_C": reading.body_charge,
            "body_charge_rate_A": reading.body_charge_rate, "gauss_defect_C": reading.gauss_defect,
            "tangent_residual": reading.tangent_residual.tolist(),
            "interior_total_current_A_m2": reading.interior_total_current.tolist()}


def run_native_pilot(model: CoupledSlab, segments: tuple[ProtocolSegment, ...], request: Mapping,
                     admission: Mapping, emit) -> dict:
    """One explicitly admitted pilot; all native/normal rates remain distinct.

    No automatic retries, preparation replay, state projection or generic time
    integrator. ``emit`` persists every record before the next native interval.
    Both raw and physical-tangent ledgers are mandatory in this first proposal.
    """
    if isinstance(model, AffineCoupledSlab):
        raise ContractError("affine_native_runner_not_admitted")
    request_id = digest(dict(request))
    if (admission.get("request_sha256") != request_id
            or not str(admission.get("coordinator_message", "")).startswith("msg_")):
        raise ContractError("native_pilot_not_admitted")
    for path, expected in admission.get("source_sha256", {}).items():
        if sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ContractError("native_source_changed_after_admission")
    if not admission.get("source_sha256"):
        raise ContractError("native_admission_missing_source_binding")
    if str(Path(__file__).resolve()) not in admission["source_sha256"]:
        raise ContractError("native_admission_does_not_bind_executing_kernel")
    if model.intervals != 8 or model.layout.size > 45:
        raise ContractError("first_native_pilot_scope")
    if digest(model.numeric_packet()) != digest(request["numeric_packet"]):
        raise ContractError("native_numeric_packet_changed")
    if [asdict(s) for s in segments] != request["segments"]:
        # JSON round trips turn tuples into lists; compare canonical JSON.
        if digest([asdict(s) for s in segments]) != digest(request["segments"]):
            raise ContractError("native_protocol_changed")
    from sksundae.ida import IDA

    started = time.perf_counter()
    budgets, controls = request["budgets"], request["controls"]
    counts = {"residual": 0, "jacobian": 0, "native_steps": 0, "normal_queries": 0}
    cumulative = {key: np.zeros(3) for key in ["raw", "physical_tangent"]}
    cumulative_uncertainty = {key: np.zeros(3) for key in cumulative}
    failure_context = None
    last_attempt = None
    last_numerical = None
    last_accepted = None
    z = np.asarray(request["z0"], float)
    rows, columns = model.graph.edges().T
    sparsity = csc_matrix((np.ones(len(rows)), (rows, columns)), shape=model.graph.shape)
    differential = np.unique(model.mass.tocoo().col)
    algebraic = np.setdiff1d(np.arange(model.layout.size), differential).tolist()

    def resource_check():
        if time.perf_counter()-started > budgets["wall_s"]:
            raise ContractError("native_wall_budget")
        if counts["residual"] > budgets["residual_calls"]:
            raise ContractError("native_residual_budget")
        if counts["native_steps"] > budgets["native_steps"]:
            raise ContractError("native_step_budget")
        # The frozen runner is macOS; its independent watchdog also measures
        # resident memory and can terminate a stuck native call.
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > budgets["rss_bytes"]:
            raise ContractError("native_rss_budget")

    def state_evidence(point):
        return state_quality_evidence(model, point, budgets)

    try:
        for segment_index, segment in enumerate(segments):
            adapter = ScaledIDAAdapter(model, segment)

            def residual(t, values, rates, output):
                nonlocal failure_context
                counts["residual"] += 1
                try:
                    resource_check()
                    output[:] = adapter.residual(t, values, rates)
                except BaseException as error:
                    if failure_context is None:
                        failure_context = {"phase": "residual", "time": float(t), "z": np.asarray(values).tolist(),
                                           "zdot": np.asarray(rates).tolist(), "reason": str(error)}
                    raise

            def jacobian(t, values, rates, residual_value, cj, output):
                counts["jacobian"] += 1
                resource_check()
                matrix = adapter.jacobian(t, values, cj, rates)
                if (matrix.nnz != model.graph.nnz or not np.array_equal(matrix.indices, model.graph.indices)
                        or not np.array_equal(matrix.indptr, model.graph.indptr) or output.shape != (model.graph.nnz,)):
                    raise ContractError("native_csc_slot_mismatch")
                output[:] = matrix.data

            initial_point = adapter.point(segment.start, z)
            _, adot = segment.inputs(segment.start)
            initial_rate = model.tangent_rate(initial_point, adot)
            initial_zdot = np.asarray(request["zdot0"]) if segment_index == 0 else initial_rate/model.S
            last_attempt = {"phase": "segment_initialization", "segment": segment.id,
                            "time": segment.start, "z": z.tolist(), "zdot": initial_zdot.tolist()}
            solver = IDA(residual, jacfn=jacobian, sparsity=sparsity,
                         algebraic_idx=algebraic, **controls)
            initial_result = solver.init_step(segment.start, z, initial_zdot)
            last_attempt = {"phase": "initialization_return", "segment": segment.id,
                            "time_hex": float(initial_result.t).hex(), "success": bool(initial_result.success),
                            "status": int(initial_result.status), "message": str(initial_result.message),
                            "z_hex": [float(v).hex() for v in initial_result.y],
                            "zdot_hex": [float(v).hex() for v in initial_result.yp]}
            initialized = snapshot_solver_result(initial_result)
            if (not initialized["success"] or initialized["time"] != segment.start
                    or not np.array_equal(initialized["z"], z) or not np.array_equal(initialized["zdot"], initial_zdot)):
                raise ContractError("native_initialization_changed_state_or_failed")
            left = adapter.point(segment.start, initialized["z"])
            left_raw = model.observe(left, model.S*initialized["zdot"], adot, "segment_initial",
                                     "right" if segment_index else "continuous")
            if last_numerical is not None and not np.array_equal(left.y, np.asarray(last_numerical["physical_y"])):
                raise ContractError("event_restart_changed_physical_state")
            metrics = state_evidence(left)
            left_tangent = model.observe(left, initial_rate, adot, "physical_tangent", left_raw.ports.side)
            initial_current_gap = float(np.max(np.abs(left_raw.ports.current-left_tangent.ports.current)))
            emit({"kind": "segment_initial", "segment": segment.id, "reading": observation_payload(left_raw),
                  "physical_tangent": observation_payload(left_tangent), "state_checks": metrics,
                  "current_rate_gap_A": initial_current_gap})
            if not metrics["passed"]: raise ContractError("native_initial_state_quality")
            if initial_current_gap > budgets["current_A"]/3: raise ContractError("initial_raw_tangent_current_budget")
            last_numerical = observation_payload(left_raw)
            if last_accepted is None: last_accepted = last_numerical
            sample_times = np.asarray(request["observation_times"][segment.id])
            sample_cursor = 0
            while sample_cursor < len(sample_times) and sample_times[sample_cursor] == segment.start:
                emit({"kind": "requested_sample", "segment": segment.id, "raw": observation_payload(left_raw),
                      "physical_tangent": observation_payload(left_tangent)})
                sample_cursor += 1
            while left.time < segment.end:
                resource_check()
                if counts["native_steps"] >= budgets["native_steps"]: raise ContractError("native_step_budget")
                last_attempt = {"phase": "native_step", "segment": segment.id, "target_time": segment.end,
                                "previous": last_numerical}
                result = solver.step(segment.end, method="onestep", tstop=segment.end)
                last_attempt = {"phase": "native_return", "segment": segment.id,
                                "time_hex": float(result.t).hex(), "success": bool(result.success),
                                "status": int(result.status), "message": str(result.message),
                                "z_hex": [float(v).hex() for v in result.y],
                                "zdot_hex": [float(v).hex() for v in result.yp]}
                native = snapshot_solver_result(result)
                if not native["success"]: raise ContractError("native_solver_failure:"+native["message"])
                t = native["time"]
                if not left.time < t <= segment.end: raise ContractError("native_interval_not_monotone")
                counts["native_steps"] += 1
                right = adapter.point(t, native["z"])
                _, rate_a = segment.inputs(t)
                origin = "stop_output" if native["status"] == 1 else "native"
                right_raw = model.observe(right, model.S*native["zdot"], rate_a, origin,
                                          "left" if t == segment.end else "continuous")
                last_numerical = observation_payload(right_raw)
                right_tangent = model.observe(right, model.tangent_rate(right, rate_a), rate_a,
                                              "physical_tangent", right_raw.ports.side)
                cache = {left.time: (left_raw, left_tangent), t: (right_raw, right_tangent)}
                active = True

                def checked_pair(raw, tangent):
                    nonlocal failure_context
                    resource_check()
                    metrics = state_evidence(raw.point)
                    current_gap = float(np.max(np.abs(raw.ports.current-tangent.ports.current)))
                    emit({"kind": "rate_pair", "segment": segment.id, "interval": counts["native_steps"],
                          "raw": observation_payload(raw), "physical_tangent": observation_payload(tangent),
                          "state_checks": metrics, "current_rate_gap_A": current_gap})
                    if not metrics["passed"]:
                        failure_context = {"phase": "state_quality", "reading": observation_payload(raw), "metrics": metrics}
                        raise ContractError("native_or_interpolant_state_quality")
                    if current_gap > budgets["current_A"]/3:
                        failure_context = {"phase": "raw_tangent_current", "raw": observation_payload(raw),
                                           "physical_tangent": observation_payload(tangent), "gap_A": current_gap}
                        raise ContractError("raw_tangent_current_budget")

                checked_pair(right_raw, right_tangent)

                def query(time_value):
                    if not active or not left.time <= time_value <= t:
                        raise ContractError("expired_or_foreign_device_interval")
                    if time_value in cache: return cache[time_value]
                    resource_check()
                    normal = solver.step(time_value, method="normal", tstop=segment.end)
                    counts["normal_queries"] += 1
                    if not normal.success or float(normal.t) != time_value:
                        raise ContractError("native_dense_query_failed")
                    p = adapter.point(time_value, normal.y)
                    _, a = segment.inputs(time_value)
                    raw = model.observe(p, model.S*np.asarray(normal.yp), a, "interpolant")
                    tangent = model.observe(p, model.tangent_rate(p, a), a, "physical_tangent")
                    cache[time_value] = (raw, tangent)
                    checked_pair(raw, tangent)
                    return raw, tangent

                try:
                    integrals = {"raw": [], "physical_tangent": []}
                    for order in [8, 16, 32]:
                        table = request["quadrature"][str(order)]
                        nodes, weights = np.asarray(table["nodes"]), np.asarray(table["weights"])
                        h = t-left.time
                        totals = {"raw": np.zeros(3), "physical_tangent": np.zeros(3)}
                        for node, weight in zip(nodes, weights):
                            when = left.time+h*(float(node)+1)/2
                            for key, reading in zip(["raw", "physical_tangent"], query(when)):
                                current = np.r_[sum(reading.ports.conduction), reading.ports.current-reading.ports.conduction]
                                totals[key] += (h*float(weight)/2)*current
                        for key in totals: integrals[key].append(totals[key])
                    delta = np.r_[right_raw.body_charge-left_raw.body_charge,
                                  right_raw.ports.charge-left_raw.ports.charge]
                    evidence = {}
                    for key in integrals:
                        evidence[key] = charge_interval_bound(delta, integrals[key], cumulative[key],
                                                             cumulative_uncertainty[key], budgets["charge_C"])
                        cumulative[key] = np.asarray(evidence[key]["cumulative_absolute_defect_C"])
                        cumulative_uncertainty[key] = np.asarray(evidence[key]["cumulative_quadrature_uncertainty_estimate_C"])
                    while sample_cursor < len(sample_times) and sample_times[sample_cursor] <= t:
                        raw, tangent = query(float(sample_times[sample_cursor]))
                        emit({"kind": "requested_sample", "segment": segment.id, "raw": observation_payload(raw),
                              "physical_tangent": observation_payload(tangent)})
                        sample_cursor += 1
                    restored = solver.step(t, method="normal", tstop=segment.end)
                    counts["normal_queries"] += 1
                    if not restored.success or float(restored.t) != t:
                        raise ContractError("native_endpoint_cursor_restore_failed")
                    restored_point = adapter.point(t, restored.y)
                    restored_raw = model.observe(restored_point, model.S*np.asarray(restored.yp), rate_a, "endpoint_restore", right_raw.ports.side)
                    checked_pair(restored_raw, model.observe(restored_point, model.tangent_rate(restored_point, rate_a),
                                                           rate_a, "physical_tangent", right_raw.ports.side))
                    interval = {"kind": "interval_charge", "segment": segment.id, "left_time": left.time,
                                "right_time": t, "numerical_origin": origin, "ledgers": evidence}
                    emit(interval)
                    if not all(v["passed"] for v in evidence.values()):
                        failure_context = interval
                        raise ContractError("native_interval_or_prefix_charge_budget")
                    last_accepted = last_numerical
                finally:
                    active = False
                z = native["z"].copy()
                left, left_raw, left_tangent = right, right_raw, right_tangent
            if sample_cursor != len(sample_times): raise ContractError("missing_frozen_observation")
        return {"status": "completed_bounded_native_pilot", "request_sha256": request_id,
                "counts": counts, "elapsed_s": time.perf_counter()-started,
                "last_numerical": last_numerical, "last_physically_accepted": last_accepted,
                "cumulative_absolute_charge_defects_C": {k:v.tolist() for k,v in cumulative.items()},
                "cumulative_quadrature_uncertainty_estimates_C": {k:v.tolist() for k,v in cumulative_uncertainty.items()},
                "prefix_total_charge_bounds_C": {k:(cumulative[k]+cumulative_uncertainty[k]).tolist() for k in cumulative},
                "complete_protocol": True, "scientific_or_G2_qualification": False}
    except BaseException as error:
        result = {"status": "failed_bounded_native_pilot", "request_sha256": request_id,
                  "reason": str(error), "exception": type(error).__name__, "traceback": traceback.format_exc(),
                  "counts": counts, "elapsed_s": time.perf_counter()-started,
                  "first_failure": failure_context or last_attempt, "last_numerical": last_numerical,
                  "last_physically_accepted": last_accepted,
                  "cumulative_absolute_charge_defects_C": {k:v.tolist() for k,v in cumulative.items()},
                  "cumulative_quadrature_uncertainty_estimates_C": {k:v.tolist() for k,v in cumulative_uncertainty.items()},
                  "prefix_total_charge_bounds_C": {k:(cumulative[k]+cumulative_uncertainty[k]).tolist() for k in cumulative},
                  "cumulative_scope": "includes the first failed attempted interval if one was fully evaluated",
                  "complete_protocol": False, "scientific_or_G2_qualification": False}
        emit({"kind": "first_failure", "failure": result})
        return result


def _fraction_float_bound(value: Fraction, *, upper: bool) -> float:
    """A directed binary64 endpoint for one finite, nonnegative rational."""
    if value < 0:
        raise ContractError("negative_rounding_bound")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError("rounding_bound_overflow")
    exact = Fraction.from_float(result)
    if upper and exact < value:
        result = math.nextafter(result, math.inf)
    elif not upper and exact > value:
        result = math.nextafter(result, -math.inf)
    return result


def affine_wrms_policy(model: AffineCoupledSlab, previous_controls: Mapping) -> dict:
    """Conservative local weights using exact encoded physical-SI inputs.

    Anew+rnew*abs(u) <= Aold+rold*abs(xref+u), including directed
    rounding of atol_z=Anew/S. This is not a global accuracy certificate.
    """
    old_r = Fraction.from_float(float(previous_controls["rtol"]))
    old_atol = np.asarray(previous_controls["atol"], float)
    if old_r <= 0 or old_atol.shape != model.S.shape or np.any(old_atol <= 0):
        raise ContractError("invalid_original_error_weights")
    references = []
    for variable in model.layout.variables:
        value = model.field(model.reference, variable.id)
        references.extend(Fraction.from_float(float(h))+Fraction.from_float(float(l))
                          for h, l in zip(value.high, value.low))
    scales = [Fraction.from_float(float(s)) for s in model.S]
    old_a = [s*Fraction.from_float(float(a)) for s, a in zip(scales, old_atol)]
    upper = [_fraction_float_bound(abs(x), upper=True) for x in references]
    upper_q = [Fraction.from_float(x) for x in upper]
    r_bound = min([old_r]+[a/(2*u) for a, u in zip(old_a, upper_q) if u])
    rnew = _fraction_float_bound(r_bound, upper=False)
    if rnew <= 0:
        raise ContractError("no_positive_representable_conservative_rtol")
    r = Fraction.from_float(rnew)
    rows, atol = [], []
    for index, (x, s, a, u) in enumerate(zip(references, scales, old_a, upper_q)):
        anew = a-r*u
        scaled = _fraction_float_bound(anew/s, upper=False)
        encoded_a = s*Fraction.from_float(scaled)
        checks = {"reference_bound": u >= abs(x), "positive_atol": scaled > 0,
                  "half_absolute_budget": r*u <= a/2, "rtol_not_larger": r <= old_r,
                  "encoded_atol_not_larger": encoded_a <= anew,
                  "triangle_inequality_margin": encoded_a+r*u <= a}
        if not all(checks.values()):
            raise ContractError("conservative_affine_weight_certificate_failed")
        rows.append({"index": index, "reference_exact": str(x), "scale_hex": float(s).hex(),
                     "old_physical_atol_exact": str(a), "reference_upper_hex": float(u).hex(),
                     "new_physical_atol_bound_exact": str(anew),
                     "encoded_physical_atol_exact": str(encoded_a),
                     "atol_z_hex": scaled.hex(), "checks": checks})
        atol.append(scaled)
    return {"schema": "solarlab.affine-wrms-proof.v1", "rtol": rnew, "atol": atol,
            "rtol_hex": rnew.hex(), "rtol_upper_exact": str(r_bound),
            "old_controls_sha256": digest(dict(previous_controls)), "components": rows,
            "bound": "S*atol_z+rnew*abs(u)<=S*old_atol_z+rold*abs(xref+u)",
            "construction": "rnew<=rold; rnew*Uref<=Aold/2; directed-down atol_z=(Aold-rnew*Uref)/S",
            "scope": "exact represented map and local denominators; finite S*z/S*zdot roundoff recorded separately",
            "global_accuracy_or_conservation_certified": False}


def affine_scaling_roundoff(model: AffineCoupledSlab, z, zdot) -> dict:
    """Exact-rational audit of the two finite binary64 adapter products."""
    vectors = {"u": frozen_array(z), "udot": frozen_array(zdot)}
    result = {"map": "binary64(S*z), binary64(S*zdot); not exact real multiplication",
              "certifies_global_physical_error": False}
    for name, values in vectors.items():
        if values.shape != model.S.shape:
            raise ContractError("scaling_audit_shape")
        actual = model.S*values
        if not np.all(np.isfinite(actual)):
            raise ContractError("nonfinite_scaled_physical_input")
        bounds = []
        for i, (s, v, a) in enumerate(zip(model.S, values, actual)):
            error = abs(Fraction.from_float(float(s))*Fraction.from_float(float(v))
                        -Fraction.from_float(float(a)))
            if error:
                bounds.append([i, _fraction_float_bound(error, upper=True).hex()])
        result[name] = {"nonzero_component_error_upper_hex": bounds,
                        "all_other_component_products_exact": True}
    return result


def prepare_affine_native_request(model: AffineCoupledSlab,
                                  segments: tuple[ProtocolSegment, ...],
                                  previous_request: Mapping) -> dict:
    """Freeze one full affine pilot from its retained absolute-state request."""
    if not isinstance(model, AffineCoupledSlab) or len(segments) != 3:
        raise ContractError("affine_first_native_scope")
    old = json.loads(json.dumps(dict(previous_request)))
    qualification_policy = old.get("qualification_policy")
    if qualification_policy is None:
        if model.intervals != 8:
            raise ContractError("affine_first_native_scope")
    else:
        qualification_policy = _qualification_scope(model, segments, qualification_policy)
        from scripts.benchmarks.qualification_policy import check_common, raw_controls
        check_common(old, qualification_policy)
        if old["controls"] != raw_controls(qualification_policy, old["numeric_packet"]):
            raise ContractError("qualification_raw_denominator_origin")
    if old["case_id"] != model.definition.id or digest(old["segments"]) != digest([asdict(s) for s in segments]):
        raise ContractError("affine_source_protocol_changed")
    for key, value in asdict(model.definition).items():
        if key != "source_path" and old["numeric_packet"]["definition"][key] != value:
            raise ContractError("affine_material_or_initial_data_changed:"+key)
    packet = model.numeric_packet()
    for key in ("x_m", "cell_edges_m", "volumes_m3", "face_areas_m2"):
        if old["numeric_packet"][key] != packet[key]:
            raise ContractError("affine_geometry_changed:"+key)
    proof = affine_wrms_policy(model, old["controls"])
    controls = dict(old["controls"])
    controls.update(rtol=proof["rtol"], atol=proof["atol"])
    controls.pop("constraints_idx", None)
    controls.pop("constraints_type", None)
    z0 = np.zeros(model.layout.size)
    initial = AffineScaledIDAAdapter(model, segments[0]).point(segments[0].start, z0)
    initial_rate = model.tangent_rate(initial, segments[0].inputs(segments[0].start)[1])
    zdot0 = initial_rate/model.S
    request = {"schema": "solarlab.affine-native-request.v1", "case_id": model.definition.id,
            "prior_request_sha256": digest(old), "numeric_packet": packet,
            "segments": [asdict(s) for s in segments], "z0": z0.tolist(), "zdot0": zdot0.tolist(),
            "initial_tangent_rate_SI": initial_rate.tolist(),
            "initial_supplied_rate_SI": (model.S*zdot0).tolist(),
            "initial_scaling_roundoff": affine_scaling_roundoff(model, z0, zdot0),
            "actual_initial_identity": initial.identity,
            "controls": controls, "weight_certificate": proof,
            "budgets": old["budgets"], "observation_times": old["observation_times"],
            "quadrature": old["quadrature"], "mandatory": old["mandatory"],
            "observation_policy": old["observation_policy"],
            "quadrature_uncertainty_scope": old["quadrature_uncertainty_scope"],
            "physical_domain_policy": {
                "admissible": "n,p>0; 0<=c<C; 0<=f<=1; signed SI remainders permitted",
                "native_zero_constraints": "omitted: the binding constrains z against zero, not xref+S*z",
                "invalid_callback_trial": "raise a nonrecoverable domain error and retain first raw z/zdot/time; no returned-NaN or clipping",
                "binding_user_recoverable_residual_return": False},
            "history_policy": "one public reference; raw z/zdot, SI primitive/rate words and protocol-bound predecessor for every sample; native and physical_tangent distinct",
            "roundoff_policy": "exact Fraction audit of finite S*z/S*zdot; final DD projection recorded, not claimed as global accuracy",
            "admission": "requires separate exact request/source/environment/wrapper-bound coordinator message",
            "refinement_or_full_device_qualification": False,
            "uncertified_by_first_pilot": old["uncertified_by_first_pilot"]}
    if qualification_policy is not None:
        request.update(qualification_policy=qualification_policy, qualification_ancestor=old)
    return request


def affine_state_quality_evidence(model: AffineCoupledSlab, point: Point, budgets: Mapping, *, evaluation=None) -> dict:
    """Affine finite-g and inventory checks at an unchanged physical state."""
    from scripts.benchmarks.precision_prototype import DD

    m = model.definition
    evaluation = model.observation_evaluation(point) if evaluation is None else evaluation
    values = model.evaluated_values(point, evaluation)
    bound = model.affine_constraint_error(point, evaluation=evaluation)
    g = values.algebraic.as_dd()
    contact = np.asarray(bound["contact_density_correction_m3"])/np.array([m.n_eq, m.n_eq, m.p_eq, m.p_eq])
    poisson = abs(g[:model.count-2]).sum()
    poisson_charge = float(poisson.hi+poisson.lo)
    charge = max(poisson_charge, max(bound["metal_charge_error_bound_C"]),
                 abs(bound["body_charge_correction_C"]))
    metrics = {"state_identity": point.identity, "state_changed": False,
               "constraint_potential_bound_V": bound["potential_error_bound_V"],
               "constraint_charge_bound_C": charge,
               "Poisson_absolute_constraint_charge_C": poisson_charge,
               "metal_charge_error_bound_C": bound["metal_charge_error_bound_C"],
               "body_charge_error_bound_C": abs(bound["body_charge_correction_C"]),
               "contact_max_relative": float(np.max(np.abs(contact))),
               "factorizations": model.factorizations, "nonlinear_remainder": 0.0,
               "trajectory_error_certified": False}
    passed = (metrics["constraint_potential_bound_V"] <= budgets["constraint_potential_V"]
              and charge <= budgets["constraint_charge_C"]
              and metrics["contact_max_relative"] <= budgets["contact_relative"])
    if m.dynamic:
        c, f = (model.field(point, name).as_dd() for name in ("c_m3", "f"))
        change = model.linear_action("ion_inventory_change", point).value.as_dd()[0]
        reference = model.linear_action("ion_inventory", model.reference).value.as_dd()[0]
        relative = abs(change/reference)
        ratio = c/m.ion_capacity
        metrics.update(ion_inventory_relative=float(relative.hi+relative.lo),
                       ion_inventory_change_words=[float(change.hi), float(change.lo)],
                       maximum_site_fraction=float(np.max(ratio.hi+ratio.lo)),
                       minimum_f=float(np.min(f.hi+f.lo)), maximum_f=float(np.max(f.hi+f.lo)))
        passed = (passed and metrics["ion_inventory_relative"] <= budgets["ion_inventory_relative"]
                  and metrics["maximum_site_fraction"] <= budgets["maximum_site_fraction"])
    metrics["passed"] = bool(passed)
    return metrics


def affine_observation_payload(reading: AffineDeviceObservation) -> dict:
    """Compact explicit DD observables; physical state is stored by history."""
    def words(value):
        return {"high_hex": [float(x).hex() for x in np.ravel(value.high)],
                "low_hex": [float(x).hex() for x in np.ravel(value.low)]}
    typed = isinstance(reading.derivative, RateView)
    displayed = reading.derivative.words[0] if typed else reading.derivative
    payload = {"point_identity": reading.point.identity, "origin": reading.origin,
            "event_side": reading.event_side,
            "physical_ydot_hex": [float(x).hex() for x in displayed],
            **{name: words(getattr(reading, name)) for name in
               ("metal_charge", "conduction_inward", "total_inward", "body_charge",
                "body_charge_rate", "gauss_defect", "interior_total_current")},
            "tangent_residual_hex": [float(x).hex() for x in reading.tangent_residual]}
    if typed:
        rate = reading.derivative
        payload.update(
            physical_ydot_role="first word for display and old field readers; not the physical consumer input",
            physical_rate={
                "identity": rate.identity, "point_identity": rate.point.identity,
                "source_identity": rate.source_identity, "mapping_identity": rate.mapping_identity,
                "frame": rate.origin, "observed_origin": reading.origin,
                "values_words_hex": [[float(v).hex() for v in word] for word in rate.words],
                "raw_coordinates_hex": [float(v).hex() for v in rate.raw_coordinates],
                "raw_rate_hex": [float(v).hex() for v in rate.raw_rate],
                "input_rate_hex": [float(v).hex() for v in rate.input_rate],
            },
            charge_integrands=words(reading.charge_integrands),
            linear_actions={name: {
                "form_identity": action.form_identity, "operand_identity": action.operand_identity,
                "source_identities": list(action.source_identities),
                "arithmetic_policy": action.arithmetic_policy,
                "absolute_error_bound_hex": [float(v).hex() for v in action.absolute_error_bound.values],
                "error_scope": "represented sources and coefficient arithmetic only; source-model and continuous integration errors excluded",
            } for name, action in reading.linear_actions.items()},
        )
    return payload


@dataclass(frozen=True, init=False)
class AffineSamplingContext:
    """Own one immutable request snapshot and its exact observation digests.

    Native sampling uses the context-owned frozen segments. External segment
    objects still take the canonical comparison path, so a mutable caller
    cannot receive a digest cached for an earlier value.
    """

    _model: AffineCoupledSlab = field(repr=False)
    _encoded_request: bytes = field(repr=False)
    source_identity: str
    reference_identity: str
    request_sha256: str
    controls_sha256: str
    segments: tuple[ProtocolSegment, ...]
    segment_sha256: tuple[str, ...]
    budgets: Mapping

    def __init__(self, model: AffineCoupledSlab, request: Mapping):
        encoded = json.dumps(dict(request), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        owned = json.loads(encoded)
        if owned["case_id"] != model.definition.id:
            raise ContractError("sampling_context_case_mismatch")
        segments = tuple(ProtocolSegment(s["id"], s["start"], s["end"],
                                         tuple(s["voltage"]), tuple(s["photons"]))
                         for s in owned["segments"])
        if len({s.id for s in segments}) != len(segments):
            raise ContractError("sampling_context_duplicate_segment")

        def immutable(value):
            if isinstance(value, dict):
                return MappingProxyType({k: immutable(v) for k, v in value.items()})
            if isinstance(value, list):
                return tuple(immutable(v) for v in value)
            return value

        for name, value in {
            "_model": model, "_encoded_request": encoded,
            "source_identity": model.source_identity,
            "reference_identity": model.reference.identity,
            "request_sha256": sha256(encoded).hexdigest(),
            "controls_sha256": digest(owned["controls"]),
            "segments": segments,
            "segment_sha256": tuple(digest(s) for s in owned["segments"]),
            "budgets": immutable(owned["budgets"]),
        }.items():
            object.__setattr__(self, name, value)

        if "qualification_policy" in owned:
            policy = _qualification_scope(model, segments, owned["qualification_policy"])
            object.__setattr__(self, "qualification_scope",
                               (model.definition.id, policy["intervals"], policy["coordinates"]))

    def request_copy(self) -> dict:
        """Return an independent copy for a controller's private working data."""
        return json.loads(self._encoded_request)

    def segment_digest(self, model: AffineCoupledSlab, segment: ProtocolSegment) -> str:
        if (model is not self._model or model.source_identity != self.source_identity
                or model.reference.identity != self.reference_identity):
            raise ContractError("foreign_or_stale_sampling_context")
        for bound, expected in zip(self.segments, self.segment_sha256):
            if segment is bound:
                return expected
            if segment.id == bound.id:
                if digest(asdict(segment)) == expected:
                    return expected
                break
        raise ContractError("native_history_segment_binding")


def affine_native_sample(model: AffineCoupledSlab, history: AffineDeviceHistory,
                         request: Mapping | AffineSamplingContext, segment: ProtocolSegment, native: Mapping,
                         predecessor: Point, *, origin: str):
    """Bind a supplied solver snapshot to its protocol and predecessor.

    This consumes a snapshot; it neither invokes nor simulates IDA.
    The caller must preserve the returned pair before interpolation.
    """
    if history.model is not model:
        raise ContractError("foreign_device_history")
    context = request if isinstance(request, AffineSamplingContext) else AffineSamplingContext(model, request)
    segment_sha256 = context.segment_digest(model, segment)
    index = next(i for i, s in enumerate(context.segments) if s.id == segment.id)
    t = float(native["time"])
    inputs, input_rate = segment.inputs(t)
    if not native["success"]:
        raise ContractError("native_snapshot_unsuccessful")
    if origin == "segment_initial":
        if t != segment.start or predecessor.time != t:
            raise ContractError("native_segment_initial_time")
        side = "continuous" if index == 0 else "right"
    elif origin in {"native", "stop_output", "endpoint_restore", "interpolant"}:
        if not segment.start <= predecessor.time < t <= segment.end:
            raise ContractError("native_history_predecessor_interval")
        side = "left" if t == segment.end else "continuous"
    else:
        raise ContractError("native_history_origin")
    z, zdot = frozen_array(native["z"]), frozen_array(native["zdot"])
    if z.shape != model.S.shape or zdot.shape != model.S.shape:
        raise ContractError("native_snapshot_shape")
    cumulative, rate = model.S*z, model.S*zdot
    point, increment, sample = history.build_sample(
        cumulative, t, inputs, predecessor, rate, input_rate,
        origin=origin, event_side=side, raw_z=z, raw_zdot=zdot)
    if origin == "segment_initial" and any(np.any(increment.field(v.id).as_dd() != 0)
                                          for v in model.layout.variables):
        raise ContractError("native_restart_changed_physical_state")
    evaluation = model.observation_evaluation(point, state_actions=True)
    raw = model.observe(point, rate, input_rate, origin, side, evaluation=evaluation)
    tangent = model.observe(point, model.tangent_rate(point, input_rate, evaluation=evaluation), input_rate,
                            "physical_tangent", side, evaluation=evaluation)
    difference = abs(raw.total_inward.as_dd()-tangent.total_inward.as_dd())
    record = {"kind": "affine_rate_pair", "request_sha256": context.request_sha256,
              "segment_id": segment.id, "segment_sha256": segment_sha256,
              "controls_sha256": context.controls_sha256, "history": sample,
              "input_slope_hex": [float(x).hex() for x in input_rate],
              "raw": affine_observation_payload(raw),
              "physical_tangent": affine_observation_payload(tangent),
              "scaling_roundoff": affine_scaling_roundoff(model, z, zdot),
              "state_checks": affine_state_quality_evidence(model, point, context.budgets, evaluation=evaluation),
              "current_rate_gap_A": float(np.max(difference.hi+difference.lo))}
    record["record_sha256"] = digest(record)
    return point, increment, raw, tangent, record


def affine_charge_accounting(model, delta, integrals, previous_defects,
                             previous_uncertainties, budget):
    """Exact retained-word gates with conservative persisted cumulative bounds.

    Float displays never participate in acceptance. The retained DD integrals
    still have their original quadrature/arithmetic uncertainty limitations;
    rational comparison does not turn them into continuum error certificates.
    """
    prior_d, prior_u = frozen_array(previous_defects), frozen_array(previous_uncertainties)
    shape = np.shape(delta.hi)
    if (len(shape) != 1 or prior_d.shape != shape or prior_u.shape != shape or len(integrals) != 3
            or any(np.shape(v.hi) != shape for v in integrals)
            or np.any(prior_d < 0) or np.any(prior_u < 0)
            or not math.isfinite(budget) or budget <= 0):
        raise ContractError("invalid_affine_charge_accounting")

    def exact_words(value):
        return [Fraction.from_float(float(h))+Fraction.from_float(float(l))
                for h, l in zip(value.hi, value.lo)]

    def upper(values):
        return [_fraction_float_bound(v, upper=True) for v in values]

    d = exact_words(delta)
    estimates = [exact_words(value) for value in integrals]
    defect = [abs(a-b) for a, b in zip(d, estimates[-1])]
    uncertainty = [abs(a-b) for a, b in zip(estimates[-1], estimates[-2])]
    cd = [Fraction.from_float(float(a))+b for a, b in zip(prior_d, defect)]
    cu = [Fraction.from_float(float(a))+b for a, b in zip(prior_u, uncertainty)]
    interval = [a+b for a, b in zip(defect, uncertainty)]
    prefix = [a+b for a, b in zip(cd, cu)]
    budget_q = Fraction.from_float(float(budget))
    # Preserve the exact encoded reference allocation used by the old gate.
    reference_q = Fraction.from_float(float(budget/3))
    checks = {"interval_total": all(v <= budget_q for v in interval),
              "prefix_total": all(v <= budget_q for v in prefix),
              "interval_reference_share": all(v <= reference_q for v in uncertainty),
              "prefix_reference_share": all(v <= reference_q for v in cu)}
    projected = [[float(v) for v in row] for row in [d]+estimates]
    projection = [sum(abs(row[i]-Fraction.from_float(view[i]))
                      for row, view in zip([d]+estimates, projected)) for i in range(shape[0])]
    return {"delta_charge_C": projected[0], "integrals_C": projected[1:],
            "absolute_defect_C": upper(defect), "quadrature_uncertainty_estimate_C": upper(uncertainty),
            "cumulative_absolute_defect_C": upper(cd),
            "cumulative_quadrature_uncertainty_estimate_C": upper(cu),
            "interval_total_bound_C": upper(interval), "prefix_total_bound_C": upper(prefix),
            "reference_estimate": "absolute16/32 difference; not a rigorous continuum certificate",
            "charge_budget_C": budget, "reference_allocation_C": budget/3,
            "output_projection_error_estimate_C": upper(projection),
            "display_projections_used_for_acceptance": False,
            "acceptance_arithmetic": "exact Fraction sums of retained DD words; prior cumulative inputs are outward-safe nonnegative binary64 bounds",
            "exact_interval_total_C": [str(v) for v in interval],
            "exact_prefix_total_C": [str(v) for v in prefix],
            "checks": checks, "passed": all(checks.values()),
            "delta_charge_words": {"high": np.asarray(delta.hi).tolist(), "low": np.asarray(delta.lo).tolist()},
            "integral_words": [{"high": np.asarray(v.hi).tolist(), "low": np.asarray(v.lo).tolist()} for v in integrals]}


_IDA_COUNTER_FIELDS = ("num_steps", "residual_evals", "linear_setups", "error_test_fails",
                       "nonlinear_iters", "nonlinear_conv_fails", "jacobian_evals")


def ida_statistics_snapshot(solver, segment_id: str, generation: int, phase: str,
                            requested_time: float, returned_time=None, *, before=None,
                            logical_initialization=False) -> dict:
    """Read public native statistics with explicit time and initialization scope.

    Current method fields can be sentinels or retained values before a first
    step; they are never filled or relabelled as last-used method fields.
    Counter differences include rejected work, not an inferred Newton path.
    """
    reader = getattr(solver, "statistics", None)
    if not callable(reader):
        raise ContractError("public_native_statistics_unavailable")
    collected = reader()
    if "parent_weight_state" in collected:
        from scripts.benchmarks.native_observation import observation_record
        collected = dict(collected, parent_weight_state=observation_record(collected["parent_weight_state"]))
    raw = json.loads(json.dumps(collected, allow_nan=False))
    if any(type(raw.get(k)) is not int or raw[k] < 0 for k in _IDA_COUNTER_FIELDS):
        raise ContractError("invalid_native_statistics_counter")
    if (type(generation) is not int or generation < 1
            or not math.isfinite(raw["current_time"])):
        raise ContractError("invalid_native_statistics_context")
    generation_field = "logical_initialization_index" if logical_initialization else "initialization_generation"
    record = {
        "kind": "native_statistics", "segment_id": segment_id,
        generation_field: generation, "phase": phase,
        "requested_time_hex": float(requested_time).hex(),
        "returned_time_hex": None if returned_time is None else float(returned_time).hex(),
        "internal_time_hex": float(raw["current_time"]).hex(),
        "method_state_valid": raw["num_steps"] > 0,
        "raw_statistics": raw,
        "counter_scope": "since this initialization; includes attempted and rejected work",
        "method_scope": "raw native current/last fields; no inferred iteration branch",
    }
    if logical_initialization:
        record["initialization_scope"] = "controller initialization ordinal; source-owned snapshots separately carry native owner/generation"
    if before is not None:
        if (before["segment_id"] != segment_id
                or before.get(generation_field) != generation):
            raise ContractError("native_statistics_generation_mismatch")
        delta = {k: raw[k]-before["raw_statistics"][k] for k in _IDA_COUNTER_FIELDS}
        if any(v < 0 for v in delta.values()):
            raise ContractError("native_statistics_counter_regressed")
        record["work_since_before"] = delta
    return record


def run_affine_native_pilot(model: AffineCoupledSlab, segments: tuple[ProtocolSegment, ...],
                            request: Mapping, admission: Mapping, emit) -> dict:
    """One admitted full protocol, retaining every raw returned IDA pair.

    Physical populations are never projected. Tangent observations use the
    unchanged raw state and have separate labels and charge ledgers. The
    outer frozen launcher owns the independent wall/RSS/total-output guard.
    """
    request_id = digest(dict(request))
    if (admission.get("request_sha256") != request_id
            or not str(admission.get("coordinator_message", "")).startswith("msg_")):
        raise ContractError("affine_native_pilot_not_admitted")
    bindings = admission.get("source_sha256", {})
    if str(Path(__file__).resolve()) not in bindings:
        raise ContractError("affine_native_executing_kernel_not_bound")
    for path, expected in bindings.items():
        if sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ContractError("affine_native_source_changed")
    if not isinstance(model, AffineCoupledSlab) or model.intervals != 8 or model.layout.size > 45:
        raise ContractError("affine_native_scope")
    if (digest(model.numeric_packet()) != digest(request["numeric_packet"])
            or digest([asdict(s) for s in segments]) != digest(request["segments"])):
        raise ContractError("affine_native_numeric_packet_changed")
    if any(k in request["controls"] for k in ("constraints_idx", "constraints_type")):
        raise ContractError("absolute_population_constraints_on_remainders")
    sampling = AffineSamplingContext(model, request)
    if sampling.request_sha256 != request_id:
        raise ContractError("affine_native_request_changed_during_snapshot")
    request = sampling.request_copy()
    segments = sampling.segments
    from sksundae.ida import IDA
    from scripts.benchmarks.precision_prototype import DD

    started = time.perf_counter()
    budgets, controls = request["budgets"], request["controls"]
    counts = {"residual": 0, "jacobian": 0, "native_steps": 0, "normal_queries": 0,
              "history_bytes": 0}
    cumulative = {key: np.zeros(3) for key in ("raw", "physical_tangent")}
    cumulative_u = {key: np.zeros(3) for key in cumulative}
    first_failure = last_attempt = last_numerical = last_accepted = None
    history = AffineDeviceHistory(model)
    z = np.asarray(request["z0"], float)
    predecessor = model.reference
    rows, columns = model.graph.edges().T
    sparsity = csc_matrix((np.ones(len(rows)), (rows, columns)), shape=model.graph.shape)
    differential = np.unique(model.mass.tocoo().col)
    algebraic = np.setdiff1d(np.arange(model.layout.size), differential).tolist()

    def save(record):
        try:
            position = emit_record(record, emit, logical_bytes=counts["history_bytes"],
                                   total_output_bytes=budgets["total_output_bytes"])
        except HistoryLimitError as error:
            raise ContractError("affine_native_history_budget") from error
        counts["history_bytes"] = position.logical_bytes
        if position.encoding == "gzip":
            counts["history_encoded_bytes"] = position.encoded_bytes

    def resource_check():
        if time.perf_counter()-started > budgets["wall_s"]:
            raise ContractError("affine_native_wall_budget")
        if counts["residual"] > budgets["residual_calls"]:
            raise ContractError("affine_native_residual_budget")
        if counts["native_steps"] > budgets["native_steps"]:
            raise ContractError("affine_native_step_budget")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > budgets["rss_bytes"]:
            raise ContractError("affine_native_rss_budget")

    def snapshot_receipt(phase, segment, native):
        return {"phase": phase, "segment_id": segment.id, "time_hex": float(native["time"]).hex(),
                "success": native["success"], "status": native["status"], "message": native["message"],
                "z_hex": [float(v).hex() for v in native["z"]],
                "zdot_hex": [float(v).hex() for v in native["zdot"]]}

    def checked_sample(segment, native, previous, origin):
        nonlocal first_failure
        resource_check()
        pair = affine_native_sample(model, history, sampling, segment, native, previous, origin=origin)
        record = pair[-1]
        save(record)
        if not record["state_checks"]["passed"]:
            if first_failure is None:
                first_failure = {"phase": "physical_state_quality", "record": record}
            raise ContractError("affine_native_or_interpolant_state_quality")
        if record["current_rate_gap_A"] > budgets["current_A"]/3:
            if first_failure is None:
                first_failure = {"phase": "raw_tangent_current", "record": record}
            raise ContractError("affine_raw_tangent_current_budget")
        return pair

    def pointer(pair):
        return {"record_sha256": pair[-1]["record_sha256"], "point_identity": pair[0].identity,
                "time_s": pair[0].time, "origin": pair[2].origin,
                "segment_id": pair[-1]["segment_id"]}

    try:
        save({"kind": "affine_reference", "request_sha256": request_id,
              "reference_digest": history.reference_digest, "reference": history.reference_record})
        for segment_index, segment in enumerate(segments):
            adapter = AffineScaledIDAAdapter(model, segment)

            def residual(t, values, rates, output):
                nonlocal first_failure
                counts["residual"] += 1
                try:
                    resource_check()
                    output[:] = adapter.residual(t, values, rates)
                except BaseException as error:
                    if first_failure is None:
                        first_failure = {"phase": "residual", "segment_id": segment.id,
                                         "time_hex": float(t).hex(), "reason": str(error),
                                         "z_hex": [float(v).hex() for v in values],
                                         "zdot_hex": [float(v).hex() for v in rates]}
                        save({"kind": "first_callback_failure", "failure": first_failure})
                    raise

            def jacobian(t, values, rates, residual_value, cj, output):
                nonlocal first_failure
                counts["jacobian"] += 1
                try:
                    resource_check()
                    save({"kind": "jacobian_callback_context", "segment_id": segment.id,
                          "initialization_generation": segment_index+1,
                          "callback_index": counts["jacobian"], "request_sha256": request_id,
                          "time_hex": float(t).hex(), "cj_hex": float(cj).hex(),
                          "z_hex": [float(v).hex() for v in values],
                          "zdot_hex": [float(v).hex() for v in rates]})
                    matrix = adapter.jacobian(t, values, cj, rates)
                    if (matrix.nnz != model.graph.nnz or not np.array_equal(matrix.indices, model.graph.indices)
                            or not np.array_equal(matrix.indptr, model.graph.indptr)
                            or output.shape != (model.graph.nnz,)):
                        raise ContractError("affine_native_csc_slot_mismatch")
                    output[:] = matrix.data
                except BaseException as error:
                    if first_failure is None:
                        first_failure = {"phase": "jacobian", "segment_id": segment.id,
                                         "time_hex": float(t).hex(), "cj_hex": float(cj).hex(),
                                         "reason": str(error), "z_hex": [float(v).hex() for v in values],
                                         "zdot_hex": [float(v).hex() for v in rates]}
                        save({"kind": "first_callback_failure", "failure": first_failure})
                    raise

            initial_point = adapter.point(segment.start, z)
            _, adot = segment.inputs(segment.start)
            initial_zdot = (np.asarray(request["zdot0"], float) if segment_index == 0 else
                            model.tangent_rate(initial_point, adot)/model.S)
            last_attempt = {"phase": "segment_initialization", "segment_id": segment.id,
                            "time_hex": segment.start.hex(), "z_hex": [float(v).hex() for v in z],
                            "zdot_hex": [float(v).hex() for v in initial_zdot]}
            save({"kind": "initialization_input", **last_attempt})
            solver = IDA(residual, jacfn=jacobian, sparsity=sparsity, algebraic_idx=algebraic, **controls)
            initialized = snapshot_solver_result(solver.init_step(segment.start, z, initial_zdot))
            last_attempt = snapshot_receipt("initialization_return", segment, initialized)
            save(ida_statistics_snapshot(solver, segment.id, segment_index+1,
                                         "initialization_return", segment.start, initialized["time"]))
            if (not initialized["success"] or initialized["time"] != segment.start
                    or not np.array_equal(initialized["z"], z)
                    or not np.array_equal(initialized["zdot"], initial_zdot)):
                raise ContractError("affine_initialization_changed_state_or_failed")
            left_pair = checked_sample(segment, initialized, predecessor, "segment_initial")
            left = left_pair[0]
            last_numerical = pointer(left_pair)
            if last_accepted is None:
                last_accepted = last_numerical
            sample_times = np.asarray(request["observation_times"][segment.id])
            cursor = 0
            while cursor < len(sample_times) and sample_times[cursor] == segment.start:
                save({"kind": "requested_sample", **last_numerical})
                cursor += 1
            while left.time < segment.end:
                resource_check()
                if counts["native_steps"] >= budgets["native_steps"]:
                    raise ContractError("affine_native_step_budget")
                last_attempt = {"phase": "native_step", "segment_id": segment.id,
                                "target_time_hex": segment.end.hex(), "previous": last_numerical}
                before_stats = ida_statistics_snapshot(solver, segment.id, segment_index+1,
                                                        "before_onestep", segment.end)
                save(before_stats)
                try:
                    native = snapshot_solver_result(solver.step(segment.end, method="onestep", tstop=segment.end))
                except BaseException:
                    try:
                        save(ida_statistics_snapshot(solver, segment.id, segment_index+1,
                                                     "onestep_exception", segment.end, before=before_stats))
                    except Exception as stats_error:
                        save({"kind": "native_statistics_unavailable", "segment_id": segment.id,
                              "initialization_generation": segment_index+1,
                              "phase": "onestep_exception", "reason": str(stats_error)})
                    raise
                save(ida_statistics_snapshot(solver, segment.id, segment_index+1,
                                             "after_onestep", segment.end, native["time"], before=before_stats))
                last_attempt = snapshot_receipt("native_return", segment, native)
                if not native["success"]:
                    raise ContractError("affine_native_solver_failure:"+native["message"])
                t = native["time"]
                if not left.time < t <= segment.end:
                    raise ContractError("affine_native_interval_not_monotone")
                counts["native_steps"] += 1
                origin = "stop_output" if native["status"] == 1 else "native"
                right_pair = checked_sample(segment, native, left, origin)
                right = right_pair[0]
                last_numerical = pointer(right_pair)
                cache = {left.time: left_pair, t: right_pair}
                active = True

                def query(when):
                    nonlocal last_attempt
                    if not active or not left.time <= when <= t:
                        raise ContractError("expired_or_foreign_affine_interval")
                    if when in cache:
                        return cache[when]
                    resource_check()
                    normal = snapshot_solver_result(solver.step(when, method="normal", tstop=segment.end))
                    counts["normal_queries"] += 1
                    last_attempt = snapshot_receipt("interpolant_return", segment, normal)
                    if not normal["success"] or normal["time"] != when:
                        raise ContractError("affine_native_dense_query_failed")
                    pair = checked_sample(segment, normal, left, "interpolant")
                    cache[when] = pair
                    return pair

                try:
                    integrals = {key: [] for key in cumulative}
                    for order in (8, 16, 32):
                        table = request["quadrature"][str(order)]
                        h = t-left.time
                        totals = {key: DD(np.zeros(3)) for key in cumulative}
                        for node, weight in zip(table["nodes"], table["weights"]):
                            pair = query(left.time+h*(float(node)+1)/2)
                            for key, reading in zip(cumulative, pair[2:4]):
                                current = model.arithmetic.concatenate((
                                    reading.conduction_inward.as_dd().sum().reshape(1),
                                    reading.total_inward.as_dd()-reading.conduction_inward.as_dd()))
                                totals[key] += DD(h)*(float(weight)/2)*current
                        for key in cumulative:
                            integrals[key].append(totals[key])
                    changes = model.finite_physical_changes(left, right, right_pair[1])
                    delta = model.arithmetic.concatenate((changes["body_charge_C"].as_dd().reshape(1),
                                                           changes["metal_charge_C"].as_dd()))
                    evidence = {}
                    for key in cumulative:
                        evidence[key] = affine_charge_accounting(model, delta, integrals[key], cumulative[key],
                                                                 cumulative_u[key], budgets["charge_C"])
                        cumulative[key] = np.asarray(evidence[key]["cumulative_absolute_defect_C"])
                        cumulative_u[key] = np.asarray(evidence[key]["cumulative_quadrature_uncertainty_estimate_C"])
                    while cursor < len(sample_times) and sample_times[cursor] <= t:
                        save({"kind": "requested_sample", **pointer(query(float(sample_times[cursor])))})
                        cursor += 1
                    restored = snapshot_solver_result(solver.step(t, method="normal", tstop=segment.end))
                    counts["normal_queries"] += 1
                    last_attempt = snapshot_receipt("endpoint_restore_return", segment, restored)
                    if not restored["success"] or restored["time"] != t:
                        raise ContractError("affine_native_endpoint_restore_failed")
                    restored_pair = checked_sample(segment, restored, left, "endpoint_restore")
                    interval = {"kind": "affine_interval_charge", "segment_id": segment.id,
                                "left": pointer(left_pair), "right": pointer(right_pair),
                                "endpoint_restore": pointer(restored_pair), "ledgers": evidence,
                                "endpoint_snapshot_equal": bool(np.array_equal(restored["z"], native["z"])),
                                "states_projected": False}
                    save(interval)
                    if not all(v["passed"] for v in evidence.values()):
                        first_failure = first_failure or interval
                        raise ContractError("affine_interval_or_prefix_charge_budget")
                    last_accepted = last_numerical
                finally:
                    active = False
                z = native["z"].copy()
                left_pair, left = right_pair, right
            if cursor != len(sample_times):
                raise ContractError("affine_missing_frozen_observation")
            predecessor = left
        result = {"status": "completed_bounded_affine_native_pilot", "complete_protocol": True}
    except BaseException as error:
        result = {"status": "failed_bounded_affine_native_pilot", "complete_protocol": False,
                  "reason": str(error), "exception": type(error).__name__, "traceback": traceback.format_exc(),
                  "first_failure": first_failure or last_attempt}
    result.update(request_sha256=request_id, counts=counts, elapsed_s=time.perf_counter()-started,
                  last_numerical=last_numerical, last_physically_accepted=last_accepted,
                  last_attempt=last_attempt,
                  cumulative_absolute_charge_defects_C={k: v.tolist() for k, v in cumulative.items()},
                  cumulative_quadrature_uncertainty_estimates_C={k: v.tolist() for k, v in cumulative_u.items()},
                  prefix_total_charge_bounds_C={k: (cumulative[k]+cumulative_u[k]).tolist() for k in cumulative},
                  cumulative_scope="includes a fully evaluated first failed attempted interval",
                  scientific_or_G2_qualification=False)
    # The runner writes this tail separately even after the streaming cap.
    return result


def _frame_words(value):
    from scripts.benchmarks.precision_prototype import PrimitiveExpansion

    return PrimitiveExpansion.from_value(value)


def _frame_scale(value, coefficient):
    """Retain each product word, rejecting unrepresentable product tails."""
    from scripts.benchmarks.precision_prototype import DD, DoubleArray

    value = _frame_words(value)
    coefficient = frozen_array(np.broadcast_to(coefficient, value.shape))
    result = _frame_words(np.zeros(value.shape))
    for word in value.words:
        if not np.any(word):
            continue
        product = DoubleArray.from_dd(DD(coefficient)*DD(word))
        # A finite DD product is not, by itself, proof against underflow.
        for a, b, hi, lo in zip(coefficient.flat, word.flat, product.high.flat,
                                product.low.flat, strict=True):
            if (Fraction(float(a))*Fraction(float(b))
                    != Fraction(float(hi))+Fraction(float(lo))):
                raise ContractError("segment_frame_product_capacity")
        result = result.add(product)
    return result


@dataclass(frozen=True)
class SegmentAffineFrame:
    """One immutable live parent origin; IDA owns only u and udot.

    The frame is fixed for an existing segment, including rejected steps.
    Four-word capacity is a representation limit, not an accuracy claim.
    """

    parent_map_identity: str
    segment_sha256: str
    logical_initialization_index: int
    predecessor_identity: str
    parent_input_sha256: str
    t0: float
    q0: object = field(repr=False, compare=False)
    v0: object = field(repr=False, compare=False)
    identity: str = field(init=False)

    def __post_init__(self):
        for name in ("parent_map_identity", "segment_sha256", "predecessor_identity", "parent_input_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ContractError("segment_frame_identity")
        if (type(self.logical_initialization_index) is not int or self.logical_initialization_index < 1
                or type(self.t0) is not float or not math.isfinite(self.t0)):
            raise ContractError("segment_frame_time_or_generation")
        q0, v0 = _frame_words(self.q0), _frame_words(self.v0)
        if len(q0.shape) != 1 or not q0.shape[0] or q0.shape != v0.shape:
            raise ContractError("segment_frame_shape")
        object.__setattr__(self, "q0", q0)
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "identity", digest(self.payload()))

    def payload(self):
        words = lambda v: [[float(x).hex() for x in word] for word in v.words]
        return {"schema": "solarlab.segment-affine-frame.v1",
                "parent_map_identity": self.parent_map_identity,
                "segment_sha256": self.segment_sha256,
                "logical_initialization_index": self.logical_initialization_index,
                "predecessor_identity": self.predecessor_identity,
                "parent_input_sha256": self.parent_input_sha256,
                "t0_hex": self.t0.hex(), "size": self.q0.shape[0],
                "q0_words_hex": words(self.q0), "v0_words_hex": words(self.v0),
                "law": "q=Q0+(t-t0)*V0+u;qdot=V0+udot",
                "weight_rounding": "RN-exact-parent-q_then-unfused-SV-v1"}

    def parent_state(self, time, u):
        from scripts.benchmarks.precision_prototype import DD

        if type(time) not in (float, int) or not math.isfinite(time):
            raise ContractError("segment_frame_time")
        u = _frame_words(u)
        if u.shape != self.q0.shape:
            raise ContractError("segment_frame_shape")
        dt = DD(float(time))-DD(self.t0)
        # Ordered, exact retained composition; no rounded t-t0 or anchor.
        return self.q0.add(_frame_scale(self.v0, dt.hi)).add(
            _frame_scale(self.v0, dt.lo)).add(u)

    def parent_rate(self, udot):
        return self.v0.add(_frame_words(udot))

    def weight_option(self):
        payload = self.payload()
        return {"schema": "sksundae.ida.parent-affine-weights.v1", "frame_identity": self.identity,
                **{k: payload[k] for k in ("t0_hex", "size", "q0_words_hex", "v0_words_hex", "weight_rounding")}}


@dataclass(frozen=True)
class AffineVoltageMap(ImmutableArrays):
    """External voltage departures mapped into unchanged physical Points.

    ``z`` is a solver coordinate. Point.y is the physical SI remainder
    ``S*z + L*(a-a_ref)`` and is never relabeled as z. The public primitive
    addition retains every input word within its declared four-word capacity.
    """

    model: AffineCoupledSlab = field(repr=False, compare=False)
    frame: SegmentAffineFrame | None = field(default=None, repr=False, compare=False)
    columns: np.ndarray = field(init=False, repr=False)
    rows: np.ndarray = field(init=False, repr=False)
    lift: np.ndarray = field(init=False, repr=False)
    reference_inputs: np.ndarray = field(init=False, repr=False)
    model_identity: str = field(init=False)
    reference_identity: str = field(init=False)
    identity: str = field(init=False)
    relation_form: PhysicalLinearForm = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        from scripts.benchmarks.precision_prototype import PrimitiveExpansion

        if not isinstance(self.model, AffineCoupledSlab):
            raise ContractError("voltage_lift_requires_physical_affine_model")
        if not callable(getattr(PrimitiveExpansion, "add", None)):
            raise ContractError("public_primitive_add_unavailable")
        m = self.model
        if m.reference.inputs.shape != (2,):
            raise ContractError("voltage_lift_input_shape")
        lift = np.zeros((m.layout.size, 2))
        # These binary64 coefficients define the map. We do not assume that
        # their discrete Poisson action is exactly zero on the physical mesh.
        lift[m.layout.offsets["phi_V"], 0] = -m.x/m.definition.length
        for name, value in (("columns", m.S), ("rows", m.Drow),
                            ("lift", lift), ("reference_inputs", m.reference.inputs)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, "model_identity", m.source_identity)
        object.__setattr__(self, "reference_identity", m.reference.identity)
        if self.frame is not None and (type(self.frame) is not SegmentAffineFrame
                or self.frame.q0.shape != self.columns.shape
                or self.frame.parent_map_identity != digest(self.payload(include_frame=False))):
            raise ContractError("segment_frame_parent_map")
        object.__setattr__(self, "identity", digest(self.payload()))
        # Prove the raw-coordinate/Point relation with the same public action.
        # This does not reconstruct a predecessor or form a second trial.
        roots = {"n_m3": m.definition.n_eq, "p_m3": m.definition.p_eq, "phi_V": 0.,
                 "c_m3": m.definition.ion_initial, "f": m.definition.f_eq}
        sources, terms, units = [], [], []
        for variable in m.layout.variables:
            if variable.id not in roots:
                raise ContractError("voltage_lift_reference_field_unsupported")
            spec = LinearSourceSpec("map_drive_"+variable.id, variable.support,
                                    variable.shape, variable.unit, self.model_identity)
            sources.append(spec)
            origin = LinearFactor("map_origin_"+variable.id, roots[variable.id],
                                  variable.unit, self.model_identity)
            for i in range(int(np.prod(variable.shape))):
                row = m.layout.offsets[variable.id].start+i
                terms += [LinearTerm(row, variable.id, i), LinearTerm(row, spec.id, i, sign=-1),
                          LinearTerm(row, None, 0, (origin,), -1)]
                units.append(variable.unit)
        object.__setattr__(self, "relation_form", PhysicalLinearForm(
            m.layout, tuple(f"raw_point_relation_{i}" for i in range(m.layout.size)),
            tuple(units), tuple(terms), self.model_identity, "state", len(terms), tuple(sources)))

    def payload(self, *, include_frame=True):
        result = {
            "schema": "solarlab.affine-voltage-map.v1",
            "physical_model": self.model_identity,
            "physical_reference": self.reference_identity,
            "layout": self.model.layout.identity,
            "columns_hex": [float(v).hex() for v in self.columns],
            "rows_hex": [float(v).hex() for v in self.rows],
            "lift_hex": [[float(v).hex() for v in row] for row in self.lift],
            "reference_inputs_hex": [float(v).hex() for v in self.reference_inputs],
            "raw_coordinate_meaning": "scaled voltage departures; other fields are scaled physical remainders",
            "physical_coordinate_meaning": "Point.y is the first word of S*z+L*(a-a_ref)",
        }
        if include_frame and self.frame is not None:
            result.update(segment_frame=self.frame.payload(),
                          raw_coordinate_meaning="fixed-segment affine remainder u; native history is u",
                          physical_coordinate_meaning="Point.y is the first retained word of S*(Q0+(t-t0)*V0+u)+L*(a-a_ref)")
        return result

    def _check(self):
        m = self.model
        if (m.source_identity != self.model_identity
                or m.reference.identity != self.reference_identity
                or not np.array_equal(m.S, self.columns)
                or not np.array_equal(m.Drow, self.rows)):
            raise ContractError("voltage_lift_source_changed")

    def _compose(self, raw, inputs, *, subtract_reference):
        from scripts.benchmarks.precision_prototype import DD, DoubleArray, PrimitiveExpansion

        self._check()
        from scripts.benchmarks.precision_prototype import PrimitiveExpansion

        inputs = frozen_array(inputs)
        if type(raw) is PrimitiveExpansion:
            if raw.shape != self.columns.shape or inputs.shape != self.reference_inputs.shape:
                raise ContractError("voltage_lift_coordinate_shape")
            result = _frame_scale(raw, self.columns)
        else:
            raw = frozen_array(raw)
            if raw.shape != self.columns.shape or inputs.shape != self.reference_inputs.shape:
                raise ContractError("voltage_lift_coordinate_shape")
            result = PrimitiveExpansion.from_value(DoubleArray.from_dd(DD(self.columns)*DD(raw)))
        if raw.shape != self.columns.shape or inputs.shape != self.reference_inputs.shape:
            raise ContractError("voltage_lift_coordinate_shape")
        for j in range(inputs.size):
            if not np.any(self.lift[:, j]):
                continue
            result = result.add(DoubleArray.from_dd(DD(self.lift[:, j])*DD(inputs[j])))
            if subtract_reference:
                result = result.add(DoubleArray.from_dd(-DD(self.lift[:, j])*DD(self.reference_inputs[j])))
        return result

    def physical_primitive(self, z, inputs, *, time=None):
        if self.frame is not None:
            z = self.frame.parent_state(time, z)
        return self._compose(z, inputs, subtract_reference=True)

    def physical_rate(self, zdot, input_rate):
        if self.frame is not None:
            zdot = self.frame.parent_rate(zdot)
        return self._compose(zdot, input_rate, subtract_reference=False)

    def bind_rate(self, point, z, zdot, input_rate):
        """Bind an actual trial/restored Point and this map's full push-forward.

        Callers retain the actual raw coordinates; no projected density or
        rate is used to recover a low word or to reconstruct solver history.
        """
        self._check()
        self.model.validate(point)
        drive = self.physical_primitive(z, point.inputs, time=point.time)
        sources = []
        for spec in self.relation_form.sources:
            selection = self.model.layout.offsets[spec.id.removeprefix("map_drive_")]
            sources.append((spec, drive.take_flat(np.arange(selection.start, selection.stop))))
        relation = point.state.linear_form(self.relation_form, arithmetic=self.model.arithmetic,
                                          point=point, sources=BoundLinearSource.bind(point, sources))
        if (np.any(relation.value.high != 0) or np.any(relation.value.low != 0)
                or np.any(relation.absolute_error_bound.values != 0)):
            raise ContractError("voltage_lift_point_raw_coordinate_mismatch")
        return RateView(point, self.physical_rate(zdot, input_rate), input_rate,
                        source_identity=self.model_identity, mapping_identity=self.identity,
                        origin="mapped-coordinate-rate", raw_coordinates=z, raw_rate=zdot)

    def trial(self, z, time, inputs, *, predecessor=None):
        primitive = self.physical_primitive(z, inputs, time=time)
        return self.model.trial(primitive, time, inputs, predecessor=predecessor,
                                transition_representation="paired-endpoints-v1")


def voltage_lift_rate_projection(model: AffineCoupledSlab, primitive) -> dict:
    """Exact audit of reducing a mapped rate to its first word.

    Historical consumers used that projection. Full-word consumers retain
    this audit as a counterfactual only, not their current arithmetic error.
    Its record shape stays unchanged so old history readers remain usable.
    """
    from scripts.benchmarks.precision_prototype import PrimitiveExpansion

    if not isinstance(primitive, PrimitiveExpansion) or primitive.shape != (model.layout.size,):
        raise ContractError("voltage_lift_rate_shape")
    f = lambda v: Fraction.from_float(float(v))
    exact = [sum((f(word[i]) for word in primitive.words), Fraction(0))
             for i in range(model.layout.size)]
    error = [f(v)-source for v, source in zip(primitive.high, exact, strict=True)]
    fields = {v.id: error[model.layout.offsets[v.id]] for v in model.layout.variables}
    m, N = model.definition, model.count
    q, nt, eps, area = map(f, (Q, m.trap_density, m.epsilon, m.area))
    volumes = list(map(f, model.geometry.volumes))
    n, p, phi = (fields[name] for name in ("n_m3", "p_m3", "phi_V"))
    c, trap = ((fields["c_m3"], fields["f"]) if m.dynamic
               else ([Fraction(0)]*N, [Fraction(0)]*N))
    rho = [q*(p[i]-n[i]+c[i]-nt*trap[i]) for i in range(N)]
    ddot = [-eps*(phi[j+1]-phi[j])/f(model.dx[j]) for j in range(N-1)]
    conduction = [q*volumes[i]*(p[i]-n[i]) for i in (0, N-1)]
    total = [conduction[0]+area*ddot[0]-volumes[0]*rho[0],
             conduction[1]-area*ddot[-1]-volumes[-1]*rho[-1]]
    body = sum((v*r for v, r in zip(volumes, rho, strict=True)), Fraction(0))
    return {
        "projection": "first word of the normalized four-word physical rate",
        "projected_rate_hex": [float(v).hex() for v in primitive.high],
        "component_error_exact": [str(v) for v in error],
        "conduction_error_A": [float(v) for v in conduction],
        "total_current_error_A": [float(v) for v in total],
        "total_current_error_exact_A": [str(v) for v in total],
        "body_charge_rate_error_A": float(body),
        "body_charge_rate_error_exact_A": str(body),
        "scope": "rate projection at unchanged physical state; no state or integral certificate",
    }


@dataclass(frozen=True)
class VoltageLiftAdapter:
    """Public physical kernel plus an explicit external affine input map.

    All arguments are independent for partial derivatives. A future native
    controller must supply protocol inputs/rates and separately obtain run
    admission; this object neither imports nor starts a native integrator.
    """

    mapping: AffineVoltageMap
    problem: object = field(init=False, repr=False)
    source_identity: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.mapping, AffineVoltageMap):
            raise ContractError("voltage_lift_map_required")
        self.mapping._check()
        object.__setattr__(self, "problem", self.mapping.model.public_problem(
            accepted_rate_mapping_identity=self.mapping.identity))
        object.__setattr__(self, "source_identity", digest({
            "map": self.mapping.identity, "partial_frame": "raw-z,raw-zdot,inputs,input-rates,time-v1"}))

    def _point_rate(self, time, z, zdot, inputs, input_rate, predecessor=None):
        point, increment = self.mapping.trial(z, time, inputs, predecessor=predecessor)
        physical_rate = self.mapping.bind_rate(point, z, zdot, input_rate)
        return point, increment, physical_rate

    def residual(self, time, z, zdot, inputs, input_rate):
        point, _, rate = self._point_rate(time, z, zdot, inputs, input_rate)
        value = self.problem.residual(point, rate, frozen_array(input_rate))
        return self.mapping.rows*self.mapping.model.project_output(value)

    def linearize(self, time, z, zdot, inputs, input_rate):
        from scripts.benchmarks.contract_prototype import SparseLinearization

        point, _, rate = self._point_rate(time, z, zdot, inputs, input_rate)
        physical = self.problem.linearize(point, rate, frozen_array(input_rate))
        mapping, structure = self.mapping, physical.structure

        def scale(matrix):
            columns = np.repeat(np.arange(matrix.shape[1]), np.diff(matrix.indptr))
            return structure.filled(matrix.data*mapping.rows[matrix.indices]*mapping.columns[columns])

        # Fy already includes the physical storage-rate contraction. Because
        # the external map is affine, there are no missing map Hessian terms.
        inputs_part = physical.inputs+physical.y @ mapping.lift
        rates_part = physical.input_rate+physical.ydot @ mapping.lift
        return SparseLinearization(
            scale(physical.y), scale(physical.ydot),
            mapping.rows[:, None]*inputs_part, mapping.rows[:, None]*rates_part,
            mapping.rows*physical.time, structure, self.source_identity)

    def jacobian(self, time, z, zdot, inputs, input_rate, cj):
        return self.linearize(time, z, zdot, inputs, input_rate).ida_matrix(cj)

    def finite_storage_increment(self, left, right, increment):
        return self.problem.storage.delta(left, right, increment)

    def observe(self, time, z, zdot, inputs, input_rate, *,
                predecessor=None, origin="algebraic_probe", side="continuous"):
        if origin == "physical_tangent":
            raise ContractError("voltage_lift_tangent_requires_named_evaluation")
        point, increment, rate = self._point_rate(time, z, zdot, inputs, input_rate, predecessor)
        observed = self.mapping.model.observe(point, rate, input_rate, origin, side)
        return point, increment, rate.values, observed, voltage_lift_rate_projection(self.mapping.model, rate.values)

    def physical_tangent_observation(self, time, z, inputs, input_rate, *, side="continuous"):
        point, _ = self.mapping.trial(z, time, inputs)
        rate = self.mapping.model.tangent_rate(point, input_rate)
        observed = self.mapping.model.observe(point, rate, input_rate, "physical_tangent", side)
        return point, rate, observed

    def port_partials(self, time, z, zdot, inputs, input_rate):
        point, _, _ = self._point_rate(time, z, zdot, inputs, input_rate)
        p = self.mapping.model.port_partials(point)
        columns, lift = diags(self.mapping.columns), self.mapping.lift
        return {
            "charge_z": p["charge_y"] @ columns,
            "charge_inputs": frozen_array(p["charge_y"] @ lift),
            "conduction_z": p["conduction_y"] @ columns,
            "conduction_zdot": p["conduction_ydot"] @ columns,
            "conduction_inputs": frozen_array(p["conduction_y"] @ lift),
            "conduction_input_rates": frozen_array(p["conduction_ydot"] @ lift),
            "current_z": p["current_y"] @ columns,
            "current_zdot": p["current_ydot"] @ columns,
            "current_inputs": frozen_array(p["current_inputs"]+p["current_y"] @ lift),
            "current_input_rates": frozen_array(p["current_input_rates"]+p["current_ydot"] @ lift),
            "body_z": frozen_array(p["body_y"]*self.mapping.columns),
            "body_inputs": frozen_array(p["body_y"] @ lift),
        }


class VoltageLiftHistory:
    """Replay the new raw coordinate map without converting old histories."""

    def __init__(self, mapping: AffineVoltageMap, *, parent_reference=None):
        from scripts.benchmarks.precision_prototype import encode_point

        if not isinstance(mapping, AffineVoltageMap):
            raise ContractError("voltage_lift_map_required")
        mapping._check()
        self.mapping = mapping
        self.reference_record = {
            "schema": "solarlab.voltage-lift-reference.v1",
            "map": mapping.payload(), "map_identity": mapping.identity,
            "physical_reference": encode_point(mapping.model.reference),
            "physical_reference_identity": mapping.reference_identity,
        }
        if parent_reference is not None:
            if (mapping.frame is None or parent_reference["map_identity"] != mapping.frame.parent_map_identity
                    or parent_reference["physical_reference_identity"] != mapping.reference_identity):
                raise ContractError("segment_frame_history_parent")
            self.reference_record = json.loads(json.dumps(parent_reference))
        self.reference_digest = digest(self.reference_record)

    def build_sample(self, z, time, inputs, predecessor, zdot, input_rate, *,
                     origin="algebraic_probe", event_side="continuous",
                     transition_representation="paired-endpoints-v1"):
        if origin not in {"algebraic_probe", "native", "interpolant", "stop_output",
                           "endpoint_restore", "segment_initial", "declared_polynomial"}:
            raise ContractError("unknown_derivative_origin")
        if event_side not in {"continuous", "left", "right"}:
            raise ContractError("unknown_event_side")
        if transition_representation not in {"paired-endpoints-v1", "four-word-v1"}:
            raise ContractError("unknown_affine_transition_representation")
        z, zdot = frozen_array(z), frozen_array(zdot)
        inputs, input_rate = frozen_array(inputs), frozen_array(input_rate)
        physical = self.mapping.physical_primitive(z, inputs, time=time)
        rate = self.mapping.physical_rate(zdot, input_rate)
        point, increment = self.mapping.model.trial(
            physical, time, inputs, predecessor=predecessor,
            transition_representation=transition_representation)
        self.mapping.model.validate(point)
        words = lambda value: [[float(x).hex() for x in word] for word in value.words]
        record = {
            "schema": ("solarlab.voltage-lift-sample.v2" if transition_representation == "paired-endpoints-v1"
                       else "solarlab.voltage-lift-sample.v1"),
            "map_identity": self.mapping.identity,
            "physical_model_identity": self.mapping.model_identity,
            "reference_digest": self.reference_digest,
            "point_identity": point.identity, "predecessor_identity": predecessor.identity,
            "time_hex": float(time).hex(),
            "inputs_hex": [float(v).hex() for v in inputs],
            "input_rates_hex": [float(v).hex() for v in input_rate],
            "raw_solver_z_hex": [float(v).hex() for v in z],
            "raw_solver_zdot_hex": [float(v).hex() for v in zdot],
            "physical_cumulative_words_hex": words(physical),
            "physical_rate_words_hex": words(rate),
            "physical_rate_projection": voltage_lift_rate_projection(self.mapping.model, rate),
            "origin": origin, "event_side": event_side,
            "raw_coordinate_frame": "scaled-voltage-departure-v1",
            "physical_rate_frame": "direct-map-push-forward-v1",
            "reference_embedded": False,
        }
        if transition_representation == "paired-endpoints-v1":
            record["transition_representation"] = transition_representation
        if self.mapping.frame is not None:
            record.update(raw_coordinate_frame="fixed-affine-state-rate-v1",
                          segment_frame_identity=self.mapping.frame.identity)
        return point, increment, rate, record

    def restore(self, reference_record, record, predecessor):
        from scripts.benchmarks.precision_prototype import decode_point

        version = record.get("schema")
        transition = "four-word-v1" if version == "solarlab.voltage-lift-sample.v1" else "paired-endpoints-v1"
        if (digest(reference_record) != self.reference_digest
                or version not in {"solarlab.voltage-lift-sample.v1", "solarlab.voltage-lift-sample.v2"}
                or version == "solarlab.voltage-lift-sample.v2" and record.get("transition_representation") != transition
                or record.get("map_identity") != self.mapping.identity
                or record.get("reference_digest") != self.reference_digest
                or record.get("predecessor_identity") != predecessor.identity):
            raise ContractError("voltage_lift_history_binding_mismatch")
        reference = decode_point(reference_record["physical_reference"], self.mapping.model.layout)
        if reference.identity != self.mapping.reference_identity:
            raise ContractError("voltage_lift_reference_decode_mismatch")
        point, increment, rate, rebuilt = self.build_sample(
            [float.fromhex(v) for v in record["raw_solver_z_hex"]], float.fromhex(record["time_hex"]),
            [float.fromhex(v) for v in record["inputs_hex"]], predecessor,
            [float.fromhex(v) for v in record["raw_solver_zdot_hex"]],
            [float.fromhex(v) for v in record["input_rates_hex"]],
            origin=record["origin"], event_side=record["event_side"],
            transition_representation=transition)
        if digest(rebuilt) != digest(record):
            raise ContractError("voltage_lift_history_word_mismatch")
        return point, increment, rate, frozen_array([float.fromhex(v) for v in record["input_rates_hex"]])


def voltage_lift_wrms_policy(mapping: AffineVoltageMap, previous_controls: Mapping,
                            segments: tuple[ProtocolSegment, ...]) -> dict:
    """Conservative denominators over every input in the full finite protocol.

    Previous controls refer to scaled physical remainders, not absolute
    populations. Triangle inequalities cover all real remainders; sampled
    states are not used to choose the bounds. This proves a local norm
    comparison only, not global state accuracy or physical conservation.
    """
    mapping._check()
    if not segments:
        raise ContractError("voltage_lift_protocol_required")
    f = lambda v: Fraction.from_float(float(v))
    old_r = f(previous_controls["rtol"])
    old_atol = frozen_array(previous_controls["atol"])
    if old_atol.shape != mapping.columns.shape or old_r <= 0 or np.any(old_atol <= 0):
        raise ContractError("invalid_original_error_weights")
    ranges = []
    for segment in segments:
        if segment.end <= segment.start:
            raise ContractError("invalid_frozen_protocol")
        bounds = []
        for endpoints in (segment.voltage, segment.photons):
            start, end = map(float, endpoints)
            duration = float(segment.end-segment.start)
            rate = float((end-start)/duration)
            # The ordinary interpolation path is a monotone composition of
            # rounded subtraction, constant multiplication and addition.
            # Include its endpoint limit as well as the explicit overrides.
            ordinary_end = float(start+float(rate*duration))
            if not all(math.isfinite(v) for v in (start, end, rate, ordinary_end)):
                raise ContractError("voltage_lift_protocol_range")
            bounds.append((min(start, end, ordinary_end), max(start, end, ordinary_end)))
        ranges.append({"segment": asdict(segment), "input_bounds": bounds})
    offsets = []
    for row in mapping.lift:
        largest = Fraction(0)
        for record in ranges:
            lower, upper = Fraction(0), Fraction(0)
            for j, (a, b) in enumerate(record["input_bounds"]):
                endpoints = [f(row[j])*(f(v)-f(mapping.reference_inputs[j])) for v in (a, b)]
                lower += min(endpoints); upper += max(endpoints)
            largest = max(largest, abs(lower), abs(upper))
        offsets.append(largest)
    scales = list(map(f, mapping.columns))
    physical_atol = [s*f(a) for s, a in zip(scales, old_atol, strict=True)]
    r_bound = min([old_r]+[a/(2*u) for a, u in zip(physical_atol, offsets) if u])
    rnew = _fraction_float_bound(r_bound, upper=False)
    if rnew <= 0:
        raise ContractError("no_positive_representable_conservative_rtol")
    r = f(rnew)
    atol, rows = [], []
    for i, (s, a, u) in enumerate(zip(scales, physical_atol, offsets, strict=True)):
        new_bound = a-r*u
        encoded = _fraction_float_bound(new_bound/s, upper=False)
        actual = s*f(encoded)
        checks = {"positive_atol": encoded > 0, "rtol_not_larger": r <= old_r,
                  "half_absolute_budget": r*u <= a/2,
                  "encoded_atol_not_larger": actual <= new_bound,
                  "triangle_margin": actual+r*u <= a}
        if not all(checks.values()):
            raise ContractError("voltage_lift_wrms_certificate_failed")
        atol.append(encoded)
        rows.append({"index": i, "scale_exact": str(s), "offset_upper_exact": str(u),
                     "old_physical_atol_exact": str(a), "new_physical_atol_exact": str(actual),
                     "atol_z_hex": encoded.hex(), "checks": checks})
    return {"schema": "solarlab.voltage-lift-wrms-proof.v1", "map_identity": mapping.identity,
            "rtol": rnew, "atol": atol, "rtol_hex": rnew.hex(),
            "old_controls_sha256": digest(dict(previous_controls)),
            "old_coordinate_frame": "scaled original physical remainders",
            "full_protocol_input_bounds": ranges, "components": rows,
            "bound": "S*atol_z+rnew*abs(y-L*(a-a_ref))<=S*old_atol_z+rold*abs(y)",
            "proof": "abs(y-b)<=abs(y)+U; rnew<=rold; Anew+rnew*U<=Aold; directed-down binary64 encoding",
            "scope": "all real physical remainders and stated protocol inputs; local error-denominator comparison only",
            "state_and_rate_projection_errors_separate": True,
            "native_admission": False, "global_accuracy_or_conservation_certified": False}


@dataclass(frozen=True)
class VoltageLiftSegmentAdapter:
    """Unexecuted controller draft: bind public callbacks to one segment.

    The caller owns resource checks and retention of the first callback
    failure. These callbacks do not start IDA, grant admission, or recover
    invalid states. The archived40/42 tests cover the underlying algebra;
    this new wiring needs its own frozen tests before native use.
    """

    adapter: VoltageLiftAdapter
    context: AffineSamplingContext
    segment: ProtocolSegment
    segment_sha256: str = field(init=False)
    source_identity: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.adapter, VoltageLiftAdapter):
            raise ContractError("voltage_lift_public_adapter_required")
        mapping = self.adapter.mapping
        segment_digest = self.context.segment_digest(mapping.model, self.segment)
        declared_map = self.context.request_copy().get("voltage_lift_map")
        parent_map_id = mapping.identity if mapping.frame is None else mapping.frame.parent_map_identity
        if not isinstance(declared_map, dict) or digest(declared_map) != parent_map_id:
            raise ContractError("voltage_lift_request_map_mismatch")
        if mapping.frame is not None:
            request = self.context.request_copy()
            _validate_segment_frame_policy(request)
            if ("segment_frame_policy" not in request or mapping.frame.segment_sha256 != segment_digest
                    or mapping.frame.t0 != self.segment.start
                    or mapping.frame.logical_initialization_index !=
                    next(i for i, s in enumerate(self.context.segments, 1) if s.id == self.segment.id)):
                raise ContractError("segment_frame_segment_binding")
        # Use the context-owned tuple inputs even when the caller supplied an
        # equal segment with mutable nested lists.
        owned = next(s for s in self.context.segments if s.id == self.segment.id)
        object.__setattr__(self, "segment", owned)
        object.__setattr__(self, "segment_sha256", segment_digest)
        object.__setattr__(self, "source_identity", digest({
            "adapter": self.adapter.source_identity,
            "request": self.context.request_sha256, "segment": segment_digest,
        }))

    def residual(self, time, z, zdot, output):
        mapping = self.adapter.mapping
        self.context.segment_digest(mapping.model, self.segment)
        if not isinstance(output, np.ndarray) or output.shape != mapping.columns.shape:
            raise ContractError("voltage_lift_residual_buffer_shape")
        inputs, input_rate = self.segment.inputs(time)
        output[:] = self.adapter.residual(time, z, zdot, inputs, input_rate)

    def jacobian(self, time, z, zdot, residual_value, cj, output):
        """Actual binding signature; fill the declared sparse CSC slots."""
        mapping = self.adapter.mapping
        self.context.segment_digest(mapping.model, self.segment)
        graph = mapping.model.graph
        if not isinstance(output, np.ndarray) or output.shape != (graph.nnz,):
            raise ContractError("voltage_lift_jacobian_buffer_shape")
        inputs, input_rate = self.segment.inputs(time)
        matrix = self.adapter.jacobian(time, z, zdot, inputs, input_rate, cj)
        if (matrix.shape != graph.shape or matrix.nnz != graph.nnz
                or not np.array_equal(matrix.indices, graph.indices)
                or not np.array_equal(matrix.indptr, graph.indptr)):
            raise ContractError("voltage_lift_csc_slot_mismatch")
        output[:] = matrix.data


def voltage_lift_initial_input(binding: VoltageLiftSegmentAdapter, z, predecessor: Point):
    """Prepare a rate at an unchanged segment-start state, without IDA.

    The desired physical tangent is distinct from the finite encoded rate.
    Encode (desired-L*adot)/S once, retain the exact rounding errors, and
    expose both residuals. No consistency or physical acceptance is inferred
    from constructing this input. The physical state is never re-encoded.
    """
    mapping, segment = binding.adapter.mapping, binding.segment
    model = mapping.model
    binding.context.segment_digest(model, segment)
    if predecessor.time != segment.start:
        raise ContractError("voltage_lift_initial_predecessor_time")
    z = frozen_array(z)
    inputs, input_rate = segment.inputs(segment.start)
    point, increment = mapping.trial(z, segment.start, inputs, predecessor=predecessor)
    model.validate(point)
    if any(np.any(increment.field(v.id).high) or np.any(increment.field(v.id).low)
           for v in model.layout.variables):
        raise ContractError("voltage_lift_restart_changed_physical_state")
    evaluation = model.observation_evaluation(point)
    desired = model.tangent_rate(point, input_rate, evaluation=evaluation)
    f = lambda v: Fraction.from_float(float(v))
    exact_coordinates = [
        (f(value)-sum((f(mapping.lift[i, j])*f(input_rate[j])
                       for j in range(input_rate.size)), Fraction(0)))/f(mapping.columns[i])
        for i, value in enumerate(desired)
    ]
    zdot = frozen_array([float(v) for v in exact_coordinates])
    mapped_rate = mapping.physical_rate(zdot, input_rate)
    represented = [sum((f(word[i]) for word in mapped_rate.words), Fraction(0))
                   for i in range(model.layout.size)]
    residual_words = lambda rate: {
        "high_hex": [float(v).hex() for v in rate.high],
        "low_hex": [float(v).hex() for v in rate.low],
    }
    problem = binding.adapter.problem
    record = {
        "schema": "solarlab.voltage-lift-initial-input.v1",
        "source_identity": binding.source_identity,
        "request_sha256": binding.context.request_sha256,
        "map_identity": mapping.identity, "segment_id": segment.id,
        "segment_sha256": binding.segment_sha256,
        "time_hex": float(segment.start).hex(),
        "point_identity": point.identity, "predecessor_identity": predecessor.identity,
        "raw_z_hex": [float(v).hex() for v in z],
        "raw_zdot_hex": [float(v).hex() for v in zdot],
        "inputs_hex": [float(v).hex() for v in inputs],
        "input_rates_hex": [float(v).hex() for v in input_rate],
        "desired_physical_tangent_hex": [float(v).hex() for v in desired],
        "ideal_coordinate_rate_exact": [str(v) for v in exact_coordinates],
        "coordinate_encoding_error_exact": [str(f(v)-q) for v, q in zip(zdot, exact_coordinates, strict=True)],
        "mapped_physical_rate_words_hex": [[float(v).hex() for v in word] for word in mapped_rate.words],
        "mapped_minus_desired_rate_exact": [str(v-f(w)) for v, w in zip(represented, desired, strict=True)],
        "projected_minus_desired_rate_exact": [str(f(v)-f(w)) for v, w in zip(mapped_rate.high, desired, strict=True)],
        "rate_projection": voltage_lift_rate_projection(model, mapped_rate),
        "desired_tangent_residual_SI": residual_words(problem.residual(point, desired, input_rate)),
        "represented_rate_residual_SI": residual_words(problem.residual(
            point, mapping.bind_rate(point, z, zdot, input_rate), input_rate)),
        "physical_rate_consumption": "public RateView full mapped words",
        "rate_projection_role": "counterfactual_first_word_only",
        "state_changed": False, "rate_source": "named physical tangent at unchanged state",
        "native_initialization_performed": False, "native_steps": 0,
        "consistency_or_physical_acceptance_certified": False,
    }
    record["record_sha256"] = digest(record)
    return point, increment, zdot, record


def voltage_lift_segment_initialization(mapping, context, segment, ordinal, parent_z, predecessor):
    """Prepare exactly one native initialization; no solver is allocated here.

    The state-only preparation map evaluates the existing initializer at the
    exact live parent state. Its finite supplied rate is then the fixed V0;
    the desired tangent is never silently substituted for that encoded rate.
    """
    request = context.request_copy()
    _validate_segment_frame_policy(request)
    if "segment_frame_policy" not in request:
        binding = VoltageLiftSegmentAdapter(VoltageLiftAdapter(mapping), context, segment)
        _, _, rate, proof = voltage_lift_initial_input(binding, parent_z, predecessor)
        return binding, frozen_array(parent_z), rate, proof, None
    if mapping.frame is not None or predecessor.time != segment.start:
        raise ContractError("segment_frame_initialization_parent")
    q0 = _frame_words(parent_z)
    zero = np.zeros(mapping.model.layout.size)
    ancestry = {"request_sha256": context.request_sha256, "parent_map_identity": mapping.identity,
                "predecessor_identity": predecessor.identity,
                "q0_words_hex": [[float(x).hex() for x in word] for word in q0.words]}
    seed = SegmentAffineFrame(mapping.identity, digest(asdict(segment)), ordinal,
                             predecessor.identity, digest(ancestry), float(segment.start), q0, zero)
    preparation = VoltageLiftSegmentAdapter(VoltageLiftAdapter(AffineVoltageMap(mapping.model, seed)),
                                           context, segment)
    point, _, supplied_v0, parent_proof = voltage_lift_initial_input(preparation, zero, predecessor)
    frame = SegmentAffineFrame(mapping.identity, seed.segment_sha256, ordinal, predecessor.identity,
                               parent_proof["record_sha256"], seed.t0, q0, supplied_v0)
    framed = AffineVoltageMap(mapping.model, frame)
    binding = VoltageLiftSegmentAdapter(VoltageLiftAdapter(framed), context, segment)
    inputs, input_rate = segment.inputs(segment.start)
    same, _ = framed.trial(zero, segment.start, inputs, predecessor=predecessor)
    rate = framed.physical_rate(zero, input_rate)
    words = lambda v: [[float(x).hex() for x in word] for word in v.words]
    if (same.identity != point.identity or words(rate) != parent_proof["mapped_physical_rate_words_hex"]
            or words(framed.physical_primitive(zero, inputs, time=segment.start)) !=
            words(mapping.physical_primitive(q0, inputs, time=segment.start))):
        raise ContractError("segment_frame_physical_handoff")
    proof = dict(parent_proof, schema="solarlab.voltage-lift-initial-input.v2",
                 source_identity=binding.source_identity, map_identity=framed.identity,
                 raw_z_hex=[float(v).hex() for v in zero], raw_zdot_hex=[float(v).hex() for v in zero],
                 segment_frame_identity=frame.identity, parent_coordinate_rate_hex=parent_proof["raw_zdot_hex"],
                 coordinate_encoding_frame="parent qdot; native udot is zero after full-word handoff")
    proof.pop("record_sha256")
    proof["record_sha256"] = digest(proof)
    record = {"kind": "voltage_lift_segment_frame", "request_sha256": context.request_sha256,
              "parent_map_identity": mapping.identity, "map_identity": framed.identity,
              "frame": frame.payload(), "frame_identity": frame.identity, "map": framed.payload(),
              "ancestry": ancestry, "preparation_frame": seed.payload(), "parent_input": parent_proof,
              "physical_handoff_words_hex": words(framed.physical_primitive(zero, inputs, time=segment.start)),
              "physical_rate_handoff_words_hex": words(rate), "native_initial_z_hex": proof["raw_z_hex"],
              "native_initial_zdot_hex": proof["raw_zdot_hex"],
              "requested_h0_unchanged": True,
              "automatic_h0_parent_rate_limiter_parity": False,
              "native_initialization_performed": False}
    record["record_sha256"] = digest(record)
    return binding, frozen_array(zero), frozen_array(zero), proof, record


def voltage_lift_native_sample(binding: VoltageLiftSegmentAdapter, history: VoltageLiftHistory,
                               native: Mapping, predecessor: Point, *, origin: str):
    """Draft sampling hook for a supplied snapshot; it never advances time.

    All mapped rate words reach the physical observer and survive in history.
    The physical tangent is evaluated at the identical Point. The retained
    first-word audit is counterfactual; no interval acceptance is inferred.
    """
    mapping, context, segment = binding.adapter.mapping, binding.context, binding.segment
    model = mapping.model
    if history.mapping is not mapping:
        raise ContractError("foreign_voltage_lift_history")
    segment_digest = context.segment_digest(model, segment)
    index = next(i for i, s in enumerate(context.segments) if s.id == segment.id)
    t = float(native["time"])
    inputs, input_rate = segment.inputs(t)
    if not native["success"]:
        raise ContractError("native_snapshot_unsuccessful")
    if origin == "segment_initial":
        if t != segment.start or predecessor.time != t:
            raise ContractError("native_segment_initial_time")
        side = "continuous" if index == 0 else "right"
    elif origin in {"native", "stop_output", "endpoint_restore", "interpolant", "declared_polynomial"}:
        if not segment.start <= predecessor.time < t <= segment.end:
            raise ContractError("native_history_predecessor_interval")
        side = "left" if t == segment.end else "continuous"
    else:
        raise ContractError("native_history_origin")
    z, zdot = frozen_array(native["z"]), frozen_array(native["zdot"])
    point, increment, rate, sample = history.build_sample(
        z, t, inputs, predecessor, zdot, input_rate, origin=origin, event_side=side)
    if origin == "segment_initial" and any(
            np.any(increment.field(v.id).high) or np.any(increment.field(v.id).low)
            for v in model.layout.variables):
        raise ContractError("voltage_lift_restart_changed_physical_state")
    evaluation = model.observation_evaluation(point, state_actions=True)
    raw = model.observe(point, mapping.bind_rate(point, z, zdot, input_rate),
                        input_rate, origin, side, evaluation=evaluation)
    tangent = model.observe(point, model.tangent_rate(point, input_rate, evaluation=evaluation),
                            input_rate, "physical_tangent", side, evaluation=evaluation)
    difference = abs(raw.total_inward.as_dd()-tangent.total_inward.as_dd())
    record = {
        "kind": "voltage_lift_rate_pair", "request_sha256": context.request_sha256,
        "map_identity": mapping.identity,
        "segment_id": segment.id, "segment_sha256": segment_digest,
        "controls_sha256": context.controls_sha256, "history": sample,
        "snapshot_status": native["status"], "snapshot_message": native["message"],
        "input_slope_hex": [float(v).hex() for v in input_rate],
        "raw": affine_observation_payload(raw),
        "physical_tangent": affine_observation_payload(tangent),
        "state_checks": affine_state_quality_evidence(model, point, context.budgets, evaluation=evaluation),
        "current_rate_gap_A": float(np.max(difference.hi+difference.lo)),
        "rate_projection": sample["physical_rate_projection"],
        "rate_projection_role": "counterfactual_first_word_only",
        "physical_rate_consumption": "public RateView full mapped words",
        "interval_or_prefix_charge_certified": False,
        "supplied_snapshot_only": True,
    }
    record["record_sha256"] = digest(record)
    return point, increment, raw, tangent, record

def voltage_lift_charge_rate_projection(model: AffineCoupledSlab, audit: Mapping):
    """Exact projected-minus-mapped errors in the three charge integrands.

    Body charge uses the sum of reservoir conduction currents. Metal charge
    uses total-minus-conduction on each side; it must not use total current
    or the whole-body storage-rate error as a substitute.
    """
    errors = [Fraction(v) for v in audit["component_error_exact"]]
    total = [Fraction(v) for v in audit["total_current_error_exact_A"]]
    if len(errors) != model.layout.size or len(total) != 2:
        raise ContractError("voltage_lift_projection_shape")
    n, p = (errors[model.layout.offsets[name]] for name in ("n_m3", "p_m3"))
    q = Fraction.from_float(float(Q))
    conduction = [q*Fraction.from_float(float(model.geometry.volumes[i]))*(p[i]-n[i])
                  for i in (0, model.count-1)]
    return (sum(conduction, Fraction(0)), total[0]-conduction[0], total[1]-conduction[1])


def voltage_lift_charge_accounting(model, delta, integrals, projection_integrals,
                                   previous_defects, previous_uncertainties,
                                   previous_projection, budget):
    """Include explicit projection estimates within the original allocations.

    The projection inputs are exact rational Gauss sums of absolute signed
    rate errors. E32+abs(E32-E16) is an estimate, just as the original16/32
    current discrepancy is an estimate; neither is a continuum certificate.
    No signed cancellation can consume less of the unchanged charge budget.
    """
    base = affine_charge_accounting(model, delta, integrals, previous_defects,
                                    previous_uncertainties, budget)
    size = len(base["exact_interval_total_C"])
    prior = frozen_array(previous_projection)
    if (prior.shape != (size,) or np.any(prior < 0) or len(projection_integrals) != 3
            or any(len(row) != size for row in projection_integrals)
            or any(not isinstance(v, Fraction) or v < 0 for row in projection_integrals for v in row)):
        raise ContractError("invalid_voltage_lift_projection_integral")
    f = lambda v: Fraction.from_float(float(v))
    projection = [b+abs(b-a) for a, b in zip(projection_integrals[1], projection_integrals[2], strict=True)]
    cumulative = [f(a)+b for a, b in zip(prior, projection, strict=True)]
    interval = [Fraction(a)+b for a, b in zip(base["exact_interval_total_C"], projection, strict=True)]
    prefix = [Fraction(a)+b for a, b in zip(base["exact_prefix_total_C"], cumulative, strict=True)]
    reference_interval = [f(a)+b for a, b in zip(base["quadrature_uncertainty_estimate_C"], projection, strict=True)]
    reference_prefix = [f(a)+b for a, b in zip(base["cumulative_quadrature_uncertainty_estimate_C"], cumulative, strict=True)]
    bound, reference_bound = f(budget), f(budget/3)
    original = dict(base["checks"])
    checks = {
        "interval_total": all(v <= bound for v in interval),
        "prefix_total": all(v <= bound for v in prefix),
        "interval_reference_share": all(v <= reference_bound for v in reference_interval),
        "prefix_reference_share": all(v <= reference_bound for v in reference_prefix),
        "original_projected_rate_checks": all(original.values()),
    }
    upper = lambda values: [_fraction_float_bound(v, upper=True) for v in values]
    base.update(
        original_projected_rate_checks=original,
        projection_absolute_gauss_sums_exact_C=[[str(v) for v in row] for row in projection_integrals],
        rate_projection_estimate_C=upper(projection),
        cumulative_rate_projection_estimate_C=upper(cumulative),
        interval_total_bound_C=upper(interval), prefix_total_bound_C=upper(prefix),
        exact_interval_total_C=[str(v) for v in interval], exact_prefix_total_C=[str(v) for v in prefix],
        exact_combined_reference_interval_C=[str(v) for v in reference_interval],
        exact_combined_reference_prefix_C=[str(v) for v in reference_prefix],
        projection_estimate="E32+abs(E32-E16), E is positive-weight quadrature of absolute exact per-sample projection errors",
        projection_integral_continuum_certified=False,
        physical_allocations_changed=False, checks=checks, passed=all(checks.values()),
    )
    return base


def _voltage_lift_nonlinear_refinement(previous_controls, ancestor_sha256, value):
    """Bind one explicitly stricter coefficient without changing its ancestor."""
    previous = previous_controls.get("nonlin_conv_coef")
    if (isinstance(previous, bool) or not isinstance(previous, (int, float))
            or not math.isfinite(previous) or previous <= 0
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value < previous
            or not isinstance(ancestor_sha256, str) or len(ancestor_sha256) != 64
            or any(c not in "0123456789abcdef" for c in ancestor_sha256)):
        raise ContractError("voltage_lift_invalid_nonlinear_refinement")
    return {"schema": "solarlab.voltage-lift-nonlinear-refinement.v1",
            "field": "nonlin_conv_coef", "previous_value": float(previous),
            "value": float(value), "previous_controls_sha256": digest(dict(previous_controls)),
            "ancestor_request_sha256": ancestor_sha256}


def _voltage_lift_nonlinear_guard(previous_controls, ancestor_sha256, policy, capacity):
    """Bind the optional native policy and its resource-only trace capacity."""
    if (not isinstance(policy, str) or policy != "first-correction-wrms-v1"
            or isinstance(capacity, bool) or not isinstance(capacity, int)
            or not 1 <= capacity <= 4096
            or not isinstance(ancestor_sha256, str) or len(ancestor_sha256) != 64
            or any(c not in "0123456789abcdef" for c in ancestor_sha256)):
        raise ContractError("voltage_lift_invalid_nonlinear_guard")
    return {"schema": "solarlab.voltage-lift-nonlinear-guard.v1", "policy": policy,
            "trace_capacity": capacity,
            "previous_controls_sha256": digest(dict(previous_controls)),
            "ancestor_request_sha256": ancestor_sha256}


def _voltage_lift_startup_step(previous_controls, ancestor_sha256, value):
    """Bind an explicit initial step without replacing the inherited controls."""
    previous = previous_controls.get("first_step", 0.0)
    maximum = previous_controls.get("max_step", 0.0)
    if (isinstance(previous, bool) or not isinstance(previous, (int, float))
            or not math.isfinite(previous) or previous < 0
            or isinstance(maximum, bool) or not isinstance(maximum, (int, float))
            or not math.isfinite(maximum) or maximum < 0
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            or (maximum > 0 and value > maximum)
            or not isinstance(ancestor_sha256, str) or len(ancestor_sha256) != 64
            or any(c not in "0123456789abcdef" for c in ancestor_sha256)):
        raise ContractError("voltage_lift_invalid_startup_step")
    return {"schema": "solarlab.voltage-lift-startup-step.v1", "field": "first_step",
            "previous_value": float(previous), "value": float(value),
            "application": "every_protocol_initialization",
            "previous_controls_sha256": digest(dict(previous_controls)),
            "ancestor_request_sha256": ancestor_sha256}


def _voltage_lift_time_weights(mapping, parent_controls, parent_proof, ancestor_sha256, kappa,
                               qualification_policy=None):
    """One parent-bound time-error experiment; no adaptive-path equivalence.

    At identical states/corrections the exact WRMS and Newton thresholds
    scale together. Matching norm histories also preserve rate/ss. Native
    rounded decisions and future weighting states need separate evidence.
    """
    size = 45
    if qualification_policy is not None:
        from scripts.benchmarks.qualification_policy import validate_policy
        qualification_policy = validate_policy(qualification_policy)
        size = qualification_policy["coordinates"]
        if mapping.model.intervals != qualification_policy["intervals"]:
            raise ContractError("qualification_time_weight_mesh")
    if (type(kappa) is not int or kappa != 1024
            or mapping.model.definition.id != "DynamicAcceptorIonPublicDeviceV1"
            or mapping.model.layout.size != size
            or parent_controls.get("nonlin_conv_coef") != 1e-8
            or parent_controls.get("nonlin_guard") != "first-correction-wrms-v1"
            or parent_controls.get("nonlin_trace_capacity") != 4096
            or parent_controls.get("first_step") != 7.8125e-7
            or not isinstance(ancestor_sha256, str) or len(ancestor_sha256) != 64
            or any(c not in "0123456789abcdef" for c in ancestor_sha256)):
        raise ContractError("voltage_lift_invalid_time_weights")
    old = {key: parent_controls[key] for key in ("rtol", "atol", "nonlin_conv_coef")}
    values = {"rtol": math.ldexp(old["rtol"], -10),
              "atol": [math.ldexp(value, -10) for value in old["atol"]],
              "nonlin_conv_coef": math.ldexp(old["nonlin_conv_coef"], 10)}
    f = lambda value: Fraction.from_float(float(value))
    pairs = [(old["rtol"], values["rtol"], Fraction(1, kappa)),
             (old["nonlin_conv_coef"], values["nonlin_conv_coef"], Fraction(kappa))]
    pairs += [(a, b, Fraction(1, kappa)) for a, b in zip(old["atol"], values["atol"], strict=True)]
    if (len(values["atol"]) != size
            or any(not math.isfinite(b) or b < sys.float_info.min or f(b) != f(a)*scale
                   for a, b, scale in pairs)):
        raise ContractError("voltage_lift_time_weights_not_exact_normal")
    policy = {
        "schema": "solarlab.voltage-lift-time-weights.v1", "kappa": kappa,
        "ancestor_request_sha256": ancestor_sha256,
        "parent_controls_sha256": digest(dict(parent_controls)),
        "parent_weight_certificate_sha256": digest(parent_proof),
        "parent_values": old, "values": values,
        "application": "every_fresh_initialization_before_first_solve",
        "exact_parameter_scaling": "rtol/atol ldexp(-10); nonlin_conv_coef ldexp(+10); positive normal binary64",
        "norm_comparison": "same-state denominators divide by1024; exact WRMS and epcon/epsNewt/toldel/m0-direct thresholds multiply by1024",
        "history_condition": "matching correction/weight histories preserve norm ratios and dimensionless rate/ss; later ss*WRMS comparison scales homogeneously",
        "local_time_error_threshold": 1,
        "rounded_native_decisions_or_adaptive_path_equal": False,
        "global_accuracy_or_conservation_certified": False,
    }
    if qualification_policy is not None:
        policy.update(schema="solarlab.voltage-lift-time-weights.v2",
                      qualification_policy_sha256=digest(qualification_policy))
    applied = dict(parent_proof, rtol=values["rtol"], atol=values["atol"],
                   rtol_hex=values["rtol"].hex(),
                   parent_weight_certificate_sha256=digest(parent_proof),
                   time_weight_policy_sha256=digest(policy))
    applied["proof"] += "; exact power-of-two contraction of both applied denominator terms"
    applied["components"] = []
    for row, atol in zip(parent_proof["components"], values["atol"], strict=True):
        scale = Fraction(row["scale_exact"])
        offset = Fraction(row["offset_upper_exact"])
        original = Fraction(row["old_physical_atol_exact"])
        absolute, r = scale*f(atol), f(values["rtol"])
        checks = {"positive_atol": atol > 0, "rtol_not_larger": r <= f(parent_proof["rtol"]),
                  "half_absolute_budget": r*offset <= original/2,
                  "encoded_atol_not_larger": absolute <= original-r*offset,
                  "triangle_margin": absolute+r*offset <= original}
        if not all(checks.values()):
            raise ContractError("voltage_lift_time_weights_certificate_failed")
        applied["components"].append(dict(row, new_physical_atol_exact=str(absolute),
                                          atol_z_hex=atol.hex(), checks=checks))
    return policy, dict(parent_controls, **values), applied


def _voltage_lift_guard_applied(raw_statistics, controls):
    """Requested policy alone cannot establish actual installation or tracing."""
    policy = controls.get("nonlin_guard")
    if policy is None:
        return
    trace = raw_statistics.get("nonlinear_guard")
    if (not isinstance(trace, Mapping) or trace.get("installed_at_capture") is not True
            or trace.get("applied_policy") != policy or trace.get("requested_policy") != policy
            or trace.get("delegate_data_verified") is not True
            or trace.get("newton_ops_verified") is not True or trace.get("complete") is not True
            or trace.get("capacity") != controls["nonlin_trace_capacity"]
            or trace.get("owner") != raw_statistics.get("observation_owner")
            or trace.get("generation") != raw_statistics.get("observation_generation")
            or trace.get("build_identity") != raw_statistics.get("nonlinear_control_state", {}).get("build_identity")):
        raise ContractError("voltage_lift_nonlinear_guard_not_applied")


def _voltage_lift_segment_startup(request, overrides):
    """Bind the single hold-only candidate to its unchanged parent controls."""
    controls = request["controls"]
    qualification_policy = request.get("qualification_policy")
    size = 45
    if qualification_policy is not None:
        from scripts.benchmarks.qualification_policy import validate_policy
        qualification_policy = validate_policy(qualification_policy)
        size = qualification_policy["coordinates"]
    if (not isinstance(overrides, Mapping) or set(overrides) != {"slow_state_hold"}
            or not isinstance(overrides["slow_state_hold"], Mapping)
            or set(overrides["slow_state_hold"]) != {"first_step"}):
        raise ContractError("voltage_lift_invalid_segment_startup")
    value = overrides["slow_state_hold"]["first_step"]
    if (type(value) not in (int, float) or value != 0 or math.copysign(1, value) != 1
            or request["case_id"] != "DynamicAcceptorIonPublicDeviceV1"
            or [s["id"] for s in request["segments"]] != [
                "dark_equilibrium_hold", "voltage_ramp", "slow_state_hold"]
            or controls.get("first_step") != 7.8125e-7
            or len(controls.get("atol", [])) != size
            or controls.get("nonlin_conv_coef") != 1.024e-5
            or controls.get("nonlin_guard") != "first-correction-wrms-v1"
            or controls.get("nonlin_trace_capacity") != 4096
            or request.get("time_weight_policy", {}).get("kappa") != 1024):
        raise ContractError("voltage_lift_invalid_segment_startup")
    policy = {"schema": "solarlab.voltage-lift-segment-startup.v1",
            "ancestor_request_sha256": request["prior_request_sha256"],
            "base_controls_sha256": digest(controls),
            "segments_sha256": digest(request["segments"]),
            "time_weight_policy_sha256": digest(request["time_weight_policy"]),
            "overrides": {"slow_state_hold": {"first_step": 0.0}},
            "application": "fresh_segment_initialization_before_first_solve"}
    if qualification_policy is not None:
        policy.update(schema="solarlab.voltage-lift-segment-startup.v2",
                      qualification_policy_sha256=digest(qualification_policy))
    return policy


def _voltage_lift_initialization_controls(request, segment):
    """Copy base controls; an explicit zero applies only to the named hold."""
    controls = dict(request["controls"])
    if "segment_startup_policy" in request:
        policy = request["segment_startup_policy"]
        if not isinstance(policy, Mapping):
            raise ContractError("voltage_lift_invalid_segment_startup")
        expected = _voltage_lift_segment_startup(request, policy.get("overrides"))
        if digest(policy) != digest(expected):
            raise ContractError("voltage_lift_segment_startup_binding")
        if digest(asdict(segment)) not in {digest(s) for s in request["segments"]}:
            raise ContractError("voltage_lift_segment_startup_unbound_segment")
        controls.update(expected["overrides"].get(segment.id, {}))
    return controls


def _segment_frame_policy(request, name):
    if type(name) is not str or name != "fixed-affine-state-rate-v1":
        raise ContractError("invalid_segment_frame_policy")
    return {"schema": "solarlab.segment-frame-policy.v1", "name": name,
            "ancestor_request_sha256": request["prior_request_sha256"],
            "parent_map_identity": request["map_identity"],
            "controls_sha256": digest(request["controls"]), "segments_sha256": digest(request["segments"]),
            "sampling_sha256": digest((request["observation_times"], request["quadrature"])),
            "weight_certificate_sha256": digest(request["weight_certificate"]),
            "segment_startup_policy_sha256": digest(request.get("segment_startup_policy")),
            "qualification_policy_sha256": digest(request.get("qualification_policy")),
            "application": "one_fixed_live_origin_at_each_existing_fresh_segment_initialization",
            "weight_rounding": "RN-exact-parent-q_then-unfused-SV-v1",
            "requested_first_steps_unchanged": True, "actual_automatic_first_step_parity_claimed": False}


def _validate_segment_frame_policy(request):
    if any(key in request for key in ("segment_frame", "frame_overrides", "parent_weight_frame")):
        raise ContractError("segment_frame_unbound_option")
    if "segment_frame_policy" in request:
        policy = request["segment_frame_policy"]
        if not isinstance(policy, Mapping):
            raise ContractError("invalid_segment_frame_policy")
        if digest(policy) != digest(_segment_frame_policy(request, policy.get("name"))):
            raise ContractError("segment_frame_policy_binding")


def _voltage_lift_frame_applied(statistics, frame, *, after_step=False):
    from scripts.benchmarks.native_observation import restore_observation_record

    value = statistics.get("parent_weight_state")
    if value is None:
        raise ContractError("segment_frame_weight_evidence_missing")
    value = restore_observation_record(value)
    if (value["schema"] != "sksundae.ida.parent-affine-weight-state.v1"
            or value["frame_identity"] != frame.identity or value["size"] != frame.q0.shape[0]
            or value["owner"] != statistics["observation_owner"]
            or value["generation"] != statistics["observation_generation"]
            or value["installed"] is not True or value["setter"] != "IDAWFtolerances"
            or value["setter_status"] != 0 or value["t0_hex"] != frame.t0.hex()
            or value["q0_words"] != tuple(w.tobytes() for w in frame.q0.words)
            or value["v0_words"] != tuple(w.tobytes() for w in frame.v0.words)
            or after_step and (value["callback_calls"] < 1 or value["callback_status"] != 0)):
        raise ContractError("segment_frame_weight_application_mismatch")


def _voltage_lift_segment_solver(ida, request, segment, ordinal, residual, jacobian,
                                 sparsity, algebraic, *, frame=None):
    """Record the exact constructor kwargs; native h0u is recorded separately."""
    controls = _voltage_lift_initialization_controls(request, segment)
    _validate_segment_frame_policy(request)
    weight_option = None
    if "segment_frame_policy" in request:
        if (type(frame) is not SegmentAffineFrame or frame.parent_map_identity != request["map_identity"]
                or frame.segment_sha256 != digest(asdict(segment)) or frame.t0 != segment.start
                or frame.logical_initialization_index != ordinal
                or frame.q0.shape != (len(request["controls"]["atol"]),)):
            raise ContractError("segment_frame_constructor_binding")
        weight_option = frame.weight_option()
    elif frame is not None:
        raise ContractError("segment_frame_unbound_option")
    recorded = json.loads(json.dumps(controls, allow_nan=False))
    extra = {} if weight_option is None else {"parent_weight_frame": weight_option}
    solver = ida(residual, jacfn=jacobian, sparsity=sparsity, algebraic_idx=algebraic, **controls, **extra)
    receipt = None
    if "segment_startup_policy" in request or weight_option is not None:
        receipt = {"kind": "voltage_lift_initialization_controls",
                   "request_sha256": digest(request), "map_identity": request["map_identity"],
                   "segment_id": segment.id, "segment_sha256": digest(asdict(segment)),
                   "logical_initialization_index": ordinal,
                   "segment_startup_policy_sha256": digest(request.get("segment_startup_policy")),
                   "constructor_controls": recorded, "constructor_controls_sha256": digest(recorded),
                   "requested_first_step_hex": float(recorded["first_step"]).hex(),
                   "evidence": "exact kwargs passed to the returned IDA constructor; not a native option getter"}
        if weight_option is not None:
            receipt.update(parent_weight_frame=weight_option,
                           segment_frame_policy_sha256=digest(request["segment_frame_policy"]),
                           parent_weight_frame_sha256=digest(weight_option),
                           evidence="exact numeric controls and parent_weight_frame passed to IDA; not a native option getter")
    return solver, receipt


def prepare_voltage_lift_native_request(mapping: AffineVoltageMap,
                                        segments: tuple[ProtocolSegment, ...],
                                        previous_request: Mapping, *,
                                        nonlin_conv_coef: float | None = None,
                                        nonlin_guard: str | None = None,
                                        nonlin_trace_capacity: int = 4096,
                                        first_step: float | None = None,
                                        time_weight_kappa: int | None = None,
                                        segment_startup_overrides: Mapping | None = None,
                                        segment_frame: str | None = None) -> dict:
    """Prepare the full original protocol; this does not authorize execution."""
    model, old = mapping.model, json.loads(json.dumps(dict(previous_request), allow_nan=False))
    qualification_policy = old.get("qualification_policy")
    if qualification_policy is not None:
        qualification_policy = _qualification_scope(model, segments, qualification_policy)
        from scripts.benchmarks.qualification_policy import options
        chosen = dict(nonlin_conv_coef=nonlin_conv_coef, nonlin_guard=nonlin_guard,
                      nonlin_trace_capacity=nonlin_trace_capacity, first_step=first_step,
                      time_weight_kappa=time_weight_kappa,
                      segment_startup_overrides=segment_startup_overrides)
        if chosen != options(qualification_policy):
            raise ContractError("qualification_unbound_control_options")
    if ((qualification_policy is None and (model.intervals != 8 or model.layout.size > 45)) or len(segments) != 3
            or old["schema"] != "solarlab.affine-native-request.v1"
            or old["case_id"] != model.definition.id
            or digest(old["segments"]) != digest([asdict(s) for s in segments])):
        raise ContractError("voltage_lift_original_case_or_protocol_changed")
    packet = model.numeric_packet()
    for key, value in asdict(model.definition).items():
        if key != "source_path" and old["numeric_packet"]["definition"][key] != value:
            raise ContractError("voltage_lift_material_or_initial_data_changed:"+key)
    for key in ("x_m", "cell_edges_m", "volumes_m3", "face_areas_m2", "column_scaling", "row_scaling"):
        if old["numeric_packet"][key] != packet[key]:
            raise ContractError("voltage_lift_geometry_or_scaling_changed:"+key)
    if any(k in old["controls"] for k in ("constraints_idx", "constraints_type")):
        raise ContractError("absolute_population_constraints_on_voltage_departures")
    proof = voltage_lift_wrms_policy(mapping, old["controls"], segments)
    controls = dict(old["controls"])
    controls.update(rtol=proof["rtol"], atol=proof["atol"])
    refinement = None
    if nonlin_conv_coef is not None:
        refinement = _voltage_lift_nonlinear_refinement(
            old["controls"], digest(old), nonlin_conv_coef)
        controls["nonlin_conv_coef"] = refinement["value"]
    guard = None
    if nonlin_guard is not None:
        guard = _voltage_lift_nonlinear_guard(
            old["controls"], digest(old), nonlin_guard, nonlin_trace_capacity)
        controls.update(nonlin_guard=guard["policy"], nonlin_trace_capacity=guard["trace_capacity"])
    elif type(nonlin_trace_capacity) is not int or nonlin_trace_capacity != 4096:
        raise ContractError("voltage_lift_invalid_nonlinear_guard")
    startup = None
    if first_step is not None:
        startup = _voltage_lift_startup_step(old["controls"], digest(old), first_step)
        controls["first_step"] = startup["value"]
    time_weights = None
    if time_weight_kappa is not None:
        time_weights, controls, proof = _voltage_lift_time_weights(
            mapping, controls, proof, digest(old), time_weight_kappa, qualification_policy)
    request = {
        "schema": "solarlab.voltage-lift-native-request.v1", "case_id": model.definition.id,
        "prior_request_sha256": digest(old), "numeric_packet": packet,
        "voltage_lift_map": mapping.payload(), "map_identity": mapping.identity,
        "segments": old["segments"], "controls": controls,
        "previous_controls": old["controls"], "weight_certificate": proof,
        "budgets": dict(old["budgets"]), "original_budgets": dict(old["budgets"]),
        "observation_times": old["observation_times"], "quadrature": old["quadrature"],
        "mandatory": dict(old["mandatory"], mapped_rate_projection_budget=True),
        "observation_policy": old["observation_policy"],
        "quadrature_uncertainty_scope": "original8/16/32 tables retain sample checks; a separately reviewed polynomial policy supplies ball radii and endpoint/input/clock/readback error bounds; empirical Gauss differences are not certificates",
        "observation_policy_authority": "native admission requires the exact reviewed request.interval_observation; the inherited observation_policy retains historical context only",
        "projection_policy": {
            "rates": "public RateView carries all direct-map words through residual and observation; first-word audit is counterfactual only",
            "point_current": "full-word raw/tangent currents and separate public row-arithmetic bounds; complete observation error allocation awaits qualification",
            "charge_integrands": ["sum(reservoir_conduction)", "left(total-conduction)", "right(total-conduction)"],
            "interval": "source-bound affine polynomial actions and bounded nonlinear physical-source integrals require every endpoint/input/clock/readback error term; first-word projection remains a counterfactual diagnostic",
            "prefix": "nonnegative exact-rational interval defects and observation bounds accumulate without resets or signed cancellation; original total and B/3 allocations remain",
            "continuum_certificate": False,
        },
        "full_word_consumer_qualification": {
            "native_policy_admitted": False,
            "local_action_scope": "affine contractions of represented state, endpoint increment and rate words with finite physical coefficients",
            "pending": ["full physical source and row error allocation", "continuous quadrature error certification",
                        "source-bound native protocol and independent refinements"],
            "physical_gates_changed": False,
        },
        "physical_domain_policy": old["physical_domain_policy"],
        "history_policy": "one original physical reference/map; preserve actual accepted native/stop and both event-side coordinates/rates, immutable source-owned polynomial snapshots, separately named declared_polynomial samples, all physical rate words and the same-state tangent; original interpolant/restore histories remain readable",
        "uncertified_by_first_pilot": old["uncertified_by_first_pilot"]+[
            "source-bound observation policy and whole-protocol independent time/state/space qualification"],
        "native_admission": False,
    }
    if qualification_policy is not None:
        request.update(qualification_policy=qualification_policy, qualification_ancestor=old)
    if refinement is not None:
        request["nonlinear_control_refinement"] = refinement
    if guard is not None:
        request["nonlinear_guard_policy"] = guard
    if startup is not None:
        request["startup_step_policy"] = startup
    if time_weights is not None:
        request["time_weight_policy"] = time_weights
    if segment_startup_overrides is not None:
        request["segment_startup_policy"] = _voltage_lift_segment_startup(
            request, segment_startup_overrides)
    if segment_frame is not None:
        request["segment_frame_policy"] = _segment_frame_policy(request, segment_frame)
    if qualification_policy is not None:
        from scripts.benchmarks.qualification_policy import check_applied_ceiling
        check_applied_ceiling(request)
    # This preparation context deliberately precedes the final request. Its
    # digest remains labelled as such; no self-referential hash is invented.
    context = AffineSamplingContext(model, request)
    z0 = np.zeros(model.layout.size)
    _, z0, zdot0, initial_record, frame_record = voltage_lift_segment_initialization(
        mapping, context, context.segments[0], 1, z0, model.reference)
    request.update(z0=z0.tolist(), zdot0=zdot0.tolist(), actual_initial_identity=initial_record["point_identity"],
                   initial_preparation=initial_record, preparation_context_sha256=context.request_sha256)
    if frame_record is not None:
        request["initial_segment_frame"] = frame_record
    return request


def validate_voltage_lift_native_request(mapping: AffineVoltageMap, segments, request: Mapping):
    """Fail closed on request/physical/allocation changes before importing IDA."""
    model = mapping.model
    if (request.get("schema") != "solarlab.voltage-lift-native-request.v1"
            or request.get("case_id") != model.definition.id
            or request.get("map_identity") != mapping.identity
            or digest(request.get("voltage_lift_map")) != mapping.identity
            or digest(model.numeric_packet()) != digest(request["numeric_packet"])
            or digest([asdict(s) for s in segments]) != digest(request["segments"])):
        raise ContractError("voltage_lift_native_request_binding")
    qualification_policy = request.get("qualification_policy")
    if qualification_policy is None and (model.intervals != 8 or model.layout.size > 45 or len(segments) != 3):
        raise ContractError("voltage_lift_native_scope")
    if qualification_policy is not None:
        qualification_policy = _qualification_scope(model, segments, qualification_policy)
        from scripts.benchmarks.qualification_policy import check_applied_ceiling, options
        check_applied_ceiling(request)
        chosen = dict(nonlin_conv_coef=request.get("nonlinear_control_refinement", {}).get("value"),
                      nonlin_guard=request.get("nonlinear_guard_policy", {}).get("policy"),
                      nonlin_trace_capacity=request.get("nonlinear_guard_policy", {}).get("trace_capacity", 4096),
                      first_step=request.get("startup_step_policy", {}).get("value"),
                      time_weight_kappa=request.get("time_weight_policy", {}).get("kappa"),
                      segment_startup_overrides=request.get("segment_startup_policy", {}).get("overrides"))
        if chosen != options(qualification_policy):
            raise ContractError("qualification_unbound_control_options")
    proof = voltage_lift_wrms_policy(mapping, request["previous_controls"], tuple(segments))
    controls = dict(request["previous_controls"], rtol=proof["rtol"], atol=proof["atol"])
    refinement = request.get("nonlinear_control_refinement")
    if refinement is not None:
        if not isinstance(refinement, Mapping):
            raise ContractError("voltage_lift_invalid_nonlinear_refinement")
        expected = _voltage_lift_nonlinear_refinement(
            request["previous_controls"], request["prior_request_sha256"], refinement.get("value"))
        if digest(refinement) != digest(expected):
            raise ContractError("voltage_lift_nonlinear_refinement_binding")
        controls["nonlin_conv_coef"] = expected["value"]
    guard = request.get("nonlinear_guard_policy")
    if guard is not None:
        if not isinstance(guard, Mapping):
            raise ContractError("voltage_lift_invalid_nonlinear_guard")
        expected = _voltage_lift_nonlinear_guard(
            request["previous_controls"], request["prior_request_sha256"],
            guard.get("policy"), guard.get("trace_capacity"))
        if digest(guard) != digest(expected):
            raise ContractError("voltage_lift_nonlinear_guard_binding")
        controls.update(nonlin_guard=expected["policy"], nonlin_trace_capacity=expected["trace_capacity"])
    if "startup_step_policy" in request:
        startup = request["startup_step_policy"]
        if not isinstance(startup, Mapping):
            raise ContractError("voltage_lift_invalid_startup_step")
        expected = _voltage_lift_startup_step(
            request["previous_controls"], request["prior_request_sha256"], startup.get("value"))
        if digest(startup) != digest(expected):
            raise ContractError("voltage_lift_startup_step_binding")
        controls["first_step"] = expected["value"]
    if "time_weight_policy" in request:
        time_weights = request["time_weight_policy"]
        if not isinstance(time_weights, Mapping):
            raise ContractError("voltage_lift_invalid_time_weights")
        expected, controls, proof = _voltage_lift_time_weights(
            mapping, controls, proof, request["prior_request_sha256"], time_weights.get("kappa"), qualification_policy)
        if digest(time_weights) != digest(expected):
            raise ContractError("voltage_lift_time_weights_binding")
    if (digest(proof) != digest(request["weight_certificate"])
            or digest(controls) != digest(request["controls"])
            or controls.get("calc_initcond") is not None
            or controls.get("linsolver") != "sparse"
            or any(k in controls for k in ("constraints_idx", "constraints_type"))):
        raise ContractError("voltage_lift_native_controls_changed")
    if "segment_startup_overrides" in request:
        raise ContractError("voltage_lift_segment_startup_unbound_override")
    if "segment_startup_policy" in request:
        for segment in segments:
            _voltage_lift_initialization_controls(request, segment)
    _validate_segment_frame_policy(request)
    if "segment_frame_policy" not in request and "initial_segment_frame" in request:
        raise ContractError("segment_frame_unbound_initialization")
    if "segment_frame_policy" in request:
        record = request.get("initial_segment_frame", {})
        if (record.get("request_sha256") != request["preparation_context_sha256"]
                or record.get("record_sha256") != digest({k: v for k, v in record.items() if k != "record_sha256"})
                or record.get("frame_identity") != digest(record.get("frame"))
                or record.get("frame", {}).get("parent_map_identity") != mapping.identity
                or record.get("frame", {}).get("segment_sha256") != digest(asdict(segments[0]))
                or record.get("frame", {}).get("logical_initialization_index") != 1
                or record.get("frame", {}).get("t0_hex") != float(segments[0].start).hex()
                or record.get("frame", {}).get("parent_input_sha256") != record.get("parent_input", {}).get("record_sha256")
                or request["initial_preparation"].get("segment_frame_identity") != record.get("frame_identity")
                or request["initial_preparation"].get("raw_z_hex") != record.get("native_initial_z_hex")
                or request["initial_preparation"].get("raw_zdot_hex") != record.get("native_initial_zdot_hex")
                or any(float(v) != 0 for v in (*request["z0"], *request["zdot0"]))):
            raise ContractError("segment_frame_initial_preparation_binding")
    if (digest(request["budgets"]) != digest(request["original_budgets"])
            or not all(request["mandatory"].values())
            or not request["mandatory"].get("mapped_rate_projection_budget")):
        raise ContractError("voltage_lift_native_gates_changed")
    qualification = request.get("full_word_consumer_qualification", {})
    if (qualification.get("native_policy_admitted") is not False
            or qualification.get("physical_gates_changed") is not False):
        raise ContractError("voltage_lift_full_word_qualification_not_reviewed")
    if (segments[0].start != 0 or any(a.end != b.start or a.voltage[1] != b.voltage[0]
                                    or a.photons[1] != b.photons[0]
                                    for a, b in zip(segments[:-1], segments[1:], strict=True))):
        raise ContractError("voltage_lift_native_protocol_discontinuity")
    for segment in segments:
        times = frozen_array(request["observation_times"][segment.id])
        if (times.ndim != 1 or times.size < 2 or times[0] != segment.start or times[-1] != segment.end
                or np.any(np.diff(times) <= 0)):
            raise ContractError("voltage_lift_native_observation_times")
    for order in (8, 16, 32):
        table = request["quadrature"][str(order)]
        nodes, weights = frozen_array(table["nodes"]), frozen_array(table["weights"])
        if (nodes.shape != (order,) or weights.shape != (order,) or np.any(np.abs(nodes) >= 1)
                or np.any(weights <= 0) or np.any(np.diff(nodes) <= 0)):
            raise ContractError("voltage_lift_native_quadrature")


def run_voltage_lift_native_pilot(mapping: AffineVoltageMap, segments: tuple[ProtocolSegment, ...],
                                  request: Mapping, admission: Mapping, emit, *,
                                  timing: RuntimeTiming | None = None) -> dict:
    """One separately admitted full protocol with unchanged physical gates.

    This controller retains actual IDA outputs before any interpolation and
    preserves both event sides. Source-owned phi/psi define separately named
    polynomial observations with explicit error bounds. They do not replace
    native rates, certify time/spatial accuracy, or restart a failed protocol.
    """
    if timing is None:
        timing = RuntimeTiming()
    model = mapping.model
    request_id = digest(dict(request))
    if (admission.get("request_sha256") != request_id
            or admission.get("map_identity") != mapping.identity
            or admission.get("voltage_lift_native_authorized") is not True
            or not str(admission.get("coordinator_message", "")).startswith("msg_")):
        raise ContractError("voltage_lift_native_pilot_not_admitted")
    validate_voltage_lift_native_request(mapping, segments, request)
    bindings = admission.get("source_sha256", {})
    if str(Path(__file__).resolve()) not in bindings:
        raise ContractError("voltage_lift_native_executing_kernel_not_bound")
    if str(Path(sys.modules[RuntimeTiming.__module__].__file__).resolve()) not in bindings:
        raise ContractError("runtime_timing_executing_source_not_bound")
    for path, expected in bindings.items():
        if sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ContractError("voltage_lift_native_source_changed")
    cancellation = Path(admission["cancel_path"]) if admission.get("cancel_path") else None
    if cancellation is not None and cancellation.exists():
        raise ContractError("voltage_lift_cancelled_before_native_import")
    sampling = AffineSamplingContext(model, request)
    if sampling.request_sha256 != request_id:
        raise ContractError("voltage_lift_request_changed_during_snapshot")
    request, segments = sampling.request_copy(), sampling.segments
    from scripts.benchmarks.native_observation import (
        AcceptedIntervalObserver, observation_record, read_native_basis_packet,
        require_loaded_observation_backend, validate_interval_observation_admission,
        prepare_charge_accumulator, charge_upper_summary,
    )
    from scripts.benchmarks.interval_observation import BallIntegrator, ChargePrefix, rational

    # Old requests and the historic Boolean remain closed. The new policy
    # requires independent review of these exact executing sources first.
    observer_policy = validate_interval_observation_admission(request, admission)
    require_loaded_observation_backend(observer_policy)
    from sksundae.ida import IDA

    started = time.perf_counter()
    budgets, controls = request["budgets"], request["controls"]
    counts = {"residual": 0, "jacobian": 0, "native_steps": 0, "onestep_returns": 0,
              "normal_queries": 0, "polynomial_queries": 0, "initializations": 0, "history_bytes": 0}
    with timing.phase('observation'):
        prefixes = {key: ChargePrefix.start((budgets["charge_C"],)*3)
                    for key in ("raw_polynomial", "same_state_affine_tangent")}
        charge_accumulator = prepare_charge_accumulator(request, observer_policy)
    previous_observer = None
    first_failure = last_attempt = last_numerical = last_accepted = None
    history = VoltageLiftHistory(mapping)
    parent_reference = history.reference_record
    z, predecessor = frozen_array(request["z0"]), model.reference
    rows, columns = model.graph.edges().T
    sparsity = csc_matrix((np.ones(len(rows)), (rows, columns)), shape=model.graph.shape)
    differential = np.unique(model.mass.tocoo().col)
    algebraic = np.setdiff1d(np.arange(model.layout.size), differential).tolist()

    def save(record):
        try:
            with timing.phase('history_io'):
                position = emit_record(record, emit, logical_bytes=counts["history_bytes"],
                                       total_output_bytes=budgets["total_output_bytes"])
        except HistoryLimitError as error:
            raise ContractError("voltage_lift_history_budget") from error
        counts["history_bytes"] = position.logical_bytes
        if position.encoding == "gzip":
            counts["history_encoded_bytes"] = position.encoded_bytes

    def resource_check():
        if cancellation is not None and cancellation.exists():
            raise ContractError("voltage_lift_cancelled")
        if time.perf_counter()-started > budgets["wall_s"]:
            raise ContractError("voltage_lift_wall_budget")
        if counts["residual"] > budgets["residual_calls"]:
            raise ContractError("voltage_lift_residual_budget")
        if counts["native_steps"] > budgets["native_steps"]:
            raise ContractError("voltage_lift_step_budget")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > budgets["rss_bytes"]:
            raise ContractError("voltage_lift_rss_budget")

    def snapshot_receipt(phase, segment, native):
        return {"phase": phase, "segment_id": segment.id, "time_hex": float(native["time"]).hex(),
                "success": native["success"], "status": native["status"], "message": native["message"],
                "z_hex": [float(v).hex() for v in native["z"]],
                "zdot_hex": [float(v).hex() for v in native["zdot"]]}

    def certify_current(observer, pair, native):
        nonlocal first_failure
        with timing.phase('observation'):
            evidence = observer.point_evidence(pair, native, budgets["current_A"])
        record = {"kind": "voltage_lift_current_certificate", "sample_record_sha256": pair[-1]["record_sha256"],
                  "evidence": observation_record(evidence)}
        save(record)
        if not evidence["passed"]:
            first_failure = first_failure or record
            raise ContractError("voltage_lift_raw_tangent_current_budget")
        return evidence

    def checked_sample(binding, native, previous, origin, observer=None):
        nonlocal first_failure
        resource_check()
        with timing.phase('observation'):
            pair = voltage_lift_native_sample(binding, history, native, previous, origin=origin)
        record = dict(pair[-1])
        record["observation_policy_sha256"] = digest(observer_policy)
        record["current_certificate_follows"] = observer is not None
        record["application_acceptance"] = "pending interval/current/state checks"
        if origin == "declared_polynomial":
            record.update(coefficient_frame_identity=native["coefficient_frame_identity"],
                          polynomial_path_identity=native["path_identity"],
                          native_getter_used=False, native_output=False)
        record.pop("record_sha256")
        record["record_sha256"] = digest(record)
        pair = (*pair[:-1], record)
        save(record)
        if not record["state_checks"]["passed"]:
            first_failure = first_failure or {"phase": "physical_state_quality", "record": record}
            raise ContractError("voltage_lift_native_or_interpolant_state_quality")
        if observer is not None:
            certify_current(observer, pair, native)
        return pair

    def pointer(pair):
        return {"record_sha256": pair[-1]["record_sha256"], "point_identity": pair[0].identity,
                "time_s": pair[0].time, "origin": pair[2].origin, "segment_id": pair[-1]["segment_id"]}

    try:
        save({"kind": "voltage_lift_reference", "request_sha256": request_id,
              "reference_digest": history.reference_digest, "reference": history.reference_record})
        for segment_index, segment in enumerate(segments):
            binding, z, initial_zdot, initial_proof, frame_record = voltage_lift_segment_initialization(
                mapping, sampling, segment, segment_index+1, z, predecessor)
            if frame_record is not None:
                save(frame_record)
                history = VoltageLiftHistory(binding.adapter.mapping, parent_reference=parent_reference)

            def residual(t, values, rates, output):
                nonlocal first_failure
                counts["residual"] += 1
                try:
                    resource_check()
                    binding.residual(t, values, rates, output)
                except BaseException as error:
                    if first_failure is None:
                        first_failure = {"phase": "residual", "segment_id": segment.id,
                                         "time_hex": float(t).hex(), "reason": str(error),
                                         "z_hex": [float(v).hex() for v in values],
                                         "zdot_hex": [float(v).hex() for v in rates]}
                        save({"kind": "first_callback_failure", "failure": first_failure})
                    raise

            def jacobian(t, values, rates, residual_value, cj, output):
                nonlocal first_failure
                counts["jacobian"] += 1
                try:
                    resource_check()
                    save({"kind": "jacobian_callback_context", "segment_id": segment.id,
                          "logical_initialization_index": segment_index+1, "map_identity": binding.adapter.mapping.identity,
                          "callback_index": counts["jacobian"], "request_sha256": request_id,
                          "time_hex": float(t).hex(), "cj_hex": float(cj).hex(),
                          "z_hex": [float(v).hex() for v in values],
                          "zdot_hex": [float(v).hex() for v in rates]})
                    binding.jacobian(t, values, rates, residual_value, cj, output)
                except BaseException as error:
                    if first_failure is None:
                        first_failure = {"phase": "jacobian", "segment_id": segment.id,
                                         "time_hex": float(t).hex(), "cj_hex": float(cj).hex(),
                                         "reason": str(error), "z_hex": [float(v).hex() for v in values],
                                         "zdot_hex": [float(v).hex() for v in rates]}
                        save({"kind": "first_callback_failure", "failure": first_failure})
                    raise

            if segment_index == 0:
                # Preparation and runtime have different context hashes. All
                # source physical data and supplied rate words must agree.
                for key in ("point_identity", "raw_z_hex", "raw_zdot_hex", "inputs_hex", "input_rates_hex",
                            "desired_physical_tangent_hex", "mapped_physical_rate_words_hex",
                            "desired_tangent_residual_SI", "represented_rate_residual_SI"):
                    if initial_proof[key] != request["initial_preparation"][key]:
                        raise ContractError("voltage_lift_initial_input_changed:"+key)
                if not np.array_equal(initial_zdot, request["zdot0"]):
                    raise ContractError("voltage_lift_frozen_initial_rate_changed")
            last_attempt = {"phase": "segment_initialization", "segment_id": segment.id,
                            "time_hex": float(segment.start).hex(), "proof": initial_proof}
            save({"kind": "voltage_lift_initialization_input", **last_attempt})
            resource_check()
            with timing.phase('ida_calls'):
                solver, control_receipt = _voltage_lift_segment_solver(
                    IDA, request, segment, segment_index+1, residual, jacobian, sparsity, algebraic,
                    frame=binding.adapter.mapping.frame)
            if control_receipt is not None:
                save(control_receipt)
            counts["initializations"] += 1
            with timing.phase('ida_calls'):
                initialized = snapshot_solver_result(solver.init_step(segment.start, z, initial_zdot))
            last_attempt = snapshot_receipt("initialization_return", segment, initialized)
            with timing.phase('ida_calls'):
                initial_stats = ida_statistics_snapshot(solver, segment.id, segment_index+1,
                    "initialization_return", segment.start, initialized["time"], logical_initialization=True)
            save(initial_stats)
            if (not initialized["success"] or initialized["time"] != segment.start
                    or not np.array_equal(initialized["z"], z)
                    or not np.array_equal(initialized["zdot"], initial_zdot)):
                raise ContractError("voltage_lift_initialization_changed_state_or_failed")
            _voltage_lift_guard_applied(initial_stats["raw_statistics"], controls)
            if frame_record is not None:
                _voltage_lift_frame_applied(initial_stats["raw_statistics"], binding.adapter.mapping.frame)
            left_pair = checked_sample(binding, initialized, predecessor, "segment_initial")
            left, last_numerical = left_pair[0], pointer(left_pair)
            left_native = initialized
            sample_times = np.asarray(request["observation_times"][segment.id])
            cursor = 0
            while cursor < len(sample_times) and sample_times[cursor] == segment.start:
                save({"kind": "requested_sample", **last_numerical})
                cursor += 1
            while left.time < segment.end:
                resource_check()
                if counts["native_steps"] >= budgets["native_steps"]:
                    raise ContractError("voltage_lift_step_budget")
                last_attempt = {"phase": "native_step", "segment_id": segment.id,
                                "target_time_hex": float(segment.end).hex(), "previous": last_numerical}
                with timing.phase('ida_calls'):
                    before_stats = ida_statistics_snapshot(solver, segment.id, segment_index+1,
                                                            "before_onestep", segment.end, logical_initialization=True)
                save(before_stats)
                try:
                    with timing.phase('ida_calls'):
                        native = snapshot_solver_result(solver.step(segment.end, method="onestep", tstop=segment.end))
                except BaseException:
                    try:
                        with timing.phase('ida_calls'):
                            failed_stats = ida_statistics_snapshot(solver, segment.id, segment_index+1,
                                                                   "onestep_exception", segment.end, before=before_stats,
                                                                   logical_initialization=True)
                        counts["native_steps"] += failed_stats["work_since_before"]["num_steps"]
                        save(failed_stats)
                    except Exception as stats_error:
                        save({"kind": "native_statistics_unavailable", "segment_id": segment.id,
                              "phase": "onestep_exception", "reason": str(stats_error)})
                    raise
                with timing.phase('ida_calls'):
                    after_stats = ida_statistics_snapshot(solver, segment.id, segment_index+1,
                                                           "after_onestep", segment.end, native["time"], before=before_stats,
                                                           logical_initialization=True)
                save(after_stats)
                counts["native_steps"] += after_stats["work_since_before"]["num_steps"]
                counts["onestep_returns"] += 1
                last_attempt = snapshot_receipt("native_return", segment, native)
                if not native["success"]:
                    raise ContractError("voltage_lift_solver_failure:"+native["message"])
                _voltage_lift_guard_applied(after_stats["raw_statistics"], controls)
                if frame_record is not None:
                    _voltage_lift_frame_applied(after_stats["raw_statistics"], binding.adapter.mapping.frame,
                                                after_step=True)
                t = native["time"]
                if not left.time < t <= segment.end:
                    raise ContractError("voltage_lift_interval_not_monotone")
                # Capture the immutable accepted interval before any output
                # query. IDA_NORMAL would invalidate consecutive-endpoint
                # authority even if it did not take an extra native step.
                with timing.phase('ida_calls'):
                    packet = solver.last_step_snapshot()
                save({"kind": "voltage_lift_native_observation_packet", "segment_id": segment.id,
                      "packet": observation_record(packet)})
                with timing.phase('observation'):
                    frame = read_native_basis_packet(packet,
                        expected_binding_identity=observer_policy["binding_identity"],
                        expected_header_sha256=observer_policy["header_sha256"], size=model.layout.size,
                        qualification_policy=request.get("qualification_policy"))
                    arithmetic = BallIntegrator(**{key: observer_policy["arithmetic"][key] for key in ("bits", "evaluations", "depth")})
                    observer = AcceptedIntervalObserver(binding, frame, arithmetic=arithmetic, previous=previous_observer)
                    observer.require_endpoint(left_native, predecessor=True)
                    observer.require_endpoint(native)
                    domain = observer.path_domain_evidence(budgets)
                save({"kind": "voltage_lift_polynomial_domain", "evidence": observation_record(domain)})
                if not domain["passed"]:
                    first_failure = first_failure or {"phase": "polynomial_domain", "evidence": observation_record(domain)}
                    raise ContractError("voltage_lift_polynomial_domain_or_inventory_budget")
                certify_current(observer, left_pair, left_native)
                if last_accepted is None:
                    last_accepted = pointer(left_pair)
                origin = "stop_output" if native["status"] == 1 else "native"
                right_pair = checked_sample(binding, native, left, origin, observer)
                right, last_numerical = right_pair[0], pointer(right_pair)
                cache = {left.time: left_pair, t: right_pair}
                active = True

                def query(when):
                    nonlocal last_attempt
                    if not active or not left.time <= when <= t:
                        raise ContractError("expired_or_foreign_voltage_lift_interval")
                    if when in cache:
                        return cache[when]
                    with timing.phase('observation'):
                        reconstructed = observer.reconstruction(when)
                    counts["polynomial_queries"] += 1
                    pair = checked_sample(binding, reconstructed, left, "declared_polynomial", observer)
                    cache[when] = pair
                    return pair

                try:
                    # The original nodes still receive the original state
                    # and instantaneous current checks. No Gauss difference
                    # is promoted to a reference-error certificate.
                    for order in (8, 16, 32):
                        table, h = request["quadrature"][str(order)], t-left.time
                        for node in table["nodes"]:
                            query(left.time+h*(float(node)+1)/2)
                    duration = observer.prepared.path.clock.tn-observer.prepared.path.clock.predecessor
                    error_allocation = rational(budgets["charge_C"])/12*duration/rational(segments[-1].end)/32
                    with timing.phase('observation'):
                        prefixes, evidence = observer.charge_evidence(left_pair, right_pair, prefixes,
                            absolute_error=error_allocation, charge_budget=budgets["charge_C"],
                            charge_accumulator=charge_accumulator)
                    while cursor < len(sample_times) and sample_times[cursor] <= t:
                        save({"kind": "requested_sample", **pointer(query(float(sample_times[cursor])))})
                        cursor += 1
                    interval = {"kind": ("voltage_lift_interval_charge_v2" if charge_accumulator is not None
                                         else "voltage_lift_interval_charge"), "segment_id": segment.id,
                                "left": pointer(left_pair), "right": pointer(right_pair),
                                "coefficient_frame_identity": frame.identity,
                                "observation": observation_record(evidence),
                                "arithmetic_calls": list(arithmetic.calls),
                                "endpoint_restore": None, "normal_query_performed": False,
                                "states_projected": False, "reference_estimates_are_continuum_certificates": False}
                    save(interval)
                    if not evidence["passed"]:
                        first_failure = first_failure or interval
                        raise ContractError("voltage_lift_interval_or_prefix_charge_budget")
                    last_accepted = last_numerical
                    previous_observer = observer
                finally:
                    active = False
                z = native["z"].copy()
                left_pair, left, left_native = right_pair, right, native
            if cursor != len(sample_times):
                raise ContractError("voltage_lift_missing_frozen_observation")
            if frame_record is not None:
                z = binding.adapter.mapping.frame.parent_state(segment.end, z)
            predecessor = left
        result = {"status": "completed_bounded_voltage_lift_native_pilot", "complete_protocol": True}
    except BaseException as error:
        result = {"status": "failed_bounded_voltage_lift_native_pilot", "complete_protocol": False,
                  "reason": str(error), "exception": type(error).__name__, "traceback": traceback.format_exc(),
                  "first_failure": first_failure or last_attempt}
    result.update(
        request_sha256=request_id, map_identity=mapping.identity, counts=counts,
        elapsed_s=time.perf_counter()-started, last_numerical=last_numerical,
        last_physically_accepted=last_accepted, last_attempt=last_attempt,
        cumulative_absolute_charge_bounds_C={k: observation_record(v.absolute_defects) for k, v in prefixes.items()},
        cumulative_observation_reference_bounds_C={k: observation_record(v.reference_errors) for k, v in prefixes.items()},
        observation_policy_sha256=digest(observer_policy),
        cumulative_scope="includes the fully evaluated first failed interval; no prefix reset at a protocol event",
        scientific_or_G2_qualification=False, DAE_time_accuracy_certified=False, continuum_space_accuracy_certified=False,
    )
    if charge_accumulator is not None:
        with timing.phase('observation'):
            result["cumulative_representation"] = observation_record(charge_upper_summary(charge_accumulator))
    return result

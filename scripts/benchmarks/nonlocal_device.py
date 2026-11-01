"""N02 small physical coupling witness, not a production device entrypoint.

The public graph has SI carrier, Poisson and contact rows.  Geometry and
assembly functions are pure metadata/algebra; constitutive callbacks are
supplied only by a separately admitted, source-bound witness runner.
No physical engine, state initializer or factorization is imported here.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
from scipy.sparse import csc_matrix

from scripts.benchmarks.contract_prototype import (
    COULOMB, PARTICLE, SECOND, VOLT, VOLUME, ContractError,
    EquationSpec, Geometry, Layout, Support, VariableSpec, frozen_array,
)
from scripts.benchmarks.nonlocal_sparse import NonlocalPlan, recycling_plan
from scripts.benchmarks.sparse_prototype import SparseGraph, TermSupport

SOURCE_CHARGE_C = 1.602176634e-19
MAX_WITNESS_NODES = 13
VARIABLES = ("n_m3", "p_m3", "phi_V")
CARRIER_ROWS = ("a_n", "b_p")


def _array(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ContractError("invalid_" + label)
    return array


def array_identity(*arrays):
    digest = sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def material_identity(material):
    def canonical(value):
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "words": np.asarray(value, dtype="<f8").tobytes().hex()}
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    return sha256(json.dumps(canonical(material), sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DeviceGraph:
    x: np.ndarray
    area_m2: float
    geometry: Geometry
    layout: Layout
    terms: tuple[TermSupport, ...]
    graph: SparseGraph
    nonlocal_plan: NonlocalPlan
    topology: str

    @property
    def nodes(self):
        return len(self.x)

    @property
    def interior(self):
        return np.arange(1, self.nodes - 1)

    def rows(self, name):
        part = self.graph.row_offsets[name]
        return np.arange(part.start, part.stop)


def physical_graph(x_m, area_m2, nonlocal_plan):
    """Compile declarations before coefficients, including all zero slots."""
    x = np.asarray(x_m, dtype=float)
    if (x.ndim != 1 or not 3 <= len(x) <= MAX_WITNESS_NODES
            or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0)
            or not np.isfinite(area_m2) or area_m2 <= 0):
        raise ContractError("small_physical_witness_geometry")
    n = len(x)
    if not isinstance(nonlocal_plan, NonlocalPlan) or nonlocal_plan.nodes != n:
        raise ContractError("nonlocal_geometry_binding")
    edges = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
    geometry = Geometry(area_m2 * np.diff(edges),
                        np.c_[np.arange(n - 1), np.arange(1, n)],
                        np.full(n - 1, area_m2))
    supports = (Support("nodes", "cell", (n,)),
                Support("interior", "cell", (n - 2,)),
                Support("contacts", "global", (2,)))
    density = PARTICLE / VOLUME
    variables = (VariableSpec("n_m3", "carriers", "nodes", (n,), density, lower=0),
                 VariableSpec("p_m3", "carriers", "nodes", (n,), density, lower=0),
                 VariableSpec("phi_V", "electrostatics", "nodes", (n,), VOLT, role="constraint"))
    equations = (
        EquationSpec("a_n", "carriers", "interior", (n - 2,), PARTICLE / SECOND,
                     derivative_support=VARIABLES),
        EquationSpec("b_p", "carriers", "interior", (n - 2,), PARTICLE / SECOND,
                     derivative_support=VARIABLES),
        EquationSpec("x_poisson", "electrostatics", "interior", (n - 2,), COULOMB,
                     role="constraint", derivative_support=VARIABLES),
        EquationSpec("y_phi_contact", "contacts", "contacts", (2,), VOLT,
                     role="constraint", derivative_support=("phi_V",)),
        EquationSpec("z_n_contact", "contacts", "contacts", (2,), density,
                     role="constraint", derivative_support=("n_m3",)),
        EquationSpec("zz_p_contact", "contacts", "contacts", (2,), density,
                     role="constraint", derivative_support=("p_m3",)),
    )
    layout = Layout(supports, variables, equations)
    eqs, specs = {e.id: e for e in equations}, {v.id: v for v in variables}
    terms = []
    for equation in equations:
        is_contact = "contact" in equation.id
        nodes = [0, n - 1] if is_contact else range(1, n - 1)
        for variable in equation.derivative_support:
            adjacent = ((equation.id == "a_n" and variable in ("n_m3", "phi_V"))
                        or (equation.id == "b_p" and variable in ("p_m3", "phi_V"))
                        or (equation.id == "x_poisson" and variable == "phi_V"))
            pairs = [(row, column) for row, node in enumerate(nodes)
                     for column in (range(node - 1, node + 2) if adjacent else [node])]
            terms.append(TermSupport("local:" + equation.id + ":" + variable,
                equation.owner, equation.id, variable, np.asarray(pairs, dtype=int),
                "port" if is_contact else "local", equation.unit / specs[variable].unit))
    for coupling in nonlocal_plan.couplings:
        if ":" in coupling.id:
            raise ContractError("ambiguous_nonlocal_term_identity")
        if any(row >= 2 * n for row in coupling.rows):
            raise ContractError("nonlocal_direct_Poisson_source_not_declared")
        for species, equation in enumerate(CARRIER_ROWS):
            rows = [row % n - 1 for row in coupling.rows
                    if row // n == species and 0 < row % n < n - 1]
            if not rows:
                continue
            for block, variable in enumerate(VARIABLES):
                columns = [column % n for column in coupling.columns if column // n == block]
                if columns:
                    terms.append(TermSupport.cartesian(
                        "nonlocal:" + coupling.id + ":" + equation + ":" + variable,
                        eqs[equation].owner, equation, variable, rows, columns,
                        unit=eqs[equation].unit / specs[variable].unit))
    topology = sha256(json.dumps({"layout": layout.identity,
        "geometry": array_identity(x, [area_m2], geometry.volumes),
        "nonlocal": nonlocal_plan.identity}, sort_keys=True).encode()).hexdigest()
    graph = SparseGraph(layout, tuple(terms), topology_identity=topology)
    return DeviceGraph(frozen_array(x), float(area_m2), geometry, layout,
                       tuple(terms), graph, nonlocal_plan, topology)


def physical_storage(plan):
    """dQ/dy in actual interior carrier rows; no storage at pinned rows."""
    mass = np.zeros((3 * plan.nodes, 3 * plan.nodes))
    for block, equation in enumerate(CARRIER_ROWS):
        mass[plan.rows(equation), block * plan.nodes + plan.interior] = plan.geometry.volumes[1:-1]
    return mass


def finite_volume_rows(plan, state, state_rate, *, charge_C, local_currents_A_m2,
                       net_local_generation_m3_s, displacement_C_m2,
                       charge_density_C_m3, contacts, recycling_m3_s=None,
                       nonlocal_currents_A_m2=None):
    """Assemble existing constitutive values once into physical FV rows.

    Face particle currents get face area, volume sources get cell volume.
    Endpoint half cells remain in reservoir and exterior-displacement ledgers,
    although their carrier solve rows are ideal Dirichlet constraints.
    """
    n = plan.nodes
    y = _array(state, (3 * n,), "physical_state")
    ydot = _array(state_rate, (3 * n,), "physical_rate")
    if np.any(y[:2 * n] <= 0) or not np.isfinite(charge_C) or charge_C <= 0:
        raise ContractError("positive_carriers_and_charge")
    currents = _array(local_currents_A_m2, (2, n - 1), "local_face_currents").copy()
    if nonlocal_currents_A_m2 is not None:
        currents += _array(nonlocal_currents_A_m2, (2, n - 1), "nonlocal_face_currents")
    generation = _array(net_local_generation_m3_s, (n,), "net_local_generation").copy()
    if recycling_m3_s is not None:
        generation += _array(recycling_m3_s, (n,), "recycling_generation")
    rho = _array(charge_density_C_m3, (n,), "charge_density")
    displacement = _array(displacement_C_m2, (n - 1,), "face_displacement")
    particle_flux = currents / charge_C
    particle_flux[0] *= -1
    rates = np.zeros((2, n))
    rates[:, :-1] -= particle_flux * plan.geometry.face_measures
    rates[:, 1:] += particle_flux * plan.geometry.face_measures
    rates += generation[None, :] * plan.geometry.volumes
    full = np.zeros(3 * n)
    for block, equation in enumerate(CARRIER_ROWS):
        full[plan.rows(equation)] = (plan.geometry.volumes[1:-1]
            * ydot[block * n + plan.interior] - rates[block, 1:-1])
    full[plan.rows("x_poisson")] = (plan.area_m2 * np.diff(displacement)
        - plan.geometry.volumes[1:-1] * rho[1:-1])
    for variable, equation in (("n_m3", "z_n_contact"), ("p_m3", "zz_p_contact"),
                               ("phi_V", "y_phi_contact")):
        field = y[plan.layout.offsets[variable]]
        full[plan.rows(equation)] = field[[0, -1]] - _array(contacts[variable], (2,), "contact_targets")
    exterior_D = np.array([displacement[0] - rho[0] * plan.geometry.volumes[0] / plan.area_m2,
                          displacement[-1] + rho[-1] * plan.geometry.volumes[-1] / plan.area_m2])
    # Positive exchange means particles supplied to the device by its reservoir.
    exchange = plan.geometry.volumes[[0, -1]] * ydot[:2 * n].reshape(2, n)[:, [0, -1]] - rates[:, [0, -1]]
    return {"residual": full, "nodal_carrier_rates_particle_s": rates,
            "face_currents_A_m2": currents, "exterior_displacement_C_m2": exterior_D,
            "reservoir_exchange_into_device_particle_s": exchange}


def component_rows(plan, normalized_value, normalized_jacobian, state_scale,
                   component_scale, kind):
    """Lift accepted N01 component values without its old 4I scaffold."""
    n = plan.nodes
    value = _array(normalized_value, (3 * n,), "component_value")
    jacobian = _array(normalized_jacobian, (3 * n, 3 * n), "component_jacobian")
    scales = _array(state_scale, (3 * n,), "component_state_scale")
    if np.any(scales <= 0) or not np.isfinite(component_scale) or component_scale <= 0:
        raise ContractError("positive_component_scales")
    if kind == "recycling_rate_density":
        weights = plan.geometry.volumes
    elif kind == "wkb_face_divergence":
        weights = np.full(n, plan.area_m2)
    else:
        raise ContractError("explicit_component_units_required")
    residual, full = np.zeros(3 * n), np.zeros((3 * n, 3 * n))
    for species, equation in enumerate(CARRIER_ROWS):
        source = species * n + plan.interior
        factor = -weights[1:-1] * component_scale
        residual[plan.rows(equation)] = factor * value[source]
        full[plan.rows(equation)] = factor[:, None] * jacobian[source] / scales[None, :]
    return residual, full


def assemble(plan, local_jacobian, coupling_jacobians, *, topology):
    """Fill declared SI blocks, rejecting even subnormal undeclared entries."""
    if topology != plan.topology:
        raise ContractError("structural_recompile_required")
    size = plan.layout.size
    local = _array(local_jacobian, (size, size), "local_jacobian")
    if set(coupling_jacobians) != {c.id for c in plan.nonlocal_plan.couplings}:
        raise ContractError("complete_nonlocal_occurrences_required")
    blocks = {"local": local}
    blocks.update({"nonlocal:" + key: _array(value, (size, size), "nonlocal_jacobian")
                   for key, value in coupling_jacobians.items()})
    allowed = {key: np.zeros((size, size), dtype=bool) for key in blocks}
    values = {}
    for term in plan.terms:
        key = "local" if term.id.startswith("local:") else ":".join(term.id.split(":")[:2])
        rows = term.pairs[:, 0] + plan.graph.row_offsets[term.equation].start
        cols = term.pairs[:, 1] + plan.layout.offsets[term.variable].start
        allowed[key][rows, cols] = True
        values[term.id] = blocks[key][rows, cols]
    for key, block in blocks.items():
        if np.any(block[~allowed[key]] != 0):
            raise ContractError("derivative_outside_declared_" + key)
    return plan.graph.assemble(values, supports=plan.terms, topology_identity=topology)


def normalized_system(matrix, residual, row_reference, variable_reference):
    shape = matrix.shape
    if shape[0] != shape[1]:
        raise ContractError("square_physical_operator_required")
    rows = _array(row_reference, (shape[0],), "row_reference")
    cols = _array(variable_reference, (shape[1],), "variable_reference")
    residual = _array(residual, (shape[0],), "physical_residual")
    if np.any(rows <= 0) or np.any(cols <= 0):
        raise ContractError("positive_dimensional_scales")
    dense = matrix.toarray() if hasattr(matrix, "toarray") else _array(matrix, shape, "matrix")
    return dense * cols[None, :] / rows[:, None], -residual / rows


def recycling_border(plan, U_normalized, V_normalized, state_scale, rate_scale):
    """Use only the source-justified photon rank, never a WKB phi rank."""
    n, rank = plan.nodes, len(plan.nonlocal_plan.couplings)
    if not rank or any(not c.id.startswith("recycling.") for c in plan.nonlocal_plan.couplings):
        raise ContractError("only_recycling_has_this_declared_border")
    u = _array(U_normalized, (3 * n, rank), "recycling_U")
    v = _array(V_normalized, (3 * n, rank), "recycling_V")
    scales = _array(state_scale, (3 * n,), "recycling_state_scale")
    if np.any(scales <= 0) or not np.isfinite(rate_scale) or rate_scale <= 0:
        raise ContractError("positive_recycling_scales")
    full_u = np.zeros((3 * n, rank))
    for block, equation in enumerate(CARRIER_ROWS):
        full_u[plan.rows(equation)] = (-plan.geometry.volumes[1:-1, None]
            * rate_scale * u[block * n + plan.interior])
    full_v = v / scales[:, None]
    return full_u, full_v


def auxiliary_system(plan, local_matrix, U, V, row_reference, variable_reference, surface_rate_reference):
    """[[J_local,U],[-V.T,I]] with dimensioned recycling closure scaling."""
    local = np.asarray(local_matrix, dtype=float)
    n = len(local)
    if local.shape != (n, n) or U.shape[0] != n or V.shape != U.shape:
        raise ContractError("physical_auxiliary_shapes")
    rows = _array(row_reference, (n,), "auxiliary_rows")
    cols = _array(variable_reference, (n,), "auxiliary_variables")
    aux = _array(surface_rate_reference, (U.shape[1],), "surface_rate_reference")
    if any(np.any(x <= 0) for x in (rows, cols, aux)):
        raise ContractError("positive_auxiliary_scales")
    # Keep structural slots even when an entire absorber is presently inactive.
    # The source declaration, not a numerical zero test, defines this pattern.
    if any(not c.id.startswith("recycling.") for c in plan.nonlocal_plan.couplings):
        raise ContractError("only_recycling_has_this_declared_border")
    pairs = set()
    for term in plan.terms:
        if term.id.startswith("local:"):
            pairs.update((int(i)+plan.graph.row_offsets[term.equation].start,
                          int(j)+plan.layout.offsets[term.variable].start) for i,j in term.pairs)
    allowed = np.zeros((n,n),bool)
    for i,j in pairs: allowed[i,j] = True
    if np.any(local[~allowed] != 0):
        raise ContractError("auxiliary_local_derivative_outside_support")
    ii, jj, data = [], [], []
    def append(i,j,value): ii.append(i); jj.append(j); data.append(value)
    for i,j in sorted(pairs): append(i,j,local[i,j]*cols[j]/rows[i])
    if len(plan.nonlocal_plan.couplings) != len(aux):
        raise ContractError("source_auxiliary_rank")
    for k,coupling in enumerate(plan.nonlocal_plan.couplings):
        receivers = [int(plan.rows(CARRIER_ROWS[r//plan.nodes])[r%plan.nodes-1])
                     for r in coupling.rows if 0 < r%plan.nodes < plan.nodes-1]
        senders = list(coupling.columns)
        if np.any(np.delete(U[:,k],receivers) != 0) or np.any(np.delete(V[:,k],senders) != 0):
            raise ContractError("auxiliary_derivative_outside_support")
        for i in receivers: append(i,n+k,U[i,k]*aux[k]/rows[i])
        for j in senders: append(n+k,j,-V[j,k]*cols[j]/aux[k])
        append(n+k,n+k,1.)
    return csc_matrix((data,(ii,jj)),shape=(n+len(aux),n+len(aux)))



def poisson_chain_support(plan):
    """Conservative full Dirichlet-Poisson inverse support, not coefficient sparsity."""
    n = plan.nodes
    poisson_rows = np.r_[plan.rows("x_poisson"), plan.rows("y_phi_contact")]
    retained_rows = np.array([row for row in range(3 * n) if row not in set(poisson_rows)])
    carrier_columns = np.arange(2 * n)
    direct = set()
    phi_affected, charged_columns = set(), set()
    row_map = {int(row): i for i, row in enumerate(retained_rows)}
    for term in plan.terms:
        rows = term.pairs[:, 0] + plan.graph.row_offsets[term.equation].start
        cols = term.pairs[:, 1] + plan.layout.offsets[term.variable].start
        for row, col in zip(rows, cols):
            if int(row) in row_map:
                if col < 2 * n:
                    direct.add((row_map[int(row)], int(col)))
                else:
                    phi_affected.add(row_map[int(row)])
            elif term.equation == "x_poisson" and col < 2 * n:
                charged_columns.add(int(col))
    chain = {(row, col) for row in phi_affected for col in charged_columns}
    return {"retained_rows": retained_rows, "poisson_rows": poisson_rows,
            "carrier_columns": carrier_columns, "phi_columns": np.arange(2 * n, 3 * n),
            "direct_edges": np.array(sorted(direct), dtype=int).reshape(-1, 2),
            "chain_edges": np.array(sorted(chain), dtype=int).reshape(-1, 2),
            "reduced_edges": np.array(sorted(direct | chain), dtype=int).reshape(-1, 2),
            "coefficient_mask_used": False}


def poisson_schur(plan, matrix, rhs, solve):
    """One explicitly counted block solve; all Poisson-chain columns survive.

    This is exact block elimination of the full linearized physical system.
    It does not project the starting state or assert g=0 at that state.
    """
    if not callable(solve):
        raise ContractError("explicit_counted_block_solver_required")
    n = plan.nodes
    full = _array(matrix, (3 * n, 3 * n), "coupled_jacobian")
    right = _array(rhs, (3 * n,), "coupled_rhs")
    support = poisson_chain_support(plan)
    rr, pr = support["retained_rows"], support["poisson_rows"]
    cc, pc = support["carrier_columns"], support["phi_columns"]
    solved = solve(full[np.ix_(pr, pc)], np.c_[full[np.ix_(pr, cc)], right[pr]])
    solved = _array(solved, (n, 2 * n + 1), "Poisson_block_solution")
    reduced = full[np.ix_(rr, cc)] - full[np.ix_(rr, pc)] @ solved[:, :2 * n]
    reduced_rhs = right[rr] - full[np.ix_(rr, pc)] @ solved[:, -1]
    return {"matrix": reduced, "rhs": reduced_rhs,
            "phi_per_carrier": -solved[:, :2 * n], "phi_rhs": solved[:, -1],
            "support": support}


def recycling_2d_declaration(nx, ny, absorber_y_ranges):
    """Source support only for y-then-x integration and uniform redistribution.

    Sender quadrature must use wx*wy and absorber *area*, not a flattened
    one-dimensional trapezoid.  Recipient FV volumes include explicit unit
    depth.  This declaration is not a physical 2D execution/size claim.
    """
    if type(nx) is not int or type(ny) is not int or min(nx, ny) < 2:
        raise ContractError("two_dimensional_axes_required")
    masks = []
    for lo, hi in absorber_y_ranges:
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= ny:
            raise ContractError("absorber_vertical_range")
        masks.append(tuple(j * nx + i for j in range(lo, hi) for i in range(nx)))
    return recycling_plan(nx * ny, masks)


def source_local_fields(plan, material, state, enter):
    """Deferred constitutive calls to existing laws, with meaningful entry counts.

    The selected witness requires the original sharp thermionic cap to remain
    strictly inactive.  An active/equal cap is retained as a branch failure,
    never replaced by uncapped transport or by another normalization.
    """
    if not callable(enter):
        raise ContractError("admitted_counted_physical_entry_required")
    enter("local_constitutive_point")
    from perovskite_sim.constants import Q
    if material["charge_C"] != Q or Q != SOURCE_CHARGE_C:
        raise ContractError("source_charge_constant_binding")
    from perovskite_sim.physics.continuity import carrier_face_currents
    from perovskite_sim.discretization.fe_operators import (
        sg_fluxes_n, sg_fluxes_p, thermionic_emission_flux,
    )
    from perovskite_sim.physics.recombination import (
        srh_recombination, radiative_recombination, auger_recombination,
    )
    n = plan.nodes
    y = _array(state, (3 * n,), "physical_state")
    electrons, holes, phi = y[:n], y[n:2 * n], y[2 * n:]
    if np.any(electrons <= 0) or np.any(holes <= 0):
        raise ContractError("nonpositive_physical_carriers")
    params = material["transport"]
    if (params.get("carrier_statistics", "maxwell_boltzmann") != "maxwell_boltzmann"
            or params.get("te_softness", 0) != 0 or params.get("exclusive_interface_faces")
            or params.get("het_recomb_despike", 0) != 0):
        raise ContractError("unsupported_N02_local_transport_branch")
    enter("carrier_face_currents")
    jn, jp = carrier_face_currents(plan.x, phi, electrons, holes, params)
    chi, gap = params.get("chi"), params.get("Eg")
    phi_n = phi if chi is None else phi + np.asarray(chi)
    phi_p = phi if chi is None else phi_n + np.asarray(gap)
    enter("uncapped_SG_electron")
    un = sg_fluxes_n(phi_n, electrons, np.diff(plan.x), params["D_n"], params["V_T"])
    enter("uncapped_SG_hole")
    up = sg_fluxes_p(phi_p, holes, np.diff(plan.x), params["D_p"], params["V_T"])
    margins = []
    for face in params.get("interface_faces", ()):
        physical_chi = np.asarray(chi if params.get("chi_te") is None else params["chi_te"])
        physical_gap = np.asarray(gap if params.get("Eg_te") is None else params["Eg_te"])
        barriers = (physical_chi[face] - physical_chi[face + 1],
                    physical_chi[face] + physical_gap[face]
                    - physical_chi[face + 1] - physical_gap[face + 1])
        for species, population, raw, barrier, richardson, dos_key in (
            ("n", electrons, un, barriers[0], params["A_star_n"], "N_C_node"),
            ("p", holes, up, barriers[1], params["A_star_p"], "N_V_node"),
        ):
            if abs(barrier) <= 0.05:
                continue
            dos = None
            if params.get("te_physical_norm", False):
                values = params.get(dos_key)
                if values is not None and all(np.isfinite(values[i]) and values[i] > 0 for i in (face, face + 1)):
                    dos = float(np.sqrt(values[face] * values[face + 1]))
            enter("thermionic_guard")
            cap = thermionic_emission_flux(float(population[face]), float(population[face + 1]),
                                          float(barrier), params["T"], float(richardson[face]), N_dos=dos)
            witness = {"face": int(face), "species": species, "uncapped_current": float(raw[face]),
                       "source_cap": float(cap), "normalization": "physical_DOS" if dos is not None else "legacy_empirical"}
            margins.append(witness)
            if not np.isfinite(cap) or not abs(raw[face]) < abs(cap):
                error = ContractError("active_or_unresolved_thermionic_cap")
                error.witness = witness
                raise error
    if not np.array_equal(jn, un) or not np.array_equal(jp, up):
        raise ContractError("declared_inactive_TE_branch_not_reproduced")
    rec = material["recombination"]
    enter("radiative_recombination")
    recombination = radiative_recombination(electrons, holes, rec["ni_sq"], rec["B_rad"])
    enter("auger_recombination")
    recombination += auger_recombination(electrons, holes, rec["ni_sq"], rec["C_n"], rec["C_p"])
    if rec["srh_enabled"]:
        enter("SRH_recombination")
        recombination += srh_recombination(electrons, holes, rec["ni_sq"], rec["tau_n"], rec["tau_p"], rec["n1"], rec["p1"])
    enter("Poisson_residual_assembly")
    eps = _array(material["epsilon_F_m"], (n,), "permittivity")
    if np.any(eps <= 0):
        raise ContractError("positive_permittivity")
    face_eps = 2 * eps[:-1] * eps[1:] / (eps[:-1] + eps[1:])
    q = float(material["charge_C"])
    rho = q * (holes - electrons + np.asarray(material["N_D_m3"]) - np.asarray(material["N_A_m3"]))
    return {"currents_A_m2": np.array([jn, jp]),
            "net_generation_m3_s": np.asarray(material["external_generation_m3_s"]) - recombination,
            "displacement_C_m2": -face_eps * np.diff(phi) / np.diff(plan.x),
            "charge_density_C_m3": rho, "thermionic_margins": margins,
            "branch_receipt": {"state": array_identity(y), "material": material_identity(material),
                               "topology": plan.topology, "strictly_inactive_TE": True}}


def source_local_jacobian(plan, material, state, cj_s_inv, branch_receipt, enter):
    """Existing SG/recombination derivatives plus actual cj*dQ/dy and pins.

    The caller must first establish the strictly inactive local cap branch at
    this exact state via source_local_fields.  The nonlocal derivative is
    added separately from source-bound N01 component data.
    """
    if not callable(enter):
        raise ContractError("admitted_counted_physical_entry_required")
    if branch_receipt != {"state": array_identity(state), "material": material_identity(material),
                          "topology": plan.topology, "strictly_inactive_TE": True}:
        raise ContractError("same_state_material_and_branch_receipt_required")
    enter("local_analytic_jacobian")
    from perovskite_sim.constants import Q
    if material["charge_C"] != Q or Q != SOURCE_CHARGE_C:
        raise ContractError("source_charge_constant_binding")
    from perovskite_sim.discretization.fe_operators import (
        sg_fluxes_n_jacobian, sg_fluxes_p_jacobian,
    )
    from perovskite_sim.physics.recombination import (
        srh_recombination_derivatives, radiative_recombination_derivatives,
        auger_recombination_derivatives,
    )
    n = plan.nodes
    y = _array(state, (3 * n,), "physical_state")
    if not np.isfinite(cj_s_inv) or cj_s_inv < 0:
        raise ContractError("physical_cj_s_inv")
    electrons, holes, phi = y[:n], y[n:2 * n], y[2 * n:]
    params, rec = material["transport"], material["recombination"]
    vt, dx, q = float(params["V_T"]), np.diff(plan.x), float(material["charge_C"])
    chi = params.get("chi")
    phi_n = phi if chi is None else phi + np.asarray(chi)
    phi_p = phi if chi is None else phi_n + np.asarray(params["Eg"])
    rates_jacobian = np.zeros((2 * n, 3 * n))
    for block, density, potential, diffusion, derivative in (
            (0, electrons, phi_n, params["D_n"], sg_fluxes_n_jacobian),
            (1, holes, phi_p, params["D_p"], sg_fluxes_p_jacobian)):
        # Each native helper evaluates and returns its SG face flux as well.
        enter("SG_jacobian_with_flux")
        tangent = derivative(potential, density, dx, diffusion, vt)
        particle_sign = (-1 if block == 0 else 1) / q
        face_derivatives = (tangent.density_left_derivative,
                            tangent.density_right_derivative,
                            tangent.potential_left_derivative,
                            tangent.potential_right_derivative)
        for face in range(n - 1):
            columns = (block * n + face, block * n + face + 1,
                       2 * n + face, 2 * n + face + 1)
            for node, incidence in ((face, -1), (face + 1, 1)):
                row = block * n + node
                for column, values in zip(columns, face_derivatives):
                    rates_jacobian[row, column] += (incidence * plan.area_m2
                                                    * particle_sign * values[face])
    enter("radiative_derivatives")
    radiative = radiative_recombination_derivatives(electrons, holes, rec["ni_sq"], rec["B_rad"])
    enter("auger_derivatives")
    auger = auger_recombination_derivatives(electrons, holes, rec["ni_sq"], rec["C_n"], rec["C_p"])
    dn = radiative.electron_density_derivative + auger.electron_density_derivative
    dp = radiative.hole_density_derivative + auger.hole_density_derivative
    if rec["srh_enabled"]:
        enter("SRH_derivatives")
        srh = srh_recombination_derivatives(electrons, holes, rec["ni_sq"], rec["tau_n"], rec["tau_p"], rec["n1"], rec["p1"])
        dn = dn + srh.electron_density_derivative
        dp = dp + srh.hole_density_derivative
    for block in (0, 1):
        rows = block * n + np.arange(n)
        rates_jacobian[rows, np.arange(n)] -= plan.geometry.volumes * dn
        rates_jacobian[rows, n + np.arange(n)] -= plan.geometry.volumes * dp
    full = cj_s_inv * physical_storage(plan)
    for block, equation in enumerate(CARRIER_ROWS):
        full[plan.rows(equation)] -= rates_jacobian[block * n + plan.interior]
    eps = np.asarray(material["epsilon_F_m"])
    face_eps = 2 * eps[:-1] * eps[1:] / (eps[:-1] + eps[1:])
    for row, node in zip(plan.rows("x_poisson"), plan.interior):
        a, b = plan.area_m2 * face_eps[node - 1] / dx[node - 1], plan.area_m2 * face_eps[node] / dx[node]
        full[row, 2 * n + node - 1:2 * n + node + 2] = [-a, a + b, -b]
        full[row, node] = q * plan.geometry.volumes[node]
        full[row, n + node] = -q * plan.geometry.volumes[node]
    for variable, equation in (("n_m3", "z_n_contact"), ("p_m3", "zz_p_contact"),
                               ("phi_V", "y_phi_contact")):
        full[plan.rows(equation), plan.layout.offsets[variable].start + np.array([0, n - 1])] = 1
    return full


def verify_component_binding(expected, observed):
    """Require an exact cached source/state/geometry binding before reuse.

    Hashes are validated by the runner against raw files, then compared here.
    A matching layout or a nearby state alone is never enough.
    """
    keys = {"source_request", "artifact", "state", "geometry", "topology"}
    if (set(expected) != keys or set(observed) != keys
            or any(not isinstance(value, str) or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value)
                   for value in [*expected.values(), *observed.values()])
            or expected != observed):
        raise ContractError("exact_component_source_state_geometry_binding_required")


def solve_diagnostics(matrix, rhs, solution):
    """Normwise and rowwise backward errors in a declared normalized system."""
    a = np.asarray(matrix, dtype=float)
    n = len(a)
    a = _array(a, (n, n), "solve_matrix")
    b = _array(rhs, (n,), "solve_rhs")
    x = _array(solution, (n,), "solve_solution")
    residual = a @ x - b
    denominator = np.abs(a) @ np.abs(x) + np.abs(b)
    rowwise = np.divide(np.abs(residual), denominator,
                        out=np.zeros(n), where=denominator != 0)
    if np.any((denominator == 0) & (residual != 0)):
        raise ContractError("nonzero_residual_on_zero_scale")
    scale = np.linalg.norm(a, np.inf) * np.linalg.norm(x, np.inf) + np.linalg.norm(b, np.inf)
    return {"residual": residual, "componentwise_backward_error": rowwise,
            "maximum_componentwise_backward_error": float(max(rowwise)),
            "normwise_backward_error": float(np.linalg.norm(residual, np.inf) / scale) if scale else 0.0}

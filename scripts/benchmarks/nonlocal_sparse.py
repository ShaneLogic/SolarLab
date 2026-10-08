"""Small P02-07 nonlocal support witness, never a production solver.

The coordinate order is (n, p, phi), each a nodal block. Graph/solve inputs
are explicitly scaled by the caller; ONE below is not a claim that physical
densities, potentials and rates have the same units. No legacy engine is
imported. The separately admitted driver supplies physical coefficients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from scripts.benchmarks.contract_prototype import (
    ONE, ContractError, EquationSpec, Layout, Support, VariableSpec, integer_indices,
)
from scripts.benchmarks.sparse_prototype import SparseGraph, TermSupport, ion_domains

# A small witness allocation guard, not an admission for a physical scale.
MAX_WITNESS_NODES = 128


def _count(value: int) -> int:
    values = integer_indices([value])
    if values.shape != (1,) or not 2 <= values[0] <= MAX_WITNESS_NODES:
        raise ContractError("nonlocal_witness_size_limit")
    return int(values[0])


def _nodes(values, count: int) -> tuple[int, ...]:
    nodes = integer_indices(values)
    if (nodes.ndim != 1 or np.any(nodes < 0) or np.any(nodes >= count)
            or len(set(nodes.tolist())) != len(nodes)):
        raise ContractError("invalid_nonlocal_nodes")
    return tuple(int(value) for value in nodes)


def _finite(values, shape):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ContractError("nonlocal_coefficient_shape_or_finiteness")
    return array


@dataclass(frozen=True)
class Coupling:
    id: str
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    # Ordered, per-energy branch labels, not coefficient values/nonzero masks.
    topology: str


@dataclass(frozen=True)
class NonlocalPlan:
    nodes: int
    couplings: tuple[Coupling, ...]

    def __post_init__(self):
        count = _count(self.nodes)
        object.__setattr__(self, "nodes", count)
        if len({c.id for c in self.couplings}) != len(self.couplings):
            raise ContractError("duplicate_nonlocal_identity")
        checked = []
        for c in self.couplings:
            if (not isinstance(c.id, str) or not c.id or c.id == "identity"
                    or not isinstance(c.topology, str)):
                raise ContractError("invalid_nonlocal_identity")
            checked.append(Coupling(c.id, _nodes(c.rows, 3 * count),
                                    _nodes(c.columns, 3 * count), c.topology))
        object.__setattr__(self, "couplings", tuple(checked))

    def to_document(self) -> dict:
        return {"schema": "solarlab.nonlocal-support.v1", "nodes": self.nodes,
                "coordinate_order": ["n", "p", "phi"],
                "couplings": [{"id": c.id, "rows": list(c.rows),
                               "columns": list(c.columns), "topology": c.topology}
                              for c in self.couplings]}

    @classmethod
    def from_document(cls, value: dict) -> NonlocalPlan:
        try:
            if (set(value) != {"schema", "nodes", "coordinate_order", "couplings"}
                    or value["schema"] != "solarlab.nonlocal-support.v1"
                    or value["coordinate_order"] != ["n", "p", "phi"]):
                raise ValueError
            couplings = []
            for row in value["couplings"]:
                if set(row) != {"id", "rows", "columns", "topology"}:
                    raise ValueError
                couplings.append(Coupling(row["id"], tuple(row["rows"]),
                                          tuple(row["columns"]), row["topology"]))
            return cls(value["nodes"], tuple(couplings))
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("invalid_nonlocal_document") from error

    @property
    def identity(self):
        return sha256(json.dumps(self.to_document(), sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def compile(self) -> SparseGraph:
        size = 3 * self.nodes
        layout = Layout((Support("nodes", "cell", (size,)),),
                        (VariableSpec("state", "nonlocal", "nodes", (size,), ONE),),
                        (EquationSpec("balance", "nonlocal", "nodes", (size,), ONE,
                                      derivative_support=("state",)),))
        diagonal = np.arange(size)
        terms = [TermSupport("identity", "nonlocal", "balance", "state",
                             np.column_stack((diagonal, diagonal)))]
        terms.extend(TermSupport.cartesian(c.id, "nonlocal", "balance", "state",
                                           c.rows, c.columns) for c in self.couplings)
        return SparseGraph(layout, terms, topology_identity=self.identity)

    def assemble(self, graph: SparseGraph, jacobians: dict, *, diagonal=1.0):
        """Assemble diagonal*I + J; reject any lost derivative, even a tiny one."""
        current = self.compile()
        if current.identity != graph.identity:
            raise ContractError("structural_recompile_required")
        if set(jacobians) != {c.id for c in self.couplings}:
            raise ContractError("nonlocal_coefficient_identity_mismatch")
        size = 3 * self.nodes
        if not np.isfinite(diagonal):
            raise ContractError("nonfinite_diagonal")
        values = {"identity": np.full(size, diagonal)}
        for term in current.supports:
            if term.id == "identity":
                continue
            array = _finite(jacobians[term.id], (size, size))
            covered = np.zeros(array.shape, dtype=bool)
            covered[term.pairs[:, 0], term.pairs[:, 1]] = True
            if np.any(array[~covered] != 0.0):
                raise ContractError("derivative_outside_declared_support")
            values[term.id] = array[term.pairs[:, 0], term.pairs[:, 1]]
        return graph.assemble(values, supports=current.supports,
                              topology_identity=current.topology_identity)


def recycling_plan(nodes: int, masks) -> NonlocalPlan:
    count = _count(nodes)
    couplings = []
    for occurrence, mask in enumerate(masks):
        selected = _nodes(mask, count)
        if tuple(sorted(selected)) != selected:
            raise ContractError("absorber_mask_must_follow_grid_order")
        carriers = selected + tuple(count + i for i in selected)
        couplings.append(Coupling(f"recycling.{occurrence}", carriers, carriers,
                                  json.dumps({"mask": selected, "occurrence": occurrence})))
    return NonlocalPlan(count, tuple(couplings))


def recycling_factors(x, n, p, b_rad, ni_sq, masks, escape, thickness):
    """MoL historical trapezoid/thickness feedback and its smooth-branch tangent.

    Returns G (m^-3 s^-1 for SI inputs), U,V with J=U@V.T in physical
    (n,p,phi) coordinates, and branch labels. At R_tot==0 a regular tangent
    is intentionally refused. Nonpositive thickness / escape>=1 / <2 mask
    nodes retain their source-defined inactive behavior and structural slots.
    """
    x = np.asarray(x, dtype=float)
    count = _count(x.size)
    x = _finite(x, (count,))
    if np.any(np.diff(x) <= 0):
        raise ContractError("nonincreasing_recycling_grid")
    n, p, b_rad, ni_sq = (_finite(v, (count,)) for v in (n, p, b_rad, ni_sq))
    plan = recycling_plan(count, masks)
    escape = _finite(escape, (len(plan.couplings),))
    thickness = _finite(thickness, escape.shape)
    u, v = np.zeros((3 * count, len(escape))), np.zeros((3 * count, len(escape)))
    generation, branches = np.zeros(count), []
    for k, coupling in enumerate(plan.couplings):
        indices = np.array([i for i in coupling.rows if i < count], dtype=int)
        if thickness[k] <= 0 or escape[k] >= 1 or len(indices) < 2:
            branches.append("inactive_parameter_or_mask")
            continue
        dx = np.diff(x[indices])
        weights = np.r_[dx, 0.0] / 2 + np.r_[0.0, dx] / 2
        total = float(weights @ (b_rad[indices] * (n[indices] * p[indices] - ni_sq[indices])))
        if not np.isfinite(total):
            raise ContractError("nonfinite_recycling_integral")
        if total == 0:
            raise ContractError("nonregular_recycling_threshold")
        if total < 0:
            branches.append("inactive_negative_integral")
            continue
        scale = (1 - escape[k]) / thickness[k]
        generation[indices] += total * scale
        u[list(coupling.rows), k] = scale
        v[indices, k] = weights * b_rad[indices] * p[indices]
        v[count + indices, k] = weights * b_rad[indices] * n[indices]
        branches.append("active_positive_integral")
    if not all(np.all(np.isfinite(a)) for a in (generation, u, v)):
        raise ContractError("nonfinite_recycling_coefficient")
    return generation, u, v, tuple(branches)


def auxiliary_matrix(plan: NonlocalPlan, u, v, *, diagonal=1.0):
    """Sparse [[dI,U],[-V^T,I]]; eliminate auxiliary variables to get dI+UV^T.

    Every absorber retains its separate occurrence, including overlaps and
    presently inactive terms. This is a normalized linear operator witness.
    """
    size, rank = 3 * plan.nodes, len(plan.couplings)
    if rank < 1 or not np.isfinite(diagonal):
        raise ContractError("invalid_auxiliary_rank_or_diagonal")
    u, v = _finite(u, (size, rank)), _finite(v, (size, rank))
    layout = Layout((Support("nodes", "cell", (size,)), Support("aux", "global", (rank,))),
                    (VariableSpec("state", "nonlocal", "nodes", (size,), ONE),
                     VariableSpec("transfer", "aux", "aux", (rank,), ONE, role="constraint")),
                    (EquationSpec("balance", "nonlocal", "nodes", (size,), ONE,
                                  derivative_support=("state", "transfer")),
                     EquationSpec("constraint", "aux", "aux", (rank,), ONE,
                                  role="constraint", derivative_support=("state", "transfer"))))
    d, a = np.arange(size), np.arange(rank)
    terms = [TermSupport("diagonal", "nonlocal", "balance", "state", np.c_[d, d]),
             TermSupport("aux.diagonal", "aux", "constraint", "transfer", np.c_[a, a], "nonlocal")]
    values = {"diagonal": np.full(size, diagonal), "aux.diagonal": np.ones(rank)}
    for k, c in enumerate(plan.couplings):
        allowed_u, allowed_v = np.zeros(size, bool), np.zeros(size, bool)
        allowed_u[list(c.rows)], allowed_v[list(c.columns)] = True, True
        if np.any(u[~allowed_u, k] != 0) or np.any(v[~allowed_v, k] != 0):
            raise ContractError("auxiliary_factor_outside_support")
        row = TermSupport.cartesian(c.id + ".row", "aux", "constraint", "state", [k], c.columns)
        col = TermSupport.cartesian(c.id + ".column", "nonlocal", "balance", "transfer", c.rows, [k])
        terms.extend((row, col))
        values[row.id], values[col.id] = -v[list(c.columns), k], u[list(c.rows), k]
    graph = SparseGraph(layout, terms, topology_identity=plan.identity)
    return graph, graph.assemble(values, supports=terms, topology_identity=plan.identity)


def wkb_plan(nodes: int, region, bounds, selectors) -> NonlocalPlan:
    """Structural device-v2 electron intraband footprint, without evaluating WKB.

    Each (a,b) contains both turning segments, as in intraband_path_flux.
    qFn depends on n and phi at the interpolated endpoints. The energy
    window/minima/peak and actions also depend on phi throughout the declared
    region, so those potential columns are never frozen away. The ordered
    bounds and extrema selectors invalidate reuse even if union edges match.
    """
    count = _count(nodes)
    region = tuple(integer_indices(region).tolist())
    if (len(region) != 4 or not 0 <= region[0] < region[1] < region[2] < region[3] <= count):
        raise ContractError("unsupported_wkb_region")
    pairs = integer_indices(bounds)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not 4 <= len(pairs) <= 128:
        raise ContractError("unsupported_wkb_quadrature")
    if np.any(np.diff(pairs[:, 0]) < 0) or np.any(np.diff(pairs[:, 1]) > 0):
        raise ContractError("wkb_paths_must_nest_in_increasing_energy_order")
    selected = tuple(integer_indices(selectors).tolist())
    # (left min node, right min node, peak node, base side 0/1).
    if (len(selected) != 4 or not region[0] <= selected[0] < region[1]
            or not region[2] <= selected[1] < region[3]
            or not region[1] <= selected[2] < region[2] or selected[3] not in (0, 1)):
        raise ContractError("unsupported_wkb_extrema")
    rows, density = set(), set()
    for a, b in pairs:
        if not region[0] <= a < b < region[3] - 1 or not a < selected[2] <= b:
            raise ContractError("wkb_turning_path_outside_region")
        rows.update(range(int(a), int(b) + 2))
        density.update((int(a), int(a) + 1, int(b), int(b) + 1))
    columns = sorted(density) + [2 * count + i for i in range(region[0], region[3])]
    token = json.dumps({"region": region, "bounds_in_energy_order": pairs.tolist(),
                        "selectors": selected, "device_version": "wkb-tunnelling-channel-device-v2"},
                       sort_keys=True, separators=(",", ":"))
    return NonlocalPlan(count, (Coupling("wkb.electron", tuple(sorted(rows)), tuple(columns), token),))


def disconnected_inventory(nodes: int, faces, diffusivity, weights):
    """Exact-D connectivity plus one inventory border per connected domain.

    This witnesses topology only. It does not remove a charged species when
    D=0, infer c0=0, or define time-dependent/frozen-ion physical semantics.
    """
    count = _count(nodes)
    domains = ion_domains(count, faces, diffusivity)
    weights = _finite(weights, (count,))
    if np.any(weights <= 0):
        raise ContractError("positive_inventory_weights_required")
    rank = len(domains.domains)
    layout = Layout((Support("nodes", "cell", (count,)), Support("totals", "global", (rank,))),
                    (VariableSpec("density", "bulk", "nodes", (count,), ONE),
                     VariableSpec("multiplier", "inventory", "totals", (rank,), ONE, role="constraint")),
                    (EquationSpec("balance", "bulk", "nodes", (count,), ONE,
                                  derivative_support=("density", "multiplier")),
                     EquationSpec("inventory", "inventory", "totals", (rank,), ONE,
                                  role="constraint", derivative_support=("density",))))
    rows = [(k, i) for k, domain in enumerate(domains.domains) for i in domain]
    row = TermSupport("inventory.row", "inventory", "inventory", "density", rows, "inventory")
    col = TermSupport("inventory.column", "bulk", "balance", "multiplier",
                      [(i, k) for k, i in rows], "inventory")
    diagonal = TermSupport("synthetic.diagonal", "bulk", "balance", "density", np.c_[np.arange(count), np.arange(count)])
    graph = SparseGraph(layout, (row, col, diagonal), topology_identity=domains.identity)
    values = {row.id: weights[row.pairs[:, 1]], col.id: weights[col.pairs[:, 0]],
              diagonal.id: np.ones(count)}
    return domains, graph, graph.assemble(values, supports=graph.supports,
                                          topology_identity=domains.identity)

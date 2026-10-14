"""Thin adapter between the R3 oracles and the new P04 implementation.

This is the ONLY file in ``tests/independent/p04`` that may touch the new
``solarlab`` P04-01 / P04-07 code.  The P04-01 hooks are wired (review R3-2)
to the public API W3a delivered on devel b1f953c (uncommitted worktree files):
``solarlab.discretization`` (+ its explicitly imported N modules
``stable_bernoulli`` and ``particle_flux``), ``solarlab.mesh`` and
``solarlab.physics.electrostatics``.  Only names exported in a module's
``__all__`` are used; no private helper is called.  The P04-07 hooks still
raise ``pytest.skip("awaiting interface: ...")`` until W3b is reviewed.

Wiring rules:

1. Call only the published public API; convert units and signs here, never
   in the tests.  Oracle conventions: SI units, +x directed particle fluxes,
   electrons z = -1, holes z = +1.
2. ``CHANGE_CLASS`` records the author's declared class (V1 Section 1.2) per
   capability; the review verifies each declaration independently.
3. Where W3a ships both an M (historical) and an N (new arithmetic) variant,
   the tests run both as separate subjects (``candidate_m`` / ``candidate_n``).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

AWAITING = "awaiting interface"

#: Author-declared change class per capability (W3a Report section 1 and 7;
#: verified in R3/ReviewW3a/ReviewV1.md).  P04-07 entries stay None.
CHANGE_CLASS: dict[str, str | None] = {
    "bernoulli": "M",               # solarlab.discretization.bernoulli*
    "bernoulli_stable": "N",        # stable_bernoulli.bernoulli_pair
    "scharfetter_gummel": "M",      # legacy_sg_current_{n,p}[_jacobian]
    "sg_particle_flux": "N",        # particle_flux.sg_particle_flux
    "face_average": "M",
    "poisson_1d": "M",              # PoissonOperator1D / PoissonFactorization
    "gauss_term_sink": "N",         # gauss_assembly.gauss_balance_with_term_sink
    "control_volumes": "M",
    "substrate_alignment": "M",
    "bulk_recombination": None,
    "trap_profile": None,
    "beer_lambert": None,
    "tmm": None,
}

#: Golden outputs with no new counterpart, with the reason (reported, not hidden).
GOLDEN_NOT_MIGRATED: dict[tuple[str, str], str] = {
    ("poisson_layered", "phi_spsolve"): (
        "legacy physics.poisson.solve_poisson (scipy spsolve fallback, not on any hot path) "
        "has no new counterpart; the LAPACK prefactored path is migrated"),
}


def _awaiting(capability: str) -> Any:
    pytest.skip(f"{AWAITING}: {capability}")


def _public(module: str, name: str) -> Any:
    """Return a published name; skip if the module is absent, fail if not public.

    A missing module (e.g. the N-class modules on an M-only commit) skips as
    "awaiting interface"; a name outside the module's ``__all__`` is a review
    failure, because the adapter may only use the published API.
    """

    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        pytest.skip(f"{AWAITING}: {module} is not present")
    if name not in getattr(mod, "__all__", ()):
        raise AssertionError(f"{module}.{name} is not in the published __all__")
    return getattr(mod, name)


def change_class(capability: str) -> str:
    value = CHANGE_CLASS.get(capability)
    if value is None:
        pytest.skip(f"{AWAITING}: change class (M/N/P) for {capability}")
    if value not in ("M", "N", "P"):
        raise AssertionError(f"invalid change class {value!r} for {capability}")
    return value


Q_LEGACY = 1.602176634e-19

#: Oracle geometry ids (p04_oracle_geometry) -> W3a published names.
def _geometry_name(oracle_id: str) -> str:
    HALF_ENDPOINT_V1 = _public("solarlab.mesh", "HALF_ENDPOINT_V1")
    INTEGER_ENDPOINT_V1 = _public("solarlab.mesh", "INTEGER_ENDPOINT_V1")
    PHYSICAL_BOUNDARY_TWO_SIDED_V1 = _public("solarlab.mesh", "PHYSICAL_BOUNDARY_TWO_SIDED_V1")
    TWO_SIDED_TRACE_V1 = _public("solarlab.mesh", "TWO_SIDED_TRACE_V1")

    table = {
        "legacy_1d_integer_endpoint_v1": INTEGER_ENDPOINT_V1,
        "legacy_2d_half_endpoint_v1": HALF_ENDPOINT_V1,
        "r1_physical_boundary_two_sided_v1": PHYSICAL_BOUNDARY_TWO_SIDED_V1,
        "qf_two_sided_trace_v1": TWO_SIDED_TRACE_V1,
    }
    return table[oracle_id]


# ---------------------------------------------------------------- P04-01


def bernoulli(x: np.ndarray) -> np.ndarray:
    """Historical element-wise B(x) (W3a M kernel)."""

    kernel = _public("solarlab.discretization", "bernoulli")

    return np.asarray(kernel(np.asarray(x, dtype=float)))


def bernoulli_derivative(x: np.ndarray) -> np.ndarray:
    """Historical element-wise B'(x) (W3a M kernel)."""

    kernel = _public("solarlab.discretization", "bernoulli_derivative")

    return np.asarray(kernel(np.asarray(x, dtype=float)))


def bernoulli_stable(x: np.ndarray) -> np.ndarray:
    """B(x) from the stable pair (W3a N kernel ``bernoulli_pair(x).forward``)."""

    bernoulli_pair = _public("solarlab.discretization.stable_bernoulli", "bernoulli_pair")

    return np.asarray(bernoulli_pair(np.asarray(x, dtype=float)).forward)


def bernoulli_derivative_stable(x: np.ndarray) -> np.ndarray:
    """B'(x) from the stable pair (``bernoulli_pair(x).forward_derivative``)."""

    bernoulli_pair = _public("solarlab.discretization.stable_bernoulli", "bernoulli_pair")

    return np.asarray(bernoulli_pair(np.asarray(x, dtype=float)).forward_derivative)


def _energy_step(psi: np.ndarray, V_T: float, z: int) -> np.ndarray:
    # W3a: D = potential-energy rise left->right in V_T units = z * dpsi / V_T.
    psi = np.asarray(psi, dtype=float)
    return z * ((psi[1:] - psi[:-1]) / V_T)


def sg_particle_flux(
    psi: np.ndarray, c: np.ndarray, h: np.ndarray, D: np.ndarray | float, V_T: float, z: int
) -> np.ndarray:
    """N variant: unified particle flux F = k (B(s) c_L - B(-s) c_R), k = D/h."""

    kernel = _public("solarlab.discretization.particle_flux", "sg_particle_flux")

    c = np.asarray(c, dtype=float)
    k = np.asarray(D, dtype=float) / np.asarray(h, dtype=float)
    return np.asarray(kernel(c[:-1], c[1:], _energy_step(psi, V_T, z), k).flux)


def sg_particle_flux_jacobian(
    psi: np.ndarray, c: np.ndarray, h: np.ndarray, D: np.ndarray | float, V_T: float, z: int
) -> dict[str, np.ndarray]:
    """N variant derivatives; dF/dpsi_{i+1} = z * dF/ds / V_T = -dF/dpsi_i."""

    kernel = _public("solarlab.discretization.particle_flux", "sg_particle_flux")

    c = np.asarray(c, dtype=float)
    k = np.asarray(D, dtype=float) / np.asarray(h, dtype=float)
    out = kernel(c[:-1], c[1:], _energy_step(psi, V_T, z), k)
    d_psi_right = z * np.asarray(out.energy_step_derivative) / V_T
    return {
        "flux": np.asarray(out.flux),
        "d_c_left": np.asarray(out.left_density_derivative),
        "d_c_right": np.asarray(out.right_density_derivative),
        "d_psi_left": -d_psi_right,
        "d_psi_right": d_psi_right,
    }


def sg_particle_flux_historical(
    psi: np.ndarray, c: np.ndarray, h: np.ndarray, D: np.ndarray | float, V_T: float, z: int
) -> np.ndarray:
    """M variant: historical electrical current converted, F_n = -J_n/q, F_p = J_p/q."""

    legacy_sg_current_n = _public("solarlab.discretization", "legacy_sg_current_n")
    legacy_sg_current_p = _public("solarlab.discretization", "legacy_sg_current_p")

    if z == -1:
        return -np.asarray(legacy_sg_current_n(psi, c, h, D, V_T)) / Q_LEGACY
    return np.asarray(legacy_sg_current_p(psi, c, h, D, V_T)) / Q_LEGACY


def sg_particle_flux_jacobian_historical(
    psi: np.ndarray, c: np.ndarray, h: np.ndarray, D: np.ndarray | float, V_T: float, z: int
) -> dict[str, np.ndarray]:
    legacy_sg_current_n_jacobian = _public("solarlab.discretization", "legacy_sg_current_n_jacobian")
    legacy_sg_current_p_jacobian = _public("solarlab.discretization", "legacy_sg_current_p_jacobian")

    sign, jac = (-1.0, legacy_sg_current_n_jacobian) if z == -1 else (1.0, legacy_sg_current_p_jacobian)
    j = jac(psi, c, h, D, V_T)
    return {
        "flux": sign * np.asarray(j.flux) / Q_LEGACY,
        "d_c_left": sign * np.asarray(j.density_left_derivative) / Q_LEGACY,
        "d_c_right": sign * np.asarray(j.density_right_derivative) / Q_LEGACY,
        "d_psi_left": sign * np.asarray(j.potential_left_derivative) / Q_LEGACY,
        "d_psi_right": sign * np.asarray(j.potential_right_derivative) / Q_LEGACY,
    }


def harmonic_face_mean(left: np.ndarray, right: np.ndarray, variant: str = "guarded") -> np.ndarray:
    """Face mean of each (left, right) pair.

    ``guarded``: ``guarded_harmonic_face_mean`` (transport channel with zero
    conductivity preserved); ``poisson``: ``harmonic_mean`` (Poisson eps
    channel, positive inputs only).
    """

    guarded_harmonic_face_mean = _public("solarlab.discretization", "guarded_harmonic_face_mean")
    harmonic_mean = _public("solarlab.discretization", "harmonic_mean")

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if variant == "poisson":
        return np.asarray(harmonic_mean(left, right))
    interleaved = np.stack([left, right]).T.ravel()
    return np.asarray(guarded_harmonic_face_mean(interleaved))[::2]


def geometry_names() -> tuple[str, ...]:
    GEOMETRY_NAMES = _public("solarlab.mesh", "GEOMETRY_NAMES")

    return tuple(GEOMETRY_NAMES)


def build_geometry(geometry: str, x: np.ndarray, interfaces: tuple[float, ...] = ()) -> Any:
    builder = _public("solarlab.mesh", "build_geometry")

    return builder(_geometry_name(geometry), np.asarray(x, dtype=float), tuple(interfaces))


def control_volume_widths(
    geometry: str, x: np.ndarray, interfaces: tuple[float, ...] = ()
) -> np.ndarray:
    """``ControlVolumeGeometry1D.cell_measures_m`` of the named geometry."""

    return np.asarray(build_geometry(geometry, x, interfaces).cell_measures_m)


def _layer_planes(thicknesses: tuple[float, ...]) -> tuple[float, ...]:
    ElectricalStack = _public("solarlab.mesh", "ElectricalStack")
    StackLayer = _public("solarlab.mesh", "StackLayer")

    stack = ElectricalStack.from_layers(
        [StackLayer(f"layer{k}", t, "electrical") for k, t in enumerate(thicknesses)])
    return stack.interface_positions_m


def _poisson_operator(geometry: str, x: np.ndarray, thicknesses: tuple[float, ...],
                      eps_r: tuple[float, ...]):
    RIGHT_LAYER_WINS = _public("solarlab.mesh", "RIGHT_LAYER_WINS")
    node_owner = _public("solarlab.mesh", "node_owner")
    PoissonOperator1D = _public("solarlab.physics.electrostatics", "PoissonOperator1D")

    planes = _layer_planes(thicknesses)
    geom = build_geometry(geometry, x, planes)
    owner = node_owner(np.asarray(x, dtype=float), thicknesses, RIGHT_LAYER_WINS)
    eps_node = np.asarray(eps_r, dtype=float)[owner]
    return geom, owner, eps_node, PoissonOperator1D.for_geometry(geom, eps_node)


def _solve_affine(residual, n: int) -> np.ndarray:
    """The residual is affine in phi: solve J phi = -r(0), then one correction."""

    from scipy.sparse.linalg import spsolve

    J = residual.jacobian
    phi = np.asarray(spsolve(J, -residual(np.zeros(n))))
    return phi - np.asarray(spsolve(J, residual(phi)))


def poisson_solve(
    geometry: str,
    x: np.ndarray,
    thicknesses: tuple[float, ...],
    eps_r: tuple[float, ...],
    rho_C_m3: tuple[float, ...],
    sheet_charge_C_m2: tuple[float, ...],
    phi_left: float,
    phi_right: float,
) -> np.ndarray:
    """Nodal potential from ``PoissonOperator1D.residual``/``jacobian``.

    Layer-wise eps/rho are assigned with the MoL ``RIGHT_LAYER_WINS`` owner;
    a sheet charge on a two-sided geometry enters through the eliminated
    Gauss jump (``add_sheet_charge_to_interior_residual``).
    """

    add_sheet_charge_to_interior_residual = _public("solarlab.physics.electrostatics", "add_sheet_charge_to_interior_residual")

    x = np.asarray(x, dtype=float)
    geom, owner, eps_node, op = _poisson_operator(geometry, x, thicknesses, eps_r)
    rho_node = np.asarray(rho_C_m3, dtype=float)[owner]
    sheets = np.asarray(sheet_charge_C_m2, dtype=float)
    if np.any(sheets != 0.0) and not geom.interfaces or (
            np.any(sheets != 0.0) and any(p.shared_node is not None for p in geom.interfaces)):
        pytest.skip(f"{AWAITING}: sheet charge on a shared-node geometry ({geometry})")

    def residual(phi):
        rows = np.array(op.residual(phi, rho_node, phi_left, phi_right))
        if np.any(sheets != 0.0):
            interior = rows[1:-1].copy()
            add_sheet_charge_to_interior_residual(interior, geom.interfaces, eps_node, sheets)
            rows[1:-1] = interior
        return rows

    residual.jacobian = op.jacobian()  # type: ignore[attr-defined]
    return _solve_affine(residual, x.size)


def poisson_solve_manufactured(
    geometry: str,
    x: np.ndarray,
    thicknesses: tuple[float, ...],
    eps_r: tuple[float, ...],
    rho_of_x: Any,
    exact_cell_charge_C_m2: np.ndarray,
    phi_left: float,
    phi_right: float,
) -> np.ndarray:
    """Manufactured source fed as the exact cell charge (rho_i = Q_i / w_i)."""

    x = np.asarray(x, dtype=float)
    geom, _, _, op = _poisson_operator(geometry, x, thicknesses, eps_r)
    rho_node = np.asarray(exact_cell_charge_C_m2, dtype=float) / np.asarray(geom.cell_measures_m)

    def residual(phi):
        return np.asarray(op.residual(phi, rho_node, phi_left, phi_right))

    residual.jacobian = op.jacobian()  # type: ignore[attr-defined]
    return _solve_affine(residual, x.size)


def poisson_vacuum_permittivity() -> float:
    VACUUM_PERMITTIVITY_F_M = _public("solarlab.discretization", "VACUUM_PERMITTIVITY_F_M")

    return float(VACUUM_PERMITTIVITY_F_M)


def gauss_balance(geometry: str, x: np.ndarray, thicknesses: tuple[float, ...],
                  eps_r: tuple[float, ...], phi: np.ndarray, rho_node: np.ndarray) -> np.ndarray:
    """N variant: Gauss balance through the public TermSink (all N rows)."""

    gauss_balance_with_term_sink = _public("solarlab.physics.electrostatics.gauss_assembly", "gauss_balance_with_term_sink")

    geom, _, _, op = _poisson_operator(geometry, np.asarray(x, dtype=float), thicknesses, eps_r)
    return np.asarray(gauss_balance_with_term_sink(op, geom, phi, rho_node))


def poisson_operator(geometry: str, x: np.ndarray, thicknesses: tuple[float, ...],
                     eps_r: tuple[float, ...]) -> Any:
    return _poisson_operator(geometry, np.asarray(x, dtype=float), thicknesses, eps_r)


def electrical_interface_alignment(
    roles: tuple[str, ...], thicknesses: tuple[float, ...]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """(full-stack interface indices, positions [m]) from ``ElectricalStack``."""

    ElectricalStack = _public("solarlab.mesh", "ElectricalStack")
    StackLayer = _public("solarlab.mesh", "StackLayer")

    stack = ElectricalStack.from_layers(
        [StackLayer(f"layer{k}", t, role) for k, (role, t) in enumerate(zip(roles, thicknesses))])
    indices = tuple(stack.full_interface_index(j) for j in range(stack.interface_count))
    return indices, tuple(stack.interface_positions_m)


# ---------------------------------------------------------------- P04-07


def bulk_recombination(
    n: np.ndarray,
    p: np.ndarray,
    params: Any,
    channels: tuple[str, ...] = ("srh", "radiative", "auger"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rate, dR/dn, dR/dp) for the requested channels only.

    ``params`` is ``p04_oracle_recombination.BulkParameters``; a channel not
    listed must be absent (no reaction instance).  TODO(W3b): public
    recombination reaction instances and their value/derivative evaluation.
    """

    return _awaiting("P04-07 bulk recombination value/derivatives")


def bulk_recombination_with_explicit_defect(
    n: np.ndarray,
    p: np.ndarray,
    params: Any,
    explicit_nodes: np.ndarray,
    defect_tau_n: float,
    defect_tau_p: float,
    defect_n1: float,
    defect_p1: float,
) -> np.ndarray:
    """Total bulk rate when an explicit neutral defect owns ``explicit_nodes``.

    TODO(W3b): public explicit-defect + lifetime-SRH composition.
    """

    return _awaiting("P04-07 explicit defect vs lifetime SRH composition")


def trap_profile(
    shape: str, x_local: np.ndarray, thickness: float, N_if: float, N_bulk: float,
    length: float, edge: str,
) -> np.ndarray:
    """N_t(x) [m^-3] for ``shape`` in {"exponential", "gaussian"}.

    TODO(W3b): public existing-trap-profile function.
    """

    return _awaiting("P04-07 trap profile")


def beer_lambert_generation(
    geometry: str,
    x: np.ndarray,
    alpha_node: np.ndarray,
    Phi: float,
    incidence: str = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """(G per node [m^-3 s^-1], control-volume weights [m]) for monochromatic BL.

    TODO(W3b): public BL generation for the named geometry; ``incidence`` is
    "left" (x=0) or "right" (x=L).
    """

    return _awaiting("P04-07 monochromatic Beer-Lambert generation")


def tmm_reflect_transmit_absorb(
    wavelengths_m: np.ndarray,
    layers: list[tuple[np.ndarray, float]],
    n_ambient: float,
    n_substrate: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, T, A per layer (M, n_wl)) at normal incidence.

    TODO(W3b): public TMM optical-resource evaluation.
    """

    return _awaiting("P04-07 TMM R/T/A")


def tmm_cell_absorbed_photons(
    wavelengths_m: np.ndarray,
    spectral_flux: np.ndarray,
    layers: list[tuple[np.ndarray, float]],
    faces_m: np.ndarray,
) -> np.ndarray:
    """Absorbed photons per cell [m^-2 s^-1] between consecutive ``faces_m``.

    TODO(W3b): public control-volume TMM generation (before dividing by w).
    """

    return _awaiting("P04-07 TMM control-volume absorbed photons")


def golden_case(case_id: str, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Evaluate a recorded golden case on the new implementation (P04-01 cases)."""

    def hx(values):
        return np.array([float.fromhex(v) for v in values])

    if case_id == "bernoulli":
        x = hx(inputs["x"])
        return {"B": bernoulli(x), "dB": bernoulli_derivative(x)}
    if case_id == "sg_jacobian":
        legacy_sg_current_n_jacobian = _public("solarlab.discretization", "legacy_sg_current_n_jacobian")
        legacy_sg_current_p_jacobian = _public("solarlab.discretization", "legacy_sg_current_p_jacobian")

        phi, dens, dx = hx(inputs["phi"]), hx(inputs["density"]), hx(inputs["dx"])
        D, VT = float.fromhex(inputs["D"]), float.fromhex(inputs["V_T"])
        out = {}
        for s, fn in (("n", legacy_sg_current_n_jacobian), ("p", legacy_sg_current_p_jacobian)):
            j = fn(phi, dens, dx, D, VT)
            for f in ("flux", "density_left_derivative", "density_right_derivative",
                      "potential_left_derivative", "potential_right_derivative"):
                out[f"{s}_{f}"] = np.asarray(getattr(j, f))
        return out
    if case_id == "poisson_layered":
        PoissonFactorization = _public("solarlab.physics.electrostatics", "PoissonFactorization")
        PoissonOperator1D = _public("solarlab.physics.electrostatics", "PoissonOperator1D")

        x, eps, rho = hx(inputs["x"]), hx(inputs["eps_r_node"]), hx(inputs["rho"])
        op = PoissonOperator1D.from_harmonic_nodes(x, eps)
        phi = PoissonFactorization.factor(op).solve(
            rho, float.fromhex(inputs["phi_left"]), float.fromhex(inputs["phi_right"]))
        return {"phi_prefactored": np.asarray(phi)}
    return _awaiting(f"golden case {case_id}")

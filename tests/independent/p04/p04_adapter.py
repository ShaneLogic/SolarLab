"""Thin adapter between the R3 oracles and the new P04 implementation.

This is the ONLY file in ``tests/independent/p04`` that may touch the new
``solarlab`` P04-01 / P04-07 code.  The P04-01 hooks are wired (review R3-2)
to the public API W3a delivered on devel b1f953c (uncommitted worktree files):
``solarlab.discretization`` (+ its explicitly imported N modules
``stable_bernoulli`` and ``particle_flux``), ``solarlab.mesh`` and
``solarlab.physics.electrostatics``.  Only names exported in a module's
``__all__`` are used; no private helper is called.  The P04-07 hooks are
wired (review R3-3) to W3b's ``solarlab.physics.recombination`` and
``solarlab.physics.generation``; legacy code is used here only to prepare
the inputs of a recorded golden case (stack, material alpha, n-k), never
for an expected value.

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
NOT_OFFERED = "not offered by the reviewed API"

#: Author-declared change class per capability (W3a Report section 1 and 7,
#: verified in R3/ReviewW3a/ReviewV1.md; W3b Report section 5 and
#: Task2/P0407SplitN1.md, verified in R3/ReviewW3b/ReviewV1.md).
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
    "bulk_recombination": "M",      # BulkRecombinationReaction (finite lifetimes)
    "srh_blocked_leg_tangent": "N", # N1: tau = +inf SRH partials NaN -> exact 0
    "trap_profile": "M",            # {exponential,gaussian}_edge_profile
    "beer_lambert": "M",            # beer_lambert_generation on a named geometry
    "tmm": "M",                     # transfer_matrix kernels + OpticalStackBinding
}

#: Golden outputs with no new counterpart, with the reason (reported, not hidden).
GOLDEN_NOT_MIGRATED: dict[tuple[str, str], str] = {
    ("poisson_layered", "phi_spsolve"): (
        "legacy physics.poisson.solve_poisson (scipy spsolve fallback, not on any hot path) "
        "has no new counterpart; the LAPACK prefactored path is migrated"),
    ("bl_ionmonger_benchmark_N30", "alpha_node"): (
        "per-node alpha is an output of the legacy material build (build_material_arrays), "
        "which is outside P04-07; the candidate consumes the legacy-built alpha as its input"),
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

    ``params`` is ``p04_oracle_recombination.BulkParameters``.  One
    ``BulkRecombinationReaction`` instance evaluates every channel; the full
    set returns its ``total``, a subset sums only the requested channel terms
    with the published ``sum_recombination_rates`` (an unlisted channel adds
    nothing, not a zero-coefficient term).
    """

    evaluation = _reaction(np.asarray(n, dtype=float).size, params).evaluate(
        np.asarray(n, dtype=float), np.asarray(p, dtype=float))
    if tuple(channels) == ("srh", "radiative", "auger"):
        term = evaluation.total
    else:
        sum_recombination_rates = _public("solarlab.physics.recombination", "sum_recombination_rates")
        term = sum_recombination_rates(*(getattr(evaluation, c) for c in channels))
    return term.rate, term.d_rate_dn, term.d_rate_dp


def _coefficients(node_count: int, params: Any, **override: Any) -> Any:
    BulkRecombinationCoefficients = _public("solarlab.physics.recombination", "BulkRecombinationCoefficients")

    values = dict(ni_sq_m6=params.ni_sq, tau_n_s=params.tau_n, tau_p_s=params.tau_p, n1_m3=params.n1,
                  p1_m3=params.p1, radiative_m3_s=params.B_rad, auger_n_m6_s=params.C_n,
                  auger_p_m6_s=params.C_p)
    values.update(override)
    return BulkRecombinationCoefficients(node_count, **values)


def _reaction(node_count: int, params: Any, explicit: Any = None, **override: Any) -> Any:
    BulkRecombinationReaction = _public("solarlab.physics.recombination", "BulkRecombinationReaction")

    return BulkRecombinationReaction(_coefficients(node_count, params, **override), explicit)


def bulk_recombination_terms(
    n: np.ndarray, p: np.ndarray, params: Any, **per_node: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-channel and total (rate, dR/dn, dR/dp); ``per_node`` overrides
    coefficients with per-node arrays (e.g. ``tau_n_s`` containing +inf)."""

    n_arr = np.asarray(n, dtype=float)
    evaluation = _reaction(n_arr.size, params, **per_node).evaluate(n_arr, np.asarray(p, dtype=float))
    return {name: (t.rate, t.d_rate_dn, t.d_rate_dp)
            for name, t in (("srh", evaluation.srh), ("radiative", evaluation.radiative),
                            ("auger", evaluation.auger), ("total", evaluation.total))}


def bulk_recombination_evaluation(
    n: np.ndarray, p: np.ndarray, params: Any,
) -> Any:
    """The public reaction evaluation, including its explicit domain predicate."""

    n_arr = np.asarray(n, dtype=float)
    return _reaction(n_arr.size, params).evaluate(n_arr, np.asarray(p, dtype=float))


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

    One ``NeutralDefectSpecies`` (n1/p1 zero off its nodes, as the legacy
    compiler stores them) in a ``NeutralExplicitDefectModel`` that owns
    exactly ``explicit_nodes``, bound into the same reaction instance.
    """

    NeutralDefectSpecies = _public("solarlab.physics.recombination", "NeutralDefectSpecies")
    NeutralExplicitDefectModel = _public("solarlab.physics.recombination", "NeutralExplicitDefectModel")

    mask = np.asarray(explicit_nodes, dtype=bool)
    digest = "c3" * 32
    species = NeutralDefectSpecies("layer[0]/r3_defect", digest, mask, float(defect_tau_n), float(defect_tau_p),
                                   np.where(mask, float(defect_n1), 0.0), np.where(mask, float(defect_p1), 0.0))
    model = NeutralExplicitDefectModel((species,), mask, (("layer[0]", digest),))
    n_arr = np.asarray(n, dtype=float)
    return _reaction(n_arr.size, params, model).evaluate(n_arr, np.asarray(p, dtype=float)).total.rate


def trap_profile(
    shape: str, x_local: np.ndarray, thickness: float, N_if: float, N_bulk: float,
    length: float, edge: str,
) -> np.ndarray:
    """N_t(x) [m^-3] for ``shape`` in {"exponential", "gaussian"}.

    The migrated historical kernels (they keep the ``length <= 0`` sentinel).
    """

    name = {"exponential": "exponential_edge_profile", "gaussian": "gaussian_edge_profile"}[shape]
    kernel = _public("solarlab.physics.recombination", name)
    return kernel(np.asarray(x_local, dtype=float), thickness, N_if, N_bulk, length, edge)


def trap_profile_binding(shape: str, x_local: np.ndarray, thickness: float, N_if: float, N_bulk: float,
                         length: float, edge: str, tau_n: Any, tau_p: Any) -> Any:
    """``TrapProfile(...).bind(...)``: (trap density, tau_n, tau_p) per node."""

    TrapProfile = _public("solarlab.physics.recombination", "TrapProfile")

    binding = TrapProfile(N_if, N_bulk, length, shape=shape, edge=edge).bind(
        np.asarray(x_local, dtype=float), thickness, tau_n, tau_p)
    return binding.trap_density_m3, binding.tau_n_s, binding.tau_p_s


def beer_lambert_generation(
    geometry: str,
    x: np.ndarray,
    alpha_node: np.ndarray,
    Phi: float,
    incidence: str = "left",
    interfaces: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """(G per node [m^-3 s^-1], control-volume weights [m]) for monochromatic BL.

    ``interfaces`` are the material planes a two-sided geometry needs.  The
    reviewed API illuminates the first node only (``INCIDENCE_SIDES`` is
    front-only, the single existing direction); a back-side request is not
    emulated here by mirroring, because that would test the adapter.
    """

    if incidence != "left":
        pytest.skip(f"{NOT_OFFERED}: back-side (x = L) incidence; the reviewed P04-07 API is front-only")
    beer_lambert_generation = _public("solarlab.physics.generation", "beer_lambert_generation")

    cv = build_geometry(geometry, x, interfaces)
    G = beer_lambert_generation(cv, np.asarray(alpha_node, dtype=float), Phi)
    return np.asarray(G), np.asarray(cv.cell_measures_m)


def tmm_reflect_transmit_absorb(
    wavelengths_m: np.ndarray,
    layers: list[tuple[np.ndarray, float]],
    n_ambient: float,
    n_substrate: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, T, A per layer (M, n_wl)) at normal incidence.

    R, T from ``tmm_reflectance`` / ``tmm_transmittance``; A_k as the
    difference of ``tmm_cumulative_absorptance`` across layer k's planes.
    """

    tmm_reflectance = _public("solarlab.physics.generation", "tmm_reflectance")
    tmm_transmittance = _public("solarlab.physics.generation", "tmm_transmittance")
    tmm_cumulative_absorptance = _public("solarlab.physics.generation", "tmm_cumulative_absorptance")

    wl = np.asarray(wavelengths_m, dtype=float)
    tmm, bounds = _tmm_layers(wl, layers)
    R = tmm_reflectance(tmm, wl, n_ambient, n_substrate)
    T = tmm_transmittance(tmm, wl, n_ambient, n_substrate)
    C = tmm_cumulative_absorptance(tmm, wl, bounds, bounds, n_ambient, n_substrate)
    return np.asarray(R), np.asarray(T), np.diff(np.asarray(C), axis=0)


def _tmm_layers(wl: np.ndarray, layers: list[tuple[Any, float]]) -> tuple[list[Any], np.ndarray]:
    TMMLayer = _public("solarlab.physics.generation", "TMMLayer")

    out = []
    for N, d in layers:
        N = np.broadcast_to(np.asarray(N, dtype=complex), wl.shape)
        out.append(TMMLayer(d=float(d), n=np.real(N).copy(), k=np.imag(N).copy()))
    return out, np.concatenate(([0.0], np.cumsum([float(d) for _, d in layers])))


def tmm_cell_absorbed_photons(
    wavelengths_m: np.ndarray,
    spectral_flux: np.ndarray,
    layers: list[tuple[np.ndarray, float]],
    faces_m: np.ndarray,
) -> np.ndarray:
    """Absorbed photons per cell [m^-2 s^-1] between consecutive ``faces_m``.

    ``tmm_absorbed_photon_flux_per_cell`` (ambient and exit index 1).
    """

    tmm_absorbed_photon_flux_per_cell = _public("solarlab.physics.generation", "tmm_absorbed_photon_flux_per_cell")

    wl = np.asarray(wavelengths_m, dtype=float)
    tmm, bounds = _tmm_layers(wl, layers)
    return np.asarray(tmm_absorbed_photon_flux_per_cell(
        tmm, wl, np.asarray(spectral_flux, dtype=float), np.asarray(faces_m, dtype=float), bounds, 1.0, 1.0))


def optical_binding(
    wavelengths_nm: np.ndarray,
    spectral_flux: np.ndarray,
    layers: list[tuple[str, Any, float, bool, bool]],
    *,
    ambient_index: float = 1.0,
    exit_index: float = 1.0,
    dependencies: Any = (),
    source_id: str = "r3-synthetic",
) -> Any:
    """``OpticalStackBinding`` from (name, N complex, thickness, incoherent, electrical)."""

    OpticalLayerSpec = _public("solarlab.physics.generation", "OpticalLayerSpec")
    OpticalStackBinding = _public("solarlab.physics.generation", "OpticalStackBinding")
    SpectralPhotonFlux = _public("solarlab.physics.generation", "SpectralPhotonFlux")

    wl_nm = np.asarray(wavelengths_nm, dtype=float)
    specs = []
    for name, N, d, incoherent, electrical in layers:
        N = np.broadcast_to(np.asarray(N, dtype=complex), wl_nm.shape)
        specs.append(OpticalLayerSpec(name, float(d), np.real(N).copy(), np.imag(N).copy(), incoherent, electrical))
    spectrum = SpectralPhotonFlux(wl_nm, np.asarray(spectral_flux, dtype=float), source_id)
    return OpticalStackBinding(tuple(specs), spectrum, ambient_index=ambient_index, exit_index=exit_index,
                               dependencies=dependencies)


def stale_profile_error() -> type:
    return _public("solarlab.physics.generation", "StaleGenerationProfileError")


def tmm_photon_budget(binding: Any) -> dict[str, Any]:
    """Integrated photon budget [m^-2 s^-1] of an ``optical_binding``."""

    budget = binding.photon_budget()
    return {
        "incident": budget.incident_m2_s, "reflected": budget.reflected_m2_s,
        "transmitted": budget.transmitted_m2_s, "electrical": budget.electrical_absorbed_m2_s,
        "parasitic": budget.parasitic_absorbed_m2_s, "layers": np.asarray(budget.layer_absorbed_m2_s),
        "closure": np.asarray(budget.spectral_closure_residual()),
    }


def optical_generation(binding: Any, geometry: str, x: np.ndarray,
                       interfaces: tuple[float, ...] = ()) -> tuple[np.ndarray, np.ndarray, Any]:
    """(G, weights, GenerationProfile) of a binding on a named geometry."""

    cv = build_geometry(geometry, x, interfaces)
    profile = binding.generation_profile(cv)
    return np.asarray(profile.source_m3_s), np.asarray(cv.cell_measures_m), profile


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
    if case_id in ("bl_synthetic", "bl_ionmonger_benchmark_N30", "tmm_ionmonger_benchmark_tmm_N30"):
        return _golden_generation(case_id, inputs, hx)
    if case_id == "tmm_synthetic_three_layer":
        tmm_reflectance = _public("solarlab.physics.generation", "tmm_reflectance")
        tmm_transmittance = _public("solarlab.physics.generation", "tmm_transmittance")
        tmm_cumulative_absorptance = _public("solarlab.physics.generation", "tmm_cumulative_absorptance")
        flux_per_cell = _public("solarlab.physics.generation", "tmm_absorbed_photon_flux_per_cell")
        TMMLayer = _public("solarlab.physics.generation", "TMMLayer")

        wl, faces, d = hx(inputs["wavelength_m"]), hx(inputs["faces"]), hx(inputs["d"])
        layers = [TMMLayer(d=float(dk), n=np.full(wl.size, float(nk)), k=hx(kk))
                  for nk, kk, dk in zip(inputs["n"], inputs["k"], d)]
        bounds = np.concatenate(([0.0], np.cumsum(d)))
        flux = np.full(wl.size, float.fromhex(inputs["spectral_flux"]))
        return {"R": tmm_reflectance(layers, wl, 1.0, 1.5), "T": tmm_transmittance(layers, wl, 1.0, 1.5),
                "C_faces": tmm_cumulative_absorptance(layers, wl, faces, bounds, 1.0, 1.5),
                "cell_absorbed": flux_per_cell(layers, wl, flux, faces, bounds, 1.0, 1.5)}
    if case_id == "recombination_total":
        from p04_oracle_recombination import BulkParameters

        par = BulkParameters(*(float.fromhex(inputs[k]) for k in
                               ("ni_sq", "tau_n", "tau_p", "n1", "p1", "B_rad", "C_n", "C_p")))
        rate, dn, dp = bulk_recombination(hx(inputs["n"]), hx(inputs["p"]), par)
        return {"rate": rate, "d_n": dn, "d_p": dp}
    if case_id == "trap_profiles":
        xl = hx(inputs["x_local"])
        args = (float.fromhex(inputs["thickness"]), float.fromhex(inputs["N_t_interface"]),
                float.fromhex(inputs["N_t_bulk"]))
        return {f"{shape}_{edge}_{length:.0e}": trap_profile(shape, xl, *args, length, edge)
                for shape in ("exponential", "gaussian") for edge in ("both", "left", "right")
                for length in inputs["lengths"]}
    return _awaiting(f"golden case {case_id}")


def _golden_generation(case_id: str, inputs: dict[str, Any], hx: Any) -> dict[str, np.ndarray]:
    """BL / TMM golden cases; legacy code only prepares the recorded inputs."""

    from pathlib import Path

    import p04_legacy as legacy

    x = hx(inputs["x"])
    if case_id == "bl_synthetic":
        G, w = beer_lambert_generation("legacy_1d_integer_endpoint_v1", x, hx(inputs["alpha_node"]),
                                       float.fromhex(inputs["Phi"]))
        return {"G": G, "dual_cell_widths": w}
    stack = legacy.device_fixture(Path(inputs["config"]).name)
    if case_id == "bl_ionmonger_benchmark_N30":
        from perovskite_sim.solver.mol import build_material_arrays

        alpha = build_material_arrays(x, stack).alpha  # input preparation (material build)
        G, w = beer_lambert_generation("legacy_1d_integer_endpoint_v1", x, alpha, float.fromhex(inputs["Phi"]))
    else:
        from perovskite_sim.physics.optical_stack import build_device_optical_stack

        OpticalLayerSpec = _public("solarlab.physics.generation", "OpticalLayerSpec")
        OpticalStackBinding = _public("solarlab.physics.generation", "OpticalStackBinding")
        SpectralPhotonFlux = _public("solarlab.physics.generation", "SpectralPhotonFlux")

        wl_nm = np.linspace(300.0, 1000.0, 200)  # the window _compute_tmm_generation models
        table = np.loadtxt(legacy.LEGACY_PACKAGE / "data" / "am15g.csv", delimiter=",", comments="#")
        spectrum = SpectralPhotonFlux.from_table(table[:, 0], table[:, 1], wl_nm, source_id="ASTM-G173-03")
        optical = build_device_optical_stack(stack, wl_nm)  # input preparation (n, k per slice)
        specs = tuple(OpticalLayerSpec(layer.name, t.d, t.n, t.k, t.incoherent, layer.role != "substrate")
                      for layer, part in zip(stack.layers, optical.physical_layer_slices)
                      for t in optical.layers[part])
        G, w, _ = optical_generation(OpticalStackBinding(specs, spectrum), "legacy_1d_integer_endpoint_v1", x)
    return {"G": G, "q_sum_G_w_A_m2": np.array([Q_LEGACY * np.sum(G * w)])}

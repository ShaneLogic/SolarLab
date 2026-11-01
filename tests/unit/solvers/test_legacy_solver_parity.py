"""Historical Newton loops, criteria and preparation (M class, bitwise).

Every value is compared with the legacy engine (read-only import) on the same
inputs by ``tobytes()``; failures compare type family and message. The legacy
loops are called directly: ``_damped_newton`` with the same residual callable,
``_solve_newton_stage`` with a stub system wrapping the same residual.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from perovskite_sim.experiments import quasi_fermi_steady_state as legacy_qf
from perovskite_sim.experiments import steady_state as legacy_ss
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver import degenerate_equilibrium as legacy_degenerate
from perovskite_sim.solver import mol as legacy_mol
from perovskite_sim.solver import newton as legacy_newton

from solarlab.solvers.nonlinear import (
    LEGACY_DEFAULT_ILLUMINATION_STAGES, LEGACY_INTERFACE_FALLBACK_ILLUMINATION_STAGES,
    LegacyNewtonStageError, legacy_continuity_current_bound, legacy_damped_fd_newton,
    legacy_illumination_stages, legacy_newton_stage, legacy_normalized_poisson_residual,
    legacy_quasi_neutral_seed, legacy_steady_state_tolerances,
)

ROOT = Path(__file__).resolve().parents[3] / "perovskite-sim"


def same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def residual_family(rng, size):
    """Smooth coupled systems with a known root, plus a matrix seed for variety."""
    A = rng.normal(size=(size, size)) + 3.0 * np.eye(size)
    root = rng.normal(size=size)
    scale = 10.0 ** rng.uniform(-2.0, 2.0)

    def F(z):
        return scale * (A @ (z - root) + 0.3 * np.sin(z - root) + 0.05 * (z - root) ** 3)
    return F, root


@pytest.mark.parametrize("seed", range(12))
def test_damped_newton_matches_the_legacy_helper_bitwise(seed):
    rng = np.random.default_rng(seed)
    size = int(rng.integers(1, 7))
    F, root = residual_family(rng, size)
    z0 = root + rng.normal(scale=float(rng.choice([0.1, 2.0, 8.0])), size=size)
    for max_it, tol, tol_step in ((60, 1e-12, 1e-9), (2, 1e-12, 1e-9), (40, 1e-3, 1e-1)):
        new = legacy_damped_fd_newton(F, z0, size, max_it=max_it, tol=tol, tol_step=tol_step)
        old = legacy_ss._damped_newton(F, z0, size, max_it=max_it, tol=tol, tol_step=tol_step)
        assert same(new[0], old[0]) and same(new[1], old[1])


def test_damped_newton_matches_the_legacy_ridge_and_stall_paths():
    def singular(z):            # column 1 never moves F: exact singular Jacobian
        return np.array([z[0] ** 2 - 4.0, z[0] - 2.0])

    def flat(z):                # minimum away from a root: the line search stalls
        return np.array([z[0] ** 2 + 1.0])

    for F, z0, size in ((singular, np.array([3.0, 1.0]), 2), (flat, np.array([0.5]), 1)):
        new = legacy_damped_fd_newton(F, z0, size, max_it=20, tol=1e-12)
        old = legacy_ss._damped_newton(F, z0, size, max_it=20, tol=1e-12)
        assert same(new[0], old[0]) and same(new[1], old[1])


def stage_systems(rng, size):
    F, root = residual_family(rng, size)

    def evaluate(z, fraction):
        return F(z) + fraction * 0.1

    legacy_system = SimpleNamespace(evaluate=lambda z, f: SimpleNamespace(residual=evaluate(z, f)))
    return evaluate, legacy_system, root


def outcome(call):
    try:
        z, iterations = call()
        return ("ok", np.asarray(z).tobytes(), iterations)
    except RuntimeError as error:
        return ("error", str(error))


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("rank_deficient", [False, True])
def test_newton_stage_matches_the_legacy_qf_stage_bitwise(seed, rank_deficient):
    rng = np.random.default_rng(100 + seed)
    size = int(rng.integers(1, 6))
    evaluate, system, root = stage_systems(rng, size)
    for fraction, start_scale, max_iterations, factor in (
            (1.0, 0.2, 40, 1.0), (1e-3, 4.0, 40, 1.0), (0.5, 4.0, 1, 1.0),
            (0.0, 30.0, 3, 2.0)):
        z0 = root + rng.normal(scale=start_scale, size=size)
        kwargs = dict(finite_difference_step=1e-7, residual_tolerance=1e-10,
                      max_iterations=max_iterations, rank_deficient=rank_deficient,
                      near_tolerance_stagnation_factor=factor)
        new = outcome(lambda: legacy_newton_stage(evaluate, z0, fraction, **kwargs))
        old = outcome(lambda: legacy_qf._solve_newton_stage(system, z0, fraction, **kwargs))
        assert new == old


def test_newton_stage_failure_messages_match_the_legacy_stage():
    def singular(z, fraction):
        return np.array([z[0] ** 2 - 4.0 + fraction, 0.0 * z[1] + 1.0])

    system = SimpleNamespace(evaluate=lambda z, f: SimpleNamespace(residual=singular(z, f)))
    z0 = np.array([3.0, 1.0])
    kwargs = dict(finite_difference_step=1e-7, residual_tolerance=1e-10, max_iterations=10)
    with pytest.raises(LegacyNewtonStageError) as new:
        legacy_newton_stage(singular, z0, 0.25, **kwargs)
    with pytest.raises(legacy_qf.QuasiFermiSteadyStateError) as old:
        legacy_qf._solve_newton_stage(system, z0, 0.25, **kwargs)
    assert str(new.value) == str(old.value)
    zero = SimpleNamespace(evaluate=lambda z, f: SimpleNamespace(residual=np.ones(2)))
    with pytest.raises(LegacyNewtonStageError) as new:
        legacy_newton_stage(lambda z, f: np.ones(2), z0, 1.0, rank_deficient=True, **kwargs)
    with pytest.raises(legacy_qf.QuasiFermiSteadyStateError) as old:
        legacy_qf._solve_newton_stage(zero, z0, 1.0, rank_deficient=True, **kwargs)
    assert str(new.value) == str(old.value)


@pytest.mark.parametrize("values", [
    LEGACY_DEFAULT_ILLUMINATION_STAGES, LEGACY_INTERFACE_FALLBACK_ILLUMINATION_STAGES,
    (1.0,), (0.0, 0.5, 1.0), (), (0.0, 0.5), (0.5, 0.5, 1.0), (-0.1, 1.0), (0.0, math.nan, 1.0),
])
@pytest.mark.parametrize("illuminated", [True, False])
def test_illumination_stages_match_the_legacy_validation(values, illuminated):
    def run(function):
        try:
            return ("ok", function(illuminated, values))
        except ValueError as error:
            return ("error", str(error))
    assert run(legacy_illumination_stages) == run(legacy_qf._validate_illumination_steps)
    assert LEGACY_DEFAULT_ILLUMINATION_STAGES == legacy_qf.DEFAULT_ILLUMINATION_STEPS
    assert (LEGACY_INTERFACE_FALLBACK_ILLUMINATION_STAGES
            == legacy_qf.INTERFACE_FALLBACK_ILLUMINATION_STEPS)


def stack_of(path):
    loader = load_scaps_yaml if "scaps_mirror" in path else load_device_from_yaml
    return loader(str(ROOT / path))


@pytest.mark.parametrize("path", [
    "configs/scaps_mirror_v2.yaml", "tests/fixtures/configs/nip_MAPbI3.yaml",
    "tests/fixtures/configs/ionmonger_benchmark.yaml", "tests/fixtures/configs/cigs_baseline.yaml",
    "tests/fixtures/configs/cSi_homojunction.yaml", "tests/fixtures/configs/dual_ion_demo.yaml",
])
def test_quasi_neutral_seed_matches_the_legacy_equilibrium_seed_bitwise(path):
    stack = stack_of(path)
    x = build_electrical_grid(stack, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = legacy_mol.build_material_arrays(x, stack)
        y = legacy_newton.solve_equilibrium(x, stack)
    state = legacy_mol.StateVec.unpack(y, x.size, mat.N_iface_state)
    n, p = legacy_quasi_neutral_seed(mat.N_D, mat.N_A, mat.ni_sq, n_left=mat.n_L,
                                     n_right=mat.n_R, p_left=mat.p_L, p_right=mat.p_R)
    assert same(n, state.n) and same(p, state.p)


def test_normalized_poisson_residual_matches_the_degenerate_lane_bitwise():
    stack = load_device_from_yaml(
        str(ROOT / "tests/fixtures/configs/csi_degenerate_pn_research.yaml"))
    x = build_electrical_grid(stack, 40)
    mat = legacy_mol.build_material_arrays(
        x, stack, carrier_statistics_transport="research_recombination_off")
    rng = np.random.default_rng(3)
    for _ in range(50):
        residual = rng.normal(scale=10.0 ** rng.uniform(-12, -2), size=x.size - 2)
        new = legacy_normalized_poisson_residual(
            residual, face_capacitance_F_m2=mat.poisson_factor.C,
            interior_widths_m=mat.poisson_factor.h_cell, acceptor_density_m3=mat.N_A,
            donor_density_m3=mat.N_D, V_bi_bc=mat.V_bi_bc, thermal_voltage_V=mat.V_T_device)
        assert same(new, legacy_degenerate._normalized_poisson_residual(residual, mat))
    # The bulk-trap lane adds the trap density to the reference (same expression).
    trap = legacy_normalized_poisson_residual(
        residual, face_capacitance_F_m2=mat.poisson_factor.C,
        interior_widths_m=mat.poisson_factor.h_cell, acceptor_density_m3=mat.N_A,
        donor_density_m3=mat.N_D, V_bi_bc=mat.V_bi_bc, thermal_voltage_V=mat.V_T_device,
        trap_density_m3=0.0)
    assert same(trap, legacy_degenerate._normalized_poisson_residual(residual, mat))


def test_steady_state_tolerances_and_current_bound_match_a_legacy_solve():
    stack = load_device_from_yaml(str(ROOT / "tests/fixtures/configs/nip_MAPbI3.yaml"))
    x = build_electrical_grid(stack, 24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = legacy_mol.build_material_arrays(x, stack)
        y0 = legacy_newton.solve_equilibrium(x, stack)
        # The legacy driver takes n_ref after its QFL-preserving Poisson relaxation.
        relaxed = legacy_ss._qfl_poisson_relax(x, mat, y0, 0.0)
        try:   # without transient assists the legacy Newton may stall from this seed
            result = legacy_ss.solve_steady_state(x, stack, 0.0, illuminated=False, y0=y0,
                                                  mat=mat, assist_times=())
            residual, bound = result.residual, result.continuity_current_bound
        except legacy_ss.SteadyStateError as error:
            result = None
            residual, bound = error.best_residual, error.continuity_current_bound
    N = x.size
    tolerances = legacy_steady_state_tolerances(
        tol=1.0e-6, tol_accept=0.5, max_continuity_current_error=0.1,
        n_ref=float(np.max(relaxed[: 2 * N])), device_width_m=float(x[-1] - x[0]))
    assert residual is not None and math.isfinite(residual)
    assert same(legacy_continuity_current_bound(tolerances, residual), bound)
    if result is not None and result.acceptance == "residual_converged":
        assert result.residual < tolerances.residual_tolerance
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="max_continuity_current_error must be finite"):
            legacy_steady_state_tolerances(tol=1e-6, tol_accept=0.5,
                                           max_continuity_current_error=bad, n_ref=1e24,
                                           device_width_m=1e-7)
    with pytest.raises(RuntimeError, match="finite continuity-current"):
        legacy_steady_state_tolerances(tol=1e-6, tol_accept=0.5,
                                       max_continuity_current_error=0.1, n_ref=0.0,
                                       device_width_m=1e-7)

"""Capture the P04 legacy golden fixture (historical reference, not truth).

Run from the repository root to (re)write the fixture::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        .venv/bin/python tests/independent/p04/p04_capture_golden.py --write

``capture_cases()`` is also imported by ``test_p04_legacy_golden.py`` to
prove the stored fixture is reproduced bit-for-bit by the legacy engine at
the recorded source digests.  Floats are stored as ``float.hex`` strings so
an M-class comparison can be bitwise.  Inputs are deterministic constants;
the two device cases read the shipped fixtures
``perovskite-sim/tests/fixtures/configs/ionmonger_benchmark{,_tmm}.yaml``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import p04_legacy as legacy  # noqa: E402

FIXTURE = _HERE / "fixtures" / "P04LegacyGoldenV1.json"
SCHEMA = "P04LegacyGoldenV1"
Q = 1.602176634e-19

BERNOULLI_X = tuple(
    sorted(
        {0.0, 1e-300, -1e-300, 701.0, 710.0, 745.0}
        | {s * v for v in (1e-12, 1e-8, 9.999e-5, 1e-4, 1e-2, 0.05, 0.5, 1.0, 2.0, 10.0, 50.0,
                           100.0, 699.0, 700.0, 1e3) for s in (1.0, -1.0)}
    )
)
THICKNESSES = (200e-9, 400e-9, 100e-9)
EPS_R = (3.0, 24.1, 10.0)
ALPHA = (0.0, 1.3e7, 0.0)
INTERVALS = (20, 40, 15)


def hexes(values) -> list[str]:
    return [float(v).hex() for v in np.asarray(values, dtype=float).ravel()]


def from_hex(values: list[str]) -> np.ndarray:
    return np.array([float.fromhex(v) for v in values])


def _layered_grid() -> np.ndarray:
    g = legacy.grid()
    return g.multilayer_grid([g.Layer(d, n) for d, n in zip(THICKNESSES, INTERVALS)])


def _right_owner(x: np.ndarray) -> np.ndarray:
    import p04_oracle_geometry as geo

    return geo.right_owner_layer_index(x, THICKNESSES)


def capture_cases() -> dict[str, dict]:
    fe, poi, gen, rec, trp, opt = (
        legacy.fe_operators(), legacy.poisson(), legacy.generation(),
        legacy.recombination(), legacy.traps(), legacy.optics(),
    )
    cases: dict[str, dict] = {}

    x = np.array(BERNOULLI_X)
    with np.errstate(all="ignore"):
        cases["bernoulli"] = {
            "function": "perovskite_sim.discretization.fe_operators.bernoulli/bernoulli_derivative",
            "inputs": {"x": hexes(x)},
            "outputs": {"B": hexes(fe.bernoulli(x)), "dB": hexes(fe.bernoulli_derivative(x))},
        }

    phi = np.array([0.0, 0.013, -0.2, -0.21, 0.4, 0.38, 1.1])
    dens = np.array([1e10, 3e12, 1e22, 5e21, 2e3, 7e5, 1e24])
    dx = np.array([1e-9, 3e-9, 2e-9, 5e-9, 1e-9, 4e-9])
    D, VT = 1.7e-4, 0.025851999786435
    jn = fe.sg_fluxes_n_jacobian(phi, dens, dx, D, VT)
    jp = fe.sg_fluxes_p_jacobian(phi, dens, dx, D, VT)
    cases["sg_jacobian"] = {
        "function": "perovskite_sim.discretization.fe_operators.sg_fluxes_{n,p}_jacobian (electric current A/m^2)",
        "inputs": {"phi": hexes(phi), "density": hexes(dens), "dx": hexes(dx), "D": float(D).hex(), "V_T": float(VT).hex()},
        "outputs": {
            f"{s}_{f}": hexes(getattr(j, f))
            for s, j in (("n", jn), ("p", jp))
            for f in ("flux", "density_left_derivative", "density_right_derivative",
                      "potential_left_derivative", "potential_right_derivative")
        },
    }

    xg = _layered_grid()
    owner = _right_owner(xg)
    eps_node = np.array(EPS_R)[owner]
    rho = Q * 1e22 * np.sin(2 * np.pi * xg / xg[-1])
    fac = poi.factor_poisson(xg, eps_node)
    cases["poisson_layered"] = {
        "function": "perovskite_sim.physics.poisson.solve_poisson / solve_poisson_prefactored",
        "inputs": {"x": hexes(xg), "eps_r_node": hexes(eps_node), "rho": hexes(rho),
                   "phi_left": 0.0.hex(), "phi_right": (1.1).hex()},
        "outputs": {"phi_spsolve": hexes(poi.solve_poisson(xg, eps_node, rho, 0.0, 1.1)),
                    "phi_prefactored": hexes(poi.solve_poisson_prefactored(fac, rho, 0.0, 1.1))},
    }

    alpha_node = np.array(ALPHA)[owner]
    G = gen.beer_lambert_generation(xg, alpha_node, 1.4e21)
    cases["bl_synthetic"] = {
        "function": "perovskite_sim.physics.generation.beer_lambert_generation",
        "inputs": {"x": hexes(xg), "alpha_node": hexes(alpha_node), "Phi": (1.4e21).hex()},
        "outputs": {"G": hexes(G), "dual_cell_widths": hexes(gen.dual_cell_widths(xg))},
    }

    from perovskite_sim.solver.mol import _compute_tmm_generation, build_material_arrays

    stack = legacy.device_fixture("ionmonger_benchmark.yaml")
    xd = legacy.electrical_grid(stack, 30)
    mat = build_material_arrays(xd, stack)
    Gd = gen.beer_lambert_generation(xd, mat.alpha, stack.Phi)
    w = gen.dual_cell_widths(xd)
    cases["bl_ionmonger_benchmark_N30"] = {
        "function": "build_material_arrays(...).alpha + beer_lambert_generation (fixture ionmonger_benchmark.yaml)",
        "inputs": {"config": "perovskite-sim/tests/fixtures/configs/ionmonger_benchmark.yaml", "N_grid": 30,
                   "x": hexes(xd), "Phi": float(stack.Phi).hex()},
        "outputs": {"G": hexes(Gd), "alpha_node": hexes(mat.alpha),
                    "q_sum_G_w_A_m2": float(Q * np.sum(Gd * w)).hex()},
    }

    stack_t = legacy.device_fixture("ionmonger_benchmark_tmm.yaml")
    xt = legacy.electrical_grid(stack_t, 30)
    Gt = _compute_tmm_generation(xt, stack_t)
    wt = gen.dual_cell_widths(xt)
    cases["tmm_ionmonger_benchmark_tmm_N30"] = {
        "function": "perovskite_sim.solver.mol._compute_tmm_generation (fixture ionmonger_benchmark_tmm.yaml; "
                    "200 wavelengths 300-1000 nm, AM1.5G)",
        "inputs": {"config": "perovskite-sim/tests/fixtures/configs/ionmonger_benchmark_tmm.yaml", "N_grid": 30,
                   "x": hexes(xt)},
        "outputs": {"G": hexes(Gt), "q_sum_G_w_A_m2": float(Q * np.sum(Gt * wt)).hex()},
    }

    n_wl = 100
    wl_nm = np.linspace(350, 750, n_wl)
    wl = wl_nm * 1e-9
    k_list = (np.where(wl_nm < 380, 0.05, 0.001), 0.4 * np.exp(-(wl_nm - 400) / 150),
              np.where(wl_nm < 420, 0.02, 0.001))
    n_list = (2.4, 2.5, 1.8)
    d_list = (50e-9, 400e-9, 150e-9)
    layers = [opt.TMMLayer(d=d, n=np.full(n_wl, n), k=k) for n, k, d in zip(n_list, k_list, d_list)]
    bounds = np.concatenate(([0.0], np.cumsum(d_list)))
    faces = np.linspace(0.0, bounds[-1], 31)
    S, _ = opt._transfer_matrix_stack(layers, wl, 1.0, 1.5)
    cases["tmm_synthetic_three_layer"] = {
        "function": "perovskite_sim.physics.optics tmm_reflectance/_transfer_matrix_stack/tmm_cumulative_absorptance/"
                    "tmm_absorbed_photon_flux_per_cell (n_ambient=1, n_substrate=1.5)",
        "inputs": {"wavelength_m": hexes(wl), "n": list(n_list), "k": [hexes(k) for k in k_list],
                   "d": hexes(d_list), "faces": hexes(faces), "spectral_flux": (3e24).hex()},
        "outputs": {
            "R": hexes(opt.tmm_reflectance(layers, wl, 1.0, 1.5)),
            "T": hexes(1.5 * np.abs(1.0 / S[:, 0, 0]) ** 2),
            "C_faces": hexes(opt.tmm_cumulative_absorptance(layers, wl, faces, bounds, 1.0, 1.5)),
            "cell_absorbed": hexes(opt.tmm_absorbed_photon_flux_per_cell(
                layers, wl, np.full(n_wl, 3e24), faces, bounds, 1.0, 1.5)),
        },
    }

    n_arr = np.array([1e10, 2.89e10, 1e20, 1e22, 1e23, 5.3e15])
    p_arr = np.array([1e22, 2.89e10, 1e20, 1e10, 1e23, 7.1e18])
    par = dict(ni_sq=2.89e10**2, tau_n=3e-9, tau_p=3e-7, n1=2.89e10, p1=2.89e10,
               B_rad=4e-17, C_n=2.8e-41, C_p=9.9e-42)
    tot = rec.total_recombination_derivatives(
        n_arr, p_arr, par["ni_sq"], par["tau_n"], par["tau_p"], par["n1"], par["p1"],
        par["B_rad"], par["C_n"], par["C_p"])
    cases["recombination_total"] = {
        "function": "perovskite_sim.physics.recombination.total_recombination_derivatives",
        "inputs": {"n": hexes(n_arr), "p": hexes(p_arr), **{k: float(v).hex() for k, v in par.items()}},
        "outputs": {"rate": hexes(tot.rate), "d_n": hexes(tot.electron_density_derivative),
                    "d_p": hexes(tot.hole_density_derivative)},
    }

    xl = np.linspace(0.0, 400e-9, 17)
    outs = {}
    for shape, fn in (("exponential", trp.exponential_edge_profile), ("gaussian", trp.gaussian_edge_profile)):
        for edge in ("both", "left", "right"):
            for length in (10e-9, 400e-9):
                outs[f"{shape}_{edge}_{length:.0e}"] = hexes(fn(xl, 400e-9, 1e23, 1e20, length, edge))
    cases["trap_profiles"] = {
        "function": "perovskite_sim.physics.traps.{exponential,gaussian}_edge_profile",
        "inputs": {"x_local": hexes(xl), "thickness": (400e-9).hex(), "N_t_interface": (1e23).hex(),
                   "N_t_bulk": (1e20).hex(), "lengths": [10e-9, 400e-9]},
        "outputs": outs,
    }
    return cases


def fixture_document() -> dict:
    head = subprocess.run(
        ["git", "-C", str(legacy.REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    extra = tuple(
        legacy.LEGACY_CONFIGS / name for name in ("ionmonger_benchmark.yaml", "ionmonger_benchmark_tmm.yaml")
    ) + tuple(sorted((legacy.LEGACY_PACKAGE / "data" / "nk").glob("*.csv")))
    return {
        "schema": SCHEMA,
        "purpose": "historical legacy outputs for M-class bitwise / N-class tolerance comparison; not truth",
        "captured_at_commit": head,
        "source_sha256": legacy.source_digests(extra),
        "cases": capture_cases(),
    }


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("pass --write to overwrite " + str(FIXTURE))
    with np.errstate(all="ignore"):
        document = fixture_document()
    FIXTURE.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(f"wrote {FIXTURE} ({len(document['cases'])} cases)")

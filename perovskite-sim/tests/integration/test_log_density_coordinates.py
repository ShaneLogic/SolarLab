import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import compute_current_components
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import (
    StateVec,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.tolerances import ComponentwiseAtol


def test_real_device_log_coordinate_observable_is_inside_refinement_envelope():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    x = multilayer_grid(
        [Layer(layer.thickness, 3) for layer in electrical_layers(stack)]
    )
    mat = build_material_arrays(x, stack)
    initial = solve_illuminated_ss(x, stack, V_app=0.0, mat=mat)
    atol = ComponentwiseAtol()
    terminal = {}

    for rtol in (3.0e-4, 1.0e-4):
        for coordinate in ("density", "research_log_density"):
            solution = run_transient(
                x,
                initial,
                (0.0, 1.0e-8),
                np.array([1.0e-8]),
                stack,
                illuminated=True,
                V_app=0.01,
                rtol=rtol,
                atol=atol,
                mat=mat,
                state_coordinates=coordinate,
            )
            assert solution.success
            state = solution.y[:, -1]
            current = compute_current_components(
                x, state, stack, 0.01, mat=mat
            ).J_total[0]
            assert np.isfinite(current)
            terminal[(rtol, coordinate)] = (state, float(current), solution)

    density_coarse = terminal[(3.0e-4, "density")]
    density_fine = terminal[(1.0e-4, "density")]
    log_coarse = terminal[(3.0e-4, "research_log_density")]
    log_fine = terminal[(1.0e-4, "research_log_density")]
    refinement_envelope = max(
        abs(density_fine[1] - density_coarse[1]),
        abs(log_fine[1] - log_coarse[1]),
    )
    roundoff_floor = 1.0e-12 * max(
        1.0, abs(density_fine[1]), abs(log_fine[1])
    )
    assert abs(density_fine[1] - log_fine[1]) <= (
        refinement_envelope + roundoff_floor
    )

    log_state = log_fine[0]
    unpacked = StateVec.unpack(log_state, len(x), mat.N_iface_state)
    assert np.all(unpacked.n > 0.0)
    assert np.all(unpacked.p > 0.0)
    active_ions = mat.P_ion0 > 0.0
    assert np.all(unpacked.P[active_ions] > 0.0)
    np.testing.assert_array_equal(unpacked.P[~active_ions], 0.0)
    assert log_fine[2].nfev <= 2 * density_fine[2].nfev

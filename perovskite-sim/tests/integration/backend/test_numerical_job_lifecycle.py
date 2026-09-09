"""Compare real 2D solves directly, in one worker, and in two workers."""

import threading

import numpy as np

from backend.jobs import JobRegistry, JobStatus
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d


def test_real_numerical_jobs_finish_and_match_direct_execution():
    stack = load_device_from_yaml("tests/fixtures/configs/twod/nip_MAPbI3_uniform.yaml")

    def solve():
        result = run_jv_sweep_2d(
            stack,
            lateral_length=300e-9,
            Nx=2,
            Ny_per_layer=3,
            V_max=0.05,
            V_step=0.05,
            settle_t=1e-8,
            lateral_bc="neumann",
            save_snapshots=False,
        )
        return {"V": result.V, "J": result.J}

    direct = solve()
    registry = JobRegistry()
    sequential = registry.submit(lambda _reporter: solve())
    status, one, error = registry.wait(sequential, timeout=30.0)
    assert status == JobStatus.DONE, error

    together = threading.Barrier(2, timeout=10.0)

    def concurrent(_reporter):
        together.wait()
        return solve()

    ids = [registry.submit(concurrent) for _ in range(2)]
    results = [one]
    for job_id in ids:
        status, result, error = registry.wait(job_id, timeout=30.0)
        assert status == JobStatus.DONE, error
        results.append(result)
    for result in results:
        np.testing.assert_array_equal(result["V"], direct["V"])
        np.testing.assert_allclose(result["J"], direct["J"], rtol=1e-8, atol=1e-8)
    assert all(not job.thread.is_alive() for job in registry._jobs.values())

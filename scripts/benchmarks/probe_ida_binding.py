"""Bounded analytic probes of the pinned IDA binding, not device qualification.

Run with the ``ida`` extra in an isolated environment. No SolarLab model or
experimental trajectory is imported. Expected missing capabilities are reported
as missing, even when this engineering probe exits successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Callable


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def run_probes() -> dict[str, Any]:
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"

    import numpy as np
    import scipy
    from scipy.sparse import csc_matrix
    import sksundae as sun
    from threadpoolctl import threadpool_info, threadpool_limits

    package_root = Path(sun.__file__).resolve().parent
    assert sun.__version__ == "1.1.3", "probe requires scikit-SUNDAE 1.1.3"
    assert sun.SUNDIALS_VERSION == "7.5.0", "probe requires SUNDIALS 7.5.0"
    sources = {}
    for path in [
        package_root / "ida/_solver.py",
        package_root / "_cy_ida.pyx",
        package_root / "py_config.pxi",
        *package_root.glob("_cy_ida*.so"),
    ]:
        if path.is_file():
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    cases: list[dict[str, Any]] = []

    def check(name: str, capability: str, fn: Callable[[], dict[str, Any]]) -> None:
        try:
            detail = fn()
            cases.append(
                dict(
                    name=name, capability=capability, observation_matched=True, **detail
                )
            )
        except Exception as exc:
            cases.append(
                dict(
                    name=name,
                    capability=capability,
                    observation_matched=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    def snapshot(result: Any) -> dict[str, Any]:
        return {
            "t": np.asarray(result.t).tolist(),
            "y": np.asarray(result.y).tolist(),
            "yp": np.asarray(result.yp).tolist(),
            "success": bool(result.success),
            "status": int(result.status),
            "nfev": int(result.nfev),
        }

    def decay(t: float, y: Any, yp: Any, res: Any) -> None:
        res[:] = yp + y

    def init_semantics() -> dict[str, Any]:
        def residual(t: float, y: Any, yp: Any, res: Any) -> None:
            res[0] = yp[0] + y[0]
            res[1] = y[0] - y[1]

        records = {}
        for mode in (None, "yp0", "y0"):
            solver = sun.ida.IDA(
                residual, algebraic_idx=[1], calc_initcond=mode, rtol=1e-9, atol=1e-11
            )
            result = solver.init_step(0.0, [1.0, 0.0], [-1.0, 0.0])
            values = np.empty(2)
            residual(0.0, result.y, result.yp, values)
            records[str(mode)] = {**snapshot(result), "residual": values.tolist()}
            if mode is None:
                assert result.success and abs(values[1]) == 1.0
            else:
                assert result.success and np.max(np.abs(values)) < 1e-7
                np.testing.assert_allclose(result.y, [1.0, 1.0], atol=1e-7)
        return {
            "records": records,
            "limitation": "init success without calc_initcond does not certify consistency; yp0 mode may also change algebraic y0",
        }

    def sparse_csc() -> dict[str, Any]:
        lengths = []

        def residual(t: float, y: Any, yp: Any, res: Any) -> None:
            res[0] = yp[0] + y[0]
            res[1] = y[0] - y[1]

        def jacobian(
            t: float, y: Any, yp: Any, res: Any, cj: float, values: Any
        ) -> None:
            lengths.append(values.shape)
            values[:] = [1.0 + cj, 1.0, -1.0]

        solver = sun.ida.IDA(
            residual,
            jacfn=jacobian,
            algebraic_idx=[1],
            linsolver="sparse",
            sparsity=csc_matrix([[1.0, 0.0], [1.0, 1.0]]),
            nthreads=1,
            rtol=1e-9,
            atol=[1e-11, 1e-11],
        )
        result = solver.solve([0.0, 0.2], [1.0, 1.0], [-1.0, -1.0])
        assert result.success and lengths and all(shape == (3,) for shape in lengths)
        error = float(np.max(np.abs(result.y[-1] - np.exp(-0.2))))
        assert error < 1e-7
        return {
            "jacobian_calls": len(lengths),
            "callback_shape": [3],
            "csc_order": ["(row0,col0)=1+cj", "(row1,col0)=1", "(row1,col1)=-1"],
            "endpoint_error": error,
            "linear_solver": "SuperLU_MT, nthreads=1",
        }

    def accepted_steps() -> dict[str, Any]:
        solver = sun.ida.IDA(decay, rtol=1e-9, atol=1e-11, max_step=0.02)
        initial = solver.init_step(0.0, [1.0], [-1.0])
        records = [snapshot(initial)]
        for _ in range(500):
            result = solver.step(0.2, method="onestep", tstop=0.2)
            assert result.success and result.t > records[-1]["t"]
            records.append(snapshot(result))
            if result.status == 1:
                break
        assert abs(records[-1]["t"] - 0.2) < 1e-13
        assert len(records) > 3
        state_error = max(abs(r["y"][0] - np.exp(-r["t"])) for r in records)
        derivative_error = max(abs(r["yp"][0] + np.exp(-r["t"])) for r in records)
        assert state_error < 1e-7 and derivative_error < 1e-6
        return {
            "records": records,
            "state_error": float(state_error),
            "derivative_error": float(derivative_error),
            "scope": "native onestep endpoints only; event roots are tested separately",
        }

    def interpolation() -> dict[str, Any]:
        solver = sun.ida.IDA(decay, rtol=1e-8, atol=1e-10, max_step=0.01)
        solver.init_step(0.0, [1.0], [-1.0])
        previous = 0.0
        for _ in range(100):
            result = solver.step(0.2, method="onestep")
            assert result.success
            if result.t > 0.04:
                break
            previous = float(result.t)
        end = float(result.t)
        mid = (previous + end) / 2
        value = solver.step(mid, method="normal")
        assert value.success and abs(value.t - mid) < 1e-13
        assert abs(value.y[0] - np.exp(-mid)) < 1e-6
        assert abs(value.yp[0] + np.exp(-mid)) < 1e-5
        try:
            outside = solver.step(previous - 5 * (end - previous), method="normal")
            rejected = not outside.success
            outside_record = snapshot(outside)
        except (ValueError, RuntimeError) as exc:
            rejected = True
            outside_record = {"error": f"{type(exc).__name__}: {exc}"}
        assert rejected
        return {
            "last_native_interval": [previous, end],
            "interpolated": snapshot(value),
            "outside_interval": outside_record,
            "arbitrary_derivative_api": hasattr(solver, "get_dky"),
            "limitation": "public interpolation only through step(normal), within last native interval; yp available, no public arbitrary-order derivative/history object",
        }

    def stop_persistence() -> dict[str, Any]:
        def constant(t: float, y: Any, yp: Any, res: Any) -> None:
            res[:] = yp - 1.0

        solver = sun.ida.IDA(constant, first_step=0.02, max_step=0.02)
        solver.init_step(0.0, [0.0], [1.0])
        first = solver.step(0.2, method="onestep", tstop=0.025)
        without_stop = [snapshot(first)]
        for _ in range(5):
            current = solver.step(0.2, method="onestep")
            assert current.success and current.status == 0
            without_stop.append(snapshot(current))
            if current.t > 0.025:
                break
        assert without_stop[-1]["t"] > 0.025
        repeated = sun.ida.IDA(constant, first_step=0.02, max_step=0.02)
        repeated.init_step(0.0, [0.0], [1.0])
        records = []
        for _ in range(5):
            current = repeated.step(0.2, method="onestep", tstop=0.025)
            records.append(snapshot(current))
            if current.status == 1:
                break
        assert records[-1]["status"] == 1 and abs(records[-1]["t"] - 0.025) < 1e-13
        return {
            "single_tstop": without_stop,
            "repeated_tstop": records,
            "persists_between_step_calls": False,
            "detail": "a previously shortened next step can still land near the old stop after it is cleared; a subsequent step passes it",
            "requirement": "resupply the active tstop on every step call; binding calls IDAClearStopTime",
        }

    def events() -> dict[str, Any]:
        def constant(t: float, y: Any, yp: Any, res: Any) -> None:
            res[:] = yp - 1.0

        def event(t: float, y: Any, yp: Any, values: Any) -> None:
            values[0] = t - 0.04

        setattr(event, "terminal", [False])
        setattr(event, "direction", [1])
        solver = sun.ida.IDA(
            constant, eventsfn=event, num_events=1, first_step=0.06, max_step=0.06
        )
        solver.init_step(0.0, [0.0], [1.0])
        root = solver.step(0.2, method="onestep")
        assert root.status == 2 and abs(root.t - 0.04) < 1e-12
        after = solver.step(0.05, method="normal")
        assert after.success and after.nfev == root.nfev
        return {
            "root": snapshot(root),
            "after_root_interpolation": snapshot(after),
            "nonterminal_event_still_stops_step": True,
            "limitation": "root return can lie within an already accepted internal step; it is not automatically a native AcceptedStep endpoint",
        }

    def ignored_return() -> dict[str, Any]:
        calls = 0

        def residual(t: float, y: Any, yp: Any, res: Any) -> int:
            nonlocal calls
            calls += 1
            res[:] = yp + y
            return 1

        result = sun.ida.IDA(residual, rtol=1e-8, atol=1e-10).solve(
            [0.0, 0.1], [1.0], [-1.0]
        )
        assert result.success and calls > 0
        assert abs(result.y[-1, 0] - np.exp(-0.1)) < 1e-6
        return {
            "returned_one_count": calls,
            "solver_success_despite_return_one": True,
            "native_recoverable_residual_status_exposed": False,
        }

    def callback_failure(cancel: bool = False) -> dict[str, Any]:
        class ProbeStopped(RuntimeError):
            pass

        cancelled = threading.Event()
        calls = 0

        def residual(t: float, y: Any, yp: Any, res: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 6:
                cancelled.set()
            if cancelled.is_set():
                raise ProbeStopped(
                    "cooperative cancellation" if cancel else "callback failure"
                )
            res[:] = yp + y

        try:
            result = sun.ida.IDA(residual).solve([0.0, 0.2], [1.0], [-1.0])
        except Exception as exc:
            assert "cooperative cancellation" in str(exc) or "callback failure" in str(
                exc
            )
            return {
                "calls": calls,
                "exception": f"{type(exc).__name__}: {exc}",
                "successful_result_published": False,
                "scope": "cooperative callback exception; no native cancel method or hard worker termination tested",
            }
        assert not result.success, (
            "callback failure was published as a successful solve"
        )
        return {
            "calls": calls,
            "result": snapshot(result),
            "successful_result_published": False,
        }

    def capacity() -> dict[str, Any]:
        def residual(t: float, y: Any, yp: Any, res: Any) -> None:
            res[:] = yp - 0.6

        result = sun.ida.IDA(
            residual, constraints_idx=[0, 1], constraints_type=[2, 2]
        ).solve([0.0, 0.5], [0.3, 0.3], [0.6, 0.6])
        end = result.y[-1]
        assert result.success and np.all(end > 0) and np.all(end < 1)
        assert float(sum(end)) > 1.1
        return {
            "endpoint": end.tolist(),
            "shared_occupancy_sum": float(sum(end)),
            "positivity_enforced_but_shared_capacity_violated": True,
            "shared_capacity_or_probability_simplex_exposed": False,
        }

    def option_limits() -> dict[str, Any]:
        rejected = {}
        for name, value in (
            ("rtol", [1e-6]),
            ("suppress_algebraic_error", True),
            ("error_weight_fn", lambda: None),
            ("nonlinear_convergence_coefficient", 0.1),
        ):
            try:
                sun.ida.IDA(decay, **{name: value})
            except (ValueError, TypeError) as exc:
                rejected[name] = f"{type(exc).__name__}: {exc}"
        assert len(rejected) == 4
        return {
            "rejected_options": rejected,
            "max_nonlin_iters_is_accuracy_control": False,
            "public_methods": [n for n in dir(sun.ida.IDA) if not n.startswith("_")],
        }

    with threadpool_limits(limits=1, user_api="blas"):
        check("initial_consistency", "limited", init_semantics)
        check("sparse_csc_jacobian", "available", sparse_csc)
        check("native_step_observation", "available", accepted_steps)
        check("dense_interpolation_bounds", "limited", interpolation)
        check("tstop_persistence", "not_persistent", stop_persistence)
        check("event_root_semantics", "limited", events)
        check(
            "residual_return_one",
            "unavailable_native_recoverable_status",
            ignored_return,
        )
        check("callback_failure", "fatal_exception_propagation", callback_failure)
        check("cooperative_cancellation", "limited", lambda: callback_failure(True))
        check("shared_site_capacity", "unavailable_as_sign_constraint", capacity)
        check("error_control_options", "limited", option_limits)
        pools_during_probes = threadpool_info()

    return {
        "schema": "solarlab.refactor.ida-binding-capabilities.v1",
        "python": sys.version,
        "executable": sys.executable,
        "binding_version": importlib.metadata.version("scikit-sundae"),
        "sundials_version": sun.SUNDIALS_VERSION,
        "binding_origin": str(package_root),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_sha256": sources,
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "documentation": "Installed 1.1.3 ida/_solver.py docstrings and _cy_ida.pyx; SUNDIALS 7.5.0",
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
        "observable_threadpools": pools_during_probes,
        "cases": cases,
        "probe_checks_passed": all(case["observation_matched"] for case in cases),
        "scientific_device_or_G2_qualification": False,
        "scope": "Tiny analytic systems only; unavailable features remain unavailable",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "probe_checks_passed": result["probe_checks_passed"],
                "cases": [
                    {k: c[k] for k in ("name", "capability", "observation_matched")}
                    for c in result["cases"]
                ],
                "scientific_device_or_G2_qualification": False,
            },
            indent=2,
        )
    )
    return 0 if result["probe_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

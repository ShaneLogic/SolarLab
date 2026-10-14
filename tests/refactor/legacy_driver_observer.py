"""Complete bounded old-driver call and read-only return observation.

No callee is replaced. A profile hook records real returned material/state
objects; the original public driver must return before complete is true.
"""
import argparse
import base64
from collections import Counter
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time
import traceback


def words(value):
    import numpy as np
    if is_dataclass(value):
        return {"type": type(value).__qualname__, "fields": {f.name: words(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not a physical byte payload")
        return {"dtype": value.dtype.str, "shape": list(value.shape), "raw_hex": value.tobytes(order="C").hex()}
    if isinstance(value, np.generic):
        return {"dtype": value.dtype.str, "raw_hex": value.tobytes().hex()}
    if type(value) is float:
        return {"binary64_be": struct.pack(">d", value).hex()}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is bytes:
        return {"bytes_b64": base64.b64encode(value).decode()}
    if isinstance(value, (tuple, list)):
        return [words(item) for item in value]
    if isinstance(value, dict):
        return {str(key): words(item) for key, item in value.items()}
    raise TypeError("unhandled returned payload: " + type(value).__qualname__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    supplied_context = None
    if args.context is not None:
        bundle = json.loads(args.context.read_bytes())
        supplied_context = bundle["document"]
        encoded_context = json.dumps(supplied_context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert hashlib.sha256(encoded_context).hexdigest() == bundle["sha256"]
        expected_schema = "solarlab.r1-execution-context.v1" if request["driver"] == "r1_step" else "solarlab.legacy-execution-context.v1"
        assert supplied_context["schema"] == expected_schema and supplied_context["driver"] == request["driver"]
        for key, path_key in (("input", "input_path"), ("reference", "reference_path"), ("prepared", "prepared_path")):
            if key not in supplied_context:
                continue
            snapshot = supplied_context[key]
            raw = base64.b64decode(snapshot["content_b64"], validate=True)
            assert hashlib.sha256(raw).hexdigest() == snapshot["sha256"]
            assert Path(request[path_key]).read_bytes() == raw
        request = {**request, "arguments": supplied_context["arguments"],
                   "environment": supplied_context["environment"], "argument_form": "driver_keywords"}
    for key, value in request["environment"].items():
        assert os.environ.get(key) == value, key
    import perovskite_sim
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.scaps_compat.loader import load_scaps_yaml
    from perovskite_sim.experiments import jv_sweep, steady_state, waveform_jv
    from perovskite_sim.solver import mol
    package = Path(perovskite_sim.__file__).resolve().parent
    if supplied_context is not None:
        bindings = (supplied_context["source_bindings"] if request["driver"] == "r1_step"
                    else supplied_context["behavior"]["sources"])
        for binding in bindings:
            path = (package / binding["id"]).resolve()
            assert path.is_relative_to(package)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    source = Path(request["input_path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == request["input_sha256"]
    stack = load_scaps_yaml(source) if request["driver"].startswith("scaps_") else load_device_from_yaml(source)
    if request.get("stack_overrides"):
        stack = replace(stack, **request["stack_overrides"])
    observed, errors = [], []
    targets = {
        ("solver/mol.py", "build_material_arrays"), ("solver/mol.py", "run_transient"),
        ("solver/newton.py", "solve_equilibrium"), ("experiments/jv_sweep.py", "build_electrical_grid"),
        ("experiments/jv_sweep.py", "_integrate_step"), ("experiments/steady_state.py", "solve_steady_state"),
        ("experiments/steady_state.py", "_transient_point_fallback"),
        ("experiments/one_dimensional_mechanism_r1.py", "build_r1_material"),
    }
    filenames = {str(package / name): name for name, _ in targets}

    def observe(frame, event, value):
        relative = filenames.get(frame.f_code.co_filename)
        if event != "return" or (relative, frame.f_code.co_name) not in targets:
            return
        try:
            if len(observed) >= 2000:
                raise ValueError("bounded observation record limit exceeded")
            row = {"source": relative, "function": frame.f_code.co_name, "returned": words(value)}
            if frame.f_code.co_name in ("solve_steady_state", "run_transient", "_integrate_step"):
                row["arguments"] = {name: words(frame.f_locals[name]) for name in
                    ("V_app", "t_span", "t_eval", "t_start", "t_end", "dt", "illuminated", "rtol", "atol", "y0")
                    if name in frame.f_locals and not callable(frame.f_locals[name])}
            observed.append(row)
        except Exception as error:
            errors.append({"source": relative, "function": frame.f_code.co_name, "error": str(error)})

    record = {"schema": "solarlab.complete-old-driver-observation.v1", "complete": False,
              "request": request, "stack": words(stack), "observed_returns": observed,
              "capture_errors": errors, "profile_does_not_replace_callees": True,
              "import_time_policy": {"ss_jacobian_reuse": steady_state._SS_JAC_REUSE,
                  "ss_gummel": steady_state._SS_GUMMEL, "rhs_finite_check": mol._RHS_FINITE_CHECK,
                  "QSS_velocity_effective_m_s": words(mol._QSS_V_TH_MS)}}
    record["observer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record["context_sha256"] = None if supplied_context is None else bundle["sha256"]
    record["driver_started_ns"] = time.monotonic_ns()
    sys.setprofile(observe)
    try:
        arguments = dict(request["arguments"])
        if request["driver"] == "scaps_transient":
            result = jv_sweep.run_jv_sweep(stack, **arguments)
        elif request["driver"] == "scaps_steady_state":
            result = steady_state.solve_voc_ss(stack, **arguments)
        elif request["driver"] == "ion_hysteresis":
            waveform = waveform_jv.JVWaveform.from_dict(arguments.pop("waveform"))
            if request.get("argument_form", "saved_request") == "saved_request":
                controls = arguments.pop("waveform_controls")
                for name in ("solver", "iface_states", "interface_boundary", "interface_transport_model"):
                    arguments.pop(name, None)
                arguments.update(rtol=controls["rtol"], atol=controls["atol_m3"])
            else:
                assert request["argument_form"] == "driver_keywords"
            result = waveform_jv.run_waveform_jv(stack, waveform, **arguments)
        elif request["driver"] == "r1_step":
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy, run_r1_step
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state, json_data
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import get_backend
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
            reference_path = Path(request["reference_path"])
            reference = json.loads(reference_path.read_bytes())
            policy = r1_policy(arguments.pop("nonlinear_factor"), time_substeps=tuple(arguments.pop("time_substeps")))
            if supplied_context is not None:
                expected_policy = json.loads(base64.b64decode(supplied_context["policy"]["content_b64"], validate=True))
                assert json_data(policy) == expected_policy
                controls = arguments.pop("controls")
                assert json_data(R1DynamicsControls.from_label(arguments["control"])) == controls
            intervals = arguments.pop("intervals")
            backend = arguments.get("backend", "legacy")
            if request.get("prepared_path"):
                prepared = get_backend(backend).decode_prepared(json.loads(Path(request["prepared_path"]).read_bytes()))
            else:
                prepared = prepare_common_state(stack, intervals, reference, policy=policy, backend=backend)
                args.output.with_name("PreparedState.json").write_text(prepared.canonical_json + "\n")
            record["prepared"] = prepared.to_dict()
            if supplied_context is not None:
                assert prepared.sha256 == supplied_context["prepared_sha256"]
            result = run_r1_step(stack, intervals, reference, prepared, policy=policy,
                                 expected_prepared_sha256=prepared.sha256, **arguments)
        else:
            raise ValueError("unknown exact old driver")
        record["result"] = words(result)
        if errors:
            raise ValueError("a required physical return could not be captured")
        record["complete"] = True
    except Exception as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if hasattr(error, "result"):
            record["failed_driver_result"] = words(error.result)
    finally:
        sys.setprofile(None)
        record["driver_finished_ns"] = time.monotonic_ns()
    record["return_counts"] = dict(Counter(row["source"] + ":" + row["function"] for row in observed))
    record["source_sha256"] = {str(Path(module.__file__).resolve().relative_to(package)): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in sorted(sys.modules.items()) if name.startswith("perovskite_sim") and getattr(module, "__file__", None)
        and Path(module.__file__).resolve().is_relative_to(package)}
    record["elapsed_seconds"] = time.monotonic() - started
    record["started_ns"] = started_ns
    record["finished_ns"] = time.monotonic_ns()
    record["pid"] = os.getpid()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    assert record["rss_bytes"] <= 4 * 1024**3
    encoded = json.dumps(record, indent=2, allow_nan=False).encode()
    assert len(encoded) <= 64 * 1024**2
    args.output.write_bytes(encoded + b"\n")
    print(json.dumps({key: record[key] for key in ("complete", "elapsed_seconds", "rss_bytes", "return_counts", "capture_errors")}, indent=2))
    if not record["complete"]:
        print(json.dumps(record["failure"], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

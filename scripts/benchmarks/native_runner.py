"""Maintained recording boundary and source-bound native runner.

Importing this module never imports a numerical backend. ``run_recorded`` is the
same recording boundary used by ``main`` and accepts an explicit controller call
for infrastructure tests. Native admission, imports and device construction occur
only in ``main``; the next immutable shim must bind this file and native_history.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import ctypes
import hashlib
import importlib.abc
import json
import os
import resource
import sys
import time
import traceback
from typing import Any, Callable, Collection

from scripts.benchmarks.native_history import HistoryLimitError, HistoryWriter

COMPLETED_STATUS = "completed_bounded_voltage_lift_native_pilot"

def output_bytes(folder, input_names):
    """Sample files once; on disappearance, reobserve once and keep the larger sum.

    Preserve stat-following file-type and relative input-exclusion behavior.
    Other I/O errors propagate; this is not an atomic or continuous census.
    """
    import stat

    def observe():
        total, changed = 0, False
        for path in folder.rglob("*"):
            try:
                info = path.stat()
            except FileNotFoundError:
                changed = True
                continue
            if stat.S_ISREG(info.st_mode) and str(path.relative_to(folder)) not in input_names:
                total += info.st_size
        return total, changed

    total, changed = observe()
    if changed:
        repeated, _ = observe()
        return max(total, repeated)
    return total

def _write_document(folder: Path, name: str, value: Any, *, total_output_bytes: int,
                    input_names: Collection[str]) -> None:
    """Retained exclusive publication, with the unchanged whole-artifact cap."""
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if output_bytes(folder, input_names) + len(data) > total_output_bytes:
        raise HistoryLimitError("native artifact publication exceeds total output cap")
    with (folder / name).open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError("short native artifact write")
        stream.flush()


def _history_state(writer: HistoryWriter) -> dict[str, Any]:
    return {"encoding": writer.encoding, "path": writer.path.name,
            "logical_bytes": writer.logical_bytes, "encoded_bytes": writer.encoded_bytes,
            "logical_sha256": writer.logical_sha256, "records": writer.records,
            "closed": writer.closed, "container_complete": writer.container_complete,
            "error": writer.error, "digest_scope": "complete record bytes written before any failed partial write"}


def _record_exception(folder: Path, error: BaseException, *, total_output_bytes: int,
                      input_names: Collection[str], writer: HistoryWriter | None = None) -> None:
    """Reporting failures become notes; they never replace the primary exception."""
    failure = {"exception": type(error).__name__, "reason": str(error),
               "traceback": "".join(traceback.format_exception(error))}
    if writer is not None:
        failure["history"] = _history_state(writer)
        try:
            writer.publish_first_failure(failure)
        except BaseException as secondary:
            error.add_note(f"first-failure publication failed: {type(secondary).__name__}: {secondary}")
    if not (folder / "RunnerFailure.json").exists():
        try:
            _write_document(folder, "RunnerFailure.json", failure,
                            total_output_bytes=total_output_bytes, input_names=input_names)
        except BaseException as secondary:
            error.add_note(f"runner-failure publication failed: {type(secondary).__name__}: {secondary}")


def run_recorded(folder: Path, controller: Callable[[HistoryWriter], dict[str, Any]], *,
                 total_output_bytes: int, input_names: Collection[str]) -> dict[str, Any]:
    """Run the real controller with the writer itself, finalize, then publish.

    An already failed controller may retain an explicitly incomplete container;
    no successful result is published after a write/finalization/publication
    failure. Inputs are explicitly excluded from the same outer artifact census.
    """
    folder = Path(folder)
    writer = None
    try:
        if any((folder / name).exists() for name in ("NativeHistory.jsonl", "NativeHistory.jsonl.gz", "NativeResult.json", "RunnerFailure.json")):
            raise FileExistsError("native attempt output already exists")
        writer = HistoryWriter(folder / "NativeHistory.jsonl.gz", encoding="gzip",
                               total_output_bytes=total_output_bytes)
        result = controller(writer)  # Do not wrap this object in an untyped callback.
        if result.get("first_failure") is not None:
            writer.publish_first_failure(result["first_failure"])
        successful = result.get("status") == COMPLETED_STATUS
        if writer.first_exception is not None:
            if successful:
                raise writer.first_exception
        else:
            writer.finish()
        # Keep all original controller fields and logical counters. Encoded bytes
        # include the footer, which the controller's last record cannot count.
        result = dict(result)
        if "counts" in result:
            counts = dict(result["counts"])
            if writer.first_exception is None and counts["history_bytes"] != writer.logical_bytes:
                raise RuntimeError("controller/history logical count mismatch")
            counts["history_encoded_bytes"] = writer.encoded_bytes
            result["counts"] = counts
        result["history"] = _history_state(writer)
        if successful and (not writer.closed or writer.container_complete is not True):
            raise RuntimeError("successful native result requires finalized history")
        _write_document(folder, "NativeResult.json", result,
                        total_output_bytes=total_output_bytes, input_names=input_names)
        return result
    except BaseException as error:
        if writer is not None:
            try:
                writer.abort(error)
            except BaseException as secondary:
                error.add_note(f"history abort failed: {type(secondary).__name__}: {secondary}")
        _record_exception(folder, error, total_output_bytes=total_output_bytes,
                          input_names=input_names, writer=writer)
        raise



def main(folder: Path, admission_path: Path, *, entry_started: float | None = None) -> int:
    """Original source/admission/native boundary; called only by an admitted shim."""
    from AdmissionBindings import hashed, validate


    folder=Path(folder)
    freeze=json.loads((folder/"NativeFreeze.json").read_text())
    admission=json.loads(Path(admission_path).read_text())
    def canonical(value):
     return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    def write(name,value):
     _write_document(folder, name, value, total_output_bytes=freeze["resources"]["total_output_bytes"],
                     input_names=freeze["watchdog"]["input_file_names"])
    if (not str(admission.get("task_id", "")).startswith("task_") or
        not str(admission.get("dispatch_id", "")).startswith("ctx_") or
        admission.get("session_id")!=freeze["session_id"] or
        admission.get("voltage_lift_native_authorized") is not True or
        admission.get("map_identity")!=freeze["map_identity"] or
        admission.get("request_sha256")!=freeze["request_sha256"] or
        admission.get("freeze_sha256")!=hashed(folder/"NativeFreeze.json") or
        not str(admission.get("coordinator_message","")).startswith("msg_")):
     raise RuntimeError("exact_native_admission_required_before_execution")
    validate(freeze,folder/"NativeFreeze.json",admission)
    started=entry_started if entry_started is not None else time.perf_counter()
    write("RuntimeStart.json",{"utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),"pgid":os.getpgrp(),"executable":sys.executable,
                              "actual_process_argv":sys.orig_argv,"declared_command":freeze["runner_command"],
                              "task_id":admission["task_id"],"dispatch_id":admission["dispatch_id"],
                              "session_id":admission["session_id"],
                              "source_task_id":freeze["task_id"],"source_dispatch_id":freeze["dispatch_id"],
                              "request_sha256":freeze["request_sha256"],"admission":admission})
    code=1;imports={};primary_error=None
    try:
     for path,expected in {**freeze["source_sha256"],**freeze["external_source_sha256"]}.items():
      if hashed(path)!=expected:raise RuntimeError("frozen_source_changed:"+path)
     if admission["source_sha256"]!=freeze["source_sha256"]:raise RuntimeError("admission_source_binding_mismatch")
     for required in (Path(__file__).resolve(), Path(sys.modules[HistoryWriter.__module__].__file__).resolve()):
      if (not required.is_relative_to(Path(freeze["isolated_root"])) or
          freeze["source_sha256"].get(str(required)) != hashed(required)):
       raise RuntimeError("maintained_recording_source_not_bound:"+str(required))
     if any(os.environ.get(k)!="1" for k in freeze["threads"]):raise RuntimeError("thread_environment_changed")
     library="/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib"
     blas=ctypes.CDLL(library);blas.BLASSetThreading.argtypes=[ctypes.c_int];blas.BLASSetThreading.restype=ctypes.c_int;blas.BLASGetThreading.restype=ctypes.c_int
     previous=blas.BLASSetThreading(1);mode=blas.BLASGetThreading()
     if mode!=1:raise RuntimeError("BLAS_thread_mode")
     class ProductionBoundary(importlib.abc.MetaPathFinder):
      def find_spec(self,fullname,path=None,target=None):
       if fullname.startswith("perovskite_sim.") and fullname not in {"perovskite_sim.constants","perovskite_sim.physics","perovskite_sim.physics.compensated"}:
        raise RuntimeError("unapproved_production_import:"+fullname)
     sys.meta_path.insert(0,ProductionBoundary())
     import numpy,scipy,sksundae,flint
     from sksundae.ida import IDA as PublicIDA
     public_api={name:callable(getattr(PublicIDA,name,None)) for name in ("statistics","last_step_snapshot")}
     if not all(public_api.values()):raise RuntimeError("composed_public_IDA_capabilities_missing")
     from threadpoolctl import threadpool_info
     from scripts.benchmarks.coupled_device_prototype import AffineCoupledSlab,AffineVoltageMap,SlabDefinition,protocol_from_plan,run_voltage_lift_native_pilot
     from scripts.benchmarks import coupled_device_prototype as kernel
     capabilities=freeze["binding_capabilities"]
     if (numpy.__version__,scipy.__version__,sksundae.__version__)!=(capabilities["numpy_version"],capabilities["scipy_version"],capabilities["binding_version"]):
      raise RuntimeError("installed_numeric_versions_changed")
     pools=threadpool_info()
     if any(p.get("num_threads",1)!=1 for p in pools):raise RuntimeError("native_threadpool_not_one")
     for name,module in list(sys.modules.items()):
      if name.startswith(("perovskite_sim","scripts.benchmarks","sksundae","flint")) and getattr(module,"__file__",None):
       path=str(Path(module.__file__).resolve());actual=hashed(path)
       if name.startswith(("perovskite_sim","scripts.benchmarks")) and not Path(path).is_relative_to(Path(freeze["isolated_root"])):
        raise RuntimeError("actual_local_import_outside_frozen_code:"+name)
       if name.startswith("sksundae") and not Path(path).is_relative_to(Path(freeze["binding_installation"])):
        raise RuntimeError("actual_binding_outside_composed_installation:"+name)
       expected=freeze["source_sha256"].get(path,freeze["external_source_sha256"].get(path))
       if actual!=expected:raise RuntimeError("actual_import_not_bound:"+name+":"+path)
       imports[name]={"path":path,"sha256":actual}
     write("EnvironmentObserved.json",{"python":sys.version,"executable":sys.executable,"numpy":numpy.__version__,
                                     "scipy":scipy.__version__,"binding":sksundae.__version__,"SUNDIALS":sksundae.SUNDIALS_VERSION,"python_flint":flint.__version__,"FLINT":flint.__FLINT_VERSION__,
                                     "thread_environment":{k:os.environ.get(k) for k in freeze["threads"]},
                                     "threadpools":pools,"Accelerate":{"observed":mode,"previous":previous},"actual_imports":imports,"public_IDA_API_before_construction":public_api})
     source=Path(freeze["model_input_root"]);plan=Path(freeze["isolated_plan"])
     request=json.loads((folder/"NativeRequest.json").read_text())
     if canonical(request)!=freeze["request_sha256"]:raise RuntimeError("native_request_changed")
     model=AffineCoupledSlab(SlabDefinition.from_plan(plan,source,request["case_id"]),8)
     segments=protocol_from_plan(plan,request["case_id"])
     if flint.__version__!="0.8.0":raise RuntimeError("unreviewed_Arb_binding")
     if request["budgets"]["total_output_bytes"] != freeze["resources"]["total_output_bytes"]:
      raise RuntimeError("controller_and_outer_output_cap_mismatch")
     result=run_recorded(folder,
                         lambda emit: run_voltage_lift_native_pilot(AffineVoltageMap(model),segments,request,admission,emit),
                         total_output_bytes=request["budgets"]["total_output_bytes"],
                         input_names=freeze["watchdog"]["input_file_names"])
     code=0 if result["status"]=="completed_bounded_voltage_lift_native_pilot" else 1
    except BaseException as error:
     primary_error=error
     _record_exception(folder,error,total_output_bytes=freeze["resources"]["total_output_bytes"],
                       input_names=freeze["watchdog"]["input_file_names"])
     code=1
    finally:
     try:
      drift={p:hashed(p) if Path(p).exists() else "missing"
             for p,v in {**freeze["source_sha256"],**freeze["external_source_sha256"]}.items()
             if not Path(p).exists() or hashed(p)!=v}
      if drift:code=1
      write("RuntimeEnd.json",{"utc":datetime.now(timezone.utc).isoformat(),"elapsed_s":time.perf_counter()-started,
                              "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              "rss_units":"Darwin ru_maxrss bytes","source_drift":drift,"exitcode":code,
                              "elapsed_scope":"runner module entry through source recheck; outer supervisor includes RuntimeEnd save and process exit"})
     except BaseException as end_error:
      code=1
      if primary_error is None:
       primary_error=end_error
      else:
       primary_error.add_note(f"runtime finalization failed: {type(end_error).__name__}: {end_error}")
      _record_exception(folder,primary_error,total_output_bytes=freeze["resources"]["total_output_bytes"],
                        input_names=freeze["watchdog"]["input_file_names"])
    return code



if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: native_runner CASE_FOLDER ADMISSION_JSON")
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))

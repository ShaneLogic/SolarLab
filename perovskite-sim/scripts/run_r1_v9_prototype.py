"""Run the frozen V9 production-backend validation window and retain all costs.

The physical request and original gates remain unchanged. This entry exercises
the production backend and codec; a run alone cannot grant complete P2 or R1-2.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from scripts.run_r1_v8_prototype import (
    PROJECT, THREAD_KEYS, INPUTS, LIMITS, ENGINEERING_LIMITS, CASE_AXES,
    TIMES_SHA256, canonical, sha, validate_digest, write, error_record,
    source_snapshot, extent, certificate_check, precision_record_check,
    peak_rss_bytes,
)
from scripts.r1_v8_timing import ExclusiveTimer

SCHEMA = "R1V9ProductionValidationContractV1"
SCOPE = "bounded_P2_backend_and_pair_evidence_migration_no_global_R1_2_qualification"
ABSOLUTE_LIMITS = {
    "elapsed_s": 340.436297710, "peak_rss_bytes": 956235776,
    "accepted_row_payload_bytes": 86417556, "case_artifact_bytes": 253170495,
}


def load_contract(path, digest, output):
    path, output = Path(path).resolve(), Path(output).resolve()
    validate_digest(digest, "request_sha256")
    if path.is_relative_to(output) or sha(path) != digest:
        raise ValueError("V9 requires the exact external request outside the result directory")
    raw = path.read_bytes()
    contract = json.loads(raw)
    if contract.get("schema") != SCHEMA or contract.get("scope") != SCOPE:
        raise ValueError("not a V9 production validation contract")
    case = contract.get("case", {})
    if set(case) != set(CASE_AXES) | {"times_s"} or any(case.get(k) != v for k,v in CASE_AXES.items()):
        raise ValueError("V9 physical case differs from the approved N16 window")
    times = case["times_s"]
    if (not isinstance(times, list) or len(times) != 114
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in times)
            or hashlib.sha256(canonical(times).encode()).hexdigest() != TIMES_SHA256):
        raise ValueError("V9 requires the unchanged 114 actual time values")
    for key, expected in (("inputs", INPUTS), ("certificate_limits", LIMITS),
                          ("engineering_limits", ENGINEERING_LIMITS),
                          ("absolute_engineering_limits", ABSOLUTE_LIMITS)):
        if contract.get(key) != expected:
            raise ValueError("changed V9 input or budget: " + key)
    execution = contract.get("execution", {})
    required = {"baseline_repeats": 3, "compensated_repeats": 1,
        "paired_baseline_ordinal": 3, "time_comparator": "minimum_elapsed_of_all_three_complete_same_source_baselines",
        "production_backend_explicit": True, "production_monkeypatch_forbidden": True,
        "expected_total_rows": 794, "expected_rows_by_substeps": {"1":114,"2":227,"4":453},
        "single_thread": True, "requires_clean_full_commit": True, "resume_from_old_state": False,
        "new_profile_runs": 0, "new_ablation_runs": 0, "new_100s_runs": 0}
    if any(execution.get(key) != value for key,value in required.items()):
        raise ValueError("V9 execution extent or work budget changed")
    if any(contract.get(key) is not False for key in ("scientifically_accepted","formal_qualification","P2_qualified")):
        raise ValueError("a V9 request cannot grant scientific qualification")
    return contract, raw


def external_json(path, expected, name):
    validate_digest(expected, name + "_sha256")
    if sha(path) != expected:
        raise ValueError("external " + name + " identity mismatch")
    return json.loads(Path(path).read_text())


def tag_nonfinite(value):
    if isinstance(value,dict):
        return {key:tag_nonfinite(child) for key,child in value.items()}
    if isinstance(value,(list,tuple)):
        return [tag_nonfinite(child) for child in value]
    if isinstance(value,float) and not math.isfinite(value):
        return {"nonfinite":repr(value)}
    return value


def function_snapshot():
    """Existing module function identities, never serialized as source proof."""
    import inspect
    return {(name,key):value for name,module in tuple(sys.modules.items())
        if name.startswith("perovskite_sim.experiments") and module is not None
        for key,value in vars(module).items() if inspect.isfunction(value)}


def check_functions(before):
    changed = [name + "." + key for (name,key),value in before.items()
               if getattr(sys.modules.get(name),key,None) is not value]
    return {"existing_function_count":len(before), "changed":sorted(changed),
            "unchanged":not changed, "scope":"identity_of_existing_module_functions_during_explicit_backend_run"}


def baseline_readiness(record):
    """A full float reference may retain only its known final operator failure."""
    reason=[]
    if not record.get("extent",{}).get("complete"):
        reason.append("incomplete_extent")
    for key in ("source_unchanged","numeric_sidecar_exact","replay_completed"):
        if record.get(key) is not True:
            reason.append(key)
    if record.get("module_function_identity",{}).get("unchanged") is not True:
        reason.append("module_function_identity")
    if record.get("four_predicates",{}).get("observer_matches_result") is not True:
        reason.append("observer_mismatch")
    if record.get("certificate_check",{}).get("limits_match_contract") is not True:
        reason.append("changed_original_limits")
    known_failure=(record.get("execution_status")=="failed"
        and record.get("original_metric_failures")==["eliminated_operator_error"]
        and record.get("failure")=={"type":"R1RunError",
            "message":"R1 controlled-step certificate failed: eliminated_operator_error"})
    healthy=(record.get("execution_status")=="completed" and record.get("failure") is None
        and record.get("four_predicates_passed") is True)
    if not (known_failure or healthy) or record.get("runner_error") or record.get("replay_error"):
        reason.append("unexpected_execution_failure")
    return {"ready":not reason,"reasons":reason,"known_float_operator_failure_retained":known_failure}


def comparison(current, baselines, request_digest):
    reasons, ratios = [], {}
    if len(baselines) != 3:
        return {"qualified":False,"status":"requires_three_complete_baselines","reasons":["baseline_count"],"ratios":{}}
    for i,baseline in enumerate(baselines,1):
        if (baseline.get("schema") != "R1V9ProductionValidationRunV1"
                or baseline.get("mode") != "baseline" or baseline.get("baseline_ordinal") != i
                or baseline.get("request_sha256") != request_digest
                or not baseline_readiness(baseline)["ready"]):
            reasons.append("incomplete_or_wrong_baseline_" + str(i))
        if any(baseline.get(k) != current.get(k) for k in ("source_commit","source_content_sha256","runtime_identity")):
            reasons.append("baseline_identity_differs_" + str(i))
    if not current.get("extent",{}).get("complete"):
        reasons.append("current_run_incomplete")
    for key,metric in (("elapsed_ratio","elapsed_s"),("peak_rss_ratio","peak_rss_bytes"),
                       ("bytes_per_row_ratio","bytes_per_row")):
        value = current.get("cost",{}).get(metric)
        refs = [b.get("cost",{}).get(metric) for b in baselines]
        if any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0 for x in [value,*refs]):
            reasons.append("unavailable_" + metric)
            continue
        ratios[key] = value/min(refs)
        if ratios[key] > ENGINEERING_LIMITS[key]:
            reasons.append("exceeded_" + key)
    return {"qualified":not reasons,"status":"within_limits" if not reasons else "not_qualified",
        "ratios":ratios,"limits":ENGINEERING_LIMITS,"reasons":reasons,
        "paired_baseline_ordinal":3,
        "baseline_elapsed_values_s":[b.get("cost",{}).get("elapsed_s") for b in baselines],
        "comparison":"all_three_baselines_minimum_is_conservative_not_a_population_confidence_bound"}


def absolute_cost_check(cost):
    reasons=[]
    for name,limit in ABSOLUTE_LIMITS.items():
        value=cost.get(name)
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0 or value > limit:
            reasons.append(name)
    return {"passed":not reasons,"limits":ABSOLUTE_LIMITS,"failed_metrics":reasons}


def run_trajectory(directory, case, mode, api, source_guard):
    """The original work plus mandatory sidecar writes and actual row telemetry."""
    directory=Path(directory)
    timer=ExclusiveTimer()
    rows, timing_rows, replay_rows = [], [], []
    result, prepared, replay, failure = {}, None, None, None
    report={"execution_status":"not_started","physical_execution_started":False}
    started=time.monotonic()
    last_observation=started

    def timed(name,call):
        return timer.call(name,call)

    with ExitStack() as stack:
        stream=stack.enter_context((directory/"AcceptedStepsV1.jsonl").open("x"))
        telemetry=stack.enter_context((directory/"RowTelemetryV1.jsonl").open("x"))
        ledger=stack.enter_context((directory/"ReplayRowsV1.jsonl").open("x"))

        def observe(row):
            nonlocal last_observation
            tick=time.monotonic()
            interval=tick-last_observation
            with timer.phase("observer_io_and_bookkeeping"):
                value=timed("data_conversion",lambda:api["json_data"](row))
                raw=timed("observer_serialization",lambda:canonical(value)+"\n")
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(value)
                item={"row":len(rows)-1,"substeps":value["substeps"],"time_s":value["time_s"],
                    "phase":value.get("phase"),"time_to_observation_s":interval,
                    "observer_conversion_and_persistence_s":time.monotonic()-tick,
                    "peak_rss_bytes":peak_rss_bytes(),"row_bytes":len(raw.encode()),
                    "scope":"event_interval_includes_initialization_or_tier_transition_when_applicable"}
                timing_rows.append(item)
                telemetry.write(canonical(item)+"\n")
                telemetry.flush()
            last_observation=time.monotonic()

        def observe_replay(item):
            with timer.phase("replay_observer_io"):
                saved=api["json_data"](item)
                replay_rows.append(saved)
                ledger.write(canonical(saved)+"\n")
                ledger.flush()

        try:
            timed("source_guard",source_guard)
            report.update(execution_status="preparing",physical_execution_started=True)
            prepared=timed("prepare_compute",api["prepare"])
            timed("preparation_write",lambda:write(directory/"PreparedV1.json",api["json_data"](prepared.to_dict())))
            report["execution_status"]="integrating"
            try:
                result=timed("integrate_compute_and_checks",lambda:api["run"](prepared,observe))
                report["execution_status"]="completed"
            except (Exception,KeyboardInterrupt) as exc:
                failure=error_record(exc)
                result=getattr(exc,"result",{})
                report["execution_status"]="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
            raw_result=result
            result=timed("data_conversion",lambda:api["json_data"](result)) if isinstance(result,dict) else {}
            timed("result_write",lambda:write(directory/"ResultV1.json",result))
            if result:
                timed("numeric_sidecar_write",lambda:api["persist_numeric"](directory/"StateArraysV1.npz",raw_result))
                timed("numeric_sidecar_readback",lambda:api["verify_numeric"](directory/"StateArraysV1.npz",result))
                report["numeric_sidecar_exact"]=True
            if prepared is not None and result:
                try:
                    replay=timed("replay_compute",lambda:api["replay"](
                        prepared,result,report["execution_status"]!="completed",observe_replay))
                    report["replay_completed"]=True
                    timed("replay_write",lambda:write(directory/"PhysicsReplayV1.json",api["json_data"](replay)))
                    receipt=getattr(replay,"replay_receipt",None)
                    if receipt is not None:
                        timed("replay_write",lambda:write(directory/"ReplayLedgerV1.json",receipt.to_dict()))
                except Exception as exc:
                    report["replay_error"]=error_record(exc)
            if failure is not None and prepared is not None and result and "failure_witness" in api:
                try:
                    witness=timed("failure_witness",lambda:api["failure_witness"](prepared,result,rows,failure))
                    timed("failure_write",lambda:write(directory/"FailureWitnessV1.json",api["json_data"](witness)))
                except Exception as exc:
                    report["failure_witness_error"]=error_record(exc)
            timed("source_guard",source_guard)
        except (Exception,KeyboardInterrupt) as exc:
            failure=error_record(exc)
            if getattr(exc,"result",None) is not None:
                timed("failure_write",lambda:write(directory/"PhaseFailureResultV1.json",api["json_data"](exc.result)))
            report["execution_status"]="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
        finally:
            if failure is not None:
                timed("failure_write",lambda:write(directory/"FailureV1.json",failure))
            for output in (stream,telemetry,ledger):
                output.flush()
                os.fsync(output.fileno())
            persisted=timed("readback",lambda:[json.loads(line) for line in
                (directory/"AcceptedStepsV1.jsonl").read_text().splitlines()])
            report["extent"]=timed("final_validation",lambda:extent(persisted,case))
            report["certificate_check"]=timed("final_validation",lambda:certificate_check(result))
            metrics=result.get("certificate",{}).get("metrics",{})
            report["original_metric_failures"]=[name for name,limit in LIMITS.items()
                if (type(metrics.get(name)) not in (int,float) or not math.isfinite(metrics[name])
                    or metrics[name]<0 or metrics[name]>limit)]
            report["precision_records"]=timed("final_validation",lambda:precision_record_check(persisted,mode))
            report["four_predicates"]={
                "run_completed":report["execution_status"]=="completed" and failure is None,
                "certificate_certified":report["certificate_check"]["passed"],
                "replay_certified":isinstance(replay,dict) and replay.get("certified") is True,
                "observer_matches_result":persisted==result.get("accepted_steps",[]) and persisted==rows}
            report["four_predicates_passed"]=all(report["four_predicates"].values()) and report["extent"]["complete"]
            npz=(directory/"StateArraysV1.npz").stat().st_size if (directory/"StateArraysV1.npz").exists() else 0
            row_bytes=(directory/"AcceptedStepsV1.jsonl").stat().st_size
            row_sidecars={name:(directory/name).stat().st_size for name in
                ("RowTelemetryV1.jsonl","ReplayRowsV1.jsonl","ReplayLedgerV1.json") if (directory/name).is_file()}
            # Mandatory repeated state words and row-associated telemetry or
            # replay receipts all count; the full case separately counts JSON copies.
            payload=row_bytes+npz+sum(row_sidecars.values())
            elapsed=time.monotonic()-started
            report["cost"]={"elapsed_s":elapsed,"peak_rss_bytes":peak_rss_bytes(),
                "accepted_rows_bytes":row_bytes,"numeric_sidecar_bytes":npz,
                "row_associated_sidecars_bytes":row_sidecars,
                "accepted_row_payload_bytes":payload,
                "bytes_per_row":payload/len(persisted) if persisted else None}
            report["phase_timing"]=timer.report(elapsed)
            report["replay_observed_rows"]=len(replay_rows)
            report["failure"]=None if failure is None else {k:failure[k] for k in ("type","message")}
    if "after_main" in api and prepared is not None and result:
        analysis_start=time.monotonic()
        try:
            report["independent_analysis"]=api["after_main"](prepared,result,replay)
        except Exception as exc:
            report["independent_analysis"]={"passed":False,"error":error_record(exc)}
        report["independent_analysis_elapsed_s"]=time.monotonic()-analysis_start
        report["independent_analysis_cost_scope"]="additional_saved_state_analysis_outside_original_main_timer"
    return report


def finalize_manifest(directory, summary):
    """Count every case file, including the two metadata files themselves."""
    summary_path=directory/"SummaryV1.json"
    manifest_path=directory/"ManifestV1.json"
    for _ in range(8):
        write(summary_path,summary)
        files={p.relative_to(directory).as_posix():{"sha256":sha(p),"bytes":p.stat().st_size}
               for p in sorted(directory.rglob("*")) if p.is_file() and p!=manifest_path}
        write(manifest_path,files)
        total=sum(v["bytes"] for v in files.values())+manifest_path.stat().st_size
        if summary["cost"].get("case_artifact_bytes")==total:
            return
        summary["cost"]["case_artifact_bytes"]=total
        summary["absolute_engineering_check"]=absolute_cost_check(summary["cost"])
    raise ValueError("case artifact accounting did not stabilize")


def run(args):
    if not (sys.flags.isolated and sys.flags.no_site):
        raise ValueError("V9 complete validation requires Python -I -S")
    if any(os.environ.get(k)!="1" for k in THREAD_KEYS):
        raise ValueError("all declared numerical thread limits must be 1")
    output=args.output.resolve()
    contract,raw=load_contract(args.request_file,args.request_sha256,output)
    budget=external_json(args.budget_file,args.budget_sha256,"budget")
    validation=external_json(args.validation_contract,args.validation_sha256,"validation_contract")
    from scripts.analyze_r1_v8_prototype import validate_budget
    validate_budget(budget)
    if (budget.get("prototype_request_sha256")!=args.request_sha256
            or contract["validation_contract_sha256"]!=args.validation_sha256
            or budget.get("validation_contract_sha256")!=args.validation_sha256):
        raise ValueError("request/budget/validation identities do not agree")
    if args.mode=="baseline" and args.baseline_ordinal not in (1,2,3):
        raise ValueError("baseline ordinal must be the frozen 1, 2 or 3")
    if args.mode=="compensated" and args.baseline_ordinal is not None:
        raise ValueError("pair run cannot masquerade as a baseline ordinal")
    baselines=[]
    if len(args.baseline_summary)!=len(args.baseline_sha256):
        raise ValueError("every baseline summary needs an external digest")
    for path,expected in zip(args.baseline_summary,args.baseline_sha256):
        baselines.append(external_json(path,expected,"baseline"))
    if args.mode=="compensated":
        if len(baselines)!=3 or any(not baseline_readiness(item)["ready"] for item in baselines):
            raise ValueError("pair execution requires all three complete verified float references")
        for index,item in enumerate(baselines,1):
            if (item.get("mode")!="baseline" or item.get("baseline_ordinal")!=index
                    or item.get("source_commit")!=args.expected_commit
                    or item.get("source_content_sha256")!=args.source_sha256
                    or item.get("request_sha256")!=args.request_sha256):
                raise ValueError("baseline source, ordinal or request differs before pair execution")
    elif baselines:
        raise ValueError("baseline runs do not consume other baseline summaries")
    initial_source=source_snapshot(args.expected_commit)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
    context=require_r1_checkout(project=PROJECT,formal=True,source_commit=args.expected_commit,
                                expected_source_sha256=args.source_sha256)
    inputs={name:context.read_bytes(spec["path"]) for name,spec in INPUTS.items()}
    if any(hashlib.sha256(inputs[name]).hexdigest()!=INPUTS[name]["sha256"] for name in inputs):
        raise ValueError("frozen physical inputs changed")
    output.mkdir(parents=True,exist_ok=False)
    (output/"RequestV1.json").write_bytes(raw)
    (output/"SourceFixtureV1.yaml").write_bytes(inputs["fixture"])
    (output/"ReferenceBindingV1.json").write_bytes(inputs["reference"])
    write(output/"SourceReceiptV1.json",{**initial_source,
        "r1_source_content_sha256":args.source_sha256,"runner_sha256":sha(__file__)})
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_limits,threadpool_info
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import get_backend
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as physics
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_pair_codec as codec
    from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
    from scripts.verify_r1_v7_precision import freeze_healthy_material
    backend=get_backend("pair" if args.mode=="compensated" else "float64")
    stack=load_device_from_yaml(output/"SourceFixtureV1.yaml")
    binding=json.loads(inputs["reference"])
    case=contract["case"]
    grid,material=build_r1_material(stack,case["intervals"])
    frozen_material=freeze_healthy_material(grid,material,source_identity=args.source_sha256)
    write(output/"FrozenHealthyMaterialV1.json",frozen_material)
    functions_before=function_snapshot()
    summary={"schema":"R1V9ProductionValidationRunV1","scope":SCOPE,"mode":args.mode,
        "baseline_ordinal":args.baseline_ordinal,"source_commit":args.expected_commit,
        "source_content_sha256":args.source_sha256,"request_sha256":args.request_sha256,
        "precision_budget_sha256":args.budget_sha256,"validation_contract_sha256":args.validation_sha256,
        "backend_representation":backend.representation_id,
        "healthy_material_frozen_before_precision_context":True,
        "started_utc":datetime.now(timezone.utc).isoformat(),
        "P2_qualified":False,"formal_qualification":False,"scientifically_accepted":False}
    summary["execution_source_scope"]="clean_commit_bound_by_outer_receipt; internal_API_records_keep_development_identity"
    summary["baseline_inputs"]=[]
    for index,(path,expected) in enumerate(zip(args.baseline_summary,args.baseline_sha256),1):
        name="InputBaselineSummary"+str(index)+"V1.json"
        (output/name).write_bytes(Path(path).read_bytes())
        summary["baseline_inputs"].append({"ordinal":index,"filename":name,"sha256":expected})
    write(output/"SummaryV1.json",summary)
    def guard():
        if source_snapshot(args.expected_commit)!=initial_source:
            raise ValueError("V9 source changed during execution")
        for path,expected in ((args.request_file,args.request_sha256),(args.budget_file,args.budget_sha256),
                              (args.validation_contract,args.validation_sha256)):
            if sha(path)!=expected:
                raise ValueError("V9 external contract changed")
        require_r1_checkout(project=PROJECT,formal=True,source_commit=args.expected_commit,
                            expected_source_sha256=args.source_sha256)
    try:
        with threadpool_limits(1):
            np.dot(np.ones((2,2)),np.ones((2,2)))
            pools=threadpool_info()
            if not pools or any(p["num_threads"]!=1 for p in pools):
                raise ValueError("numerical libraries are not demonstrably single-threaded")
            runtime={"executable":sys.executable,"python":sys.version,"numpy":np.__version__,
                "scipy":scipy.__version__,"platform":sys.platform,"machine":platform.machine(),
                "host":platform.node(),"threads":{k:os.environ[k] for k in THREAD_KEYS},"blas":pools}
            summary["runtime_identity"]=runtime
            if any(item.get("runtime_identity")!=runtime for item in baselines):
                raise ValueError("baseline runtime differs before pair execution")
            write(output/"EnvironmentV1.json",runtime)
            def persist_numeric(path,result):
                if protocol.nonfinite_numeric_paths(result):
                    with Path(path).open("wb") as stream:
                        np.savez_compressed(stream,**codec.numeric_arrays(result,allow_nonfinite=True))
                else:
                    codec.write_numeric_sidecar(path,result)
            def verify_numeric(path,result):
                if codec.contains_nonfinite_tags(result):
                    return codec.verify_failed_numeric_sidecar(path,result)
                return codec.verify_numeric_sidecar(path,result)
            api={"json_data":lambda value:tag_nonfinite(states.json_data(value)),
                "prepare":lambda:states.prepare_common_state(stack,16,binding,policy=protocol.r1_policy(),backend=backend),
                "run":lambda prepared,observer:protocol.run_r1_step(stack,16,binding,prepared,
                    control="D",amplitude_V=.005,times_s=case["times_s"],
                    policy=protocol.r1_policy(.1,time_substeps=(1,2,4)),physics_evidence=True,
                    accepted_step_observer=observer,expected_prepared_sha256=prepared.sha256,backend=backend),
                "persist_numeric":persist_numeric,
                "verify_numeric":verify_numeric,
                "replay":lambda prepared,result,incomplete,observer:physics.verify_r1_step_physics(
                    stack,16,binding,prepared,result,allow_incomplete=incomplete,
                    backend=backend,row_observer=observer)}
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_reconstruction import rebuild_failure_witness
            api["failure_witness"]=lambda prepared,result,rows,failure:rebuild_failure_witness(
                stack,16,binding,prepared,result,backend=backend)
            if backend.is_pair:
                from scripts.analyze_r1_v9_prototype import analyze_records
                from scripts.verify_r1_v9_precision import digest
                def analyze(prepared,result,replay):
                    prepared_record=states.json_data(prepared.to_dict())
                    return analyze_records(result["accepted_steps"],frozen_material,prepared_record,result,
                        context={"source_digest":digest(result["source"]),
                            "prepared_sha256":prepared_record["sha256"],"result_sha256":result["sha256"],
                            "saved_rows_digest":digest(result["accepted_steps"]),
                            "source_commit":args.expected_commit,"request_sha256":args.request_sha256},
                        output=output/"IndependentAnalysis",budget=budget,
                        replay_receipt=getattr(replay,"replay_receipt",None))
                api["after_main"]=analyze
            summary.update(run_trajectory(output,case,args.mode,api,guard))
        guard()
        summary["source_unchanged"]=True
    except (Exception,KeyboardInterrupt) as exc:
        summary.update(execution_status="failed",runner_error=error_record(exc),four_predicates_passed=False)
        write(output/"RunnerFailureV1.json",summary["runner_error"])
        try:
            guard()
            summary["source_unchanged"]=True
        except Exception as source_exc:
            summary["source_unchanged"]=False
            summary["source_error"]=error_record(source_exc)
    summary["module_function_identity"]=check_functions(functions_before)
    summary.setdefault("cost",{})
    summary["engineering_check"]=comparison(summary,baselines,args.request_sha256)
    write(output/"PhaseTimingV1.json",summary.get("phase_timing",{}))
    if all((output/name).exists() for name in ("ReplayRowsV1.jsonl","ResultV1.json","PreparedV1.json")):
        write(output/"ReplayReceiptV1.json",{"schema":"R1V9ReplayReceiptV1",
            "source_commit":args.expected_commit,"source_content_sha256":args.source_sha256,
            "request_sha256":args.request_sha256,"result_file_sha256":sha(output/"ResultV1.json"),
            "prepared_file_sha256":sha(output/"PreparedV1.json"),
            "replay_rows_sha256":sha(output/"ReplayRowsV1.jsonl"),
            "replay_ledger_sha256":sha(output/"ReplayLedgerV1.json") if (output/"ReplayLedgerV1.json").exists() else None,
            "physics_replay_sha256":sha(output/"PhysicsReplayV1.json") if (output/"PhysicsReplayV1.json").exists() else None,
            "checked_rows":summary.get("replay_observed_rows",0),
            "source_unchanged":summary["source_unchanged"],
            "scope":"outer_commit_request_binding_for_actual_replay_callbacks_not_a_self_authenticated_independence_claim"})
    finalize_manifest(output,summary)
    print(json.dumps({k:summary.get(k) for k in ("execution_status","four_predicates_passed",
        "source_unchanged","engineering_check","absolute_engineering_check","P2_qualified")},indent=2))
    ok=(summary.get("four_predicates_passed") and summary.get("numeric_sidecar_exact")
        and summary["source_unchanged"] and summary["module_function_identity"]["unchanged"]
        and (args.mode!="compensated" or (
            summary.get("precision_records",{}).get("record_fields_present")
            and summary.get("independent_analysis",{}).get("passed")
            and summary["engineering_check"]["qualified"]
            and summary["absolute_engineering_check"]["passed"])))
    return 0 if ok else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit","source-sha256","request-sha256","budget-sha256","validation-sha256"):
        parser.add_argument("--"+name,required=True)
    for name in ("request-file","budget-file","validation-contract","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--mode",choices=("baseline","compensated"),required=True)
    parser.add_argument("--baseline-ordinal",type=int)
    parser.add_argument("--baseline-summary",type=Path,action="append",default=[])
    parser.add_argument("--baseline-sha256",action="append",default=[])
    return run(parser.parse_args())


if __name__=="__main__":
    raise SystemExit(main())

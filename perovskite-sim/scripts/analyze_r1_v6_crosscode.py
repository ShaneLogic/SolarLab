"""Reduce V6 independent refinements without fitting to a cross-code residual."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def envelope(values):
    """Predeclared conservative observed spread, not Richardson extrapolation."""
    if len(values) < 3:
        raise ValueError("a convergence ladder needs three points")
    differences = [abs(b-a) for a, b in zip(values, values[1:])]
    return {"values": values, "adjacent_differences": differences,
            "uncertainty": max(differences),
            "last_difference_decreased": differences[-1] <= differences[-2]}


def classify_fault(healthy, faulty, reference, budget):
    """A small faulty residual is not evidence of a sound independent test."""
    hd, fd = abs(healthy-reference), abs(faulty-reference)
    if budget is None:
        return {"healthy_discrepancy":hd,"faulty_discrepancy":fd,
                "detects_under_observed_budget":None,"status":"unknown_budget"}
    return {"healthy_discrepancy":hd,"faulty_discrepancy":fd,
            "fault_makes_agreement_better":fd<hd,
            "normal_within_budget":hd<=budget,"fault_outside_budget":fd>budget,
            "detects_under_observed_budget":hd<=budget and fd>budget,
            "status":"detected" if hd<=budget and fd>budget else "not_demonstrated"}


def qualification(unknowns, nondecreasing_axes, denominator_ratio=None):
    return (not unknowns and not nondecreasing_axes
            and (denominator_ratio is None or denominator_ratio>10))


def validate_new_df(record):
    """A populated final time array cannot prove an untruncated PDE solution."""
    if record.get("status")!="completed" or record.get("matlab_max_computational_threads")!=1:
        raise ValueError("DF run is not completed with one computational thread")
    profiles=[v for k,v in record.items() if k=="equilibrium" or k.startswith(("ctl","ramp_"))]
    if len(profiles)!=3+2*len(record["request"]["times_s"]):
        raise ValueError("DF run has missing requested profiles")
    for profile in profiles:
        if (profile.get("state_time_rows")!=profile.get("time_rows")
                or profile.get("state_nodes")!=profile.get("nx")
                or profile.get("finite_history") is not True):
            raise ValueError("DF state/time history is truncated or nonfinite")
        if any(not math.isfinite(v) for v in profile.values() if isinstance(v,(float,int))):
            raise ValueError("DF profile scalar is nonfinite")


def r1_rows(record, tier):
    return {row["t_s"]: row for row in record["rows"] if row["substeps"] == tier}


def df_rows(record, control="B"):
    return {row["t_s"]: row for key, row in record.items() if key.startswith("ctl"+control+"_")}


def observable(rows, name):
    if name == "g_ion25":
        denominator = rows[1e-8]["Jion_nm25"]
        if denominator == 0:
            raise ValueError("zero ratio denominator")
        return rows[1e-6]["Jion_nm25"]/denominator
    if name == "delta_electron50":
        return rows[1e-6]["Jn_nm50"]-rows[1e-8]["Jn_nm50"]
    field, time = {"ion25_10ns": ("Jion_nm25",1e-8), "ion25_1us": ("Jion_nm25",1e-6),
                   "ion50_10ns": ("Jion_nm50",1e-8), "electron50_10ns": ("Jn_nm50",1e-8),
                   "electron130_1ns": ("Jn_nm130",1e-9)}[name]
    return rows[time][field]


def build_report(root, historical):
    used = []
    def load(path):
        used.append(Path(path))
        return read(path)
    r64=load(root/"R1V5Base/B_N64_F0p1_T4_none/ObservablesV1.json")
    r128=load(root/"R1V5Base/B_N128_F0p1_T4_none/ObservablesV1.json")
    r256=load(root/"R1V5Base/B_N256_F0p1_T4_none/ObservablesV1.json")
    rloose=load(root/"R1V5Imported/B_N256_F1p0_T4.json")
    rtight=load(root/"R1V5N256T4Tight/B_N256_F0p01_T4_none/ObservablesV1.json")
    failed_finer=load(root/"R1V5N256Tight/B_N256_F0p01_T8_none/ObservablesV1.json")
    healthy128=load(root/"R1V5Tight/B_N128_F0p01_T8_none/ObservablesV1.json")
    loose128=load(root/"R1V5Loose/B_N128_F1p0_T4_none/ObservablesV1.json")
    faulty128=load(root/"R1V5DnFault2/B_N128_F0p01_T8_Dn_p1/ObservablesV1.json")
    healthyA=load(root/"R1V5Tight/A_N128_F0p01_T8_none/ObservablesV1.json")
    faultyA=load(root/"R1V5DnFault2/A_N128_F0p01_T8_Dn_p1/ObservablesV1.json")
    df={name:load(historical/"reference"/(name+".json")) for name in (
        "Eps0FixG1","Eps0FixG2","Eps0FixG4","Eps0FixR12","Eps0FixR10",
        "Eps0FixW005","Eps0FixW04","Eps0FixS1e7","Eps0FixS1e11","Eps0FixDefect")}
    new={name:load(root/"DFVerified"/(name+".json")) for name in (
        "FreshG1","TimeHalf","TimeQuarter","Tol9","Tol10","Dn_p1","Dion_p1")}
    constants=load(root/"DFVerified/KBTMatch.json")
    for record in [*new.values(),constants]:
        validate_new_df(record)
    contract=load(root/"ContractV1.json")
    metrics={}
    for name in ("g_ion25","ion25_10ns","ion25_1us","ion50_10ns", "electron50_10ns", "electron130_1ns","delta_electron50"):
        rv=lambda r,t:observable(r1_rows(r,t),name)
        dv=lambda d:observable(df_rows(d),name)
        axes={
            "r1_space_N64_128_256": envelope([rv(r,16) for r in (r64,r128,r256)]),
            "r1_time_4_8_16": envelope([rv(rtight,t) for t in (4,8,16)]),
            "r1_nonlinear_1_0p1_0p01": envelope([rv(r,16) for r in (rloose,r256,rtight)]),
            "df_space_G1_G2_G4": envelope([dv(df[k]) for k in ("Eps0FixG1","Eps0FixG2","Eps0FixG4")]),
            "df_time_maxstep_1_half_quarter": envelope([dv(new[k]) for k in ("FreshG1","TimeHalf","TimeQuarter")]),
            "df_tolerance_8_9_10": envelope([dv(new[k]) for k in ("FreshG1","Tol9","Tol10")]),
            "df_ramp_10_11_12": envelope([dv(df[k]) for k in ("Eps0FixR10","Eps0FixG2","Eps0FixR12")]),
            "df_width_0p4_0p1_0p05_nm": envelope([dv(df[k]) for k in ("Eps0FixW04","Eps0FixG2","Eps0FixW005")]),
            "df_contact_1e7_1e9_1e11": envelope([dv(df[k]) for k in ("Eps0FixS1e7","Eps0FixG2","Eps0FixS1e11")]),
        }
        defect_diagnostic={"values":[dv(df["Eps0FixG2"]),dv(df["Eps0FixDefect"])],
            "observed_change":abs(dv(df["Eps0FixG2"])-dv(df["Eps0FixDefect"])),
            "uncertainty":None, "included_in_budget":False,
            "reason":"historical Et translation is wrong (audit: SRH about 573x); this is a perturbation, not a valid defect-model bound"}
        qscale=1.602176634e-19/constants["constants"]["e_C"]
        corrected=dv(constants)*(1 if name=="g_ion25" else qscale)
        axes["df_kbt_and_q_constants"]={"values":[dv(new["FreshG1"]),corrected],
            "uncertainty":abs(dv(new["FreshG1"])-corrected),
            "kind":"effective-kBT input sensitivity plus exact electrical-current charge conversion"}
        # The chosen comparison has an independently checked finer R1 time
        # grid. DF uses its actual G4 value, not a fitted continuum limit.
        rvalue=rv(rtight,16); dvalue=dv(df["Eps0FixG4"])
        budget=sum(axis["uncertainty"] for axis in axes.values())
        metrics[name]={"units":"dimensionless" if name=="g_ion25" else "A/cm^2",
            "r1":rvalue,"driftfusion":dvalue,"absolute_discrepancy":abs(rvalue-dvalue),
            "budget":budget,"normal_within_observed_budget":abs(rvalue-dvalue)<=budget,
            "axes":axes,"nondecreasing_axes":[k for k,v in axes.items() if v.get("last_difference_decreased") is False],
            "historical_defect_diagnostic":defect_diagnostic,
            "budget_coverage_complete":False,
            "formal_budget_established":False,
            "formal_budget_reason":"model limiting-equation equivalence is unestablished; measured variant spreads are not a proven bound",
            "budget_character":"partial empirical sum of independent observed refinement/known-variant spreads; interface-defect equivalence is unknown, not zero"}
        if name=="g_ion25":
            metrics[name]["dynamic_departure_from_one"]=abs(dvalue-1)
            metrics[name]["dynamic_content_to_budget"]=abs(dvalue-1)/budget
            fault_signal=dv(new["Dion_p1"])-dv(new["FreshG1"])
            metrics[name]["Dion_parameter_sensitivity"]={"healthy_df_G1":dv(new["FreshG1"]),
                "faulty_df_G1":dv(new["Dion_p1"]),"change":fault_signal,
                "signal_to_budget":abs(fault_signal)/budget,
                "faulty_df_to_healthy_r1_discrepancy":abs(dv(new["Dion_p1"])-rvalue),
                "outside_observed_budget":abs(dv(new["Dion_p1"])-rvalue)>budget,
                "scope":"DF input parameter perturbation only; no R1 ion implementation fault tested"}
        if name.startswith("electron") or name=="delta_electron50":
            healthy=rv(healthy128,32); faulty=rv(faulty128,32)
            local_time=envelope([rv(healthy128,t) for t in (8,16,32)])
            faulty_time=envelope([rv(faulty128,t) for t in (8,16,32)])
            local_nonlinear=envelope([rv(r,16) for r in (loose128,r128,healthy128)])
            # Match the actual fault comparison's N128/T32 state. Keep the
            # broad measured spatial envelope, and replace N256 time/F axes.
            fault_budget=(budget-axes["r1_time_4_8_16"]["uncertainty"]
                -axes["r1_nonlinear_1_0p1_0p01"]["uncertainty"]
                +max(local_time["uncertainty"],faulty_time["uncertainty"])
                +local_nonlinear["uncertainty"])
            metrics[name]["Dn_implementation_fault"]={"r1_healthy_N128":healthy,"r1_faulty_N128":faulty,
                **classify_fault(healthy,faulty,dvalue,fault_budget), "fault_signal":faulty-healthy,
                "own_observed_budget":fault_budget,"healthy_N128_time":local_time,
                "faulty_N128_time":faulty_time,"healthy_N128_nonlinear":local_nonlinear,
                "qualified_detection":qualification(["faulty spatial/nonlinear and model equivalence unknown"], []),
                "qualification_unknowns":["faulty-state spatial/nonlinear refinements absent","common continuum model error unestablished"],
                "A_healthy":rv(healthyA,32),"A_faulty":rv(faultyA,32),
                "healthy_B_minus_A":healthy-rv(healthyA,32),"faulty_B_minus_A":faulty-rv(faultyA,32),
                "df_input_Dn_p1_signal":dv(new["Dn_p1"])-dv(new["FreshG1"]),
                "scope":"R1 runtime material D_n_face*1.01 before common preparation and both A/B runs; independent of CSV parameter probe"}
    denominator=metrics["ion25_10ns"]
    metrics["g_ion25"]["denominator_magnitude_to_budget"]=abs(denominator["driftfusion"])/denominator["budget"]
    metrics["g_ion25"]["denominator_resolved"]=abs(denominator["driftfusion"])>10*denominator["budget"]
    metrics["g_ion25"]["qualified_dynamic_validation"]=qualification(
        ["model equivalence unknown", "ion implementation fault not tested"],
        metrics["g_ion25"]["nondecreasing_axes"],metrics["g_ion25"]["denominator_magnitude_to_budget"])
    metrics["g_ion25"]["qualification_unknowns"]=[
        "DF tolerance and finite-interface-width ratio ladders are not monotone",
        "common continuum model error unestablished", "no ion implementation fault was tested"]
    replay_max=max(abs(new["FreshG1"][key][field]-df["Eps0FixG1"][key][field])
                   for key in ("ctlB_1em08","ctlB_1em06") for field in ("Jion_nm25","Jn_nm50"))
    return {"schema":"R1V6CrossCodeAnalysisV1", "contract_sha256":digest(root/"ContractV1.json"),
        "scope":"development_K2_evidence_not_R1_2_or_D3_acceptance", "contract":contract,
        "historical_DF_base_reproduced_max_difference_A_cm2":replay_max,
        "metrics":metrics,"electronic_independent_validation_ready":all(
            metrics[name]["Dn_implementation_fault"]["qualified_detection"]
            for name in ("electron50_10ns","electron130_1ns")),
        "finer_r1_failure":{k:v for k,v in failed_finer.items() if k!="rows"},
        "finer_N128_electron_increment_ladder":{
            str(t):observable(r1_rows(healthy128,t),"delta_electron50") for t in (8,16,32)},
        "unknowns":{
            "continuum_model_equivalence_error":"unknown: finite-width/contact/defect sweeps are observed variants, not a proof of common limiting equations",
            "interface_defect_model_error":"unknown: historical Eps0FixDefect has incorrect Et and is excluded from the budget",
            "N256_T32_complete_response":"unknown: solver failed during its first interval",
            "ion_implementation_fault_detection":"not tested; DF input sensitivity cannot substitute",
            "Dn_vs_Nc_fault_identifiability":"not established; no root-cause claim from a single scalar current",
            "formal_error_bound":"not established; all computed budgets are independent empirical envelopes"},
        "limitations":["Finite-interface and surface-defect sweeps bound observed variants, not proof of identical continuum models",
            "Axes need not converge monotonically at their floating-point floor; all observed noise is retained",
            "Dn faulty R1 run is N128; primary healthy comparison is N256. Shared N128 healthy/fault sides are reported explicitly",
            "No ion implementation mutation was tested; DF Dion parameter sensitivity is not equivalent evidence",
            "Long-window, SRH and dynamic-interface-defect validation are outside this cross-code deliverable"],
        "sources":{str(p):digest(p) for p in used}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--historical",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    report=build_report(args.root,args.historical)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for name, metric in report["metrics"].items():
        print(name,"difference",metric["absolute_discrepancy"],"independent observed budget",metric["budget"],
              "normal in budget",metric["normal_within_observed_budget"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())

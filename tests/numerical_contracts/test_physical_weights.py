"""Manufactured metadata and optional native EWT checks; no SolarLab model."""
import atexit
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import unittest

import numpy as np

from scripts.benchmarks import physical_tolerance as pt
from scripts.benchmarks.native_readback import physical_weight_state

CENSUS = dict(model_constructions=0, SolarLab_F=0, SolarLab_J=0, SolarLab_Poisson=0,
              Arb=0, native_initialization_attempts=0, native_step_calls=0,
              manufactured_residual_calls=0, scratch_weight_evaluations=0)
RECEIPTS = []


def packed(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if hasattr(value, "items"):
        return {k: packed(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [packed(v) for v in value]
    return value


def save_receipt():
    name = os.environ.get("SOLARLAB_PHYSICAL_EWT_RECEIPT")
    if name:
        Path(name).write_text(json.dumps({"scope": "manufactured EWT only; no device qualification",
                                        "calls": CENSUS, "records": packed(RECEIPTS)}, indent=2)+"\n")


atexit.register(save_receipt)


def expansion(values):
    rows = [[] for _ in range(4)]
    for value in values:
        remainder = Fraction(value)
        for row in rows:
            x = float(remainder); row.append(x.hex()); remainder -= Fraction(x)
        if remainder:
            raise ValueError("manufactured four-word capacity")
    return rows


def fixture(nodes=9, dynamic=True, profile=(1e-5, 1e-8), time=0.25):
    names = ("c_m3", "f", "n_m3", "p_m3", "phi_V") if dynamic else ("n_m3", "p_m3", "phi_V")
    offsets = {k: [j*nodes, (j+1)*nodes] for j,k in enumerate(names)}
    roots = {"c_m3": 2.0**40, "n_m3": 2.0**35, "p_m3": 2.0**38, "f": 0.5, "phi_V": 0.0}
    fields = {k: {"high": [roots[k]]*nodes, "low": [0.0]*nodes} for k in names}
    scale = [2.0**20 if k.endswith("m3") else 1.0 for k in names for _ in range(nodes)]
    packet = {"column_scaling": scale, "variable_offsets": offsets,
              "physical_reference_fields": fields,
              "definition": {"temperature": 300.0, "area": 1e-12, "length": 1e-6,
                             "ni": 2e12, "ion_initial": 1e13 if dynamic else 0.0, "trap_density": 2e14},
              "initial_reference": {"payload": {"fields": {
                  k: {w: [v.hex() for v in values] for w,values in f.items()} for k,f in fields.items()}}}}
    policy = pt.make_policy(packet, {"r": profile[0], "alpha": profile[1]})
    size = len(scale)
    lift = [[0.0, 0.0] for _ in range(size)]
    for i in range(*offsets["phi_V"]):
        lift[i][0] = -float(i-offsets["phi_V"][0])/(nodes-1)
    mapping = {"columns_hex": [v.hex() for v in scale], "lift_hex": [[v.hex() for v in r] for r in lift],
               "reference_inputs_hex": [0.125.hex(), 0.0.hex()]}
    segment = {"id": "manufactured", "start": 0.0, "end": 1.0, "voltage": [0.1, 0.2], "photons": [0.0, 0.0]}
    previous = dict(rtol=1e-12, atol=[1e-15]*size, first_step=0.0, max_step=0.1, max_order=5,
                    max_num_steps=200000, max_nonlin_iters=4, max_conv_fails=10, calc_initcond=None, linsolver="sparse", nthreads=1)
    request = {"numeric_packet": packet, "physical_tolerance_policy": policy, "previous_controls": previous,
               "controls": pt.controls(previous, policy), "weight_certificate": policy,
               "budgets": {"wall_s": 1}, "original_budgets": {"wall_s": 1}, "native_admission": False,
               "voltage_lift_map": mapping, "map_identity": pt.digest(mapping), "segments": [segment]}
    frame = {"parent_map_identity": request["map_identity"], "segment_sha256": pt.digest(segment),
             "logical_initialization_index": 1, "t0_hex": float(time).hex(), "size": size,
             "q0_words_hex": expansion([Fraction(1,4)]*size),
             "v0_words_hex": expansion([Fraction(1,256)]*size),
             "weight_rounding": "RN-exact-parent-q_then-unfused-SV-v1"}
    return request, frame, segment, pt.weight_option(request, frame, segment)


def oracle(option, when, raw):
    n = option["size"]; f = lambda v: Fraction(float.fromhex(v))
    start,end = (float.fromhex(option[k]) for k in ("protocol_start_hex", "protocol_end_hex"))
    first,last,rates = ([float.fromhex(v) for v in option[k]] for k in ("inputs_start_hex", "inputs_end_hex", "input_rates_hex"))
    a = first if when == start else last if when == end else [x+v*(when-start) for x,v in zip(first,rates)]
    states = []
    for i in range(n):
        q = sum((f(w[i]) for w in option["q0_words_hex"]),Fraction(raw[i]))
        q += (Fraction(when)-f(option["t0_hex"]))*sum((f(w[i]) for w in option["v0_words_hex"]),Fraction(0))
        x = sum((f(w[i]) for w in option["reference_words_hex"]),Fraction(0)) + f(option["scale_hex"][i])*q
        x += sum((f(option["lift_hex"][i][j])*(Fraction(a[j])-f(option["reference_inputs_hex"][j])) for j in range(2)),Fraction(0))
        states.append(float(x))
    weights = [float.fromhex(s)/(float.fromhex(option["rtol_hex"])*abs(x)+float.fromhex(a))
               for s,x,a in zip(option["scale_hex"],states,option["atol_physical_hex"])]
    return np.array(states), np.array(weights)


class PhysicalMetadataTests(unittest.TestCase):
    def test_reader_rejects_forged_scalar_and_weight_receipts(self):
        request,frame,segment,option=fixture()
        n=option["size"]; raw=np.zeros(n); x,w=oracle(option,0.5,raw)
        words=lambda values: np.array([float.fromhex(v) for v in values],dtype="<f8").tobytes()
        record=dict(schema="sksundae.ida.physical-affine-weight-state.v1",frame_identity=pt.digest(frame),
                    profile_identity=option["profile_identity"],owner="manufactured",generation=1,size=n,
                    installed=True,setter="IDAWFtolerances",setter_status=0,t0_hex=option["t0_hex"],rtol_hex=option["rtol_hex"],
                    weight_rounding=option["weight_rounding"],q0_words=tuple(words(v) for v in option["q0_words_hex"]),
                    v0_words=tuple(words(v) for v in option["v0_words_hex"]),atol=words(option["atol_physical_hex"]),
                    scale=words(option["scale_hex"]),reference_words=b"".join(words(v) for v in option["reference_words_hex"]),
                    lift=b"".join(words(v) for v in option["lift_hex"]),protocol_start_hex=option["protocol_start_hex"],
                    protocol_end_hex=option["protocol_end_hex"],callback_calls=1,callback_status=0,failure_component=None,
                    basis_time_hex=0.5.hex(),basis_u=raw.tobytes(),rounded_physical=x.tobytes(),computed_weights=w.tobytes())
        for key,source in (("inputs_start","inputs_start_hex"),("inputs_end","inputs_end_hex"),
                           ("input_rates","input_rates_hex"),("reference_inputs","reference_inputs_hex")):
            record[key]=tuple(float.fromhex(v) for v in option[source])
        physical_weight_state(record,request,frame,segment,owner="manufactured",generation=1)
        for key,value in (("generation",True),("generation",2),("setter_status",False),("callback_status",False),
                          ("profile_identity","0"*64),("computed_weights",bytes(len(w.tobytes())))):
            bad=dict(record); bad[key]=value
            with self.assertRaises(ValueError):
                physical_weight_state(bad,request,frame,segment,owner="manufactured",generation=1)

    def test_separate_exact_physical_charge_budget(self):
        from perovskite_sim.constants import Q
        request,_,_,_=fixture()
        before=deepcopy(request["budgets"])
        policy=pt.physical_charge_policy(request)
        d=request["numeric_packet"]["definition"]
        qref=Fraction(Q)*Fraction(d["area"])*Fraction(d["length"])*Fraction(d["trap_density"])
        self.assertEqual(Fraction(policy["Qref_C"]),qref)
        self.assertEqual(pt.physical_charge_budgets(request),(qref/10**8,)*3)
        self.assertEqual(request["budgets"],before)
        self.assertEqual(policy["integration_uncertainty_fraction"],[1,3])

    def test_profiles_and_all_supported_field_layouts(self):
        for nodes,dynamic in ((9,False),(17,False),(9,True),(17,True),(33,True),(65,True)):
            for profile in pt.PROFILES:
                request,frame,segment,option = fixture(nodes,dynamic,profile)
                self.assertEqual(pt.validate_request(request),request["physical_tolerance_policy"])
                self.assertEqual(option["size"],nodes*(5 if dynamic else 3))
                self.assertEqual(request["controls"]["nonlin_conv_coef"],1e-8)
                with self.assertRaisesRegex(ValueError,"signed_balance_integration_pending"):
                    pt.require_balance_integration(request)

    def test_malformed_mixed_and_forged_metadata_rejected(self):
        request,_,_,_ = fixture()
        for r,a in ((True,1e-8),(1e-5,1e-9),(-1,1e-8),(float("nan"),1e-8)):
            with self.assertRaises(ValueError): pt.make_policy(request["numeric_packet"],{"r":r,"alpha":a})
        for kind in ("time_weight_policy","diagnostic_accuracy_policy","native_admission","controls","physical_tolerance_policy"):
            bad=deepcopy(request)
            if kind in ("time_weight_policy","diagnostic_accuracy_policy"): bad[kind]={}
            elif kind=="native_admission": bad[kind]=True
            elif kind=="controls": bad[kind]["nonlin_conv_coef"]*=65536
            else: bad[kind]["atol_SI"][0]*=2
            with self.assertRaises(ValueError): pt.validate_request(bad)

    def test_reference_lift_and_moving_frame_invariance(self):
        for nodes in (9,17):
            _,_,_,option=fixture(nodes); n=option["size"]; raw=[0.125]*n; when=0.5
            x,w=oracle(option,when,raw)
            translated=deepcopy(option)
            translated["reference_words_hex"][0]=[(float.fromhex(a)+float.fromhex(s)*0.125).hex() for a,s in zip(option["reference_words_hex"][0],option["scale_hex"])]
            translated["q0_words_hex"]=expansion([Fraction(float.fromhex(q))-Fraction(1,8) for q in option["q0_words_hex"][0]])
            np.testing.assert_array_equal(w,oracle(translated,when,raw)[1])
            moved=deepcopy(option); moved["t0_hex"]=when.hex()
            moved["q0_words_hex"]=expansion([Fraction(float.fromhex(q))+Fraction(1,4)*Fraction(float.fromhex(v)) for q,v in zip(option["q0_words_hex"][0],option["v0_words_hex"][0])])
            np.testing.assert_array_equal(x,oracle(moved,when,raw)[0])
            lifted=deepcopy(option); lifted["reference_inputs_hex"][0]=0.25.hex()
            lifted["q0_words_hex"]=expansion([Fraction(float.fromhex(q))+Fraction(float.fromhex(l[0]))*Fraction(1,8)/Fraction(float.fromhex(s)) for q,l,s in zip(option["q0_words_hex"][0],option["lift_hex"],option["scale_hex"])])
            np.testing.assert_array_equal(w,oracle(lifted,when,raw)[1])

    def test_both_device_execution_barriers(self):
        from scripts.benchmarks.coupled_device_prototype import validate_voltage_lift_native_request
        from scripts.benchmarks.native_readback import _Protocol
        request,_,_,_=fixture()
        with self.assertRaisesRegex(ValueError,"signed_balance_integration_pending"):
            validate_voltage_lift_native_request(None,None,request)
        with self.assertRaisesRegex(ValueError,"signed_balance_integration_pending"):
            _Protocol(request,None)


@unittest.skipUnless(os.environ.get("SOLARLAB_PHYSICAL_EWT_NATIVE_TEST")=="1", "explicit manufactured native admission required")
class PhysicalNativeTests(unittest.TestCase):
    def test_full_terms_preserve_near_zero_cancellation(self):
        from sksundae.ida import IDA
        request,frame,segment,option=fixture()
        packet=request["numeric_packet"]
        for name in ("c_m3","n_m3","p_m3"):
            count=len(packet["physical_reference_fields"][name]["low"])
            packet["physical_reference_fields"][name]["low"]=[2.0**-40]*count
            packet["initial_reference"]["payload"]["fields"][name]["low"]=[(2.0**-40).hex()]*count
        policy=pt.make_policy(packet,{"r":1e-5,"alpha":1e-8})
        request.update(physical_tolerance_policy=policy,weight_certificate=policy,
                       controls=pt.controls(request["previous_controls"],policy))
        option=pt.weight_option(request,frame,segment)
        frame["q0_words_hex"]=expansion([-sum((Fraction(float.fromhex(row[i])) for row in option["reference_words_hex"]),Fraction(0))/Fraction(float.fromhex(option["scale_hex"][i]))
                                         +Fraction(1,2**200)+Fraction(1,2**260) for i in range(option["size"])])
        frame["v0_words_hex"]=expansion([Fraction(1,256)+Fraction(1,2**70)+Fraction(1,2**130)+Fraction(1,2**190)]*option["size"])
        option=pt.weight_option(request,frame,segment)
        raw=np.full(option["size"],-2.0**-200)
        def residual(t,y,yp,out):
            CENSUS["manufactured_residual_calls"]+=1; out[:]=yp+y
        solver=IDA(residual,physical_weight_frame=option,rtol=request["controls"]["rtol"],
                   atol=request["controls"]["atol"],calc_initcond=None)
        CENSUS["native_initialization_attempts"]+=1
        self.assertTrue(solver.init_step(0.25,raw,-raw).success)
        CENSUS["scratch_weight_evaluations"]+=1; record=solver.evaluate_initial_parent_weights()
        x,w=oracle(option,0.25,raw)
        self.assertGreater(x[0],0)
        state=dict(record["parent_weight_state"])
        np.testing.assert_array_equal(np.frombuffer(state["rounded_physical"],dtype="<f8"),x)
        np.testing.assert_array_equal(np.frombuffer(record["weights"],dtype="<f8"),w)
        physical_weight_state(state,request,frame,segment,owner=record["owner"],generation=record["generation"])
        RECEIPTS.append({"full_term_cancellation":record})

    def test_legacy_default_and_parent_option_remain_distinct(self):
        from sksundae.ida import IDA
        def residual(t,y,yp,out):
            CENSUS["manufactured_residual_calls"]+=1; out[:]=yp+y
        solver=IDA(residual,rtol=1e-5,atol=1e-8,calc_initcond=None,first_step=1e-3)
        y=np.ones(2)
        CENSUS["native_initialization_attempts"]+=1; self.assertTrue(solver.init_step(0.,y,-y).success)
        self.assertNotIn("parent_weight_state",solver.statistics())
        CENSUS["native_step_calls"]+=1; self.assertTrue(solver.step(.1,method="onestep",tstop=.1).success)
        packet=solver.last_step_snapshot()
        np.testing.assert_array_equal(np.frombuffer(packet["error_weights"],dtype="<f8"),np.full(2,1.0/(1e-5+1e-8)))
        _,frame,_,_=fixture()
        parent={k:frame[k] for k in ("t0_hex","size","q0_words_hex","v0_words_hex","weight_rounding")}
        parent.update(schema="sksundae.ida.parent-affine-weights.v1",frame_identity=pt.digest(frame))
        solver=IDA(residual,parent_weight_frame=parent,rtol=1e-5,atol=1e-8,calc_initcond=None)
        y=np.full(parent["size"],0.125)
        CENSUS["native_initialization_attempts"]+=1; self.assertTrue(solver.init_step(.25,y,-y).success)
        CENSUS["scratch_weight_evaluations"]+=1; record=solver.evaluate_initial_parent_weights()
        self.assertEqual(record["parent_weight_state"]["schema"],"sksundae.ida.parent-affine-weight-state.v1")
        np.testing.assert_array_equal(np.frombuffer(record["weights"],dtype="<f8"),np.full(parent["size"],1.0/(1e-5*.375+1e-8)))
        RECEIPTS.append({"legacy_default":packet,"legacy_parent":record})

    def test_actual_weights_time_rejections_refresh_and_resize(self):
        from sksundae.ida import IDA
        def residual(t,y,yp,out):
            CENSUS["manufactured_residual_calls"]+=1; out[:]=yp+y
        for nodes,dynamic in ((9,False),(17,False),(9,True),(17,True)):
            request,frame,segment,option=fixture(nodes,dynamic); n=option["size"]; y=np.full(n,0.125)
            settings=dict(rtol=request["controls"]["rtol"],atol=request["controls"]["atol"],calc_initcond=None,
                          first_step=0.1,max_step=0.1,max_num_steps=100,nonlin_conv_coef=1e-8,
                          nonlin_guard="first-correction-wrms-v1",nonlin_trace_capacity=4096)
            solver=IDA(residual,physical_weight_frame=option,**settings)
            CENSUS["native_initialization_attempts"]+=1; self.assertTrue(solver.init_step(0.25,y,-y).success)
            CENSUS["scratch_weight_evaluations"]+=1; initial=solver.evaluate_initial_parent_weights()
            self.assertFalse(initial["solver_applied_weights"]); self.assertTrue(initial["native_memory_phi_and_ewt_unchanged"])
            np.testing.assert_array_equal(np.frombuffer(initial["weights"],dtype="<f8"),oracle(option,0.25,y)[1])
            previous=y.tobytes(); prior_time=0.25
            for _ in range(2):
                CENSUS["native_step_calls"]+=1; result=solver.step(1.0,method="onestep",tstop=1.0)
                self.assertTrue(result.success); packet=solver.last_step_snapshot(); state=dict(packet["parent_weight_state"])
                self.assertEqual(state["basis_u"],previous); self.assertEqual(float.fromhex(state["basis_time_hex"]),prior_time)
                self.assertEqual(state["computed_weights"],packet["error_weights"])
                physical_weight_state(state,request,frame,segment,owner=packet["owner"],generation=packet["generation"])
                RECEIPTS.append({"size":n,"packet":packet}); previous=packet["basis"]["phi"][:n*8]; prior_time=float(result.t)
            self.assertGreater(solver.statistics()["error_test_fails"],0)
            newer=deepcopy(option); newer["t0_hex"]=1.0.hex(); newer["frame_identity"]="b"*64
            CENSUS["native_initialization_attempts"]+=1; self.assertTrue(solver.init_step(1.0,y,-y,physical_weight_frame=newer).success)
            CENSUS["scratch_weight_evaluations"]+=1; endpoint=solver.evaluate_initial_parent_weights()
            np.testing.assert_array_equal(np.frombuffer(endpoint["weights"],dtype="<f8"),oracle(newer,1.0,y)[1])
            RECEIPTS.append({"same_size_reinit_endpoint":endpoint})
            smaller=fixture(9,not dynamic)[3]; m=smaller["size"]; z=np.full(m,0.125)
            CENSUS["native_initialization_attempts"]+=1; self.assertTrue(solver.init_step(0.25,z,-z,physical_weight_frame=smaller).success)
            CENSUS["scratch_weight_evaluations"]+=1; resized=solver.evaluate_initial_parent_weights()
            np.testing.assert_array_equal(np.frombuffer(resized["weights"],dtype="<f8"),oracle(smaller,0.25,z)[1])
            RECEIPTS.append({"resized":resized,"capability":IDA.physical_weight_capability()})

    def test_actual_mixed_and_malformed_options(self):
        from sksundae.ida import IDA
        option=fixture()[3]
        residual=lambda t,y,yp,out: out.__setitem__(slice(None),yp+y)
        with self.assertRaises(ValueError): IDA(residual,physical_weight_frame=option,parent_weight_frame=option)
        for key,value in (("rtol_hex",True),("scale_hex",["nan"]*45),("protocol_end_hex",0.0.hex()),("profile_identity","x"*64)):
            bad=deepcopy(option); bad[key]=value
            with self.assertRaises((ValueError,TypeError)): IDA(residual,physical_weight_frame=bad)


if __name__ == "__main__":
    unittest.main()

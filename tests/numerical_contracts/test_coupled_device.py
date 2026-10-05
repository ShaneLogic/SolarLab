"""Independent constituent/finite-volume/derivative checks; no time integrator."""

from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import importlib.util
from pathlib import Path
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import issparse
import yaml

from perovskite_sim.constants import K_B, Q
from scripts.benchmarks.contract_prototype import ContractError
from scripts.benchmarks.coupled_device_prototype import (
    CoupledSlab, ScaledIDAAdapter, SlabDefinition, protocol_from_plan,
    digest, prepare_native_request, run_native_pilot, snapshot_solver_result,
    state_quality_evidence, charge_interval_bound,
)

REPO = Path(__file__).resolve().parents[2]
PLAN = Path(os.environ["REAL_DEVICE_PLAN"])
GATES = json.loads(PLAN.read_text())["readiness_contract"]
CASES = GATES["cases"]


def dec(value):
    return Decimal.from_float(float(value))


def log_case(request, data, checks):
    data.update(test=request.node.nodeid, checks={k: bool(v) for k, v in checks.items()}, passed=bool(all(checks.values())))
    def encode(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, Decimal): return str(value)
        raise TypeError(type(value).__name__)
    if path := os.environ.get("COUPLED_CASE_LOG"):
        with Path(path).open("a") as f: f.write(json.dumps(data, default=encode, allow_nan=False)+"\n")
    assert all(checks.values()), {k: bool(v) for k, v in checks.items() if not v}


@lru_cache(maxsize=8)
def model(case_id, area=1.0):
    return CoupledSlab(SlabDefinition.from_plan(PLAN, REPO, case_id, area=area), 8)


def point(model, family):
    initial = model.initial()
    if family == "equilibrium": return initial
    x = model.x/model.definition.length
    fields = {v.id: model.field(initial, v.id).copy() for v in model.layout.variables}
    fields["n_m3"] *= 1+0.05*np.sin(np.pi*x)
    fields["p_m3"] *= 1-0.03*np.sin(np.pi*x)
    fields["phi_V"] = -0.04*x+0.002*np.sin(np.pi*x)
    if model.definition.dynamic:
        fields["c_m3"] *= 1+0.05*np.sin(2*np.pi*x)
        fields["f"] += 0.001*np.sin(np.pi*x)
        if family == "near_boundary":
            fields["c_m3"] = model.definition.ion_capacity*(0.005+0.985*x)
            fields["f"] = 1e-6+(1-2e-6)*x
    elif family == "near_boundary":
        fields["n_m3"] *= 100
        fields["p_m3"] *= 0.01
    return model.point(fields, 0.0, [0.04, model.definition.photon_reference*0.4])


def db(x):
    return Decimal(1) if x == 0 else x/(x.exp()-1)


def decimal_kernel(model, y, inputs, ydot=None, precision=80):
    """Independent direct full-SG and node-by-node finite-volume equations."""
    with localcontext() as ctx:
        ctx.prec = precision
        m = model.definition
        get = lambda field: y[model.layout.offsets[field]]
        n, p, phi = get("n_m3"), get("p_m3"), get("phi_V")
        N = len(n); q, area, vt, eps = map(dec, [Q, m.area, m.vt, m.epsilon])
        volume, dx = list(map(dec, model.geometry.volumes)), list(map(dec, model.dx))
        cn, cp, nt, n1, p1 = map(dec, [m.capture_n, m.capture_p, m.trap_density, m.n1, m.p1])
        c = get("c_m3") if m.dynamic else [Decimal(0)]*N
        f = get("f") if m.dynamic else [(cn*n[i]+cp*p1)/(cn*(n[i]+n1)+cp*(p[i]+p1)) for i in range(N)]
        rn = [cn*(n[i]*(1-f[i])-n1*f[i]) for i in range(N)]
        rp = [cp*(p[i]*f[i]-p1*(1-f[i])) for i in range(N)]
        if not m.dynamic:
            # The exact QSS closure makes the two capture channels identical.
            common = [cn*cp*(n[i]*p[i]-n1*p1)/(cn*(n[i]+n1)+cp*(p[i]+p1)) for i in range(N)]
            rn, rp = common, common
        jn, jp, fi, displacement = [], [], [], []
        for i in range(N-1):
            xi = (phi[i+1]-phi[i])/vt
            jn.append(q*dec(m.mu_n)*vt/dx[i]*(db(xi)*n[i+1]-db(-xi)*n[i]))
            jp.append(q*dec(m.mu_p)*vt/dx[i]*(db(xi)*p[i]-db(-xi)*p[i+1]))
            displacement.append(-eps*(phi[i+1]-phi[i])/dx[i])
            if m.dynamic:
                cap = dec(m.ion_capacity)
                xi += -(1-c[i+1]/cap).ln()+(1-c[i]/cap).ln()
                fi.append(dec(m.diffusion_ion)/dx[i]*(db(xi)*c[i]-db(-xi)*c[i+1]))
            else: fi.append(Decimal(0))
        rates = {v: [Decimal(0)]*N for _, v, _ in model.rate_fields}
        for var, flux in [("n_m3", [-v/q for v in jn]), ("p_m3", [v/q for v in jp])]+([("c_m3", fi)] if m.dynamic else []):
            for i, value in enumerate(flux):
                rates[var][i] -= area*value; rates[var][i+1] += area*value
        edges = list(map(dec, model.edges)); alpha = dec(m.alpha)
        for i in range(N):
            generation = area*inputs[1]*((-alpha*edges[i]).exp()-(-alpha*edges[i+1]).exp())
            rates["n_m3"][i] += generation-volume[i]*nt*rn[i]
            rates["p_m3"][i] += generation-volume[i]*nt*rp[i]
            if m.dynamic: rates["f"][i] = volume[i]*nt*(rn[i]-rp[i])
        rho = [q*(p[i]-n[i]+(c[i]-dec(m.ion_initial)-nt*f[i] if m.dynamic else 0)) for i in range(N)]
        g = [area*(displacement[i]-displacement[i-1])-volume[i]*rho[i] for i in range(1, N-1)]
        g += [phi[0], phi[-1]+inputs[0], n[0]-dec(m.n_eq), n[-1]-dec(m.n_eq), p[0]-dec(m.p_eq), p[-1]-dec(m.p_eq)]
        rate = [rates[var][int(node)] for _, var, nodes in model.rate_fields for node in nodes]
        derivative = [Decimal(0)]*len(y) if ydot is None else ydot
        storage_rate = [volume[int(i)]*(nt if var == "f" else 1)*derivative[model.layout.offsets[var].start+int(i)]
                        for _, var, nodes in model.rate_fields for i in nodes]
        residual = [a-b for a, b in zip(storage_rate, rate, strict=True)]+g
        ndot, pdot, phidot = (derivative[model.layout.offsets[v]] for v in ["n_m3", "p_m3", "phi_V"])
        cdot, fdot = ([derivative[model.layout.offsets[v]] for v in ["c_m3", "f"]]
                       if m.dynamic else ([Decimal(0)]*N, [Decimal(0)]*N))
        rhodot = [q*(pdot[i]-ndot[i]+cdot[i]-nt*fdot[i]) for i in range(N)]
        ddot = [-eps*(phidot[i+1]-phidot[i])/dx[i] for i in range(N-1)]
        outer_D = [displacement[0]-rho[0]*volume[0]/area, displacement[-1]+rho[-1]*volume[-1]/area]
        outer_Ddot = [ddot[0]-rhodot[0]*volume[0]/area, ddot[-1]+rhodot[-1]*volume[-1]/area]
        icon = [q*(volume[i]*(pdot[i]-ndot[i])+rates["n_m3"][i]-rates["p_m3"][i]) for i in [0, N-1]]
        qm = [area*outer_D[0], -area*outer_D[1]]
        total = [icon[0]+area*outer_Ddot[0], icon[1]-area*outer_Ddot[1]]
        return {"F": residual, "R": rate, "g": g, "Jn": jn, "Jp": jp, "Fi": fi,
                "Rn": rn, "Rp": rp, "D": displacement, "Qmetal": qm, "Icond": icon, "Itotal": total,
                "Qbody": [sum(volume[i]*rho[i] for i in range(N))]}


def compare(actual, reference, scale, atol, rtol):
    errors, limits = [], []
    with localcontext() as ctx:
        ctx.prec = 80
        for a, b, s in zip(np.asarray(actual).ravel(), reference, np.broadcast_to(scale, np.asarray(actual).shape).ravel(), strict=True):
            e = abs((dec(a)-b)*dec(s)); limit = dec(atol)+dec(rtol)*abs(b*dec(s))
            errors.append(float(e)); limits.append(float(limit))
    return {"scaled_absolute_errors": errors, "scaled_limits": limits,
            "passed": all(a <= b for a, b in zip(errors, limits, strict=True)),
            "max_budget_fraction": max((a/b if b else 0 for a, b in zip(errors, limits, strict=True)), default=0)}


def exact_initial_reference(m, precision=100):
    cfg = yaml.safe_load(Path(m.source_path).read_text()); layer = cfg["layers"][0]
    trap = layer["bulk_defects"][0]
    d = lambda value: Decimal(str(value))
    with localcontext() as ctx:
        ctx.prec = precision
        vt = d(K_B)*d(cfg["device"]["T"])/d(Q)
        gap, energy, nc, nv = map(d, [layer["Eg"], trap["distribution"]["center_eV_above_vb"], layer["Nc300"], layer["Nv300"]])
        n1, p1 = nc*(-(gap-energy)/vt).exp(), nv*(-energy/vt).exp()
        ni2 = nc*nv*(-gap/vt).exp(); ni = ni2.sqrt()
        k = trap["kinetics"]
        cn, cp = d(k["sigma_n_m2"])*d(k["thermal_velocity_n_m_s"]), d(k["sigma_p_m2"])*d(k["thermal_velocity_p_m_s"])
        nt = d(trap["distribution"]["total_density_m3"])
        def state(u):
            n, p = ni*u.exp(), ni*(-u).exp()
            f = (cn*n+cp*p1)/(cn*(n+n1)+cp*(p+p1))
            return n, p, f
        lo, hi = Decimal(-100), Decimal(100)
        for _ in range(256 if m.dynamic else 0):
            mid = (lo+hi)/2; n, p, f = state(mid)
            if p-n-nt*f > 0: lo = mid
            else: hi = mid
        n, p, f = state((lo+hi)/2 if m.dynamic else Decimal(0))
        return {"n1": n1, "p1": p1, "ni": ni, "n_eq": n, "p_eq": p, "f_eq": f}


@pytest.mark.parametrize("case_id", CASES)
def test_initial_source_numeric_packet(case_id, request):
    m = model(case_id); refs = exact_initial_reference(m.definition)
    errors = {k: float(abs(dec(getattr(m.definition, k))/v-1)) for k, v in refs.items()}
    initial = m.initial(); packet = m.numeric_packet()
    if folder := os.environ.get("COUPLED_PACKET_DIR"):
        path = Path(folder)/(case_id+".json")
        with path.open("x") as f: json.dump(packet, f, indent=2); f.write("\n")
    log_case(request, {"family": "source_and_initial", "case": case_id,
                       "packet": packet, "Decimal100_reference": refs, "relative_errors": errors,
                       "initial_residual_scaled": m.Drow*m.residual(initial, m.tangent_rate(initial, np.zeros(2)), np.zeros(2))},
             {"source_equilibrium_reference": max(errors.values()) <= GATES["initial_reference_relative_limit"],
              "proper_node_count": m.count == 9, "sparse_square": m.graph.shape == (m.layout.size, m.layout.size),
              "volume_once": abs(sum(m.geometry.volumes)-m.definition.area*m.definition.length) <= 2e-16*m.definition.area*m.definition.length})


@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("family", GATES["points"])
def test_values_and_independent_ports(case_id, family, request):
    m = model(case_id); p = point(m, family); ydot = m.S*np.sin(np.arange(m.layout.size)+0.3)*1e5
    a = np.array([2e5, 1e20]); ref = decimal_kernel(m, list(map(dec, p.y)), list(map(dec, p.inputs)), list(map(dec, ydot)))
    e, obs = m.evaluate(p), m.observe(p, ydot, a)
    quantities = {"F": (m.residual(p, ydot, a), m.Drow), "Jn": (e.electron_current, 1),
                  "Jp": (e.hole_current, 1), "Fi": (e.ion_flux, 1/max(m.definition.ion_initial, 1)),
                  "Qmetal": (obs.ports.charge, 1), "Icond": (obs.ports.conduction, 1),
                  "Itotal": (obs.ports.current, 1), "Qbody": ([obs.body_charge], 1)}
    checks = {k: compare(value, ref[k], scale, GATES["value_scaled_atol"], GATES["value_scaled_rtol"])
              for k, (value, scale) in quantities.items()}
    log_case(request, {"family": "independent_values", "case": case_id, "point": family, "point_identity": p.identity, "comparisons": checks},
             {"independent_values": all(v["passed"] for v in checks.values()),
              "value_path_has_no_derivative_matrices": e.rate_jacobian is None and not e.nodal_derivatives,
              "raw_rate_preserved": np.array_equal(obs.derivative, ydot)})


def directions(m, p):
    fields = {v.id: m.field(p, v.id) for v in m.layout.variables}
    vectors = []
    for variable in m.layout.variables:
        v = np.zeros(m.layout.size); state = fields[variable.id]
        scale = (np.full(m.count, m.definition.vt) if variable.id == "phi_V" else
                 np.minimum(state, 1-state) if variable.id == "f" else
                 np.minimum(state, m.definition.ion_capacity-state) if variable.id == "c_m3" else state)
        v[m.layout.offsets[variable.id]] = 0.1*scale*np.cos(np.arange(m.count)+0.2)
        vectors.append((variable.id, v))
    vectors.append(("coupled", sum(v for _, v in vectors)))
    return [(name+("+" if sign == 1 else "-"), sign*v) for name, v in vectors for sign in [1, -1]]


@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("family", GATES["points"])
@pytest.mark.parametrize("cj", GATES["cj_s_inv"])
def test_analytic_sparse_directions_and_fd_ladder(case_id, family, cj, request):
    m = model(case_id); p = point(m, family); ydot = m.S*np.sin(np.arange(m.layout.size)+0.3)*1e5
    adot = np.array([2e5, 1e20]); J = m.sparse_jacobian(p, cj)
    dense = m.common_system().linearize(p, ydot, adot).ida_matrix(cj)
    records = []
    for name, direction in directions(m, p):
        references = []
        for precision in [80, 100]:
            with localcontext() as ctx:
                ctx.prec = precision
                h = Decimal("1e-22")
                y, yp, dy = list(map(dec, p.y)), list(map(dec, ydot)), list(map(dec, direction))
                values = []
                for sign in [1, -1]:
                    values.append(decimal_kernel(m, [a+sign*h*b for a,b in zip(y,dy)], list(map(dec,p.inputs)),
                                                 [a+sign*h*dec(cj)*b for a,b in zip(yp,dy)], precision)["F"])
                references.append([(a-b)/(2*h) for a,b in zip(*values)])
        actual = J @ direction
        exact = compare(actual, references[0], m.Drow, GATES["jacobian_scaled_atol"], GATES["jacobian_scaled_rtol"])
        errors = []
        for eps in GATES["fd_epsilon_ladder"]:
            plus = m.coordinates.point(p.y+eps*direction, p.time, p.inputs)
            minus = m.coordinates.point(p.y-eps*direction, p.time, p.inputs)
            fd = (m.residual(plus, ydot+cj*eps*direction, adot)-m.residual(minus, ydot-cj*eps*direction, adot))/(2*eps)
            errors.append(float(np.max(np.abs(m.Drow*(fd-actual)))/(1+np.max(np.abs(m.Drow*actual)))))
        plateau = any(all(e <= GATES["fd_plateau_relative_limit"] for e in errors[i:i+GATES["fd_required_adjacent_points"]])
                      for i in range(len(errors)-GATES["fd_required_adjacent_points"]+1))
        uncertainty = compare(np.zeros(m.layout.size), [a-b for a,b in zip(*references)], m.Drow,
                              GATES["jacobian_scaled_atol"]*GATES["reference_fraction_of_gate"], 0)
        records.append({"direction": name, "reference": exact, "reference_uncertainty": uncertainty,
                        "physical_direction": direction.tolist(), "fd_errors": errors, "plateau_passed": plateau})
    log_case(request, {"family": "sparse_jacobian", "case": case_id, "point": family, "cj": cj,
                       "graph_identity": m.graph.identity, "nnz": J.nnz, "directions": records,
                       "physical_y": p.y, "physical_ydot": ydot, "inputs": p.inputs, "input_rates": adot,
                       "epsilon_ladder": GATES["fd_epsilon_ladder"]},
             {"decimal_Jvp": all(v["reference"]["passed"] for v in records),
              "reference_uncertainty": all(v["reference_uncertainty"]["passed"] for v in records),
              "all_fd_plateaus": all(v["plateau_passed"] for v in records),
              "common_dense_crosscheck": np.array_equal(J.toarray(), dense),
              "sparse_structure_complete": np.count_nonzero((J.toarray() != 0) & (dense == 0)) == 0})


@pytest.mark.parametrize("case_id", CASES)
def test_input_storage_and_external_scaling(case_id, request):
    m = model(case_id); p = point(m, "perturbed"); segment = protocol_from_plan(PLAN, case_id)[0]
    adapter = ScaledIDAAdapter(m, segment); initial = m.initial(); yp = m.tangent_rate(initial, np.array([0.0, 0.0]))
    z = initial.y/m.S; zdot = yp/m.S
    mapping = np.max(np.abs(adapter.residual(0, z, zdot)-m.Drow*m.residual(adapter.point(0,z), m.S*zdot, np.zeros(2))))
    partials = m.common_system().linearize(p, np.ones(m.layout.size), np.ones(2))
    input_checks = []
    for j, scale in enumerate([0.1, max(m.definition.photon_reference, 2e16)]):
        inputs = list(map(dec,p.inputs)); h=Decimal('1e-22'); da=[Decimal(0),Decimal(0)];da[j]=dec(scale)
        values=[]
        with localcontext() as ctx:
            ctx.prec=80
            for sign in [1,-1]:values.append(decimal_kernel(m,list(map(dec,p.y)),[a+sign*h*b for a,b in zip(inputs,da)])["F"])
            ref=[(a-b)/(2*h) for a,b in zip(*values)]
        input_checks.append(compare(partials.inputs[:,j]*scale,ref,m.Drow,GATES['jacobian_scaled_atol'],GATES['jacobian_scaled_rtol']))
    right, inc = m.coordinates.advance(p, 1e-5*directions(m,p)[-1][1], 1e-6, p.inputs)
    expected_delta = m.common_system().storage.delta(p,right,inc)
    storage_error = float(np.max(np.abs(adapter.finite_storage_increment(p,right,inc)-expected_delta)))
    conservative = m.common_system().conservative_residual(p,right,inc)
    log_case(request,{"family":"input_storage_scaling","case":case_id,"input_derivatives":input_checks,
                      "scaling_error":mapping,"storage_delta_error":storage_error},
             {"input_columns":all(v['passed'] for v in input_checks),"external_scale_once":mapping==0,
              "physical_storage_once":storage_error==0,"conservative_residual":np.allclose(m.conservative_residual(p,right,inc),conservative,rtol=1e-14,atol=1e-12),
              "explicit_time_and_input_rate_zero":np.all(partials.time==0) and np.all(partials.input_rate==0),
              "conservative_jacobian":np.allclose(m.conservative_jacobian(p,right).toarray(),m.common_system().conservative_jacobian(p,right),rtol=1e-14,atol=1e-20)})


@pytest.mark.parametrize("case_id",CASES)
@pytest.mark.parametrize("family",GATES['points'])
def test_charge_identities_area_and_affine_tangent(case_id,family,request):
    m=model(case_id);p=point(m,family);a=np.array([1e5,1e20]);rate=m.tangent_rate(p,a)
    obs=m.observe(p,rate,a,'physical_tangent');other=model(case_id,0.37);p2=point(other,family)
    obs2=other.observe(p2,rate,a,'physical_tangent');e=m.evaluate(p)
    qref=Q*m.definition.area*m.definition.length*max(m.definition.ni,m.definition.ion_initial,m.definition.trap_density)
    jref=Q*m.definition.vt*(m.definition.mu_n+m.definition.mu_p)*m.definition.ni/m.definition.length
    errors={'body_conduction_A':abs(obs.body_charge_rate-sum(obs.ports.conduction)),
            'total_terminal_A':abs(sum(obs.ports.current)),
            'gauss_residual_relation_C':abs(obs.gauss_defect+sum(e.algebraic[:m.count-2])),
            'all_face_current_A_m2':float(np.ptp(obs.interior_total_current))}
    scaled=[(obs2.ports.charge,0.37*obs.ports.charge),(obs2.ports.current,0.37*obs.ports.current),
            (other.storage_value(p2),0.37*m.storage_value(p)),(other.evaluate(p2).rate,0.37*e.rate)]
    area_errors=[float(np.max(np.abs(x-y))/(1+np.max(np.abs(y)))) for x,y in scaled]
    bound=m.affine_constraint_error(p)
    log_case(request,{'family':'charge_area_tangent','case':case_id,'point':family,'errors':errors,
                      'charge_limit_C':qref*GATES['charge_identity_Qref_fraction'],'current_limit_A':jref*GATES['current_identity_Jref_fraction'],
                      'area_errors':area_errors,'affine_constraint':bound,'physical_point_identity':p.identity},
             {'body_uses_conduction_only':errors['body_conduction_A']<=jref*GATES['current_identity_Jref_fraction'],
              'terminal_KCL':errors['total_terminal_A']<=jref*GATES['current_identity_Jref_fraction'],
              'finite_gauss_identity':errors['gauss_residual_relation_C']<=qref*GATES['charge_identity_Qref_fraction'],
              'all_faces':errors['all_face_current_A_m2']<=jref*GATES['current_identity_Jref_fraction'],
              'area_applied_once':max(area_errors)<=GATES['area_relative_limit'],
              'model_geometry_reference_bound':p.identity!=p2.identity,
              'no_projection':bound['point_identity']==p.identity and not bound['state_changed'],
              'one_cached_factor':m.factorizations==1})


@pytest.mark.parametrize("case_id",CASES)
def test_protocol_domains_and_history(case_id,request):
    m=model(case_id);segments=protocol_from_plan(PLAN,case_id);p=m.initial()
    obs=m.observe(p,m.tangent_rate(p,np.zeros(2)),np.zeros(2),'physical_tangent')
    original=p.identity; failures=[]
    for field,value in [('n_m3',-1.0)]+([('c_m3',m.definition.ion_capacity),('f',1.01)] if m.definition.dynamic else []):
        fields={v.id:m.field(p,v.id).copy() for v in m.layout.variables};fields[field][0]=value
        with pytest.raises(ContractError) as error:m.point(fields)
        failures.append(error.value.reason)
    with pytest.raises(ValueError):obs.derivative[0]=0
    joins=[(a.inputs(a.end)[0].tolist(),b.inputs(b.start)[0].tolist()) for a,b in zip(segments,segments[1:])]
    log_case(request,{'family':'protocol_domain_history','case':case_id,'segments':[asdict(v) for v in segments],
                      'join_values':joins,'guard_reasons':failures},
             {'complete_history_duration':segments[-1].end==(9.2 if m.definition.dynamic else 1.2e-6),
              'joins_continuous':all(a==b for a,b in joins),'history_unchanged':p.identity==original,
              'all_guards':len(failures)==(3 if m.definition.dynamic else 1)})


@pytest.mark.parametrize("case_id",CASES)
@pytest.mark.parametrize("family",GATES['points'])
def test_exterior_port_state_rate_derivatives(case_id,family,request):
    m=model(case_id);p=point(m,family);rate=m.S*np.sin(np.arange(m.layout.size)+0.3)*1e5
    partials=m.port_partials(p);records=[]
    for name,direction in directions(m,p):
        with localcontext() as ctx:
            ctx.prec=80;h=Decimal('1e-22');y=list(map(dec,p.y));yp=list(map(dec,rate));dy=list(map(dec,direction))
            values=[]
            for sign in [1,-1]:
                values.append(decimal_kernel(m,[a+sign*h*b for a,b in zip(y,dy)],list(map(dec,p.inputs)),
                                             [a+sign*h*Decimal('1000')*b for a,b in zip(yp,dy)]))
            ref_i=[(a-b)/(2*h) for a,b in zip(values[0]['Itotal'],values[1]['Itotal'])]
            ref_q=[(a-b)/(2*h) for a,b in zip(values[0]['Qmetal'],values[1]['Qmetal'])]
        current=partials['current_y']@direction+1000*(partials['current_ydot']@direction)
        charge=partials['charge_y']@direction
        records.append({'direction':name,'current':compare(current,ref_i,1,GATES['jacobian_scaled_atol'],GATES['jacobian_scaled_rtol']),
                        'charge':compare(charge,ref_q,1,GATES['jacobian_scaled_atol'],GATES['jacobian_scaled_rtol'])})
    log_case(request,{'family':'port_derivatives','case':case_id,'point':family,'directions':records},
             {'all_port_directions':all(v['current']['passed'] and v['charge']['passed'] for v in records),
              'full_sparse_state_rate_partials':issparse(partials['current_y']) and issparse(partials['current_ydot'])})


@pytest.mark.parametrize('case_id', CASES)
def test_affine_finite_constraint_includes_contact_half_cells(case_id, request):
    m = model(case_id); p = point(m, 'near_boundary'); identity = p.identity
    bound = m.affine_constraint_error(p)
    correction = np.zeros(m.layout.size)
    correction[m.layout.offsets['phi_V']] = bound['potential_correction_V']
    for field, values in zip(['n_m3', 'p_m3'], np.reshape(bound['contact_density_correction_m3'], (2, 2))):
        offsets = m.layout.offsets[field]
        correction[[offsets.start, offsets.stop-1]] = values
    shadow = m.coordinates.point(p.y+correction, p.time, p.inputs)
    # This is an independent test point, never an accepted-state projection.
    ref = decimal_kernel(m, list(map(dec, p.y)), list(map(dec, p.inputs)))
    corrected = decimal_kernel(m, list(map(dec, shadow.y)), list(map(dec, shadow.inputs)))
    qref = Q*m.definition.area*m.definition.length*max(m.definition.ni, m.definition.ion_initial, m.definition.trap_density)
    delta_q = np.asarray([float(a-b) for a,b in zip(corrected['Qmetal'], ref['Qmetal'])])
    delta_body = float(corrected['Qbody'][0]-ref['Qbody'][0])
    metal_error = float(np.max(np.abs(delta_q-np.asarray(bound['metal_charge_correction_C']))))
    body_error = abs(delta_body-bound['body_charge_correction_C'])
    storage_error = float(np.max(np.abs(m.dynamic_mass @ correction)))
    residual = float(np.max(np.abs(m.Drow[m.dynamic_count:]*m.evaluate(shadow).algebraic)))
    log_case(request, {'family': 'affine_finite_g_contacts', 'case': case_id, 'bound': bound,
                      'metal_correction_error_C': metal_error, 'body_correction_error_C': body_error,
                      'independent_corrected_constraint_scaled_max': residual,
                      'differential_storage_correction': storage_error},
             {'charge_correction_includes_half_cells': max(metal_error,body_error)<=qref*GATES['charge_identity_Qref_fraction'],
              'affine_constraints_all_rows': residual<=GATES['value_scaled_atol'],
              'differential_storage_preserved': storage_error==0,
              'original_state_unchanged': p.identity==identity and not bound['state_changed']})


@pytest.mark.parametrize('case_id', CASES)
@pytest.mark.parametrize('cj', [0.0, 1000.0])
def test_scaled_csc_retains_zero_slots(case_id, cj, request):
    m = model(case_id); segment = protocol_from_plan(PLAN, case_id)[0]
    adapter = ScaledIDAAdapter(m, segment); z = m.initial().y/m.S
    p = adapter.point(0.0, z); physical = m.sparse_jacobian(p, cj)
    scaled = adapter.jacobian(0.0, z, cj)
    dense = m.Drow[:,None]*physical.toarray()*m.S[None,:]
    max_error = float(np.max(np.abs(scaled.toarray()-dense)))
    log_case(request, {'family': 'scaled_csc_structure', 'case': case_id, 'cj': cj,
                      'nnz_slots': scaled.nnz, 'actual_zero_values': int(np.count_nonzero(scaled.data==0)),
                      'external_jacobian_error': max_error},
             {'same_declared_slots': scaled.nnz==m.graph.nnz and np.array_equal(scaled.indices,m.graph.indices)
                                     and np.array_equal(scaled.indptr,m.graph.indptr),
              'scaled_jacobian_exact': max_error==0,
              'finite_all_values': np.isfinite(scaled.data).all()})


@pytest.mark.parametrize('case_id', CASES)
def test_native_request_is_exact_but_cannot_execute_without_admission(case_id, request):
    m = model(case_id); segments = protocol_from_plan(PLAN,case_id); initial = m.initial()
    packet = prepare_native_request(m,segments)
    with pytest.raises(ContractError,match='native_pilot_not_admitted'):
        run_native_pilot(m,segments,packet,{},lambda record: pytest.fail('unadmitted record'))
    restored = json.loads(json.dumps(packet,allow_nan=False))
    checks = state_quality_evidence(m,ScaledIDAAdapter(m,segments[0]).point(0,np.asarray(packet['z0'])),packet['budgets'])
    samples = packet['observation_times']
    coverage = all(len(samples[s.id])>=33 and samples[s.id][0]==s.start and samples[s.id][-1]==s.end
                   and np.all(np.diff(samples[s.id])>0) for s in segments)
    if m.definition.dynamic:
        hold = segments[-1];extra = hold.start+np.geomspace(1e-7,9.0,33)
        extra[-1] = hold.end
        coverage = coverage and all(t in samples[hold.id] for t in extra)
    packet_path = None
    if base := os.environ.get('COUPLED_PACKET_DIR'):
        packet_path = str(Path(base)/(case_id+'.NativeRequest.json'))
        Path(packet_path).write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
    log_case(request, {'family':'prospective_native_request','case':case_id,
                      'request_sha256':digest(packet),'packet_path':packet_path,'initial_checks':checks,
                      'samples_per_segment':{k:len(v) for k,v in samples.items()},
                      'unadmitted_guard':'native_pilot_not_admitted'},
             {'complete_protocol':segments[-1].end==(9.2 if m.definition.dynamic else 1.2e-6),
              'roundtrip_identity':digest(packet)==digest(restored),
              'frozen_samples_complete':coverage,'physical_initial_eligibility':checks['passed'],
              'no_native_backend_import':'sksundae' not in sys.modules,
              'history_unchanged':initial.identity==m.initial().identity,
              'no_trajectory_qualification':not checks['trajectory_error_certified'] and not packet['refinement_or_full_device_qualification']})


def test_native_snapshot_does_not_borrow_interpolator_buffers(request):
    values=np.array([1.,2.]);rates=np.array([3.,4.])
    snapshot=snapshot_solver_result(SimpleNamespace(t=1.,y=values,yp=rates,success=True,status=0,message='test only'))
    values[:]=99;rates[:]=88
    with pytest.raises(ValueError):snapshot['z'].setflags(write=True)
    log_case(request,{'family':'native_history_buffer_ownership','backend_executed':False},
             {'values_independent':np.array_equal(snapshot['z'],[1.,2.]),
              'rates_independent':np.array_equal(snapshot['zdot'],[3.,4.])})


@pytest.mark.parametrize('case_id', CASES)
@pytest.mark.parametrize('family', GATES['points'])
def test_public_problem_consumes_only_values_for_residual(case_id, family, monkeypatch, request):
    m=model(case_id);p=point(m,family)
    ydot=m.S*np.sin(np.arange(m.layout.size)+0.3)*1e5;adot=np.array([2e5,1e20])
    dense=m.common_system().linearize(p,ydot,adot)
    direct=m.residual(p,ydot,adot)
    right,increment=m.coordinates.advance(p,1e-5*directions(m,p)[-1][1],1e-6,p.inputs)
    problem=m.public_problem();evaluation=m.evaluate;calls=[]
    def observe_evaluate(point,*,derivatives=False):
        calls.append(derivatives)
        return evaluation(point,derivatives=derivatives)
    def forbidden_dense():raise AssertionError('public problem used dense convenience path')
    monkeypatch.setattr(m,'evaluate',observe_evaluate)
    monkeypatch.setattr(m,'common_system',forbidden_dense)
    residual=problem.residual(p,ydot,adot);value_calls=list(calls);calls.clear()
    partials=problem.linearize(p,ydot,adot);derivative_calls=list(calls)
    be=problem.conservative_jacobian(p,right)
    storage=problem.storage.delta(p,right,increment)
    comparisons={name:bool(np.array_equal(getattr(partials,name).toarray() if name in {'y','ydot'} else getattr(partials,name),
                                        getattr(dense,name))) for name in ['y','ydot','inputs','input_rate','time']}
    log_case(request,{'family':'public_implicit_problem','case':case_id,'point':family,
                      'source_identity':problem.source_identity,'structure_identity':problem.structure.identity,
                      'residual_derivative_flags':value_calls,'linearization_derivative_flags':derivative_calls,
                      'partial_comparisons':comparisons,'conservative_nnz':be.nnz},
             {'value_path_only':value_calls==[False],'one_analytic_derivative_evaluation':derivative_calls==[True],
              'identical_physical_residual':np.array_equal(residual,direct),'all_chain_partials':all(comparisons.values()),
              'source_bound_storage':problem.storage.source_identity==m.source_identity,
              'finite_storage_through_public_API':np.array_equal(storage,m.storage_delta(p,right,increment)),
              'conservative_fixed_slots':be.nnz==m.graph.nnz and np.array_equal(be.indices,m.graph.indices)
                                          and np.array_equal(be.indptr,m.graph.indptr)})


@pytest.mark.parametrize('case_name,defect,uncertainty,prior_d,prior_u,expected_failure', [
    ('separate_terms_pass_but_sum_fails',0.85,0.25,0.0,0.0,'interval_total'),
    ('separate_intervals_pass_but_reference_prefix_fails',0.0,0.2,0.0,0.2,'prefix_reference_share'),
    ('interval_passes_but_total_prefix_fails',0.2,0.1,0.75,0.0,'prefix_total'),
    ('within_all_componentwise_allocations',0.1,0.1,0.1,0.1,None),
])
def test_charge_combined_interval_and_prefix_allocations(case_name,defect,uncertainty,prior_d,prior_u,expected_failure,request):
    delta=np.ones(3);last=delta-np.array([0.0,defect,0.0]);middle=last-np.array([0.0,uncertainty,0.0])
    evidence=charge_interval_bound(delta,[middle,middle,last],[0.,prior_d,0.],[0.,prior_u,0.],1.0)
    expected_total=defect+uncertainty
    expected_prefix=prior_d+prior_u+expected_total
    checks={'expected_classification':evidence['passed']==(expected_failure is None),
            'specific_failure':expected_failure is None or not evidence['checks'][expected_failure],
            'combined_interval_retained':abs(evidence['interval_total_bound_C'][1]-expected_total)<=1e-15,
            'combined_prefix_retained':abs(evidence['prefix_total_bound_C'][1]-expected_prefix)<=1e-15,
            'unrelated_components_unchanged':all(evidence['prefix_total_bound_C'][i]==0 for i in [0,2]),
            'reference_is_explicit_estimate':'not a rigorous continuum certificate' in evidence['reference_estimate']}
    log_case(request,{'family':'charge_allocation_controls','case':case_name,'evidence':evidence},checks)


@pytest.mark.parametrize('used,rss,live,expected', [
    (268435457,1024,True,'output_cap'),
    (0,None,True,'rss_measurement_unavailable'),
    (0,None,False,None),
    (268435456,2147483648,True,None),
])
def test_native_watchdog_output_and_rss_exit_race(used,rss,live,expected,request):
    path=Path(os.environ['COUPLED_WATCHDOG_SOURCE'])
    spec=importlib.util.spec_from_file_location('coupled_watchdog_controls',path)
    watchdog=importlib.util.module_from_spec(spec);spec.loader.exec_module(watchdog)
    budgets={'total_output_bytes':268435456,'rss_bytes':2147483648,'wall_s':180}
    actual=watchdog.watchdog_decision(used,rss,live,1.0,budgets)
    log_case(request,{'family':'native_watchdog_controls','output_bytes':used,'rss_bytes':rss,
                      'child_live':live,'expected':expected,'observed':actual},
             {'guard_outcome':actual==expected,'native_import_forbidden':'sksundae' not in sys.modules})


@lru_cache(maxsize=8)
def affine_model(case_id, area=1.0):
    from scripts.benchmarks.coupled_device_prototype import AffineCoupledSlab

    return AffineCoupledSlab(SlabDefinition.from_plan(PLAN, REPO, case_id, area=area), 8)


def affine_design():
    return json.loads(Path(os.environ["AFFINE_DESIGN"]).read_text())


def word_decimals(value):
    """Both exact input words, with no reference conversion to binary64."""
    with localcontext() as ctx:
        ctx.prec = 100
        return [dec(hi)+dec(lo) for hi,lo in zip(value.high.ravel(),value.low.ravel(),strict=True)]


def compare_words(actual, reference, scale, atol, rtol):
    values = word_decimals(actual)
    scales = np.broadcast_to(scale, actual.shape).ravel()
    with localcontext() as ctx:
        ctx.prec = 100
        errors = [abs((value-ref)*dec(s)) for value,ref,s in zip(values,reference,scales,strict=True)]
        limits = [dec(atol)+dec(rtol)*abs(ref*dec(s)) for ref,s in zip(reference,scales,strict=True)]
    return {"errors":[float(v) for v in errors], "limits":[float(v) for v in limits],
            "passed":all(e<=b for e,b in zip(errors,limits,strict=True)),
            "max_budget_fraction":max((float(e/b) if b else (0.0 if not e else 1e300)
                                      for e,b in zip(errors,limits,strict=True)),default=0.0)}


def affine_reference(m, p, rate, precision=100):
    values = [v for variable in m.layout.variables for v in word_decimals(p.state.field(variable.id))]
    return decimal_kernel(m, values, list(map(dec,p.inputs)), list(map(dec,rate)), precision)


@pytest.mark.parametrize("case_id", CASES)
def test_affine_initial_reference_and_public_residual(case_id, request):
    """This smallest slice needs no new public trial or native backend."""
    m = affine_model(case_id); p=m.initial(); design=affine_design()
    rate=np.zeros(m.layout.size); adot=np.zeros(2)
    e=m.evaluate(p); observed=m.observe(p,rate,adot); problem=m.public_problem()
    ref=affine_reference(m,p,rate,100); ref80=affine_reference(m,p,rate,80)
    nc08=design["NC08"]
    checks={}
    metrics={"residual":compare_words(problem.residual(p,rate,adot),ref["F"],m.Drow,
                                      GATES["value_scaled_atol"],GATES["value_scaled_rtol"])}
    for name,value in (("Jn",e.electron_current),("Jp",e.hole_current)):
        metrics[name]=compare_words(value,ref[name],1.0,nc08["weak_flux"]["atol_A_m2"],nc08["weak_flux"]["rtol"])
    for name,value in (("Rn",e.reaction_n),("Rp",e.reaction_p)):
        metrics[name]=compare_words(value,ref[name],m.definition.trap_density,
                                   nc08["capture"]["atol"],nc08["capture"]["rtol"])
    for name,value in (("Icond",observed.conduction_inward),("Itotal",observed.total_inward)):
        metrics[name]=compare_words(value,ref[name],1.0,
                                   m.definition.area*nc08["weak_flux"]["atol_A_m2"],nc08["weak_flux"]["rtol"])
    charge_atol=Q*sum(m.geometry.volumes)*nc08["density_increment_atol"]
    for name,value in (("Qmetal",observed.metal_charge),("Qbody",observed.body_charge)):
        metrics[name]=compare_words(value,ref[name],1.0,charge_atol,nc08["density_increment_rtol"])
    checks.update({name:v["passed"] for name,v in metrics.items()})
    expected={"n_m3":m.definition.n_eq,"p_m3":m.definition.p_eq,"phi_V":0.0}
    if m.definition.dynamic:expected.update(c_m3=m.definition.ion_initial,f=m.definition.f_eq)
    checks["original_reference_words_unchanged"]=all(
        np.all(m.field(p,name).high==value) and np.all(m.field(p,name).low==0) for name,value in expected.items())
    checks["cumulative_coordinate_zero"]=np.array_equal(p.y,np.zeros(m.layout.size))
    checks["initial_reference_not_replaced_by_equilibrium_oracle"]=all(
        any(v!=0 for v in word_decimals(value))==any(v!=0 for v in ref[name])
        for name,value in (("Rn",e.reaction_n),("Rp",e.reaction_p),("Qbody",observed.body_charge)))
    checks["sparse_linearization_preserved"]=problem.linearize(p,rate,adot).ida_matrix(1.0).nnz==m.graph.nnz
    checks["native_import_forbidden"]="sksundae" not in sys.modules
    reference_uncertainty={name:max((abs(a-b) for a,b in zip(ref[name],ref80[name])),default=Decimal(0))
                           for name in ("F","Rn","Rp","Icond","Qmetal","Qbody")}
    log_case(request,{"family":"affine_initial_reference","case":case_id,"metrics":metrics,
                      "reference_uncertainty":reference_uncertainty,
                      "reference_point_identity":p.identity,"initial_coordinate":p.y,
                      "reference_capture_n_per_s":word_decimals(e.reference_reaction_n),
                      "reference_capture_p_per_s":word_decimals(e.reference_reaction_p),
                      "reference_body_charge_C":word_decimals(observed.body_charge),
                      "physical_charge_atol_C":charge_atol,"scope":"initial algebra only; no trial/native qualification"},
             checks)


def affine_exact_input(m, cumulative):
    """Independent reference from supplied primitive words, before projection."""
    from scripts.benchmarks.precision_prototype import PrimitiveExpansion

    # Test inputs are immutable public expansions or explicitly supplied floats.
    words = cumulative.words if isinstance(cumulative,PrimitiveExpansion) else (np.asarray(cumulative),)
    base = [v for variable in m.layout.variables for v in word_decimals(m.field(m.reference,variable.id))]
    with localcontext() as ctx:
        ctx.prec=100
        return [b+sum((dec(word[i]) for word in words),Decimal(0)) for i,b in enumerate(base)]


def affine_point(m, family):
    x=m.x/m.definition.length
    u=np.zeros(m.layout.size)
    if family=="equilibrium":
        return m.trial(u,0.0,(0.0,0.0))[0],u
    u[m.layout.offsets["n_m3"]]=m.definition.n_eq*0.05*np.sin(np.pi*x)
    u[m.layout.offsets["p_m3"]]=-m.definition.p_eq*0.03*np.sin(np.pi*x)
    u[m.layout.offsets["phi_V"]]=-0.04*x+0.002*np.sin(np.pi*x)
    if m.definition.dynamic:
        u[m.layout.offsets["c_m3"]]=m.definition.ion_initial*0.05*np.sin(2*np.pi*x)
        u[m.layout.offsets["f"]]=0.001*np.sin(np.pi*x)
        if family=="near_boundary":
            u[m.layout.offsets["c_m3"]]=m.definition.ion_capacity*(0.005+0.985*x)-m.definition.ion_initial
            u[m.layout.offsets["f"]]=1e-6+(1-2e-6)*x-m.definition.f_eq
    elif family=="near_boundary":
        u[m.layout.offsets["n_m3"]]=100*(m.definition.n_eq+u[m.layout.offsets["n_m3"]])-m.definition.n_eq
        u[m.layout.offsets["p_m3"]]=0.01*(m.definition.p_eq+u[m.layout.offsets["p_m3"]])-m.definition.p_eq
    return m.trial(u,0.0,(0.04,0.4*m.definition.photon_reference))[0],u


def affine_independent_reference(m,p,cumulative,rate,precision=100):
    return decimal_kernel(m,affine_exact_input(m,cumulative),list(map(dec,p.inputs)),
                          list(map(dec,rate)),precision)


@pytest.mark.parametrize("case_id",CASES)
@pytest.mark.parametrize("family",GATES["points"])
def test_affine_kernel_values_and_half_cells(case_id,family,request):
    m=affine_model(case_id);p,u=affine_point(m,family)
    rate=m.S*(0.17+np.sin(np.arange(m.layout.size)))
    e=m.evaluate(p);o=m.observe(p,rate,[0.1,0.3*m.definition.photon_reference])
    ref=affine_independent_reference(m,p,u,rate)
    ref80=affine_independent_reference(m,p,u,rate,80)
    design=affine_design();nc=design["NC08"]
    metrics={"F":compare_words(m.public_problem().residual(p,rate,[0.1,0.3*m.definition.photon_reference]),
                              ref["F"],m.Drow,GATES["value_scaled_atol"],GATES["value_scaled_rtol"])}
    for name,value in (("Jn",e.electron_current),("Jp",e.hole_current)):
        metrics[name]=compare_words(value,ref[name],1,nc["weak_flux"]["atol_A_m2"],nc["weak_flux"]["rtol"])
    for name,value in (("Icond",o.conduction_inward),("Itotal",o.total_inward)):
        metrics[name]=compare_words(value,ref[name],1,m.definition.area*nc["weak_flux"]["atol_A_m2"],nc["weak_flux"]["rtol"])
    qscale=1/(Q*m.definition.ion_initial*m.definition.area*m.definition.length) if m.definition.dynamic else 1/(Q*m.definition.ni*m.definition.area*m.definition.length)
    for name,value in (("Qmetal",o.metal_charge),("Qbody",o.body_charge)):
        metrics[name]=compare_words(value,ref[name],qscale,GATES["value_scaled_atol"],GATES["value_scaled_rtol"])
    with localcontext() as ctx:
        ctx.prec=100
        uncertainty={}
        for name in metrics:
            scales=(m.Drow if name=="F" else np.full(len(ref[name]),qscale if name in {"Qmetal","Qbody"} else 1.0))
            uncertainty[name]=[abs(a-b)*dec(s) for a,b,s in zip(ref[name],ref80[name],scales,strict=True)]
    checks={name:value["passed"] for name,value in metrics.items()}
    checks["reference_uncertainty_share"]=all(
        all(float(v)<=bound/3 for v,bound in zip(values,metrics[k]["limits"],strict=True))
        for k,values in uncertainty.items())
    checks["blocking_ion_exterior_definition"]=all(v==0 for v in word_decimals(e.ion_exterior_flux))
    checks["affine_estimator_preserves_point"]=not m.affine_constraint_error(p)["state_changed"]
    checks["source_reference_unchanged"]=m.reference.identity==m.initial().identity
    log_case(request,{"family":"affine_values","case":case_id,"point_family":family,
                      "point_identity":p.identity,"cumulative_SI":u,"physical_rates":rate,
                      "metrics":metrics,"reference_uncertainty":uncertainty,
                      "blocking_exterior_ion_flux":0.0,"geometry_volumes":m.geometry.volumes},checks)


@pytest.mark.parametrize("case_id",CASES)
@pytest.mark.parametrize("size",["zero","tiny_positive","tiny_negative","half_ulp_positive","half_ulp_negative"])
def test_affine_signed_weak_density_consumption(case_id,size,request):
    from scripts.benchmarks.precision_prototype import DoubleArithmetic, PrimitiveExpansion

    m=affine_model(case_id);u=np.zeros(m.layout.size)
    common=np.zeros(m.layout.size)
    # A common spatial remainder forces the weak localized change to survive
    # after an absolute DD projection can already lose its third term.
    common[m.layout.offsets["p_m3"]]=1.0
    previous,_=m.trial(common,1e-3,(0.0,0.0))
    magnitude=(0.0 if size=="zero" else 1e-40 if "tiny" in size else 0.5*np.spacing(m.definition.p_eq))
    if "negative" in size:magnitude=-magnitude
    u[m.layout.offsets["p_m3"].start+1]=magnitude
    arithmetic=DoubleArithmetic()
    cumulative=PrimitiveExpansion.from_value(arithmetic.freeze(
        arithmetic.add(arithmetic.array(common),arithmetic.array(u))))
    p,inc=m.trial(cumulative,2e-3,(0.0,0.0),predecessor=previous)
    finite=m.finite_physical_changes(previous,p,inc);e=m.evaluate(p)
    zero=np.zeros(m.layout.size)
    left_ref=affine_independent_reference(m,previous,common,zero)
    right_ref=affine_independent_reference(m,p,cumulative,zero)
    nc=affine_design()["NC08"]
    with localcontext() as ctx:
        ctx.prec=100
        capture={name:[b-a for a,b in zip(left_ref[name],right_ref[name])] for name in ("Rn","Rp")}
        qbody=[right_ref["Qbody"][0]-left_ref["Qbody"][0]]
        qmetal=[b-a for a,b in zip(left_ref["Qmetal"],right_ref["Qmetal"])]
    metrics={"hole_current":compare_words(e.hole_current,right_ref["Jp"],1,nc["weak_flux"]["atol_A_m2"],nc["weak_flux"]["rtol"]),
             "capture_n":compare_words(finite["capture_n_per_s"],capture["Rn"],m.definition.trap_density,nc["capture"]["atol"],nc["capture"]["rtol"]),
             "capture_p":compare_words(finite["capture_p_per_s"],capture["Rp"],m.definition.trap_density,nc["capture"]["atol"],nc["capture"]["rtol"])}
    charge_atol=Q*sum(m.geometry.volumes)*nc["density_increment_atol"]
    metrics["body_charge"]=compare_words(finite["body_charge_C"],qbody,1,charge_atol,nc["density_increment_rtol"])
    metrics["metal_charge"]=compare_words(finite["metal_charge_C"],qmetal,1,charge_atol,nc["density_increment_rtol"])
    inc_values=word_decimals(inc.field("p_m3"))
    checks={name:value["passed"] for name,value in metrics.items()}
    increment_field=inc.field("p_m3")
    expected_words=u[m.layout.offsets["p_m3"]]
    checks["exact_physical_increment"]=(
        increment_field.high.tobytes()==expected_words.tobytes()
        and increment_field.low.tobytes()==np.zeros_like(expected_words).tobytes())
    checks["zero_or_signed_weak_current"]=(all(v==0 for v in word_decimals(e.hole_current)) if magnitude==0
                                          else word_decimals(e.hole_current)[0]*dec(magnitude)<0)
    checks["weak_finite_storage_nonzero"]=(all(v==0 for v in word_decimals(finite["storage"])) if magnitude==0
                                           else any(v!=0 for v in word_decimals(finite["storage"])))
    checks["reference_retained"]=m.reference.identity==m.initial().identity
    if "tiny" in size:
        checks["absolute_projection_can_match_while_driver_survives"]=(
            np.array_equal(m.field(previous,"p_m3").high,m.field(p,"p_m3").high)
            and np.array_equal(m.field(previous,"p_m3").low,m.field(p,"p_m3").low))
    log_case(request,{"family":"affine_weak_consumption","case":case_id,"size":size,
                      "increment_m3":magnitude,"actual_increment":inc_values,"metrics":metrics,
                      "actual_increment_words":{"high":increment_field.high,"low":increment_field.low},
                      "first_hole_current_A_m2":word_decimals(e.hole_current)[0],
                      "finite_body_charge_C":word_decimals(finite["body_charge_C"]),
                      "left_identity":previous.identity,"right_identity":p.identity},checks)


@pytest.mark.parametrize("case_id",CASES)
def test_affine_signed_domains_and_old_native_guard(case_id,request):
    from scripts.benchmarks.coupled_device_prototype import AffineScaledIDAAdapter

    m=affine_model(case_id);u=np.zeros(m.layout.size)
    for var in m.layout.variables:
        if var.id!="phi_V":u[m.layout.offsets[var.id]]=-0.1*m.field(m.reference,var.id).high
    u[m.layout.offsets["phi_V"]]=-0.02*m.x/m.definition.length
    p,_=m.trial(u,0.0,(0.02,0.0));m.validate(p)
    with pytest.raises(ContractError,match="affine_native_request"):
        prepare_native_request(m,protocol_from_plan(PLAN,case_id))
    with pytest.raises(ContractError,match="affine_native_runner_not_admitted"):
        run_native_pilot(m,protocol_from_plan(PLAN,case_id),{}, {},lambda value:None)
    adapter=AffineScaledIDAAdapter(m,protocol_from_plan(PLAN,case_id)[0])
    z=u/m.S;actual=adapter.point(0,z)
    bad=u.copy();bad[m.layout.offsets["n_m3"]]=-m.definition.n_eq
    with pytest.raises(ContractError):
        invalid,_=m.trial(bad,0,(0.0,0.0));m.validate(invalid)
    log_case(request,{"family":"affine_domains","case":case_id,
                      "signed_remainders":u,"actual_scaled_input":actual.y,
                      "physical_positive_fields":{v.id:m.project_output(m.field(p,v.id)) for v in m.layout.variables}},
             {"negative_remainders_valid":bool(np.any(p.y<0)),
              "scale_once":np.array_equal(actual.y,m.S*z),
              "absolute_density_native_constraint_not_reused":True,
              "native_import_forbidden":"sksundae" not in sys.modules})


@pytest.mark.parametrize("case_id",CASES)
def test_affine_compact_history_and_area(case_id,request):
    from scripts.benchmarks.coupled_device_prototype import AffineDeviceHistory
    from scripts.benchmarks.precision_prototype import PrimitiveExpansion

    m=affine_model(case_id);h=AffineDeviceHistory(m)
    u=np.zeros(m.layout.size);u[m.layout.offsets["p_m3"]]=1.0
    p1,_=m.trial(u,1e-3,(0.0,0.0));du=np.zeros_like(u);du[m.layout.offsets["p_m3"].start+1]=1e-40
    cumulative=PrimitiveExpansion((u,du,np.zeros_like(u),np.zeros_like(u)))
    p2,inc=m.trial(cumulative,2e-3,(0.0,0.0),predecessor=p1)
    rates=m.S*(0.17+np.sin(np.arange(m.layout.size)))
    frames=[h.sample(p1,u,m.reference,rates,[0.0,0.0],event_side="left"),
            h.sample(p2,cumulative,p1,-rates,[0.0,0.0],event_side="right")]
    encoded=json.dumps({"reference":h.reference_record,"frames":frames},sort_keys=True)
    decoded=json.loads(encoded);r1,_,v1,_=h.restore(decoded["reference"],decoded["frames"][0],m.reference)
    r2,di,v2,_=h.restore(decoded["reference"],decoded["frames"][1],r1)
    damaged=json.loads(json.dumps(frames[1]));damaged["predecessor_identity"]="foreign"
    with pytest.raises(ContractError,match="history_reference_or_predecessor_mismatch"):
        h.restore(h.reference_record,damaged,r1)
    with pytest.raises(ContractError,match="raw_solver_values_required"):
        h.sample(p2,cumulative,p1,rates,[0.0,0.0],origin="native")
    alt=affine_model(case_id,0.37);ap1,_=alt.trial(u,1e-3,(0.0,0.0))
    ap2,ai=alt.trial(cumulative,2e-3,(0.0,0.0),predecessor=ap1)
    b=m.finite_physical_changes(p1,p2,inc);a=alt.finite_physical_changes(ap1,ap2,ai)
    with localcontext() as ctx:
        ctx.prec=100
        area_errors={name:max((abs(x/dec(0.37)/y-1) if y else abs(x)
                              for x,y in zip(word_decimals(a[name]),word_decimals(b[name]))),default=Decimal(0))
                     for name in ("storage","body_charge_C","metal_charge_C")}
    checks={"history_identity":(r1.identity,r2.identity)==(p1.identity,p2.identity),
            "rates_preserved":np.array_equal(v1,rates) and np.array_equal(v2,-rates),
            "finite_weak_change_preserved":di.field("p_m3").identity_bytes()==inc.field("p_m3").identity_bytes(),
            "one_reference_only":all("reference" not in frame and not frame["reference_embedded"] for frame in frames),
            "event_sides_preserved":[f["event_side"] for f in decoded["frames"]]==["left","right"],
            "nonunit_area_once":all(float(v)<=GATES["area_relative_limit"] for v in area_errors.values())}
    log_case(request,{"family":"affine_history_area","case":case_id,"area_errors":area_errors,
                      "reference_bytes":len(json.dumps(h.reference_record).encode()),
                      "sample_bytes":[len(json.dumps(f).encode()) for f in frames],
                      "combined_json_bytes":len(encoded.encode()),
                      "raw_solver_data":"absent and explicitly algebraic; no native samples fabricated",
                      "restored_identities":[r1.identity,r2.identity]},checks)


def affine_directions(m):
    """Signed physical blocks and one coupled direction; all columns also get an oracle."""
    x=m.x/m.definition.length
    block_shape=0.2+0.5*np.sin(np.pi*x)
    interior_shape=np.sin(np.pi*x);interior_shape[[0,-1]]=0.0
    base=[]
    for variable in m.layout.variables:
        d=np.zeros(m.layout.size)
        d[m.layout.offsets[variable.id]]=m.S[m.layout.offsets[variable.id]]*(
            interior_shape if variable.id=="f" else block_shape)
        base.append((variable.id,d))
    base.append(("coupled",sum(((-1)**i*d for i,(_,d) in enumerate(base)),np.zeros(m.layout.size))))
    return [(name,sign,sign*d) for name,d in base for sign in (-1,1)]


@pytest.mark.parametrize("case_id",CASES)
@pytest.mark.parametrize("family",GATES["points"])
def test_affine_complete_sparse_jacobian_ladder(case_id,family,request):
    """Identity-column oracle is explicitly limited to N8 and <=45 unknowns."""
    from scripts.benchmarks.coupled_device_prototype import AffineScaledIDAAdapter

    m=affine_model(case_id);p,u=affine_point(m,family);N=m.layout.size
    assert m.intervals==8 and N<=45
    rate=m.S*(0.17+np.sin(np.arange(N)))
    adot=np.array([0.1,0.3*m.definition.photon_reference])
    problem=m.public_problem();linear=problem.linearize(p,rate,adot)
    y=affine_exact_input(m,u);a=list(map(dec,p.inputs));v=list(map(dec,rate))
    h=Decimal("1e-22")
    state_refs={};rate_refs={}
    with localcontext() as ctx:
        ctx.prec=100
        for precision in (80,100):
            sy=[];sv=[]
            for column in range(N):
                step=h*dec(m.S[column])
                yp,ym=list(y),list(y);vp,vm=list(v),list(v)
                yp[column]+=step;ym[column]-=step
                vp[column]+=step;vm[column]-=step
                fp=decimal_kernel(m,yp,a,v,precision)["F"];fm=decimal_kernel(m,ym,a,v,precision)["F"]
                gp=decimal_kernel(m,y,a,vp,precision)["F"];gm=decimal_kernel(m,y,a,vm,precision)["F"]
                sy.append([(b-c)/(2*h)*dec(s) for b,c,s in zip(fp,fm,m.Drow)])
                sv.append([(b-c)/(2*h)*dec(s) for b,c,s in zip(gp,gm,m.Drow)])
            state_refs[precision]=sy;rate_refs[precision]=sv
        matrix_evidence=[];matrix_pass=True;reference_pass=True
        for cj in GATES["cj_s_inv"]:
            actual=(m.Drow[:,None]*linear.ida_matrix(cj).toarray())*m.S[None,:]
            errors=[];limits=[];uncertainties=[]
            for row in range(N):
                er=[];li=[];un=[]
                for col in range(N):
                    expected=state_refs[100][col][row]+dec(cj)*rate_refs[100][col][row]
                    r80=state_refs[80][col][row]+dec(cj)*rate_refs[80][col][row]
                    error=abs(dec(actual[row,col])-expected)
                    limit=dec(GATES["jacobian_scaled_atol"])+dec(GATES["jacobian_scaled_rtol"])*abs(expected)
                    uncertainty=abs(r80-expected)
                    matrix_pass &= error<=limit;reference_pass &= uncertainty<=limit/3
                    er.append(float(error));li.append(float(limit));un.append(float(uncertainty))
                errors.append(er);limits.append(li);uncertainties.append(un)
            matrix_evidence.append({"cj_s_inv":cj,"errors":errors,"limits":limits,"reference_uncertainty":uncertainties})

        fa_metrics=[]
        for column,scale in enumerate((m.definition.vt,m.definition.photon_reference)):
            plus,minus=list(a),list(a);plus[column]+=h*dec(scale)
            one_sided=column==1 and a[column]==0
            if not one_sided:minus[column]-=h*dec(scale)
            rp=decimal_kernel(m,y,plus,v,100)["F"];rm=decimal_kernel(m,y,minus,v,100)["F"]
            expected=[(b-c)/((1 if one_sided else 2)*h)*dec(s) for b,c,s in zip(rp,rm,m.Drow)]
            actual=linear.inputs[:,column]*scale*m.Drow
            errors=[abs(dec(x)-z) for x,z in zip(actual,expected)]
            limits=[dec(GATES["jacobian_scaled_atol"])+dec(GATES["jacobian_scaled_rtol"])*abs(z) for z in expected]
            fa_metrics.append({"column":column,"input_scale":scale,"one_sided_at_photon_boundary":one_sided,"errors":[float(x) for x in errors],
                               "limits":[float(x) for x in limits],"passed":all(e<=b for e,b in zip(errors,limits))})

        # A conservative residual derivative has M-h*R_y on differential rows
        # and the unscaled affine constraint derivative on algebraic rows.
        be_h=0.001
        right,increment=m.trial(u,be_h,p.inputs,predecessor=m.reference)
        be=(m.Drow[:,None]*problem.conservative_jacobian(m.reference,right).toarray())*m.S[None,:]
        be_errors=[];be_pass=True
        for row in range(N):
            entries=[]
            for col in range(N):
                expected=(rate_refs[100][col][row]+dec(be_h)*state_refs[100][col][row]
                          if row<m.dynamic_count else state_refs[100][col][row])
                error=abs(dec(be[row,col])-expected)
                limit=dec(GATES["jacobian_scaled_atol"])+dec(GATES["jacobian_scaled_rtol"])*abs(expected)
                be_pass &= error<=limit
                entries.append(float(error))
            be_errors.append(entries)

    ladder=[];plateau_pass=True
    for cj in GATES["cj_s_inv"]:
        J=linear.ida_matrix(cj)
        for name,sign,direction in affine_directions(m):
            target=m.Drow*(J @ direction)
            denominator=1+float(np.max(np.abs(target)))
            errors=[]
            for eps in GATES["fd_epsilon_ladder"]:
                plus,_=m.trial(u+eps*direction,p.time,p.inputs)
                minus,_=m.trial(u-eps*direction,p.time,p.inputs)
                fp=problem.residual(plus,rate+cj*eps*direction,adot).as_dd()
                fm=problem.residual(minus,rate-cj*eps*direction,adot).as_dd()
                from scripts.benchmarks.precision_prototype import DoubleArray
                fd=m.Drow*m.project_output(DoubleArray.from_dd((fp-fm)/(2*eps)))
                errors.append(float(np.max(np.abs(fd-target)))/denominator)
            qualifies=[e<=GATES["fd_plateau_relative_limit"] for e in errors]
            adjacent=any(all(qualifies[i:i+GATES["fd_required_adjacent_points"]])
                         for i in range(len(qualifies)-GATES["fd_required_adjacent_points"]+1))
            plateau_pass &= adjacent
            ladder.append({"cj_s_inv":cj,"direction":name,"sign":sign,
                           "physical_direction":direction.tolist(),"eps":GATES["fd_epsilon_ladder"],
                           "normalized_errors":errors,"adjacent_plateau":adjacent})
    later,_=m.trial(u,0.037,p.inputs)
    value0=problem.residual(p,rate,adot);value1=problem.residual(later,rate,adot)
    adapter_pass=True
    adapter_errors=[]
    if family=="equilibrium":
        adapter=AffineScaledIDAAdapter(m,protocol_from_plan(PLAN,case_id)[0])
        for cj,oracle in zip(GATES["cj_s_inv"],matrix_evidence,strict=True):
            actual=adapter.jacobian(0.0,np.zeros(N),cj,rate/m.S).toarray()
            with localcontext() as ctx:
                ctx.prec=100
                errors=[[float(abs(dec(actual[row,col])-(state_refs[100][col][row]+dec(cj)*rate_refs[100][col][row])))
                         for col in range(N)] for row in range(N)]
            adapter_pass &= all(e<=b for er,br in zip(errors,oracle["limits"],strict=True) for e,b in zip(er,br,strict=True))
            adapter_errors.append({"cj_s_inv":cj,"independent_scaled_errors":errors})
    checks={"all_state_rate_columns":bool(matrix_pass),"reference_uncertainty_share":bool(reference_pass),
            "input_derivatives":all(v["passed"] for v in fa_metrics),"BE_all_columns":bool(be_pass),
            "all_declared_FD_ladders":bool(plateau_pass),
            "Ft_and_Fadot_zero":np.all(linear.time==0) and np.all(linear.input_rate==0)
                               and value0.identity_bytes()==value1.identity_bytes(),
            "sparse_slots_preserved":all(linear.ida_matrix(cj).nnz==m.graph.nnz for cj in GATES["cj_s_inv"]),
            "independent_adapter_scale_at_equilibrium":bool(adapter_pass),
            "no_native_import":"sksundae" not in sys.modules}
    log_case(request,{"family":"affine_full_jacobian","case":case_id,"point_family":family,
                      "cumulative_SI":u,"physical_rates":rate,"input_rates":adot,"dimension":N,
                      "matrix_oracle":"every identity column, Decimal80/100 direct finite volume, h=1e-22",
                      "matrix_evidence":matrix_evidence,"input_evidence":fa_metrics,
                      "BE_step_s":be_h,"BE_errors":be_errors,"FD_ladders":ladder,"adapter_errors":adapter_errors,
                      "native_trajectory":False},checks)


@lru_cache(maxsize=2)
def affine_pilot_request(case_id):
    from scripts.benchmarks.coupled_device_prototype import prepare_affine_native_request

    sources = json.loads(Path(os.environ["AFFINE_NATIVE_PRIOR"]).read_text())
    old = json.loads(Path(sources[case_id]).read_text())
    m = affine_model(case_id)
    return prepare_affine_native_request(m, protocol_from_plan(PLAN, case_id), old), old


@pytest.mark.parametrize("case_id", CASES)
def test_affine_native_weight_proof_and_frozen_request(case_id, request):
    m = affine_model(case_id)
    proposed, old = affine_pilot_request(case_id)
    proof = proposed["weight_certificate"]
    r = Fraction.from_float(proposed["controls"]["rtol"])
    old_r = Fraction.from_float(old["controls"]["rtol"])
    checked = 0
    smallest_margin = None
    for row, new_a, old_a in zip(proof["components"], proposed["controls"]["atol"], old["controls"]["atol"]):
        xref = Fraction(row["reference_exact"])
        s = Fraction.from_float(float.fromhex(row["scale_hex"]))
        # Independent exact envelope points include depletion, either sign,
        # and a zero reference; feasibility is not an assumption of the proof.
        for multiplier in [Fraction(-2), Fraction(-1), -1+Fraction(1,2**26),
                           Fraction(0), Fraction(1,7), Fraction(1), Fraction(2**20)]:
            u = multiplier*(abs(xref)+s)
            lhs = s*Fraction.from_float(new_a)+r*abs(u)
            rhs = s*Fraction.from_float(old_a)+old_r*abs(xref+u)
            margin = rhs-lhs
            assert margin >= 0
            smallest_margin = margin if smallest_margin is None else min(smallest_margin, margin)
            checked += 1
    if output := os.environ.get("AFFINE_NATIVE_PACKET_DIR"):
        with (Path(output)/("NativeRequest"+case_id+".json")).open("x") as stream:
            json.dump(proposed,stream,indent=2,allow_nan=False);stream.write("\n")
    checks = {"positive_scalar_rtol": 0 < r <= old_r,
              "positive_vector_atol": min(proposed["controls"]["atol"]) > 0,
              "exact_component_certificates": all(all(row["checks"].values()) for row in proof["components"]),
              "signed_denominator_envelopes": checked == 7*m.layout.size,
              "physical_and_resource_budgets_unchanged": proposed["budgets"] == old["budgets"],
              "full_protocol_unchanged": digest(proposed["segments"]) == digest(old["segments"]),
              "all_observations_unchanged": proposed["observation_times"] == old["observation_times"],
              "original_constraints_not_applied_to_signed_u": not any(k in proposed["controls"] for k in ("constraints_idx","constraints_type")),
              "no_global_certificate": not proof["global_accuracy_or_conservation_certified"],
              "no_native_import": "sksundae" not in sys.modules}
    log_case(request,{"family":"affine_native_weight_readiness","case":case_id,
                      "rtol_hex":float(r).hex(),"request_sha256":digest(proposed),
                      "exact_envelope_checks":checked,"smallest_denominator_margin_exact":str(smallest_margin),
                      "native_trajectory":False},checks)


@pytest.mark.parametrize("case_id", CASES)
def test_affine_native_scale_roundoff_independent_decimal(case_id, request):
    from scripts.benchmarks.coupled_device_prototype import affine_scaling_roundoff

    m = affine_model(case_id)
    values = np.resize(np.array([0.0,-0.75,np.nextafter(1.,2.),1e-40,-1e-30,.17]),m.layout.size)
    rates = np.resize(np.array([.3,-1e-30,0.,np.nextafter(.5,0.),-.17,1e9]),m.layout.size)
    evidence = affine_scaling_roundoff(m,values,rates)
    comparisons, nonzero = 0, 0
    with localcontext() as ctx:
        ctx.prec = 220
        for name, vector in (("u",values),("udot",rates)):
            bounds = {int(i):dec(float.fromhex(h)) for i,h in evidence[name]["nonzero_component_error_upper_hex"]}
            for i,(s,v,a) in enumerate(zip(m.S,vector,m.S*vector)):
                error = abs(dec(s)*dec(v)-dec(a))
                assert error <= bounds.get(i,Decimal(0))
                assert bool(error) == (i in bounds)
                comparisons += 1; nonzero += bool(error)
    log_case(request,{"family":"affine_scale_roundoff","case":case_id,"component_checks":comparisons,
                      "nonzero_products":nonzero,"bounds":evidence,"native_trajectory":False},
             {"all_components_checked":comparisons==2*m.layout.size,"nontrivial_roundoff":nonzero>0,
              "not_global_error_claim":not evidence["certifies_global_physical_error"]})


@pytest.mark.parametrize("case_id", CASES)
def test_affine_native_protocol_history_and_payload(case_id, request):
    from scripts.benchmarks.coupled_device_prototype import AffineDeviceHistory, affine_native_sample

    m = affine_model(case_id)
    proposal, _ = affine_pilot_request(case_id)
    segments = protocol_from_plan(PLAN,case_id)
    history = AffineDeviceHistory(m)
    snapshot = {"time":segments[0].start,"z":np.array(proposal["z0"]),
                "zdot":np.array(proposal["zdot0"]),"success":True,"status":0,"message":"synthetic algebraic snapshot; no backend"}
    start = time.perf_counter()
    initial = affine_native_sample(m,history,proposal,segments[0],snapshot,m.reference,origin="segment_initial")
    sample_seconds = time.perf_counter()-start
    left_snapshot = {**snapshot,"time":segments[0].end}
    left = affine_native_sample(m,history,proposal,segments[0],left_snapshot,initial[0],origin="stop_output")
    right_snapshot = {**snapshot,"time":segments[1].start}
    right = affine_native_sample(m,history,proposal,segments[1],right_snapshot,left[0],origin="segment_initial")
    restored = history.restore(history.reference_record,right[-1]["history"],left[0])
    altered = replace(segments[1],voltage=(segments[1].voltage[0],segments[1].voltage[1]+.01))
    with pytest.raises(ContractError,match="native_history_segment_binding"):
        affine_native_sample(m,history,proposal,altered,right_snapshot,left[0],origin="segment_initial")
    changed = {**right_snapshot,"z":right_snapshot["z"].copy()}
    changed["z"][m.layout.offsets["p_m3"].start+1]=-.125
    with pytest.raises(ContractError,match="native_restart_changed_physical_state"):
        affine_native_sample(m,history,proposal,segments[1],changed,left[0],origin="segment_initial")
    checks = {"left_right_sides":left[-1]["history"]["event_side"]=="left" and right[-1]["history"]["event_side"]=="right",
              "raw_vectors_present":right[-1]["history"]["raw_solver"] is not None,
              "independent_slope_binding":right[-1]["input_slope_hex"]==[float(v).hex() for v in segments[1].inputs(segments[1].start)[1]],
              "restored_exact_identity":restored[0].identity==right[0].identity,
              "accepted_predecessor_named":right[-1]["history"]["predecessor_identity"]==left[0].identity,
              "no_reference_duplication":not right[-1]["history"]["reference_embedded"],
              "native_and_tangent_separate":right[-1]["raw"]["origin"]=="segment_initial" and right[-1]["physical_tangent"]["origin"]=="physical_tangent",
              "no_native_import":"sksundae" not in sys.modules}
    log_case(request,{"family":"affine_native_history_readiness","case":case_id,
                      "sample_bytes":len(json.dumps(initial[-1],separators=(",",":")).encode())+1,
                      "reference_bytes":len(json.dumps(history.reference_record,separators=(",",":")).encode())+1,
                      "one_pair_elapsed_s":sample_seconds,"timing_scope":"one nonnative initial pair, includes all quality checks and exact scaling audit; not a trajectory prediction",
                      "initial_state_quality":initial[-1]["state_checks"],"native_trajectory":False},checks)


@pytest.mark.parametrize("case_id", CASES)
def test_affine_native_domain_and_finite_constraint_policy(case_id, request):
    from scripts.benchmarks.coupled_device_prototype import AffineScaledIDAAdapter, affine_state_quality_evidence

    m = affine_model(case_id)
    proposal,_ = affine_pilot_request(case_id)
    adapter = AffineScaledIDAAdapter(m,protocol_from_plan(PLAN,case_id)[0])
    z = np.array(proposal["z0"])
    z[m.layout.offsets["p_m3"].start+1] = -2*m.definition.p_eq/m.S[m.layout.offsets["p_m3"].start+1]
    with pytest.raises(ContractError,match="physical_state_outside_domain"):
        adapter.residual(0.,z,np.zeros_like(z))
    base = affine_state_quality_evidence(m,m.reference,proposal["budgets"])
    u = np.zeros(m.layout.size);u[m.layout.offsets["p_m3"].start]=.02*m.definition.p_eq
    invalid,_ = m.trial(u,0.,[0.,0.])
    changed = affine_state_quality_evidence(m,invalid,proposal["budgets"])
    log_case(request,{"family":"affine_native_domain_readiness","case":case_id,
                      "reference":base,"contact_perturbation":changed,"native_trajectory":False},
             {"initial_finite_g_passes":base["passed"],"bad_contact_fails":not changed["passed"],
              "state_untouched":not changed["state_changed"],"single_constant_factor":m.factorizations==1,
              "constraint_body_half_cell_retained":changed["body_charge_error_bound_C"]>0,
              "no_global_state_certificate":not changed["trajectory_error_certified"]})


@pytest.mark.parametrize("case_id", CASES)
def test_affine_native_charge_projection_and_prefix(case_id, request):
    from scripts.benchmarks.coupled_device_prototype import affine_charge_accounting
    from scripts.benchmarks.precision_prototype import DD

    m = affine_model(case_id)
    proposal,_ = affine_pilot_request(case_id)
    budget = proposal["budgets"]["charge_C"]
    delta = DD(np.array([.2,.1,-.3])*budget,np.array([1.,-1.,1.])*budget*1e-20)
    integrals = [delta+DD(np.full(3,.02*budget)) for _ in range(3)]
    first = affine_charge_accounting(m,delta,integrals,np.zeros(3),np.zeros(3),budget)
    fail = affine_charge_accounting(m,delta,integrals,np.full(3,.99*budget),np.zeros(3),budget)
    log_case(request,{"family":"affine_native_charge_readiness","case":case_id,"first":first,
                      "prefix_failure":fail,"native_trajectory":False},
             {"small_combined_error_passes":first["passed"],"prefix_failure_retained":not fail["passed"],
              "interval_not_confused_with_prefix":fail["checks"]["interval_total"] and not fail["checks"]["prefix_total"],
              "DD_low_words_retained":any(v!=0 for v in first["delta_charge_words"]["low"]),
              "projection_uncertainty_recorded":all(v>=0 for v in first["output_projection_error_estimate_C"])})


@pytest.mark.parametrize("case_id", CASES)
def test_affine_native_admission_blocks_before_backend(case_id, request):
    from scripts.benchmarks.coupled_device_prototype import run_affine_native_pilot

    m = affine_model(case_id)
    proposal,_ = affine_pilot_request(case_id)
    with pytest.raises(ContractError,match="affine_native_pilot_not_admitted"):
        run_affine_native_pilot(m,protocol_from_plan(PLAN,case_id),proposal,{},lambda x:None)
    log_case(request,{"family":"affine_native_admission_readiness","case":case_id,"native_trajectory":False},
             {"no_native_import":"sksundae" not in sys.modules})


@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("boundary", ["zero","equal","above","below","negative_above",
                                       "prefix_equal","prefix_above","reference_equal",
                                       "reference_above","reference_prefix_above"])
def test_affine_native_charge_strict_retained_word_boundaries(case_id,boundary,request):
    from scripts.benchmarks.coupled_device_prototype import affine_charge_accounting
    from scripts.benchmarks.precision_prototype import DD

    m = affine_model(case_id)
    proposal,_ = affine_pilot_request(case_id)
    budget = proposal["budgets"]["charge_C"]
    evidence=[]
    with localcontext() as ctx:
        ctx.prec=220
        b, share = dec(budget),dec(budget/3)
        def oracle_words(value):
            return [dec(h)+dec(l) for h,l in zip(value.hi,value.lo)]
        for component in range(3):
            hi,lo,pd,pu=(np.zeros(3) for _ in range(4))
            ih,il=np.zeros(3),np.zeros(3)
            if boundary in {"equal","above","below","negative_above"}:
                hi[component]=-budget if boundary=="negative_above" else budget
                lo[component]={"equal":0.,"above":1e-40,"below":-1e-40,"negative_above":-1e-40}[boundary]
            if boundary in {"prefix_equal","prefix_above"}:
                pd[component]=budget
                if boundary=="prefix_above":lo[component]=1e-40
            if boundary in {"reference_equal","reference_above","reference_prefix_above"}:
                ih[component]=budget/3 if boundary!="reference_prefix_above" else 0.
                il[component]=0. if boundary=="reference_equal" else 1e-40
                hi,lo=ih.copy(),il.copy()
                if boundary=="reference_prefix_above":pu[component]=budget/3
            delta=DD(hi,lo);integrals=[DD(np.zeros(3)),DD(np.zeros(3)),DD(ih,il)]
            d=oracle_words(delta);values=[oracle_words(v) for v in integrals]
            defects=[abs(a-c) for a,c in zip(d,values[-1])]
            u=[abs(a-c) for a,c in zip(values[-1],values[-2])]
            cd=[dec(a)+c for a,c in zip(pd,defects)]
            cu=[dec(a)+c for a,c in zip(pu,u)]
            expected={"interval_total":all(a+c<=b for a,c in zip(defects,u)),
                      "prefix_total":all(a+c<=b for a,c in zip(cd,cu)),
                      "interval_reference_share":all(v<=share for v in u),
                      "prefix_reference_share":all(v<=share for v in cu)}
            result=affine_charge_accounting(m,delta,integrals,pd,pu,budget)
            assert result["checks"]==expected
            assert result["passed"]==all(expected.values())
            for name,exact in [("cumulative_absolute_defect_C",cd),
                               ("cumulative_quadrature_uncertainty_estimate_C",cu),
                               ("prefix_total_bound_C",[a+c for a,c in zip(cd,cu)])]:
                assert all(dec(actual)>=value for actual,value in zip(result[name],exact))
            evidence.append({"component":component,"delta_words_hex":[float(hi[component]).hex(),float(lo[component]).hex()],
                             "expected":expected,"candidate":result,
                             "oracle":"Decimal220 exact binary-word sums; independent of Fraction candidate"})
    log_case(request,{"family":"affine_strict_charge_gate","case":case_id,"boundary":boundary,
                      "budget_hex":budget.hex(),"components":evidence,"native_trajectory":False},
             {"all_body_and_metal_components":len(evidence)==3,
              "exact_retained_word_comparison":all(not row["candidate"]["display_projections_used_for_acceptance"] for row in evidence)})


@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("family", GATES["points"])
def test_affine_bound_evaluation_reuses_exact_values(case_id,family,monkeypatch,request):
    from scripts.benchmarks.coupled_device_prototype import affine_state_quality_evidence,affine_observation_payload

    m=affine_model(case_id);p,u=affine_point(m,family)
    rate=m.S*(.17+np.sin(np.arange(m.layout.size)));adot=np.array([.1,.3*m.definition.photon_reference])
    proposal,_=affine_pilot_request(case_id)
    raw=m.observe(p,rate,adot)
    tangent_rate=m.tangent_rate(p,adot)
    tangent=m.observe(p,tangent_rate,adot,"physical_tangent")
    quality=affine_state_quality_evidence(m,p,proposal["budgets"])
    bound=m.affine_constraint_error(p)
    actual_evaluate=m.evaluate;count=[0]
    def counted(*args,**kwargs):
        count[0]+=1
        return actual_evaluate(*args,**kwargs)
    monkeypatch.setattr(m,"evaluate",counted)
    evaluation=m.observation_evaluation(p)
    shared_raw=m.observe(p,rate,adot,evaluation=evaluation)
    shared_rate=m.tangent_rate(p,adot,evaluation=evaluation)
    shared_tangent=m.observe(p,shared_rate,adot,"physical_tangent",evaluation=evaluation)
    shared_quality=affine_state_quality_evidence(m,p,proposal["budgets"],evaluation=evaluation)
    shared_bound=m.affine_constraint_error(p,evaluation=evaluation)
    other,_=m.trial(u,p.time+.001,p.inputs)
    with pytest.raises(ContractError,match="foreign_or_stale_observation_evaluation"):
        m.observe(other,rate,adot,evaluation=evaluation)
    with pytest.raises(ContractError,match="foreign_or_stale_observation_evaluation"):
        m.tangent_rate(p,adot,evaluation=replace(evaluation,source_identity="foreign"))
    with pytest.raises(ValueError):evaluation.values.algebraic.high.setflags(write=True)
    with pytest.raises(TypeError):evaluation.values.nodal_rates["n_m3"]=None
    log_case(request,{"family":"bound_affine_evaluation","case":case_id,"point_family":family,
                      "point_identity":p.identity,"full_evaluation_count":count[0],"native_trajectory":False},
             {"one_evaluation":count[0]==1,"raw_words_equal":affine_observation_payload(shared_raw)==affine_observation_payload(raw),
              "tangent_words_equal":affine_observation_payload(shared_tangent)==affine_observation_payload(tangent),
              "rates_bitwise_equal":np.array_equal(shared_rate,tangent_rate),"quality_equal":shared_quality==quality,
              "constraint_bound_equal":shared_bound==bound,"no_mutable_sparse_cache":evaluation.values.rate_jacobian is None})


@pytest.mark.parametrize("case_id", CASES)
def test_affine_native_pair_evaluates_once(case_id,monkeypatch,request):
    from scripts.benchmarks.coupled_device_prototype import AffineDeviceHistory,affine_native_sample

    m=affine_model(case_id);proposal,_=affine_pilot_request(case_id);segments=protocol_from_plan(PLAN,case_id)
    history=AffineDeviceHistory(m)
    snapshot={"time":0.,"z":np.array(proposal["z0"]),"zdot":np.array(proposal["zdot0"]),
              "success":True,"status":0,"message":"synthetic snapshot; no native solver"}
    actual=m.evaluate;count=[0]
    def counted(*args,**kwargs):
        count[0]+=1
        return actual(*args,**kwargs)
    monkeypatch.setattr(m,"evaluate",counted)
    before=time.perf_counter()
    pair=affine_native_sample(m,history,proposal,segments[0],snapshot,m.reference,origin="segment_initial")
    elapsed=time.perf_counter()-before
    restored=history.restore(history.reference_record,pair[-1]["history"],m.reference)
    log_case(request,{"family":"affine_one_evaluation_pair","case":case_id,"full_evaluation_count":count[0],
                      "one_pair_elapsed_s":elapsed,"sample_bytes":len(json.dumps(pair[-1],separators=(",",":")).encode())+1,
                      "raw_pair_point_identity":pair[0].identity,"native_trajectory":False},
             {"one_evaluation":count[0]==1,"full_public_history_restores":restored[0].identity==pair[0].identity,
              "raw_rates_retained":pair[-1]["history"]["raw_solver"] is not None,
              "separate_tangent_origin":pair[-1]["physical_tangent"]["origin"]=="physical_tangent"})

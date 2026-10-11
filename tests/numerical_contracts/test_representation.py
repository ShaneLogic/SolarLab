"""Manufactured representation checks; no genuine device, Poisson or native calls."""
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import (
    ContractError, Layout, Linearization, ONE, PARTICLE, RateView, StateView,
    Support, VariableSpec, VOLT, VOLUME,
)
from scripts.benchmarks.precision_prototype import (
    DoubleArray, FrameInputExpansion, RelativeCoordinates, encode_point, physical_state_words,
)
from scripts.benchmarks.representation_prototype import (
    PhysicalAllowance, PhysicalRepresentation, UnitExpression, rate_values, state_values,
)


def decimal(value):
    value = Fraction(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


@pytest.fixture
def physical():
    layout = Layout((Support("one", "cell", (1,)), Support("two", "cell", (2,))), (
        VariableSpec("n_m3", "carrier", "one", (1,), PARTICLE/VOLUME),
        VariableSpec("phi_V", "potential", "two", (2,), VOLT),
        VariableSpec("f", "trap", "one", (1,), ONE)), ())
    coordinates = RelativeCoordinates(layout, {v.id: "linear" for v in layout.variables})
    reference = coordinates.initial(StateView(layout, [
        ("n_m3", DoubleArray([8.])), ("phi_V", DoubleArray([0., 0.])), ("f", DoubleArray([0.]))]), inputs=[0., 0.])
    values, low = np.zeros(layout.size), np.zeros(layout.size)
    values[layout.offsets["n_m3"]] = 1.
    values[layout.offsets["phi_V"]] = [0.125, -0.25]
    low[layout.offsets["n_m3"]] = 2.**-65
    low[layout.offsets["phi_V"]] = [2.**-70, -2.**-72]
    zeros = np.zeros(layout.size)
    prior, _ = coordinates.trial(reference, FrameInputExpansion((zeros,)*12), 0.125, [0.25, 1.], predecessor=reference)
    point, _ = coordinates.trial(reference, FrameInputExpansion((values, low)+(zeros,)*10), 0.375, [0.375, 1.25], predecessor=prior)
    rates = np.zeros(layout.size); rates[layout.offsets["n_m3"]] = 0.75
    rates[layout.offsets["phi_V"]] = [0.5, -0.75]
    rate = RateView(point, FrameInputExpansion((rates, low/8)+(zeros,)*10), [0.125, -0.5], source_identity="7"*64,
        mapping_identity="manufactured-physical-rate", origin="physical-rate", raw_coordinates=point.y, raw_rate=rates)
    model = SimpleNamespace(layout=layout, reference=reference, source_identity="7"*64,
        trial=lambda value,time,inputs,**kw: coordinates.trial(reference,value,time,inputs,**kw),
        validate=lambda point: None)  # Manufactured public construction, no physical evaluator.
    return model, point, rate


def chart(physical, **kwargs):
    model, _, _ = physical
    return PhysicalRepresentation(model.layout, model.source_identity, model.reference, **kwargs)


def allowance(physical, width=Fraction(1, 1000)):
    model, point, _ = physical
    return PhysicalAllowance(point.identity, model.reference.identity, model.source_identity,
                             model.layout.identity, (width,)*model.layout.size)


def independent_inverse(rep, image, coordinates=None, rates=None):
    z = image.coordinates if coordinates is None else coordinates
    zd = image.coordinate_rate if rates is None else rates
    fields, velocities = [], []
    with localcontext() as ctx:
        ctx.prec = 110
        for i,(name,scale) in enumerate(zip(rep.names,rep.scales)):
            s, x, v = decimal(scale), decimal(float(z[i])), decimal(float(zd[i]))
            if name in rep.references:
                y = decimal(rep.references[name])*(s*x).exp()
                dy = s*y*v
            else:
                y = s*x-(decimal(rep.gauge) if name == rep.potential_field else 0)
                dy = s*v
            fields.append(y); velocities.append(dy)
    return fields, velocities


@pytest.mark.parametrize("mode", ["gauge", "log", "units", "combined"])
def test_actual_forward_inverse_and_original_history_are_distinct(physical, mode):
    units = {"phi_V": UnitExpression(VOLT, Fraction(1,1000), "mV"),
             "f": UnitExpression(ONE, Fraction(1,100), "percent")}
    if mode == "units": units["n_m3"] = UnitExpression(PARTICLE/VOLUME, 10**6, "cm^-3")
    rep = chart(physical, gauge_offset_V=Fraction(3,8) if mode in {"gauge","combined"} else 0,
        reference_densities={"n_m3": 32} if mode in {"log","combined"} else None,
        column_units=units if mode in {"units","combined"} else None)
    model, point, rate = physical
    original, original_rate = encode_point(point), rate.identity
    image = rep.forward(point, rate)
    inverse, expected = rep.inverse(image), independent_inverse(rep,image)
    with localcontext() as ctx:
        ctx.prec = 110
        for values, refs in zip((inverse.state_si,inverse.rate_si),expected):
            for v, exact in zip(values,refs):
                assert abs(decimal(v.center)-exact) <= decimal(v.radius)+Decimal("1e-100")
    assert encode_point(image.point) == original and image.rate.identity == original_rate
    assert image.point is point and image.rate is rate
    assert inverse.inputs_si == tuple(Fraction(float(x)) for x in point.inputs)
    assert inverse.input_rates_si == tuple(Fraction(float(x)) for x in rate.input_rate)
    assert image.contact_potentials_V == (rep.gauge,rep.gauge-Fraction(3,8))
    assert image.contact_rates_V_s == (0,-Fraction(1,8))
    j = model.layout.offsets["f"].start
    assert image.coordinates[j] == image.coordinate_rate[j] == 0
    assert inverse.state_si[j].center == inverse.rate_si[j].center == 0
    assert all(len(words) == 14 for words in physical_state_words(point.state).values())
    # This inverse is an actual public mapped12 trial, never a cached Point.
    restored, restored_rate, _, evidence = rep.physical_trial(model,image)
    assert restored.identity != point.identity
    assert encode_point(restored)["payload"]["authority"]["transition"]["previous_point"] == point.identity
    assert state_values(restored) == tuple(v.center for v in evidence.state_si)
    assert rate_values(restored_rate) == tuple(v.center for v in evidence.rate_si)
    assert encode_point(point) == original


def test_physical_gauge_moves_contacts_and_preserves_fields_and_port_work(physical):
    rep = chart(physical, gauge_offset_V=2)
    _, point, rate = physical
    image = rep.forward(point,rate)
    y, yd = state_values(point), rate_values(rate)
    phi = point.state.layout.offsets["phi_V"]
    original = y[phi.start:phi.stop]
    gauged = rep.physical_expression(point)[phi.start:phi.stop]
    assert gauged==tuple(x+rep.gauge for x in original)
    assert rep.physical_expression(rep.reference)[phi.start:phi.stop]==(rep.gauge,rep.gauge)
    assert gauged[1]-gauged[0] == original[1]-original[0]
    assert gauged[0]-image.contact_potentials_V[0] == original[0]
    assert gauged[1]-image.contact_potentials_V[1] == original[1]+Fraction(3,8)
    # A manufactured displacement/conduction port and voltage work use
    # potential differences and rates, not the absolute potential origin.
    current = (original[1]-original[0])*Fraction(7,2)+(yd[phi.stop-1]-yd[phi.start])*Fraction(5,8)
    gauged_current = (gauged[1]-gauged[0])*Fraction(7,2)+(yd[phi.stop-1]-yd[phi.start])*Fraction(5,8)
    assert current == gauged_current
    assert current*Fraction(3,8) == gauged_current*(image.contact_potentials_V[0]-image.contact_potentials_V[1])
    with pytest.raises(ContractError,match="contact_input_mismatch"):
        rep.inverse(replace(image,contact_potentials_V=(Fraction(),-Fraction(3,8))))


def test_independent_reference_changes_log_law_not_physical_material(physical):
    first, second = chart(physical,reference_densities={"n_m3":2}), chart(physical,reference_densities={"n_m3":64})
    assert first != second
    model, point, rate = physical
    a,b = first.forward(point,rate),second.forward(point,rate)
    i = model.layout.offsets["n_m3"].start
    assert a.coordinates[i] != b.coordinates[i]
    assert a.coordinate_rate[i] == b.coordinate_rate[i]
    with localcontext() as ctx:
        ctx.prec=100
        shift = decimal(float(b.coordinates[i]))-decimal(float(a.coordinates[i]))
        assert abs(shift+(Decimal(32)).ln()) <= Decimal(4)*decimal(np.spacing(abs(a.coordinates[i])))
    assert first.reference is second.reference is model.reference
    for rep,image in ((first,a),(second,b)):
        promise = rep.error_weights(image,allowance(physical))
        assert all(e*3<=d for e,d in zip(promise.representation_debit_si,promise.widths_si))


@pytest.mark.parametrize("mode", ["gauge", "log", "units", "combined"])
def test_original_si_promise_with_one_combined_debit_and_finite_errors(physical,mode):
    kwargs = {}
    if mode in {"gauge","combined"}: kwargs["gauge_offset_V"] = Fraction(3,8)
    if mode in {"log","combined"}: kwargs["reference_densities"] = {"n_m3":32}
    if mode in {"units","combined"}: kwargs["column_units"] = {"phi_V":UnitExpression(VOLT,Fraction(1,1000),"mV")}
    rep=chart(physical,**kwargs); _,point,rate=physical
    image=rep.forward(point,rate); promise=rep.error_weights(image,allowance(physical))
    decoded=rep.inverse(image); original=state_values(point)
    errors=tuple(abs(v.center-y)+v.radius for v,y in zip(decoded.state_si,original))
    assert promise.representation_debit_si == errors
    assert promise.eta == max(e/d for e,d in zip(errors,promise.widths_si))
    assert promise.beta+promise.eta == 1
    probes=[np.array([1.,-1.,1.,-1.])/2, np.array([-1.,1.,-1.,1.])/2]
    probes += [np.eye(4)[j]*sign*1.9 for j in range(4) for sign in (-1,1)]
    for normalized in probes:
        dz=normalized/promise.weights
        # Evaluate at the actual rounded raw coordinates, including addition
        # roundoff in the measured solver error, before checking the implication.
        z=image.coordinates+dz
        assessed=rep.assess_error(image,promise,coordinates=z)
        assert assessed["accepted"] and assessed["combined_physical_error_used_once"]
        assert assessed["original_physical_wrms_upper_squared"]<=1
        actual_dz=tuple(Fraction(float(x))-Fraction(float(y)) for x,y in zip(z,image.coordinates))
        raw=sum((d*Fraction(float(w)))**2 for d,w in zip(actual_dz,promise.weights))/4
        assert raw<=1
        physical_value,_=independent_inverse(rep,image,coordinates=z)
        with localcontext() as ctx:
            ctx.prec=100
            exposure=sum(((v-decimal(y))/decimal(d))**2 for v,y,d in zip(physical_value,original,promise.widths_si))/4
            assert exposure<=1


def test_internal_units_change_actual_coordinates_rates_jacobian_and_weights(physical):
    _,point,rate=physical
    base=chart(physical); units=chart(physical,column_units={"phi_V":UnitExpression(VOLT,Fraction(1,1000),"mV")})
    a,b=base.forward(point,rate),units.forward(point,rate)
    phi=point.state.layout.offsets["phi_V"]
    assert np.array_equal(b.coordinates[phi],1000*a.coordinates[phi])
    assert np.array_equal(b.coordinate_rate[phi],1000*a.coordinate_rate[phi])
    n=point.y.size
    p=Linearization(np.eye(n),np.eye(n)*3,np.ones((n,2)),np.ones((n,2))*2,np.ones(n)*5)
    transformed,rounding=units.pullback(point,rate,p)
    j=phi.start
    assert Fraction(float(transformed.y[j,j]))-Fraction(1,1000) in (-rounding["y"][j*n+j],rounding["y"][j*n+j])
    assert transformed.inputs is p.inputs and transformed.input_rate is p.input_rate and transformed.time is p.time
    wa,wb=base.error_weights(a,allowance(physical)),units.error_weights(b,allowance(physical))
    assert wb.weights[j]<wa.weights[j]/999
    # Reusing unconverted weights would reject an otherwise valid identical
    # physical perturbation by a factor of about1000; equality is not the rule.
    assert (0.5/wb.weights[j])*wa.weights[j]>100


def test_nonlinear_storage_pullback_includes_nonzero_rate_input_and_time(physical):
    rep=chart(physical,reference_densities={"n_m3":32})
    _,point,rate=physical
    y,yd=state_values(point),rate_values(rate); size=len(y)
    i=point.state.layout.offsets["n_m3"].start; j=point.state.layout.offsets["phi_V"].start
    n,nd,phi,a,ad,t=y[i],yd[i],y[j],Fraction(float(point.inputs[0])),Fraction(float(rate.input_rate[0])),Fraction(point.time)
    # Q=n^3/3+a*n^2/2+t*n; R=n*phi+a^2+t^2.
    fy,fd=np.zeros((size,size)),np.zeros((size,size))
    exact_fy=(2*n+a)*nd+n*ad+1-phi; mass=n*n+a*n+t
    fy[i,i],fy[i,j],fd[i,i]=float(exact_fy),float(-n),float(mass)
    pa,pr,pt=np.zeros((size,2)),np.zeros((size,2)),np.zeros(size)
    pa[i,0],pr[i,0],pt[i]=float(n*nd-2*a),float(n*n/2),float(nd-2*t)
    physical_partials=Linearization(fy,fd,pa,pr,pt)
    out,rounding=rep.pullback(point,rate,physical_partials)
    direct=exact_fy*n+mass*nd
    input_arithmetic=abs(Fraction(float(fy[i,i]))-exact_fy)*n+abs(Fraction(float(fd[i,i]))-mass)*abs(nd)
    assert abs(Fraction(float(out.y[i,i]))-direct)<=input_arithmetic+rounding["y"][i*size+i]
    assert abs(Fraction(float(out.y[i,i]))-exact_fy*n)>mass*nd/2
    assert out.inputs is pa and out.input_rate is pr and out.time is pt
    assert rounding["B"][i]==nd and rounding["A"][i]==n
    # Independent direct joint difference at nonzero rates for several cj.
    with localcontext() as ctx:
        ctx.prec=100
        N,ND,PH,A,AD,T=map(decimal,(n,nd,phi,a,ad,t))
        def f(x,v):
            return (x*x+A*x+T)*v+x*x*AD/2+x-x*PH-A*A-T*T
        for cj in (0,3,1000):
            h=Decimal(2)**-40; c=Decimal(cj)
            xp,xm=N*h.exp(),N*(-h).exp()
            vp,vm=xp*(ND/N+c*h),xm*(ND/N-c*h)
            difference=(f(xp,vp)-f(xm,vm))/(2*h)
            expected=decimal(direct+mass*n*cj)
            # The retained100-digit difference resolves the known O(h^2)
            # truncation; this is a manufactured derivative check, not a gate.
            assert abs(difference-expected)<Decimal("1e-12")


def test_binding_domain_units_zero_and_mutation_rejections(physical):
    model,point,rate=physical; rep=chart(physical); image=rep.forward(point,rate)
    with pytest.raises(FrozenInstanceError): rep.gauge=1
    shape=image.coordinates.shape
    with pytest.warns(DeprecationWarning):
        image.coordinates.shape=(2,2)
    assert image.coordinates.shape==shape
    with pytest.raises(ContractError,match="column_dimension"):
        chart(physical,column_units={"n_m3":UnitExpression(VOLT,1,"wrong")})
    with pytest.raises(ContractError,match="log_reference_dimension_or_domain"):
        chart(physical,reference_densities={"phi_V":1})
    with pytest.raises(ContractError,match="log_reference_dimension_or_domain"):
        chart(physical,reference_densities={"n_m3":0})
    with pytest.raises(ContractError,match="foreign_image"):
        chart(physical,gauge_offset_V=1).inverse(image)
    with pytest.raises(ContractError,match="allowance_binding"):
        rep.error_weights(image,replace(allowance(physical),source_identity="8"*64))
    with pytest.raises(ContractError):
        rep.forward(point,RateView(point,rate.values,rate.input_rate,source_identity="8"*64,
            mapping_identity=rate.mapping_identity,origin=rate.origin,raw_coordinates=rate.raw_coordinates,raw_rate=rate.raw_rate))
    with pytest.raises(ContractError,match="arithmetic_share_exceeded"):
        rep.error_weights(image,allowance(physical,Fraction(1,10**100)))
    with pytest.raises(ContractError,match="trial_model_mismatch"):
        rep.physical_trial(SimpleNamespace(source_identity="8"*64),image)
    promise=rep.error_weights(image,allowance(physical))
    z=image.coordinates.copy();z[point.state.layout.offsets["n_m3"].start]+=1
    assert not rep.assess_error(image,promise,coordinates=z)["accepted"]
    with pytest.raises(ContractError,match="weight_image_mismatch"):
        rep.assess_error(image,replace(promise,image_identity="wrong"),coordinates=image.coordinates)


def test_sparse_pullback_preserves_declared_zero_slots_and_units(physical):
    from scripts.benchmarks.contract_prototype import SparseLinearization, SparseStructure
    model,point,rate=physical
    rep=chart(physical,reference_densities={"n_m3":32},
              column_units={"phi_V":UnitExpression(VOLT,Fraction(1,1000),"mV")})
    structure=SparseStructure((4,4),np.tile(np.arange(4),4),np.arange(0,17,4),"manufactured-full-slots",model.layout.identity)
    y=np.array([[1.,0.,2.,0.],[0.,3.,0.,4.],[5.,0.,0.,0.],[0.,0.,0.,6.]])
    yd=np.diag([7.,8.,9.,10.]); pa=np.arange(8).reshape(4,2); pr=pa/4; pt=np.arange(4)
    dense=Linearization(y,yd,pa,pr,pt)
    sparse=SparseLinearization(structure.filled(y.ravel(order="F")),structure.filled(yd.ravel(order="F")),
                               pa,pr,pt,structure,model.source_identity)
    a,ae=rep.pullback(point,rate,dense); b,be=rep.pullback(point,rate,sparse)
    assert b.structure.identity==structure.identity and b.y.nnz==b.ydot.nnz==16
    assert np.array_equal(a.y,b.y.toarray()) and np.array_equal(a.ydot,b.ydot.toarray())
    assert tuple(np.asarray(ae["y"],dtype=object).reshape(4,4).ravel(order="F"))==be["y"]
    with pytest.raises(ContractError,match="linearization_source_mismatch"):
        rep.pullback(point,rate,replace(sparse,source_identity="8"*64))


def test_zero_log_density_rejected_without_floor_and_public_factory(physical):
    from scripts.benchmarks.coupled_device_prototype import AffineCoupledSlab
    model,point,rate=physical
    rep=AffineCoupledSlab.physical_representation(model,reference_densities={"n_m3":32})
    assert rep.reference is model.reference and rep.right_contact_sign==-1
    remainder=np.zeros(4);remainder[model.layout.offsets["n_m3"]]=-8
    zero,_=model.trial(FrameInputExpansion.from_value(remainder),point.time,point.inputs,
                       predecessor=point,transition_representation="paired-endpoints-v1")
    zero_rate=RateView(zero,rate.values,rate.input_rate,source_identity=model.source_identity,
        mapping_identity="manufactured-zero-rate-binding",origin="physical-rate",raw_coordinates=zero.y,raw_rate=rate.raw_rate)
    assert state_values(zero)[model.layout.offsets["n_m3"].start]==0
    with pytest.raises(ContractError,match="log_nonpositive_density"):
        rep.forward(zero,zero_rate)


def test_extreme_unit_roundoff_does_not_silently_change_physical_promise(physical):
    rep=chart(physical,gauge_offset_V=2**55); _,point,rate=physical
    image=rep.forward(point,rate)
    with pytest.raises(ContractError,match="arithmetic_share_exceeded"):
        rep.error_weights(image,allowance(physical))
    with pytest.raises(ContractError,match="invalid_unit_expression"):
        UnitExpression(VOLT,0,"zero-scale")

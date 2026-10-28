"""Exact acb witnesses for local flux reuse; no solver or qualification.

The two reference bodies are copied from committed interval_observation.py
8118af7b (also byte-identical to frozen 20751ff). An explicit baseline path
checks that provenance in the locked review run; ordinary tests stay portable.
"""
import ast
from fractions import Fraction as F
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import textwrap
import time
from types import MethodType, SimpleNamespace as NS

import pytest

flint = pytest.importorskip("flint")
SOURCE = Path(__file__).resolve().parents[2] / "scripts/benchmarks/interval_observation.py"
SPEC = importlib.util.spec_from_file_location("_flux_reuse_candidate", SOURCE)
candidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate
SPEC.loader.exec_module(candidate)
BASELINE_SHA = "8f9c5b1688ed533953b6d25750d44441d19aa9b5deacdc64cc68c5e1e6a020f4"


def old_tangent_metal_flux(self, arithmetic, coordinate, analytic=False):
    """Closed form of the constant Dirichlet Poisson tangent action."""
    carrier, ion = self.fluxes(arithmetic, coordinate, analytic)
    number = arithmetic.number
    total = [j+number(self.q)*f for j, f in zip(carrier, ion, strict=True)]
    charge_rates = [number(-self.q*self.area)*ion[0]]
    charge_rates += [number(self.area)*(a-b) for a, b in zip(total[:-1], total[1:], strict=True)]
    charge_rates += [number(self.q*self.area)*ion[-1]]
    return tuple(sum((-number(w)*b for w, b in zip(weights, charge_rates, strict=True)), number(0))
                 for weights in self.weights)


def old_physical_charge_flux(self, arithmetic, coordinate, analytic=False):
    """Independent body and metal F in A, including both metal terms."""
    a = arithmetic
    carrier, _ = self.fluxes(a, coordinate, analytic)
    body = (a.number(self.area)*(carrier[0]-carrier[-1])
            +a.polynomial(self.capture_charge_rate[0]+self.capture_charge_rate[-1],coordinate))
    metal = self.tangent_metal_flux(a,coordinate,analytic)
    return (body, *(flux+a.polynomial(poly,coordinate) for flux,poly in
                    zip(metal,self.tangent_metal_polynomial,strict=True)))


def check_reference_bodies(path):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASELINE_SHA
    cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == "SlabPathObserver")
    for reference in (old_tangent_metal_flux, old_physical_charge_flux):
        name = reference.__name__.removeprefix("old_")
        original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        copied = ast.parse(textwrap.dedent(inspect.getsource(reference))).body[0]
        assert ast.dump(ast.Module(body=original.body, type_ignores=[])) == ast.dump(ast.Module(body=copied.body, type_ignores=[]))


def observer(*, dynamic, near_zero, old=False, nodes=9):
    """Small exact polynomial state with nonzero, mixed body/metal terms."""
    o = object.__new__(candidate.SlabPathObserver)
    p = candidate.Polynomial
    epsilon = F(1, 2**80) if near_zero else F(1, 16)
    o.q, o.area = F(1602176634, 10**28), F(1, 10**8)
    o.dx = tuple(F(i+4, 10**8) for i in range(nodes-1))
    length = sum(o.dx)
    o.weights = (tuple(sum(o.dx[i:], F(0))/length for i in range(nodes)),
                 tuple(sum(o.dx[:i], F(0))/length for i in range(nodes)))
    o.m = NS(mu_n=F(1, 50), mu_p=F(1, 100), vt=F(1, 40), dynamic=dynamic,
             ion_capacity=F(10**25), diffusion_ion=F(1, 10**15))
    n = tuple(p((F(10**20)*(1+i*epsilon), F(10**19)*epsilon*(i+1), F(10**18)*epsilon)) for i in range(nodes))
    holes = tuple(p((F(2*10**20)*(1-i*epsilon/10), -F(10**19)*epsilon*(i+1))) for i in range(nodes))
    ions = tuple(p((F(3*10**24)*(1+i*epsilon/10), F(10**22)*epsilon*(i+1))) for i in range(nodes))
    o.path = NS(fields={"n_m3": n, "p_m3": holes, "c_m3": ions})
    o.xi = tuple(p((epsilon*(i-3), epsilon/8, -epsilon/32)) for i in range(nodes-1))
    o.dn = tuple(b-a for a,b in zip(n[:-1],n[1:]))
    o.dp = tuple(b-a for a,b in zip(holes[:-1],holes[1:]))
    o.dc = tuple(b-a for a,b in zip(ions[:-1],ions[1:]))
    o.capture_charge_rate = tuple(p((F((-1)**i*(i+1), 10**11), F(i+2,10**12))) for i in range(nodes))
    o.tangent_metal_polynomial = (p((F(3,10**8), F(-2,10**9))), p((F(-5,10**8), F(7,10**9))))
    if old:
        o.tangent_metal_flux = MethodType(old_tangent_metal_flux, o)
        o.physical_charge_flux = MethodType(old_physical_charge_flux, o)
    return o


def ball_words(value):
    """Exact midpoint/radius mantissa-exponent pairs, including imaginary part."""
    assert value.is_finite()
    return tuple(tuple(tuple(int(x) for x in part.man_exp()) for part in (component.mid(), component.rad()))
                 for component in (value.real, value.imag))


def vector_words(values):
    return tuple(ball_words(value) for value in values)


COORDINATES = ("zero", "tiny", "point", "real_interval", "complex_point", "complex_interval")


def coordinate(kind):
    a, b = flint.arb, flint.acb
    return {"zero": lambda: b(0), "tiny": lambda: b(a((1,-100))),
            "point": lambda: b(a((1,-3))),
            "real_interval": lambda: b(a((1,-3),(1,-10))),
            "complex_point": lambda: b(a((1,-3)),a((1,-8))),
            "complex_interval": lambda: b(a((1,-3),(1,-10)),a(0,(1,-12)))}[kind]()


def instrument(o, *, capture=True):
    calls, snapshots = [], []
    original = o.fluxes
    def counted(arithmetic, u, analytic=False):
        calls.append((arithmetic, u, analytic))
        result = original(arithmetic, u, analytic)
        assert all(type(row) is tuple for row in result)
        if capture:
            snapshots.append((result, tuple(vector_words(row) for row in result)))
        return result
    o.fluxes = counted
    return calls, snapshots


def unchanged(snapshots):
    for values, before in snapshots:
        assert tuple(vector_words(row) for row in values) == before


def test_reference_bodies_match_captured_commit():
    path = os.environ.get("SOLARLAB_FLUX_BASELINE_SOURCE")
    if not path:
        pytest.skip("explicit committed baseline artifact required for provenance witness")
    check_reference_bodies(path)


@pytest.mark.parametrize("bits", [128,256,512])
@pytest.mark.parametrize("dynamic", [False,True])
@pytest.mark.parametrize("near_zero", [False,True])
@pytest.mark.parametrize("kind", COORDINATES)
@pytest.mark.parametrize("analytic", [False,True])
def test_exact_ball_equivalence_and_local_reuse(bits, dynamic, near_zero, kind, analytic):
    before_precision = flint.ctx.prec
    old = observer(dynamic=dynamic, near_zero=near_zero, old=True)
    new = observer(dynamic=dynamic, near_zero=near_zero)
    old_calls, old_values = instrument(old)
    new_calls, new_values = instrument(new)
    arithmetic = candidate.BallIntegrator(bits=bits)
    with flint.ctx.workprec(bits):
        u = coordinate(kind); before_coordinate = ball_words(u)
        expected = vector_words(old.physical_charge_flux(arithmetic,u,analytic))
        actual = vector_words(new.physical_charge_flux(arithmetic,u,analytic))
        assert actual == expected
        assert len(set(actual)) == 3
        assert len(old_calls) == 2 and len(new_calls) == 1
        assert all(a is arithmetic and point is u and flag is analytic for a,point,flag in old_calls+new_calls)
        assert ball_words(u) == before_coordinate and flint.ctx.prec == bits
        unchanged(old_values); unchanged(new_values)
        # Ordinary standalone tangent still evaluates flux once and preserves
        # its exact outputs, independently of the optimized physical path.
        old_calls.clear(); new_calls.clear()
        assert vector_words(old.tangent_metal_flux(arithmetic,u,analytic)) == vector_words(new.tangent_metal_flux(arithmetic,u,analytic))
        assert len(old_calls) == len(new_calls) == 1
        assert arithmetic.calls == []
    assert flint.ctx.prec == before_precision


@pytest.mark.parametrize("method", ["physical_charge_flux", "tangent_metal_flux"])
def test_first_flux_exception_is_preserved(method):
    failure = RuntimeError("manufactured flux refusal")
    def refuse(*args):
        raise failure
    for old in (False, True):
        o = observer(dynamic=True, near_zero=True, old=old)
        o.fluxes = refuse
        with pytest.raises(RuntimeError) as caught:
            getattr(o,method)(candidate.BallIntegrator(),flint.acb(0),True)
        assert caught.value is failure


def test_existing_tangent_override_keeps_original_routing():
    old = observer(dynamic=True, near_zero=False, old=True)
    new = observer(dynamic=True, near_zero=False)
    calls = []
    def override(a,u,analytic):
        calls.append((u,analytic))
        return a.number(7),a.number(-3)
    old.tangent_metal_flux = new.tangent_metal_flux = override
    a = candidate.BallIntegrator()
    with flint.ctx.workprec(256):
        u = coordinate("real_interval")
        assert vector_words(old.physical_charge_flux(a,u,True)) == vector_words(new.physical_charge_flux(a,u,True))
    assert calls == [(u,True),(u,True)]


def test_mismatched_flux_tuple_exception_is_unchanged():
    messages = []
    for old in (False,True):
        o = observer(dynamic=True, near_zero=False, old=old)
        o.fluxes = lambda a,u,analytic: ((a.number(1),a.number(2)),())
        with pytest.raises(ValueError) as caught:
            o.physical_charge_flux(candidate.BallIntegrator(),flint.acb(0),False)
        messages.append(str(caught.value))
    assert messages[0] == messages[1]


def bounded_sample(path):
    """One fixed same-input old/new sample, not a G2 replay or speed forecast."""
    check_reference_bodies(os.environ["SOLARLAB_FLUX_BASELINE_SOURCE"])
    definitions = [(d,n,k,a) for d in (False,True) for n in (False,True)
                   for k in ("point","real_interval","complex_interval") for a in (False,True)]
    results, encodings = {}, {}
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(number, frame):
        raise TimeoutError("one flux sample path exceeded 30 seconds")
    signal.signal(signal.SIGALRM, expired)
    try:
        for variant in ("old","new"):
            batch = [(observer(dynamic=d,near_zero=n,old=variant=="old"),k,analytic) for d,n,k,analytic in definitions]
            counters = [instrument(o,capture=False)[0] for o,_,_ in batch]
            arithmetic = candidate.BallIntegrator(bits=256)
            started, cpu = time.monotonic(),time.process_time()
            signal.setitimer(signal.ITIMER_REAL,30)
            encoded = []
            with flint.ctx.workprec(256):
                for _ in range(8):
                    for o,kind,analytic in batch:
                        if time.monotonic()-started >= 30:
                            raise TimeoutError("flux sample budget")
                        encoded.append(vector_words(o.physical_charge_flux(arithmetic,coordinate(kind),analytic)))
            signal.setitimer(signal.ITIMER_REAL,0)
            elapsed = time.monotonic()-started
            assert elapsed < 30
            encodings[variant] = encoded
            results[variant] = {"wall_s":elapsed,"CPU_s":time.process_time()-cpu,"physical_calls":len(encoded),
                "flux_calls":sum(map(len,counters)),"exact_encoding_sha256":hashlib.sha256(json.dumps(encoded).encode()).hexdigest()}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
    assert encodings["old"] == encodings["new"]
    output = {"scope":"manufactured polynomial point/interval function sample only","variants":results,
        "definitions":definitions,"repetitions":8,"precision_bits":256,"exact_encodings_equal":True,
        "encodings":encodings["new"],"qualification":False,"full_observation_or_solver_calls":0,
        "interpretation":"single bounded sample including identical instrumentation; no full-run speedup estimate"}
    Path(path).write_text(json.dumps(output,indent=2)+"\n")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-output",required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(bounded_sample(args.sample_output),indent=2))

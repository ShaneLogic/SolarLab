"""Independent polynomial oracle; no Main/native/NumPy/Arb imports.

Main supplies the existing model's stored SI constants, spacings and volumes.
The executable self-check uses a declared rational algebra witness only, not
SCAPS material values and not a DAE trajectory. No tolerance is accepted here.
Polynomial coefficients are in s=(t-t0)/H, on [0,1].
"""
from fractions import Fraction as F
import json
import resource
import signal
import time
from pathlib import Path


def p(*values):
    return tuple(F.from_float(x) if type(x) is float else F(x) for x in values)


def add(*terms):
    return tuple(sum((t[i] if i < len(t) else F(0) for t in terms), F(0))
                 for i in range(max(map(len, terms))))


def scale(a, x):
    return tuple(F(a)*v for v in x)


def mul(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return tuple(out)


def rate(x, H):
    return tuple(F(i)*x[i]/H for i in range(1, len(x))) or p(0)


def delta(x):
    return sum(x[1:], F(0))


def integral(x, H):
    return H*sum((v/F(i+1) for i, v in enumerate(x)), F(0))


def expect(k, fields):
    """q[C/particle], A[m2], dx[m], volumes[m3], H[s], other SI constants.

    Fields n,p,c have units m-3, f is dimensionless, voltage has V. Every
    supplied value is an exact Fraction of stored data, with no recomputed
    equilibrium or default coefficients. Uniform n,p,c,f and phi_i=-x_i*V/L
    eliminate SG Bernoulli terms and make the vacancy ratio exactly one.
    """
    q, A, H = k['q'], k['area'], k['H']
    L, W = sum(k['dx'], F(0)), sum(k['volumes'], F(0))
    WL, WR = k['volumes'][0], k['volumes'][-1]
    n, h, c, f, v = (fields[key] for key in ('n', 'p', 'c', 'f', 'voltage'))
    one_minus_f = add(p(1), scale(-1, f))
    rn = scale(k['cn'], add(mul(n, one_minus_f), scale(-k['n1'], f)))
    rp = scale(k['cp'], add(mul(h, f), scale(-k['p1'], one_minus_f)))
    capture = add(rn, scale(-1, rp))
    rho = scale(q, add(h, scale(-1, n), c, p(-k['c0']), scale(-k['Nt'], f)))
    J = scale(q/L, mul(add(scale(k['mu_n'], n), scale(k['mu_p'], h)), v))
    Fi = scale(k['D']/(L*k['vt']), mul(c, v))
    C = A*k['epsilon']/L
    cap = [scale(-q*w*k['Nt'], capture) for w in (WL, WR)]
    carrier_storage = [scale(q*w, rate(add(h, scale(-1, n)), H)) for w in (WL, WR)]
    metals = [add(scale(sign*C, v), scale(-w, rho)) for sign, w in ((1, WL), (-1, WR))]
    raw_metal = [rate(x, H) for x in metals]
    raw_cond = [add(scale(sign*A, J), cap[i], carrier_storage[i])
                for i, sign in enumerate((1, -1))]
    tangent_cond = [add(scale(sign*A, J), cap[i]) for i, sign in enumerate((1, -1))]
    tangent_metal = [add(scale(sign*C, rate(v, H)), scale(sign*q*A, Fi), scale(-1, cap[i]))
                     for i, sign in enumerate((1, -1))]
    raw_current = [add(a, b) for a, b in zip(raw_cond, raw_metal)]
    tangent_current = [add(a, b) for a, b in zip(tangent_cond, tangent_metal)]
    body = scale(W, rho)
    charges = [body, *metals]
    raw_rows = [add(*raw_cond), *raw_metal]
    tangent_rows = [add(*tangent_cond), *tangent_metal]
    raw_integrals = [integral(x, H) for x in raw_rows]
    tangent_integrals = [integral(x, H) for x in tangent_rows]
    return {'charge_polynomials_C': charges, 'charge_deltas_C': [delta(x) for x in charges],
            'carrier_current_density_A_m2': J, 'ion_particle_flux_m2_s': Fi,
            'capture_occupancy_rate_s_1': capture, 'raw_charge_integrands_A': raw_rows,
            'tangent_charge_integrands_A': tangent_rows,
            'raw_integrals_C': raw_integrals, 'tangent_integrals_C': tangent_integrals,
            'raw_defects_C': [delta(x)-a for x, a in zip(charges, raw_integrals)],
            'tangent_defects_C': [delta(x)-a for x, a in zip(charges, tangent_integrals)],
            'raw_current_A': raw_current, 'tangent_current_A': tangent_current,
            'raw_minus_tangent_current_A': [add(a, scale(-1, b)) for a, b in zip(raw_current, tangent_current)],
            'ion_inventory_change_particles': W*delta(c),
            'gauss_defect_C': add(*charges),
            'input_rate_V_s': rate(v, H)}


# Independent source: Stage02/NativeMapReviewV1/FractionOracle.py
SOURCE_SHA256 = '6c643ad05c38f875780600c896ca167e4cc2727d396986e8770866c38b5b6474'

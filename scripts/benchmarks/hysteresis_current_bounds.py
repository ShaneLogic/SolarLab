"""Pointwise enclosures for the saved, zero-ion-flux D0/C0 HI device law.

The reference inputs are exact captured binary states/materials. A bounded
positive-series expm1 supplies SG enclosures; production SG/RHS/Poisson are not
called. No trajectory, initial state, integrator or scientific gate is created.
Unknown trajectory density errors stay separate from these pointwise bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import math


PRECISION = 96
SERIES_TERMS = 64
DOWN = Context(prec=PRECISION, rounding=ROUND_FLOOR)
UP = Context(prec=PRECISION, rounding=ROUND_CEILING)


class UnsupportedPoint(ValueError):
    pass


def exact(value) -> Fraction:
    if isinstance(value, bool):
        raise UnsupportedPoint("boolean is not a physical scalar")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
        return Fraction(value)
    value = float(value)
    if not math.isfinite(value):
        raise UnsupportedPoint("finite encoded input required")
    return Fraction.from_float(value)


@dataclass(frozen=True)
class _Interval:
    lo: Decimal
    hi: Decimal

    def __post_init__(self):
        if not self.lo.is_finite() or not self.hi.is_finite() or self.lo > self.hi:
            raise UnsupportedPoint("invalid finite enclosure")

    @classmethod
    def of(cls, value):
        if isinstance(value, cls):
            return value
        q = exact(value)
        n, d = Decimal(q.numerator), Decimal(q.denominator)
        return cls(DOWN.divide(n, d), UP.divide(n, d))

    def __add__(self, other):
        b = self.of(other)
        return self.__class__(DOWN.add(self.lo, b.lo), UP.add(self.hi, b.hi))

    __radd__ = __add__

    def __neg__(self):
        return self.__class__(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + (-self.of(other))

    def __rsub__(self, other):
        return self.of(other) - self

    def __mul__(self, other):
        b = self.of(other)
        ends = ((self.lo, b.lo), (self.lo, b.hi), (self.hi, b.lo), (self.hi, b.hi))
        return self.__class__(min(DOWN.multiply(a, c) for a, c in ends),
                              max(UP.multiply(a, c) for a, c in ends))

    __rmul__ = __mul__

    def __truediv__(self, other):
        b = self.of(other)
        if b.lo <= 0 <= b.hi:
            raise UnsupportedPoint("division enclosure includes zero")
        inverse = self.__class__(DOWN.divide(Decimal(1), b.hi), UP.divide(Decimal(1), b.lo))
        return self * inverse

    def absolute_upper(self):
        return max(self.lo.copy_abs(), self.hi.copy_abs())

    def record(self):
        return [str(self.lo), str(self.hi)]


def upper_float(value) -> float:
    value = exact(value)
    result = float(value)
    if not math.isfinite(result):
        raise UnsupportedPoint("bound does not fit a finite reported float")
    return math.nextafter(result, math.inf) if exact(result) < value else result


def error_upper(value, enclosure: _Interval) -> Decimal:
    return (_Interval.of(value)-enclosure).absolute_upper()


def _expm1_bounds(x: Fraction) -> _Interval:
    """Positive Taylor terms and a geometric tail, followed by exact identities.

    Scale |x| to u<=1/8. After term N, the remaining series is bounded by
    term_(N+1)/(1-u/(N+2)). Directed decimal operations enclose every step.
    expm1(2u)=r*(r+2), expm1(-u)=-r/(1+r), so no exp implementation is trusted.
    """
    x = exact(x)
    if not x:
        return _Interval.of(0)
    if abs(x) > 8192:
        raise UnsupportedPoint("SG argument exceeds this bounded arithmetic domain")
    u, squarings = abs(x), 0
    while u > Fraction(1, 8):
        u /= 2
        squarings += 1
    ub = _Interval.of(u)
    term = total = ub
    for k in range(2, SERIES_TERMS+1):
        term = term*ub/k
        total = total+term
    tail = (term*ub/(SERIES_TERMS+1))/(1-ub/(SERIES_TERMS+2))
    result = _Interval(total.lo, (total+tail).hi)
    for _ in range(squarings):
        result = result*(result+2)
    return -result/(1+result) if x < 0 else result


def bernoulli_bounds(x: Fraction) -> _Interval:
    x = exact(x)
    return _Interval.of(1) if not x else _Interval.of(x)/_expm1_bounds(x)


def poisson_exact(C, widths, rho, right, left=Fraction(0)):
    """Exact linear map for the encoded face coefficients and volume widths."""
    C, widths, rho = tuple(map(exact, C)), tuple(map(exact, widths)), tuple(map(exact, rho))
    if (len(C)+1 != len(rho) or len(widths)+2 != len(rho)
            or any(c <= 0 for c in C) or any(w <= 0 for w in widths)):
        raise UnsupportedPoint("invalid Poisson geometry")
    prefix = [Fraction(0)]
    for r, width in zip(rho[1:-1], widths, strict=True):
        prefix.append(prefix[-1]+r*width)
    resistance = sum((1/c for c in C), Fraction(0))
    d0 = (exact(left)-exact(right)-sum((s/c for s, c in zip(prefix, C, strict=True)), Fraction(0)))/resistance
    displacement = [d0+s for s in prefix]
    phi = [exact(left)]
    for d, c in zip(displacement, C, strict=True):
        phi.append(phi[-1]-d/c)
    return phi, displacement


def poisson_rate_bounds(C, widths, rho_rate, voltage_rate, polarity):
    """Enclose differentiated Poisson, including the applied-voltage derivative."""
    prefix = [_Interval.of(0)]
    for r, width in zip(rho_rate[1:-1], widths, strict=True):
        prefix.append(prefix[-1]+r*width)
    resistance = sum((1/exact(c) for c in C), Fraction(0))
    d0 = (_Interval.of(polarity*exact(voltage_rate))-sum((s/c for s, c in zip(prefix, C, strict=True)), _Interval.of(0)))/resistance
    return [d0+s for s in prefix]


def density_to_potential_radius(C, widths, n_radius, p_radius, ion_radius,
                                charge, *, left_radius=0, right_radius=0):
    """Positive Poisson inverse propagates an explicit physical density box."""
    if n_radius is None or p_radius is None or ion_radius is None:
        raise UnsupportedPoint("trajectory density error radii are unknown")
    radii = [list(map(exact, r)) for r in (n_radius, p_radius, ion_radius)]
    if (any(v < 0 for row in radii for v in row)
            or exact(left_radius) < 0 or exact(right_radius) < 0 or exact(charge) <= 0):
        raise UnsupportedPoint("nonnegative uncertainty radii required")
    rho_radius = [exact(charge)*(a+b+c) for a,b,c in zip(*radii,strict=True)]
    phi, _ = poisson_exact(C,widths,rho_radius,exact(right_radius),exact(left_radius))
    if any(x < 0 for x in phi):
        raise ArithmeticError("positive Poisson inverse violated")
    return phi


def current_change_from_state_box(C, sensitivity, *, voltage_rate_radius):
    """A bound for the self-consistent semidiscrete current, not a trajectory estimate."""
    if voltage_rate_radius is None or exact(voltage_rate_radius) < 0:
        raise UnsupportedPoint("explicit voltage-derivative uncertainty required")
    resistance = sum((1/exact(c) for c in C), Fraction(0))
    total = _Interval.of(voltage_rate_radius)
    for c, row in zip(C,sensitivity,strict=True):
        bound = _Interval.of(Decimal(row['electron']['box_current_change_upper_A_m2']))
        bound = bound+Decimal(row['hole']['box_current_change_upper_A_m2'])
        total = total+bound/exact(c)
    return (total/resistance).hi


def sg_bounds(phi, n, p, dx, dn, dp, vt, charge, polarity):
    """Homogeneous MB SG law, using B(-x)=B(x)+x and exact density differences."""
    if (polarity not in (-1, 1) or exact(vt) <= 0 or exact(charge) <= 0
            or any(exact(w) <= 0 for w in dx)
            or any(exact(d) < 0 for d in (*dn, *dp))):
        raise UnsupportedPoint("positive SG geometry and nonnegative diffusivity required")
    electron, hole = [], []
    for i, width in enumerate(dx):
        xi = (exact(phi[i+1])-exact(phi[i]))/exact(vt)
        b = bernoulli_bounds(xi)
        scale = polarity*exact(charge)/exact(width)
        electron.append(scale*exact(dn[i])*(b*(exact(n[i])-exact(n[i+1]))+xi*exact(n[i])))
        hole.append(scale*exact(dp[i])*(b*(exact(p[i+1])-exact(p[i]))+xi*exact(p[i+1])))
    return electron, hole


def sg_sensitivity_bounds(phi, n, p, dx, dn, dp, vt, charge, phi_radius,
                          *, n_radius, p_radius):
    """A box bound; density radii must be explicit for a trajectory-error claim.

    B>0 and -1<B'<0 follow from B(x)=1/integral_0^1 exp(tx)dt and
    B(-x)=B(x)+x. Thus the field derivative is a convex combination of the
    two nonnegative densities. This bounds phi and density effects jointly.
    Missing density radii are rejected; exact-saved-state callers pass explicit
    zeros without claiming that the physical trajectory error is zero.
    """
    if n_radius is None or p_radius is None:
        raise UnsupportedPoint("trajectory density error radii are unknown")
    nr, pr = list(map(exact, n_radius)), list(map(exact, p_radius))
    er = list(map(exact, phi_radius))
    if (len(nr) != len(n) or len(pr) != len(p) or len(er) != len(phi)
            or any(e < 0 for e in nr+pr+er)
            or any(exact(x)-e < 0 for values, radii in ((n, nr), (p, pr)) for x, e in zip(values, radii, strict=True))):
        raise UnsupportedPoint("nonnegative physical density box required")
    records = []
    for i, width in enumerate(dx):
        xi = (exact(phi[i+1])-exact(phi[i]))/exact(vt)
        radius_xi = (er[i]+er[i+1])/exact(vt)
        bplus = bernoulli_bounds(xi-radius_xi).hi
        bminus = bernoulli_bounds(-(xi+radius_xi)).hi
        species = []
        for values, radii, diffusivity, hole in ((n, nr, dn, False), (p, pr, dp, True)):
            k = exact(charge)*exact(diffusivity[i])/exact(width)
            left = _Interval.of(k)*(bplus if hole else bminus)
            right = _Interval.of(k)*(bminus if hole else bplus)
            phi_coefficient = _Interval.of(k*max(exact(values[i])+radii[i], exact(values[i+1])+radii[i+1])/exact(vt))
            bound = left*radii[i]+right*radii[i+1]+phi_coefficient*(er[i]+er[i+1])
            species.append({'left_density_coefficient_upper':str(left.hi),
                            'right_density_coefficient_upper':str(right.hi),
                            'each_phi_coefficient_upper':str(phi_coefficient.hi),
                            'box_current_change_upper_A_m2':str(bound.hi)})
        records.append({'electron':species[0], 'hole':species[1]})
    return records


def qualify_point(point: dict, arrays: dict, packet: dict, charge: float) -> dict:
    """Bound one represented snapshot against its mathematical semidiscrete law."""
    n, p, ions = (list(map(exact, point[k])) for k in ('n_m3', 'p_m3', 'positive_ions_m3'))
    count = len(n)
    a = lambda name: list(map(exact, arrays['material.'+name]))
    C, widths = a('poisson_factor.C'), a('poisson_factor.h_cell')
    x = arrays['initial.x_m']
    dx = [exact(float(x[i+1])-float(x[i])) for i in range(count-1)]
    source_widths = [exact(.5*(float(dx[i])+float(dx[i+1]))) for i in range(count-2)]
    if widths != source_widths or widths != a('dx_cell')[1:-1]:
        raise UnsupportedPoint("Poisson and continuity do not share encoded control volumes")
    pol = point['junction_polarity']
    if (pol not in (-1, 1) or packet['has_dynamic_traps_or_negative_ions']
            or any(a('chi')) or any(a('Eg')) or any(a('N_t_node'))
            or any(v <= 0 for v in n+p) or any(v < 0 for v in ions)
            or any(v is not None for v in packet['contacts_and_potential']['effective_S'].values())):
        raise UnsupportedPoint("only positive bulk MB D0/C0 snapshots with pinned contacts are covered")
    if any(a('D_ion_face')) and (any(ions) or any(a('P_ion0'))):
        raise UnsupportedPoint("nonzero mobile-ion flux needs a separate bound")
    if list(map(exact, point['state'])) != n+p+ions:
        raise UnsupportedPoint("snapshot and packed state differ")
    q, vt = exact(charge), exact(packet['contacts_and_potential']['V_T_device'])
    rho = [q*(pi-ni+ci-bg-na+nd) for ni,pi,ci,bg,na,nd in zip(n,p,ions,a('P_ion0'),a('N_A'),a('N_D'),strict=True)]
    right = exact(packet['contacts_and_potential']['V_bi_bc'])-pol*exact(point['voltage_V'])
    phi, _ = poisson_exact(C, widths, rho, right)
    phi_radius = [abs(v-exact(s)) for v,s in zip(phi,point['phi_V'],strict=True)]
    jn, jp = sg_bounds(phi,n,p,dx,a('D_n_face'),a('D_p_face'),vt,q,pol)
    jn_stored_phi, jp_stored_phi = sg_bounds(point['phi_V'],n,p,dx,a('D_n_face'),a('D_p_face'),vt,q,pol)
    sensitivity = sg_sensitivity_bounds(point['phi_V'],n,p,dx,a('D_n_face'),a('D_p_face'),vt,q,phi_radius,
                                        n_radius=[0]*count, p_radius=[0]*count)
    jc = [en+hp for en,hp in zip(jn,jp,strict=True)]
    rho_rate = [_Interval.of(0)]+[pol*(jc[i+1]-jc[i])/widths[i] for i in range(count-2)]+[_Interval.of(0)]
    dr = poisson_rate_bounds(C,widths,rho_rate,point['voltage_rate_V_s'],pol)
    phi_rate = [_Interval.of(0)]
    for value, capacitance in zip(dr, C, strict=True):
        phi_rate.append(phi_rate[-1]-value/capacitance)
    endpoint_rate = exact(-pol*exact(point['voltage_rate_V_s']))
    if not exact(phi_rate[-1].lo) <= endpoint_rate <= exact(phi_rate[-1].hi):
        raise ArithmeticError("differentiated voltage constraint not enclosed")

    # Independent exact SRH/radiative/Auger source and per-carrier continuity.
    # G-R is shared; it cancels algebraically in charge, rather than relying on
    # cancellation of two rounded huge carrier rates.
    material = {name:a(name) for name in ('ni_sq','tau_n','tau_p','n1','p1','B_rad','C_n','C_p','G_optical')}
    rn, rp = [_Interval.of(0)], [_Interval.of(0)]
    for i in range(1,count-1):
        denominator = material['tau_p'][i]*(n[i]+material['n1'][i])+material['tau_n'][i]*(p[i]+material['p1'][i])
        if denominator <= 0:
            raise UnsupportedPoint("nonpositive SRH denominator")
        excess = n[i]*p[i]-material['ni_sq'][i]
        R = excess/denominator + material['B_rad'][i]*excess + (material['C_n'][i]*n[i]+material['C_p'][i]*p[i])*excess
        G = material['G_optical'][i] if point['illuminated'] else Fraction(0)
        rn.append(-pol*(jn[i]-jn[i-1])/(q*widths[i-1])+(G-R))
        rp.append(pol*(jp[i]-jp[i-1])/(q*widths[i-1])+(G-R))
        recombined = q*(rp[-1]-rn[-1])
        if max(recombined.lo,rho_rate[i].lo) > min(recombined.hi,rho_rate[i].hi):
            raise ArithmeticError("independent carrier/charge enclosures disagree")
    rn.append(_Interval.of(0)); rp.append(_Interval.of(0))

    yd = list(map(exact, point['ydot_m3_s']))
    if any(yd[2*count:]) or any(point['J_ion_A_m2']):
        raise UnsupportedPoint("saved ion rate/current is not zero")
    encoded_charge_rate = [q*(yd[i+count]-yd[i]+yd[i+2*count]) for i in range(count)]
    saved_jc = [exact(en)+exact(hp)+exact(ion) for en,hp,ion in zip(point['J_n_A_m2'],point['J_p_A_m2'],point['J_ion_A_m2'],strict=True)]
    encoded_flux_rate = [Fraction(0)]+[pol*(saved_jc[i+1]-saved_jc[i])/widths[i] for i in range(count-2)]+[Fraction(0)]
    faces = []
    for i in range(count-1):
        arithmetic = [error_upper(point['J_n_A_m2'][i],jn_stored_phi[i]),error_upper(point['J_p_A_m2'][i],jp_stored_phi[i])]
        potential = [Decimal(sensitivity[i][key]['box_current_change_upper_A_m2']) for key in ('electron','hole')]
        component_errors = [error_upper(point['J_n_A_m2'][i],jn[i]),error_upper(point['J_p_A_m2'][i],jp[i])]
        for actual_error, ae, pe in zip(component_errors,arithmetic,potential,strict=True):
            # Each compared reference has its own interval radius. The very
            # small radii are retained instead of assuming exact decimal values.
            allowance = UP.add(UP.add(ae,pe),UP.add(UP.subtract(jn[i].hi,jn[i].lo),UP.subtract(jp[i].hi,jp[i].lo)))
            if actual_error > allowance:
                raise ArithmeticError("SG sensitivity bound does not cover the reference enclosure")
        d_error = error_upper(point['Ddot_A_m2'][i],dr[i])
        current = exact(float(point['J_cond_A_m2'][i])-pol*float(point['Ddot_A_m2'][i]))
        reference_current = jc[i]-pol*dr[i]
        aggregation = abs(current-(exact(point['J_cond_A_m2'][i])-pol*exact(point['Ddot_A_m2'][i])))
        composed = UP.add(error_upper(point['J_cond_A_m2'][i],jc[i]),d_error)
        composed = UP.add(composed, _Interval.of(aggregation).hi)
        direct = error_upper(current,reference_current)
        if direct > UP.add(composed,UP.subtract(reference_current.hi,reference_current.lo)):
            raise ArithmeticError("joint current bound failed")
        faces.append({'Jn_reference_A_m2':jn[i].record(),'Jp_reference_A_m2':jp[i].record(),
                      'Ddot_reference_A_m2':dr[i].record(),'Jtotal_reference_A_m2':reference_current.record(),
                      'SG_arithmetic_upper_A_m2':list(map(str,arithmetic)),
                      'SG_phi_effect_upper_A_m2':list(map(str,potential)),
                      'sensitivity':sensitivity[i],
                      'Ddot_error_upper_A_m2':str(d_error),'total_current_error_upper_A_m2':str(direct),
                      'conservative_composed_current_error_upper_A_m2':str(composed)})
    rate_errors = {
        'stored_rho_dot':max(error_upper(v,b) for v,b in zip(point['rho_dot_C_m3_s'],rho_rate,strict=True)),
        'exact_charge_of_stored_ydot':max(error_upper(v,b) for v,b in zip(encoded_charge_rate,rho_rate,strict=True)),
        'exact_divergence_of_stored_conduction':max(error_upper(v,b) for v,b in zip(encoded_flux_rate,rho_rate,strict=True))}
    return {'phase':point['phase_id'],'time_s':point['time_s'],'event_side':point['event_side'],
            'voltage_V':point['voltage_V'],'voltage_rate_V_s':point['voltage_rate_V_s'],
            'reference_input_meaning':'exact saved binary state/materials; mathematical charge and boundary arithmetic; encoded face widths',
            'phi_error_upper_V':upper_float(max(phi_radius)),
            'rho_aggregation_error_upper_C_m3':upper_float(max(abs(r-exact(s)) for r,s in zip(rho,point['rho_C_m3'],strict=True))),
            'right_boundary_roundoff_V':str(abs(right-exact(point['phi_V'][-1]))),
            'max_phi_dot_error_upper_V_s':upper_float(max(error_upper(v,b) for v,b in zip(point['phi_dot_V_s'],phi_rate,strict=True))),
            'carrier_RHS_error_upper_m3_s':{'n':upper_float(max(error_upper(v,b) for v,b in zip(yd[:count],rn,strict=True))),
                                          'p':upper_float(max(error_upper(v,b) for v,b in zip(yd[count:2*count],rp,strict=True)))},
            'distinct_charge_rate_errors_upper_C_m3_s':{k:upper_float(v) for k,v in rate_errors.items()},
            'faces':faces,'max_Ddot_error_upper_A_m2':max(upper_float(Decimal(f['Ddot_error_upper_A_m2'])) for f in faces),
            'max_total_current_error_upper_A_m2':max(upper_float(Decimal(f['total_current_error_upper_A_m2'])) for f in faces),
            'left_current_error_upper_A_m2':upper_float(Decimal(faces[0]['total_current_error_upper_A_m2'])),
            'density_error_radii_for_pointwise_reference':'zero only by exact-saved-input definition; trajectory radii are unknown',
            'continuous_trajectory_error_bound':None,'reference_eligibility':False}

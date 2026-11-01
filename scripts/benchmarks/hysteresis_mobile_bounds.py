"""Independent enclosures for exact saved Dref inputs, not trajectory errors.

The caller supplies the unchanged, source-bound hysteresis_current_bounds
module as ``base``. No production transport, RHS or Poisson function is called.
The actual captured branch is single-positive-ion diffusion-only steric SG.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction


LOG_TERMS = 64
CLIP_UPPER_HEX = "0x1.ffffde7210be9p-1"


def _positive_log_series(base, t):
    """2*atanh(t), 0<=t<=1/3, with an explicit geometric remainder."""
    if not 0 <= t <= Fraction(1, 3):
        raise base.UnsupportedPoint("logarithm series argument outside bound")
    if not t:
        return base._Interval.of(0)
    x = base._Interval.of(t)
    square, term, total = x*x, x, x
    for index in range(1, LOG_TERMS):
        term = term*square
        total = total+term/(2*index+1)
    # All omitted positive terms have ratio at most t^2; their denominators
    # are at least 2*N+1. Outward operations retain the complete remainder.
    tail = (term*square)/(2*LOG_TERMS+1)/(1-square)
    return 2*base._Interval(total.lo, (total+tail).hi)


def log_bounds(base, value):
    """Bound log of a positive rational using exact powers-of-two reduction."""
    value = base.exact(value)
    if value <= 0:
        raise base.UnsupportedPoint("positive logarithm input required")
    if value < 1:
        return -log_bounds(base, 1/value)
    shift = value.numerator.bit_length()-value.denominator.bit_length()
    if shift > 32:
        raise base.UnsupportedPoint("logarithm input exceeds the declared material domain")
    scaled = value/Fraction(1 << shift)
    if scaled < 1:
        shift -= 1
        scaled *= 2
    result = _positive_log_series(base, (scaled-1)/(scaled+1))
    return result if not shift else result+shift*_positive_log_series(base, Fraction(1, 3))


def _bernoulli_interval(base, value):
    # B is decreasing on the real axis; reuse the independently bounded
    # scalar Bernoulli at both exact decimal endpoints.
    return base._Interval(base.bernoulli_bounds(base.exact(value.hi)).lo,
                          base.bernoulli_bounds(base.exact(value.lo)).hi)


def mobile_flux_bounds(base, phi, ions, dx, diffusion, limits, vt, clip_upper):
    """Positive particle flux, including the captured vacancy-log drive."""
    exact, interval = base.exact, base._Interval.of
    phi, ions = list(map(exact, phi)), list(map(exact, ions))
    dx, diffusion, limits = [list(map(exact, row)) for row in (dx, diffusion, limits)]
    count, vt, clip_upper = len(ions), exact(vt), exact(clip_upper)
    if (count < 2 or len(phi) != count or len(limits) != count
            or len(dx) != count-1 or len(diffusion) != count-1
            or vt <= 0 or not 0 < clip_upper < 1 or any(c < 0 for c in ions)
            or any(d < 0 for d in diffusion) or any(w <= 0 for w in dx)
            or any(limit <= 0 for limit in limits)):
        raise base.UnsupportedPoint("invalid single-ion physical inputs")
    vacancies = [1-min(max(c/limit, Fraction(0)), clip_upper)
                 for c, limit in zip(ions, limits, strict=True)]
    fluxes, drives = [], []
    for i, width in enumerate(dx):
        # mu_ex,right-mu_ex,left = log(vacancy_left/vacancy_right).
        drive = interval((phi[i+1]-phi[i])/vt)+log_bounds(base, vacancies[i]/vacancies[i+1])
        b = _bernoulli_interval(base, drive)
        fluxes.append(diffusion[i]/width*(b*(ions[i]-ions[i+1])-drive*ions[i+1]))
        drives.append(drive)
    return fluxes, drives, vacancies


def _carrier_rates(base, arrays, n, p, jn, jp, widths, q, polarity, illuminated):
    interval, exact = base._Interval.of, base.exact
    material = {name: list(map(exact, arrays["material."+name])) for name in
                ("ni_sq", "tau_n", "tau_p", "n1", "p1", "B_rad", "C_n", "C_p", "G_optical")}
    rn, rp = [interval(0)], [interval(0)]
    for i in range(1, len(n)-1):
        denominator = material["tau_p"][i]*(n[i]+material["n1"][i])+material["tau_n"][i]*(p[i]+material["p1"][i])
        if denominator <= 0:
            raise base.UnsupportedPoint("nonpositive SRH denominator")
        excess = n[i]*p[i]-material["ni_sq"][i]
        recombination = excess/denominator+material["B_rad"][i]*excess+(material["C_n"][i]*n[i]+material["C_p"][i]*p[i])*excess
        generation = material["G_optical"][i] if illuminated else Fraction(0)
        rn.append(-polarity*(jn[i]-jn[i-1])/(q*widths[i-1])+generation-recombination)
        rp.append(polarity*(jp[i]-jp[i-1])/(q*widths[i-1])+generation-recombination)
    return rn+[interval(0)], rp+[interval(0)]


def qualify_point(base, point, arrays, packet, charge):
    """Enclose the represented mobile state under its fixed semidiscrete law."""
    exact, interval = base.exact, base._Interval.of
    a = lambda name: list(map(exact, arrays["material."+name]))
    n, p, ions = [list(map(exact, point[key])) for key in ("n_m3", "p_m3", "positive_ions_m3")]
    count = len(n)
    if count < 2 or len(p) != count or len(ions) != count or list(map(exact, point["state"])) != n+p+ions:
        raise base.UnsupportedPoint("packed state or species shape differs")
    transport, contact = packet["ion_transport"], packet["contacts_and_potential"]
    polarity = point["junction_polarity"]
    if (transport["steric_diffusion_only"] is not True or transport["species"] != "single_positive"
            or transport["effective_shared_site"] is not False
            or transport["clip_upper_binary64"] != CLIP_UPPER_HEX
            or transport["external_particle_flux"] != "zero_both_ends"
            or packet["has_dynamic_traps_or_negative_ions"]
            or point.get("negative_ions_m3") is not None or point.get("trap_occupancy") is not None
            or type(polarity) is not int or polarity not in (-1, 1) or polarity != contact["junction_polarity"]
            or type(point["illuminated"]) is not bool
            or contact["left_phi_V"] != 0 or any(value is not None for value in contact["effective_S"].values())
            or any(a("chi")) or any(a("Eg")) or any(a("N_t_node"))
            or any(value <= 0 for value in n+p) or any(value < 0 for value in ions)):
        raise base.UnsupportedPoint("outside the captured single-ion MB Dref source domain")
    q, vt = exact(charge), exact(contact["V_T_device"])
    if q <= 0 or vt <= 0:
        raise base.UnsupportedPoint("positive charge and thermal voltage required")
    x = arrays["initial.x_m"]
    dx = [exact(float(x[i+1])-float(x[i])) for i in range(count-1)]
    widths, coefficients = a("poisson_factor.h_cell"), a("poisson_factor.C")
    original_interior = [exact(.5*(float(dx[i])+float(dx[i+1]))) for i in range(count-2)]
    if widths != original_interior or widths != a("dx_cell")[1:-1]:
        raise base.UnsupportedPoint("encoded Poisson and continuity widths differ")
    ion_widths = [dx[0]]+original_interior+[dx[-1]]
    rho = [q*(hp-en+ion-bg-na+nd) for en,hp,ion,bg,na,nd in
           zip(n,p,ions,a("P_ion0"),a("N_A"),a("N_D"),strict=True)]
    right = exact(contact["V_bi_bc"])-polarity*exact(point["voltage_V"])
    phi, _ = base.poisson_exact(coefficients, widths, rho, right)
    phi_error = [abs(v-exact(s)) for v,s in zip(phi,point["phi_V"],strict=True)]
    jn, jp = base.sg_bounds(phi,n,p,dx,a("D_n_face"),a("D_p_face"),vt,q,polarity)
    sn, sp = base.sg_bounds(point["phi_V"],n,p,dx,a("D_n_face"),a("D_p_face"),vt,q,polarity)
    clip = exact(float.fromhex(CLIP_UPPER_HEX))
    diffusion, limits = a("D_ion_face"), a("P_lim_node")
    particle, drives, vacancies = mobile_flux_bounds(base,phi,ions,dx,diffusion,limits,vt,clip)
    stored_phi_particle, _, _ = mobile_flux_bounds(base,point["phi_V"],ions,dx,diffusion,limits,vt,clip)
    ji = [-polarity*q*value for value in particle]
    si = [-polarity*q*value for value in stored_phi_particle]
    total = [en+hp+ion for en,hp,ion in zip(jn,jp,ji,strict=True)]
    padded = [interval(0)]+particle+[interval(0)]
    ion_rate = [-(padded[i+1]-padded[i])/width for i,width in enumerate(ion_widths)]
    rho_rate = [q*ion_rate[0]]+[polarity*(total[i+1]-total[i])/widths[i] for i in range(count-2)]+[q*ion_rate[-1]]
    rn, rp = _carrier_rates(base,arrays,n,p,jn,jp,widths,q,polarity,point["illuminated"])
    for en,hp,ion,charge_rate in zip(rn,rp,ion_rate,rho_rate,strict=True):
        combined = q*(hp-en+ion)
        if max(combined.lo,charge_rate.lo) > min(combined.hi,charge_rate.hi):
            raise ArithmeticError("independent species and charge enclosures disagree")
    d_rate = base.poisson_rate_bounds(coefficients,widths,rho_rate,point["voltage_rate_V_s"],polarity)
    phi_rate = [interval(0)]
    for d,c in zip(d_rate,coefficients,strict=True):
        phi_rate.append(phi_rate[-1]-d/c)
    endpoint = -polarity*exact(point["voltage_rate_V_s"])
    if not exact(phi_rate[-1].lo) <= endpoint <= exact(phi_rate[-1].hi):
        raise ArithmeticError("voltage-rate boundary not enclosed")
    carrier_sensitivity = base.sg_sensitivity_bounds(point["phi_V"],n,p,dx,a("D_n_face"),a("D_p_face"),vt,q,phi_error,n_radius=[0]*count,p_radius=[0]*count)
    faces = []
    for i in range(count-1):
        arithmetic = [base.error_upper(point[key][i],bound) for key,bound in
                      zip(("J_n_A_m2","J_p_A_m2","J_ion_A_m2"),(sn[i],sp[i],si[i]),strict=True)]
        ion_phi_effect = interval(q*diffusion[i]*max(ions[i],ions[i+1])/(dx[i]*vt))*(phi_error[i]+phi_error[i+1])
        potential = [Decimal(carrier_sensitivity[i][key]["box_current_change_upper_A_m2"]) for key in ("electron","hole")]+[ion_phi_effect.hi]
        for key,bound,ae,pe,stored_bound in zip(("J_n_A_m2","J_p_A_m2","J_ion_A_m2"),(jn[i],jp[i],ji[i]),arithmetic,potential,(sn[i],sp[i],si[i]),strict=True):
            allowance = interval(ae)+pe+interval(bound.hi)-bound.lo+interval(stored_bound.hi)-stored_bound.lo
            if base.error_upper(point[key][i],bound) > allowance.hi:
                raise ArithmeticError("component potential sensitivity fails to enclose error")
        current = exact(float(point["J_cond_A_m2"][i])-polarity*float(point["Ddot_A_m2"][i]))
        reference = total[i]-polarity*d_rate[i]
        aggregation = abs(current-(exact(point["J_cond_A_m2"][i])-polarity*exact(point["Ddot_A_m2"][i])))
        displacement_error = base.error_upper(point["Ddot_A_m2"][i],d_rate[i])
        composed = interval(base.error_upper(point["J_cond_A_m2"][i],total[i]))+displacement_error+aggregation
        direct = base.error_upper(current,reference)
        if direct > (composed+interval(reference.hi)-reference.lo).hi:
            raise ArithmeticError("composed current bound failed")
        signed_error = interval(current)-reference
        excludes_zero = reference.lo > 0 or reference.hi < 0
        magnitude_min = min(abs(exact(reference.lo)),abs(exact(reference.hi))) if excludes_zero else Fraction(0)
        magnitude_max = max(abs(exact(reference.lo)),abs(exact(reference.hi)))
        allowance_min = (Fraction(1,1000)+magnitude_min/Fraction(10000))/12
        allowance_max = (Fraction(1,1000)+magnitude_max/Fraction(10000))/12
        error_excludes_zero = signed_error.lo > 0 or signed_error.hi < 0
        error_min = min(abs(exact(signed_error.lo)),abs(exact(signed_error.hi))) if error_excludes_zero else Fraction(0)
        screen = "within" if exact(direct) <= allowance_min else "exceeds" if error_min > allowance_max else "indeterminate"
        faces.append({"Jn_reference_A_m2":jn[i].record(),"Jp_reference_A_m2":jp[i].record(),
            "Jion_reference_A_m2":ji[i].record(),"particle_flux_reference_m2_s":particle[i].record(),
            "vacancy_drive_reference":drives[i].record(),"Ddot_reference_A_m2":d_rate[i].record(),
            "Jtotal_reference_A_m2":reference.record(),"arithmetic_upper_A_m2":list(map(str,arithmetic)),
            "represented_phi_effect_upper_A_m2":list(map(str,potential)),
            "Ddot_error_upper_A_m2":str(displacement_error),"total_current_error_upper_A_m2":str(direct),
            "composed_current_error_upper_A_m2":str(composed.hi),
            "initial_observable_axis_allowance_formula":"(1/1000+abs(J_ref)/10000)/12 A/m2",
            "initial_observable_axis_allowance_min_A_m2":str(allowance_min),
            "pointwise_axis_screen":screen})
    yd = list(map(exact,point["ydot_m3_s"]))
    if len(yd) != 3*count:
        raise base.UnsupportedPoint("saved derivative shape differs")
    exact_saved_charge_rate = [q*(yd[count+i]-yd[i]+yd[2*count+i]) for i in range(count)]
    saved_charge_flux = [exact(en)+exact(hp)+exact(ion) for en,hp,ion in
                         zip(point["J_n_A_m2"],point["J_p_A_m2"],point["J_ion_A_m2"],strict=True)]
    saved_flux_charge_rate = [polarity*exact(point["J_ion_A_m2"][0])/ion_widths[0]]
    saved_flux_charge_rate += [polarity*(saved_charge_flux[i+1]-saved_charge_flux[i])/widths[i] for i in range(count-2)]
    saved_flux_charge_rate += [-polarity*exact(point["J_ion_A_m2"][-1])/ion_widths[-1]]
    max_error = lambda values,bounds: base.upper_float(max(base.error_upper(v,b) for v,b in zip(values,bounds,strict=True)))
    return {"phase":point["phase_id"],"time_s":point["time_s"],"event_side":point["event_side"],
        "voltage_V":point["voltage_V"],"voltage_rate_V_s":point["voltage_rate_V_s"],
        "reference_input_meaning":"exact saved binary states/materials;original encoded widths and diffusion-only steric law",
        "phi_error_upper_V":base.upper_float(max(phi_error)),
        "rho_aggregation_error_upper_C_m3":base.upper_float(max(abs(r-exact(v)) for r,v in zip(rho,point["rho_C_m3"],strict=True))),
        "RHS_error_upper_m3_s":{"n":max_error(yd[:count],rn),"p":max_error(yd[count:2*count],rp),"positive_ion":max_error(yd[2*count:],ion_rate)},
        "rho_rate_error_upper_C_m3_s":{"stored_rho_dot":max_error(point["rho_dot_C_m3_s"],rho_rate),
            "exact_charge_of_stored_ydot":max_error(exact_saved_charge_rate,rho_rate),
            "exact_divergence_of_stored_conduction":max_error(saved_flux_charge_rate,rho_rate)},
        "ion_rate_reference_m3_s":[value.record() for value in ion_rate],
        "rho_rate_reference_C_m3_s":[value.record() for value in rho_rate],
        "phi_rate_reference_V_s":[value.record() for value in phi_rate],
        "phi_dot_error_upper_V_s":max_error(point["phi_dot_V_s"],phi_rate),
        "reference_ion_inventory_rate":"exactly zero by telescoping identical face fluxes and zero external flux",
        "stored_ion_inventory_rate_m2_s":str(sum((width*rate for width,rate in zip(ion_widths,yd[2*count:],strict=True)),Fraction(0))),
        "minimum_reference_vacancy":str(min(vacancies)),"faces":faces,
        "max_total_current_error_upper_A_m2":max(base.upper_float(Decimal(f["total_current_error_upper_A_m2"])) for f in faces),
        "left_current_error_upper_A_m2":base.upper_float(Decimal(faces[0]["total_current_error_upper_A_m2"])),
        "density_error_radii":"zero only by exact-saved-input definition;trajectory radii unknown",
        "continuous_trajectory_error_bound":None,"reference_eligibility":False}

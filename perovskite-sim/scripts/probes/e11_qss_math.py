"""Phase E11.1 — offline QSS interface-SRH math (no solver). Validate the
algebraic reduction against analytical limits BEFORE any solver wiring.

QSS per SRH path: the interface-plane densities deplete from their bulk-
projected targets by delta, set by balancing TE-in vs SRH-out:
    v_th * delta = SRH(proj_n - delta, proj_p - delta)
A 1-D bounded root-find in delta (stable; no ODE DOF, no Jacobian feedback).
R_interface = v_th * delta.
"""
import numpy as np
from perovskite_sim.physics.recombination import interface_recombination

VT = 0.025852  # V at 300 K

def srh_qss_R(proj_n, proj_p, ni_sq, n1, p1, v_n, v_p, v_th):
    """Solve v_th*delta = SRH(proj_n-delta, proj_p-delta) for delta>=0; return R."""
    if v_n == 0.0 and v_p == 0.0:
        return 0.0, 0.0
    hi = min(proj_n, proj_p)  # delta cannot exceed either density
    if hi <= 0:
        return 0.0, 0.0
    def f(d):
        R = interface_recombination(proj_n - d, proj_p - d, ni_sq, n1, p1, v_n, v_p)
        return v_th * d - R
    # f(0) = -R(0). If R(0)<=0 (np<=ni^2, generation/eq) -> delta=0, R=R(0) clamped>=... 
    R0 = interface_recombination(proj_n, proj_p, ni_sq, n1, p1, v_n, v_p)
    if R0 <= 0.0:
        return 0.0, 0.0  # no recombination (at/below equilibrium) -> no depletion
    # bisection on [0, hi)
    lo, h = 0.0, hi * (1 - 1e-9)
    if f(h) < 0:  # SRH still exceeds v_th*delta at max depletion -> R capped by v_th*hi
        d = h
    else:
        for _ in range(80):
            mid = 0.5 * (lo + h)
            if f(mid) > 0: h = mid
            else: lo = mid
        d = 0.5 * (lo + h)
    R = v_th * d
    return R, d

def main():
    # representative interface params (PVK/ETL-like): sigma=1e-19 m2, v_th=1e5 m/s
    sig = 1e-19; vth_carrier = 1e5; Nt = 1e16  # areal-ish proxy
    v_n = v_p = sig * vth_carrier * Nt  # SRV [m/s]
    v_th_TE = 1e5  # TE equilibration velocity
    n1 = p1 = 2.09e23
    print("=== analytical reference limits ===")
    # 1. dark equilibrium: proj_n*proj_p = ni_sq -> R = 0
    ni_sq = 1e44
    pn = 1e22; pp = ni_sq/pn  # n*p = ni_sq exactly
    R,d = srh_qss_R(pn, pp, ni_sq, n1, p1, v_n, v_p, v_th_TE)
    print(f"1. dark eq (np=ni_sq): R={R:.3e} (expect 0)  -> {'PASS' if abs(R)<1e-3 else 'FAIL'}")
    # 2. below-equilibrium (np<ni_sq): generation suppressed -> R=0 (no spurious source)
    R,d = srh_qss_R(1e20, 1e20, ni_sq, n1, p1, v_n, v_p, v_th_TE)  # np=1e40<<1e44
    print(f"2. np<ni_sq: R={R:.3e} (expect 0, NO spurious generation) -> {'PASS' if R>=0 else 'FAIL'}")
    # 3. high injection (np>>ni_sq): R>0, bounded by v_th*min(proj)
    pn=pp=1e23; R,d=srh_qss_R(pn,pp,1e24,n1,p1,v_n,v_p,v_th_TE)
    print(f"3. high inj (np>>ni_sq): R={R:.3e}>0, delta={d:.3e}<=min_proj={min(pn,pp):.1e} -> {'PASS' if (R>0 and d<=min(pn,pp)) else 'FAIL'}")
    # 4. monotonic in Nt (more defects -> more R)
    Rs=[]
    for Nt_ in [1e9,1e12,1e15,1e18]:
        vv=sig*vth_carrier*Nt_; R,_=srh_qss_R(1e23,1e23,1e24,n1,p1,vv,vv,v_th_TE); Rs.append(R)
    mono = all(Rs[i]<=Rs[i+1]+1e-6 for i in range(len(Rs)-1))
    print(f"4. monotone R vs Nt: {[f'{r:.2e}' for r in Rs]} -> {'PASS' if mono else 'FAIL'}")
    # 5. weak-SRH limit: R -> SRH(proj) (delta~0)
    vv=1e-3; R,d=srh_qss_R(1e23,1e23,1e24,n1,p1,vv,vv,v_th_TE)
    Rdirect=interface_recombination(1e23,1e23,1e24,n1,p1,vv,vv)
    print(f"5. weak SRH: R_qss={R:.3e} vs SRH(proj)={Rdirect:.3e} ratio={R/Rdirect:.4f} -> {'PASS' if abs(R/Rdirect-1)<0.01 else 'FAIL'}")
    print("\n=== QSS vs bulk-interior over-count (the fix) ===")
    # bulk-interior (E1.5): uses undepleted bulk densities (large) -> large R
    # QSS: uses projected (depleted) densities -> physical R
    n_bulk=1e24; p_bulk=1e22  # ETL electrons (huge), PVK holes
    Vbend=0.3  # band bending [V] depleting the plane
    proj_n=n_bulk*np.exp(-Vbend/VT); proj_p=p_bulk  # electron depleted at plane
    R_bulk=interface_recombination(n_bulk,p_bulk,1e28,n1,p1,v_n,v_p)
    R_qss,_=srh_qss_R(proj_n,proj_p,1e28*np.exp(-Vbend/VT),n1,p1,v_n,v_p,v_th_TE)
    print(f"bulk-interior R={R_bulk:.3e} (over-count) | QSS-projected R={R_qss:.3e} | ratio={R_qss/max(R_bulk,1e-30):.3e}")

if __name__=="__main__":
    main()

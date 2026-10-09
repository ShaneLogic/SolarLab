"""Manufactured independent checks; no saved physical point is evaluated."""
from __future__ import annotations

import copy
from decimal import Context, Decimal, localcontext
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load("bound_base", REPO/"scripts/benchmarks/hysteresis_current_bounds.py")
mobile = load("mobile_bound", REPO/"scripts/benchmarks/hysteresis_mobile_bounds.py")


def manufactured(polarity=1, diffusion=1, zero_stock=False):
    """Direct 160-digit SG and weighted-current solution on four nodes."""
    with localcontext(Context(prec=160)):
        ions = list(map(Decimal, [0,0,0,0] if zero_stock else [1,2,1,0]))
        widths = list(map(Decimal, [1, 2, 4]))
        cells = [Decimal(1), Decimal("1.5"), Decimal(3), Decimal(4)]
        coeff = [Decimal(1), Decimal("0.5"), Decimal("0.25")]
        flux = []
        for left,right,width in zip(ions[:-1],ions[1:],widths,strict=True):
            vacancy_ratio = (1-left/10)/(1-right/10)
            drive = vacancy_ratio.ln()
            # Different formula from the implementation: independent exp and
            # two Bernoulli evaluations, rather than its series and B(-x) law.
            bp = Decimal(1) if not drive else drive/(drive.exp()-1)
            bm = Decimal(1) if not drive else (-drive)/((-drive).exp()-1)
            flux.append(Decimal(diffusion)*(bp*left-bm*right)/width)
        jion = [-polarity*f for f in flux]
        current = sum((j/c for j,c in zip(jion,coeff,strict=True)),Decimal(0))/sum((1/c for c in coeff),Decimal(0))
        d_rate = [polarity*(j-current) for j in jion]
        phi_rate = [Decimal(0)]
        for rate,c in zip(d_rate,coeff,strict=True):
            phi_rate.append(phi_rate[-1]-rate/c)
        full = [Decimal(0)]+flux+[Decimal(0)]
        rate = [-(full[i+1]-full[i])/width for i,width in enumerate(cells)]
        assert abs(sum((r*w for r,w in zip(rate,cells,strict=True)),Decimal(0))) < Decimal("1e-150")
        point = {"phase_id":"manufactured_ramp","time_s":0.,"event_side":"ramp_start_right",
            "voltage_V":0.,"voltage_rate_V_s":0.,"junction_polarity":polarity,"illuminated":False,
            "n_m3":[1.]*4,"p_m3":[1.]*4,"positive_ions_m3":list(map(float,ions)),
            "negative_ions_m3":None,"trap_occupancy":None,"phi_V":[0.]*4,"rho_C_m3":[0.]*4,
            "ydot_m3_s":[0.]*8+list(map(float,rate)),"rho_dot_C_m3_s":list(map(float,rate)),
            "phi_dot_V_s":list(map(float,phi_rate)),"Ddot_A_m2":list(map(float,d_rate)),
            "J_n_A_m2":[0.]*3,"J_p_A_m2":[0.]*3,"J_ion_A_m2":list(map(float,jion)),
            "J_cond_A_m2":list(map(float,jion))}
        point["state"] = point["n_m3"]+point["p_m3"]+point["positive_ions_m3"]
        arrays = {"initial.x_m":[0.,1.,3.,7.],"material.poisson_factor.C":list(map(float,coeff)),
            "material.poisson_factor.h_cell":[1.5,3.],"material.dx_cell":list(map(float,cells)),
            "material.P_ion0":list(map(float,ions)),"material.D_n_face":[1.]*3,"material.D_p_face":[1.]*3,
            "material.D_ion_face":[float(diffusion)]*3,"material.P_lim_node":[10.]*4}
        for key in ("N_A","N_D","chi","Eg","N_t_node","B_rad","C_n","C_p","G_optical"):
            arrays["material."+key] = [0.]*4
        for key in ("ni_sq","tau_n","tau_p","n1","p1"):
            arrays["material."+key] = [1.]*4
        packet = {"has_dynamic_traps_or_negative_ions":False,
            "contacts_and_potential":{"effective_S":dict.fromkeys(("S_n_L","S_p_L","S_n_R","S_p_R")),
                "V_T_device":1.,"V_bi_bc":0.,"V_bi_eff":0.,"junction_polarity":polarity,"left_phi_V":0.},
            "ion_transport":{"species":"single_positive","steric_diffusion_only":True,
                "effective_shared_site":False,"external_particle_flux":"zero_both_ends",
                "clip_upper_binary64":mobile.CLIP_UPPER_HEX}}
        return point,arrays,packet,current


class BoundsTests(unittest.TestCase):
    def test_log_encloses_independent_high_precision_values(self):
        for rational in (Fraction(1),Fraction(2),Fraction(3,2),Fraction(1,3),Fraction(1,1000000),Fraction(2**20),Fraction(1000000000001,1000000000000)):
            with self.subTest(value=rational), localcontext(Context(prec=160)):
                oracle = (Decimal(rational.numerator)/Decimal(rational.denominator)).ln()
                result = mobile.log_bounds(base,rational)
                self.assertLessEqual(result.lo,oracle)
                self.assertGreaterEqual(result.hi,oracle)
                self.assertLess(result.hi-result.lo,Decimal("1e-58"))

    def test_log_domain(self):
        for value in (0,-1,True,2**34):
            with self.subTest(value=value), self.assertRaises(base.UnsupportedPoint):
                mobile.log_bounds(base,value)

    def test_zero_diffusion_uniform_and_clipped_cases(self):
        cases = [([1,2],[10,10],0,0),([1,1],[10,10],1,0),([10,20],[1,1],1,-10)]
        for ions,limits,diffusion,expected in cases:
            with self.subTest(ions=ions):
                flux,_,_ = mobile.mobile_flux_bounds(base,[0,0],ions,[1],[diffusion],limits,1,Fraction.from_float(.999999))
                self.assertLessEqual(flux[0].lo,Decimal(expected));self.assertGreaterEqual(flux[0].hi,Decimal(expected))
                self.assertEqual(flux[0].lo,flux[0].hi)

    def test_vacancy_log_sign_with_closed_flux(self):
        flux,_,_ = mobile.mobile_flux_bounds(base,[0,0],[1,2],[1],[1],[4,4],1,Fraction.from_float(.999999))
        with localcontext(Context(prec=160)):
            oracle = -4*Decimal("1.5").ln()
            self.assertLessEqual(flux[0].lo,oracle);self.assertGreaterEqual(flux[0].hi,oracle)

    def test_nonuniform_point_and_both_current_signs(self):
        results = []
        for polarity in (1,-1):
            point,arrays,packet,current = manufactured(polarity)
            result = mobile.qualify_point(base,point,arrays,packet,1)
            self.assertFalse(result["reference_eligibility"])
            self.assertIsNone(result["continuous_trajectory_error_bound"])
            self.assertLess(result["max_total_current_error_upper_A_m2"],1e-14)
            self.assertLess(result["RHS_error_upper_m3_s"]["positive_ion"],1e-14)
            self.assertEqual(len(result["ion_rate_reference_m3_s"]),4)
            self.assertEqual(len(result["rho_rate_reference_C_m3_s"]),4)
            for face in result["faces"]:
                lo,hi = map(Decimal,face["Jtotal_reference_A_m2"])
                self.assertLessEqual(lo,current);self.assertGreaterEqual(hi,current)
                self.assertEqual(face["pointwise_axis_screen"],"within")
            results.append(float(current))
        self.assertEqual(results[0],-results[1])

    def test_zero_diffusion_full_point(self):
        point,arrays,packet,_ = manufactured(diffusion=0)
        result = mobile.qualify_point(base,point,arrays,packet,1)
        self.assertEqual(result["max_total_current_error_upper_A_m2"],0.)
        self.assertEqual(result["RHS_error_upper_m3_s"]["positive_ion"],0.)

    def test_zero_stock_is_separate_from_zero_diffusion(self):
        point,arrays,packet,_ = manufactured(diffusion=1,zero_stock=True)
        self.assertTrue(all(value == 1 for value in arrays["material.D_ion_face"]))
        self.assertTrue(all(value == 0 for value in point["positive_ions_m3"]+arrays["material.P_ion0"]))
        result = mobile.qualify_point(base,point,arrays,packet,1)
        self.assertEqual(result["max_total_current_error_upper_A_m2"],0.)

    def test_unsupported_branch_and_changed_width(self):
        p,a,k,_ = manufactured()
        bad = copy.deepcopy(k);bad["ion_transport"]["steric_diffusion_only"] = False
        with self.assertRaises(base.UnsupportedPoint):mobile.qualify_point(base,p,a,bad,1)
        bad = copy.deepcopy(k);bad["ion_transport"]["effective_shared_site"] = True
        with self.assertRaises(base.UnsupportedPoint):mobile.qualify_point(base,p,a,bad,1)
        bad = copy.deepcopy(a);bad["material.poisson_factor.h_cell"][0] = 1.
        with self.assertRaises(base.UnsupportedPoint):mobile.qualify_point(base,p,bad,k,1)
        bad = copy.deepcopy(p);bad["junction_polarity"] = True
        with self.assertRaises(base.UnsupportedPoint):mobile.qualify_point(base,bad,a,k,1)

    def test_invalid_ion_domain(self):
        for ions,limits in (([-1,1],[10,10]),([1,2],[0,10])):
            with self.assertRaises(base.UnsupportedPoint):
                mobile.mobile_flux_bounds(base,[0,0],ions,[1],[1],limits,1,Fraction.from_float(.999999))

    def test_no_production_imports(self):
        self.assertFalse(any(name.startswith("perovskite_sim") for name in sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)

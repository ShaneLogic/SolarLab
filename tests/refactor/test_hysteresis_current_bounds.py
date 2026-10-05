"""Distinct enclosure and domain checks; no trajectory or production solver."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys
import unittest


SOURCE = Path(__file__).resolve().parents[2]/'scripts/benchmarks/hysteresis_current_bounds.py'
spec = importlib.util.spec_from_file_location('hi_current_bounds_test_subject', SOURCE)
cb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cb
spec.loader.exec_module(cb)


class TestDirectedArithmetic(unittest.TestCase):
    def contains(self, interval, value):
        self.assertLessEqual(F(interval.lo), value)
        self.assertGreaterEqual(F(interval.hi), value)

    def test_directed_operations_against_fraction(self):
        for a,b in ((F(1,3),F(-5,7)),(F(10)**80,F(1,10**90)),
                    (F(-3,10**100),F(7,10**80)),(F(1,2),F(1,2))):
            with self.subTest(a=a,b=b):
                ia,ib = cb._Interval.of(a),cb._Interval.of(b)
                self.contains(ia+ib,a+b)
                self.contains(ia-ib,a-b)
                self.contains(ia*ib,a*b)
                self.contains(ia/ib,a/b)
        with self.assertRaises(cb.UnsupportedPoint):
            cb._Interval.of(1)/cb._Interval(Decimal(-1),Decimal(1))

    def test_positive_series_tail_encloses_independent_decimal260(self):
        for x in (F(0),F(1,10**80),F(-1,10**80),F(1,5),F(-1,5),F(12),F(-12),F(80),F(-80)):
            with self.subTest(x=x),localcontext() as context:
                context.prec = 260
                value = (Decimal(x.numerator)/Decimal(x.denominator)).exp()-1
                self.contains(cb._expm1_bounds(x),F(value))
                if x:
                    self.contains(cb.bernoulli_bounds(x),F(Decimal(x.numerator)/Decimal(x.denominator)/value))
                else:
                    self.assertEqual(cb.bernoulli_bounds(x).record(),['1','1'])

    def test_physical_domains_fail_explicitly(self):
        for value in (True,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises((cb.UnsupportedPoint,ValueError)):
                cb.exact(value)
        with self.assertRaises(cb.UnsupportedPoint):
            cb.bernoulli_bounds(F(8193))
        with self.assertRaises(cb.UnsupportedPoint):
            cb.sg_bounds([0,1],[1,1],[1,1],[0],[1],[1],1,1,1)


class TestConditionalPhysicalBounds(unittest.TestCase):
    def test_density_box_propagates_through_positive_poisson_inverse(self):
        radius=cb.density_to_potential_radius([F(2),F(2)],[F(1)],
                  [F(0),F(1,10),F(0)],[F(0),F(1,5),F(0)],[F(0)]*3,F(2))
        self.assertEqual(radius,[F(0),F(3,20),F(0)])
        with self.assertRaisesRegex(cb.UnsupportedPoint,'unknown'):
            cb.density_to_potential_radius([2,2],[1],None,[0]*3,[0]*3,1)

    def test_one_volume_charge_and_capacitor_rates_with_both_port_signs(self):
        for pol in (-1,1):
            for slope in (F(-3,10),F(0),F(3,10)):
                for charge_rate in (F(0),F(3,7)):
                    right = -pol*slope
                    middle = (charge_rate+2*right)/4
                    expected = [-2*middle,2*(middle-right)]
                    dr = cb.poisson_rate_bounds([F(2),F(2)],[F(1)],
                         [cb._Interval.of(0),cb._Interval.of(charge_rate),cb._Interval.of(0)],slope,pol)
                    for interval,value in zip(dr,expected,strict=True):
                        self.assertLessEqual(F(interval.lo),value)
                        self.assertGreaterEqual(F(interval.hi),value)
                    if charge_rate == 0:
                        # area=3; physical left/right electrode rates have opposite signs.
                        self.assertEqual(F(dr[0].lo)*3,3*pol*slope)
                        self.assertEqual(F(dr[-1].hi)*-3,-3*pol*slope)

    def test_joint_density_and_potential_sensitivity_uses_explicit_radii(self):
        phi,n,p = [F(0),F(0)],[F(2),F(3)],[F(5),F(7)]
        phi_error = [F(1,1000)]*2
        nr,pr = [F(1,100)]*2,[F(1,50)]*2
        args = (phi,n,p,[F(1)],[F(2)],[F(3)],F(1),F(1),phi_error)
        bounds = cb.sg_sensitivity_bounds(*args,n_radius=nr,p_radius=pr)[0]
        with localcontext() as context:
            context.prec=160
            d=lambda x:Decimal(x.numerator)/Decimal(x.denominator)
            for sp,sn,sh in ((-1,-1,1),(1,1,-1),(1,-1,1),(-1,1,-1)):
                ph=[phi[0]+sp*phi_error[0],phi[1]-sp*phi_error[1]]
                nn=[n[0]+sn*nr[0],n[1]-sn*nr[1]]
                pp=[p[0]+sh*pr[0],p[1]-sh*pr[1]]
                xi=d(ph[1]-ph[0]);b=xi/(xi.exp()-1)
                current_n=2*(b*(d(nn[0])-d(nn[1]))+xi*d(nn[0]))
                current_p=3*(b*(d(pp[1])-d(pp[0]))+xi*d(pp[1]))
                self.assertLessEqual(abs(current_n-Decimal(-2)),Decimal(bounds['electron']['box_current_change_upper_A_m2']))
                self.assertLessEqual(abs(current_p-Decimal(6)),Decimal(bounds['hole']['box_current_change_upper_A_m2']))
        with self.assertRaisesRegex(cb.UnsupportedPoint,'unknown'):
            cb.sg_sensitivity_bounds(*args,n_radius=None,p_radius=pr)

    @staticmethod
    def neutral_fixture():
        arrays={'initial.x_m':[0.,.5,1.]}
        for name,value in [('P_ion0',0),('N_A',0),('N_D',0),('chi',0),('Eg',0),('N_t_node',0),
                           ('ni_sq',4),('tau_n',1),('tau_p',1),('n1',1),('p1',1),('B_rad',0),
                           ('C_n',0),('C_p',0),('G_optical',0)]:
            arrays['material.'+name]=[value]*3
        arrays.update({'material.poisson_factor.C':[2.,2.],'material.poisson_factor.h_cell':[.5],
                       'material.dx_cell':[.5,.5,.5],'material.D_ion_face':[0.,0.],
                       'material.D_n_face':[.1,.1],'material.D_p_face':[.2,.2]})
        packet={'has_dynamic_traps_or_negative_ions':False,
                'contacts_and_potential':{'effective_S':{'nL':None,'nR':None,'pL':None,'pR':None},
                                          'V_T_device':1.,'V_bi_bc':0.}}
        point={'n_m3':[2.]*3,'p_m3':[2.]*3,'positive_ions_m3':[0.]*3,'state':[2.]*6+[0.]*3,
               'junction_polarity':1,'voltage_V':0.,'voltage_rate_V_s':0.,'phi_V':[0.]*3,
               'rho_C_m3':[0.]*3,'rho_dot_C_m3_s':[0.]*3,'phi_dot_V_s':[0.]*3,'Ddot_A_m2':[0.]*2,
               'J_n_A_m2':[0.]*2,'J_p_A_m2':[0.]*2,'J_ion_A_m2':[0.]*2,'J_cond_A_m2':[0.]*2,
               'ydot_m3_s':[0.]*9,'illuminated':False,'phase_id':'analytic','time_s':0.,'event_side':'right'}
        return point,arrays,packet

    def test_exact_neutral_limit_and_unknown_trajectory_error(self):
        point,arrays,packet=self.neutral_fixture()
        result=cb.qualify_point(point,arrays,packet,1.)
        self.assertEqual(result['max_Ddot_error_upper_A_m2'],0.)
        self.assertEqual(result['max_total_current_error_upper_A_m2'],0.)
        self.assertIsNone(result['continuous_trajectory_error_bound'])
        self.assertFalse(result['reference_eligibility'])

    def test_nonzero_inventory_flux_and_negative_physical_state_rejected(self):
        for kind in ('negative','mobile'):
            point,arrays,packet=self.neutral_fixture()
            if kind=='negative':
                point['n_m3'][1]=-1.
            else:
                arrays['material.D_ion_face']=[1.,1.]
                point['positive_ions_m3'][1]=1.
            with self.subTest(kind=kind),self.assertRaises(cb.UnsupportedPoint):
                cb.qualify_point(point,arrays,packet,1.)


if __name__=='__main__':
    unittest.main()

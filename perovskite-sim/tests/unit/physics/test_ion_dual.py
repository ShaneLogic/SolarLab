"""Unit tests for dual-species ion migration."""
import numpy as np
import pytest

from perovskite_sim.physics.ion_migration import (
    ion_continuity_rhs,
    ion_continuity_rhs_neg,
)
from perovskite_sim.constants import V_T


@pytest.fixture
def uniform_grid():
    """Simple 20-node uniform grid, 400 nm."""
    return np.linspace(0, 400e-9, 20)


class TestNegativeIonRHS:

    def test_zero_D_gives_zero_rhs(self, uniform_grid):
        """With D_ion_neg = 0, dP_neg/dt should be zero everywhere."""
        x = uniform_grid
        N = len(x)
        phi = np.zeros(N)
        P_neg = np.full(N, 1e25)
        dP = ion_continuity_rhs_neg(x, phi, P_neg, D_I=0.0, V_T=V_T, P_lim=1e30)
        np.testing.assert_allclose(dP, 0.0, atol=1e-10)

    def test_uniform_P_zero_field_gives_zero_rhs(self, uniform_grid):
        """Uniform concentration in zero field → no flux → zero RHS."""
        x = uniform_grid
        N = len(x)
        phi = np.zeros(N)
        P_neg = np.full(N, 1e25)
        dP = ion_continuity_rhs_neg(x, phi, P_neg, D_I=1e-17, V_T=V_T, P_lim=1e30)
        np.testing.assert_allclose(dP, 0.0, atol=1e10)

    def test_reversed_drift_vs_positive(self, uniform_grid):
        """Under the same field, negative ions should drift opposite to positive.

        In a uniform field pointing right (phi increases left to right),
        positive ions drift right (current in field direction) and
        negative ions drift left (current against field).
        """
        x = uniform_grid
        N = len(x)
        # Linear potential: field points right
        phi = np.linspace(0, 0.5, N)
        P = np.full(N, 1e25)
        D = 1e-17

        dP_pos = ion_continuity_rhs(x, phi, P, D_I=D, V_T=V_T, P_lim=1e30)
        dP_neg = ion_continuity_rhs_neg(x, phi, P, D_I=D, V_T=V_T, P_lim=1e30)

        # In a field pointing right:
        # Positive species accumulates at the right → dP_pos > 0 at right
        # Negative species accumulates at the left → dP_neg > 0 at left
        # Interior nodes (exclude BCs at 0 and -1)
        mid = N // 2
        # The signs should be opposite in bulk
        pos_right_bias = np.mean(dP_pos[mid:-1])
        neg_right_bias = np.mean(dP_neg[mid:-1])
        assert pos_right_bias * neg_right_bias < 0, \
            "Positive and negative ion drift should be opposite"

    def test_zero_flux_bcs(self, uniform_grid):
        """Zero-flux BCs: boundary node RHS depends only on interior flux."""
        x = uniform_grid
        N = len(x)
        phi = np.linspace(0, 0.5, N)
        # Non-uniform concentration to ensure flux ≠ 0
        P_neg = np.linspace(1e25, 2e25, N)
        dP = ion_continuity_rhs_neg(x, phi, P_neg, D_I=1e-17, V_T=V_T, P_lim=1e30)
        # Total ion content should be conserved (sum of dP * dx_cell = 0)
        dx = np.diff(x)
        dx_cell = np.empty(N)
        dx_cell[0] = dx[0]
        dx_cell[-1] = dx[-1]
        dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        total_change = np.sum(dP * dx_cell)
        np.testing.assert_allclose(total_change, 0.0, atol=1e10,
                                   err_msg="Ion conservation violated")


class TestChargeDensity:

    def test_dual_species_charge_density(self):
        """Charge density should include both ionic species."""
        from perovskite_sim.solver.mol import _charge_density
        from perovskite_sim.constants import Q
        N = 5
        n = np.full(N, 1e16)
        p = np.full(N, 1e16)
        P_pos = np.full(N, 1.5e25)
        P0_pos = np.full(N, 1e25)
        P_neg = np.full(N, 1.5e25)
        P0_neg = np.full(N, 1e25)
        N_A = np.zeros(N)
        N_D = np.zeros(N)

        # Single species
        rho_single = _charge_density(p, n, P_pos, P0_pos, N_A, N_D)
        # Dual species
        rho_dual = _charge_density(p, n, P_pos, P0_pos, N_A, N_D,
                                   P_neg=P_neg, P_neg0=P0_neg)
        # Difference should be -Q * (P_neg - P0_neg)
        expected_diff = -Q * (P_neg - P0_neg)
        np.testing.assert_allclose(rho_dual - rho_single, expected_diff, rtol=1e-10)

    def test_single_species_backward_compat(self):
        """Without P_neg, charge density should be identical to original."""
        from perovskite_sim.solver.mol import _charge_density
        N = 5
        n = np.full(N, 1e16)
        p = np.full(N, 1e16)
        P = np.full(N, 1e25)
        P0 = np.full(N, 1e25)
        N_A = np.zeros(N)
        N_D = np.zeros(N)
        rho = _charge_density(p, n, P, P0, N_A, N_D)
        rho_none = _charge_density(p, n, P, P0, N_A, N_D, P_neg=None, P_neg0=None)
        np.testing.assert_array_equal(rho, rho_none)

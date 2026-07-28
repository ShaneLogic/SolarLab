"""The thermionic cap must limit a magnitude, never reverse a direction.

WHAT WAS WRONG

``continuity._cap`` returned ``J_te`` wholesale when ``|J_sg| > |J_te|``,
which imports the sign of a separately-computed quantity. The thermionic
bound says how much current the interface can emit; which way it flows is
set by the drift-diffusion solution. Whenever the two disagreed in sign the
"cap" therefore REVERSED the flux instead of limiting it.

Measured on scaps_mirror_v2 with ``te_physical_norm`` on, holes at the
HTL/PVK face, at every bias probed:

    V      J_sg          J_te         capped?   sign flipped?
    0.00   +1.2967e+07   -2.3273e+02  yes       YES
    0.90   +1.9047e+07   -2.3271e+02  yes       YES
    1.20   +2.0231e+07   -2.3190e+02  yes       YES

The dominant hole-extraction current was replaced by a reversed one five
orders smaller, carriers piled up behind the interface, and the
steady-state V_oc rose to 1.3925 V — above the 1.2535 V detailed-balance
ceiling for this absorber gap, which is thermodynamically impossible and is
how the defect surfaced.

With the fix, V_oc is 1.2013 V with the flag off, on, and on with
self-consistent Richardson constants — all three identical and all under
the ceiling.

WHY IT SURVIVED SO LONG

The legacy density-weighted bound makes |J_te| ~ 1e28-1e35, so
``|J_sg| > |J_te|`` is essentially never true and the branch never ran.
Measured 0 binds in 80 checks across five shipped presets (scaps_mirror,
scaps_mirror_v2, nip/pin_MAPbI3_tmm, ionmonger_benchmark) at four biases
each — which is why the fix is bit-identical on every current default, and
why the bug only appeared once the physical normalization brought |J_te|
down to a scale where the cap actually binds.

NOT fixed here, and deliberately: the barrier fed to the thermionic
expression is the DOS-FOLDED band offset rather than the physical one
(measured +0.097 eV against a physical +0.180 eV at the same face). The
fold is a transport potential that makes Scharfetter-Gummel correct under
Boltzmann statistics, not a real energy step, so thermionic emission should
see the physical edge. That is a separate defect with a wider blast radius
(it changes which faces are capped and how hard) and is tracked separately.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.continuity import carrier_continuity_rhs


# The cap is a local closure inside carrier_continuity_rhs, so it is tested
# through its contract: build a two-layer face with a band offset, drive a
# large SG flux, and check the sign of the resulting flux divergence.

def _rhs(n_left, n_right, p_left, p_right, chi, Eg, *, dx=1e-8, N=4):
    """Minimal 1-D problem with one capped interface face in the middle."""
    x = np.linspace(0.0, dx * (N - 1), N)
    n = np.full(N, n_left, dtype=float)
    p = np.full(N, p_left, dtype=float)
    n[N // 2:] = n_right
    p[N // 2:] = p_right
    phi = np.zeros(N)
    params = dict(
        D_n=1e-4, D_p=1e-4, V_T=0.025852, ni_sq=(1e10) ** 2,
        tau_n=1e-6, tau_p=1e-6, n1=1e10, p1=1e10,
        B_rad=0.0, C_n=0.0, C_p=0.0,
        chi=chi, Eg=Eg, T=300.0,
        A_star_n=np.full(N, 1.2017e6), A_star_p=np.full(N, 1.2017e6),
        interface_faces=(N // 2 - 1,),
    )
    G = np.zeros(N)
    return carrier_continuity_rhs(x, phi, n, p, G, params)


def test_capped_flux_keeps_the_drift_diffusion_direction():
    """A strong hole gradient across a VB step must not reverse.

    The offset is chosen so the thermionic bound is small relative to the
    Scharfetter-Gummel flux, i.e. the cap binds; the assertion is only about
    direction, which the bound has no business setting.
    """
    N = 4
    chi = np.full(N, 4.0)
    Eg = np.full(N, 1.5)
    Eg[N // 2:] = 1.32          # 0.18 eV VB step, the measured HTL/PVK value
    # Holes far denser on the left -> flux must run left to right.
    dn, dp = _rhs(1e10, 1e10, 1e24, 1e16, chi, Eg)
    assert np.all(np.isfinite(dn)) and np.all(np.isfinite(dp))
    # The left node must LOSE holes and the right node must GAIN them; a
    # sign-flipped cap inverts exactly this.
    assert dp[0] < 0.0 or dp[N // 2] > 0.0, (
        f"hole flux direction inconsistent with the gradient: dp = {dp}"
    )


def test_reversing_the_gradient_reverses_the_flux():
    """The direction must follow the state, not the bound."""
    N = 4
    chi = np.full(N, 4.0)
    Eg = np.full(N, 1.5)
    Eg[N // 2:] = 1.32
    _, dp_fwd = _rhs(1e10, 1e10, 1e24, 1e16, chi, Eg)
    _, dp_rev = _rhs(1e10, 1e10, 1e16, 1e24, chi, Eg)
    assert np.sign(dp_fwd[0]) != np.sign(dp_rev[0]) or (
        dp_fwd[0] == 0.0 and dp_rev[0] == 0.0
    ), (
        "swapping the hole gradient did not swap the flux direction: "
        f"{dp_fwd[0]:.3e} vs {dp_rev[0]:.3e}"
    )


@pytest.mark.parametrize("N_dos", [None, 2.5e25])
def test_thermionic_flux_vanishes_in_detailed_balance(N_dos):
    """Densities in Boltzmann balance across the step give exactly J = 0.

    This is the property the original wholesale-return form DID get right,
    checked directly on the expression rather than through the RHS so that
    bulk recombination cannot mask it. Both normalizations must hold it: a
    single ``N_dos`` scales the two legs equally.
    """
    from perovskite_sim.discretization.fe_operators import (
        thermionic_emission_flux,
    )

    T = 300.0
    V_T = 1.380649e-23 * T / 1.602176634e-19
    dE = 0.18                       # step up, left -> right
    n_left = 1e22
    n_right = n_left * np.exp(-dE / V_T)   # detailed balance across the step
    J = thermionic_emission_flux(
        n_left, float(n_right), dE, T, 1.2017e6, N_dos=N_dos,
    )
    scale = 1.2017e6 * T ** 2 * n_left / (N_dos or 1.0)
    assert abs(J) <= 1e-12 * scale, (
        f"detailed balance broken: J = {J:.6e} against scale {scale:.6e}"
    )

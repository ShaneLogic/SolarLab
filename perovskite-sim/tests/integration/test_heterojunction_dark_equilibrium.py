"""Heterojunction dark equilibrium — detailed balance (review §7 #2).

At ``V_app = 0`` in the dark, the discrete device state must be a true
thermodynamic equilibrium, INDEPENDENT of how large the band offsets or the
effective-DOS contrasts between adjacent layers are:

  (a) every face electron / hole / total current density vanishes, and
  (b) the electron and hole quasi-Fermi levels coincide into a single
      spatially FLAT ``E_F``.

For the Scharfetter-Gummel discretisation this is exact, not asymptotic.
Zero SG electron flux at a face forces ``n_{i+1}/n_i = exp((phi_n,i+1 -
phi_n,i)/V_T)`` (Bernoulli identity ``B(-x) = e^x B(x)``) with the transport
potential ``phi_n = phi + chi_transport``.  With the effective-DOS fold active
(``DeviceStack.dos_band_potentials``, default ON since 2026-06),

    chi_transport      = chi + V_T ln(N_C / N_C_ref)
    (chi + Eg)_transport = chi + Eg - V_T ln(N_V / N_V_ref)

so the zero-flux state reproduces the physical Boltzmann relation exactly and
``E_Fn = -phi - chi + V_T ln(n/N_C)`` comes out flat.  The same algebra for
holes gives flat ``E_Fp``.  Differencing the two identities across one face
yields the *closed form of the pre-fix defect*, which this file also pins:

    with the fold OFF,  E_Fn(i+1) - E_Fn(i) = -V_T ln(N_C,i+1 / N_C,i)
                        E_Fp(i+1) - E_Fp(i) = +V_T ln(N_V,i+1 / N_V,i)

i.e. a spurious kT*ln(DOS-ratio) quasi-Fermi step at every DOS-contrast
heterojunction — the documented root cause of the SolarLab-vs-SCAPS V_oc gap.

One auxiliary exact statement falls out and is pinned separately: the
contact-to-contact quasi-Fermi offset is an ALGEBRAIC identity in the boundary
conditions alone (no settling required),

    E_Fn[N-1] - E_Fp[0] == stack.compute_V_bi() - MaterialArrays.V_bi_bc

valid whenever ``N_C == N_V`` in the two contact layers.  A genuinely flat E_F
therefore REQUIRES the Poisson Dirichlet drop to equal the work-function-derived
built-in potential, which is why the synthetic stacks below set
``V_bi = compute_V_bi()`` — otherwise the flatness assertions would be measuring
a YAML contact mismatch rather than heterojunction transport.

Preset: ``configs/scaps_mirror.yaml`` (3 electrical layers, no glass substrate,
no het_recomb_despike, no flat_band_contacts, D_ion = 0 so J_ion == 0 exactly,
N_C == N_V in every layer).  ``optical_material`` is stripped because the run is
dark — TMM would only add build cost.  Cost is ~0.4 s per settled case at
``N_GRID = 30``; this fits the DEFAULT suite (no ``slow`` marker).  N_GRID is
PINNED: 10 intervals per layer is on the cheap side of a very steep cost cliff.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.experiments.band_diagram import (
    _DARK_SETTLE,
    _QFL_MASK,
    _node_band_params,
    compute_band_diagram,
)
from perovskite_sim.experiments.jv_sweep import (
    compute_current_components,
    extract_spatial_snapshot,
)
from perovskite_sim.experiments.steady_state import _grid_for
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.scaps_compat.materials import ni_from_dos
from perovskite_sim.solver.mol import build_material_arrays, run_transient
from perovskite_sim.solver.newton import solve_equilibrium

CONFIG = str(Path(__file__).resolve().parents[2] / "configs" / "scaps_mirror.yaml")

# 10 intervals per electrical layer (the `30 // len(elec)` convention used by
# tests/unit/solver/test_dos_band_potentials.py). PINNED, not incidental.
N_GRID = 30

# Wall-time guard on every Radau leg: a future grid/preset change fails fast
# instead of hanging the default suite.
MAX_NFEV = 100_000

# --------------------------------------------------------------------------
# Tolerances. Every number is fixed a priori from physics or from an exact
# algebraic bound — none is read back from a run.
# --------------------------------------------------------------------------

# AM1.5G photon-flux-limited short-circuit current of a 1.53 eV absorber
# (~25 mA/cm^2). An EXTERNAL standard, bracketed by SCAPS's 26.3 and SolarLab's
# 23.1 mA/cm^2 on this very stack — deliberately NOT the J_sc of this run.
J_REF = 250.0  # A/m^2

# Any mechanism that breaks detailed balance at a heterojunction (a
# recombination term referenced to a bulk-asymptotic rather than a discrete
# equilibrium n*p, a TE leg whose two halves do not cancel, an interface-state
# block that pumps carriers) operates at the same rate scale as the channels
# that carry J_sc under illumination, i.e. O(10-100 %) of J_REF -- 2-4 decades
# above these bounds. Below 1 % of J_sc the dark baseline cannot bias a
# dark-J-V / EQE / impedance reading at the percent level.
J_FACE_TOL = 1.0e-2 * J_REF     # 2.5 A/m^2, any face
J_TERMINAL_TOL = 1.0e-3 * J_REF  # 0.25 A/m^2, face 0 = the reported terminal J

# Quasi-Fermi flatness. Signal side: the error class under test produces
# V_T*ln(DOS ratio); the smallest contrast in the matrix (25x) gives 83.2 meV,
# so this sits 16x below the smallest signal. Noise side: the contact-BC
# contribution is removed by construction (V_bi := compute_V_bi(), see
# test_contact_bc_offset_is_an_algebraic_identity); the interface-SRH sink at
# dark equilibrium is bounded by S_eff = sigma*v_th*N_t*cf = 1e-2 m/s acting on
# ni_PVK^2, i.e. <~1e-16 A/m^2 (a sub-nanovolt step); integration error enters
# E_F only LOGARITHMICALLY (a 1e-4 relative density error is V_T*ln(1.0001) =
# 2.6 ueV) plus a Poisson term from delta_rho = q*(delta_p - delta_n) over the
# 20/25 nm quasi-neutral contact layers, bounded by delta_rho*L^2/(2*eps)
# <~ 0.2 mV. So 5 meV is ~25x above the largest legitimate residual and 16x
# below the smallest signal; it is also 0.19 kT at 300 K, i.e. below any
# thermodynamically meaningful quasi-Fermi step. (The incumbent
# tests/unit/experiments/test_band_diagram.py::test_equilibrium_fermi_level_is_flat
# uses 150 meV, which cannot distinguish a flat E_F from a fully broken 83 meV
# DOS step; this is 30x tighter on purpose.)
FLAT_TOL = 5.0e-3  # eV

# Same noise budget as FLAT_TOL; 3 meV = 0.12 kT.
STEP_TOL = 3.0e-3  # eV

# 0.72 x the predicted 83.2 meV: fires on a genuine breakdown of the known
# fold-OFF defect rather than on its exact magnitude.
NOT_FLAT_FLOOR = 60.0e-3  # eV

# Exact algebraic identity -> only IEEE round-off on ~5 eV magnitudes separates
# the two sides. 1e-9 eV is ~1e5 ULP of headroom and 4e-8 kT.
BC_IDENTITY_TOL = 1.0e-9  # eV

# At least this fraction of nodes must carry a defined quasi-Fermi level, or the
# flatness assertion would be vacuous.
MIN_COVERAGE = 0.90


# --------------------------------------------------------------------------
# Synthetic stacks
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Case:
    """One (band-offset, DOS-contrast, fold) probe point."""

    chi_etl: float          # ETL electron affinity [eV]
    eg_etl: float           # ETL band gap [eV]
    dos: tuple              # (N_C=N_V) per layer: HTL, PVK, ETL [m^-3]
    fold: bool              # DeviceStack.dos_band_potentials

    @property
    def dEc(self) -> float:
        """E_C(ETL) - E_C(PVK) = chi_PVK - chi_ETL. > 0 spike, < 0 cliff."""
        return 3.94 - self.chi_etl


# PVK reference: chi = 3.94 eV, Eg = 1.53 eV.
#   spike: chi_ETL = 3.74 -> dEc = +0.20 eV, dEv = +0.17 eV
#   cliff: chi_ETL = 4.24 -> dEc = -0.30 eV, dEv = +1.17 eV
# Both |dEc| >= 0.20 eV, so the 0.05 eV thermionic-emission threshold is crossed
# in every case and the TE cap's equilibrium behaviour is genuinely exercised.
# The 25x DOS contrast is the measured HTL/PVK ratio of the shipped preset;
# HTL and ETL are given the SAME DOS so the fold-OFF error stays internal (both
# contacts remain V_bi-consistent) and the signature is two clean +/-83.2 meV
# steps rather than a smeared tilt.
_DOS_FLAT = (1.0e25, 1.0e25, 1.0e25)
_DOS_25X = (2.5e26, 1.0e25, 2.5e26)
DOS_RATIO = _DOS_25X[0] / _DOS_25X[1]  # 25.0

CASES = {
    # fold ON — the invariant must hold for every one of these.
    "spike_dos1": Case(3.74, 1.90, _DOS_FLAT, True),
    "cliff_dos1": Case(4.24, 2.40, _DOS_FLAT, True),
    "spike_dos25": Case(3.74, 1.90, _DOS_25X, True),
    "cliff_dos25": Case(4.24, 2.40, _DOS_25X, True),
    # fold OFF — control (no DOS contrast) + the two known-defect points.
    "spike_dos1_nofold": Case(3.74, 1.90, _DOS_FLAT, False),
    "spike_dos25_nofold": Case(3.74, 1.90, _DOS_25X, False),
    "cliff_dos25_nofold": Case(4.24, 2.40, _DOS_25X, False),
}

FOLD_ON = [k for k, c in CASES.items() if c.fold]
FOLD_OFF_CONTRAST = ["spike_dos25_nofold", "cliff_dos25_nofold"]


def _build_stack(case: Case):
    """``dataclasses.replace`` a shipped preset into the requested geometry.

    Per case the ETL gets (chi, Eg) and every layer gets (Nc300, Nv300) plus a
    REBUILT ``ni = sqrt(N_C N_V) exp(-Eg/2V_T)`` so the mass-action constant
    stays consistent with the DOS — skipping that silently breaks n*p = ni^2 and
    would invalidate the whole test.  ``V_bi`` is recomputed from
    ``compute_V_bi()`` AFTER all replacements: a 0.30 eV chi_ETL shift moves the
    required contact drop by 0.30 V, and the shipped manual 1.30 V would then
    impose a 300 meV quasi-Fermi tilt unrelated to the invariant under test.
    """
    stack = load_scaps_yaml(CONFIG)
    elec = electrical_layers(stack)
    assert len(elec) == 3 and len(stack.layers) == 3, (
        "this test assumes the 3-electrical-layer, substrate-free scaps_mirror "
        f"stack; got {[L.name for L in stack.layers]}"
    )
    layers = []
    for i, layer in enumerate(stack.layers):
        p = layer.params
        chi = case.chi_etl if i == 2 else p.chi
        eg = case.eg_etl if i == 2 else p.Eg
        dos = case.dos[i]
        layers.append(dataclasses.replace(layer, params=dataclasses.replace(
            p, chi=chi, Eg=eg, Nc300=dos, Nv300=dos,
            ni=ni_from_dos(dos, dos, eg),
            optical_material=None, n_optical=None,
        )))
    stack = dataclasses.replace(
        stack, layers=tuple(layers), dos_band_potentials=case.fold,
    )
    # N_C == N_V per layer is what makes the contact-BC identity exact.
    for layer in electrical_layers(stack):
        assert layer.params.Nc300 == layer.params.Nv300
    return dataclasses.replace(stack, V_bi=stack.compute_V_bi())


# --------------------------------------------------------------------------
# Dark settle (shared across every assertion for a given case)
# --------------------------------------------------------------------------

_SETTLED: dict[str, tuple] = {}


def _settle(key: str):
    """``solve_equilibrium`` seed + the band_diagram dark ladder, cached.

    Unlike ``experiments/band_diagram.py``'s dark branch — which does
    ``if sol.success: y = ...`` and therefore silently reuses the previous,
    less-settled state when a Radau leg fails — this asserts success on EVERY
    leg, so a non-convergent settle can never masquerade as a converged one.
    """
    if key in _SETTLED:
        return _SETTLED[key]
    stack = _build_stack(CASES[key])
    x = _grid_for(stack, N_GRID)
    mat = build_material_arrays(x, stack)
    y = solve_equilibrium(x, stack)
    for t in _DARK_SETTLE:
        sol = run_transient(
            x, y, (0.0, t), np.array([t]), stack,
            illuminated=False, V_app=0.0, mat=mat,
            max_step=t / 8.0, max_nfev=MAX_NFEV,
        )
        assert sol.success, (
            f"[{key}] dark settle leg t={t:g} s did not converge: "
            f"{getattr(sol, 'message', '')}"
        )
        y = sol.y[:, -1]
    _SETTLED[key] = (stack, x, mat, y)
    return _SETTLED[key]


def _quasi_fermi(x, y, stack, mat):
    """E_Fn / E_Fp exactly as ``compute_band_diagram`` builds them.

    Reuses ``band_diagram._node_band_params`` (physical per-layer chi/Eg/N_C/N_V,
    NOT the DOS-folded transport arrays) and ``band_diagram._QFL_MASK``.  Pinned
    against the public API by ``test_inline_quasi_fermi_matches_compute_band_diagram``.
    """
    chi, Eg, Nc, Nv = _node_band_params(x, stack)
    s = extract_spatial_snapshot(x, y, stack, 0.0, mat=mat)
    V_T = mat.V_T_device
    n = np.maximum(s.n, 1.0e-30)
    p = np.maximum(s.p, 1.0e-30)
    E_C = -s.phi - chi
    E_V = E_C - Eg
    E_Fn = np.where(n / Nc > _QFL_MASK, E_C + V_T * np.log(n / Nc), np.nan)
    E_Fp = np.where(p / Nv > _QFL_MASK, E_V - V_T * np.log(p / Nv), np.nan)
    return E_Fn, E_Fp


def _resolved_fermi(E_Fn, E_Fp):
    """E_Fn where defined, else E_Fp (the resolution used by test_band_diagram)."""
    return np.where(~np.isnan(E_Fn), E_Fn, E_Fp)


def _interface_nodes(x, stack):
    """Node index sitting exactly on each internal layer boundary."""
    elec = electrical_layers(stack)
    edges = np.cumsum([L.thickness for L in elec])[:-1]
    return [int(np.argmin(np.abs(x - e))) for e in edges]


def _leg_scale(x, y, stack, mat):
    """Magnitude of the larger SG leg at every face [A/m^2] — DIAGNOSTIC ONLY.

    Reported in failure messages so a regression is a number rather than a
    boolean.  Deliberately NOT asserted on: it is simultaneously far too weak
    where it is large (1e-3 x 1e7 A/m^2 is meaningless) and pure noise where the
    minority carrier sits at the integrator floor.  Asserting on it would be a
    tolerance chosen to fit the numerics rather than the physics.
    """
    from perovskite_sim.constants import Q

    dx = np.diff(x)
    s = extract_spatial_snapshot(x, y, stack, 0.0, mat=mat)
    V_T = mat.V_T_device
    xi_n = ((s.phi + mat.chi)[1:] - (s.phi + mat.chi)[:-1]) / V_T
    leg_n = Q * mat.D_n_face / dx * np.maximum(
        bernoulli(xi_n) * s.n[1:], bernoulli(-xi_n) * s.n[:-1])
    phi_p = s.phi + mat.chi + mat.Eg
    xi_p = (phi_p[1:] - phi_p[:-1]) / V_T
    leg_p = Q * mat.D_p_face / dx * np.maximum(
        bernoulli(xi_p) * s.p[:-1], bernoulli(-xi_p) * s.p[1:])
    return leg_n, leg_p


# --------------------------------------------------------------------------
# (a) every face current vanishes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(CASES))
def test_dark_equilibrium_face_currents_vanish(key):
    """No face carries current at V=0 in the dark, for ANY offset / DOS ratio."""
    stack, x, mat, y = _settle(key)
    cc = compute_current_components(x, y, stack, 0.0, mat=mat)
    leg_n, leg_p = _leg_scale(x, y, stack, mat)

    # D_ion = 0 on this preset, so the ionic channel must be identically zero.
    assert np.all(cc.J_ion == 0.0), f"[{key}] J_ion not identically zero"

    for name, J, leg in (("J_n", cc.J_n, leg_n),
                         ("J_p", cc.J_p, leg_p),
                         ("J_total", cc.J_total, np.maximum(leg_n, leg_p))):
        i = int(np.argmax(np.abs(J)))
        rel = abs(J[i]) / leg[i] if leg[i] > 0.0 else float("nan")
        assert abs(J[i]) < J_FACE_TOL, (
            f"[{key}] {name} does not vanish at dark equilibrium: worst face "
            f"{i}/{J.size - 1}, |{name}| = {abs(J[i]):.6e} A/m^2 "
            f"(bound {J_FACE_TOL:g} = 1% of J_REF={J_REF:g}); "
            f"SG leg at that face = {leg[i]:.3e} A/m^2, |J|/leg = {rel:.3e}"
        )


@pytest.mark.parametrize("key", list(CASES))
def test_dark_equilibrium_terminal_current_vanishes(key):
    """Face 0 is what ``_compute_current`` reports — the V=0 anchor of every
    dark-J-V / EQE / impedance baseline, so it gets the tighter bound."""
    stack, x, mat, y = _settle(key)
    cc = compute_current_components(x, y, stack, 0.0, mat=mat)
    assert abs(cc.J_total[0]) < J_TERMINAL_TOL, (
        f"[{key}] terminal J_total[0] = {cc.J_total[0]:.6e} A/m^2 at dark "
        f"equilibrium (bound {J_TERMINAL_TOL:g} = 0.1% of J_REF={J_REF:g})"
    )


# --------------------------------------------------------------------------
# (b) the quasi-Fermi levels are flat
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", FOLD_ON)
def test_dark_equilibrium_fermi_level_is_flat(key):
    """With the effective-DOS fold active, E_F is flat for ANY band offset and
    ANY DOS contrast. This is the review §7 #2 invariant proper."""
    stack, x, mat, y = _settle(key)
    E_Fn, E_Fp = _quasi_fermi(x, y, stack, mat)
    E_F = _resolved_fermi(E_Fn, E_Fp)
    good = ~np.isnan(E_F)
    assert good.mean() >= MIN_COVERAGE, (
        f"[{key}] only {good.sum()}/{good.size} nodes carry a defined "
        f"quasi-Fermi level ({good.mean():.1%}) — flatness would be vacuous"
    )

    span = float(np.max(E_F[good]) - np.min(E_F[good]))
    # Diagnostics: interface nodes are where a DOS/offset bug lands.
    iface = _interface_nodes(x, stack)
    keep = good.copy()
    keep[iface] = False
    span_bulk = float(np.max(E_F[keep]) - np.min(E_F[keep]))
    worst = int(np.nanargmax(np.abs(E_F - np.median(E_F[good]))))
    assert span < FLAT_TOL, (
        f"[{key}] E_F is NOT flat at dark equilibrium: "
        f"max-min = {span * 1e3:.3f} meV over {good.sum()} nodes "
        f"(bound {FLAT_TOL * 1e3:g} meV). Worst node {worst} "
        f"(interface nodes = {iface}), E_F[{worst}] = {E_F[worst]:.6f} eV vs "
        f"median {np.median(E_F[good]):.6f} eV. "
        f"Span EXCLUDING interface nodes = {span_bulk * 1e3:.3f} meV. "
        f"V_T*ln(DOS ratio {DOS_RATIO:g}) = "
        f"{mat.V_T_device * math.log(DOS_RATIO) * 1e3:.3f} meV."
    )


@pytest.mark.parametrize("key", FOLD_ON)
def test_dark_equilibrium_np_equals_ni_squared(key):
    """E_Fn == E_Fp wherever both are defined, i.e. |V_T ln(np/ni^2)| < 5 meV.

    Every recombination channel in the code is proportional to (n*p - ni^2), so
    this is the statement that R = 0 in every channel at equilibrium.
    """
    stack, x, mat, y = _settle(key)
    E_Fn, E_Fp = _quasi_fermi(x, y, stack, mat)
    both = (~np.isnan(E_Fn)) & (~np.isnan(E_Fp))
    if not both.any():
        pytest.skip(
            f"[{key}] no node has BOTH carriers above the {_QFL_MASK:g} "
            "deep-minority mask — the split is undefined on this device"
        )
    gap = np.abs(E_Fn[both] - E_Fp[both])
    assert gap.max() < FLAT_TOL, (
        f"[{key}] n*p != ni^2 at equilibrium: max|E_Fn - E_Fp| = "
        f"{gap.max() * 1e3:.4f} meV over {int(both.sum())} node(s) "
        f"(bound {FLAT_TOL * 1e3:g} meV)"
    )


def test_fold_off_without_dos_contrast_is_still_flat():
    """Control: with dos_band_potentials OFF but no DOS contrast, E_F is flat.

    This proves any fold-OFF failure below is DOS-driven, not band-offset-driven.
    """
    key = "spike_dos1_nofold"
    stack, x, mat, y = _settle(key)
    E_Fn, E_Fp = _quasi_fermi(x, y, stack, mat)
    E_F = _resolved_fermi(E_Fn, E_Fp)
    good = ~np.isnan(E_F)
    span = float(np.max(E_F[good]) - np.min(E_F[good]))
    assert span < FLAT_TOL, (
        f"[{key}] fold-OFF control with equal DOS in all layers is not flat: "
        f"{span * 1e3:.3f} meV (bound {FLAT_TOL * 1e3:g} meV)"
    )


@pytest.mark.parametrize("key", FOLD_OFF_CONTRAST)
def test_fold_off_with_dos_contrast_shows_the_predicted_kT_ln_step(key):
    """The known pre-fix defect, asserted QUANTITATIVELY (not xfail'd).

    Differencing the zero-flux Boltzmann relation against the physical one over
    one face with the fold OFF gives, exactly,

        E_Fn(i+1) - E_Fn(i) = -V_T ln(N_C,i+1 / N_C,i)
        E_Fp(i+1) - E_Fp(i) = +V_T ln(N_V,i+1 / N_V,i)

    On this stack that is -/+ V_T ln 25 = -/+ 83.2 meV, matching the repo's own
    independent measurement of 84 meV at HTL/PVK on scaps_mirror_v2.
    """
    stack, x, mat, y = _settle(key)
    V_T = mat.V_T_device
    E_Fn, E_Fp = _quasi_fermi(x, y, stack, mat)
    E_F = _resolved_fermi(E_Fn, E_Fp)
    good = ~np.isnan(E_F)
    span = float(np.max(E_F[good]) - np.min(E_F[good]))
    assert span > NOT_FLAT_FLOOR, (
        f"[{key}] fold-OFF E_F span {span * 1e3:.3f} meV did not exceed the "
        f"{NOT_FLAT_FLOOR * 1e3:g} meV floor — the documented DOS-step defect "
        f"appears to have changed character"
    )

    _, _, Nc, Nv = _node_band_params(x, stack)
    iface = _interface_nodes(x, stack)
    assert len(iface) == 2
    i_hp, i_pe = iface  # HTL/PVK, PVK/ETL

    # HTL/PVK is read on the hole level (electrons are deep minority in the HTL).
    got_p = E_Fp[i_hp] - E_Fp[i_hp - 1]
    want_p = V_T * math.log(Nv[i_hp] / Nv[i_hp - 1])
    assert got_p == pytest.approx(want_p, abs=STEP_TOL), (
        f"[{key}] HTL/PVK E_Fp step {got_p * 1e3:.3f} meV != predicted "
        f"+V_T ln(N_V ratio) = {want_p * 1e3:.3f} meV "
        f"(tol {STEP_TOL * 1e3:g} meV)"
    )
    # PVK/ETL is read on the electron level (holes are deep minority in the ETL).
    got_n = E_Fn[i_pe] - E_Fn[i_pe - 1]
    want_n = -V_T * math.log(Nc[i_pe] / Nc[i_pe - 1])
    assert got_n == pytest.approx(want_n, abs=STEP_TOL), (
        f"[{key}] PVK/ETL E_Fn step {got_n * 1e3:.3f} meV != predicted "
        f"-V_T ln(N_C ratio) = {want_n * 1e3:.3f} meV "
        f"(tol {STEP_TOL * 1e3:g} meV)"
    )


# --------------------------------------------------------------------------
# Root cause of the fold-ON failure: the DOS fold is applied more than once
# on the node that sits exactly on a layer boundary.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", ["scaps_mirror.yaml", "scaps_mirror_v2.yaml"])
@pytest.mark.parametrize("n_grid", [30, 45, 80])
def test_dos_fold_is_applied_exactly_once_per_node(cfg, n_grid):
    """Every node's transport potential must be ONE layer's folded (chi, Eg).

    ``build_material_arrays`` assigns the base per-layer arrays with ``=`` over
    inclusive, OVERLAPPING masks, so the node that lands exactly on a layer
    boundary is taken by the LATER (right) layer.  The effective-DOS fold that
    follows uses ``+=`` over the same overlapping masks, so that node
    accumulates the sum of both neighbours' corrections on top of the right
    layer's base chi/Eg.  Cheap and settle-free: this reads the cache directly
    on the SHIPPED presets.
    """
    stack = load_scaps_yaml(
        str(Path(__file__).resolve().parents[2] / "configs" / cfg))
    assert stack.dos_band_potentials
    x = _grid_for(stack, n_grid)
    mat = build_material_arrays(x, stack)
    elec = electrical_layers(stack)
    V_T = mat.V_T_device
    ref = next((L.params for L in elec if L.role == "absorber"), elec[0].params)

    def folded(p):
        dC = V_T * math.log(p.Nc300 / ref.Nc300)
        dV = V_T * math.log(p.Nv300 / ref.Nv300)
        return p.chi + dC, p.Eg - dC - dV

    edges = np.cumsum([L.thickness for L in elec])[:-1]
    for k, edge in enumerate(edges):
        j = int(np.argmin(np.abs(x - edge)))
        left, right = folded(elec[k].params), folded(elec[k + 1].params)
        got = (mat.chi[j], mat.Eg[j])
        ok = (np.allclose(got, left, atol=1e-12)
              or np.allclose(got, right, atol=1e-12))
        assert ok, (
            f"[{cfg} N={n_grid}] node {j} on the "
            f"{elec[k].name}/{elec[k + 1].name} boundary carries a hybrid "
            f"transport potential: mat.chi = {got[0]:.6f}, mat.Eg = "
            f"{got[1]:.6f}; left-layer folded = ({left[0]:.6f}, {left[1]:.6f}), "
            f"right-layer folded = ({right[0]:.6f}, {right[1]:.6f}). "
            f"Excess over the right layer = "
            f"{(got[0] - right[0]) * 1e3:.3f} meV in chi and "
            f"{(got[1] - right[1]) * 1e3:.3f} meV in Eg; "
            f"V_T*ln(N_C_left/N_C_ref) = "
            f"{V_T * math.log(elec[k].params.Nc300 / ref.Nc300) * 1e3:.3f} meV"
        )


# --------------------------------------------------------------------------
# Auxiliary exact identities + reuse pin
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", ["scaps_mirror.yaml", "scaps_mirror_v2.yaml"])
def test_contact_bc_offset_is_an_algebraic_identity(cfg):
    """``E_Fn[N-1] - E_Fp[0] == compute_V_bi() - V_bi_bc``, no settling needed.

    Substituting phi[0] = 0, phi[N-1] = V_bi_bc, n[N-1] = n_R, p[0] = p_L into
    the quasi-Fermi definitions and expanding ``compute_V_bi``'s ``_fermi_level``
    with ni = sqrt(N_C N_V) exp(-Eg/2V_T) leaves a residual of exactly
    (V_T/2)*ln(N_C_L/N_V_L) + (V_T/2)*ln(N_V_R/N_C_R), which vanishes when
    N_C == N_V in the two contact layers (true of every scaps_mirror layer).

    Consequence, and the reason the synthetic stacks above override V_bi: the
    SHIPPED preset pins V_bi = 1.30 while compute_V_bi() returns ~1.2940, so its
    dark E_F is tilted by ~-6.0 meV — above the 5 meV flatness tolerance and
    nothing to do with heterojunction transport.
    """
    stack = load_scaps_yaml(
        str(Path(__file__).resolve().parents[2] / "configs" / cfg))
    x = _grid_for(stack, N_GRID)
    mat = build_material_arrays(x, stack)
    y = solve_equilibrium(x, stack)  # un-settled seed: the identity is in the BCs
    for layer in (electrical_layers(stack)[0], electrical_layers(stack)[-1]):
        assert layer.params.Nc300 == layer.params.Nv300

    E_Fn, E_Fp = _quasi_fermi(x, y, stack, mat)
    lhs = float(E_Fn[-1] - E_Fp[0])
    rhs = float(stack.compute_V_bi() - mat.V_bi_bc)
    assert lhs == pytest.approx(rhs, abs=BC_IDENTITY_TOL), (
        f"[{cfg}] contact quasi-Fermi offset {lhs:.12f} eV != "
        f"compute_V_bi() - V_bi_bc = {rhs:.12f} eV "
        f"(compute_V_bi = {stack.compute_V_bi():.6f}, "
        f"V_bi_bc = {mat.V_bi_bc:.6f})"
    )


def test_inline_quasi_fermi_matches_compute_band_diagram():
    """Pin the private-API reuse: the settle-once optimisation used by this file
    must reproduce the shipped ``compute_band_diagram`` bit for bit."""
    cfg_stack = load_scaps_yaml(CONFIG)
    x = _grid_for(cfg_stack, N_GRID)
    mat = build_material_arrays(x, cfg_stack)
    y = solve_equilibrium(x, cfg_stack)
    for t in _DARK_SETTLE:
        sol = run_transient(
            x, y, (0.0, t), np.array([t]), cfg_stack,
            illuminated=False, V_app=0.0, mat=mat,
            max_step=t / 8.0, max_nfev=MAX_NFEV,
        )
        assert sol.success
        y = sol.y[:, -1]
    E_Fn, E_Fp = _quasi_fermi(x, y, cfg_stack, mat)

    bd = compute_band_diagram(cfg_stack, 0.0, illuminated=False, N_grid=N_GRID)
    np.testing.assert_array_equal(bd.x, x)
    np.testing.assert_array_equal(np.isnan(bd.E_Fn), np.isnan(E_Fn))
    np.testing.assert_array_equal(np.isnan(bd.E_Fp), np.isnan(E_Fp))
    assert np.nanmax(np.abs(bd.E_Fn - E_Fn)) < 1.0e-12
    assert np.nanmax(np.abs(bd.E_Fp - E_Fp)) < 1.0e-12

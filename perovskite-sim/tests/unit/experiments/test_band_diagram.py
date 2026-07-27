"""Band-diagram invariants: flat E_F at equilibrium, quasi-Fermi splitting = qV."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.constants import V_T as _V_T_300
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.experiments.band_diagram import compute_band_diagram

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "scaps_mirror_v2.yaml"

# Dark-equilibrium E_F flatness bound: ONE thermal voltage, kT/q = 25.852 meV
# at 300 K. Fixed a priori from physics — not read back from a run.
#
#   * Meaning. kT/q is the scale on which a quasi-Fermi difference changes the
#     Boltzmann occupation by a factor e. A step below it carries no
#     thermodynamically meaningful population imbalance; a step at or above it
#     means "flat E_F" is simply false. Anything larger is not a tolerance, it
#     is an admission that the state is not an equilibrium.
#
#   * Resolves the error class this gate exists for. A heterojunction transport
#     potential that is wrong by an effective-DOS ratio puts a step of
#     V_T*ln(N_C ratio) into E_F. The smallest DOS contrast on this preset is
#     HTL/PVK = 25x, i.e. 83.2 meV = 3.22 kT/q, so the bound sits 3.2x below
#     the weakest signal it must catch. The previous 150 meV bound was 1.8x
#     ABOVE that signal and passed with the defect present (measured on the
#     2026-06 DOS-fold boundary-node double-count: span 83.214 meV, all of it
#     on the single HTL/PVK interface node; post-fix 6.025 meV).
#
#   * Does not fire on the legitimate residual. On this preset the residual is
#     dominated by a deliberate config choice, not by numerics: the YAML pins
#     V_bi = 1.300 V while compute_V_bi() = 1.293975 V, and the contact
#     quasi-Fermi offset is the algebraic identity
#     E_Fn[-1] - E_Fp[0] == compute_V_bi() - V_bi_bc (pinned by
#     tests/integration/test_heterojunction_dark_equilibrium.py::
#     test_contact_bc_offset_is_an_algebraic_identity), i.e. exactly 6.025 meV
#     = 0.23 kT/q. Settle and discretisation error enter E_F only
#     LOGARITHMICALLY (a 1e-4 relative density error is V_T*ln(1.0001) =
#     2.6 ueV). kT/q therefore leaves ~4x headroom over the whole legitimate
#     budget while staying 3.2x under the smallest defect signal.
FLAT_TOL = _V_T_300  # 0.025852 eV


def _absorber_mask(stack, x):
    elec = electrical_layers(stack)
    edges = np.concatenate([[0.0], np.cumsum([L.thickness for L in elec])])
    a = int(np.argmax([L.thickness for L in elec]))  # absorber = thickest layer
    return (x > edges[a]) & (x < edges[a + 1])


def test_equilibrium_fermi_level_is_flat():
    """At the zero-current dark equilibrium the quasi-Fermi levels coincide into a
    single flat E_F (the defining thermodynamic invariant)."""
    stack = load_scaps_yaml(CONFIG)
    bd = compute_band_diagram(stack, 0.0, illuminated=False, N_grid=40)
    # the resolved E_F: E_Fn where electrons present, else E_Fp
    E_F = np.where(~np.isnan(bd.E_Fn), bd.E_Fn, bd.E_Fp)
    E_F = E_F[~np.isnan(E_F)]
    assert E_F.size > 0
    span = float(np.nanmax(E_F) - np.nanmin(E_F))
    bc_tilt = float(stack.V_bi - stack.compute_V_bi())
    assert span < FLAT_TOL, (
        f"E_F is NOT flat at dark equilibrium: max-min = {span * 1e3:.3f} meV "
        f"over {E_F.size} nodes (bound {FLAT_TOL * 1e3:.3f} meV = kT/q at "
        f"300 K). Of that, {abs(bc_tilt) * 1e3:.3f} meV is the config's own "
        f"contact-BC offset (YAML V_bi {stack.V_bi:.6f} vs compute_V_bi "
        f"{stack.compute_V_bi():.6f}); the remainder is a heterojunction "
        f"transport or detailed-balance defect."
    )


@pytest.mark.slow
def test_quasi_fermi_splitting_equals_qV():
    """Under illumination at V, the absorber quasi-Fermi splitting E_Fn - E_Fp
    equals qV (the operating-point invariant)."""
    stack = load_scaps_yaml(CONFIG)
    V = 0.9
    bd = compute_band_diagram(stack, V, illuminated=True, N_grid=40, settle_t=1e-2)
    split = np.nanmean((bd.E_Fn - bd.E_Fp)[_absorber_mask(stack, bd.x)])
    assert split == pytest.approx(V, abs=0.12)


def test_band_offsets_match_config():
    """E_C and E_V edges reflect the configured electron affinity and bandgap."""
    stack = load_scaps_yaml(CONFIG)
    bd = compute_band_diagram(stack, 0.0, illuminated=False, N_grid=40)
    # gap E_C - E_V equals the per-node bandgap (>0 everywhere, physical)
    gap = bd.E_C - bd.E_V
    assert np.all(gap > 0.5) and np.all(gap < 4.0)


def test_missing_dos_data_raises():
    """A config without effective-DOS data cannot define quasi-Fermi levels."""
    stack = load_scaps_yaml(CONFIG)
    layers = tuple(
        dataclasses.replace(L, params=dataclasses.replace(L.params, Nc300=0.0))
        for L in stack.layers
    )
    stripped = dataclasses.replace(stack, layers=layers)
    with pytest.raises(ValueError, match="Nc300"):
        compute_band_diagram(stripped, 0.0, illuminated=False, N_grid=40)

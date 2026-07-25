"""Steady-state terminal-current convergence certificate (review F-13).

At true steady state charge conservation makes ``J(x)`` uniform, so the
face-to-face disagreement measures how far a settled state actually is
from that limit. ``_compute_current_ss`` reports the median, which is
robust to a few outlying faces but says nothing about convergence on its
own — these tests pin the certificate that now travels with it.

The arithmetic is exercised against a stubbed face-current array so the
contract is tested without a solver run: median over *all* faces (the
historical reported value, unchanged), spread over the *interior* faces
only (contact-adjacent faces excluded — their displacement term is
physical external-circuit current).
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.experiments import jv_sweep
from perovskite_sim.experiments.jv_sweep import (
    _compute_current_ss,
    _compute_current_ss_with_spread,
)


@pytest.fixture
def stub_faces(monkeypatch):
    """Replace the face-current computation with a fixed array."""
    def _install(values):
        arr = np.asarray(values, dtype=float)
        monkeypatch.setattr(
            jv_sweep, "_total_current_faces",
            lambda *a, **kw: arr,
        )
        return arr
    return _install


def _call():
    return _compute_current_ss_with_spread(
        np.zeros(3), np.zeros(3), object(), 0.0, mat=None,
    )


def test_spread_ignores_contact_adjacent_faces(stub_faces):
    """Boundary faces swing on ionic-rich stacks; that is physical
    displacement current, not a conservation error."""
    stub_faces([-1700.0, 10.0, 10.0, 10.0, 1700.0])
    J, spread = _call()
    assert J == pytest.approx(10.0)
    assert spread == pytest.approx(0.0), (
        "boundary faces leaked into the certificate"
    )


def test_spread_reports_interior_disagreement(stub_faces):
    stub_faces([0.0, 10.0, 12.0, 9.0, 0.0])
    _J, spread = _call()
    assert spread == pytest.approx(3.0)


def test_converged_uniform_current_has_zero_spread(stub_faces):
    stub_faces([5.0] * 6)
    J, spread = _call()
    assert J == pytest.approx(5.0)
    assert spread == pytest.approx(0.0)


def test_median_matches_the_legacy_helper(stub_faces):
    """The reported current is unchanged by the certificate addition."""
    stub_faces([-1700.0, 10.0, 11.0, 9.0, 1700.0])
    J_pair, _spread = _call()
    J_legacy = _compute_current_ss(
        np.zeros(3), np.zeros(3), object(), 0.0, mat=None,
    )
    assert J_pair == J_legacy


def test_short_grid_falls_back_to_all_faces(stub_faces):
    """Fewer than three faces leaves nothing interior to compare."""
    stub_faces([4.0, 6.0])
    J, spread = _call()
    assert J == pytest.approx(5.0)
    assert spread == pytest.approx(2.0)


def test_certificate_is_reported_not_gated():
    """The spread is a diagnostic, not a pass/fail threshold.

    Measurement showed it is neither monotone in settling time (the
    ionmonger TMM preset's spread grows 43x going from a 1 ms to a 100 ms
    settle, while the median moves <0.02 %) nor stable under mesh
    refinement (~80x between 12 and 20 nodes per layer), so any fixed
    tolerance would reject converged runs and its natural remedy
    ("settle longer") is backwards. This test pins the absence of such a
    gate so it is not reintroduced without new evidence.
    """
    from perovskite_sim.experiments import eqe as eqe_mod
    from perovskite_sim.experiments import suns_voc as suns_mod

    assert not hasattr(jv_sweep, "_SS_SPREAD_RTOL")
    for mod in (eqe_mod, suns_mod):
        assert not hasattr(mod, "_SS_SPREAD_RTOL")

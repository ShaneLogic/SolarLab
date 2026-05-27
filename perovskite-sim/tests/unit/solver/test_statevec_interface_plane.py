"""Phase E3 Sprint 6 Day 1 — StateVec extension for interface-plane state.

Add an OPTIONAL ``iface_state`` block at the END of the packed state
vector. When None (default), packing + unpacking is bit-identical to
the pre-E3 behaviour (legacy bulk-only state).

Layout when active:
  y = [n[0:N], p[0:N], P[0:N], P_neg[0:N]?, iface_state[0:4*N_iface]]

The iface_state block carries 4 unknowns per heterointerface k in this
order: (n_1s, p_1s, n_2s, p_2s). Block at the end so legacy slicing
(y[:N], y[N:2N], y[2N:3N]) stays bit-identical for the bulk portion.

Contract pinned by this test file:
1. pack(n, p, P) without iface_state → identical to legacy 3N vector.
2. pack(n, p, P, P_neg) without iface_state → identical to legacy 4N.
3. pack(n, p, P, iface_state=...) → 3N + len(iface_state).
4. unpack with N_iface_state=0 → identical to legacy.
5. unpack with N_iface_state>0 → returns StateVec with iface_state set.
6. Round-trip pack → unpack → pack reproduces input bit-identically.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.solver.mol import StateVec


N = 5
N_IFACE = 2


def _toy_arrays():
    n = np.arange(N, dtype=float)
    p = np.arange(N, dtype=float) + N
    P = np.arange(N, dtype=float) + 2 * N
    return n, p, P


def test_pack_legacy_3n_no_iface_state():
    """pack(n, p, P) without iface_state → bit-identical to legacy."""
    n, p, P = _toy_arrays()
    y = StateVec.pack(n, p, P)
    assert y.shape == (3 * N,)
    assert np.array_equal(y[:N], n)
    assert np.array_equal(y[N:2*N], p)
    assert np.array_equal(y[2*N:3*N], P)


def test_pack_legacy_4n_with_p_neg():
    """pack(n, p, P, P_neg) → bit-identical to legacy 4N."""
    n, p, P = _toy_arrays()
    P_neg = np.arange(N, dtype=float) + 3 * N
    y = StateVec.pack(n, p, P, P_neg=P_neg)
    assert y.shape == (4 * N,)
    assert np.array_equal(y[3*N:4*N], P_neg)


def test_pack_with_iface_state_appends_at_end():
    """pack with iface_state appends a 4*N_iface block at the end."""
    n, p, P = _toy_arrays()
    iface = np.arange(4 * N_IFACE, dtype=float) + 100.0
    y = StateVec.pack(n, p, P, iface_state=iface)
    assert y.shape == (3 * N + 4 * N_IFACE,)
    assert np.array_equal(y[:N], n)
    assert np.array_equal(y[N:2*N], p)
    assert np.array_equal(y[2*N:3*N], P)
    assert np.array_equal(y[3*N:3*N + 4*N_IFACE], iface)


def test_pack_with_p_neg_and_iface_state():
    """pack(n, p, P, P_neg, iface_state) → 4N + 4*N_iface, iface at end."""
    n, p, P = _toy_arrays()
    P_neg = np.arange(N, dtype=float) + 3 * N
    iface = np.arange(4 * N_IFACE, dtype=float) + 100.0
    y = StateVec.pack(n, p, P, P_neg=P_neg, iface_state=iface)
    assert y.shape == (4 * N + 4 * N_IFACE,)
    assert np.array_equal(y[4*N:4*N + 4*N_IFACE], iface)


def test_unpack_legacy_no_iface_state():
    """unpack with N_iface_state=0 → identical to legacy unpack."""
    n, p, P = _toy_arrays()
    y = StateVec.pack(n, p, P)
    sv = StateVec.unpack(y, N, N_iface_state=0)
    assert np.array_equal(sv.n, n)
    assert np.array_equal(sv.p, p)
    assert np.array_equal(sv.P, P)
    assert sv.iface_state is None or sv.iface_state.size == 0


def test_unpack_with_iface_state_active():
    """unpack with N_iface_state>0 → StateVec.iface_state populated."""
    n, p, P = _toy_arrays()
    iface = np.arange(4 * N_IFACE, dtype=float) + 100.0
    y = StateVec.pack(n, p, P, iface_state=iface)
    sv = StateVec.unpack(y, N, N_iface_state=N_IFACE)
    assert sv.iface_state is not None
    assert np.array_equal(sv.iface_state, iface)


def test_round_trip_pack_unpack_pack_bit_identical():
    """pack → unpack → pack reproduces input bit-identically."""
    n, p, P = _toy_arrays()
    iface = np.linspace(1e10, 1e20, 4 * N_IFACE)
    y_orig = StateVec.pack(n, p, P, iface_state=iface)
    sv = StateVec.unpack(y_orig, N, N_iface_state=N_IFACE)
    y_repack = StateVec.pack(sv.n, sv.p, sv.P, iface_state=sv.iface_state)
    assert np.array_equal(y_orig, y_repack)


def test_unpack_legacy_default_kwarg_bit_identical():
    """unpack with default N_iface_state kwarg (0) → bit-identical to pre-E3.

    Required for all existing callers of unpack(y, N) without kwarg to
    keep working unmodified.
    """
    n, p, P = _toy_arrays()
    y = StateVec.pack(n, p, P)
    sv_default = StateVec.unpack(y, N)  # legacy call signature
    assert np.array_equal(sv_default.n, n)
    assert np.array_equal(sv_default.p, p)
    assert np.array_equal(sv_default.P, P)
    assert sv_default.P_neg is None

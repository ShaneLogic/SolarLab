from __future__ import annotations

import numpy as np

from perovskite_sim._compat import numpy_compat


def test_trapezoid_falls_back_to_trapz(monkeypatch):
    monkeypatch.delattr(numpy_compat.np, "trapezoid", raising=False)

    y = np.array([0.0, 1.0, 0.0])
    x = np.array([0.0, 1.0, 2.0])

    assert numpy_compat.trapezoid(y, x) == np.trapz(y, x)

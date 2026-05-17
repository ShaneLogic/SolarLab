"""Small NumPy compatibility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def trapezoid(
    y: Any,
    x: Any | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> Any:
    """Use NumPy's trapezoid integration with a trapz fallback for NumPy 1.x."""

    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(y, x=x, dx=dx, axis=axis)

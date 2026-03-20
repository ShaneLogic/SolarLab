import numpy as np


def beer_lambert_generation(
    x: np.ndarray, alpha, Phi: float
) -> np.ndarray:
    """G(x) = Phi * alpha(x) * exp(-∫₀ˣ alpha dx') [m⁻³ s⁻¹].

    Supports scalar alpha (uniform device) or array alpha (layered device).
    For array alpha the cumulative optical depth is integrated from x[0]
    using the trapezoidal rule, so generation is zero in non-absorbing layers
    and peaks at the front of the absorbing region.
    """
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim == 0:
        # Scalar: standard single-layer Beer-Lambert
        return float(Phi) * float(alpha) * np.exp(-float(alpha) * x)
    # Array: cumulative optical depth via mid-face trapezoidal integration
    dx = np.diff(x)
    alpha_mid = 0.5 * (alpha[:-1] + alpha[1:])   # (N-1,) face-average
    cum_depth = np.zeros_like(x)
    cum_depth[1:] = np.cumsum(alpha_mid * dx)
    return Phi * alpha * np.exp(-cum_depth)

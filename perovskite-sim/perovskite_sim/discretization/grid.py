from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Layer:
    thickness: float   # metres
    N: int             # number of grid intervals in this layer


def tanh_grid(N: int, L: float, alpha: float = 3.0) -> np.ndarray:
    """N+1 points on [0, L] with boundary concentration parameter alpha."""
    xi = np.linspace(-1.0, 1.0, N + 1)
    x = L * (1.0 + np.tanh(alpha * xi) / np.tanh(alpha)) / 2.0
    return x


def multilayer_grid(layers: list[Layer], alpha: float = 3.0) -> np.ndarray:
    """Concatenated tanh grid for a stack of layers."""
    segments = []
    offset = 0.0
    for k, layer in enumerate(layers):
        x_seg = tanh_grid(layer.N, layer.thickness, alpha) + offset
        if k > 0:
            x_seg = x_seg[1:]   # drop duplicate interface point
        segments.append(x_seg)
        offset += layer.thickness
    return np.concatenate(segments)

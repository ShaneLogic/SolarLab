"""Independent high-precision Gauss-law check of all four charge increments."""

from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.ion_aware_impedance import _potential_increment
from perovskite_sim.physics.poisson import factor_poisson


@pytest.mark.parametrize("polarity", [-1, 1])
def test_nonlinear_increment_matches_high_precision_integrated_gauss_law(polarity):
    x = np.array([0.0, 1e-11, 3e-9, 7e-8, 2e-7])
    eps_r = np.array([4.0, 4.0, 10.0, 17.0, 17.0])
    base = np.repeat([1e25, 2e24, 3e24, 4e24], x.size)
    coordinate = np.array([
        0, 1e-12, -2e-12, 3e-12, 0,
        0, -2e-12, 3e-12, 1e-12, 0,
        0, 3e-12, 1e-12, -2e-12, 0,
        0, -1e-12, 2e-12, 1e-12, 0,
    ])
    voltage_increment = 1e-12
    material = SimpleNamespace(poisson_factor=factor_poisson(x, eps_r), junction_polarity=polarity)
    layout = SimpleNamespace(n_nodes=x.size, state_indices=tuple(range(base.size)))
    delta_phi = _potential_increment(material, layout, base, coordinate, voltage_increment)
    eps_face = EPS_0 * 2 * eps_r[:-1] * eps_r[1:] / (eps_r[:-1] + eps_r[1:])
    actual = -polarity * eps_face * np.diff(delta_phi) / np.diff(x)

    # Integrate Gauss' law and enforce the voltage integral. This reference
    # uses no Poisson matrix, factorization or floating-point density subtraction.
    with localcontext() as context:
        context.prec = 70
        def dec(value):
            return Decimal(str(value))
        dx = [dec(right) - dec(left) for left, right in zip(x[:-1], x[1:])]
        eps = [dec(EPS_0) * 2 * dec(left) * dec(right) / (dec(left) + dec(right))
               for left, right in zip(eps_r[:-1], eps_r[1:])]
        charge = [sum(dec(sign) * dec(Q) * dec(base[block * x.size + node])
                      * (dec(coordinate[block * x.size + node]).exp() - 1)
                      for block, sign in enumerate([-1, 1, 1, -1]))
                  for node in range(x.size)]
        prefix = [Decimal(0)]
        for node in range(1, x.size - 1):
            prefix.append(prefix[-1] + charge[node] * (dx[node - 1] + dx[node]) / 2)
        series = [width / permittivity for width, permittivity in zip(dx, eps)]
        left = (dec(polarity) * dec(voltage_increment)
                - sum(value * weight for value, weight in zip(prefix, series))) / sum(series)
        expected = np.array([float(dec(polarity) * (left + value)) for value in prefix])

    scale = float(np.max(np.abs(expected)))
    assert np.max(np.abs(actual - expected)) / scale < 2e-10
    assert delta_phi[0] == 0.0
    assert delta_phi[-1] == -polarity * voltage_increment

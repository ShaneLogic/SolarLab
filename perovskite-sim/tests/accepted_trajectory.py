"""Observe Radau endpoints under docs/AcceptedTrajectoryValidationContractV1.md."""

from contextlib import contextmanager

import numpy as np
import pytest
from scipy.integrate import Radau


class AcceptedTrajectoryMonitor:
    def __init__(
        self, n_nodes, initial, *, weights=None, ion_capacities=(), inventory_rtol=1e-10
    ):
        self.n_nodes = n_nodes
        self.capacities = tuple(
            np.asarray(capacity).reshape(n_nodes) for capacity in ion_capacities
        )
        self.blocks = 2 + len(self.capacities)
        initial = np.asarray(initial).reshape(self.blocks, n_nodes)
        self.weights = None if weights is None else np.asarray(weights).reshape(n_nodes)
        if self.capacities and self.weights is None:
            raise ValueError(
                "ion trajectory checks require physical integration weights"
            )
        if any(np.any(capacity <= 0.0) for capacity in self.capacities):
            raise ValueError("ion capacities must be positive")
        self.initial_inventory = [float(self.weights @ block) for block in initial[2:]]
        self.inventory_rtol = inventory_rtol
        self.times = []
        self.minima = np.full(self.blocks, np.inf)
        self.maximum_site_fraction = np.zeros(len(self.capacities))
        self.maximum_inventory_drift = np.zeros(len(self.capacities))
        self.maximum_inventory_change = np.zeros(len(self.capacities))
        self.violations = set()
        self.first_violation = None

    def _violation(self, kind, time):
        self.violations.add(kind)
        if self.first_violation is None:
            self.first_violation = {"kind": kind, "time_s": float(time)}

    def observe(self, time, state):
        values = np.asarray(state).reshape(self.blocks, self.n_nodes)
        self.times.append(float(time))
        if not np.all(np.isfinite(values)):
            self._violation("nonfinite_state", time)
            return
        self.minima = np.minimum(self.minima, np.min(values, axis=1))
        for index, name in enumerate(("electron", "hole")):
            if np.any(values[index] <= 0.0):
                self._violation(f"{name}_nonpositive", time)
        for index, capacity in enumerate(self.capacities):
            density = values[index + 2]
            if np.any(density < 0.0):
                self._violation(f"ion_{index}_negative", time)
            fraction = float(np.max(density / capacity))
            self.maximum_site_fraction[index] = max(
                self.maximum_site_fraction[index], fraction
            )
            if fraction > 1.0:
                self._violation(f"ion_{index}_capacity", time)
            inventory = float(self.weights @ density)
            reference = self.initial_inventory[index]
            self.maximum_inventory_change[index] = max(
                self.maximum_inventory_change[index], abs(inventory - reference)
            )
            if reference > 0.0:
                drift = abs(inventory / reference - 1.0)
                self.maximum_inventory_drift[index] = max(
                    self.maximum_inventory_drift[index], drift
                )
                if drift > self.inventory_rtol:
                    self._violation(f"ion_{index}_inventory", time)
            elif np.any(density != 0.0):
                self._violation(f"ion_{index}_zero_population_changed", time)

    def report(self):
        violations = sorted(self.violations)
        if len(self.times) < 2:
            violations.append("no_accepted_steps")
        return {
            "passed": not violations,
            "accepted_step_count": max(0, len(self.times) - 1),
            "accepted_times_s": self.times.copy(),
            "minimum_density_m3": [
                float(value) if np.isfinite(value) else None for value in self.minima
            ],
            "maximum_ion_site_fraction": self.maximum_site_fraction.tolist(),
            "initial_ion_inventory": self.initial_inventory.copy(),
            "maximum_relative_ion_inventory_drift": [
                float(value) if reference > 0.0 else None
                for value, reference in zip(
                    self.maximum_inventory_drift, self.initial_inventory
                )
            ],
            "maximum_absolute_ion_inventory_change": self.maximum_inventory_change.tolist(),
            "ion_inventory_rtol": self.inventory_rtol,
            "violations": violations,
            "first_violation": self.first_violation,
        }

    def assert_valid(self):
        report = self.report()
        assert report["passed"], report


@contextmanager
def observe_accepted_radau(module, monitor):
    original = module.solve_ivp

    class ObservedRadau(Radau):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            monitor.observe(self.t, self.y)

        def step(self):
            message = super().step()
            if self.status != "failed":
                monitor.observe(self.t, self.y)
            return message

    def solve(*args, **kwargs):
        if kwargs.get("method") != "Radau":
            raise AssertionError(
                "accepted-trajectory observer expects the declared Radau method"
            )
        return original(*args, **dict(kwargs, method=ObservedRadau))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "solve_ivp", solve)
        yield monitor

"""The R1 Newton closure includes the electrodes as well as internal faces."""
from types import SimpleNamespace

import numpy as np

from perovskite_sim.experiments.one_dimensional_mechanism_r1 import PhysicalInterfaceIonSystem
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem


def test_internal_face_agreement_cannot_hide_contact_mismatch(monkeypatch):
    internal = np.array([1., 1.])
    raw = (np.zeros(2), internal, np.zeros((1, 2)), np.zeros((1, 2)), 0., 0.)
    monkeypatch.setattr(PhysicalInterfaceIonSystem, "transient_current_metrics", lambda *args: raw)
    system = object.__new__(ControlledPhysicalInterfaceIonSystem)
    system.material = SimpleNamespace(poisson_factor=SimpleNamespace(C=np.ones(2)))
    system.widths = np.ones(3)
    system.polarity = 1.
    system.storage_increment = lambda *args: np.zeros(3)
    system._increment_charge_density = lambda *args: np.zeros(3)
    system.potential_increment = lambda *args: np.zeros(3)
    state = SimpleNamespace(current_n=np.array([1., 2.]), current_p=np.zeros(2))
    result = system.transient_current_metrics(state, state, 1.)
    assert result[4] == .5  # range(1,1,1,2) / max(1,1,1,2)
    for observed, expected in zip(result[:4], raw[:4]):
        np.testing.assert_array_equal(observed, expected)
    assert result[5] == raw[5]

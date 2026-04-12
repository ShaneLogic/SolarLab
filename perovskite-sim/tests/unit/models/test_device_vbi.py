"""Tests for DeviceStack.compute_V_bi()."""
from __future__ import annotations

import pytest
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.models.device import DeviceStack, LayerSpec


def _minimal_params(**overrides) -> MaterialParams:
    """Return a MaterialParams with sensible defaults, overridden by kwargs."""
    defaults = dict(
        eps_r=24.1,
        mu_n=2e-4,
        mu_p=2e-4,
        D_ion=0.0,
        P_lim=0.0,
        P0=0.0,
        ni=1e11,
        tau_n=3e-9,
        tau_p=3e-9,
        n1=1e11,
        p1=1e11,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=0.0,
        Eg=0.0,
    )
    defaults.update(overrides)
    return MaterialParams(**defaults)


class TestComputeVbi:
    def test_compute_vbi_ionmonger_stack(self):
        """IonMonger-like n-i-p stack should give V_bi in the 0.8–1.5 V range.

        Uses ni=1 for HTL/ETL (wide-bandgap transport layers) matching the
        ionmonger_benchmark.yaml config (Courtier 2019).
        """
        htl = LayerSpec(
            name="HTL",
            thickness=200e-9,
            params=_minimal_params(chi=2.1, Eg=3.0, N_A=1e24, ni=1.0),
            role="HTL",
        )
        absorber = LayerSpec(
            name="Absorber",
            thickness=400e-9,
            params=_minimal_params(chi=3.7, Eg=1.6, ni=2.89e10),
            role="absorber",
        )
        etl = LayerSpec(
            name="ETL",
            thickness=100e-9,
            params=_minimal_params(chi=4.0, Eg=3.2, N_D=1e24, ni=1.0),
            role="ETL",
        )
        stack = DeviceStack(layers=(htl, absorber, etl))
        v_bi = stack.compute_V_bi()
        assert 0.8 < v_bi < 1.5, f"V_bi = {v_bi} out of expected range"

    def test_compute_vbi_zero_offsets_falls_back(self):
        """When all chi and Eg are zero, fall back to the manual V_bi field."""
        layer_a = LayerSpec(
            name="L1",
            thickness=100e-9,
            params=_minimal_params(chi=0.0, Eg=0.0, N_A=1e20),
            role="HTL",
        )
        layer_b = LayerSpec(
            name="L2",
            thickness=100e-9,
            params=_minimal_params(chi=0.0, Eg=0.0, N_D=1e20),
            role="ETL",
        )
        stack = DeviceStack(layers=(layer_a, layer_b), V_bi=1.1)
        assert stack.compute_V_bi() == pytest.approx(1.1)

    def test_compute_vbi_symmetric_doping_gives_positive(self):
        """Symmetric p-type / n-type contacts should still yield V_bi > 0."""
        p_layer = LayerSpec(
            name="p-contact",
            thickness=100e-9,
            params=_minimal_params(chi=3.9, Eg=1.6, N_A=1e22, ni=1e11),
            role="HTL",
        )
        n_layer = LayerSpec(
            name="n-contact",
            thickness=100e-9,
            params=_minimal_params(chi=3.9, Eg=1.6, N_D=1e22, ni=1e11),
            role="ETL",
        )
        stack = DeviceStack(layers=(p_layer, n_layer))
        v_bi = stack.compute_V_bi()
        assert v_bi > 0, f"V_bi = {v_bi} should be positive"

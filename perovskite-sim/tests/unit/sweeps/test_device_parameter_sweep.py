from __future__ import annotations

import dataclasses

import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
    cm3_to_m3,
    cms_to_ms,
    make_pilot_points,
    srh_n1_p1_from_trap_depth,
)


def _stack():
    return load_device_from_yaml("configs/solarscale_nip_band_aligned.yaml")


def _by_role(stack, role):
    return next(layer for layer in electrical_layers(stack) if layer.role == role)


def test_unit_conversions():
    assert cm3_to_m3(1e10) == pytest.approx(1e16)
    assert cms_to_ms(1e7) == pytest.approx(1e5)


def test_trap_depth_midgap_maps_to_ni():
    n1, p1 = srh_n1_p1_from_trap_depth(3.2e13, 1.6, 0.0, reference="midgap")
    assert n1 == pytest.approx(3.2e13)
    assert p1 == pytest.approx(3.2e13)


def test_trap_depth_below_cb_keeps_product_equal_ni_squared():
    ni = 3.2e13
    n1, p1 = srh_n1_p1_from_trap_depth(ni, 1.6, 0.3, reference="below_cb")
    assert n1 * p1 == pytest.approx(ni**2)
    assert n1 > p1


def test_apply_etl_delta_ec_sets_chi_and_syncs_vbi():
    stack = _stack()
    absorber = _by_role(stack, "absorber").params
    point = SweepPoint("p", "etl_delta_ec", "0.25 eV", {"etl_delta_ec_eV": 0.25})
    swept = apply_sweep_point(stack, point)
    etl = _by_role(swept, "ETL").params
    assert etl.chi == pytest.approx(absorber.chi - 0.25)
    assert swept.V_bi == pytest.approx(swept.compute_V_bi())


def test_apply_htl_delta_ev_sets_valence_offset():
    stack = _stack()
    absorber = _by_role(stack, "absorber").params
    point = SweepPoint("p", "htl_delta_ev", "0.2 eV", {"htl_delta_ev_eV": 0.2})
    swept = apply_sweep_point(stack, point)
    htl = _by_role(swept, "HTL").params
    delta_ev = absorber.chi + absorber.Eg - htl.chi - htl.Eg
    assert delta_ev == pytest.approx(0.2)


def test_apply_doping_and_srv():
    stack = _stack()
    point = SweepPoint(
        "p",
        "combined",
        "doping and interface",
        {"etl_doping_cm3": 1e18, "absorber_doping_cm3": 1e12, "interface_srv_cm_s": 1e4},
    )
    swept = apply_sweep_point(stack, point)
    etl = _by_role(swept, "ETL").params
    absorber = _by_role(swept, "absorber").params
    assert etl.N_D == pytest.approx(1e24)
    assert etl.N_A == pytest.approx(0.0)
    assert absorber.N_A == pytest.approx(1e18)
    assert absorber.N_D == pytest.approx(0.0)
    assert any(pair == pytest.approx((1e2, 1e2)) for pair in swept.interfaces)


def test_apply_absorber_defect_density_scales_tau():
    stack = _stack()
    absorber0 = _by_role(stack, "absorber").params
    point = SweepPoint(
        "p",
        "absorber_defect_density",
        "1e18 cm^-3",
        {"absorber_defect_density_cm3": 1e18},
    )
    swept = apply_sweep_point(stack, point)
    absorber = _by_role(swept, "absorber").params
    assert absorber.trap_N_t_bulk == pytest.approx(1e24)
    assert absorber.tau_n == pytest.approx(absorber0.tau_n * 1e22 / 1e24)
    assert absorber.tau_p == pytest.approx(absorber0.tau_p * 1e22 / 1e24)


def test_pilot_points_cover_requested_axes():
    axes = {point.axis for point in make_pilot_points()}
    assert {
        "etl_delta_ec",
        "htl_delta_ev",
        "etl_doping",
        "absorber_doping",
        "absorber_defect_depth",
        "absorber_defect_density",
        "interface_trap_density",
        "interface_srv",
    } <= axes

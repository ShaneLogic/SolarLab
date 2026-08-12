"""The standard YAML loader must understand the same interface schema as the
inline-device path the frontend uses.

Before this, ``models/config_loader.py`` parsed ``device.interfaces`` (raw SRV
pairs) but never ``device.interface_defects``, so a plain YAML config could not
express a trap level. That gap is not cosmetic: an ``InterfaceDefect`` does not
merely supply different n1/p1 numbers, it activates the E_t-aware cross-carrier
evaluation path, and it carries ~95 % of the interface-recombination effect
(measured -143 mV of V_oc versus -6.8 mV for the bare SRV on the band-aligned
screening template). The two paths sharing one parser is also what keeps them
from drifting, which they have done before.
"""
from __future__ import annotations

import textwrap

import pytest

from perovskite_sim.models.config_loader import (
    interfaces_from_device_dict,
    load_device_from_yaml,
)
from perovskite_sim.models.device import (
    InterfaceDefect,
    electrical_interface_defects,
    electrical_interfaces,
)

# One SCAPS-style defect block and the SRV it must produce:
#   v = sigma[cm^2] * v_th[cm/s] * N_t[cm^-2] * 1e-2   ->  m/s
DEFECT = {
    "sigma_n_cm2": 1.0e-19,
    "sigma_p_cm2": 2.0e-19,
    "v_th_cm_s": 1.0e7,
    "N_t_cm2": 1.0e12,
    "E_t_eV_below_cb": 0.8,
}
V_N_EXPECTED = 1.0e-19 * 1.0e7 * 1.0e12 * 1.0e-2      # 1e-2 m/s
V_P_EXPECTED = 2.0e-19 * 1.0e7 * 1.0e12 * 1.0e-2      # 2e-2 m/s


class TestParser:
    """Unit-level behaviour of the shared parser."""

    def test_absent_keys_are_legacy_empty(self):
        """No interface keys at all must stay bit-identical to the old path."""
        assert interfaces_from_device_dict({}, n_layers=4) == ((), ())

    def test_legacy_srv_pairs_only(self):
        dev = {"interfaces": [[0.0, 0.0], [0.1, 0.2], [0.3, 0.4]]}
        ifaces, defects = interfaces_from_device_dict(dev, n_layers=4)
        assert ifaces == ((0.0, 0.0), (0.1, 0.2), (0.3, 0.4))
        assert defects == ()          # no defect schema supplied

    def test_defect_block_derives_srv_and_trap(self):
        dev = {"interface_defects": [None, dict(DEFECT), None]}
        ifaces, defects = interfaces_from_device_dict(dev, n_layers=4)
        assert ifaces[1] == pytest.approx((V_N_EXPECTED, V_P_EXPECTED))
        assert ifaces[0] == (0.0, 0.0) and ifaces[2] == (0.0, 0.0)
        assert defects[1] == InterfaceDefect(
            E_t_eV=0.8,
            calibration_factor=1.0,
            N_t_cm2=1.0e12,
        )
        assert defects[0] is None and defects[2] is None

    def test_defect_takes_precedence_over_legacy_pair(self):
        dev = {
            "interfaces": [[0.0, 0.0], [9.9, 9.9], [0.5, 0.5]],
            "interface_defects": [None, dict(DEFECT), None],
        }
        ifaces, defects = interfaces_from_device_dict(dev, n_layers=4)
        assert ifaces[1] == pytest.approx((V_N_EXPECTED, V_P_EXPECTED))
        assert ifaces[2] == (0.5, 0.5)          # legacy survives on null slots
        assert defects[1] is not None and defects[2] is None

    def test_calibration_factor_roundtrips(self):
        dev = {"interface_defects": [dict(DEFECT, calibration_factor=0.02)]}
        _, defects = interfaces_from_device_dict(dev, n_layers=2)
        assert defects[0].calibration_factor == pytest.approx(0.02)

    def test_short_lists_pad_to_the_interface_count(self):
        dev = {"interfaces": [[0.1, 0.1]]}
        ifaces, _ = interfaces_from_device_dict(dev, n_layers=4)
        assert len(ifaces) == 3
        assert ifaces[1] == (0.0, 0.0) and ifaces[2] == (0.0, 0.0)


class TestYamlRoundTrip:
    """The loader must carry both schemas through to the DeviceStack."""

    def _write(self, tmp_path, device_extra: str = ""):
        # Dedent FIRST, then splice: interpolating into the template before
        # dedent lets the injected block's own indentation redefine the common
        # prefix and corrupts the whole document.
        body = textwrap.dedent("""\
            device:
              V_bi: 1.0
              Phi: 2.5e21
              mode: fast
            layers:
              - name: htl
                role: HTL
                thickness: 200.0e-9
                eps_r: 3.0
                mu_n: 1.0e-10
                mu_p: 1.0e-6
                ni: 1.0
                N_D: 0.0
                N_A: 2.0e23
                chi: 2.05
                Eg: 3.05
                D_ion: 0.0
                P_lim: 1.0e30
                P0: 0.0
                tau_n: 1.0e-9
                tau_p: 1.0e-9
                n1: 1.0
                p1: 1.0
                B_rad: 0.0
                C_n: 0.0
                C_p: 0.0
                alpha: 0.0
              - name: abs
                role: absorber
                thickness: 400.0e-9
                eps_r: 24.1
                mu_n: 2.0e-4
                mu_p: 2.0e-4
                ni: 3.2e13
                N_D: 0.0
                N_A: 0.0
                chi: 3.7
                Eg: 1.6
                D_ion: 0.0
                P_lim: 1.0e30
                P0: 0.0
                tau_n: 1.0e-6
                tau_p: 1.0e-6
                n1: 3.2e13
                p1: 3.2e13
                B_rad: 0.0
                C_n: 0.0
                C_p: 0.0
                alpha: 1.3e7
              - name: etl
                role: ETL
                thickness: 100.0e-9
                eps_r: 10.0
                mu_n: 1.0e-5
                mu_p: 1.0e-10
                ni: 1.0
                N_D: 1.0e24
                N_A: 0.0
                chi: 4.0
                Eg: 3.2
                D_ion: 0.0
                P_lim: 1.0e30
                P0: 0.0
                tau_n: 1.0e-9
                tau_p: 1.0e-9
                n1: 1.0
                p1: 1.0
                B_rad: 0.0
                C_n: 0.0
                C_p: 0.0
                alpha: 0.0
            """)
        if device_extra:
            body = body.replace("layers:\n", device_extra + "layers:\n", 1)
        p = tmp_path / "cfg.yaml"
        p.write_text(body)
        return p

    def test_no_interface_keys_loads_empty(self, tmp_path):
        stack = load_device_from_yaml(str(self._write(tmp_path)))
        assert stack.interfaces == ()
        assert stack.interface_defects == ()

    def test_defects_reach_the_stack(self, tmp_path):
        # dedent to column 0 first, THEN indent: dedent folds any leading
        # indentation into the common prefix and strips it away.
        extra = textwrap.indent(textwrap.dedent("""\
            interface_defects:
              - sigma_n_cm2: 1.0e-19
                sigma_p_cm2: 2.0e-19
                v_th_cm_s: 1.0e7
                N_t_cm2: 1.0e12
                E_t_eV_below_cb: 0.8
              -
            """), "  ")
        stack = load_device_from_yaml(str(self._write(tmp_path, extra)))
        assert stack.interfaces[0] == pytest.approx((V_N_EXPECTED, V_P_EXPECTED))
        assert stack.interface_defects[0].E_t_eV == pytest.approx(0.8)
        assert stack.interface_defects[1] is None
        # no substrate here, so the electrical views are the full tuples
        assert electrical_interfaces(stack) == stack.interfaces
        assert electrical_interface_defects(stack) == stack.interface_defects


class TestShippedConfigsUnchanged:
    """Every config in the repo must load exactly as it did before."""

    @pytest.mark.parametrize("name", [
        "nip_MAPbI3", "pin_MAPbI3", "ionmonger_benchmark",
        "solarscale_nip_band_aligned", "cigs_baseline",
    ])
    def test_configs_without_the_key_stay_empty(self, name):
        stack = load_device_from_yaml(f"configs/{name}.yaml")
        assert stack.interface_defects == ()


class TestParityWithInlinePath:
    """YAML and the frontend's inline-device path must agree — they have
    drifted before, which is why they now share one parser."""

    def test_backend_inline_matches_the_shared_parser(self):
        from backend.main import stack_from_dict

        dev = {
            "V_bi": 1.0, "Phi": 2.5e21, "mode": "fast",
            "interfaces": [[0.0, 0.0], [0.7, 0.7]],
            "interface_defects": [None, dict(DEFECT)],
        }
        layer = dict(
            name="l", role="absorber", thickness=4.0e-7, eps_r=24.1,
            mu_n=2e-4, mu_p=2e-4, ni=3.2e13, N_D=0.0, N_A=0.0, chi=3.7, Eg=1.6,
            D_ion=0.0, P_lim=1e30, P0=0.0, tau_n=1e-6, tau_p=1e-6,
            n1=3.2e13, p1=3.2e13, B_rad=0.0, C_n=0.0, C_p=0.0, alpha=1.3e7,
        )
        cfg = {"device": dev, "layers": [layer, dict(layer, name="m"),
                                         dict(layer, name="n")]}
        stack = stack_from_dict(cfg)
        ifaces, defects = interfaces_from_device_dict(dev, n_layers=3)
        # pytest.approx refuses nested tuples, so compare slot by slot
        assert len(stack.interfaces) == len(ifaces)
        for got, want in zip(stack.interfaces, ifaces):
            assert got == pytest.approx(want)
        assert stack.interface_defects == defects

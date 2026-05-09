"""R5: 1D vs 2D matched-stack J-V (spec § 9, R5)."""
import json
from pathlib import Path

import yaml

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams

PEROVSKITE_SIM_ROOT = Path(__file__).resolve().parents[3] / "perovskite-sim"
CFG_1D = PEROVSKITE_SIM_ROOT / "configs" / "nip_MAPbI3.yaml"
CFG_2D = PEROVSKITE_SIM_ROOT / "configs" / "twod" / "nip_MAPbI3_singleGB.yaml"
OUT = Path(__file__).resolve().parents[1] / "figures" / "data" / "r5_jv_curves.json"


def stack_from_dict(cfg: dict) -> DeviceStack:
    """Build a DeviceStack from a YAML config dict (simplified from backend.main)."""
    dev = cfg.get("device", {}) or {}
    layers: list[LayerSpec] = []

    for layer_cfg in cfg.get("layers", []) or []:
        p = MaterialParams(
            eps_r=float(layer_cfg["eps_r"]),
            mu_n=float(layer_cfg["mu_n"]),
            mu_p=float(layer_cfg["mu_p"]),
            D_ion=float(layer_cfg["D_ion"]),
            P_lim=float(layer_cfg["P_lim"]),
            P0=float(layer_cfg["P0"]),
            ni=float(layer_cfg["ni"]),
            tau_n=float(layer_cfg["tau_n"]),
            tau_p=float(layer_cfg["tau_p"]),
            n1=float(layer_cfg["n1"]),
            p1=float(layer_cfg["p1"]),
            B_rad=float(layer_cfg["B_rad"]),
            C_n=float(layer_cfg["C_n"]),
            C_p=float(layer_cfg["C_p"]),
            alpha=float(layer_cfg["alpha"]),
            N_A=float(layer_cfg["N_A"]),
            N_D=float(layer_cfg["N_D"]),
            chi=float(layer_cfg.get("chi", 0.0)),
            Eg=float(layer_cfg.get("Eg", 0.0)),
        )
        layers.append(
            LayerSpec(
                name=str(layer_cfg["name"]),
                thickness=float(layer_cfg["thickness"]),
                params=p,
                role=str(layer_cfg.get("role", "absorber")),
            )
        )

    interfaces = tuple(
        (float(iface.get("v_n", 0.0)), float(iface.get("v_p", 0.0)))
        for iface in dev.get("interfaces", [])
    )

    stack = DeviceStack(
        layers=tuple(layers),
        interfaces=interfaces,
        V_bi=float(dev.get("V_bi", 0.0)),
    )
    return stack


def main():
    with open(CFG_1D) as f:
        cfg_dict_1d = yaml.safe_load(f)
    with open(CFG_2D) as f:
        cfg_dict_2d = yaml.safe_load(f)

    stack_1d = stack_from_dict(cfg_dict_1d)
    stack_2d = stack_from_dict(cfg_dict_2d)

    print("Running 1D J-V sweep...")
    res_1d = run_jv_sweep(stack_1d)

    print("Running 2D J-V sweep...")
    res_2d = run_jv_sweep_2d(
        stack_2d,
        lateral_length=500e-9,  # 500 nm grain size
        Nx=30,
        V_max=1.4,
        V_step=0.05,
        Ny_per_layer=20,
        settle_t=1e-7,
        save_snapshots=False,
    )

    # For 1D, use forward scan metrics (standard reference)
    metrics_1d = res_1d.metrics_fwd

    # For 2D, use the single metrics object
    metrics_2d = res_2d.metrics

    # Sign-normalise 2D J so both arrays use 1D's "photocurrent positive at V=0"
    # convention. The 2D solver returns J with the opposite sign; metrics.J_sc
    # already reports the magnitude, but the raw J array still inverts.
    j_2d = [-j for j in res_2d.J]

    # The simulator's metrics.PCE is a *fraction* (e.g. 0.288 means 28.8 %).
    # Convert to percentage for the JSON so downstream figure scripts can
    # annotate "PCE = X.Y %" directly.
    payload = {
        "config_1d": str(CFG_1D),
        "config_2d": str(CFG_2D),
        "v_1d": list(res_1d.V_fwd),
        "j_1d": list(res_1d.J_fwd),
        "v_2d": list(res_2d.V),
        "j_2d": j_2d,
        "metrics_1d": {
            "voc_V": metrics_1d.V_oc,
            "jsc_A_per_m2": metrics_1d.J_sc,
            "ff": metrics_1d.FF,
            "pce_pct": metrics_1d.PCE * 100.0,
        },
        "metrics_2d": {
            "voc_V": metrics_2d.V_oc,
            "jsc_A_per_m2": metrics_2d.J_sc,
            "ff": metrics_2d.FF,
            "pce_pct": metrics_2d.PCE * 100.0,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    print(f"  1D: V_oc={payload['metrics_1d']['voc_V']:.4f} V, "
          f"PCE={payload['metrics_1d']['pce_pct']:.2f} %")
    print(f"  2D: V_oc={payload['metrics_2d']['voc_V']:.4f} V, "
          f"PCE={payload['metrics_2d']['pce_pct']:.2f} %")


if __name__ == "__main__":
    main()

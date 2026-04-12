from __future__ import annotations
import yaml
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.models.device import DeviceStack, LayerSpec


def _f(v) -> float:
    """Cast YAML value to float (handles string scientific notation)."""
    return float(v)


def load_device_from_yaml(path: str) -> DeviceStack:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    dev = cfg["device"]
    layers = []
    for layer_cfg in cfg["layers"]:
        p = MaterialParams(
            eps_r=_f(layer_cfg["eps_r"]),
            mu_n=_f(layer_cfg["mu_n"]),
            mu_p=_f(layer_cfg["mu_p"]),
            D_ion=_f(layer_cfg["D_ion"]),
            P_lim=_f(layer_cfg["P_lim"]),
            P0=_f(layer_cfg["P0"]),
            ni=_f(layer_cfg["ni"]),
            tau_n=_f(layer_cfg["tau_n"]),
            tau_p=_f(layer_cfg["tau_p"]),
            n1=_f(layer_cfg["n1"]),
            p1=_f(layer_cfg["p1"]),
            B_rad=_f(layer_cfg["B_rad"]),
            C_n=_f(layer_cfg["C_n"]),
            C_p=_f(layer_cfg["C_p"]),
            alpha=_f(layer_cfg["alpha"]),
            N_A=_f(layer_cfg["N_A"]),
            N_D=_f(layer_cfg["N_D"]),
            chi=_f(layer_cfg.get("chi", 0.0)),
            Eg=_f(layer_cfg.get("Eg", 0.0)),
            A_star_n=_f(layer_cfg.get("A_star_n", 1.2017e6)),
            A_star_p=_f(layer_cfg.get("A_star_p", 1.2017e6)),
            D_ion_neg=_f(layer_cfg.get("D_ion_neg", 0.0)),
            P0_neg=_f(layer_cfg.get("P0_neg", 0.0)),
            P_lim_neg=_f(layer_cfg.get("P_lim_neg", 1e30)),
            Nc300=float(layer_cfg["Nc300"]) if "Nc300" in layer_cfg else None,
            Nv300=float(layer_cfg["Nv300"]) if "Nv300" in layer_cfg else None,
            mu_T_gamma=_f(layer_cfg.get("mu_T_gamma", -1.5)),
            E_a_ion=_f(layer_cfg.get("E_a_ion", 0.58)),
            trap_N_t_interface=float(layer_cfg["trap_N_t_interface"]) if "trap_N_t_interface" in layer_cfg else None,
            trap_N_t_bulk=float(layer_cfg["trap_N_t_bulk"]) if "trap_N_t_bulk" in layer_cfg else None,
            trap_decay_length=float(layer_cfg["trap_decay_length"]) if "trap_decay_length" in layer_cfg else None,
            optical_material=layer_cfg.get("optical_material"),
            n_optical=float(layer_cfg["n_optical"]) if "n_optical" in layer_cfg else None,
        )
        layers.append(LayerSpec(
            name=layer_cfg["name"],
            thickness=_f(layer_cfg["thickness"]),
            params=p,
            role=layer_cfg["role"],
        ))
    # Interface recombination velocities: list of [v_n, v_p] per internal interface
    raw_interfaces = dev.get("interfaces", [])
    interfaces = tuple(
        (float(pair[0]), float(pair[1])) for pair in raw_interfaces
    )
    return DeviceStack(
        layers=layers,
        V_bi=_f(dev.get("V_bi", 1.1)),
        Phi=_f(dev.get("Phi", 2.5e21)),
        interfaces=interfaces,
        T=_f(dev.get("T", 300.0)),
        mode=str(dev.get("mode", "full")),
    )

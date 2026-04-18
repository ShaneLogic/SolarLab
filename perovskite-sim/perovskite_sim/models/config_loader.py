from __future__ import annotations
import yaml
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.models.device import DeviceStack, LayerSpec


def load_simulation_hints(path: str) -> dict:
    """Read the optional top-level ``simulation_hints`` block from a
    device YAML and return it as a dict (empty if absent).

    The hints surface solver-level recommendations that are tied to the
    physical stack rather than to any individual experiment. Recognised
    keys today:

    - ``min_N_grid`` : int. Minimum per-stack grid count below which the
      default drift-diffusion solve is known to produce unphysical
      results (e.g. thick c-Si wafers, multi-µm CIGS absorbers where
      the absorber-to-Debye-length ratio is large). The backend/UI can
      raise the user-facing N_grid default to this value, or warn when
      a user drops below it.
    - ``notes`` : str. Free-text caveats for the UI to surface next to
      the Run button.

    Hints are advisory and have no effect on a pure library-level
    ``run_*`` call — the solver does not read them. Consumers that want
    to enforce them should do so explicitly.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return dict(cfg.get("simulation_hints", {}) or {})


def _f(v) -> float:
    """Cast YAML value to float (handles string scientific notation)."""
    return float(v)


def _parse_bool(v) -> bool:
    """Parse a YAML value as bool, tolerating quoted strings like "false".

    PyYAML 1.1 normally returns Python bools for bare true/false, but a
    quoted "false" comes through as a str and bool("false") is True.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "on"}
    return bool(v)


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
            B_rad_T_gamma=_f(layer_cfg.get("B_rad_T_gamma", 0.0)),
            varshni_alpha=_f(layer_cfg.get("varshni_alpha", 0.0)),
            varshni_beta=_f(layer_cfg.get("varshni_beta", 0.0)),
            trap_N_t_interface=float(layer_cfg["trap_N_t_interface"]) if "trap_N_t_interface" in layer_cfg else None,
            trap_N_t_bulk=float(layer_cfg["trap_N_t_bulk"]) if "trap_N_t_bulk" in layer_cfg else None,
            trap_decay_length=float(layer_cfg["trap_decay_length"]) if "trap_decay_length" in layer_cfg else None,
            trap_profile_shape=str(layer_cfg.get("trap_profile_shape", "exponential")),
            optical_material=layer_cfg.get("optical_material"),
            n_optical=float(layer_cfg["n_optical"]) if "n_optical" in layer_cfg else None,
            incoherent=_parse_bool(layer_cfg.get("incoherent", False)),
            v_sat_n=_f(layer_cfg.get("v_sat_n", 0.0)),
            v_sat_p=_f(layer_cfg.get("v_sat_p", 0.0)),
            ct_beta_n=_f(layer_cfg.get("ct_beta_n", 2.0)),
            ct_beta_p=_f(layer_cfg.get("ct_beta_p", 2.0)),
            pf_gamma_n=_f(layer_cfg.get("pf_gamma_n", 0.0)),
            pf_gamma_p=_f(layer_cfg.get("pf_gamma_p", 0.0)),
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

    # Selective / Schottky contact surface recombination velocities
    # (Phase 3.3 — Apr 2026). YAML may supply them as a nested
    # ``contacts:`` block or as flat top-level device keys. Missing
    # keys leave the contact as ohmic Dirichlet (None sentinel) so
    # pre-3.3 configs load unchanged.
    def _opt_S(value):
        if value is None:
            return None
        return float(value)

    contacts_cfg = dev.get("contacts", {}) or {}
    left_cfg = contacts_cfg.get("left", {}) or {}
    right_cfg = contacts_cfg.get("right", {}) or {}
    S_n_left = _opt_S(dev.get("S_n_left", left_cfg.get("S_n")))
    S_p_left = _opt_S(dev.get("S_p_left", left_cfg.get("S_p")))
    S_n_right = _opt_S(dev.get("S_n_right", right_cfg.get("S_n")))
    S_p_right = _opt_S(dev.get("S_p_right", right_cfg.get("S_p")))

    return DeviceStack(
        layers=layers,
        V_bi=_f(dev.get("V_bi", 1.1)),
        Phi=_f(dev.get("Phi", 2.5e21)),
        interfaces=interfaces,
        T=_f(dev.get("T", 300.0)),
        mode=str(dev.get("mode", "full")),
        S_n_left=S_n_left,
        S_p_left=S_p_left,
        S_n_right=S_n_right,
        S_p_right=S_p_right,
    )

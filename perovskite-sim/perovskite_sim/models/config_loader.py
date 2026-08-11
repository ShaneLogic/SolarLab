from __future__ import annotations
from collections.abc import Mapping, Sequence
import math
from typing import Any, Optional

import yaml
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.physics.doping import validate_doping_profile_params
from perovskite_sim.twod.microstructure import load_microstructure_from_yaml_block


def electrical_grid_from_config_dict(
    cfg: Mapping[str, Any],
    layers: Sequence[LayerSpec],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Parse the optional top-level executable ``electrical_grid`` block.

    Both maps are keyed by electrical-layer name and must cover those layers
    exactly; substrate names are neither required nor accepted. Returning
    ordered tuples keeps :class:`DeviceStack` independent of YAML map order.
    An absent block returns empty tuples, preserving the legacy grid path.
    """
    if "electrical_grid" not in cfg:
        return (), ()

    block = cfg["electrical_grid"]
    if not isinstance(block, Mapping):
        raise ValueError("electrical_grid must be a mapping")
    required_keys = {"interval_weights", "alphas"}
    actual_block_keys = set(block)
    if actual_block_keys != required_keys:
        missing = sorted(required_keys - actual_block_keys)
        extra = sorted(repr(key) for key in actual_block_keys - required_keys)
        raise ValueError(
            "electrical_grid must contain exactly interval_weights and alphas; "
            f"missing={missing}, extra={extra}"
        )

    electrical = tuple(layer for layer in layers if layer.role != "substrate")
    layer_names = tuple(layer.name for layer in electrical)
    if len(set(layer_names)) != len(layer_names):
        raise ValueError(
            "electrical_grid requires unique names for all electrical layers"
        )
    expected_names = set(layer_names)

    def _positive_map(key: str) -> tuple[float, ...]:
        raw = block[key]
        if not isinstance(raw, Mapping):
            raise ValueError(f"electrical_grid.{key} must be a mapping")
        actual_names = set(raw)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(repr(name) for name in actual_names - expected_names)
            raise ValueError(
                f"electrical_grid.{key} must cover exactly the electrical "
                f"layers; missing={missing}, extra={extra}"
            )
        values: list[float] = []
        for name in layer_names:
            value = raw[name]
            if isinstance(value, bool):
                raise ValueError(
                    f"electrical_grid.{key}[{name!r}] must be finite and positive"
                )
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"electrical_grid.{key}[{name!r}] must be finite and positive"
                ) from exc
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(
                    f"electrical_grid.{key}[{name!r}] must be finite and positive"
                )
            values.append(number)
        return tuple(values)

    return _positive_map("interval_weights"), _positive_map("alphas")


def interfaces_from_device_dict(
    dev: dict, n_layers: int
) -> tuple[tuple[tuple[float, float], ...], tuple[Optional[InterfaceDefect], ...]]:
    """Parse a ``device:`` block's interface schemas into the two tuples
    ``DeviceStack`` carries.

    This is the single source of truth for both entry points — the YAML loader
    below and ``backend/main.py:stack_from_dict`` (the inline-device path the
    frontend uses). They have drifted apart before; sharing the parser is what
    stops it happening again.

    Two schemas are accepted, both FULL-layer-aligned (one slot per internal
    interface, ``n_layers - 1`` of them, substrate included):

    ``interfaces``
        Legacy list of ``[v_n, v_p]`` surface-recombination velocities in m/s.

    ``interface_defects``
        SCAPS-style list of ``{sigma_n_cm2, sigma_p_cm2, v_th_cm_s, N_t_cm2,
        E_t_eV_below_cb, calibration_factor?}`` or ``None`` per slot. The SRV
        pair is derived from the kinetic identity ``v = sigma * v_th * N_t``
        (the ``1e-2`` converts cm/s to m/s), and an :class:`InterfaceDefect`
        carries the trap depth.

    Where both supply a slot, the defect wins and the legacy pair survives only
    on ``None`` slots. Populating a defect is not a cosmetic difference: it
    activates the E_t-aware cross-carrier evaluation path, which carries most
    of the interface-recombination effect (measured -143 mV of V_oc versus
    -6.8 mV for a bare SRV of the same magnitude).

    Absent keys return ``((), ())`` — byte-identical to the pre-schema
    behaviour, so every config that does not opt in is unaffected.
    """
    legacy_pairs = list(dev.get("interfaces") or [])
    defect_blocks = list(dev.get("interface_defects") or [])
    if not legacy_pairs and not defect_blocks:
        return (), ()

    n_iface = max(0, n_layers - 1)
    iface_list: list[tuple[float, float]] = []
    defect_list: list[Optional[InterfaceDefect]] | None = (
        [] if defect_blocks else None
    )
    for k in range(n_iface):
        pair = legacy_pairs[k] if k < len(legacy_pairs) else (0.0, 0.0)
        block = defect_blocks[k] if k < len(defect_blocks) else None
        if block is None:
            iface_list.append((float(pair[0]), float(pair[1])))
            if defect_list is not None:
                defect_list.append(None)
            continue
        v_th = float(block["v_th_cm_s"])
        N_t = float(block["N_t_cm2"])
        iface_list.append((
            float(block["sigma_n_cm2"]) * v_th * N_t * 1.0e-2,
            float(block["sigma_p_cm2"]) * v_th * N_t * 1.0e-2,
        ))
        assert defect_list is not None
        defect_list.append(InterfaceDefect(
            E_t_eV=float(block["E_t_eV_below_cb"]),
            calibration_factor=float(block.get("calibration_factor", 1.0)),
            iface_state_calibration_factor=float(
                block.get("iface_state_calibration_factor", 1.0)
            ),
            N_t_cm2=N_t,
        ))
    return (
        tuple(iface_list),
        tuple(defect_list) if defect_list is not None else (),
    )


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


def material_params_from_dict(layer_cfg: dict) -> MaterialParams:
    """Build a MaterialParams from one layer dict (standard schema).

    Single source of truth for layer-field parsing, shared by the YAML loader
    here and the backend inline-device path (``backend.main.stack_from_dict``).
    Keeping both on this one function is what stops the two parsers from
    drifting — the inline path historically dropped fields (TE A_star, effective
    DOS, trap profiles, dual-ion, temperature scaling) the loader carried, which
    silently disabled that physics for UI-built devices.
    """
    params = MaterialParams(
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
        v_th=_f(layer_cfg.get("v_th", 1.0e5)),
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
        trap_edge=str(layer_cfg.get("trap_edge", "both")),
        optical_material=layer_cfg.get("optical_material"),
        n_optical=float(layer_cfg["n_optical"]) if "n_optical" in layer_cfg else None,
        incoherent=_parse_bool(layer_cfg.get("incoherent", False)),
        v_sat_n=_f(layer_cfg.get("v_sat_n", 0.0)),
        v_sat_p=_f(layer_cfg.get("v_sat_p", 0.0)),
        ct_beta_n=_f(layer_cfg.get("ct_beta_n", 2.0)),
        ct_beta_p=_f(layer_cfg.get("ct_beta_p", 2.0)),
        pf_gamma_n=_f(layer_cfg.get("pf_gamma_n", 0.0)),
        pf_gamma_p=_f(layer_cfg.get("pf_gamma_p", 0.0)),
        Eg_back=float(layer_cfg["Eg_back"]) if "Eg_back" in layer_cfg else None,
        chi_back=float(layer_cfg["chi_back"]) if "chi_back" in layer_cfg else None,
        grading_profile=str(layer_cfg.get("grading_profile", "linear")),
        grading_direction=str(layer_cfg.get("grading_direction", "front_to_back")),
        grading_bowing=_f(layer_cfg.get("grading_bowing", 0.0)),
        grading_char_length=float(layer_cfg["grading_char_length"]) if "grading_char_length" in layer_cfg else None,
        grading_N_mult=int(layer_cfg.get("grading_N_mult", 1)),
        N_A_bulk=float(layer_cfg["N_A_bulk"]) if "N_A_bulk" in layer_cfg else None,
        N_D_bulk=float(layer_cfg["N_D_bulk"]) if "N_D_bulk" in layer_cfg else None,
        doping_profile_shape=(
            str(layer_cfg["doping_profile_shape"])
            if "doping_profile_shape" in layer_cfg else None
        ),
        doping_decay_length=(
            float(layer_cfg["doping_decay_length"])
            if "doping_decay_length" in layer_cfg else None
        ),
        doping_edge=str(layer_cfg.get("doping_edge", "front")),
    )
    validate_doping_profile_params(params)
    return params


def load_device_from_yaml(path: str) -> DeviceStack:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    dev = cfg["device"]
    layers = []
    for layer_cfg in cfg["layers"]:
        p = material_params_from_dict(layer_cfg)
        layers.append(LayerSpec(
            name=layer_cfg["name"],
            thickness=_f(layer_cfg["thickness"]),
            params=p,
            role=layer_cfg["role"],
        ))
    grid_interval_weights, grid_alphas = electrical_grid_from_config_dict(
        cfg, layers
    )
    # Interface recombination: legacy [v_n, v_p] SRV pairs and/or SCAPS-style
    # defect blocks. Shared with the backend inline-device path so the two
    # cannot drift.
    interfaces, interface_defects = interfaces_from_device_dict(dev, len(layers))

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

    # Lateral microstructure (2D Stage B). Top-level YAML block, optional.
    # Empty / missing → Microstructure() (no GBs); strict-key validation
    # happens inside load_microstructure_from_yaml_block.
    microstructure = load_microstructure_from_yaml_block(cfg.get("microstructure"))

    return DeviceStack(
        layers=layers,
        V_bi=_f(dev.get("V_bi", 1.1)),
        Phi=_f(dev.get("Phi", 2.5e21)),
        grid_interval_weights=grid_interval_weights,
        grid_alphas=grid_alphas,
        jv_solver_policy=str(dev.get("jv_solver_policy", "general")),
        interfaces=interfaces,
        interface_defects=interface_defects,
        T=_f(dev.get("T", 300.0)),
        mode=str(dev.get("mode", "full")),
        interface_plane_projection=(
            str(dev.get("interface_plane_projection", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        dos_band_potentials=(
            str(dev.get("dos_band_potentials", True)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        te_physical_norm=(
            str(dev.get("te_physical_norm", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        ion_steric_diffusion_only=(
            str(dev.get("ion_steric_diffusion_only", True)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        ion_steric_shared_site=(
            str(dev.get("ion_steric_shared_site", True)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        autoloop_generated_lever=(
            str(dev.get("autoloop_generated_lever", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        flat_band_contacts=(
            str(dev.get("flat_band_contacts", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        flat_band_metal_contacts=(
            str(dev.get("flat_band_metal_contacts", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        contact_phi_B_eV=_f(dev.get("contact_phi_B_eV", 0.0)),
        interface_two_sided=(
            str(dev.get("interface_two_sided", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        interface_shared_occupancy=(
            str(dev.get("interface_shared_occupancy", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        interface_plane_closure=(
            str(dev.get("interface_plane_closure", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        interface_plane_generation=(
            str(dev.get("interface_plane_generation", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        het_recomb_despike=float(dev.get("het_recomb_despike", 0.0)),
        band_grading=(
            str(dev.get("band_grading", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        interface_tunneling=(
            str(dev.get("interface_tunneling", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        tunnel_mass_eff=_f(dev.get("tunnel_mass_eff", 0.2)),
        S_n_left=S_n_left,
        S_p_left=S_p_left,
        S_n_right=S_n_right,
        S_p_right=S_p_right,
        microstructure=microstructure,
    )

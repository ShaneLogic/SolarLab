from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

from perovskite_sim.models.config_loader import load_device_from_yaml, _parse_bool
from perovskite_sim.models.device import DeviceStack


SUPPORTED_JUNCTION_MODELS = frozenset({"ideal_ohmic"})
SUPPORTED_LIGHT_DIRECTIONS = frozenset({"top_first"})


@dataclass(frozen=True)
class JunctionLayer:
    name: str
    thickness: float
    optical_material: str
    incoherent: bool = False


@dataclass(frozen=True)
class TandemConfig:
    top_cell: DeviceStack
    bottom_cell: DeviceStack
    junction_stack: tuple[JunctionLayer, ...]
    junction_model: str
    light_direction: str
    benchmark: dict | None


def _resolve(base: Path, ref: str) -> str:
    p = Path(ref)
    if not p.is_absolute():
        p = (base.parent / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"tandem sub-cell YAML not found: {p}")
    return str(p)


def load_tandem_from_yaml(path: str) -> TandemConfig:
    cfg_path = Path(path).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: YAML is empty or not a mapping")

    if cfg.get("device_type") != "tandem_2T_monolithic":
        raise ValueError(
            f"{path}: expected device_type=tandem_2T_monolithic, got "
            f"{cfg.get('device_type')!r}"
        )

    try:
        tandem = cfg["tandem"]
    except KeyError as e:
        raise ValueError(f"{path}: missing required field tandem") from e

    try:
        top_ref = tandem["top_cell"]
    except KeyError as e:
        raise ValueError(f"{path}: missing required field tandem.top_cell") from e

    try:
        bot_ref = tandem["bottom_cell"]
    except KeyError as e:
        raise ValueError(f"{path}: missing required field tandem.bottom_cell") from e

    top_path = _resolve(cfg_path, top_ref)
    bot_path = _resolve(cfg_path, bot_ref)
    top_cell = load_device_from_yaml(top_path)
    bottom_cell = load_device_from_yaml(bot_path)

    try:
        junction_model = tandem["junction"]["model"]
    except KeyError as e:
        raise ValueError(f"{path}: missing required field tandem.junction.model") from e

    if junction_model not in SUPPORTED_JUNCTION_MODELS:
        raise ValueError(
            f"junction.model={junction_model!r} not supported in v1. "
            f"Supported: {sorted(SUPPORTED_JUNCTION_MODELS)}"
        )

    light_direction = tandem.get("light_direction", "top_first")
    if light_direction not in SUPPORTED_LIGHT_DIRECTIONS:
        raise ValueError(
            f"light_direction={light_direction!r} not supported in v1. "
            f"Supported: {sorted(SUPPORTED_LIGHT_DIRECTIONS)}"
        )

    junction_raw = cfg.get("junction_stack", []) or []
    layers: list[JunctionLayer] = []
    for idx, j in enumerate(junction_raw):
        for required_key in ("name", "thickness_nm", "optical_material"):
            if required_key not in j:
                raise ValueError(
                    f"{path}: junction_stack[{idx}] is missing required field "
                    f"'{required_key}'"
                )
        thickness_nm = float(j["thickness_nm"])
        if thickness_nm <= 0:
            raise ValueError(
                f"{path}: junction_stack[{idx}] has invalid thickness_nm="
                f"{thickness_nm!r}; must be > 0"
            )
        layers.append(
            JunctionLayer(
                name=j["name"],
                thickness=thickness_nm * 1e-9,
                optical_material=j["optical_material"],
                incoherent=_parse_bool(j.get("incoherent", False)),
            )
        )
    junction_stack = tuple(layers)

    return TandemConfig(
        top_cell=top_cell,
        bottom_cell=bottom_cell,
        junction_stack=junction_stack,
        junction_model=junction_model,
        light_direction=light_direction,
        benchmark=cfg.get("benchmark"),
    )

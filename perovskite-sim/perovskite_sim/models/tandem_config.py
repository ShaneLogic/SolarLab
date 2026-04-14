from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

from perovskite_sim.models.config_loader import load_device_from_yaml
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

    if cfg.get("device_type") != "tandem_2T_monolithic":
        raise ValueError(
            f"{path}: expected device_type=tandem_2T_monolithic, got "
            f"{cfg.get('device_type')!r}"
        )

    tandem = cfg["tandem"]
    top_path = _resolve(cfg_path, tandem["top_cell"])
    bot_path = _resolve(cfg_path, tandem["bottom_cell"])
    top_cell = load_device_from_yaml(top_path)
    bottom_cell = load_device_from_yaml(bot_path)

    junction_model = tandem["junction"]["model"]
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
    junction_stack = tuple(
        JunctionLayer(
            name=j["name"],
            thickness=float(j["thickness_nm"]) * 1e-9,
            optical_material=j["optical_material"],
            incoherent=bool(j.get("incoherent", False)),
        )
        for j in junction_raw
    )

    return TandemConfig(
        top_cell=top_cell,
        bottom_cell=bottom_cell,
        junction_stack=junction_stack,
        junction_model=junction_model,
        light_direction=light_direction,
        benchmark=cfg.get("benchmark"),
    )

"""Layer stacks, substrate prefixes and electrical-interface alignment.

Optical-only ``substrate`` layers may only form a contiguous prefix. The
electrical grid starts at the first electrical layer, so interface data that is
aligned to the full layer list must be shifted by the substrate prefix before
it is indexed by electrical interface number. Positions are accumulated in
layer order with binary64 additions, the order every historical grid used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, TypeVar

__all__ = ["ElectricalStack", "StackLayer", "SUBSTRATE_ROLE"]

SUBSTRATE_ROLE = "substrate"
T = TypeVar("T")


@dataclass(frozen=True)
class StackLayer:
    """One layer of the full optical/electrical stack."""

    name: str
    thickness_m: float
    role: str

    def __post_init__(self) -> None:
        thickness = float(self.thickness_m)
        if not self.name or not self.role:
            raise ValueError("layer name and role must be non-empty")
        if not thickness > 0.0 or thickness == float("inf"):
            raise ValueError(f"layer {self.name!r} thickness must be finite and positive")
        object.__setattr__(self, "thickness_m", thickness)


@dataclass(frozen=True)
class ElectricalStack:
    """The electrical view of a full stack, with explicit index alignment."""

    layers: tuple[StackLayer, ...]
    substrate_prefix: int = field(init=False)
    electrical: tuple[StackLayer, ...] = field(init=False)
    optical_offset_m: float = field(init=False)
    interface_positions_m: tuple[float, ...] = field(init=False)
    thickness_m: float = field(init=False)

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not all(isinstance(layer, StackLayer) for layer in layers):
            raise TypeError("layers must be StackLayer values")
        seen_electrical = False
        for layer in layers:
            if layer.role == SUBSTRATE_ROLE:
                if seen_electrical:
                    raise ValueError(
                        "substrate layers must form a contiguous prefix of "
                        "stack.layers (mid-stack or trailing substrate layers "
                        "are not supported)"
                    )
            else:
                seen_electrical = True
        prefix = 0
        for layer in layers:
            if layer.role != SUBSTRATE_ROLE:
                break
            prefix += 1
        electrical = layers[prefix:]
        if not electrical:
            raise ValueError("stack has no electrical layers")
        offset = 0.0
        for layer in layers[:prefix]:
            offset += layer.thickness_m
        positions: list[float] = []
        position = 0.0
        for layer in electrical[:-1]:
            position += layer.thickness_m
            positions.append(position)
        total = 0.0
        for layer in electrical:
            total += layer.thickness_m
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "substrate_prefix", prefix)
        object.__setattr__(self, "electrical", electrical)
        object.__setattr__(self, "optical_offset_m", offset)
        object.__setattr__(self, "interface_positions_m", tuple(positions))
        object.__setattr__(self, "thickness_m", total)

    @classmethod
    def from_layers(cls, layers: Sequence[StackLayer]) -> ElectricalStack:
        return cls(tuple(layers))

    @property
    def interface_count(self) -> int:
        return len(self.electrical) - 1

    def full_interface_index(self, electrical_index: int) -> int:
        """Index into data aligned with the full layer list (``len(layers)-1``)."""
        if type(electrical_index) is not int or not 0 <= electrical_index < self.interface_count:
            raise IndexError("electrical interface index outside the stack")
        return electrical_index + self.substrate_prefix

    def electrical_interface_data(self, full_aligned: Sequence[T], *,
                                  pad_missing: bool = False) -> tuple[T | None, ...]:
        """Slice full-layer-aligned per-interface data to electrical interfaces.

        ``pad_missing=False`` keeps the historical interface-pair behaviour (a
        short tuple stays short); ``pad_missing=True`` keeps the historical
        interface-defect behaviour (missing slots become ``None``).
        """
        start = self.substrate_prefix
        desired = max(0, self.interface_count)
        selected: list[T | None] = list(tuple(full_aligned)[start:start + desired])
        if pad_missing:
            selected.extend([None] * (desired - len(selected)))
        return tuple(selected)

    def optical_coordinates(self, electrical_x_m: float) -> float:
        """Map an electrical coordinate into the full optical stack."""
        return float(electrical_x_m) + self.optical_offset_m

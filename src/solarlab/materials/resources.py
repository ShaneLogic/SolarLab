"""Content-bound optical tables and explicitly supplied fixed generation data.

No path discovery, spectral model or profile_file loader is provided. Tables
own immutable byte storage; NumPy access returns a fresh read-only view, so a
caller's shape/dtype edits cannot mutate the stored table or another consumer.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from solarlab.materials.source import SourceDocument
from solarlab.units import normalize_quantity

__all__ = ["ResourceTable", "ResourceLibrary"]


@dataclass(frozen=True, slots=True)
class ResourceTable:
    name: str
    kind: str
    source: SourceDocument
    shape: tuple[int, int] = field(init=False)
    columns: tuple[str, ...] = field(init=False)
    units: tuple[str, ...] = field(init=False)
    _values: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.source, SourceDocument):
            raise ValueError("resource requires a name and supplied source bytes")
        text = self.source.content.decode("utf-8")
        columns: tuple[str, ...]
        units: tuple[str, ...]
        if self.kind in {"nk", "spectrum"}:
            rows = list(csv.reader(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")))
            if self.kind == "nk":
                if not rows or rows.pop(0) != ["wavelength_nm", "n", "k"]:
                    raise ValueError("nk source requires wavelength_nm,n,k columns")
                columns, units = ("wavelength", "n", "k"), ("m", "1", "1")
            else:
                if rows and rows[0] == ["wavelength_nm", "spectral_flux_m-2_s-1_m-1"]:
                    rows.pop(0)
                columns, units = ("wavelength", "spectral_photon_flux"), ("m", "m^-3/s")
            if any(len(row) != len(columns) for row in rows):
                raise ValueError("resource row has an unexpected column count")
            numbers = [tuple([normalize_quantity(row[0] + " nm", "m"),
                              *(normalize_quantity(value, unit) for value, unit in zip(row[1:], units[1:]))]) for row in rows]
        elif self.kind == "fixed_generation":
            def exact_mapping(pairs: list[tuple[str, object]]) -> dict[str, object]:
                if len({name for name, _ in pairs}) != len(pairs):
                    raise ValueError("duplicate fixed-generation source key")
                return dict(pairs)
            data = json.loads(text, object_pairs_hook=exact_mapping)
            if not isinstance(data, dict) or set(data) != {"coordinate_m", "generation_m3_s"}:
                raise ValueError("fixed generation requires coordinate_m and generation_m3_s")
            if any(not isinstance(data[key], list) for key in data) or len(data["coordinate_m"]) != len(data["generation_m3_s"]):
                raise ValueError("fixed-generation arrays must have matching lengths")
            columns, units = ("coordinate", "generation"), ("m", "m^-3/s")
            numbers = [(normalize_quantity(x, "m"), normalize_quantity(g, "m^-3/s"))
                       for x, g in zip(data["coordinate_m"], data["generation_m3_s"])]
        else:
            raise ValueError(f"unsupported resource kind: {self.kind}")
        array: np.ndarray = np.asarray(numbers, dtype="<f8")
        if len(numbers) < 2 or not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError("resource requires at least two finite nonnegative rows")
        if np.any(np.diff(array[:, 0]) <= 0):
            raise ValueError("resource coordinates must be strictly increasing")
        if self.kind in {"nk", "spectrum"} and np.any(array[:, 0] <= 0):
            raise ValueError("wavelength must be positive")
        if self.kind == "nk" and np.any(array[:, 1] <= 0):
            raise ValueError("refractive index n must be positive")
        object.__setattr__(self, "shape", (len(numbers), len(columns)))
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "_values", array.tobytes(order="C"))

    @property
    def array(self) -> np.ndarray:
        return np.frombuffer(self._values, dtype="<f8").reshape(self.shape)

    @property
    def content_sha256(self) -> str:
        meta = json.dumps((self.name, self.kind, self.source.id, self.source.sha256,
                           self.shape, self.columns, self.units), separators=(",", ":")).encode()
        return hashlib.sha256(meta + self._values).hexdigest()


@dataclass(frozen=True, slots=True)
class ResourceLibrary:
    tables: tuple[ResourceTable, ...] = ()

    def __post_init__(self) -> None:
        tables = tuple(self.tables)
        if any(not isinstance(table, ResourceTable) for table in tables):
            raise ValueError("resource library requires validated tables")
        if len({table.name for table in tables}) != len(tables):
            raise ValueError("duplicate resource name")
        object.__setattr__(self, "tables", tuple(sorted(tables, key=lambda table: table.name)))

    def get(self, name: str, kind: str) -> ResourceTable:
        for table in self.tables:
            if table.name == name:
                if table.kind != kind:
                    raise ValueError(f"resource {name!r}: expected {kind}, found {table.kind}")
                return table
        raise ValueError(f"missing supplied {kind} resource: {name!r}")

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(json.dumps([(table.name, table.content_sha256) for table in self.tables], separators=(",", ":")).encode()).hexdigest()

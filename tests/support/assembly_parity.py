"""Representation-neutral, bitwise-by-default assembly comparison.

F and J are the steady residual and its coordinate Jacobian. Physical storage
is S, S_u=dS/du, and M=F_udot=-S_u. Step/AC operators and boundary masks are
additional named quantities; none is silently substituted for steady F/J.
Dense and sparse inputs compare by their represented entries, not sparse
index ordering. Baseline and candidate captures must come from independently
selected implementations; repeated current-source evaluation proves only
determinism, never migration parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, issparse


@dataclass(frozen=True)
class Block:
    """A named half-open coordinate range; row blocks use the same layout."""

    name: str
    start: int
    stop: int

    @property
    def indices(self) -> slice:
        return slice(self.start, self.stop)


def _dense(value: Any) -> np.ndarray:
    if issparse(value):
        entries = value.tocoo(copy=True)
        entries.sum_duplicates()
        result = np.zeros(entries.shape, dtype=entries.dtype)
        # Assignment preserves explicitly stored signed zeros. Sparse toarray
        # may instead add each entry to +0, concealing a bitwise difference.
        result[entries.row, entries.col] = entries.data
    else:
        result = np.asarray(value)
    if result.dtype.kind not in "biufc" or not np.all(np.isfinite(result)):
        raise ValueError("assembly quantities must be finite numeric arrays")
    return result


def _vector(value: Any, size: int) -> np.ndarray:
    array = _dense(value)
    if array.shape == (size, 1):
        array = array[:, 0]
    if array.shape != (size,):
        raise ValueError("assembly vector shape differs from the block layout")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _matrix(value: Any, size: int) -> Any:
    if getattr(value, "shape", np.shape(value)) != (size, size):
        raise ValueError("assembly matrix shape differs from the block layout")
    if not issparse(value):
        dense = np.array(_dense(value), copy=True)
        dense.setflags(write=False)
        return dense
    result = csc_matrix(value, copy=True)
    # Coalesce only the represented matrix entries, without numerical rounding
    # policy changes, dropping small values, or eliminating signed zeros.
    result.sum_duplicates()
    result.sort_indices()
    if not np.all(np.isfinite(result.data)):
        raise ValueError("assembly quantities must be finite")
    for array in (result.data, result.indices, result.indptr):
        array.setflags(write=False)
    return result


@dataclass(frozen=True)
class Extraction:
    """A capture usable without importing any eventual production Assembler.

    ``blocks`` accepts either Block records or ``{name: (start, stop)}``.
    S_u is exposed as both ``dS_du`` and the ``S_u`` property. Quantities absent
    from the original interface stay None. ``extra`` carries the original
    step residual/Jacobian or AC system quantities. Boundary masks are global
    row vectors and are compared just as strictly as numerical coefficients.
    """

    F: Any
    J: Any
    blocks: tuple[Block, ...] | Mapping[str, tuple[int, int]]
    S: Any = None
    dS_du: Any = None
    M: Any = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    boundary_masks: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        blocks = (tuple(Block(name, *bounds) for name, bounds in self.blocks.items())
                  if isinstance(self.blocks, Mapping) else tuple(self.blocks))
        end, names = 0, set()
        for block in blocks:
            if (not block.name or block.name in names or type(block.start) is not int
                    or type(block.stop) is not int or block.start != end or block.stop <= block.start):
                raise ValueError("blocks must be unique, contiguous named half-open ranges")
            names.add(block.name)
            end = block.stop
        if not blocks:
            raise ValueError("at least one variable block is required")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "F", _vector(self.F, end))
        object.__setattr__(self, "J", _matrix(self.J, end))
        if self.S is not None:
            object.__setattr__(self, "S", _vector(self.S, end))
        for name in ("dS_du", "M"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _matrix(value, end))
        extra = {}
        for name, value in self.extra.items():
            if name in {"F", "J", "S", "dS_du", "M"} or name.startswith("mask:"):
                raise ValueError("extra quantity name shadows a canonical quantity")
            shape = getattr(value, "shape", np.shape(value))
            extra[name] = _matrix(value, end) if shape == (end, end) else _vector(value, end)
        masks = {}
        for name, value in self.boundary_masks.items():
            mask = _vector(value, end)
            if mask.dtype != np.dtype(bool):
                raise ValueError("boundary-row masks must be explicit boolean vectors")
            masks[name] = mask
        object.__setattr__(self, "extra", MappingProxyType(extra))
        object.__setattr__(self, "boundary_masks", MappingProxyType(masks))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def S_u(self) -> Any:
        return self.dS_du

    @property
    def size(self) -> int:
        return int(self.F.size)

    def quantities(self) -> dict[str, Any]:
        values = {name: getattr(self, name) for name in ("F", "J", "S", "dS_du", "M")
                  if getattr(self, name) is not None}
        values.update(self.extra)
        values.update({"mask:" + name: mask for name, mask in self.boundary_masks.items()})
        return values


@dataclass(frozen=True)
class Tolerance:
    """An explicit exception for one quantity; zero defaults retain every bit."""

    atol: float = 0.0
    rtol: float = 0.0
    justification: str = ""

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in (self.atol, self.rtol)):
            raise ValueError("parity tolerances must be finite and nonnegative")
        if (self.atol or self.rtol) and not self.justification.strip():
            raise ValueError("every nonzero parity tolerance requires a stated justification")

    @property
    def bitwise(self) -> bool:
        return self.atol == self.rtol == 0.0


@dataclass(frozen=True)
class BlockDifference:
    quantity: str
    row_block: str
    column_block: str | None
    max_absolute: float
    max_relative: float
    changed_entries: int
    bitwise_equal: bool
    passed: bool
    tolerance: Tolerance


@dataclass(frozen=True)
class Comparison:
    differences: tuple[BlockDifference, ...]
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues and all(item.passed for item in self.differences)

    def assert_equal(self) -> None:
        failures = [f"{item.quantity}:{item.row_block}/{item.column_block} "
                    f"abs={item.max_absolute:.17g}, rel={item.max_relative:.17g}, "
                    f"changed={item.changed_entries}" for item in self.differences if not item.passed]
        if not self.passed:
            raise AssertionError("assembly parity failed: " + "; ".join((*self.issues, *failures)))


def _difference(reference: np.ndarray, candidate: np.ndarray, quantity: str,
                row: str, column: str | None, tolerance: Tolerance) -> BlockDifference:
    a, b = np.ascontiguousarray(reference), np.ascontiguousarray(candidate)
    same_dtype = a.dtype == b.dtype
    if same_dtype:
        words_a = a.reshape(-1).view(np.uint8).reshape(a.size, a.dtype.itemsize)
        words_b = b.reshape(-1).view(np.uint8).reshape(b.size, b.dtype.itemsize)
        changed = int(np.count_nonzero(np.any(words_a != words_b, axis=1)))
    else:
        changed = a.size
    dtype = np.result_type(a.dtype, b.dtype, np.float64)
    delta = abs(b.astype(dtype) - a.astype(dtype))
    denominator = np.maximum(abs(a.astype(dtype)), abs(b.astype(dtype)))
    relative = np.divide(delta, denominator, out=np.zeros_like(delta), where=denominator != 0)
    exact = same_dtype and changed == 0
    passed = exact if tolerance.bitwise else same_dtype and bool(np.all(
        delta <= tolerance.atol + tolerance.rtol * abs(a.astype(dtype))))
    return BlockDifference(quantity, row, column, float(np.max(delta, initial=0.0)),
                           float(np.max(relative, initial=0.0)), changed, exact, passed, tolerance)


def compare(reference: Extraction, candidate: Extraction,
            *, tolerances: Mapping[str, Tolerance] | None = None) -> Comparison:
    """Report every vector block/matrix block pair without a hidden epsilon.

    Relative differences use |candidate-reference|/max(|candidate|,|reference|),
    with 0/0 reported as zero. Explicit tolerances use the usual reference-
    scaled bound atol+rtol*|reference|. Bitwise mode also detects signed zeros,
    dtype changes and a one-ulp difference. Layout/quantity mismatches always
    fail, regardless of numeric tolerances. Source provenance may differ.
    """
    left, right = reference.quantities(), candidate.quantities()
    chosen = {} if tolerances is None else dict(tolerances)
    if set(chosen) - set(left):
        raise ValueError("tolerance names must identify an extracted reference quantity")
    if any(not isinstance(value, Tolerance) for value in chosen.values()):
        raise TypeError("each per-quantity tolerance must be an explicit Tolerance")
    issues = []
    if reference.blocks != candidate.blocks:
        issues.append("named block layout differs")
    if left.keys() != right.keys():
        issues.append("quantity inventory differs")
    if reference.size != candidate.size:
        return Comparison((), tuple((*issues, "global coordinate count differs")))
    differences = []
    for name in sorted(left.keys() & right.keys()):
        a, b = _dense(left[name]), _dense(right[name])
        if a.shape != b.shape:
            issues.append(name + " shape differs")
            continue
        tolerance = chosen.get(name, Tolerance())
        for row in reference.blocks:
            if not isinstance(row, Block):  # Narrow the public mapping/tuple input type.
                raise TypeError("normalized block records required")
            if a.ndim == 1:
                differences.append(_difference(a[row.indices], b[row.indices], name, row.name, None, tolerance))
            else:
                for column in reference.blocks:
                    if not isinstance(column, Block):
                        raise TypeError("normalized block records required")
                    region = (row.indices, column.indices)
                    differences.append(_difference(a[region], b[region], name, row.name, column.name, tolerance))
    return Comparison(tuple(differences), tuple(issues))

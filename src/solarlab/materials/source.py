"""Explicit immutable source bytes, independent of files and installation paths."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

__all__ = ["SourceDocument"]


@dataclass(frozen=True, slots=True)
class SourceDocument:
    id: str
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("source requires an explicit ID")
        if type(self.content) is not bytes or not self.content or len(self.content) > 2**22:
            raise ValueError("source content requires nonempty bounded bytes")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()



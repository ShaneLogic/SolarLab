"""Machine-readable contract failure."""

from __future__ import annotations

__all__ = ["ContractError"]


class ContractError(ValueError):
    """An input or assembly contract failed, with a machine-readable reason."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)

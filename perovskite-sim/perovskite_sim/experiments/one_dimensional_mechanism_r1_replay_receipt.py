"""In-process evidence of completed R1 equation replay.

This leaf module has no numerical imports. A JSON ledger is transferable
evidence to bind to a trusted outer artifact, but cannot impersonate a live
receipt from this process. Receipts never claim independence of shared laws.
"""
from __future__ import annotations

from dataclasses import dataclass
import json


_REPLAY_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class R1PhysicsReplayReceipt:
    canonical_json: str

    def __init__(self, ledger, *, _issuer=None):
        if _issuer is not _REPLAY_ISSUER:
            raise TypeError("a replay receipt must be issued by actual equation reconstruction")
        object.__setattr__(self, "canonical_json", json.dumps(
            ledger, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def to_dict(self):
        return json.loads(self.canonical_json)


class R1PhysicsReplayReport(dict):
    def __init__(self, report, receipt):
        super().__init__(report)
        self.replay_receipt = receipt


def _issue_replay_receipt(ledger):
    return R1PhysicsReplayReceipt(ledger, _issuer=_REPLAY_ISSUER)

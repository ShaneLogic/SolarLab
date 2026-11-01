"""Injected run persistence, independent of storage and execution implementations.

These signatures describe the existing RunStore lifecycle, event and artifact
operations. Documents remain opaque, storage-validated JSON objects; the port
does not compute scientific identities or change their meaning. Attempt tokens
and validated artifact payloads are supplied by the adapter's type parameters,
so this layer need not import either implementation.

Store construction, closing, local paths, recovery and archive administration
remain the owner's responsibility and are deliberately outside this port.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

__all__ = ["PersistenceReader", "PersistencePort"]

_Token = TypeVar("_Token")
_Payload = TypeVar("_Payload", contravariant=True)


@runtime_checkable
class PersistenceReader(Protocol):
    """Read existing run/attempt records and bounded, ordered event pages.

    Returned records retain the producer's identities and terminal result.
    Reading never recovers an attempt or changes its terminal state.
    """

    store_id: str

    def get_run(self, run_id: str) -> dict[str, Any]: ...

    def attempts(self, run_id: str) -> list[dict[str, Any]]: ...

    def events(
        self, *, after: int = 0, limit: int = 100, run_id: str | None = None,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class PersistencePort(PersistenceReader, Protocol[_Token, _Payload]):
    """The existing fenced run writer and verified artifact reader.

    Enqueue is idempotent for the exact stored request. Claim, event, publication
    and terminal operations preserve attempt/owner/generation authority; a retry
    creates a new attempt. Completion verifies its artifact references before
    committing the unique terminal event. Implementations keep their existing
    conflict and integrity errors, including refusal to mutate a read-only store.
    """

    def enqueue(
        self, run_id: str, *, opaque_identity: dict[str, Any], input_metadata: dict[str, Any],
    ) -> dict[str, Any]: ...

    def claim(
        self, run_id: str, *, expected_attempt_id: str, expected_generation: int, owner_id: str,
    ) -> _Token: ...

    def append_event(self, token: _Token, *, event_id: str, payload: dict[str, Any]) -> int: ...

    def publish(self, token: _Token, name: str, payload: _Payload) -> dict[str, Any]: ...

    def complete(
        self, token: _Token, *, state: str, result: dict[str, Any], artifacts: Sequence[str] = (),
    ) -> int: ...

    def fail_queued(
        self, run_id: str, *, expected_attempt_id: str, expected_generation: int,
        result: dict[str, Any],
    ) -> int: ...

    def cancel(
        self, run_id: str, *, expected_attempt_id: str, expected_generation: int, reason: str,
    ) -> int: ...

    def retry(
        self, run_id: str, *, expected_attempt_id: str, expected_generation: int,
    ) -> dict[str, Any]: ...

    def artifact(self, artifact_id: str) -> dict[str, Any]: ...

    def read_artifact(self, artifact_id: str) -> bytes: ...

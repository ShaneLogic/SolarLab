"""Explicit portable source locators and immutable byte/symbol verification.

No checkout, Git repository or installed resource is discovered implicitly.
The caller supplies a source bundle and its trusted commit/content association;
as with BuildSource, this is not a release-coverage certificate.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from solarlab.config.identity import BuildSource
from solarlab.materials.source import SourceDocument

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Commit = Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or path.as_posix() != value or any(part in {".", ".."} for part in value.split("/")) or "\\" in value or ":" in value:
        raise ValueError("source path must be an explicit normalized bundle-relative POSIX path")
    return value


class SourceRef(Record):
    id: Annotated[str, Field(min_length=1)]
    path: str
    sha256: Digest
    source_commit: Commit

    _path = field_validator("path")(relative_path)


class SymbolRef(Record):
    source_id: Annotated[str, Field(min_length=1)]
    symbol: Annotated[str, Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")]


@dataclass(frozen=True, slots=True)
class SourceSet:
    references: tuple[SourceRef, ...]
    documents: tuple[SourceDocument, ...]

    def __post_init__(self) -> None:
        refs, documents = tuple(self.references), tuple(self.documents)
        if not refs or any(type(ref) is not SourceRef for ref in refs):
            raise ValueError("source references are required")
        if any(type(source) is not SourceDocument for source in documents):
            raise ValueError("actual immutable SourceDocument bytes are required")
        if len({ref.id for ref in refs}) != len(refs) or len({(ref.path, ref.source_commit) for ref in refs}) != len(refs):
            raise ValueError("duplicate or ambiguous source ID/path/commit")
        if len({doc.id for doc in documents}) != len(documents) or {doc.id for doc in documents} != {ref.id for ref in refs}:
            raise ValueError("source bundle has missing, unexpected or ambiguous documents")
        actual = {doc.id: doc for doc in documents}
        for ref in refs:
            if actual[ref.id].sha256 != ref.sha256:
                raise ValueError(f"{ref.path}: source SHA-256 mismatch")
        for commit in {ref.source_commit for ref in refs}:
            BuildSource(commit, tuple(actual[ref.id] for ref in refs if ref.source_commit == commit))
        object.__setattr__(self, "references", refs)
        object.__setattr__(self, "documents", documents)

    def document(self, id: str) -> SourceDocument:
        for document in self.documents:
            if document.id == id:
                return document
        raise ValueError(f"missing source ID: {id}")

    def reference(self, id: str) -> SourceRef:
        for reference in self.references:
            if reference.id == id:
                return reference
        raise ValueError(f"missing source ID: {id}")

    def at_path(self, path: str) -> SourceDocument:
        relative_path(path)
        matches = [ref for ref in self.references if ref.path == path]
        if len(matches) != 1:
            raise ValueError(f"source path is missing or ambiguous: {path}")
        return self.document(matches[0].id)

    def verify_symbol(self, ref: SymbolRef) -> None:
        source = self.document(ref.source_id)
        if not self.reference(ref.source_id).path.endswith(".py"):
            raise ValueError("symbol must be bound to a Python source document")
        try:
            body = ast.parse(source.content).body
        except (SyntaxError, UnicodeError) as error:
            raise ValueError("invalid bound Python source") from error
        for index, name in enumerate(ref.symbol.split(".")):
            matches = [node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name]
            if len(matches) != 1:
                raise ValueError(f"missing or ambiguous symbol {ref.symbol} in {source.id}")
            selected = matches[0]
            if index < len(ref.symbol.split(".")) - 1 and not isinstance(selected, ast.ClassDef):
                raise ValueError("only module or class member bindings are supported")
            body = selected.body


def read_sources(root: str | Path, references: tuple[SourceRef, ...]) -> SourceSet:
    """Load an explicit directory bundle, including after relocation/install.

    V2 YAMLs and historical source bundles are caller-owned data; they are not
    implicitly bundled package resources or resolved relative to the CWD.
    """
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("an absolute explicit source bundle root is required")
    root = root.resolve(strict=True)
    documents = []
    for ref in references:
        path = (root / relative_path(ref.path)).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"source escapes or is absent from the supplied bundle: {ref.path}")
        documents.append(SourceDocument(ref.id, path.read_bytes()))
    return SourceSet(references, tuple(documents))

"""Solver-free retained-consumer checks derived from MilestoneA/RegenerateV1.

Keep its source scope: non-test Python/notebook files under src, examples,
scripts and perovskite-sim, excluding the legacy library and W03 internals.
Inspect named and literal dynamic imports, plus literal browser wire transports.
Computed import/URL targets are not an execution or whole-program proof.
Unlike the archival renderer, this guard needs no archive files or Node install.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


SCRIPT_SCOPE = (
    "All active script and registry executor entries in the P00 graph, "
    "not just these representative paths"
)
RESEARCH_SCOPE = (
    "All precision, backend, replay, certificate and archive consumers in the P00 graph"
)
_JS_COMMENTS = re.compile(
    r"""('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|"""
    + chr(96) + r"(?:\\.|[^" + chr(96) + r"\\])*" + chr(96)
    + r")|//[^\n]*|/\*[\s\S]*?\*/"
)
_TRANSPORT = re.compile(r"\b(?:fetch|EventSource)\s*(?:<[^;{}()]*>)?\s*\(")
_WIRE = re.compile(r"/api/[A-Za-z0-9_./-]+")


@dataclass(frozen=True)
class Consumer:
    path: str
    line: int
    target: str
    cell: int | None = None


def python_scope(path: str) -> bool:
    """Same path exclusions as the acceptance audit; .py notebooks are scripts."""
    return (
        path.endswith((".py", ".ipynb"))
        and path.startswith(("src/", "examples/", "scripts/", "perovskite-sim/"))
        and not path.startswith((
            "perovskite-sim/tests/", "perovskite-sim/perovskite_sim/",
            "src/solarlab_server/legacy_engine/",
        ))
    )


def browser_scope(path: str) -> bool:
    return (
        path.startswith("web/src/") and path.endswith((".ts", ".tsx"))
        and ".test." not in path and "/__tests__/" not in path
    )


def import_consumers(path: str, source: str, cell: int | None = None) -> list[Consumer]:
    """Parse source, never import it; include literal importlib aliases."""
    if not any(word in source for word in ("perovskite_sim", "backend")):
        return []
    tree = ast.parse(source, filename=path)
    loaders = {"import_module", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            loaders.update(a.asname or a.name for a in node.names if a.name == "import_module")
    found: list[Consumer] = []
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets = [node.module + "." + name.name for name in node.names]
        elif isinstance(node, ast.Import):
            targets = [name.name for name in node.names]
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            loader = (isinstance(function, ast.Name) and function.id in loaders
                      or isinstance(function, ast.Attribute)
                      and function.attr in {"import_module", "__import__"})
            argument = node.args[0]
            if loader and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                targets = [argument.value]
        for target in targets:
            if target.split(".")[0] not in {"perovskite_sim", "backend"}:
                continue
            # backend/jobs.py -> backend/progress.py is internal app support,
            # not a second direct legacy-library consumer (AcceptanceV2).
            if path.startswith("perovskite-sim/backend/") and target.split(".")[0] == "backend":
                continue
            found.append(Consumer(path, getattr(node, "lineno"), target, cell))
    return found


def scan_consumers(root: Path, paths: Iterable[str]) -> tuple[Consumer, ...]:
    found: list[Consumer] = []
    for path in sorted(set(paths)):
        if not python_scope(path) and not browser_scope(path):
            continue
        file = root / path
        if not file.is_file():
            continue  # Explicit registered deletions are checked independently.
        source = file.read_text()
        if browser_scope(path):
            # Preserve strings/templates while removing comments. Transport
            # calls and URL literals may occur on separate lines in the file.
            code = _JS_COMMENTS.sub(
                lambda match: match.group(1) or "\n" * match.group().count("\n"), source
            )
            if _TRANSPORT.search(code):
                found.extend(Consumer(path, code.count("\n", 0, m.start()) + 1, m.group())
                             for m in _WIRE.finditer(code))
        elif path.endswith(".ipynb"):
            for index, cell in enumerate(json.loads(source).get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                code = cell.get("source", [])
                code = "".join(code) if isinstance(code, list) else code
                # The audit excludes IPython shell/magic directives. Preserve
                # line numbering and parse all remaining Python, failing closed
                # on malformed code instead of silently dropping a whole cell.
                code = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                                 for line in code.splitlines())
                found.extend(import_consumers(path, code, index))
        else:
            found.extend(import_consumers(path, source))
    return tuple(sorted(set(found), key=lambda c: (c.path, c.cell or -1, c.line, c.target)))


def historical(row: dict[str, Any]) -> bool:
    return row.get("historical") is True or row.get("status") == "historical"


def registration_violations(
    root: Path,
    registry: dict[str, Any],
    p00: dict[str, Any],
    consumers: Iterable[Consumer],
    lanes: dict[str, Any] | None = None,
) -> list[str]:
    """Require live path/owner/exit records or a proved frozen-P00 umbrella."""
    errors = []
    active = [g for g in registry["consumer_groups"] if not historical(g)]
    for row in (registry["consumer_groups"] + registry.get("data_adapters", [])
                + [registry["execution_facade"]]):
        if historical(row):
            continue
        identity = row.get("id", row.get("module", "unnamed registration"))
        if not row.get("owner") or not row.get("exit_tasks"):
            errors.append(f"{identity}: missing owner or exit tasks")
        paths = row.get("paths", [row["path"]] if "path" in row else [])
        for path in paths:
            if not (root / path).is_file():
                errors.append(f"{identity}: registered path missing: {path}")
    explicit = {path for group in active if group.get("owner") and group.get("exit_tasks")
                for path in group.get("paths", [])}
    scripts = any(g.get("id") == "legacy-cli-and-registries"
                  and g.get("inventory_scope") == SCRIPT_SCOPE
                  and g.get("owner") and g.get("exit_tasks") for g in active)
    research = any(g.get("id") == "retained-r1-and-reverse-research-consumers"
                   and g.get("inventory_scope") == RESEARCH_SCOPE
                   and g.get("owner") and g.get("exit_tasks") for g in active)
    old_scripts = {s["current_path"] for s in p00["sources"]
                   if s.get("path_base") != "archive"}
    missing = set()
    for consumer in consumers:
        path = consumer.path
        covered_script = (
            scripts and path in old_scripts and path.endswith(".py")
            and path.startswith(("scripts/", "perovskite-sim/scripts/",
                                 "perovskite-sim/notebooks/"))
        )
        if path not in explicit and not covered_script:
            missing.add(path)
    errors.extend(f"unregistered direct legacy consumer: {path}" for path in sorted(missing))
    if lanes is not None:
        old_lanes = {
            c["origin_key"].removeprefix("lane:"): c
            for c in p00["capabilities"] if c.get("origin_key", "").startswith("lane:")
        }
        for name, lane in sorted(lanes.items()):
            executor = str(lane.get("executor", ""))
            # The accepted audit covers every configured refinement lane;
            # a module prefix cannot exempt it from its exact P00 registration.
            old = old_lanes.get(name, {})
            if not ((scripts or research)
                    and executor == old.get("registered_contract", {}).get("executor")):
                errors.append(f"unregistered direct legacy refinement executor: {name}: {executor}")
    return errors

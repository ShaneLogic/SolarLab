"""Audit actual workflow pytest directories without maintaining a file allowlist."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shlex
from typing import Any


@dataclass(frozen=True)
class Selection:
    job: str
    directories: tuple[str, ...]
    markers: str
    ignored_files: tuple[str, ...] = ()


def workflow_selections(workflow: dict[str, Any]) -> tuple[Selection, ...]:
    selected = []
    default = workflow.get("defaults", {}).get("run", {}).get("working-directory", ".")
    for job, body in workflow["jobs"].items():
        if body.get("if"):
            continue  # Optional/manual jobs cannot conceal gaps in ordinary CI.
        job_cwd = body.get("defaults", {}).get("run", {}).get("working-directory", default)
        for step in body.get("steps", []):
            if step.get("working-directory", job_cwd) != ".":
                continue  # The legacy project's separate smoke suite is out of root scope.
            for line in step.get("run", "").splitlines():
                if "-m pytest" not in line:
                    continue
                argv = shlex.split(line)
                argv = argv[argv.index("pytest") + 1:]
                if any(arg == "-k" or arg.startswith(("--ignore-glob", "--deselect")) for arg in argv):
                    raise ValueError(f"{job}: exclusions belong in explicit test prerequisites, not hidden CLI deselection")
                positional, ignored = [], []
                index = 0
                while index < len(argv):
                    argument = argv[index]
                    if argument == "--ignore":
                        index += 1
                        if index == len(argv):
                            raise ValueError(f"{job}: --ignore requires an explicit file")
                        ignored.append(argv[index])
                    elif argument.startswith("--ignore="):
                        ignored.append(argument.split("=", 1)[1])
                    else:
                        positional.append(argument)
                    index += 1
                directories = tuple(arg for arg in positional if arg == "tests" or arg.startswith("tests/"))
                if any(".py" in path or "::" in path for path in directories):
                    raise ValueError(f"{job}: root CI must discover directories, not list test files")
                if not directories:
                    continue
                markers = argv[argv.index("-m") + 1] if "-m" in argv else "not slow"
                selected.append(Selection(job, directories, markers, tuple(ignored)))
    return tuple(selected)


def marker_names(nodes: list[ast.AST]) -> set[str]:
    result = set()
    for node in nodes:
        for value in ast.walk(node):
            if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Attribute)
                    and value.value.attr == "mark"):
                result.add(value.attr)
    return result


def source_test_markers(source: str) -> tuple[set[str], ...]:
    tree = ast.parse(source)
    module = marker_names([node.value for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)])
    result = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            result.append(module | marker_names(node.decorator_list))
        if isinstance(node, ast.ClassDef):
            class_markers = module | marker_names(node.decorator_list)
            result.extend(class_markers | marker_names(child.decorator_list) for child in node.body
                          if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and child.name.startswith("test_"))
    return tuple(result)


def matches(expression: str, markers: set[str]) -> bool:
    def visit(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in markers
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not visit(node.operand)
        if isinstance(node, ast.BoolOp):
            values = [visit(value) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise ValueError("unsupported CI marker expression")
    return visit(ast.parse(expression, mode="eval").body)


def audit_files(files: dict[str, str], selections: tuple[Selection, ...],
                exclusions: dict[str, Any]) -> dict[str, Any]:
    """Audit a source mapping, also usable with synthetic in-memory files."""
    excluded = exclusions["files"]
    for name, reason in excluded.items():
        if not isinstance(reason, str) or not reason.strip() or name not in files:
            raise ValueError(f"stale or unexplained exclusion: {name}")
    for selection in selections:
        for name in selection.ignored_files:
            if name not in excluded:
                raise ValueError(f"undocumented CI exclusion: {name}")
    uncovered, assignments = [], {}
    duplicates, conflicts = {}, []
    for name, source in sorted(files.items()):
        path = PurePosixPath(name)
        markers = source_test_markers(source)
        jobs = sorted({selection.job for selection in selections
                       if name not in selection.ignored_files
                       and any(path.is_relative_to(directory) for directory in selection.directories)
                       and any(matches(selection.markers, value) for value in markers)})
        assignments[name] = jobs
        if len(jobs) > 1:
            duplicates[name] = jobs
        if jobs and name in excluded:
            conflicts.append(name)
        if not jobs and name not in excluded:
            uncovered.append(name)
    if duplicates:
        raise ValueError(f"duplicate CI job assignments: {duplicates}")
    if conflicts:
        raise ValueError(f"covered files conflict with exclusions: {conflicts}")
    return {"test_files": len(files), "uncovered_files": uncovered,
            "duplicate_files": duplicates, "conflicting_exclusions": conflicts,
            "explicit_file_exclusions": excluded, "assignments": assignments}


def audit(root: Path, selections: tuple[Selection, ...], exclusions: dict[str, Any]) -> dict[str, Any]:
    files = {path.relative_to(root).as_posix(): path.read_text()
             for path in (root / "tests").rglob("test_*.py")}
    return audit_files(files, selections, exclusions)

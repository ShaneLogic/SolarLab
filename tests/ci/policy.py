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
    isolated_group: str | None = None


def workflow_selections(workflow: dict[str, Any]) -> tuple[Selection, ...]:
    selected = []
    default = workflow.get("defaults", {}).get("run", {}).get("working-directory", ".")
    for job, body in workflow["jobs"].items():
        if body.get("if"):
            continue  # Optional/manual jobs cannot conceal gaps in ordinary CI.
        job_cwd = body.get("defaults", {}).get("run", {}).get("working-directory", default)
        for step in body.get("steps", []):
            if step.get("if"):
                continue
            if step.get("working-directory", job_cwd) != ".":
                continue  # The legacy project's separate smoke suite is out of root scope.
            group = step.get("env", {}).get("PYTEST_CI_ISOLATED_GROUP")
            if group is not None and (not isinstance(group, str) or not group.strip()):
                raise ValueError(f"{job}: an isolated process group requires a name")
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
                if any("::" in path for path in directories):
                    raise ValueError(f"{job}: root CI must not select individual test nodes")
                if group is None and any(".py" in path for path in directories):
                    raise ValueError(f"{job}: root CI must discover directories, not list test files")
                if not directories:
                    continue
                markers = argv[argv.index("-m") + 1] if "-m" in argv else "not slow"
                selected.append(Selection(job, directories, markers, tuple(ignored), group))
    return tuple(selected)


def marker_names(nodes: list[ast.AST]) -> set[str]:
    result = set()
    for node in nodes:
        for value in ast.walk(node):
            if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Attribute)
                    and value.value.attr == "mark"):
                result.add(value.attr)
    return result


def source_test_node_markers(source: str) -> dict[str, set[str]]:
    tree = ast.parse(source)
    module = marker_names([node.value for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)])
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            result[node.name] = module | marker_names(node.decorator_list)
        if isinstance(node, ast.ClassDef):
            class_markers = module | marker_names(node.decorator_list)
            result.update({f"{node.name}::{child.name}": class_markers | marker_names(child.decorator_list)
                           for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and child.name.startswith("test_")})
    return result


def source_test_markers(source: str) -> tuple[set[str], ...]:
    return tuple(source_test_node_markers(source).values())


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


def source_test_nodes(source: str) -> set[str]:
    return set(source_test_node_markers(source))


def audit_fixture_inventory(root: Path, captured: dict[str, str], portable: dict[str, str]) -> list[str]:
    """Inventory serialized data without relying on a particular capture schema.

    JSON can store interned binary64 digests, hex words or literal arrays;
    NumPy files and compressed readback bundles need the same explicit review.
    Portable input/readback/metadata declarations do not authorize comparisons
    of newly computed floating outputs to historical words.
    """
    # Treat every non-source/documentation asset as a candidate, including
    # unfamiliar suffixes. A new encoding must not bypass the inventory.
    source_suffixes = {".py", ".pyc", ".pyo", ".md"}
    files = {path.relative_to(root).as_posix() for path in (root / "tests").rglob("*")
             if path.is_file() and path.suffix not in source_suffixes
             and "__pycache__" not in path.parts and path.name != ".DS_Store"}
    files.discard("tests/ci/exclusions.json")  # This policy is not a numerical fixture.
    if captured.keys() & portable.keys():
        raise ValueError("fixture cannot be both captured and portable")
    for name, reason in {**captured, **portable}.items():
        if not isinstance(reason, str) or not reason.strip() or (not name.startswith("env:") and name not in files):
            raise ValueError(f"stale or unexplained fixture classification: {name}")
    missing = sorted(files - captured.keys() - portable.keys())
    if missing:
        raise ValueError(f"unclassified serialized fixtures: {missing}")
    return sorted(files)


def platform_fixture_consumers(sources: dict[str, str], fixtures: dict[str, str]) -> dict[str, list[str]]:
    """Conservatively follow fixture references, helpers, imports and pytest fixtures.

    Classification is separate from skipping: input-byte and metadata checks
    stay portable, while comparisons to captured floating outputs need the
    capture-environment guard. Literal paths, basenames and named external
    fixture environment variables are all inventoried by the policy.
    """
    symbols: dict[tuple[str, str], ast.AST] = {}
    imported: dict[tuple[str, str], tuple[str, str]] = {}
    modules: dict[tuple[str, str], str] = {}
    tests = set()

    def module_file(module: str) -> str | None:
        path = module.replace(".", "/") + ".py"
        for candidate in (path, "tests/" + path):
            if candidate in sources:
                return candidate
        candidates = [name for name in sources if PurePosixPath(name).name == path]
        return candidates[0] if len(candidates) == 1 else None

    for path, source in sources.items():
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols[path, node.name] = node
            elif isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols[path, node.name + "::" + method.name] = method
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and node.value is not None:
                        symbols[path, target.id] = node.value
            elif isinstance(node, ast.ImportFrom) and node.module:
                target = module_file(node.module)
                if target is not None:
                    for alias in node.names:
                        imported[path, alias.asname or alias.name] = (target, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    target = module_file(alias.name)
                    if target is not None:
                        modules[path, alias.asname or alias.name] = target
        tests.update((path, name) for name in source_test_nodes(source)
                     if PurePosixPath(path).name.startswith("test_"))

    def resolve(path: str, name: str) -> tuple[str, str] | None:
        if (path, name) in symbols:
            return path, name
        if (path, name) in imported:
            return imported[path, name]
        for parent in PurePosixPath(path).parents:
            key = (str(parent / "conftest.py"), name)
            if key in symbols:
                return key
        return None

    direct: dict[tuple[str, str], set[str]] = {}
    edges: dict[tuple[str, str], set[tuple[str, str]]] = {}
    origins: dict[tuple[str, str], set[str]] = {}
    attributes: list[tuple[tuple[str, str], str]] = []
    for key, node in symbols.items():
        path, _ = key
        direct[key], edges[key], origins[key] = set(), set(), set()
        for part in ast.walk(node):
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                for fixture in fixtures:
                    label = fixture.removeprefix("env:")
                    if part.value == label or part.value.endswith("/" + label) or part.value == PurePosixPath(label).name:
                        direct[key].add(fixture)
                target = module_file(part.value.removesuffix(".py")) if part.value.endswith(".py") else None
                if target is not None:
                    origins[key].add(target)
            if isinstance(part, (ast.Name, ast.arg)):
                dependency = resolve(path, part.id if isinstance(part, ast.Name) else part.arg)
                if dependency is not None:
                    edges[key].add(dependency)
            if isinstance(part, ast.Attribute):
                if isinstance(part.value, ast.Name) and (path, part.value.id) in modules:
                    edges[key].add((modules[path, part.value.id], part.attr))
                else:
                    attributes.append((key, part.attr))
            # runpy.run_path returns an export dictionary, not a module.
            # Follow the named export through the same actual-file origins.
            if isinstance(part, ast.Subscript) and isinstance(part.slice, ast.Constant) and isinstance(part.slice.value, str):
                attributes.append((key, part.slice.value))
            if isinstance(part, ast.Call):
                export = None
                if isinstance(part.func, ast.Attribute) and part.func.attr == "get" and part.args:
                    export = part.args[0]
                elif isinstance(part.func, ast.Name) and part.func.id == "getattr" and len(part.args) > 1:
                    export = part.args[1]
                if isinstance(export, ast.Constant) and isinstance(export.value, str):
                    attributes.append((key, export.value))
            if isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute) and part.func.attr in {
                    "usefixtures", "getfixturevalue"}:
                for argument in part.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        dependency = resolve(path, argument.value)
                        if dependency is not None:
                            edges[key].add(dependency)
    def propagate(values: dict[tuple[str, str], set[str]]) -> None:
        changed = True
        while changed:
            changed = False
            for key, dependencies in edges.items():
                before = len(values[key])
                for dependency in dependencies:
                    values[key].update(values.get(dependency, ()))
                changed |= len(values[key]) != before

    # Resolve dynamically loaded oracle attributes through their actual file,
    # not every unrelated module that happens to export a constant named FIXTURE.
    propagate(origins)
    for key, attribute in attributes:
        edges[key].update((path, attribute) for path in origins[key])
    propagate(direct)
    return {path + "::" + name: sorted(direct[path, name])
            for path, name in sorted(tests) if direct[path, name]}


def audit_platform_fixtures(sources: dict[str, str], fixtures: dict[str, str],
                            classified: list[str]) -> dict[str, list[str]]:
    consumers = platform_fixture_consumers(sources, fixtures)
    marked = {path + "::" + name for path, source in sources.items()
              if PurePosixPath(path).name.startswith("test_")
              for name, markers in source_test_node_markers(source).items() if "platform_bound" in markers}
    missing = sorted((set(consumers) | marked) - set(classified))
    if missing:
        raise ValueError(f"unclassified platform-bound fixture consumers: {missing}")
    unmarked = sorted(set(consumers) - marked)
    if unmarked:
        raise ValueError(f"platform-bound fixture consumers lack explicit pytest markers: {unmarked}")
    return consumers


def audit_files(files: dict[str, str], selections: tuple[Selection, ...],
                exclusions: dict[str, Any]) -> dict[str, Any]:
    """Audit a source mapping, also usable with synthetic in-memory files."""
    excluded = exclusions["files"]
    for name, reason in excluded.items():
        if not isinstance(reason, str) or not reason.strip() or name not in files:
            raise ValueError(f"stale or unexplained exclusion: {name}")
    groups = exclusions.get("isolated_groups", {})
    grouped_files: set[str] = set()
    group_report = {}
    for group, definition in groups.items():
        paths = definition.get("files", [])
        if (not definition.get("reason", "").strip() or not definition.get("job")
                or not paths or len(paths) != len(set(paths))
                or any(path not in files or path in grouped_files for path in paths)):
            raise ValueError(f"stale or unexplained isolated group: {group}")
        grouped_files.update(paths)
        invocations = [selection for selection in selections if selection.isolated_group == group]
        if len(invocations) != 1:
            raise ValueError(f"isolated group {group} must run exactly once; found {len(invocations)}")
        selected = invocations[0]
        if (selected.job != definition["job"] or selected.directories != tuple(paths)
                or selected.ignored_files):
            raise ValueError(f"isolated group {group} must use its exact files and job")
        for path in paths:
            if any("slow" not in markers and not matches(selected.markers, markers)
                   for markers in source_test_markers(files[path])):
                raise ValueError(f"isolated group {group} hides fast tests in {path}")
        group_report[group] = {"job": selected.job, "files": paths, "invocations": len(invocations)}
    for selection in selections:
        if selection.isolated_group is not None and selection.isolated_group not in groups:
            raise ValueError(f"undocumented isolated group: {selection.isolated_group}")
        for name in selection.ignored_files:
            if name not in excluded and name not in grouped_files:
                raise ValueError(f"undocumented CI exclusion: {name}")
    archive_nodes = exclusions.get("archive_dependent_tests", [])
    platform_nodes = exclusions.get("platform_bound", [])
    for nodes, label in ((archive_nodes, "archive-dependent"), (platform_nodes, "platform-bound")):
        for node in nodes:
            name, _, test = node.partition("::")
            reason = exclusions.get("conditional_tests", {}).get(node)
            if (name not in files or test not in source_test_nodes(files[name])
                    or not isinstance(reason, str) or not reason.strip()):
                raise ValueError(f"stale or unexplained {label} test: {node}")
    uncovered, assignments = [], {}
    duplicates, repeated, conflicts = {}, {}, []
    for name, source in sorted(files.items()):
        path = PurePosixPath(name)
        markers = source_test_markers(source)
        candidates = [selection for selection in selections if name not in selection.ignored_files
                      and any(path.is_relative_to(directory) for directory in selection.directories)]
        jobs = sorted({selection.job for selection in candidates
                       if any(matches(selection.markers, value) for value in markers)})
        assignments[name] = jobs
        if len(jobs) > 1:
            duplicates[name] = jobs
        if any(sum(matches(selection.markers, value) for selection in candidates) > 1 for value in markers):
            repeated[name] = [selection.job for selection in candidates]
        if jobs and name in excluded:
            conflicts.append(name)
        if not jobs and name not in excluded:
            uncovered.append(name)
    if duplicates:
        raise ValueError(f"duplicate CI job assignments: {duplicates}")
    if repeated:
        raise ValueError(f"duplicate CI process invocations: {repeated}")
    if conflicts:
        raise ValueError(f"covered files conflict with exclusions: {conflicts}")
    return {"test_files": len(files), "uncovered_files": uncovered,
            "duplicate_files": duplicates, "duplicate_invocations": repeated, "conflicting_exclusions": conflicts,
            "isolated_groups": group_report, "archive_dependent_tests": archive_nodes,
            "platform_bound": platform_nodes,
            "explicit_file_exclusions": excluded, "assignments": assignments}


def audit(root: Path, selections: tuple[Selection, ...], exclusions: dict[str, Any]) -> dict[str, Any]:
    files = {path.relative_to(root).as_posix(): path.read_text()
             for path in (root / "tests").rglob("test_*.py")}
    return audit_files(files, selections, exclusions)

"""CI guards for GitHub MathJax macro compatibility and local Markdown links."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]

DISALLOWED_MATHJAX_MACROS = (
    "operatorname",
    "DeclareMathOperator",
    "newcommand",
    "renewcommand",
    "providecommand",
    "choose",
    "bm",
    "SI",
    "si",
    "pu",
    "ce",
)
MACRO_PATTERN = re.compile(
    r"\\(?:" + "|".join(DISALLOWED_MATHJAX_MACROS) + r")\b"
)
MD_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
IMG_SRC_PATTERN = re.compile(r'<img\s+[^>]*src="([^"]+)"')

ACTIVE_GUIDE_DOCS = (
    "README.md",
    "CLAUDE.md",
    "docs/DevelopmentGuide.md",
    "docs/RetiredModelsV1.md",
    "docs/contracts/README.md",
    "reproducibility/README.md",
    "examples/research_1d/README.md",
    "examples/research_2d/README.md",
)


def _tracked_markdown_files() -> list[Path]:
    out = subprocess.check_output(
        ["git", "ls-files", "*.md"], cwd=ROOT, text=True
    ).splitlines()
    return [ROOT / rel for rel in out if (ROOT / rel).is_file()]


def _iter_non_code_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    in_fence = False
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append((lineno, raw))
    return lines


def test_markdown_files_contain_no_disallowed_github_mathjax_macros() -> None:
    violations: list[str] = []
    for path in _tracked_markdown_files():
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in _iter_non_code_lines(path):
            match = MACRO_PATTERN.search(line)
            if match is not None:
                violations.append(
                    f"{rel}:{lineno}: disallowed macro {match.group(0)!r}"
                )
            stripped = line.strip()
            if "$$" in line and stripped != "$$":
                violations.append(
                    f"{rel}:{lineno}: inline/single-line $$ fence (put $$ on its own line)"
                )
            if stripped.startswith("|") and "$" in stripped:
                for span in re.finditer(r"\$(?!\$)(.+?)\$", line):
                    if re.search(r"(?<!\\)\|", span.group(1)):
                        violations.append(
                            f"{rel}:{lineno}: unescaped '|' inside table math {span.group(0)!r}"
                        )
    assert not violations, "\n".join(violations)


def test_active_readme_and_guide_relative_links_resolve() -> None:
    broken: list[str] = []
    for rel in ACTIVE_GUIDE_DOCS:
        doc_path = ROOT / rel
        assert doc_path.is_file(), f"missing expected documentation file: {rel}"
        for lineno, line in _iter_non_code_lines(doc_path):
            targets = [
                *MD_LINK_PATTERN.findall(line),
                *IMG_SRC_PATTERN.findall(line),
            ]
            for raw_target in targets:
                target = raw_target.strip().strip("<>")
                if not target or target.startswith(
                    ("http://", "https://", "mailto:", "#")
                ):
                    continue
                path_part = target.split("#", 1)[0]
                if not path_part:
                    continue
                resolved = (doc_path.parent / path_part).resolve()
                if not resolved.exists():
                    broken.append(f"{rel}:{lineno}: broken link -> {raw_target!r}")
    assert not broken, "\n".join(broken)

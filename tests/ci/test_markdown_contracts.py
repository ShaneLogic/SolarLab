"""CI guards for GitHub cmark-gfm / MathJax compatibility and local Markdown links."""

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
    "colorbox",
    "fcolorbox",
)
MACRO_PATTERN = re.compile(
    r"\\(?:" + "|".join(DISALLOWED_MATHJAX_MACROS) + r")\b"
)
INLINE_CODE_PATTERN = re.compile(r"(`+)([^`]*?)\1")
MD_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
IMG_SRC_PATTERN = re.compile(r'<img\s+[^>]*src="([^"]+)"')

ACTIVE_GUIDE_DOCS = (
    "README.md",
    "CLAUDE.md",
    "docs/DevelopmentGuide.md",
    "docs/RetiredModelsV1.md",
    "docs/BeerLambertReferenceProtocolV1.md",
    "docs/PhysicsCorrectionsScopeV1.md",
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


def _check_inline_gfm_math_line(rel: str, lineno: int, line: str) -> list[str]:
    """Validate cmark-gfm inline $...$ delimiter and backslash-escape invariants."""
    errors: list[str] = []
    scrubbed = INLINE_CODE_PATTERN.sub("CODE", line)
    if "$" not in scrubbed:
        return errors

    dollar_positions = [
        m.start() for m in re.finditer(r"(?<!\\)\$", scrubbed)
    ]
    if len(dollar_positions) % 2 != 0:
        errors.append(f"{rel}:{lineno}: unbalanced inline '$' delimiters")
        return errors

    brace_underscore_spans = 0
    for idx in range(0, len(dollar_positions), 2):
        s_pos = dollar_positions[idx]
        e_pos = dollar_positions[idx + 1]
        span = scrubbed[s_pos + 1 : e_pos]

        if s_pos > 0 and scrubbed[s_pos - 1] in "-[" or (
            s_pos > 0 and scrubbed[s_pos - 1].isalnum()
        ):
            errors.append(
                f"{rel}:{lineno}: invalid character {scrubbed[s_pos - 1]!r} "
                f"immediately before opening '$' in ${span}$"
            )
        if e_pos + 1 < len(scrubbed) and scrubbed[e_pos + 1].isalnum():
            errors.append(
                f"{rel}:{lineno}: alphanumeric character {scrubbed[e_pos + 1]!r} "
                f"immediately after closing '$' in ${span}$"
            )
        if span.endswith(")") and e_pos + 1 < len(scrubbed) and scrubbed[e_pos + 1] == ")":
            errors.append(
                f"{rel}:{lineno}: cmark-gfm breaks on ')$)' boundary in (${span}$)"
            )
        if re.search(r"\\[%,]", span):
            errors.append(
                f"{rel}:{lineno}: CommonMark strips backslash in '\\%' or '\\,' "
                f"inside single-dollar math ${span}$"
            )
        if any(ord(ch) > 127 for ch in span):
            errors.append(
                f"{rel}:{lineno}: non-ASCII character inside inline math ${span}$"
            )
        if span.count("*") >= 2:
            errors.append(
                f"{rel}:{lineno}: multiple '*' in inline math ${span}$ "
                "(use \\ast to prevent GFM <em> emphasis collision)"
            )
        if re.search(r"\}_(?=[{\\])", span):
            brace_underscore_spans += 1
        if line.lstrip().startswith("|") and re.search(r"(?<!\\)\|", span):
            errors.append(
                f"{rel}:{lineno}: unescaped '|' inside table math ${span}$"
            )

    if brace_underscore_spans > 0 and scrubbed.count("_") >= 2 and len(dollar_positions) >= 4:
        errors.append(
            f"{rel}:{lineno}: '}}_' inside multi-span inline math risks GFM '_' emphasis across '$'"
        )
    return errors


def test_markdown_files_contain_no_disallowed_github_mathjax_macros() -> None:
    violations: list[str] = []
    for path in _tracked_markdown_files():
        rel = path.relative_to(ROOT).as_posix()
        in_display_math = False
        for lineno, line in _iter_non_code_lines(path):
            match = MACRO_PATTERN.search(line)
            if match is not None:
                violations.append(
                    f"{rel}:{lineno}: disallowed macro {match.group(0)!r}"
                )
            if "$$" in line:
                if line != "$$":
                    violations.append(
                        f"{rel}:{lineno}: '$$' fence must stand alone at column 0"
                    )
                else:
                    in_display_math = not in_display_math
                continue
            if not in_display_math:
                violations.extend(_check_inline_gfm_math_line(rel, lineno, line))
        if in_display_math:
            violations.append(f"{rel}: unclosed '$$' display math block")
    assert not violations, "\n".join(violations)


def test_active_readme_and_guide_relative_links_resolve() -> None:
    broken: list[str] = []
    for rel in ACTIVE_GUIDE_DOCS:
        doc_path = ROOT / rel
        assert doc_path.is_file(), f"missing expected documentation file: {rel}"
        for lineno, line in _iter_non_code_lines(doc_path):
            scrubbed = INLINE_CODE_PATTERN.sub("", line)
            targets = [
                *MD_LINK_PATTERN.findall(scrubbed),
                *IMG_SRC_PATTERN.findall(scrubbed),
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

"""Tagged-text parser. Hard-fails on raw `_` / `^` to enforce spec § 7.3."""
import re
from typing import Literal, List, Tuple

Role = Literal["normal", "sub", "sup"]
Segment = Tuple[str, Role]

_TAG_RE = re.compile(r"\[(sub|sup)\](.*?)\[/\1\]")


def parse_segments(text: str) -> List[Segment]:
    if "_" in text:
        raise ValueError(
            f"underscore literal found in slide text: {text!r}. "
            "Use [sub]…[/sub] instead (spec § 7.3)."
        )
    if "^" in text:
        raise ValueError(
            f"caret literal found in slide text: {text!r}. "
            "Use [sup]…[/sup] instead (spec § 7.3)."
        )
    if "[sub]" in text and "[/sub]" not in text:
        raise ValueError(f"unmatched [sub] tag in {text!r}")
    if "[sup]" in text and "[/sup]" not in text:
        raise ValueError(f"unmatched [sup] tag in {text!r}")

    out: List[Segment] = []
    cursor = 0
    for match in _TAG_RE.finditer(text):
        if match.start() > cursor:
            out.append((text[cursor:match.start()], "normal"))
        out.append((match.group(2), match.group(1)))
        cursor = match.end()
    if cursor < len(text):
        out.append((text[cursor:], "normal"))
    return out


def write_segments(paragraph, segments: List[Segment], size_bold, color):
    """Append each segment as a pptx text run; sub/sup via XML baseline."""
    from pptx.oxml.ns import qn
    from theme.fonts import apply

    for text, role in segments:
        run = paragraph.add_run()
        run.text = text
        apply(run, size_bold, color)
        if role == "sub":
            run._r.get_or_add_rPr().set("baseline", "-25000")
        elif role == "sup":
            run._r.get_or_add_rPr().set("baseline", "30000")

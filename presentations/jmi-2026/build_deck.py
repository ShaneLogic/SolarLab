"""End-to-end PPTX builder. Reads copy.yaml, emits output/jmi-2026.pptx."""
import argparse
from pathlib import Path
import yaml
from pptx import Presentation
from pptx.util import Inches

from theme.layouts import add_cover, add_divider, add_content


def build(*, copy_path: Path = Path("copy.yaml"),
          out_path: Path = Path("output/jmi-2026.pptx")) -> int:
    spec = yaml.safe_load(copy_path.read_text())
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)            # 16:9
    total_content = spec["deck"]["total_content_slides"]
    for s in spec["slides"]:
        kind = s["kind"]
        if kind == "cover":
            add_cover(prs,
                      title=spec["deck"]["title"],
                      authors=spec["deck"]["authors"],
                      affiliation=spec["deck"]["affiliation"])
        elif kind == "divider":
            add_divider(prs, number=s["number"], title=s["title"])
        elif kind == "content":
            fig = s.get("figure")
            fig_path = Path(fig) if fig else None
            if fig_path and not fig_path.exists():
                raise FileNotFoundError(
                    f"Slide {s['index']} references missing figure {fig_path}. "
                    "Run figure scripts first.")
            add_content(prs,
                        section_label=s["section_label"],
                        slide_index=s["index"],
                        total=total_content,
                        title=s["title"],
                        subtitle=s["subtitle"],
                        bullets=s.get("bullets") or [],
                        figure_path=fig_path)
        else:
            raise ValueError(f"unknown slide kind: {kind}")
    out_path.parent.mkdir(exist_ok=True)
    prs.save(str(out_path))
    return len(prs.slides)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--copy", default="copy.yaml")
    p.add_argument("--out", default="output/jmi-2026.pptx")
    args = p.parse_args()
    n = build(copy_path=Path(args.copy), out_path=Path(args.out))
    print(f"built {args.out} with {n} slides")

"""Cover, divider, content layout builders (spec § 7.4 & § 7.5)."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

from theme import palette, fonts
from theme.runs import parse_segments, write_segments


_BLANK_LAYOUT = 6        # PowerPoint blank layout index


def _add_blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[_BLANK_LAYOUT])


def _add_textbox(slide, left, top, width, height, text, role,
                 color, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    write_segments(p, parse_segments(text), role, color)
    return tb


def add_cover(prs, *, title, authors, affiliation):
    slide = _add_blank(prs)
    sw, sh = prs.slide_width, prs.slide_height
    _add_textbox(slide, Emu(0), Emu(int(sh * 0.40)), sw, Inches(1),
                 title, fonts.TITLE, palette.INK, PP_ALIGN.CENTER)
    _add_textbox(slide, Emu(0), Emu(int(sh * 0.55)), sw, Inches(0.5),
                 authors, fonts.SUBTITLE, palette.BODY, PP_ALIGN.CENTER)
    _add_textbox(slide, Emu(0), Emu(int(sh * 0.62)), sw, Inches(0.4),
                 affiliation, fonts.CAPTION, palette.HAIRLINE, PP_ALIGN.CENTER)
    # accent line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Emu(int(sw * 0.42)), Emu(int(sh * 0.52)),
                                  Emu(int(sw * 0.16)), Emu(int(sh * 0.004)))
    line.fill.solid()
    line.fill.fore_color.rgb = palette.ACCENT
    line.line.fill.background()
    return slide


def add_divider(prs, *, number, title):
    """Full-bleed numeral divider (spec § 7.5)."""
    slide = _add_blank(prs)
    sw, sh = prs.slide_width, prs.slide_height
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, sw, sh)
    bg.fill.solid()
    bg.fill.fore_color.rgb = palette.INK
    bg.line.fill.background()
    _add_textbox(slide, Emu(int(sw * 0.06)), Emu(int(sh * 0.10)),
                 Emu(int(sw * 0.6)), Inches(4),
                 number, fonts.DIVIDER_NUMERAL, palette.ACCENT)
    _add_textbox(slide, Emu(int(sw * 0.06)), Emu(int(sh * 0.78)),
                 Emu(int(sw * 0.88)), Inches(0.6),
                 title, fonts.DIVIDER_TITLE, palette.BODY)
    return slide


def add_content(prs, *, section_label, slide_index, total, title,
                subtitle, bullets, figure_path=None):
    slide = _add_blank(prs)
    sw, sh = prs.slide_width, prs.slide_height
    # header row
    _add_textbox(slide, Inches(0.5), Inches(0.3), Inches(7), Inches(0.3),
                 section_label.upper(), fonts.SECTION_LABEL, palette.BODY)
    _add_textbox(slide, Inches(7.5), Inches(0.3), Inches(2), Inches(0.3),
                 f"{slide_index} / {total}", fonts.SLIDE_COUNTER,
                 palette.HAIRLINE, align=PP_ALIGN.RIGHT)
    # body title + subtitle
    _add_textbox(slide, Inches(0.5), Inches(0.8), Inches(9), Inches(0.6),
                 title, fonts.TITLE, palette.INK)
    _add_textbox(slide, Inches(0.5), Inches(1.4), Inches(9), Inches(0.4),
                 subtitle, fonts.SUBTITLE, palette.BODY)
    # figure (left) + bullets (right)
    if figure_path:
        # Fit picture inside a 5.4 x 3.2 inch box, preserving native aspect.
        from PIL import Image
        with Image.open(str(figure_path)) as im:
            src_w, src_h = im.size
        max_w_in, max_h_in = 5.4, 3.2
        src_aspect = src_w / src_h
        if src_aspect > (max_w_in / max_h_in):
            fig_w_in = max_w_in
            fig_h_in = max_w_in / src_aspect
        else:
            fig_h_in = max_h_in
            fig_w_in = max_h_in * src_aspect
        # Center inside the 0.5..5.9 / 2.0..5.2 box.
        x_in = 0.5 + (max_w_in - fig_w_in) / 2.0
        y_in = 2.0 + (max_h_in - fig_h_in) / 2.0
        slide.shapes.add_picture(str(figure_path),
                                 Inches(x_in), Inches(y_in),
                                 width=Inches(fig_w_in),
                                 height=Inches(fig_h_in))
        bullet_left = Inches(6.2)
        bullet_w = Inches(3.3)
    else:
        bullet_left = Inches(0.5)
        bullet_w = Inches(9)
    if bullets:
        tb = slide.shapes.add_textbox(bullet_left, Inches(2.0),
                                      bullet_w, Inches(3.2))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT
            write_segments(p, parse_segments(f"•  {line}"),
                           fonts.BODY, palette.BODY)
    # footer
    _add_textbox(slide, Inches(0.5), Inches(5.05), Inches(7), Inches(0.3),
                 "Chen et al. — JMI 2026", fonts.FOOTER, palette.HAIRLINE)
    _add_textbox(slide, Inches(7.5), Inches(5.05), Inches(2), Inches(0.3),
                 "SOLARLAB", fonts.FOOTER, palette.ACCENT, align=PP_ALIGN.RIGHT)
    return slide

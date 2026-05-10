"""Arial-only size table (spec § 7.2)."""
from pptx.util import Pt

FONT_NAME = "Arial"

# (size_pt, bold) by role
TITLE = (Pt(28), True)            # cover only
CONTENT_TITLE = (Pt(22), True)    # content slides — fits long titles on one line
SUBTITLE = (Pt(15), False)
BODY = (Pt(13), False)
CAPTION = (Pt(11), False)
FOOTER = (Pt(10), False)
DIVIDER_NUMERAL = (Pt(250), True)
DIVIDER_TITLE = (Pt(36), True)
SECTION_LABEL = (Pt(11), False)         # uppercase header on content slides
SLIDE_COUNTER = (Pt(11), False)

def apply(run, size_bold, color):
    """Set run font to Arial with given (size, bold) and RGBColor."""
    run.font.name = FONT_NAME
    run.font.size = size_bold[0]
    run.font.bold = size_bold[1]
    run.font.color.rgb = color

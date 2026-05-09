"""P1 graphite + crimson palette (spec § 7.1)."""
from pptx.dml.color import RGBColor

INK = RGBColor(0x1c, 0x25, 0x33)         # titles, axes, divider bg
BODY = RGBColor(0x47, 0x55, 0x69)         # body text, section labels
HAIRLINE = RGBColor(0xcb, 0xd5, 0xe1)     # rules, slide counter
TINT = RGBColor(0xf5, 0xf7, 0xfa)         # background tint
ACCENT = RGBColor(0xc0, 0x39, 0x2b)       # crimson — divider numerals, brand mark
WHITE = RGBColor(0xff, 0xff, 0xff)

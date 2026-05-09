from pptx.dml.color import RGBColor
from theme.palette import INK, BODY, HAIRLINE, TINT, ACCENT

def test_palette_hex_values():
    assert INK == RGBColor(0x1c, 0x25, 0x33)
    assert BODY == RGBColor(0x47, 0x55, 0x69)
    assert HAIRLINE == RGBColor(0xcb, 0xd5, 0xe1)
    assert TINT == RGBColor(0xf5, 0xf7, 0xfa)
    assert ACCENT == RGBColor(0xc0, 0x39, 0x2b)

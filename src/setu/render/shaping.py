"""Correct Kannada text shaping via HarfBuzz.

Plain glyph-per-codepoint rendering (what Pillow's default text drawing
does without a raqm build) is wrong for Kannada: consonant clusters form
conjunct glyphs, and vowel signs reorder around the base consonant. This
module shapes text properly using HarfBuzz before anything gets rasterised
— see the discussion in the project history for why this is not optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import uharfbuzz as hb

from setu.render.fonts import FontSpec

_HB_UNITS_PER_EM = 1000  # HarfBuzz's scale is independent of the font's own units-per-em


@dataclass(frozen=True)
class ShapedGlyph:
    glyph_id: int
    x_advance: float  # in font design units at the requested pixel size
    x_offset: float
    y_offset: float


@dataclass(frozen=True)
class ShapedLine:
    glyphs: list[ShapedGlyph]
    missing_glyph_codepoints: list[int]  # codepoints in `text` HarfBuzz couldn't map to a glyph


class ShapingFont:
    """Wraps one HarfBuzz font instance for a given FontSpec and pixel size."""

    def __init__(self, spec: FontSpec, pixel_size: int):
        with open(spec.path, "rb") as f:
            font_data = f.read()
        face = hb.Face(hb.Blob(font_data), spec.face_index)
        self.hb_font = hb.Font(face)
        self.hb_font.scale = (_HB_UNITS_PER_EM, _HB_UNITS_PER_EM)
        self.pixel_size = pixel_size
        self.spec = spec

    def shape_line(self, text: str, language: str = "kn", script: str = "Knda") -> ShapedLine:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        # Force Kannada script/language rather than trusting guesses on
        # short strings (guess_segment_properties can misdetect script on
        # a handful of characters, especially punctuation-only runs).
        buf.script = script
        buf.language = language
        buf.direction = "ltr"
        hb.shape(self.hb_font, buf)

        scale = self.pixel_size / _HB_UNITS_PER_EM
        glyphs = [
            ShapedGlyph(
                glyph_id=info.codepoint,
                x_advance=pos.x_advance * scale,
                x_offset=pos.x_offset * scale,
                y_offset=pos.y_offset * scale,
            )
            for info, pos in zip(buf.glyph_infos, buf.glyph_positions)
        ]

        # Glyph id 0 is HarfBuzz's .notdef — the font has no glyph for
        # whatever codepoint produced it. This is exactly the failure mode
        # that would silently drop archaic letters (ಱ, ೞ) if a font lacks
        # them; surface it instead of rendering a blank box unnoticed.
        # Use `.cluster` (the source character index), not positional zip
        # against `text` — shaping can merge or reorder characters into a
        # different number of glyphs, so glyph N is not character N.
        missing = [
            ord(text[info.cluster])
            for info in buf.glyph_infos
            if info.codepoint == 0 and info.cluster < len(text)
        ]
        return ShapedLine(glyphs=glyphs, missing_glyph_codepoints=missing)

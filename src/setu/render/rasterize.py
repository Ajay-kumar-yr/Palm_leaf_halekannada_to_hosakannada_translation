"""Turn HarfBuzz-shaped glyphs into a clean grayscale line-image bitmap.

This is the "just draw correctly shaped text" half of the renderer.
Damage (Part 3.2 of the roadmap: texture, ink bleed, warp, skew, holes) is
applied afterwards by `setu.render.damage` — keep this module free of any
of that so the two concerns stay testable independently.
"""

from __future__ import annotations

import freetype
import numpy as np

from setu.render.fonts import FontSpec
from setu.render.shaping import ShapedLine, ShapingFont

# freetype uses 26.6 fixed-point for sizes/positions: 1 pixel == 64 units.
_FT_SUBPIXEL = 64


def render_clean_line(
    text: str,
    font_spec: FontSpec,
    pixel_size: int = 48,
    line_height: int = 64,
    pad_x: int = 8,
) -> tuple[np.ndarray, list[int]]:
    """Render `text` in `font_spec` to a grayscale uint8 array, ink=255 on
    a black (0) background, height `line_height`, variable width.

    Returns (image, missing_glyph_codepoints) — the caller decides whether
    missing glyphs (see `ShapingFont.shape_line`) are fatal for this line.
    """
    shaper = ShapingFont(font_spec, pixel_size)
    shaped: ShapedLine = shaper.shape_line(text)

    face = freetype.Face(font_spec.path)
    if font_spec.face_index:
        face = freetype.Face(font_spec.path)
        face.select_charmap(freetype.FT_ENCODING_UNICODE)
    face.set_pixel_sizes(0, pixel_size)

    # First pass: rasterise every glyph once, record per-glyph bitmaps and
    # metrics, and accumulate the pen position to find the required canvas
    # width and the vertical extent above/below the baseline.
    pen_x = 0.0
    glyph_bitmaps = []  # (bitmap_2d, draw_x, draw_y_from_top_of_bitmap, top_bearing)
    max_ascent = 0
    max_descent = 0

    for g in shaped.glyphs:
        face.load_glyph(g.glyph_id, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
        bmp = face.glyph.bitmap
        if bmp.width and bmp.rows:
            buf = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.width)
        else:
            buf = np.zeros((0, 0), dtype=np.uint8)

        top = face.glyph.bitmap_top  # distance from baseline to top of bitmap
        left = face.glyph.bitmap_left
        draw_x = pen_x + g.x_offset + left
        glyph_bitmaps.append((buf, draw_x, top))

        max_ascent = max(max_ascent, top)
        max_descent = max(max_descent, bmp.rows - top)
        pen_x += g.x_advance

    canvas_width = int(np.ceil(pen_x)) + 2 * pad_x
    canvas_width = max(canvas_width, 1)
    canvas_height = line_height

    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    # Baseline placed so the tallest ascender/descender combo is centred
    # with a little headroom, then clamped into the fixed line_height.
    content_height = max_ascent + max_descent
    baseline_y = min(
        canvas_height - 4, max(4, (canvas_height - content_height) // 2 + max_ascent)
    )

    for buf, draw_x, top in glyph_bitmaps:
        if buf.size == 0:
            continue
        x0 = int(round(draw_x)) + pad_x
        y0 = baseline_y - top
        h, w = buf.shape
        x1, y1 = x0 + w, y0 + h
        # Clip against canvas bounds — a mis-shaped or oversized glyph
        # should not crash generation; it should be visibly clipped so a
        # human reviewing sample output notices something is off.
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x1, canvas_width), min(y1, canvas_height)
        if cx0 >= cx1 or cy0 >= cy1:
            continue
        src = buf[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        # Additive blend (max, not overwrite) so overlapping glyph ink
        # (common with reordered vowel signs) doesn't clip one glyph out.
        canvas[cy0:cy1, cx0:cx1] = np.maximum(canvas[cy0:cy1, cx0:cx1], src)

    return canvas, shaped.missing_glyph_codepoints

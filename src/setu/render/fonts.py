"""Font inventory for rendering old-Kannada text.

Each entry names a font file already present on this machine plus a
one-line licence note. Before adding a font here, check its licence —
CLAUDE.md / the roadmap (Part 3.2 of v2, still true in v4) calls this out
explicitly. Fonts that render Kannada script must also carry Kannada
glyph coverage for the archaic letters ಱ and ೞ where possible; if a font
lacks them, HarfBuzz will report `.notdef` glyphs for those codepoints and
`shaping.shape_line` will flag it (see `missing_glyphs`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FontSpec:
    name: str
    path: str
    face_index: int
    licence_note: str


# Windows ships Nirmala UI with full Indic script coverage. It is not an
# open-licence font — fine for local development and internal reports, but
# it must NOT ship as a redistributed asset (e.g. baked into a demo
# checkpoint or included as a repo file) without checking Microsoft's font
# EULA first. Treat this as a placeholder until Noto Sans Kannada / Tunga /
# other explicitly-licensed Kannada fonts are added — see the "Ask before"
# rule in CLAUDE.md for adding new font assets.
_CANDIDATES = [
    FontSpec(
        name="Nirmala UI",
        path=r"C:\Windows\Fonts\Nirmala.ttc",
        face_index=0,
        licence_note="Microsoft system font — dev/local use only, do not redistribute.",
    ),
]


def available_fonts() -> list[FontSpec]:
    """Return the FontSpecs whose font file actually exists on disk."""
    return [f for f in _CANDIDATES if Path(f.path).exists()]


def require_fonts() -> list[FontSpec]:
    fonts = available_fonts()
    if not fonts:
        raise RuntimeError(
            "No usable Kannada fonts found. Add at least one FontSpec to "
            "src/setu/render/fonts.py pointing at a real font file, and "
            "check its licence before using it for anything beyond local "
            "development."
        )
    return fonts

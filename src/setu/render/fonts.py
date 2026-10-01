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


# Navilu (S.P. Aravinda, SIL Open Font License 1.1 -- freely usable and
# redistributable, unlike Nirmala below) is a genuine handwriting-style
# Kannada font, not a UI/print font -- verified to cover both archaic
# letters (ಱ, ೞ) and all but 2 of the 65 WX-scheme characters (candrabindu
# and avagraha, both absent from the S1 corpus anyway). This replaced
# Nirmala UI as the only font after visual comparison through the full
# damage pipeline showed Nirmala's uniform, geometrically clean letterforms
# read as "typed text pasted on a photo" rather than inscribed writing --
# see data/external_models/fonts/navilu/PROVENANCE.md.
#
# Nirmala UI is kept registered but unused by default: it's a Windows
# system font, not an open licence -- fine for local dev, but must NOT
# ship as a redistributed asset (e.g. baked into a demo checkpoint or
# committed as a repo file) without checking Microsoft's font EULA. See
# the "Ask before" rule in CLAUDE.md for adding new font assets.
_CANDIDATES = [
    FontSpec(
        name="Navilu",
        path="data/external_models/fonts/navilu/Navilu.ttf",
        face_index=0,
        licence_note="SIL Open Font License 1.1 -- freely usable and redistributable.",
    ),
]

_NIRMALA_FALLBACK = FontSpec(
    name="Nirmala UI",
    path=r"C:\Windows\Fonts\Nirmala.ttc",
    face_index=0,
    licence_note="Microsoft system font — dev/local use only, do not redistribute.",
)


def available_fonts() -> list[FontSpec]:
    """Return the FontSpecs whose font file actually exists on disk. Falls
    back to Nirmala UI only if Navilu itself is missing (e.g. not yet
    downloaded on this machine) -- rendering should never silently produce
    zero fonts just because the preferred one isn't present yet."""
    found = [f for f in _CANDIDATES if Path(f.path).exists()]
    if not found and Path(_NIRMALA_FALLBACK.path).exists():
        return [_NIRMALA_FALLBACK]
    return found


def glyph_fallback_font() -> FontSpec | None:
    """Navilu (549 glyphs) is missing danda/double-danda (।॥) -- real,
    common verse-boundary marks in old-Kannada poetry, affecting ~15% of
    the S1 corpus. Rather than silently dropping that much data (and
    disproportionately dropping the more traditional/liturgical verses
    that use dandas), the caller should retry a line that fails on the
    preferred font with this one before giving up. Nirmala UI has full
    coverage; a per-line font fallback is a small, deliberate compromise,
    not a silent inconsistency -- generate.py records which font was
    actually used per line in the manifest."""
    return _NIRMALA_FALLBACK if Path(_NIRMALA_FALLBACK.path).exists() else None


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

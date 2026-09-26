"""Background and hole/damage patches sourced from real HKHPL leaves.

Per the roadmap (v4 Part 6.1): synthetic images should be *conditioned on*
HKHPL, not invented from nothing — that's what lets the report say the
damage simulator models the appearance of the target collection rather
than making up noise. `data/real/` is where those source photos belong.

Until real HKHPL photos are added there, this module falls back to a
procedural placeholder texture so development isn't blocked — but it warns
loudly and exactly once per process, because silently training on
procedural noise instead of HKHPL-derived texture is the kind of thing
CLAUDE.md rule 6 ("never quietly change a setting") is there to prevent.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

_WARNED = False


def _warn_placeholder_once(real_dir: Path) -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    warnings.warn(
        f"No real HKHPL images found in {real_dir} — using a procedural "
        "placeholder texture instead. This is fine for pipeline "
        "development, but S1/S2 generated for actual training or reported "
        "numbers must use real HKHPL-derived patches. Add photos to "
        "data/real/ before that point.",
        stacklevel=2,
    )


def _list_real_images(real_dir: Path) -> list[Path]:
    """Recursively find real HKHPL photos under `real_dir`.

    The HKHPL drop nests photos several folders deep (e.g.
    `dataset/Dataset/Dataset/{train,val,test}/`,
    `dataset/hd_images/hd_images/`) rather than sitting flat in `real_dir`,
    so this walks the whole tree. `Ground_Truth_images` is excluded: those
    are pre-binarized single-channel outputs from the dataset's original
    binarization benchmark, not photos, and using them as a texture/hole
    source would defeat the point of conditioning on real leaf appearance
    (and per CLAUDE.md, Palmira must only ever see the original photo, so
    binarized crops shouldn't leak in here either).
    """
    if not real_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return [
        p
        for p in real_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in exts
        and not any("ground_truth" in part.lower() for part in p.parts)
    ]


def _procedural_leaf_patch(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A rough beige/brown fibrous-looking placeholder, height x width, uint8."""
    h, w = size
    base = rng.normal(loc=178, scale=10, size=(h, w))
    # Coarse low-frequency mottling so it doesn't look like flat noise.
    coarse = rng.normal(loc=0, scale=1, size=(max(h // 8, 1), max(w // 8, 1)))
    coarse = np.array(Image.fromarray(coarse.astype(np.float32)).resize((w, h), Image.BILINEAR))
    patch = base + coarse * 12
    return np.clip(patch, 0, 255).astype(np.uint8)


def sample_patch(
    real_dir: Path, size: tuple[int, int], rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    """Return (grayscale patch of `size` = (height, width), is_real).

    Draws a random crop from a random real HKHPL image if any are present
    in `real_dir`; otherwise returns a procedural placeholder and warns.
    """
    images = _list_real_images(real_dir)
    if not images:
        _warn_placeholder_once(real_dir)
        return _procedural_leaf_patch(size, rng), False

    h, w = size
    path = images[rng.integers(len(images))]
    img = Image.open(path).convert("L")
    iw, ih = img.size
    if iw < w or ih < h:
        img = img.resize((max(iw, w), max(ih, h)))
        iw, ih = img.size
    x0 = int(rng.integers(0, iw - w + 1))
    y0 = int(rng.integers(0, ih - h + 1))
    crop = img.crop((x0, y0, x0 + w, y0 + h))
    return np.array(crop, dtype=np.uint8), True

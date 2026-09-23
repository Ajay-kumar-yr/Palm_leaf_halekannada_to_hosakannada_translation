"""The five damage effects baked into S1/S2 images at generation time
(roadmap v4 §2.2): background texture, ink bleed, page warp, baseline
skew, holes and damage. These are the "slow, generate once" effects —
the cheap on-the-fly ones (rotation, brightness, blur, stretch) applied
fresh every time an image is used during training live in `augment.py`,
not here.

Everything in this module operates on a grayscale "ink mask" array
(uint8, 0 = no ink, 255 = full ink) as produced by
`rasterize.render_clean_line`, and `compose_damage` is the only function
other code should call — it runs all five effects in a fixed order and
returns the final damaged line image ready to save.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from setu.render.textures import sample_patch


@dataclass
class DamageParams:
    """Recorded per-image alongside the manifest entry so any run is
    reproducible and so a reviewer can see exactly what was applied
    (CLAUDE.md rule 4: a result without its config does not exist)."""

    skew_degrees: float
    bleed_radius: float
    bleed_smear_px: int
    warp_amplitude_px: float
    warp_wavelength_px: float
    texture_is_real: bool
    hole_count: int
    hole_is_real: list[bool] = field(default_factory=list)


def apply_baseline_skew(ink: np.ndarray, rng: np.random.Generator, max_degrees: float = 4.0) -> tuple[np.ndarray, float]:
    """Rotate the whole line a few degrees, as an unevenly-held leaf or an
    unevenly-cut line-segmentation box would produce."""
    degrees = float(rng.uniform(-max_degrees, max_degrees))
    img = Image.fromarray(ink)
    rotated = img.rotate(
        degrees, resample=Image.BILINEAR, expand=True, fillcolor=0
    )
    return np.array(rotated, dtype=np.uint8), degrees


def apply_ink_bleed(
    ink: np.ndarray, rng: np.random.Generator, max_radius: float = 1.6, max_smear_px: int = 2
) -> tuple[np.ndarray, float, int]:
    """Smudge ink edges, as absorption into palm-leaf fibre would. A
    Gaussian blur softens edges; an optional small directional smear
    (averaging the image with a shifted copy) mimics ink dragging along
    the leaf grain in one direction rather than spreading symmetrically."""
    radius = float(rng.uniform(0.4, max_radius))
    img = Image.fromarray(ink).filter(ImageFilter.GaussianBlur(radius))
    smeared = np.array(img, dtype=np.float32)

    smear_px = int(rng.integers(0, max_smear_px + 1))
    if smear_px > 0:
        shifted = np.roll(smeared, smear_px, axis=1)
        shifted[:, :smear_px] = 0
        smeared = np.maximum(smeared, shifted * 0.6)

    return np.clip(smeared, 0, 255).astype(np.uint8), radius, smear_px


def apply_page_warp(
    ink: np.ndarray, rng: np.random.Generator, max_amplitude_px: float = 3.0
) -> tuple[np.ndarray, float, float]:
    """Gentle vertical sinusoidal displacement per column, as a curled or
    bent leaf would produce. Implemented as a manual column-wise vertical
    shift so it needs nothing beyond numpy — no scipy dependency."""
    h, w = ink.shape
    amplitude = float(rng.uniform(0.0, max_amplitude_px))
    wavelength = float(rng.uniform(w * 0.6, w * 1.6))  # a fraction of a full sine cycle over the line
    phase = float(rng.uniform(0, 2 * np.pi))

    cols = np.arange(w)
    shift = amplitude * np.sin(2 * np.pi * cols / wavelength + phase)

    warped = np.zeros_like(ink)
    row_idx = np.arange(h)[:, None]  # (h, 1)
    src_rows = row_idx - shift[None, :]  # (h, w), floating source row per column
    src_rows_floor = np.floor(src_rows).astype(int)
    frac = src_rows - src_rows_floor

    for dr, weight_fn in ((0, lambda f: 1 - f), (1, lambda f: f)):
        sr = src_rows_floor + dr
        valid = (sr >= 0) & (sr < h)
        sr_clamped = np.clip(sr, 0, h - 1)
        gathered = np.take_along_axis(ink, sr_clamped, axis=0).astype(np.float32)
        weight = weight_fn(frac)
        warped = warped + np.where(valid, gathered * weight, 0).astype(np.float32)

    return np.clip(warped, 0, 255).astype(np.uint8), amplitude, wavelength


def apply_background_and_holes(
    ink: np.ndarray,
    rng: np.random.Generator,
    real_dir: Path,
    max_holes: int = 3,
    hole_size_range: tuple[int, int] = (6, 22),
) -> tuple[np.ndarray, bool, int, list[bool]]:
    """Composite the ink onto a real (or placeholder) background texture,
    then punch in a few holes/damaged patches on top — this order matters:
    holes must be able to erase both ink and background, as real leaf
    damage does, not just sit under the text."""
    h, w = ink.shape
    bg, bg_is_real = sample_patch(real_dir, (h, w), rng)

    # Ink drawn as darker-than-background strokes (palm-leaf styli
    # traditionally darken the leaf, e.g. via lampblack rubbed into
    # incised grooves) — alpha-blend by ink intensity.
    alpha = (ink.astype(np.float32) / 255.0)[..., None]
    ink_color = 40.0  # dark stroke value
    composed = bg.astype(np.float32)[..., None] * (1 - alpha) + ink_color * alpha
    composed = composed[..., 0]

    n_holes = int(rng.integers(0, max_holes + 1))
    hole_is_real: list[bool] = []
    for _ in range(n_holes):
        hh = int(rng.integers(*hole_size_range))
        hw = int(rng.integers(*hole_size_range))
        hh, hw = min(hh, h), min(hw, w)
        if hh < 2 or hw < 2:
            continue
        patch, patch_is_real = sample_patch(real_dir, (hh, hw), rng)
        hole_is_real.append(patch_is_real)
        y0 = int(rng.integers(0, h - hh + 1))
        x0 = int(rng.integers(0, w - hw + 1))
        # Soft-edged mask so the hole doesn't look like a pasted rectangle.
        yy, xx = np.mgrid[0:hh, 0:hw]
        cy, cx = hh / 2, hw / 2
        dist = np.sqrt(((yy - cy) / (hh / 2)) ** 2 + ((xx - cx) / (hw / 2)) ** 2)
        mask = np.clip(1.2 - dist, 0, 1)
        composed[y0 : y0 + hh, x0 : x0 + hw] = (
            composed[y0 : y0 + hh, x0 : x0 + hw] * (1 - mask) + patch.astype(np.float32) * mask
        )

    return np.clip(composed, 0, 255).astype(np.uint8), bg_is_real, n_holes, hole_is_real


def compose_damage(
    ink: np.ndarray, rng: np.random.Generator, real_dir: Path
) -> tuple[np.ndarray, DamageParams]:
    """Run all five baked-in damage effects in order and return the final
    image plus the parameters used, for the manifest."""
    skewed, skew_deg = apply_baseline_skew(ink, rng)
    bled, bleed_radius, smear_px = apply_ink_bleed(skewed, rng)
    warped, warp_amp, warp_wl = apply_page_warp(bled, rng)
    final, bg_real, n_holes, hole_real = apply_background_and_holes(warped, rng, real_dir)

    params = DamageParams(
        skew_degrees=skew_deg,
        bleed_radius=bleed_radius,
        bleed_smear_px=smear_px,
        warp_amplitude_px=warp_amp,
        warp_wavelength_px=warp_wl,
        texture_is_real=bg_real,
        hole_count=n_holes,
        hole_is_real=hole_real,
    )
    return final, params

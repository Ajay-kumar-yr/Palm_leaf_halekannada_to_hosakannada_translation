"""The damage effects baked into S1/S2 images at generation time (roadmap
v4 §2.2): stroke distortion, background texture, ink bleed, page warp,
baseline skew, holes and damage. These are the "slow, generate once"
effects — the cheap on-the-fly ones (rotation, brightness, blur, stretch)
applied fresh every time an image is used during training live in
`augment.py`, not here.

Stroke distortion was added after the fact (not one of the original five)
because a vector-font glyph's perfectly uniform, geometrically clean
outline is what makes a rendered line read as "typed text pasted onto a
photo" rather than hand-inscribed writing -- no amount of background/ink
compositing trickery fixes that if the strokes themselves are still
mechanically perfect. It's a smooth per-glyph elastic warp, deliberately
gentler than a from-scratch handwriting synthesis effort (that's out of
scope, same reasoning as the roadmap's already-cut effects) but enough to
break the too-perfect look while staying legible.

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

    distortion_max_px: float
    distortion_grid_px: int
    skew_degrees: float
    bleed_radius: float
    bleed_smear_px: int
    warp_amplitude_px: float
    warp_wavelength_px: float
    texture_is_real: bool
    hole_count: int
    hole_is_real: list[bool] = field(default_factory=list)


def apply_stroke_distortion(
    ink: np.ndarray, rng: np.random.Generator, max_disp_px: float = 0.35, grid_spacing_px: int = 10
) -> tuple[np.ndarray, float, int]:
    """Smooth elastic warp of the glyph shapes themselves, applied before
    any line-level transform. A coarse grid of random per-control-point
    displacements is bicubic-upsampled to a full per-pixel (dy, dx) field
    -- smooth because it's an upsample of a sparse grid, not per-pixel
    noise, which is what keeps whole letters coherent instead of
    shredding them. Bilinear resampling (manual, matching the style of
    apply_page_warp below -- no scipy dependency).

    Both parameters were tuned empirically against the full pipeline (this
    function's effect compounds with skew/bleed/warp downstream, so
    tuning it in isolation is misleading): grid_spacing_px much below ~9
    lets independent random offsets land within a single glyph's height
    and tears letters apart; max_disp_px much above ~0.4 becomes
    illegible once skew and page-warp stack on top of it.
    """
    h, w = ink.shape
    gh, gw = h // grid_spacing_px + 2, w // grid_spacing_px + 2
    dy_coarse = rng.uniform(-max_disp_px, max_disp_px, size=(gh, gw)).astype(np.float32)
    dx_coarse = rng.uniform(-max_disp_px, max_disp_px, size=(gh, gw)).astype(np.float32)
    dy = np.array(Image.fromarray(dy_coarse).resize((w, h), Image.BICUBIC), dtype=np.float32)
    dx = np.array(Image.fromarray(dx_coarse).resize((w, h), Image.BICUBIC), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_y, src_x = yy + dy, xx + dx
    y0 = np.clip(np.floor(src_y).astype(int), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(np.floor(src_x).astype(int), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(src_y - y0, 0, 1)
    fx = np.clip(src_x - x0, 0, 1)

    ink_f = ink.astype(np.float32)
    top = ink_f[y0, x0] * (1 - fx) + ink_f[y0, x1] * fx
    bot = ink_f[y1, x0] * (1 - fx) + ink_f[y1, x1] * fx
    distorted = np.clip(top * (1 - fy) + bot * fy, 0, 255).astype(np.uint8)
    return distorted, max_disp_px, grid_spacing_px


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

    # Ink darkens the LOCAL background multiplicatively rather than
    # replacing it with a flat value -- a flat replace (bg*(1-alpha) +
    # 40*alpha) makes every stroke read as a uniform overlay pasted on top
    # of the leaf, ignoring whatever grain/shading/staining is already at
    # that pixel. Real stylus-incised writing follows the leaf's surface:
    # ink pools darker in a locally rough/stained patch, lighter on a
    # locally clean one, so the same grain visible in the bare background
    # should still be visible faintly through the strokes. Multiplying by
    # a retained-brightness factor (rather than an additive/replace blend)
    # keeps that per-pixel texture proportional through the ink.
    alpha = (ink.astype(np.float32) / 255.0)[..., None]
    ink_retained_brightness = 0.15  # fraction of local bg value kept where fully inked
    composed = bg.astype(np.float32)[..., None] * (1 - alpha * (1 - ink_retained_brightness))
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
    """Run all baked-in damage effects in order and return the final image
    plus the parameters used, for the manifest."""
    distorted, dist_max, dist_grid = apply_stroke_distortion(ink, rng)
    skewed, skew_deg = apply_baseline_skew(distorted, rng)
    bled, bleed_radius, smear_px = apply_ink_bleed(skewed, rng)
    warped, warp_amp, warp_wl = apply_page_warp(bled, rng)
    final, bg_real, n_holes, hole_real = apply_background_and_holes(warped, rng, real_dir)

    params = DamageParams(
        distortion_max_px=dist_max,
        distortion_grid_px=dist_grid,
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

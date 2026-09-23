"""Cheap augmentation applied fresh every time an image is used during
training (roadmap v4 §2.2), as opposed to the five effects in `damage.py`
which are baked in once at generation time. Kept separate and deliberately
lightweight — Part 11 of the roadmap is explicit that expensive per-batch
warping here would starve this machine's 6-core CPU, which is why warp,
texture and holes are NOT here.

Not wired into a dataloader yet — this is the standalone building block
for whenever the recogniser's training loop is built.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def augment_line(
    img: np.ndarray,
    rng: np.random.Generator,
    max_rotation_deg: float = 1.5,
    brightness_range: tuple[float, float] = (0.85, 1.15),
    max_blur_radius: float = 0.6,
    max_stretch: float = 0.08,
) -> np.ndarray:
    """Apply a small random rotation, brightness shift, mild blur, and
    slight horizontal stretch. Cheap enough to run every epoch, every
    image, without becoming the training bottleneck."""
    pil_img = Image.fromarray(img)

    angle = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
    pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)

    stretch = float(rng.uniform(1 - max_stretch, 1 + max_stretch))
    w, h = pil_img.size
    pil_img = pil_img.resize((max(1, int(w * stretch)), h), Image.BILINEAR)

    factor = float(rng.uniform(*brightness_range))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)

    blur_radius = float(rng.uniform(0, max_blur_radius))
    if blur_radius > 0:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))

    return np.array(pil_img, dtype=np.uint8)

"""Real-photo binarization step for the actual inference pipeline (Palmira
-> cut lines -> **this** -> CRNN -> soft bridge -> modernizer). Synthetic
S1/S2 images never go through this -- they're already rendered as clean
grayscale line images with no separate binarization step needed.

Model: the same U-Net (S.P. Sajjan et al., "U-Net: Convolutional Neural
Network for binarization of Historical Kannada Handwritten Palm Leaf
Manuscripts") used to produce HKHPL's own Ground_Truth_images -- i.e. the
one binarizer with a direct, demonstrated track record on this exact
dataset, rather than a generic thresholding method picked arbitrarily.
Source: https://github.com/sajjanvsl/U-Net-CNN-for-binarization-of-Historical-Kannada-Handwritten-Palm-Leaf-Manuscripts
(MIT licensed). Weights (unet_best_weights.pth, sha256
aef2dc11584ba489217cf7f41642583059046f2341efda1178b4867238186f44) are
committed nowhere -- gitignored like every other model checkpoint in this
repo; re-download from the source above if missing.

Per CLAUDE.md: Palmira must receive the ORIGINAL photo, never this
binarized output (Palmira was trained on colour photos; black-and-white
is out-of-distribution for it) -- this step runs strictly AFTER Palmira
has already found and cut the lines, on each cut line crop, immediately
before the CRNN.

Preprocessing/postprocessing replicated exactly from the source repo's
run.py (not reinterpreted): RGB uint8 -> [0,1] float tensor, model forward
(sigmoid activation is baked into the model itself), then a per-image
min-max normalize and invert. That inversion is deliberate on the source
author's part, not a mistake to "fix" -- it's what actually produces
dark-ink-on-light-page output from this particular model's raw sigmoid
range, confirmed by matching HKHPL's own Ground_Truth_images convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

WEIGHTS_PATH = Path("data/external_models/sajjan_unet/unet_best_weights.pth")


def load_binarizer(weights_path: Path = WEIGHTS_PATH, device: str | torch.device = "cpu"):
    import segmentation_models_pytorch as smp  # deferred: only needed on the real-photo inference path

    model = smp.Unet(in_channels=3, classes=1, activation="sigmoid", encoder_weights=None)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def binarize(image: Image.Image, model, device: str | torch.device = "cpu") -> np.ndarray:
    """image: a real photo (or a Palmira-cut line crop) as a PIL RGB image.
    Returns a grayscale uint8 array, ink-dark-on-light -- matching HKHPL's
    own Ground_Truth_images convention and the rest of this project's
    saved-image convention (see setu.render.generate)."""
    rgb = image.convert("RGB")
    x = torch.from_numpy(np.array(rgb)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)
    y = model(x)[0, 0].cpu()
    y_norm = 1 - (y - y.min()) / (y.max() - y.min())
    return (y_norm.numpy() * 255).astype(np.uint8)

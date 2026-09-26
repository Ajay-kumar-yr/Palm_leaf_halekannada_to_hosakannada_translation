"""CLAUDE.md rule 3: before any training run longer than 10 minutes, prove
the model can memorise 8 examples to near-zero error first. If it can't,
something is broken -- do not start the real run.

This renders 8 short Kannada lines directly (clean, undamaged -- damage
augmentation is not the point of this check; the point is proving the
CRNN + CTC training loop itself is wired correctly), trains the CRNN on
just those 8 examples, and asserts CER reaches ~0. Two of the eight lines
exercise the archaic letters ಱ and ೞ specifically, since those are the
letters a syllable-level (rather than WX) vocabulary would starve of
examples -- worth checking the whole image -> WX pipeline handles them
before any real run.

Usage:
    python -m setu.recogniser.memorize_check
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from torch import nn

from setu.data import wx
from setu.eval.metrics import cer
from setu.recogniser.model import CRNN, greedy_decode
from setu.render.fonts import require_fonts
from setu.render.rasterize import render_clean_line
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6: set random seeds in every entry point

LINES = [
    "ನಮಸ್ಕಾರ",              # ನಮಸ್ಕಾರ (namaskara)
    "ಕರ್ನಾಟಕ",              # ಕರ್ನಾಟಕ (Karnataka)
    "ಕನ್ನಡ",                          # ಕನ್ನಡ (Kannada)
    "ಮಱೆ",                                      # ಮಱೆ (archaic rY)
    "ಪೞೆಯ",                                # ಪೞೆಯ (archaic zY)
    "ಹಳ್ಳಿ",                          # ಹಳ್ಳಿ (village)
    "ಬೆಂಗಳೂರು",        # ಬೆಂಗಳೂರು (Bengaluru)
    "ಸಂಸ್ಕ್ರೃತ",  # ಸಂಸ್ಕೃತ (Sanskrit)
]

LINE_HEIGHT = 64
PIXEL_SIZE = 44
MAX_EPOCHS = 400
TARGET_CER = 0.0
LOG_EVERY = 25


def build_batch() -> tuple[torch.Tensor, list[list[int]], list[int]]:
    fonts = require_fonts()
    font = fonts[0]

    images = []
    label_seqs = []
    for text in LINES:
        img, missing = render_clean_line(text, font, PIXEL_SIZE, LINE_HEIGHT)
        if missing:
            raise RuntimeError(
                f"Font {font.name!r} is missing glyphs for {text!r} "
                f"(codepoints {[hex(c) for c in missing]}) -- pick a "
                f"different memorisation line or font before proceeding."
            )
        images.append(img)
        symbols = wx.encode(text)
        label_seqs.append([wx.SYMBOL_TO_INDEX[s] for s in symbols])

    max_w = max(img.shape[1] for img in images)
    batch = np.zeros((len(images), 1, LINE_HEIGHT, max_w), dtype=np.float32)
    widths = []
    for i, img in enumerate(images):
        h, w = img.shape
        batch[i, 0, :, :w] = img.astype(np.float32) / 255.0
        widths.append(w)

    return torch.from_numpy(batch), label_seqs, widths


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    images, label_seqs, widths = build_batch()
    model = CRNN()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    input_lengths = torch.tensor([model.output_length(w) for w in widths], dtype=torch.long)
    target_lengths = torch.tensor([len(s) for s in label_seqs], dtype=torch.long)
    targets = torch.tensor([idx for seq in label_seqs for idx in seq], dtype=torch.long)

    for w, ilen, tlen in zip(widths, input_lengths.tolist(), target_lengths.tolist()):
        if ilen < tlen:
            raise RuntimeError(
                f"CTC input length {ilen} < target length {tlen} for a line of "
                f"width {w}px -- CTC cannot align this; use a wider render."
            )

    run_dir = start_run(
        "crnn_memorize_check",
        {
            "seed": SEED,
            "lines": LINES,
            "line_height": LINE_HEIGHT,
            "pixel_size": PIXEL_SIZE,
            "max_epochs": MAX_EPOCHS,
            "target_cer": TARGET_CER,
            "vocab_size": len(wx.VOCAB),
        },
    )

    final_epoch, final_loss, final_cer, predictions = None, None, None, None
    model.train()
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        log_probs = model(images)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        # Decode without switching to eval(): with only 8 fixed examples,
        # BatchNorm's running mean/var (updated by a slow moving average)
        # hasn't converged to match this exact tiny batch, which makes
        # eval-mode CER lag behind the training loss and never reach 0
        # even once the model has clearly fit the data. Using the same
        # batch statistics for decode as for the loss keeps this check
        # honest about what it's actually testing (can the architecture +
        # training loop fit 8 examples), not eval-mode generalisation --
        # that's a separate concern for the real validation split later.
        with torch.no_grad():
            decoded = greedy_decode(model(images), input_lengths)

        cers = [
            cer(ref, hyp) for ref, hyp in zip(label_seqs, decoded)
        ]
        mean_cer = sum(cers) / len(cers)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  mean CER {mean_cer:.4f}")

        if mean_cer <= TARGET_CER:
            final_epoch, final_loss, final_cer = epoch, loss.item(), mean_cer
            predictions = decoded
            break
    else:
        final_epoch, final_loss, final_cer = MAX_EPOCHS, loss.item(), mean_cer
        predictions = decoded

    passed = final_cer <= TARGET_CER
    results = {
        "passed": passed,
        "epoch_reached": final_epoch,
        "final_loss": final_loss,
        "final_mean_cer": final_cer,
        "per_line": [
            {
                "text": LINES[i],
                "target_wx": wx.to_string(wx.encode(LINES[i])),
                "predicted_wx": wx.to_string([wx.INDEX_TO_SYMBOL[idx] for idx in predictions[i]]),
                "cer": cer(label_seqs[i], predictions[i]),
            }
            for i in range(len(LINES))
        ],
    }
    finish_run(run_dir, results)

    print(f"\nRun folder: {run_dir}")
    if passed:
        print(f"PASS -- memorised all 8 examples (CER {final_cer:.4f}) by epoch {final_epoch}.")
    else:
        print(
            f"FAIL -- CER {final_cer:.4f} after {final_epoch} epochs, did not reach "
            f"{TARGET_CER}. Per CLAUDE.md rule 3: something is broken in the model or "
            f"training loop -- do NOT start a real training run until this passes."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

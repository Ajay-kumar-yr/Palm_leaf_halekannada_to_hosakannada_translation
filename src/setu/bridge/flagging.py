"""Confidence-based flagging (CLAUDE.md Week 3: "confidence flagging";
reported as "one chart: accuracy on flagged versus unflagged lines" --
the point is showing that low-confidence lines really are where the
system is more often wrong, which is what makes flagging a useful
triage signal for a human reviewer rather than a random split).

CLAUDE.md doesn't pin down the exact aggregation/threshold (unlike the
bridge's precise per-frame procedure), so this makes one explicit,
documented choice: flag a line if its MEAN top-1 frame confidence (over
real, non-padding frames) falls below a threshold. Mean rather than min
-- a single noisy frame shouldn't flag an otherwise-confident line, which
is what min would do; mean reflects the line's overall reliability.
Reuses bridge machinery directly (same per-frame confidence computation
as soft_bridge.measure_confidence_distribution), per CLAUDE.md's own
framing ("reuses bridge machinery").

The threshold itself needs tuning against real validation data once a
trained CRNN exists -- same caveat as the bridge's temperature -- so it's
a required argument here, not a hardcoded default.
"""

from __future__ import annotations

import torch


def mean_line_confidence(log_probs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
    """log_probs: (T, B, C) CTC log-softmax output. input_lengths: (B,).
    Returns (B,): each line's mean top-1 confidence over its real frames."""
    t, b = log_probs.shape[0], log_probs.shape[1]
    device = log_probs.device
    frame_idx = torch.arange(t, device=device).unsqueeze(1)  # (T, 1)
    valid_mask = frame_idx < input_lengths.unsqueeze(0)  # (T, B)

    top1_conf = log_probs.exp().max(dim=-1).values  # (T, B)
    masked = top1_conf * valid_mask  # zero out padding frames
    return masked.sum(dim=0) / input_lengths.clamp(min=1).float()  # (B,)


def flag_lines(log_probs: torch.Tensor, input_lengths: torch.Tensor, threshold: float) -> torch.Tensor:
    """Returns (B,) bool: True = flagged (mean confidence below threshold,
    i.e. likely unreliable and worth a human reviewer's attention)."""
    return mean_line_confidence(log_probs, input_lengths) < threshold


def flagged_vs_unflagged_accuracy(
    is_flagged: torch.Tensor, line_cers: list[float]
) -> dict:
    """is_flagged: (B,) bool. line_cers: per-line CER (same order), e.g.
    from setu.eval.metrics.cer. Returns the comparison CLAUDE.md's chart
    is built from: mean CER (lower = more accurate) on each side. Empty
    groups report None rather than a misleading 0.0/nan."""
    if len(line_cers) != is_flagged.shape[0]:
        raise ValueError(f"is_flagged and line_cers must align, got {is_flagged.shape[0]} and {len(line_cers)}")

    flagged_cers = [c for c, f in zip(line_cers, is_flagged.tolist()) if f]
    unflagged_cers = [c for c, f in zip(line_cers, is_flagged.tolist()) if not f]

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "n_flagged": len(flagged_cers),
        "n_unflagged": len(unflagged_cers),
        "mean_cer_flagged": _mean(flagged_cers),
        "mean_cer_unflagged": _mean(unflagged_cers),
    }

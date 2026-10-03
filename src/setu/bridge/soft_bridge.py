"""The soft bridge -- the project's one novel contribution (CLAUDE.md:
"the only genuinely new part of the project"). Carries the recogniser's
full per-frame confidence distribution into the modernizer instead of
collapsing to a single argmax string first, so the modernizer sees what
the recogniser was unsure about rather than inheriting a premature,
irreversible commitment.

CTC gives one distribution per image FRAME (a vertical strip), not per
letter -- most frames are blank/repeats, and there is no clean "top-5 per
letter position" without collapsing frames first, which is itself the
hard decision this bridge exists to avoid. Everything here operates per
frame.

Per-frame procedure (CLAUDE.md): temperature dial -> softmax -> keep top-5
symbols (blank included, not special-cased) -> renormalise those 5 to sum
to 1 -> blend embeddings by confidence -> optionally pool every 2-4 frames
by averaging -> feed to the modernizer's encoder. Frames where blank
confidence exceeds ~0.95 may be dropped -- defensible as removing padding,
not choosing between competing letters (distinct from collapsing, and
worth being able to explain that distinction under questioning).

Temperature is NOT fixed at 1: CTC models are typically overconfident
(98-99% on the top choice), which would make the blend numerically
near-identical to argmax and erase the B3-vs-B4 difference. The actual
temperature value is tuned against real validation data (CLAUDE.md: Week
3, measure the fraction of frames with top-1 confidence below 0.9; raise
temperature if that fraction is tiny) -- that tuning needs a trained CRNN,
which doesn't exist yet, so `temperature` is a required argument here,
not a hardcoded default, and `measure_confidence_distribution` below is
the tool for doing that measurement once a checkpoint exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SoftBridgeConfig:
    temperature: float
    top_k: int = 5
    pool_size: int = 1  # 1 = no pooling; CLAUDE.md allows 2-4
    blank_index: int = 0
    blank_drop_threshold: float | None = 0.95  # None disables frame-dropping


def apply_soft_bridge(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    src_embed_table: torch.Tensor,
    config: SoftBridgeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """log_probs: (T, B, C) CTC log-softmax output from the CRNN (C = WX
    vocab + blank). input_lengths: (B,) real (unpadded) frame counts.
    src_embed_table: (C, H) the modernizer's src_embed.weight -- CRNN
    output classes and the modernizer's encoder vocabulary are both
    indexed over setu.data.wx.VOCAB + a shared blank/pad at index 0, so
    they line up directly with no remapping.

    Returns (blended_embeds: (T', B, H), key_padding_mask: (B, T') -- True
    = ignore) ready for Modernizer.encode_embeds(). T' == T if pool_size=1.
    """
    t, b, c = log_probs.shape
    device = log_probs.device

    # 1-3: temperature -> softmax -> top-k -> renormalise.
    # softmax(log_probs / T) == softmax(logits / T) for any T: log_probs is
    # logits shifted by a per-frame constant (log_probs = logits -
    # logsumexp(logits)), and softmax is invariant to any uniform
    # (same-for-every-class) additive shift -- so temperature scaling can
    # be applied directly to the model's log-softmax output without
    # needing raw pre-softmax logits.
    temp_probs = F.softmax(log_probs / config.temperature, dim=-1)  # (T, B, C)
    topk_probs, topk_idx = temp_probs.topk(config.top_k, dim=-1)  # (T, B, k)
    renorm_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # (T, B, k)

    # 4: blend embeddings by (renormalised) confidence.
    gathered = src_embed_table[topk_idx]  # (T, B, k, H)
    blended = (renorm_probs.unsqueeze(-1) * gathered).sum(dim=2)  # (T, B, H)

    # Padding mask: frames beyond each example's real length...
    frame_idx = torch.arange(t, device=device).unsqueeze(1)  # (T, 1)
    key_padding_mask = frame_idx >= input_lengths.unsqueeze(0)  # (T, B) -> transpose below

    # ...plus (not instead of) frames where blank dominates. This is
    # dropping by MASKING, not by removing/compacting positions: the
    # transformer's padding mask already has "ignore this position" built
    # in, so marking a blank-dominated frame as padding is both simpler
    # and more honest than deleting it -- nothing here chooses between
    # competing letters, it just stops attending to frames that are ~all
    # blank, which is what "removing padding" actually means.
    if config.blank_drop_threshold is not None:
        blank_conf = temp_probs[..., config.blank_index]  # (T, B)
        key_padding_mask = key_padding_mask | (blank_conf > config.blank_drop_threshold)

    key_padding_mask = key_padding_mask.transpose(0, 1)  # (B, T)

    # 5: optional pooling every pool_size frames (average) to shorten the
    # sequence. A pooled position is padding only if EVERY frame it
    # draws from is padding (if at least one real frame survives, pool it
    # in -- padded/dropped frames contribute their (irrelevant) embedding
    # value but don't skew the average meaningfully since blank's own
    # embedding is also a learned, meaningful vector, not zero).
    if config.pool_size > 1:
        pad_len = (-t) % config.pool_size
        if pad_len > 0:
            blended = F.pad(blended, (0, 0, 0, 0, 0, pad_len))  # pad T dim
            key_padding_mask = F.pad(key_padding_mask, (0, pad_len), value=True)
        t_pooled = (t + pad_len) // config.pool_size
        blended = blended.view(t_pooled, config.pool_size, b, blended.shape[-1]).mean(dim=1)
        key_padding_mask = (
            key_padding_mask.view(b, t_pooled, config.pool_size).all(dim=2)
        )

    return blended, key_padding_mask


def measure_confidence_distribution(log_probs: torch.Tensor, input_lengths: torch.Tensor) -> dict:
    """CLAUDE.md, Week 3: measure the fraction of (real, non-padding)
    frames with top-1 confidence below 0.9, to decide whether temperature
    needs raising (CTC models are typically overconfident -- if this
    fraction is tiny, the blend is numerically close to argmax and the
    B3-vs-B4 comparison has nothing to show). Run this over a validation
    batch once a trained CRNN checkpoint exists; there is nothing to
    measure before that."""
    t, b = log_probs.shape[0], log_probs.shape[1]
    device = log_probs.device
    frame_idx = torch.arange(t, device=device).unsqueeze(1)
    valid_mask = frame_idx < input_lengths.unsqueeze(0)  # (T, B)

    probs = log_probs.exp()
    top1_conf = probs.max(dim=-1).values  # (T, B)
    valid_conf = top1_conf[valid_mask]

    return {
        "n_frames": int(valid_mask.sum().item()),
        "mean_top1_confidence": float(valid_conf.mean().item()),
        "fraction_below_0.9": float((valid_conf < 0.9).float().mean().item()),
        "fraction_below_0.99": float((valid_conf < 0.99).float().mean().item()),
    }

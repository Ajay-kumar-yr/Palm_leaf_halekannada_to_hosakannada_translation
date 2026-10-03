"""Correctness checks for setu.bridge.soft_bridge -- plain assert-based,
no pytest (not in requirements.txt). Tested against synthetic CTC
log-probs since no trained CRNN checkpoint exists yet; real temperature
tuning needs real validation data (see soft_bridge.measure_confidence_distribution).

Run directly:
    python tests/test_soft_bridge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from setu.bridge.soft_bridge import SoftBridgeConfig, apply_soft_bridge, measure_confidence_distribution
from setu.data import wx
from setu.modernizer.model import Modernizer
from setu.modernizer.vocab import build_vocab


def _fake_log_probs(t: int, b: int, c: int, seed: int, blank_bias: float = 3.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(t, b, c, generator=g) * 0.5
    logits[..., 0] += blank_bias  # real CTC output is blank-heavy
    return torch.log_softmax(logits, dim=-1)


def _src_embed_table() -> torch.Tensor:
    vocab = build_vocab(["ಕನ್ನಡ ನಾಡು", "ಹಳೆಯ ಗ್ರಂಥ"])
    model = Modernizer(len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id)
    return model.src_embed.weight, model


def test_shapes_and_padding_respected() -> None:
    t, b, c, h = 20, 3, len(wx.VOCAB) + 1, 256
    log_probs = _fake_log_probs(t, b, c, seed=0)
    input_lengths = torch.tensor([20, 15, 10])
    src_embed_table, _ = _src_embed_table()

    cfg = SoftBridgeConfig(temperature=2.0, top_k=5, pool_size=1, blank_drop_threshold=0.95)
    blended, mask = apply_soft_bridge(log_probs, input_lengths, src_embed_table, cfg)
    assert blended.shape == (t, b, h), blended.shape
    assert mask.shape == (b, t), mask.shape
    assert mask[1, 15:].all(), "frames beyond input_length must be masked"
    assert not mask[1, :15].all(), "not all real frames should be masked"
    print("OK  shapes correct, out-of-length padding respected")


def test_pooling_shortens_sequence() -> None:
    t, b, c = 20, 3, len(wx.VOCAB) + 1
    log_probs = _fake_log_probs(t, b, c, seed=1)
    input_lengths = torch.tensor([20, 15, 10])
    src_embed_table, _ = _src_embed_table()

    cfg = SoftBridgeConfig(temperature=2.0, pool_size=4, blank_drop_threshold=0.95)
    blended, mask = apply_soft_bridge(log_probs, input_lengths, src_embed_table, cfg)
    expected_t = (t + 3) // 4
    assert blended.shape[0] == expected_t, blended.shape
    assert mask.shape == (b, expected_t), mask.shape
    print(f"OK  pool_size=4 shortens T={t} -> T'={expected_t}")


def test_feeds_modernizer_encoder_cleanly() -> None:
    t, b, c = 20, 3, len(wx.VOCAB) + 1
    log_probs = _fake_log_probs(t, b, c, seed=2)
    input_lengths = torch.tensor([20, 15, 10])
    src_embed_table, model = _src_embed_table()

    cfg = SoftBridgeConfig(temperature=2.0, blank_drop_threshold=0.95)
    blended, mask = apply_soft_bridge(log_probs, input_lengths, src_embed_table, cfg)
    memory = model.encode_embeds(blended, mask)
    assert memory.shape == (t, b, 256), memory.shape
    assert not torch.isnan(memory).any(), "NaN in encoder output"
    print("OK  blended embeddings feed Modernizer.encode_embeds() without error")


def test_low_temperature_converges_to_argmax_embedding() -> None:
    """The defining correctness property: at T -> 0, the confidence-
    weighted blend must collapse to exactly the argmax symbol's own
    embedding (B4 degenerates to B3's behaviour in the limit) -- if this
    doesn't hold, the blend formula itself is wrong, not just imprecise."""
    t, b, c = 20, 3, len(wx.VOCAB) + 1
    log_probs = _fake_log_probs(t, b, c, seed=3)
    input_lengths = torch.tensor([20, 15, 10])
    src_embed_table, _ = _src_embed_table()

    cfg = SoftBridgeConfig(temperature=0.01, top_k=5, blank_drop_threshold=None)
    blended, _ = apply_soft_bridge(log_probs, input_lengths, src_embed_table, cfg)
    argmax_embeds = src_embed_table[log_probs.argmax(dim=-1)]
    max_diff = (blended - argmax_embeds).abs().max().item()
    assert max_diff < 1e-3, f"low-temperature blend should match argmax embedding, max diff {max_diff}"
    print(f"OK  low-temperature blend converges to argmax embedding (max diff {max_diff:.2e})")


def test_blank_dominant_frames_are_dropped() -> None:
    t, c, h = 10, len(wx.VOCAB) + 1, 8
    g = torch.Generator().manual_seed(4)
    src_embed_table = torch.randn(c, h, generator=g)

    logits = torch.randn(t, 1, c, generator=g) * 0.3
    logits[0:5, 0, 0] += 10.0  # frames 0-4: blank totally dominates
    log_probs = torch.log_softmax(logits, dim=-1)
    input_lengths = torch.tensor([10])

    cfg = SoftBridgeConfig(temperature=1.0, blank_drop_threshold=0.95)
    _, mask = apply_soft_bridge(log_probs, input_lengths, src_embed_table, cfg)
    assert mask[0, :5].all(), "high-blank-confidence frames should be dropped (masked)"
    assert not mask[0, 5:].any(), "normal frames should not be dropped"

    cfg_nodrop = SoftBridgeConfig(temperature=1.0, blank_drop_threshold=None)
    _, mask2 = apply_soft_bridge(log_probs, input_lengths, src_embed_table, cfg_nodrop)
    assert not mask2[0].any(), "blank_drop_threshold=None must disable frame-dropping entirely"
    print("OK  blank-dominant frame dropping (and disabling it) both work correctly")


def test_measure_confidence_distribution() -> None:
    t, b, c = 20, 3, len(wx.VOCAB) + 1
    log_probs = _fake_log_probs(t, b, c, seed=5)
    input_lengths = torch.tensor([20, 15, 10])
    stats = measure_confidence_distribution(log_probs, input_lengths)
    assert stats["n_frames"] == int(input_lengths.sum().item())
    assert 0.0 <= stats["mean_top1_confidence"] <= 1.0
    assert 0.0 <= stats["fraction_below_0.9"] <= 1.0
    print(f"OK  confidence distribution measured: {stats}")


def main() -> None:
    test_shapes_and_padding_respected()
    test_pooling_shortens_sequence()
    test_feeds_modernizer_encoder_cleanly()
    test_low_temperature_converges_to_argmax_embedding()
    test_blank_dominant_frames_are_dropped()
    test_measure_confidence_distribution()
    print("\nAll soft bridge tests passed.")


if __name__ == "__main__":
    main()

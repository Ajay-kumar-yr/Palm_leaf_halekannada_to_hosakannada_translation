"""Correctness checks for setu.bridge.flagging -- plain assert-based, no
pytest. Run directly:
    python tests/test_flagging.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from setu.bridge.flagging import flag_lines, flagged_vs_unflagged_accuracy, mean_line_confidence
from setu.data import wx


def _four_line_batch():
    c, t = len(wx.VOCAB) + 1, 10
    logits = torch.zeros(t, 4, c)
    logits[:, 0, 0] = 20.0  # near-certain every frame
    logits[:, 1, :] = 0.01 * torch.randn(t, c, generator=torch.Generator().manual_seed(0))  # near-uniform
    logits[:, 2, 0] = 2.0
    logits[:, 3, 0] = 0.5
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs, torch.tensor([10, 10, 10, 10])


def test_confidence_ordering_matches_construction() -> None:
    log_probs, input_lengths = _four_line_batch()
    conf = mean_line_confidence(log_probs, input_lengths)
    assert conf[0] > conf[2] > conf[3] > conf[1], conf.tolist()
    print(f"OK  confidence ordering: {[round(c, 3) for c in conf.tolist()]}")


def test_flag_lines_respects_threshold() -> None:
    log_probs, input_lengths = _four_line_batch()
    flags = flag_lines(log_probs, input_lengths, threshold=0.5)
    assert not flags[0], "near-certain line should not be flagged"
    assert flags[1], "near-uniform line should be flagged"
    print(f"OK  flag_lines at threshold=0.5: {flags.tolist()}")


def test_flag_lines_respects_padding() -> None:
    """A line's confidence must only be computed over its real frames --
    padding frames (garbage/zero activations beyond input_length) must
    not pull the average down."""
    c, t = len(wx.VOCAB) + 1, 10
    logits = torch.zeros(t, 1, c)
    logits[:5, 0, 0] = 20.0  # confident real frames
    logits[5:, 0, :] = 0.0  # padding -- would read as low-confidence if wrongly included
    log_probs = torch.log_softmax(logits, dim=-1)
    conf_with_correct_length = mean_line_confidence(log_probs, torch.tensor([5]))
    conf_if_padding_included = mean_line_confidence(log_probs, torch.tensor([10]))
    assert conf_with_correct_length.item() > conf_if_padding_included.item()
    print("OK  padding frames excluded from confidence average")


def test_flagged_vs_unflagged_accuracy_grouping() -> None:
    is_flagged = torch.tensor([False, True, True, False])
    line_cers = [0.1, 0.5, 0.7, 0.2]
    result = flagged_vs_unflagged_accuracy(is_flagged, line_cers)
    assert result["n_flagged"] == 2
    assert result["n_unflagged"] == 2
    assert abs(result["mean_cer_flagged"] - 0.6) < 1e-9
    assert abs(result["mean_cer_unflagged"] - 0.15) < 1e-9
    print(f"OK  flagged-vs-unflagged grouping: {result}")


def test_flagged_vs_unflagged_handles_empty_group() -> None:
    is_flagged = torch.tensor([False, False])
    result = flagged_vs_unflagged_accuracy(is_flagged, [0.1, 0.2])
    assert result["mean_cer_flagged"] is None, "empty group should report None, not 0.0 or nan"
    print("OK  empty flagged group reports None rather than a misleading number")


def test_mismatched_lengths_raise() -> None:
    try:
        flagged_vs_unflagged_accuracy(torch.tensor([True, False]), [0.1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatched is_flagged/line_cers length")
    print("OK  mismatched-length inputs raise rather than silently misaligning")


def main() -> None:
    test_confidence_ordering_matches_construction()
    test_flag_lines_respects_threshold()
    test_flag_lines_respects_padding()
    test_flagged_vs_unflagged_accuracy_grouping()
    test_flagged_vs_unflagged_handles_empty_group()
    test_mismatched_lengths_raise()
    print("\nAll flagging tests passed.")


if __name__ == "__main__":
    main()

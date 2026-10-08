"""Checks for the demo-time per-character view of the soft bridge.

Hand-built distributions, so the expected answer is known exactly rather
than read off a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from setu.bridge.line_alternatives import (  # noqa: E402
    analyse_line,
    annotated_text,
    argmax_text,
    summarise,
    variant_readings,
)
from setu.data import wx  # noqa: E402

C = len(wx.VOCAB) + 1
BLANK = 0


def frame(dist: dict[int, float]) -> torch.Tensor:
    p = torch.full((C,), 1e-9)
    for i, v in dist.items():
        p[i] = v
    return p.log()


def idx(sym: str) -> int:
    return wx.SYMBOL_TO_INDEX[sym]


def test_repeats_collapse_and_blanks_drop() -> None:
    k, a = idx("k"), idx("a")
    lp = torch.stack([
        frame({BLANK: 0.9}),
        frame({k: 0.95}), frame({k: 0.9}),      # one character, two frames
        frame({BLANK: 0.8}),
        frame({a: 0.99}),
    ])
    symbols, slots = analyse_line(lp, input_length=5)
    assert symbols == ["k", "a"], symbols
    assert len(slots) == 2
    assert argmax_text(symbols) == wx.decode(["k", "a"])
    print(f"OK  repeats collapse, blanks dropped -> {argmax_text(symbols)!r}")


def test_peak_frame_is_used_not_the_run_edge() -> None:
    k = idx("k")
    lp = torch.stack([
        frame({k: 0.55, idx("K"): 0.45}),   # ragged edge
        frame({k: 0.99, idx("K"): 0.01}),   # peak -- this is the honest frame
        frame({k: 0.50, idx("K"): 0.50}),   # ragged edge
    ])
    _, slots = analyse_line(lp, input_length=3, uncertain_below=0.9)
    assert slots[0].frame == 1, slots[0].frame
    assert not slots[0].uncertain, "peak frame is confident, so the char is not flagged"
    print("OK  attribution uses the run's peak frame, not its edges")


def test_confusable_pair_is_flagged_with_both_readings() -> None:
    # The demo's best case: ದ vs ಧ, the recogniser genuinely undecided.
    # Full syllables (consonant + vowel), as real WX sequences are.
    p, a, x, X = idx("p"), idx("a"), idx("x"), idx("X")
    lp = torch.stack([
        frame({p: 0.99}), frame({a: 0.99}),
        frame({x: 0.52, X: 0.48}), frame({a: 0.99}),
    ])
    symbols, slots = analyse_line(lp, input_length=4, uncertain_below=0.9)
    assert symbols == ["p", "a", "x", "a"], symbols
    assert [s.uncertain for s in slots] == [False, False, True, False]
    assert slots[2].alternatives[0].symbol == "x"
    assert slots[2].alternatives[1].symbol == "X"

    readings = variant_readings(symbols, slots)
    texts = [t for t, _ in readings]
    assert "ಪದ" in texts, texts
    assert "ಪಧ" in texts, f"the rival whole-line reading must survive: {texts}"
    assert all("್" not in t for t in texts), f"no spurious viramas: {texts}"

    ann = annotated_text(symbols, slots)
    assert "ಪದ" in ann and "ಪಧ" in ann, ann
    print(f"OK  confusable pair -> readings {texts}")
    indented = ann.replace(chr(10), chr(10) + "      ")
    print("    prompt block:\n      " + indented)


def test_argmax_discards_what_the_bridge_keeps() -> None:
    x, X = idx("x"), idx("X")
    lp = torch.stack([frame({x: 0.52, X: 0.48}), frame({idx("a"): 0.99})])
    symbols, slots = analyse_line(lp, input_length=2, uncertain_below=0.9)
    s = summarise(slots)
    assert s["n_uncertain"] == 1
    # argmax keeps 0.52 and throws away the rest; the bridge carries it.
    assert abs(s["mean_discarded_weight_at_uncertain"] - 0.48) < 0.02, s
    assert s["n_chars"] == 2
    assert argmax_text(symbols) == wx.decode(["x", "a"]), "B3 sees one reading only"
    print(f"OK  argmax discards {s['mean_discarded_weight_at_uncertain']:.2f} at the flagged char")


def test_confident_line_is_not_annotated() -> None:
    lp = torch.stack([frame({idx("k"): 0.999}), frame({idx("a"): 0.995})])
    symbols, slots = analyse_line(lp, input_length=2, uncertain_below=0.9)
    ann = annotated_text(symbols, slots)
    assert ann == f"reading: {argmax_text(symbols)}", ann
    assert "undecided" not in ann
    assert summarise(slots)["n_uncertain"] == 0
    print("OK  a confident line reads identically through both branches")


def test_temperature_raises_flagging() -> None:
    lp = torch.stack([frame({idx("x"): 0.80, idx("X"): 0.20})])
    _, cold = analyse_line(lp, 1, temperature=1.0, uncertain_below=0.9)
    _, hot = analyse_line(lp, 1, temperature=2.0, uncertain_below=0.9)
    assert hot[0].top1_prob < cold[0].top1_prob, (cold[0].top1_prob, hot[0].top1_prob)
    print(f"OK  temperature softens top-1 {cold[0].top1_prob:.3f} -> {hot[0].top1_prob:.3f}")


def main() -> None:
    test_repeats_collapse_and_blanks_drop()
    test_peak_frame_is_used_not_the_run_edge()
    test_confusable_pair_is_flagged_with_both_readings()
    test_argmax_discards_what_the_bridge_keeps()
    test_confident_line_is_not_annotated()
    test_temperature_raises_flagging()
    print("\nAll line-alternatives tests passed.")


if __name__ == "__main__":
    main()

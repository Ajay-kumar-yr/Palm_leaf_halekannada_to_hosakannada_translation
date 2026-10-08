"""Turn a line's per-frame CTC distributions into per-character
alternatives — the form the demo's modernizer can consume (DEMO_PLAN.md).

This is a **demo-time adaptation of the soft bridge, not the bridge
itself**, and the report must say so. The measured claim stays the
frame-level one (`bridge/recovery_examples.py`: on frozen-test frames
where top-1 was wrong, the correct symbol was still in the top-5 80.6%
of the time, carried with mean weight 0.200 against argmax's 0.000).
What this module adds is a way to *show* that to a reader and to hand it
to an LLM modernizer, which takes text rather than embeddings.

Steps 1-3 are the bridge's own, deliberately reusing its arithmetic
rather than re-deriving it: temperature → softmax → top-k → renormalise.
The new part is attribution — mapping surviving frames back to the
output characters they produced, so an alternative can be named as "at
this character" instead of "at frame 273".

Attribution rule: CTC greedy decoding collapses a run of repeated frames
into one character. For each collapsed character this takes its **peak
frame** (the frame in that run where the chosen symbol is most
confident) and reads the alternatives there. The peak frame is the one
place the recogniser was most committed, so alternatives that survive
*there* are the honest ones; sampling a run's ragged edges would
overstate uncertainty.

A character is flagged when its renormalised top-1 probability falls
below `uncertain_below`. Flagged characters are exactly where argmax
throws information away.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from setu.data import wx


@dataclass(frozen=True)
class Alternative:
    symbol: str          # WX symbol
    prob: float          # renormalised over the kept top-k
    kannada: str         # how this symbol alone renders (may carry a virama)


@dataclass(frozen=True)
class CharSlot:
    index: int                      # position in the decoded symbol sequence
    frame: int                      # peak frame this character was read at
    chosen: str                     # WX symbol argmax committed to
    alternatives: list[Alternative]  # includes `chosen`, highest first
    uncertain: bool

    @property
    def top1_prob(self) -> float:
        return self.alternatives[0].prob if self.alternatives else 1.0


def _decode_single(symbol: str) -> str:
    try:
        return wx.decode([symbol])
    except Exception:
        return symbol


def analyse_line(
    log_probs: torch.Tensor,
    input_length: int,
    temperature: float = 1.0,
    top_k: int = 5,
    uncertain_below: float = 0.9,
    blank_index: int = 0,
) -> tuple[list[str], list[CharSlot]]:
    """log_probs: (T, C) or (T, 1, C) for ONE line.

    Returns (decoded WX symbols, one CharSlot per decoded symbol).
    """
    if log_probs.dim() == 3:
        if log_probs.shape[1] != 1:
            raise ValueError(f"one line at a time; got batch {log_probs.shape[1]}")
        log_probs = log_probs[:, 0]
    lp = log_probs[:input_length]

    # Bridge steps 1-3, identical to soft_bridge.apply_soft_bridge.
    probs = F.softmax(lp / temperature, dim=-1)              # (T, C)
    topk_probs, topk_idx = probs.topk(top_k, dim=-1)         # (T, k)
    renorm = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    best = lp.argmax(dim=-1)                                  # (T,) greedy path
    symbols: list[str] = []
    slots: list[CharSlot] = []

    prev = None
    run_frames: list[int] = []

    def close_run(cls: int, frames: list[int]) -> None:
        if cls == blank_index or not frames:
            return
        # Peak frame: where this symbol was most confident in its run.
        peak = max(frames, key=lambda t: float(probs[t, cls]))
        alts = [
            Alternative(symbol=wx.INDEX_TO_SYMBOL[int(i)] if int(i) != blank_index else "<blank>",
                        prob=float(p),
                        kannada="" if int(i) == blank_index else _decode_single(wx.INDEX_TO_SYMBOL[int(i)]))
            for p, i in zip(renorm[peak].tolist(), topk_idx[peak].tolist())
        ]
        chosen = wx.INDEX_TO_SYMBOL[cls]
        symbols.append(chosen)
        slots.append(CharSlot(
            index=len(symbols) - 1, frame=peak, chosen=chosen, alternatives=alts,
            uncertain=alts[0].prob < uncertain_below if alts else False,
        ))

    for t, cls in enumerate(best.tolist()):
        if cls == prev:
            run_frames.append(t)
            continue
        close_run(prev, run_frames) if prev is not None else None
        prev, run_frames = cls, [t]
    if prev is not None:
        close_run(prev, run_frames)

    return symbols, slots


def argmax_text(symbols: list[str]) -> str:
    """What B3 sends onward: one string, uncertainty discarded."""
    try:
        return wx.decode(symbols)
    except Exception:
        return "".join(symbols)


def annotated_text(symbols: list[str], slots: list[CharSlot], max_alternatives: int = 3,
                   min_prob: float = 0.02) -> str:
    """What B4 sends onward: the committed reading, plus the rival whole-line
    readings the recogniser was still holding.

    Alternatives are given as **whole lines**, not inline brackets.
    Kannada is an abugida: a consonant's rendered form depends on the
    vowel that follows it, so splicing a single symbol's decoding into a
    string produces spurious viramas (ಪ್ದ್ where the line reads ಪದ). Only
    decoding a complete symbol sequence is correct, which is also the
    form a reader — or an LLM — can actually judge.
    """
    lines = [f"reading: {argmax_text(symbols)}"]
    rivals = [(t, p) for t, p in variant_readings(symbols, slots, max_variants=max_alternatives + 1)[1:]
              if p >= min_prob]
    if rivals:
        lines.append("the recogniser was undecided here; it also kept:")
        lines += [f"  {t}  ({p:.2f})" for t, p in rivals]
    return "\n".join(lines)


def variant_readings(symbols: list[str], slots: list[CharSlot], max_variants: int = 8) -> list[tuple[str, float]]:
    """Whole-line readings the bridge still keeps alive: the argmax
    reading, then one per (uncertain character, alternative) swap, most
    probable first.

    Whole lines rather than isolated characters because a Kannada reader
    judges a word, not a glyph — and because this is what makes a demo
    example legible: "argmax reads X; the bridge also held Y, which
    reads as a real word".
    """
    readings: list[tuple[str, float]] = [(argmax_text(symbols), 1.0)]
    swaps: list[tuple[float, int, str]] = []
    for slot in slots:
        if not slot.uncertain:
            continue
        for alt in slot.alternatives:
            if alt.symbol in ("<blank>", slot.chosen) or alt.prob < 1e-3:
                continue
            swaps.append((alt.prob, slot.index, alt.symbol))
    swaps.sort(reverse=True)
    for prob, idx, sym in swaps[: max_variants - 1]:
        variant = list(symbols)
        variant[idx] = sym
        readings.append((argmax_text(variant), prob))
    return readings


def summarise(slots: list[CharSlot]) -> dict:
    n_unc = sum(1 for s in slots if s.uncertain)
    return {
        "n_chars": len(slots),
        "n_uncertain": n_unc,
        "fraction_uncertain": n_unc / len(slots) if slots else 0.0,
        "mean_top1": sum(s.top1_prob for s in slots) / len(slots) if slots else None,
        # Weight argmax discards: everything the bridge keeps beyond its
        # single choice, at exactly the characters where that matters.
        "mean_discarded_weight_at_uncertain": (
            sum(1.0 - s.top1_prob for s in slots if s.uncertain) / n_unc if n_unc else 0.0
        ),
    }

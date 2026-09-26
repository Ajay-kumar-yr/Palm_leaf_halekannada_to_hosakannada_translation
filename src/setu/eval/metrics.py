"""The single source of truth for evaluation metrics (CLAUDE.md rule 5:
metrics are computed ONLY here -- never inline, never invented elsewhere).

Only edit_distance/CER are implemented so far, needed by the recogniser's
8-example memorisation sanity check (CLAUDE.md rule 3). chrF++, BLEU, and
POS accuracy are added in Week 3/4 when B0/B3/B4 are actually compared.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def edit_distance(ref: Sequence[T], hyp: Sequence[T]) -> int:
    """Levenshtein distance between two token sequences (chars, or WX symbols)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def cer(ref: Sequence[T], hyp: Sequence[T]) -> float:
    """Character (or symbol) error rate: edit_distance / len(ref).

    An empty `ref` is defined as CER 0.0 if `hyp` is also empty, else 1.0
    per non-empty hyp token -- there is no well-defined ratio to report
    otherwise, and silently returning 0.0/0.0 = nan would be worse.
    """
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else float(len(hyp))
    return edit_distance(ref, hyp) / len(ref)

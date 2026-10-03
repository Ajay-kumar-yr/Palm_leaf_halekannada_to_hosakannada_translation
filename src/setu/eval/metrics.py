"""The single source of truth for evaluation metrics (CLAUDE.md rule 5:
metrics are computed ONLY here -- never inline, never invented elsewhere).

edit_distance/CER (needed by the recogniser's 8-example memorisation
sanity check, CLAUDE.md rule 3) plus chrF++ and BLEU for the B0/B3/B4
modernization comparison. POS accuracy is added separately in Week 4
(hand-checked tagging, not a metric computed over model output the same
way).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import sacrebleu

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


@dataclass(frozen=True)
class CorpusScore:
    score: float
    detail: str  # the library's own human-readable breakdown, for the report/appendix


def chrf_plus_plus(hyps: list[str], refs: list[str]) -> CorpusScore:
    """chrF++ (CLAUDE.md: the PRIMARY modernization metric) -- character
    n-gram F-score plus word unigram/bigram F-score (word_order=2 is the
    standard definition of "chrF++" as opposed to plain chrF). Corpus-level
    (one score over the whole test split via sacrebleu's corpus_score, not
    an average of per-line scores -- the standard way to report this, and
    not equivalent to averaging individual sentence scores).

    hyps/refs are plain strings (whole lines), not pre-tokenized sequences
    -- chrF operates on characters directly and sacrebleu handles its own
    internal word-boundary splitting for the word-order component.
    """
    if len(hyps) != len(refs):
        raise ValueError(f"hyps and refs must be the same length, got {len(hyps)} and {len(refs)}")
    result = sacrebleu.CHRF(word_order=2).corpus_score(hyps, [refs])
    return CorpusScore(score=result.score, detail=str(result))


def bleu(hyps: list[str], refs: list[str]) -> CorpusScore:
    """BLEU (CLAUDE.md: SECONDARY -- "BLEU alone is misleading here: plain
    copying already scores well given vocabulary overlap, and the
    patent's reported 0.81 should be read with that in mind"). Corpus-
    level via sacrebleu with its standard tokenizer and smoothing
    defaults, for comparability with how BLEU is normally reported.

    Note on short lines: BLEU's standard 4-gram geometric mean can read 0
    for very short sentences (a line under 4 words has no 4-grams to
    match at all, independent of translation quality) -- expected
    behaviour of the metric, not a bug here, and part of why CLAUDE.md
    treats chrF++ as primary rather than fixing this with a different
    smoothing method unasked.
    """
    if len(hyps) != len(refs):
        raise ValueError(f"hyps and refs must be the same length, got {len(hyps)} and {len(refs)}")
    result = sacrebleu.BLEU().corpus_score(hyps, [refs])
    return CorpusScore(score=result.score, detail=str(result))

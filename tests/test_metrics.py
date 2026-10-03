"""Correctness checks for setu.eval.metrics -- plain assert-based, no
pytest (not in requirements.txt). Run directly:
    python tests/test_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from setu.eval.metrics import bleu, cer, chrf_plus_plus, edit_distance


def test_edit_distance_and_cer_basics() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "abc") == 0
    assert cer("abc", "abc") == 0.0
    assert cer("abc", "abd") == 1 / 3
    assert cer("", "") == 0.0
    assert cer("", "xyz") == 3.0
    print("OK  edit_distance / cer basics")


def test_chrf_identical_corpus_is_100() -> None:
    refs = ["ಕನ್ನಡ ನಾಡು ಚೆಂದ", "ಹಳೆಯ ಗ್ರಂಥ ಓದುವುದು"]
    result = chrf_plus_plus(refs, refs)
    assert abs(result.score - 100.0) < 1e-6, result.score
    print(f"OK  chrF++ identical corpus = {result.score}")


def test_chrf_distinguishes_similarity() -> None:
    refs = ["ಕನ್ನಡ ನಾಡು ಚೆಂದ"]
    close_hyp = ["ಕನ್ನಡ ನಾಡು ಚಂದ"]  # one character off
    far_hyp = ["ಬೇರೆಯೇ ಪದಗಳು ಇಲ್ಲಿವೆ"]  # unrelated
    close_score = chrf_plus_plus(close_hyp, refs).score
    far_score = chrf_plus_plus(far_hyp, refs).score
    assert close_score > far_score, (close_score, far_score)
    assert 0.0 <= far_score <= close_score <= 100.0
    print(f"OK  chrF++ close={close_score:.1f} > far={far_score:.1f}")


def test_bleu_identical_corpus_is_100() -> None:
    # multiple lines, long enough to have 4-grams -- avoids BLEU's
    # documented short-sentence degeneracy (see metrics.py docstring)
    refs = [
        "ಕನ್ನಡ ನಾಡು ಚೆಂದ ಇರುವ ನಾಡು ಎಂದು ಎಲ್ಲರೂ ಹೇಳುತ್ತಾರೆ",
        "ಹಳೆಯ ಗ್ರಂಥಗಳನ್ನು ಓದುವುದು ಬಹಳ ಸಂತೋಷದ ವಿಷಯವಾಗಿದೆ",
    ]
    result = bleu(refs, refs)
    assert abs(result.score - 100.0) < 1e-6, result.score
    print(f"OK  BLEU identical corpus = {result.score}")


def test_bleu_short_sentence_degeneracy_is_expected() -> None:
    """Document the quirk rather than hide it: a line under 4 words has
    no 4-grams at all, so standard BLEU reads 0 regardless of quality --
    this is why CLAUDE.md treats BLEU as secondary, not a bug to patch."""
    refs = ["ಕನ್ನಡ ನಾಡು"]
    result = bleu(refs, refs)
    assert result.score == 0.0, f"expected the documented degeneracy, got {result.score}"
    print("OK  short-sentence BLEU degeneracy confirmed (expected, documented)")


def test_mismatched_lengths_raise() -> None:
    try:
        chrf_plus_plus(["a"], ["a", "b"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatched hyps/refs length")
    print("OK  mismatched-length inputs raise rather than silently misalign")


def main() -> None:
    test_edit_distance_and_cer_basics()
    test_chrf_identical_corpus_is_100()
    test_chrf_distinguishes_similarity()
    test_bleu_identical_corpus_is_100()
    test_bleu_short_sentence_degeneracy_is_expected()
    test_mismatched_lengths_raise()
    print("\nAll metrics tests passed.")


if __name__ == "__main__":
    main()

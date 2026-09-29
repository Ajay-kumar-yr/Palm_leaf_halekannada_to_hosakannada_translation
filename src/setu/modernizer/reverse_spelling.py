"""Reverse-spelling rule engine: turns real modern-Kannada text into
synthetic old-Kannada-*styled* text, for S3 (roadmap v4 data table: "S3 |
old/modern text pairs, no images, rule-generated | 100-300k | Pre-training
the modernizer"). S3 pairs are pretraining data ONLY -- CLAUDE.md rule 9
restricts *real scholarly* modern text to S2's reported end-to-end numbers,
never our own rule engine's output; nothing here is ever scored against.

Three rules, all grounded in documented old-vs-modern Kannada orthographic
conventions (not invented):

1. Anusvara-to-homorganic-nasal (deterministic). Modern printed Kannada
   freely uses ಂ (anusvara) before any stop consonant; older orthography
   (and formal/Sanskritic spelling generally) more often spells out the
   homorganic nasal of that consonant's varga instead -- ಂ+ಕ -> ಙ್+ಕ,
   ಂ+ಚ -> ಞ್+ಚ, ಂ+ಟ -> ಣ್+ಟ, ಂ+ತ -> ನ್+ತ, ಂ+ಪ -> ಮ್+ಪ. This is a real,
   productive Sanskrit/Kannada sandhi convention, not a guess, and is
   applied deterministically (always reversible in principle) rather than
   probabilistically.

2. ರ -> ಱ (probabilistic, intervocalic only). ಱ (archaic trill r) merged
   into ರ when modern Kannada orthography stopped distinguishing them --
   ಱ never occurs word-initially or word-finally in genuine old-Kannada
   text, so the rule only fires on an intervocalic ರ (preceded and
   followed by a vowel-bearing syllable), matching that distributional
   constraint even though it cannot know which specific words historically
   had ಱ without a lexicon.

3. ಳ -> ೞ (probabilistic, intervocalic only). Same story for ೞ (archaic
   zha), which merged into ಳ. Same word-medial-only constraint.

Rules 2-3 are approximate by necessity (a real old-Kannada word either had
ಱ/ೞ or it didn't -- there is no rule that recovers this from modern
spelling alone), which is exactly why this is pretraining data, never a
reported number: it teaches the modernizer's encoder roughly what
old-looking spelling variance looks like before fine-tuning on the much
smaller, real S2 pairs and the recogniser's actual noisy output. It is
never held up as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ANUSVARA_TO_NASAL: dict[str, str] = {}
for _nasal, _stops in [
    ("ಙ", "ಕಖಗಘ"),
    ("ಞ", "ಚಛಜಝ"),
    ("ಣ", "ಟಠಡಢ"),
    ("ನ", "ತಥದಧ"),
    ("ಮ", "ಪಫಬಭ"),
]:
    for _stop in _stops:
        _ANUSVARA_TO_NASAL[_stop] = _nasal

VIRAMA = "್"
_VOWEL_BEARING_MATRAS = set("ಾಿೀುೂೃೆೇೈೊೋೌ")


def _is_vowel_bearing(ch: str) -> bool:
    """True for an independent vowel, or a consonant not immediately
    followed by virama (i.e. it carries some vowel, inherent 'a' or a
    matra) -- used to test the intervocalic constraint for rules 2-3."""
    return ("ಅ" <= ch <= "ಔ") or ch in _VOWEL_BEARING_MATRAS


def apply_anusvara_rule(text: str) -> str:
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "ಂ" and nxt in _ANUSVARA_TO_NASAL:
            out.append(_ANUSVARA_TO_NASAL[nxt])
            out.append(VIRAMA)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def apply_probabilistic_rule(text: str, target: str, replacement: str, rate: float, rng: np.random.Generator) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch != target:
            continue
        prev_ch = chars[i - 1] if i > 0 else ""
        next_ch = chars[i + 1] if i + 1 < len(chars) else ""
        intervocalic = _is_vowel_bearing(prev_ch) and _is_vowel_bearing(next_ch)
        if intervocalic and rng.random() < rate:
            chars[i] = replacement
    return "".join(chars)


@dataclass(frozen=True)
class ReverseSpellingConfig:
    ra_to_rYa_rate: float = 0.15
    la_to_zYa_rate: float = 0.15
    seed: int = 0


def old_style_text(text: str, cfg: ReverseSpellingConfig, rng: np.random.Generator) -> str:
    text = apply_anusvara_rule(text)
    text = apply_probabilistic_rule(text, "ರ", "ಱ", cfg.ra_to_rYa_rate, rng)
    text = apply_probabilistic_rule(text, "ಳ", "ೞ", cfg.la_to_zYa_rate, rng)
    return text

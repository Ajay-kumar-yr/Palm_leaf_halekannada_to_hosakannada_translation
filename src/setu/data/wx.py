"""Kannada Unicode <-> WX-style romanization, exact round-trip.

Per CLAUDE.md: the recogniser's output alphabet is WX, not Kannada script
directly, because Kannada has 600-900 distinct rendered syllables while WX
needs only ~50-60 symbols -- important because with only 25,000 training
lines, a syllable-level vocabulary would starve rare/archaic letters
(notably ಱ and ೞ) of training examples. WX must also convert back to
Kannada script exactly.

This is a phonemic scheme in the traditional WX-notation style used across
Indian-language NLP (the same k/K/g/G/f, c/C/j/J/F, t/T/d/D/N, w/W/x/X/n
consonant-varga convention), extended with two Kannada-only consonants
(ಱ, ೞ) that Devanagari-based WX tables don't need. Kannada's dependent
vowel signs (matras) are NOT given separate symbols from their independent
vowel counterparts -- a matra just means "attach this vowel to the
preceding consonant instead of writing it standalone" -- which is what
keeps the vocabulary near the ~50-60 target instead of the ~70-90 a
glyph-per-codepoint encoding would need.

The public representation of a WX-encoded line is `list[str]`, one entry
per symbol (this is what the CRNN vocabulary and CTC labels actually are);
`to_string`/`from_string` give a "|"-joined form for manifests/debugging
only. Symbols are never concatenated into a bare string for parsing, so
there is no prefix-collision risk between symbols of different lengths.

encode() raises on any character it has no mapping for (space, the
Kannada block, and the danda/double-danda punctuation shared with
Devanagari are covered) rather than silently dropping or mis-encoding it
-- CLAUDE.md rule 7, fail loudly.
"""

from __future__ import annotations

# --- Consonants -------------------------------------------------------
# codepoint -> WX symbol. Five-row varga order, then approximants/
# sibilants, then the two Kannada-only archaic consonants.
CONSONANTS: dict[str, str] = {
    "ಕ": "k", "ಖ": "K", "ಗ": "g", "ಘ": "G", "ಙ": "f",
    "ಚ": "c", "ಛ": "C", "ಜ": "j", "ಝ": "J", "ಞ": "F",
    "ಟ": "t", "ಠ": "T", "ಡ": "d", "ಢ": "D", "ಣ": "N",
    "ತ": "w", "ಥ": "W", "ದ": "x", "ಧ": "X", "ನ": "n",
    "ಪ": "p", "ಫ": "P", "ಬ": "b", "ಭ": "B", "ಮ": "m",
    "ಯ": "y", "ರ": "r", "ಲ": "l", "ವ": "v",
    "ಶ": "S", "ಷ": "R", "ಸ": "s", "ಹ": "h",
    "ಳ": "lY",       # ಳ retroflex l
    "ಱ": "rY",       # ಱ archaic trill r -- old-Kannada marker
    "ೞ": "zY",       # ೞ archaic zha -- old-Kannada marker
}

# --- Vowels -------------------------------------------------------------
# One WX symbol per vowel quality, shared between the independent-vowel
# codepoint and its dependent matra codepoint (the phonemic collapse).
_VOWEL_SYMBOLS = {
    "a": "a", "aa": "A", "i": "i", "ii": "I", "u": "u", "uu": "U",
    "vr": "q", "e": "e", "ee": "E", "ai": "Y", "o": "o", "oo": "O", "au": "V",
}

INDEP_VOWELS: dict[str, str] = {
    "ಅ": _VOWEL_SYMBOLS["a"], "ಆ": _VOWEL_SYMBOLS["aa"],
    "ಇ": _VOWEL_SYMBOLS["i"], "ಈ": _VOWEL_SYMBOLS["ii"],
    "ಉ": _VOWEL_SYMBOLS["u"], "ಊ": _VOWEL_SYMBOLS["uu"],
    "ಋ": _VOWEL_SYMBOLS["vr"],
    "ಎ": _VOWEL_SYMBOLS["e"], "ಏ": _VOWEL_SYMBOLS["ee"],
    "ಐ": _VOWEL_SYMBOLS["ai"],
    "ಒ": _VOWEL_SYMBOLS["o"], "ಓ": _VOWEL_SYMBOLS["oo"],
    "ಔ": _VOWEL_SYMBOLS["au"],
}

MATRAS: dict[str, str] = {
    "ಾ": _VOWEL_SYMBOLS["aa"], "ಿ": _VOWEL_SYMBOLS["i"],
    "ೀ": _VOWEL_SYMBOLS["ii"], "ು": _VOWEL_SYMBOLS["u"],
    "ೂ": _VOWEL_SYMBOLS["uu"], "ೃ": _VOWEL_SYMBOLS["vr"],
    "ೆ": _VOWEL_SYMBOLS["e"], "ೇ": _VOWEL_SYMBOLS["ee"],
    "ೈ": _VOWEL_SYMBOLS["ai"],
    "ೊ": _VOWEL_SYMBOLS["o"], "ೋ": _VOWEL_SYMBOLS["oo"],
    "ೌ": _VOWEL_SYMBOLS["au"],
}

VIRAMA = "್"

# --- Other signs and punctuation ----------------------------------------
OTHER_SIGNS: dict[str, str] = {
    "ಂ": "M",   # anusvara
    "ಃ": "H",   # visarga
    "ಁ": "z",   # candrabindu (rare)
    "ಽ": "Z",   # avagraha
}

PUNCTUATION: dict[str, str] = {
    " ": "sp",
    "।": "danda",    # ।  verse-line marker, common in old-Kannada poetry
    "॥": "ddanda",   # ॥  double danda
}

# --- Derived tables -------------------------------------------------------
_ENCODE_DIRECT: dict[str, str] = {**CONSONANTS, **INDEP_VOWELS, **OTHER_SIGNS, **PUNCTUATION}
_CONSONANT_CODEPOINTS = set(CONSONANTS)
_VOWEL_SYMBOL_TO_MATRA_CODEPOINT = {sym: cp for cp, sym in MATRAS.items()}
_VOWEL_SYMBOL_TO_INDEP_CODEPOINT = {sym: cp for cp, sym in INDEP_VOWELS.items()}
_SYMBOL_TO_CONSONANT_CODEPOINT = {sym: cp for cp, sym in CONSONANTS.items()}
_SYMBOL_TO_OTHER_CODEPOINT = {sym: cp for cp, sym in OTHER_SIGNS.items()}
_SYMBOL_TO_PUNCT_CODEPOINT = {sym: cp for cp, sym in PUNCTUATION.items()}

VOWEL_SYMBOLS = set(_VOWEL_SYMBOLS.values())

BLANK = "<blank>"
VOCAB: list[str] = sorted(
    set(CONSONANTS.values())
    | VOWEL_SYMBOLS
    | set(OTHER_SIGNS.values())
    | set(PUNCTUATION.values())
)
SYMBOL_TO_INDEX = {sym: i + 1 for i, sym in enumerate(VOCAB)}  # 0 reserved for blank
INDEX_TO_SYMBOL = {i: sym for sym, i in SYMBOL_TO_INDEX.items()}
INDEX_TO_SYMBOL[0] = BLANK

# --- Kannada digits: an OPT-IN extension, not part of VOCAB ---------------
#
# Real palm-leaf pages carry numerals -- verse numbers, dates, counts. They
# are only 1.9% of characters but appear in 36% of lines (57% on some
# pages, 100% on one), so a recogniser that cannot emit them cannot be
# trained on those lines at all.
#
# They are kept OUT of VOCAB deliberately. VOCAB's 56 symbols (+ blank =
# 57 CTC classes) are what the reported synthetic results were measured
# with -- the 1.53% frozen-test CER, the cached S2 distributions, the
# trained modernizer's input embeddings. Growing VOCAB would invalidate
# every one of those checkpoints. The real-handwriting fine-tune is a
# separate model for a separate purpose, so it (and only it) opts in via
# EXTENDED_VOCAB. Approved by the user 2026-10-08 (CLAUDE.md: "ask before
# ... changing the vocabulary").
#
# Symbols are prefixed "#" so they cannot collide with any WX symbol, and
# the digits are appended AFTER the base vocabulary so every existing
# class index keeps its meaning: a 57-class checkpoint loads into a
# 67-class model by copying the first 57 output rows (see
# setu.recogniser.model.expand_classifier).
DIGITS: dict[str, str] = {chr(0x0CE6 + i): f"#{i}" for i in range(10)}  # ೦..೯
_SYMBOL_TO_DIGIT = {sym: cp for cp, sym in DIGITS.items()}
DIGIT_SYMBOLS: list[str] = [DIGITS[chr(0x0CE6 + i)] for i in range(10)]

EXTENDED_VOCAB: list[str] = VOCAB + DIGIT_SYMBOLS
EXTENDED_SYMBOL_TO_INDEX = {sym: i + 1 for i, sym in enumerate(EXTENDED_VOCAB)}
EXTENDED_INDEX_TO_SYMBOL = {i: sym for sym, i in EXTENDED_SYMBOL_TO_INDEX.items()}
EXTENDED_INDEX_TO_SYMBOL[0] = BLANK

assert all(EXTENDED_SYMBOL_TO_INDEX[s] == SYMBOL_TO_INDEX[s] for s in VOCAB), (
    "extending the vocabulary must not renumber existing symbols -- a checkpoint "
    "trained on VOCAB would silently mean something different"
)


def tables(digits: bool) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """(vocab, symbol->index, index->symbol) for the chosen vocabulary."""
    if digits:
        return EXTENDED_VOCAB, EXTENDED_SYMBOL_TO_INDEX, EXTENDED_INDEX_TO_SYMBOL
    return VOCAB, SYMBOL_TO_INDEX, INDEX_TO_SYMBOL


def encode(text: str, digits: bool = False) -> list[str]:
    """Convert Kannada-script text to a list of WX symbols.

    Raises ValueError on any character without a mapping, rather than
    dropping or mis-encoding it silently.

    `digits=True` also accepts Kannada numerals (೦-೯), which are an
    opt-in extension used only by the real-handwriting fine-tune -- see
    EXTENDED_VOCAB. With the default False they raise, exactly as before,
    so every existing caller keeps the vocabulary its checkpoint expects.
    """
    symbols: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if ch in _CONSONANT_CODEPOINTS:
            symbols.append(CONSONANTS[ch])
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                i += 2  # bare consonant, no vowel -- consume the virama
                continue
            if nxt in MATRAS:
                symbols.append(MATRAS[nxt])
                i += 2
                continue
            symbols.append(_VOWEL_SYMBOLS["a"])  # inherent vowel
            i += 1
            continue

        if digits and ch in DIGITS:
            symbols.append(DIGITS[ch])
            i += 1
            continue

        if ch in INDEP_VOWELS:
            symbols.append(INDEP_VOWELS[ch])
            i += 1
            continue

        if ch in OTHER_SIGNS:
            symbols.append(OTHER_SIGNS[ch])
            i += 1
            continue

        if ch in PUNCTUATION:
            symbols.append(PUNCTUATION[ch])
            i += 1
            continue

        if ch == VIRAMA or ch in MATRAS:
            raise ValueError(
                f"Unexpected matra/virama {ch!r} (U+{ord(ch):04X}) not "
                f"preceded by a consonant at position {i} in {text!r}"
            )

        raise ValueError(
            f"No WX mapping for {ch!r} (U+{ord(ch):04X}) at position {i} "
            f"in {text!r} -- extend setu.data.wx if this is legitimate text"
        )
    return symbols


def decode(symbols: list[str]) -> str:
    """Convert a list of WX symbols back to exact Kannada-script text.

    Digit symbols (#0-#9) are always accepted: a model that can emit them
    must be able to have its output read back."""
    out: list[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        sym = symbols[i]

        if sym in _SYMBOL_TO_CONSONANT_CODEPOINT:
            out.append(_SYMBOL_TO_CONSONANT_CODEPOINT[sym])
            nxt = symbols[i + 1] if i + 1 < n else None
            if nxt == _VOWEL_SYMBOLS["a"]:
                i += 2  # bare consonant codepoint already carries inherent a
                continue
            if nxt in VOWEL_SYMBOLS:
                out.append(_VOWEL_SYMBOL_TO_MATRA_CODEPOINT[nxt])
                i += 2
                continue
            out.append(VIRAMA)  # next thing (or end) is not a vowel -> bare consonant
            i += 1
            continue

        if sym in VOWEL_SYMBOLS:
            out.append(_VOWEL_SYMBOL_TO_INDEP_CODEPOINT[sym])
            i += 1
            continue

        if sym in _SYMBOL_TO_OTHER_CODEPOINT:
            out.append(_SYMBOL_TO_OTHER_CODEPOINT[sym])
            i += 1
            continue

        if sym in _SYMBOL_TO_PUNCT_CODEPOINT:
            out.append(_SYMBOL_TO_PUNCT_CODEPOINT[sym])
            i += 1
            continue

        if sym in _SYMBOL_TO_DIGIT:
            out.append(_SYMBOL_TO_DIGIT[sym])
            i += 1
            continue

        raise ValueError(f"Unknown WX symbol {sym!r} at position {i}")

    return "".join(out)


def to_string(symbols: list[str]) -> str:
    return "|".join(symbols)


def from_string(s: str) -> list[str]:
    return s.split("|") if s else []

"""Phonetic romanisation -> Kannada script, for typing on a QWERTY keyboard.

Not the same thing as `setu.data.wx`. WX is the recogniser's internal
alphabet: terse, case-loaded (`k`/`K`, `x`/`X`), chosen for a small CTC
class count. Nobody can type it fluently. This is an ITRANS-style scheme
a Kannada speaker can type by ear -- `kannaDa` -> ಕನ್ನಡ -- for hand
transcribing manuscript lines.

Kept here, with a test, so the browser transliterator is not the only
copy of the table: `make_transcriber.py` emits this same mapping into
the page, and `test_translit.py` checks the two agree.

Conventions, following ITRANS as closely as Kannada allows:
  vowels      a aa/A i ii/I u uu/U R e E/ee ai o O/oo au
  consonants  k kh g gh ~N | ch chh j jh ~n | T Th D Dh N
              t th d dh n | p ph b bh m | y r l v/w sh Sh s h L
  archaic     rY (ಱ), lY (ೞ)  -- the letters that mark a text as old
  marks       M anusvara, H visarga, q explicit virama
  also        | -> danda, || -> double danda, 0-9 -> Kannada digits
"""

from __future__ import annotations

VIRAMA = "್"

INDEP_VOWELS = {
    "a": "ಅ", "aa": "ಆ", "A": "ಆ", "i": "ಇ", "ii": "ಈ",
    "I": "ಈ", "u": "ಉ", "uu": "ಊ", "U": "ಊ", "R": "ಋ",
    "e": "ಎ", "ee": "ಏ", "E": "ಏ", "ai": "ಐ",
    "o": "ಒ", "oo": "ಓ", "O": "ಓ", "au": "ಔ",
}

MATRAS = {
    "a": "", "aa": "ಾ", "A": "ಾ", "i": "ಿ", "ii": "ೀ",
    "I": "ೀ", "u": "ು", "uu": "ೂ", "U": "ೂ", "R": "ೃ",
    "e": "ೆ", "ee": "ೇ", "E": "ೇ", "ai": "ೈ",
    "o": "ೊ", "oo": "ೋ", "O": "ೋ", "au": "ೌ",
}

CONSONANTS = {
    "k": "ಕ", "kh": "ಖ", "g": "ಗ", "gh": "ಘ", "~N": "ಙ",
    "ch": "ಚ", "Ch": "ಛ", "chh": "ಛ", "j": "ಜ", "jh": "ಝ",
    "~n": "ಞ",
    "T": "ಟ", "Th": "ಠ", "D": "ಡ", "Dh": "ಢ", "N": "ಣ",
    "t": "ತ", "th": "ಥ", "d": "ದ", "dh": "ಧ", "n": "ನ",
    "p": "ಪ", "ph": "ಫ", "f": "ಫ", "b": "ಬ", "bh": "ಭ",
    "m": "ಮ",
    "y": "ಯ", "r": "ರ", "rY": "ಱ", "l": "ಲ", "L": "ಳ",
    "lY": "ೞ", "v": "ವ", "w": "ವ",
    "sh": "ಶ", "Sh": "ಷ", "S": "ಷ", "s": "ಸ", "h": "ಹ",
}

SIGNS = {"M": "ಂ", "H": "ಃ"}
DIGITS = {str(i): chr(0x0CE6 + i) for i in range(10)}

# Longest first, so "chh" wins over "ch" and "aa" over "a".
_CONS_KEYS = sorted(CONSONANTS, key=len, reverse=True)
_VOWEL_KEYS = sorted(MATRAS, key=len, reverse=True)


def _match(text: str, i: int, keys: list[str]) -> str | None:
    for k in keys:
        if text.startswith(k, i):
            return k
    return None


def to_kannada(text: str) -> str:
    """Transliterate romanised input. Characters with no mapping -- Kannada
    already typed, spaces, punctuation -- pass through unchanged, so a
    half-typed line is never mangled."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("||", i):
            out.append("॥")
            i += 2
            continue
        ch = text[i]
        if ch == "|":
            out.append("।")
            i += 1
            continue
        if ch in DIGITS:
            out.append(DIGITS[ch])
            i += 1
            continue

        cons = _match(text, i, _CONS_KEYS)
        if cons:
            i += len(cons)
            out.append(CONSONANTS[cons])
            if i < n and text[i] == "q":  # explicit bare consonant
                out.append(VIRAMA)
                i += 1
                continue
            vowel = _match(text, i, _VOWEL_KEYS)
            if vowel:
                out.append(MATRAS[vowel])
                i += len(vowel)
            else:
                out.append(VIRAMA)  # cluster or word-final
            continue

        vowel = _match(text, i, sorted(INDEP_VOWELS, key=len, reverse=True))
        if vowel:
            out.append(INDEP_VOWELS[vowel])
            i += len(vowel)
            continue

        if ch in SIGNS:
            out.append(SIGNS[ch])
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def js_tables() -> str:
    """The same tables as a JS object literal, for make_transcriber.py."""
    import json
    return json.dumps({
        "indep": INDEP_VOWELS, "matra": MATRAS, "cons": CONSONANTS,
        "signs": SIGNS, "digits": DIGITS, "virama": VIRAMA,
    }, ensure_ascii=False)

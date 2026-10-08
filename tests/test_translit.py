"""The browser transliterator must agree with setu.data.translit.

make_transcriber.py emits that module's tables into the page, so the two
can only drift if the emitted JSON stops matching the Python tables --
which is exactly what this checks. A drift would be silent: the page
would keep accepting input and quietly produce different Kannada from
everything measured here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from setu.data import translit  # noqa: E402
from setu.data.translit import to_kannada  # noqa: E402


def test_known_words() -> None:
    cases = {
        "kannaDa": "ಕನ್ನಡ",
        "namaskAra": "ನಮಸ್ಕಾರ",
        "mugdhe": "ಮುಗ್ಧೆ",      # the ದ/ಧ demo word
        "jIva": "ಜೀವ",
        "tatva": "ತತ್ವ",
        "saMga": "ಸಂಗ",
        "duHkha": "ದುಃಖ",
        "rAm": "ರಾಮ್",           # word-final bare consonant
        "||": "॥",
        "12": "೧೨",
    }
    for src, want in cases.items():
        got = to_kannada(src)
        assert got == want, f"{src!r} -> {got!r}, expected {want!r}"
    print(f"OK  {len(cases)} known words transliterate correctly")


def test_archaic_letters_reachable() -> None:
    # CLAUDE.md singles these out; a transcriber that cannot type them
    # would silently normalise exactly the letters that mark a text old.
    assert to_kannada("marYe") == "ಮಱೆ"
    assert to_kannada("palYeya") == "ಪೞೆಯ"
    print("OK  ಱ and ೞ are typeable (rY, lY)")


def test_longest_match_wins() -> None:
    assert to_kannada("aa") == "ಆ", "aa must beat a+a"
    assert to_kannada("kha") == "ಖ", "kh must beat k+h"
    assert to_kannada("Sha") == "ಷ"
    print("OK  longest-match ordering holds")


def test_unmapped_passes_through() -> None:
    # A half-typed line must never be mangled: Kannada already present,
    # spaces and punctuation survive untouched.
    assert to_kannada("ಕನ್ನಡ") == "ಕನ್ನಡ"
    assert to_kannada("ka ka") == "ಕ ಕ"
    assert to_kannada("?") == "?"
    print("OK  unmapped input passes through unchanged")


def test_js_tables_match_python() -> None:
    tables = json.loads(translit.js_tables())
    assert tables["cons"] == translit.CONSONANTS
    assert tables["matra"] == translit.MATRAS
    assert tables["indep"] == translit.INDEP_VOWELS
    assert tables["signs"] == translit.SIGNS
    assert tables["digits"] == translit.DIGITS
    assert tables["virama"] == translit.VIRAMA
    print("OK  emitted JS tables match the Python tables")


def main() -> None:
    test_known_words()
    test_archaic_letters_reachable()
    test_longest_match_wins()
    test_unmapped_passes_through()
    test_js_tables_match_python()
    print("\nAll transliteration tests passed.")


if __name__ == "__main__":
    main()

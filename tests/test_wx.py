"""Round-trip and vocabulary-size checks for setu.data.wx.

Plain assert-based (no pytest dependency -- not in requirements.txt, and
CLAUDE.md says ask before adding one). Run directly:
    python tests/test_wx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from setu.data import wx


def check_round_trip(text: str, label: str) -> None:
    symbols = wx.encode(text)
    back = wx.decode(symbols)
    assert back == text, f"[{label}] round-trip failed: {text!r} -> {symbols} -> {back!r}"
    # to_string/from_string must also round-trip through the symbol list
    assert wx.from_string(wx.to_string(symbols)) == symbols, f"[{label}] string round-trip failed"
    print(f"OK  [{label}] {text!r} -> {wx.to_string(symbols)}")


def test_vocab_size_in_target_range() -> None:
    n = len(wx.VOCAB)
    assert 45 <= n <= 65, f"expected ~50-60 WX symbols per roadmap, got {n}"
    print(f"OK  vocab size = {n} symbols (+ blank = {n + 1} CTC classes)")


def test_every_codepoint_round_trips() -> None:
    for cp in list(wx.CONSONANTS) + list(wx.INDEP_VOWELS) + list(wx.OTHER_SIGNS):
        check_round_trip(cp, f"lone codepoint U+{ord(cp):04X}")


def test_archaic_letters() -> None:
    # ಱ (archaic trill r) and ೞ (archaic zha) are the letters that mark a
    # text as old Kannada -- CLAUDE.md calls out exactly these two as the
    # ones a syllable-level vocabulary would starve of training examples.
    # "ಮಱೆ" / "ಪೞೆಯ" are illustrative old-Kannada-style spellings exercising
    # ಱ and ೞ (the historical ಱ->ರ / ೞ->ಳ sound shifts into modern Kannada
    # are well documented; these strings are for symbol coverage, not
    # claimed as verified corpus attestations).
    check_round_trip("ಮಱೆ", "rY word (ಮಱೆ)")
    check_round_trip("ಪೞೆಯ", "zY word (ಪೞೆಯ)")


def test_consonant_clusters_and_virama() -> None:
    # ಕ್ತ = ka + virama + ta -> bare "k" (no vowel) followed by "t"+"a"
    check_round_trip("ಕ್ತ", "ka-virama-ta cluster")
    # word-final bare consonant (no trailing vowel/virama in the source)
    check_round_trip("ಅಕ್", "trailing virama (bare consonant at end)")


def test_matras_and_independent_vowels() -> None:
    # ಕಾ (ka + aa-matra) vs ಆ (independent aa) must use the same WX vowel
    # symbol "A" but decode back to different Unicode codepoints.
    ka_aa = wx.encode("ಕಾ")
    indep_aa = wx.encode("ಆ")
    assert ka_aa[-1] == indep_aa[0] == "A"
    check_round_trip("ಕಾ", "ka + aa-matra")
    check_round_trip("ಆ", "independent aa")


def test_space_and_danda() -> None:
    check_round_trip("ಕ ಕ।ಕ॥", "space + danda + double danda")


def test_encode_rejects_unmapped_character() -> None:
    try:
        wx.encode("hello")  # Latin text has no mapping -- must fail loudly
    except ValueError:
        pass
    else:
        raise AssertionError("encode() silently accepted unmapped Latin text")
    print("OK  encode() fails loudly on unmapped characters")


def main() -> None:
    test_vocab_size_in_target_range()
    test_every_codepoint_round_trips()
    test_archaic_letters()
    test_consonant_clusters_and_virama()
    test_matras_and_independent_vowels()
    test_space_and_danda()
    test_encode_rejects_unmapped_character()
    print("\nAll WX round-trip tests passed.")


if __name__ == "__main__":
    main()

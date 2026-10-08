"""Checks for build B's consensus over repeated vision-model readings.

This is the part that can fail silently: if the votes are built by
position instead of by alignment, one inserted character shifts every
later position and manufactures disagreement across the whole line --
which would look like rich, demo-worthy uncertainty while being an
artifact. Fixed inputs here, so the right answer is known.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from setu.demo.build_demo_real import (  # noqa: E402
    annotated_block,
    consensus,
    uncertain_slots,
    variant_lines,
)


def test_unanimous_readings_flag_nothing() -> None:
    base, slots = consensus(["ಪದ", "ಪದ", "ಪದ"])
    assert base == "ಪದ"
    assert uncertain_slots(slots, agree_below=1.0) == []
    print("OK  unanimous samples flag no characters")


def test_one_disagreement_is_found_with_its_frequency() -> None:
    # 4 of 5 read ದ, one read ಧ -- the ದ/ಧ confusion again.
    base, slots = consensus(["ಪದ", "ಪದ", "ಪಧ", "ಪದ", "ಪದ"])
    assert base == "ಪದ", base
    unc = uncertain_slots(slots, agree_below=1.0)
    assert len(unc) == 1, unc
    assert unc[0]["index"] == 1
    assert abs(unc[0]["agreement"] - 0.8) < 1e-6, unc[0]
    alts = {a["symbol"]: a["prob"] for a in unc[0]["alternatives"]}
    assert abs(alts["ಪ"] if "ಪ" in alts else alts["ದ"] - 0.8) < 1e-6 or abs(alts["ದ"] - 0.8) < 1e-6
    assert abs(alts["ಧ"] - 0.2) < 1e-6, alts
    print(f"OK  single disagreement found at index 1: {alts}")


def test_insertion_does_not_cascade() -> None:
    """The bug this file exists for. One reading inserts a character at
    the front; every later character still agrees."""
    base, slots = consensus(["ಕನ್ನಡ", "ಕನ್ನಡ", "ಪಕನ್ನಡ"])
    assert base == "ಕನ್ನಡ", base
    unc = uncertain_slots(slots, agree_below=1.0)
    # Positional comparison would flag nearly every character here.
    assert len(unc) <= 1, f"an insertion cascaded into {len(unc)} contested characters"
    print(f"OK  a leading insertion contests {len(unc)} character(s), not the whole line")


def test_medoid_beats_an_outlier() -> None:
    """The base must be the reading closest to the others, not the first
    sample -- otherwise one bad read becomes the spine."""
    base, _ = consensus(["ಕುಕುಕುಕು", "ಜೀವ ತತ್ವ", "ಜೀವ ತತ್ವ", "ಜೀವ ತತ್ವ"])
    assert base == "ಜೀವ ತತ್ವ", base
    print("OK  medoid chosen over an outlier first sample")


def test_deletion_counts_as_a_vote() -> None:
    base, slots = consensus(["ಪದ", "ಪದ", "ಪ"])
    unc = uncertain_slots(slots, agree_below=1.0)
    assert len(unc) == 1
    alts = {a["symbol"]: a["prob"] for a in unc[0]["alternatives"]}
    assert "" in alts, f"a reading that omitted the character must vote: {alts}"
    print(f"OK  an omission is counted as a vote: {alts}")


def test_variants_are_whole_lines_and_b3_is_one_string() -> None:
    base, slots = consensus(["ಪದ", "ಪದ", "ಪಧ", "ಪದ"])
    unc = uncertain_slots(slots, agree_below=1.0)
    variants = variant_lines(base, unc, max_variants=5)
    texts = [t for t, _ in variants]
    assert texts[0] == "ಪದ"
    assert "ಪಧ" in texts, texts
    block = annotated_block(base, variants)
    assert "ಪದ" in block and "ಪಧ" in block
    # B3 is the base alone: the rival reading must not reach it.
    assert "ಪಧ" not in base
    print(f"OK  B4 carries {texts}; B3 carries only {base!r}")


def test_empty_and_single_sample_are_safe() -> None:
    assert consensus([])[0] == ""
    assert consensus(["", "  "])[0] == ""
    base, slots = consensus(["ಪದ"])
    assert base == "ಪದ" and len(slots) == 2
    assert uncertain_slots(slots, agree_below=1.0) == []
    print("OK  empty and single-sample inputs handled")


def main() -> None:
    test_unanimous_readings_flag_nothing()
    test_one_disagreement_is_found_with_its_frequency()
    test_insertion_does_not_cascade()
    test_medoid_beats_an_outlier()
    test_deletion_counts_as_a_vote()
    test_variants_are_whole_lines_and_b3_is_one_string()
    test_empty_and_single_sample_are_safe()
    print("\nAll consensus tests passed.")


if __name__ == "__main__":
    main()

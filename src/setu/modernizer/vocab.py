"""Character-level vocabulary for the modernizer's OUTPUT side (real modern
Kannada text, e.g. S2/S3's `modern_text` column) -- built from data, not
hand-specified, since modern Kannada text carries ordinary punctuation,
occasional digits/Latin citation fragments, etc. (measured: 138 distinct
characters across the S3 corpus alone). The INPUT side uses
setu.data.wx.VOCAB directly (the fixed ~56-symbol WX alphabet) -- no
separate vocab needed there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


@dataclass
class CharVocab:
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    @property
    def pad_id(self) -> int:
        return self.char_to_idx[PAD]

    @property
    def bos_id(self) -> int:
        return self.char_to_idx[BOS]

    @property
    def eos_id(self) -> int:
        return self.char_to_idx[EOS]

    @property
    def unk_id(self) -> int:
        return self.char_to_idx[UNK]

    def __len__(self) -> int:
        return len(self.char_to_idx)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            ch = self.idx_to_char.get(i, UNK)
            if ch == EOS:
                break
            if ch in (PAD, BOS):
                continue
            chars.append(ch)
        return "".join(chars)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.char_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "CharVocab":
        char_to_idx = json.loads(path.read_text(encoding="utf-8"))
        idx_to_char = {v: k for k, v in char_to_idx.items()}
        return CharVocab(char_to_idx, idx_to_char)


def build_vocab(texts: list[str]) -> CharVocab:
    chars = sorted(set().union(*[set(t) for t in texts])) if texts else []
    char_to_idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for ch in chars:
        if ch not in char_to_idx:
            char_to_idx[ch] = len(char_to_idx)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return CharVocab(char_to_idx, idx_to_char)

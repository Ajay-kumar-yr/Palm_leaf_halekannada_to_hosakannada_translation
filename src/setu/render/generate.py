"""CLI / library entry point that turns a text corpus into rendered,
damaged line images plus a manifest — the shared machinery behind both S1
(old text only) and S2 (old text + modern text) generation.

Input format: one line of source text per row. For S1, each row is just
old-Kannada text. For S2, each row is `old_text\tmodern_text` (tab
separated) — pass `--paired` so the modern-text column is captured in the
manifest instead of being treated as part of the old text.

Usage:
    python -m setu.render.generate --corpus path/to/lines.txt \
        --out-dir data/s1 --manifest data/s1/manifest.jsonl --seed 0

This does not yet fetch the actual Kannada Wikisource / KannadaLit4NLP
corpora — it takes whatever line-delimited text file it's given. Building
the corpus itself (source selection, cleaning) is separate work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from setu.render.damage import compose_damage
from setu.render.fonts import require_fonts
from setu.render.rasterize import render_clean_line


def generate_dataset(
    corpus_path: Path,
    out_dir: Path,
    manifest_path: Path,
    real_dir: Path,
    seed: int,
    paired: bool = False,
    pixel_size_range: tuple[int, int] = (36, 56),
    line_height: int = 64,
    limit: int | None = None,
) -> None:
    rng = np.random.default_rng(seed)  # CLAUDE.md rule 5: seed every entry point
    fonts = require_fonts()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        line.rstrip("\n")
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        lines = lines[:limit]

    n_written, n_skipped = 0, 0
    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for idx, raw_line in enumerate(lines):
            if paired:
                parts = raw_line.split("\t", 1)
                if len(parts) != 2:
                    n_skipped += 1
                    continue
                old_text, modern_text = parts
            else:
                old_text, modern_text = raw_line, None

            font = fonts[rng.integers(len(fonts))]
            pixel_size = int(rng.integers(*pixel_size_range))

            clean, missing = render_clean_line(old_text, font, pixel_size, line_height)
            if missing:
                # A font missing a glyph for this line's text is exactly
                # the silent-failure mode this project is built to avoid
                # (see CLAUDE.md: archaic letters sit in the rare tail).
                # Skip the line rather than render it wrong, and record why.
                n_skipped += 1
                manifest_f.write(
                    json.dumps(
                        {
                            "id": f"line_{idx:06d}",
                            "skipped": True,
                            "reason": "missing_glyphs",
                            "missing_codepoints": [hex(c) for c in missing],
                            "text": old_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            damaged, params = compose_damage(clean, rng, real_dir)

            image_id = f"line_{idx:06d}"
            image_path = images_dir / f"{image_id}.png"
            # compose_damage's output is already ink-dark-on-light (real texture
            # background, dark ink strokes) -- its own docstring says "ready to
            # save". No further inversion here: a `255 - damaged` used to be
            # applied on top, which flipped it to a bright-ink/dark-background
            # negative on disk (harmless for training, since train.py's loader
            # used to invert back -- but wrong for anything reading the PNG
            # directly, and just confusing to look at).
            Image.fromarray(damaged).save(image_path)

            record = {
                "id": image_id,
                "text": old_text,
                # .as_posix(), not str(): rendering runs on Windows (fonts.py needs a
                # Windows font path) but training runs on WSL/Linux per CLAUDE.md --
                # str() on Windows gives backslash separators, which Linux treats as a
                # literal filename character rather than a path separator, so
                # `data_dir / image_path` silently resolves to a nonexistent path.
                "image_path": image_path.relative_to(out_dir).as_posix(),
                "font": font.name,
                "pixel_size": pixel_size,
                "damage": vars(params),
            }
            if paired:
                record["modern_text"] = modern_text
            manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} images to {images_dir}, skipped {n_skipped}. Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--paired", action="store_true", help="corpus lines are old_text<TAB>modern_text (for S2)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    generate_dataset(
        corpus_path=args.corpus,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        real_dir=args.real_dir,
        seed=args.seed,
        paired=args.paired,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

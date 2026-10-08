"""Persistent Palmira segmentation worker (run in the `palmira` conda env).

Palmira needs Python 3.7 + detectron2 + the compiled DefGrid CUDA
extension; the rest of the pipeline needs Python 3.14 + torch 2.14.
Neither can import the other, so the demo app talks to Palmira through
the filesystem instead of in-process.

Why a worker rather than a subprocess per upload: loading the model
costs ~15-20 s, which would be paid on every single upload and make the
demo feel broken. Here it is paid once at startup.

Protocol -- deliberately the simplest thing that cannot half-work:

    inbox/<job>.json     {"image": "<abs path>", "out_dir": "<abs path>"}
    outbox/<job>.json    {"ok": true, "lines": [...], "overlay": "..."}
                         or {"ok": false, "error": "..."}

The worker writes its result to a temporary name and renames it into
place, so a reader never sees a half-written file. A job file is deleted
once picked up, so a crash mid-job loses that job rather than looping on
it forever.

Run it from the Palmira checkout (it needs configs/, pretrained/,
predictor.py and defgrid/ relative to cwd):

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate palmira
    cd ~/Palmira
    python /mnt/d/.../data/raw_corpus/palmira_worker.py
"""
# NOTE: this file runs under the palmira conda env's Python 3.7, not the
# 3.14 the rest of the project uses. Keep it 3.7-compatible: no walrus,
# no positional-only params, and no Path.unlink(missing_ok=) (3.8+).
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.data.detection_utils import read_image

sys.path.insert(0, str(Path.cwd()))
from predictor import VisualizationDemo  # noqa: E402

from run_palmira_survey import build_cfg  # noqa: E402

ROOT = Path("/mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation")
INBOX = ROOT / "data" / "demo_jobs" / "inbox"
OUTBOX = ROOT / "data" / "demo_jobs" / "outbox"
LINE_CLASS = "Character Line Segment"
READY = ROOT / "data" / "demo_jobs" / "worker_ready"


def unlink_quietly(p: Path) -> None:
    """Path.unlink(missing_ok=True) is 3.8+; this env is 3.7."""
    try:
        p.unlink()
    except OSError:
        pass


def mask_filled_crop(img_bgr, mask, box):
    """Same treatment as run_palmira_crops.py: blank everything outside
    the (slightly dilated) line mask, so a slanted neighbour intruding
    into this line's bounding box does not get read as part of it."""
    x0, y0, x1, y1 = box
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    crop = img_bgr[y0:y1, x0:x1].copy()
    k = max(3, (y1 - y0) // 10 | 1)
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    inside = crop[m.astype(bool)]
    fill = np.median(inside, axis=0).astype(np.uint8) if len(inside) else np.uint8([255, 255, 255])
    crop[~m.astype(bool)] = fill
    return crop


def segment(demo, image_path: Path, out_dir: Path) -> dict:
    img = read_image(str(image_path), format="BGR")
    pred, vis = demo.run_on_image(img)
    inst = pred["instances"].to("cpu")
    names = demo.metadata.thing_classes

    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = out_dir / "palmira_overlay.jpg"
    try:
        vis.save(str(overlay))
    except Exception:
        overlay = None

    records = []
    for i, cls in enumerate(inst.pred_classes.tolist()):
        if names[cls] != LINE_CLASS:
            continue
        mask = inst.pred_masks[i].numpy()
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        records.append({
            "box": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            "mask": mask, "cy": float(ys.mean()), "score": float(inst.scores[i]),
        })
    records.sort(key=lambda r: r["cy"])  # reading order, top to bottom

    lines = []
    for n, r in enumerate(records):
        x0, y0, x1, y1 = r["box"]
        rel = f"line_{n:02d}.png"
        cv2.imwrite(str(out_dir / rel), mask_filled_crop(img, r["mask"], r["box"]))
        lines.append({"index": n, "file": str(out_dir / rel), "box_xyxy": [x0, y0, x1, y1],
                      "width": x1 - x0, "height": y1 - y0, "score": round(r["score"], 4)})

    # Every detected class, so the UI can show what Palmira tagged, not
    # just the lines we keep.
    tags = {}
    for cls in inst.pred_classes.tolist():
        name = names[cls] if cls < len(names) else f"class_{cls}"
        tags[name] = tags.get(name, 0) + 1

    return {"ok": True, "lines": lines, "n_lines": len(lines), "tags": tags,
            "overlay": str(overlay) if overlay else None,
            "image_size": [int(img.shape[1]), int(img.shape[0])]}


def main() -> None:
    INBOX.mkdir(parents=True, exist_ok=True)
    OUTBOX.mkdir(parents=True, exist_ok=True)
    print("loading Palmira (once) ...", flush=True)
    demo = None
    try:
        demo = VisualizationDemo(build_cfg("cuda"))
    except Exception:
        print("CUDA predictor failed, falling back to CPU", flush=True)
        demo = VisualizationDemo(build_cfg("cpu"))
    READY.write_text(str(os.getpid()), encoding="utf-8")
    print(f"ready (pid {os.getpid()}); watching {INBOX}", flush=True)

    try:
        while True:
            jobs = sorted(INBOX.glob("*.json"))
            if not jobs:
                time.sleep(0.3)
                continue
            for job in jobs:
                try:
                    spec = json.loads(job.read_text(encoding="utf-8"))
                except Exception:
                    unlink_quietly(job)
                    continue
                unlink_quietly(job)  # claim it before working
                name = job.stem
                try:
                    result = segment(demo, Path(spec["image"]), Path(spec["out_dir"]))
                except Exception as e:
                    result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                              "traceback": traceback.format_exc()[-1500:]}
                    torch.cuda.empty_cache()
                tmp = OUTBOX / f".{name}.tmp"
                tmp.write_text(json.dumps(result), encoding="utf-8")
                tmp.replace(OUTBOX / f"{name}.json")  # atomic: no half-read
                print(f"  {name}: {'ok ' + str(result.get('n_lines')) + ' lines' if result['ok'] else result['error'][:80]}",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        unlink_quietly(READY)
        print("worker stopped", flush=True)


if __name__ == "__main__":
    main()

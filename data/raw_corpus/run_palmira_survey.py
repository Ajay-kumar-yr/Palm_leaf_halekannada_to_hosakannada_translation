"""One-off script (run inside the `palmira` conda env on WSL, not part of
setu/ -- Palmira has its own isolated, fragile dependency stack, kept
fully separate per CLAUDE.md: "used exactly as downloaded, no
fine-tuning"). Runs Palmira over every REAL HKHPL page (original color
photo, never B&W -- Palmira was trained on photos, per CLAUDE.md/the
earlier original-vs-BW check) and records per-page instance counts, for
the Week 1 roadmap task this session never actually finished: "run
Palmira over every page... (this also produces the §6.4 segmentation
number)".

Dedupes to the 738 physically-unique photos (of 1,259 file paths) --
train/val/test are mirrors of the same files for the dataset's original
segmentation benchmark, not 1,259 distinct pages.

GPU-OOM fallback to CPU per-image: the 4GB RTX 3050 was already confirmed
(in the earlier original-vs-BW check) to be insufficient for some large
real photos (e.g. a 4096x3072 facing-page pair) even for a SINGLE image.

Writes one JSON line per image to palmira_survey_results.jsonl
incrementally (so a crash partway through loses only the current image,
not prior progress) and saves a handful of visualized outputs for a
human to spot-check.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from defgrid.config import add_defgrid_maskhead_config
from predictor import VisualizationDemo

REAL_DIR = Path("/mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/data/real")
SURVEY_CSV = Path(
    "/mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/data/splits/hkhpl_page_survey.csv"
)
OUT_JSONL = Path.home() / "palmira_survey_results.jsonl"
VIS_OUT_DIR = Path.home() / "palmira_survey_vis_sample"
N_VIS_SAMPLES = 20


def build_cfg(device: str):
    cfg = get_cfg()
    add_defgrid_maskhead_config(cfg)
    cfg.merge_from_file("configs/palmira/Palmira.yaml")
    cfg.merge_from_list(
        ["MODEL.WEIGHTS", "pretrained/Palmira_indiscapes.pth", "MODEL.DEVICE", device]
    )
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg


def unique_real_image_paths() -> list[Path]:
    """Same dedup logic as the earlier demo-page-selection pass: one path
    per unique (width,height,sharpness,contrast,brightness) signature."""
    rows = list(csv.DictReader(SURVEY_CSV.open(encoding="utf-8")))
    seen, paths = set(), []
    for r in rows:
        sig = (r["width"], r["height"], r["sharpness"], r["contrast"], r["brightness"])
        if sig in seen or not r["width"]:
            continue
        seen.add(sig)
        paths.append(REAL_DIR / r["path"].replace("\\", "/"))  # CSV written on Windows
    return paths


def main() -> None:
    paths = unique_real_image_paths()
    print(f"{len(paths)} unique real photos to process")

    already_done = set()
    if OUT_JSONL.exists():
        for line in OUT_JSONL.open(encoding="utf-8"):
            already_done.add(json.loads(line)["path"])
        print(f"Resuming: {len(already_done)} already processed")

    demo_gpu = VisualizationDemo(build_cfg("cuda"))
    demo_cpu = None  # built lazily only if a GPU OOM actually happens

    VIS_OUT_DIR.mkdir(exist_ok=True)
    n_vis_saved = 0

    with OUT_JSONL.open("a", encoding="utf-8") as f_out:
        for i, path in enumerate(paths):
            rel = str(path.relative_to(REAL_DIR))
            if rel in already_done:
                continue
            try:
                img = read_image(str(path), format="BGR")
            except Exception as e:
                f_out.write(json.dumps({"path": rel, "error": f"read failed: {e}"}) + "\n")
                f_out.flush()
                continue

            t0 = time.time()
            device_used = "cuda"
            try:
                try:
                    predictions, vis_output = demo_gpu.run_on_image(img)
                except RuntimeError as e:
                    # This torch build (python3.7, pinned for the DefGrid CUDA
                    # extension) predates torch.cuda.OutOfMemoryError as a
                    # distinct class -- OOM surfaces as a plain RuntimeError
                    # with "out of memory" in the message, so match on that
                    # instead of the exception type.
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    if demo_cpu is None:
                        print("  GPU OOM -- building CPU predictor (first fallback, one-time cost)")
                        demo_cpu = VisualizationDemo(build_cfg("cpu"))
                    device_used = "cpu"
                    predictions, vis_output = demo_cpu.run_on_image(img)
            except Exception as e:
                # A genuine model-internal bug (e.g. a reshape error when the
                # detector finds zero candidate regions on some image) must
                # not kill the whole batch over one page -- Palmira is used
                # exactly as downloaded, no fine-tuning, so the fix belongs
                # here (skip and record), not inside its own model code.
                f_out.write(json.dumps({"path": rel, "error": f"inference failed: {e}"}) + "\n")
                f_out.flush()
                torch.cuda.empty_cache()
                continue
            elapsed = time.time() - t0

            instances = predictions.get("instances")
            class_counts: dict[str, int] = {}
            if instances is not None and len(instances) > 0:
                names = demo_gpu.metadata.thing_classes
                for cls_id in instances.pred_classes.tolist():
                    name = names[cls_id] if cls_id < len(names) else f"cls_{cls_id}"
                    class_counts[name] = class_counts.get(name, 0) + 1

            record = {
                "path": rel,
                "device": device_used,
                "seconds": round(elapsed, 2),
                "n_instances": len(instances) if instances is not None else 0,
                "class_counts": class_counts,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            if n_vis_saved < N_VIS_SAMPLES and i % max(len(paths) // N_VIS_SAMPLES, 1) == 0:
                try:
                    vis_output.save(str(VIS_OUT_DIR / f"{Path(rel).stem}.jpg"))
                    n_vis_saved += 1
                except Exception:
                    pass

            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(paths)} ({device_used}, {elapsed:.1f}s, {len(class_counts)} classes)")

    print(f"Done. Results: {OUT_JSONL}, {n_vis_saved} sample visualizations in {VIS_OUT_DIR}")


if __name__ == "__main__":
    main()

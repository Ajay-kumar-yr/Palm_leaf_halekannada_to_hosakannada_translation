"""The two end-to-end pipelines behind the demo UI, with caching.

    REAL photo      Palmira -> crops -> binarize -> LLM x N ensemble
                    -> consensus + alternatives -> modernizer (B3 | B4)

    SYNTHETIC line  CRNN -> per-frame CTC distributions
                    -> soft bridge -> alternatives -> modernizer (B3 | B4)

The two differ in what produces the uncertainty, which is the whole
point of showing both: on synthetic lines it is our own recogniser's
CTC posterior (the project's actual contribution, 80.6% of wrong frames
still hold the right symbol); on real crops our recogniser cannot read
at all, so the same *principle* is shown with an ensemble over a
borrowed reader, which recovers 26.0%. See RESULTS.md §2.

**Caching is not an optimisation here, it is a safety requirement.**
Free-tier quota is 20 requests per key per day per model; a demo that
calls out live can die in front of an examiner. Every result is keyed
by a hash of the image bytes plus the settings that affect the answer,
so a previously-seen image replays instantly and spends nothing. Only a
genuinely new upload calls out.

Palmira runs in a different Python environment and is reached through
`data/raw_corpus/palmira_worker.py` over a filesystem mailbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
JOBS = REPO_ROOT / "data" / "demo_jobs"
CACHE = REPO_ROOT / "data" / "demo_cache" / "pipeline"
WORK = REPO_ROOT / "data" / "demo_work"
WORKER_READY = JOBS / "worker_ready"

LINE_HEIGHT = 64
MIN_HEIGHT = 32
CRNN_CKPT = REPO_ROOT / "runs" / "20261006T120718Z_crnn_train_s1" / "best_model.pt"


# --------------------------------------------------------------------- routing

def classify_input(path: Path) -> tuple[str, str]:
    """('real' | 'synthetic', why).

    A rendered S1/S2 line is a single strip -- very wide, short, and
    already one line. A manuscript photo is a whole page. Aspect ratio
    separates them cleanly (S1 lines run 12:1 and up; pages are nearer
    4:3), and the caller can override, so this only has to be right
    usually, not always.
    """
    with Image.open(path) as im:
        w, h = im.size
    aspect = w / max(h, 1)
    if aspect >= 6.0:
        return "synthetic", f"{w}x{h}, aspect {aspect:.1f}:1 — a single line strip"
    return "real", f"{w}x{h}, aspect {aspect:.1f}:1 — a full page, needs line segmentation"


def image_key(image_path: Path, **settings) -> str:
    """Cache key: the image bytes plus every setting that changes the
    answer. The parameter is `image_path`, not `path`, because callers
    pass a `path=` setting naming the route."""
    h = hashlib.sha256(Path(image_path).read_bytes())
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()[:24]


# A picked file is read straight from disk, so its bytes are stable and
# `image_key` is exact. Uploads still go through `gr.Image`, which
# re-encodes to lossy webp (its `format` argument is not honoured in
# Gradio 6), so an uploaded copy of a staged file is a DIFFERENT image
# as far as the cache is concerned. That is why the demo picks files
# rather than uploading them -- see `build()`.


def real_key(image: Path, samples: int, temperature: float, max_lines: int,
             read: bool, modern: bool) -> str:
    """The cache key for the real route. ONE definition, used by both
    `run_real` and the UI's `--no-live` gate.

    It was two definitions, and they drifted: `binarizer` was added here
    and not to the gate, `resize`/`ckpt` likewise for the synthetic key,
    so the gate computed a key that could never match a stored one.
    `is_cached` was therefore always False, and `--no-live` -- the mode
    the demo is supposed to be presented in -- refused every image
    including the ones just pre-cached. Live mode hid it, because the
    run functions consult their own cache afterwards.
    """
    return image_key(image, path="real", n=int(samples), t=float(temperature),
                     m=int(max_lines), read=read, modern=modern,
                     binarizer="sajjan_unet")


def synthetic_key(image: Path, temperature: float = 1.5, top_k: int = 5,
                  uncertain_below: float = 0.9, modern: bool = False) -> str:
    """The cache key for the synthetic route. See `real_key`."""
    return image_key(image, path="synthetic", t=temperature, k=top_k, u=uncertain_below,
                     modern=modern, resize=False, ckpt=CRNN_CKPT.parent.name)


def cached(key: str) -> dict | None:
    f = CACHE / f"{key}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        d["from_cache"] = True
        return d
    return None


def store(key: str, payload: dict) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    payload["from_cache"] = False
    (CACHE / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return payload


# --------------------------------------------------------------------- Palmira

def worker_alive() -> bool:
    """The ready file exists AND the process it names is still running.

    The file is written once at startup and nothing removes it when the
    worker dies or the machine reboots, so existence alone reported a
    worker from the previous day as running. The real route then
    accepted the job, waited out its 180s timeout and failed -- the
    worst possible version of this, because the page had already told
    the presenter the worker was up.

    The pid is only meaningful on the host that wrote it; the app and
    the worker both run inside WSL, which is the only supported way to
    run the real route (DEMO_RUNBOOK.md). Where /proc is absent we fall
    back to trusting the file rather than refusing to work.
    """
    if not WORKER_READY.exists():
        return False
    try:
        pid = int(WORKER_READY.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    proc = Path("/proc")
    if not proc.is_dir():
        return True
    return (proc / str(pid)).exists()


def segment_page(image: Path, timeout: float = 180.0) -> dict:
    """Hand the page to the Palmira worker and wait for its answer."""
    if not worker_alive():
        return {"ok": False, "error":
                "Palmira worker is not running. Start it in the palmira conda env:\n"
                "  conda activate palmira && cd ~/Palmira && python "
                "/mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/"
                "data/raw_corpus/palmira_worker.py"}
    (JOBS / "inbox").mkdir(parents=True, exist_ok=True)
    (JOBS / "outbox").mkdir(parents=True, exist_ok=True)
    job = uuid.uuid4().hex[:12]
    out_dir = WORK / job
    (JOBS / "inbox" / f"{job}.json").write_text(
        json.dumps({"image": str(Path(image).resolve()), "out_dir": str(out_dir)}),
        encoding="utf-8")
    result_path = JOBS / "outbox" / f"{job}.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if result_path.exists():
            d = json.loads(result_path.read_text(encoding="utf-8"))
            result_path.unlink(missing_ok=True)
            return d
        time.sleep(0.25)
    return {"ok": False, "error": f"Palmira worker did not answer within {timeout:.0f}s"}


# ----------------------------------------------------------------- binarizer

_BINARIZER: list = [None]


def binarize_crop(path: Path, out: Path) -> Path:
    from setu.demo.binarize import binarize, load_binarizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _BINARIZER[0] is None:
        _BINARIZER[0] = load_binarizer(device=dev)
    arr = binarize(Image.open(path), _BINARIZER[0], device=dev)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out)
    return out


# ---------------------------------------------------------------------- CRNN

_CRNN: list = [None]


def load_crnn(checkpoint: Path = CRNN_CKPT, digits: bool = False):
    from setu.data import wx
    from setu.recogniser.model import CRNN

    if _CRNN[0] is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = len(wx.tables(digits)[0]) + 1
        m = CRNN(num_classes=n).to(dev)
        m.load_state_dict(torch.load(checkpoint, map_location=dev))
        m.eval()
        _CRNN[0] = (m, dev)
    return _CRNN[0]


def crnn_line(path: Path, resize: bool, digits: bool = False):
    """(log_probs (T,C), input_length, the array actually fed in)."""
    from setu.recogniser.model import WIDTH_DOWNSAMPLE

    model, dev = load_crnn(digits=digits)
    im = Image.open(path).convert("L")
    if resize:
        w = max(WIDTH_DOWNSAMPLE, round(im.width * LINE_HEIGHT / im.height))
        im = im.resize((w, LINE_HEIGHT), Image.LANCZOS)
    a = np.array(im, dtype=np.uint8)
    if a.shape[0] < MIN_HEIGHT:
        pad = MIN_HEIGHT - a.shape[0]
        a = np.pad(a, ((pad // 2, pad - pad // 2), (0, 0)), constant_values=int(np.median(a)))
    x = torch.from_numpy(a.astype(np.float32) / 255.0)[None, None].to(dev)
    with torch.no_grad():
        lp = model(x)
    return lp[:, 0].cpu(), max(1, a.shape[1] // WIDTH_DOWNSAMPLE), a


# ------------------------------------------------------------------- results

@dataclass
class LineResult:
    index: int
    crop: str
    binarized: str | None = None
    b3_text: str = ""
    b4_block: str = ""
    rivals: list = field(default_factory=list)
    uncertain: list = field(default_factory=list)
    n_chars: int = 0
    n_uncertain: int = 0
    samples: list = field(default_factory=list)
    b3_modern: str | None = None
    b4_modern: str | None = None

    def differ(self) -> bool | None:
        if self.b3_modern is None or self.b4_modern is None:
            return None
        return self.b3_modern != self.b4_modern

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["branches_differ"] = self.differ()
        return d


# ----------------------------------------------------------- synthetic path

def run_synthetic(image: Path, temperature: float = 1.5, top_k: int = 5,
                  uncertain_below: float = 0.9, modernize_fn=None,
                  progress=None) -> dict:
    """CRNN -> CTC distributions -> soft bridge -> modernizer.

    This is the project's own machinery end to end: our recogniser, our
    per-frame posterior, our bridge. The uncertainty here is the real
    thing, not an ensemble standing in for it.
    """
    from setu.bridge.line_alternatives import (
        analyse_line, annotated_text, argmax_text, summarise, variant_readings)

    # `resize` and the checkpoint belong in the key: both change the
    # answer, and leaving them out meant a fix to the resize behaviour
    # silently replayed the old, wrong result from cache.
    key = synthetic_key(image, temperature=temperature, top_k=top_k,
                        uncertain_below=uncertain_below, modern=modernize_fn is not None)
    hit = cached(key)
    if hit:
        return hit

    if progress:
        progress("Reading the line with the CRNN")
    # NOT resized. S1 images are what the CRNN trained on, at their own
    # scale (around 172px tall, text rendered at pixel size 36-56);
    # squashing one to height 64 shrinks the glyphs to roughly 15px and
    # the reading collapses to noise. Only real crops get normalised,
    # because they are not in the training distribution to begin with.
    lp, n_frames, _ = crnn_line(image, resize=False)
    symbols, slots = analyse_line(lp, n_frames, temperature=temperature, top_k=top_k,
                                  uncertain_below=uncertain_below)
    if progress:
        progress("Applying the soft bridge")
    b3 = argmax_text(symbols)
    block = annotated_text(symbols, slots)
    rivals = [{"text": t, "prob": round(p, 3)} for t, p in variant_readings(symbols, slots)]
    summary = summarise(slots)

    line = LineResult(
        index=0, crop=str(image), b3_text=b3, b4_block=block, rivals=rivals,
        n_chars=summary["n_chars"], n_uncertain=summary["n_uncertain"],
        uncertain=[{"index": s.index, "chosen": s.chosen, "top1": round(s.top1_prob, 3),
                    "alternatives": [{"kannada": a.kannada, "prob": round(a.prob, 3)}
                                     for a in s.alternatives if a.symbol != "<blank>"][:4]}
                   for s in slots if s.uncertain],
    )
    if modernize_fn:
        if progress:
            progress("Modernizing both branches")
        line.b3_modern = modernize_fn(b3, "b3")
        line.b4_modern = modernize_fn(block, "b4")

    return store(key, {
        "route": "synthetic", "image": str(image),
        "steps": ["CRNN", "soft bridge (CTC posterior)", "modernizer"],
        "uncertainty_source": "our CRNN's per-frame CTC posterior -- the project's own "
                              "soft bridge (RESULTS.md 2.1: 80.6% of wrong frames still "
                              "hold the correct symbol in the top-5)",
        "lines": [line.as_dict()],
    })


# ---------------------------------------------------------------- real path

def run_real(image: Path, samples: int = 5, temperature: float = 0.8,
             max_lines: int = 6, read_fn=None, modernize_fn=None,
             progress=None) -> dict:
    """Palmira -> crops -> binarize -> LLM ensemble -> modernizer.

    Our CRNN cannot read real crops (0.696 CER), so the reading is done
    by a vision model and the uncertainty comes from disagreement
    between repeated reads. Same claim as the soft bridge, different
    mechanism -- and weaker: 26.0% recovery against the bridge's 80.6%
    (RESULTS.md 2.2).
    """
    from setu.demo.build_demo_real import (
        annotated_block, consensus, uncertain_slots, variant_lines)

    key = real_key(image, samples, temperature, max_lines,
                   read=read_fn is not None, modern=modernize_fn is not None)
    hit = cached(key)
    if hit:
        return hit

    # A page needs Palmira; an already-cut line crop does not, and
    # sending one through segmentation would just find itself. Aspect
    # separates them the same way the router does.
    with Image.open(image) as _im:
        pre_cut = _im.size[0] / max(_im.size[1], 1) >= 6.0
    if pre_cut:
        if progress:
            progress("Input is already a single line -- skipping Palmira")
        seg = {"ok": True, "n_lines": 1, "tags": None, "overlay": None,
               "pre_cut": True,
               "lines": [{"index": 0, "file": str(image)}]}
    else:
        if progress:
            progress("Segmenting the page with Palmira")
        seg = segment_page(image)
    if not seg.get("ok"):
        return {"route": "real", "image": str(image), "error": seg.get("error"),
                "lines": [], "from_cache": False}

    out_lines = []
    for rec in seg["lines"][:max_lines]:
        i = rec["index"]
        crop = Path(rec["file"])
        if progress:
            progress(f"Binarizing line {i + 1} of {min(len(seg['lines']), max_lines)}")
        binz = binarize_crop(crop, WORK / "binarized" / f"{crop.stem}_{image_key(crop)[:8]}_bin.png")

        line = LineResult(index=i, crop=str(crop), binarized=str(binz))
        if read_fn:
            if progress:
                progress(f"Reading line {i + 1} ({samples} passes)")
            texts = [t for t in read_fn(binz, samples, temperature) if t and t.strip()]
            line.samples = texts
            base, slots = consensus(texts)
            unc = uncertain_slots(slots, agree_below=1.0)
            variants = variant_lines(base, unc, max_variants=5)
            line.b3_text = base
            line.b4_block = annotated_block(base, variants)
            line.rivals = [{"text": t, "prob": round(p, 3)} for t, p in variants]
            line.n_chars = len(slots)
            line.n_uncertain = len(unc)
            line.uncertain = [{"index": s["index"], "chosen": s["chosen"],
                               "top1": s["agreement"],
                               "alternatives": s["alternatives"][:4]} for s in unc[:12]]
            if modernize_fn and base:
                if progress:
                    progress(f"Modernizing line {i + 1}")
                line.b3_modern = modernize_fn(line.b3_text, "b3")
                line.b4_modern = modernize_fn(line.b4_block, "b4")
        out_lines.append(line.as_dict())

    return store(key, {
        "route": "real", "image": str(image),
        "steps": (["already a single line crop"] if seg.get("pre_cut") else
                  ["Palmira segmentation", "line crops"]) +
                 ["U-Net binarization", f"vision-model ensemble (x{samples})", "modernizer"],
        "uncertainty_source": "disagreement between repeated reads of a vision model -- NOT "
                              "the CTC soft bridge. Same claim, different mechanism, and "
                              "weaker: 26.0% recovery against the bridge's 80.6% "
                              "(RESULTS.md 2.2)",
        "palmira": {"tags": seg.get("tags"), "n_lines": seg.get("n_lines"),
                    "overlay": seg.get("overlay"), "image_size": seg.get("image_size")},
        "lines": out_lines,
    })

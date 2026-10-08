"""SETU demo: upload a manuscript image and watch the pipeline run.

Two pipelines, chosen by what is uploaded:

    a full page photo   Palmira segmentation -> line crops -> U-Net
                        binarization -> vision-model ensemble ->
                        modernizer (B3 argmax vs B4 carrying uncertainty)

    a rendered line     CRNN -> per-frame CTC posterior -> soft bridge ->
                        modernizer (B3 vs B4)

The split is not cosmetic. On synthetic lines the uncertainty is our own
recogniser's CTC posterior -- the project's actual contribution, where
80.6% of wrong top-1 frames still hold the correct symbol. On real
crops our recogniser cannot read at all (0.687 CER), so the same
principle is shown with an ensemble over a borrowed reader, which
recovers 34.2%. The page says so rather than blurring the two.

Run it (WSL, setu-venv, with the Palmira worker running in its own
conda env for the real path):

    export PYTHONPATH=src && python -m setu.demo.app
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr  # noqa: E402

from setu.demo import pipeline as P  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ALLOW_LIVE = True  # set False to refuse anything not already cached

CSS = """
.setu-note {border-left:3px solid #3c4a7a; padding:10px 14px; background:#fffdf8;
            font-size:.88rem; line-height:1.6; border-radius:4px}
.kn {font-family:"Noto Serif Kannada","Nirmala UI",serif; font-size:1.05rem; line-height:1.95}
footer {display:none !important}
"""


# ------------------------------------------------------------------ readers

def make_reader():
    """Vision-model reader for the real path: N independent reads."""
    from setu.demo.build_demo_real import sample_readings
    from setu.label.vlm_label import KeyRing, collect_keys, load_env

    ring = KeyRing(collect_keys(load_env(REPO_ROOT / ".env"), "GEMINI_API_KEY"))

    def read(path: Path, n: int, temperature: float) -> list[str]:
        png = Path(path).read_bytes()
        return [s["text"] for s in
                sample_readings(ring, READ_MODEL, png, n, temperature)]

    return read, ring


def make_modernizer(ring):
    from setu.demo.modernize_llm import modernize

    def fn(text: str, branch: str) -> str:
        try:
            return modernize(text, branch, MODERN_MODEL,
                             REPO_ROOT / "data" / "demo_cache", ring=ring)["modern"]
        except Exception as e:
            return f"[modernizer unavailable: {type(e).__name__}]"

    return fn


READ_MODEL = "gemini-3.5-flash"
MODERN_MODEL = "gemini-3.1-flash-lite"


# ------------------------------------------------------------------ renderer

def _branch_md(line: dict) -> str:
    differ = line.get("branches_differ")
    head = {True: "**the two branches disagree**", False: "both branches agree",
            None: "modernizer not run"}[differ]
    rivals = "".join(
        f"\n  - `{r['prob']:.2f}`  <span class='kn'>{r['text']}</span>"
        for r in (line.get("rivals") or [])[1:5])
    unc = ", ".join(
        f"<span class='kn'>{u['alternatives'][0]['kannada'] if u.get('alternatives') else u['chosen']}</span>"
        f" ({u['top1']:.2f})" for u in (line.get("uncertain") or [])[:8])
    return f"""
#### Line {line['index'] + 1} — {line['n_uncertain']}/{line['n_chars']} characters uncertain — {head}

| | |
|---|---|
| **B3** reading (argmax commits) | <span class='kn'>{line['b3_text'] or '—'}</span> |
| **B3** modernized | <span class='kn'>{line.get('b3_modern') or '—'}</span> |
| **B4** modernized (uncertainty carried) | <span class='kn'>{line.get('b4_modern') or '—'}</span> |

*Readings B4 also kept:*{rivals or ' none'}

*Uncertain characters:* {unc or 'none'}
"""


def render(result: dict) -> str:
    if result.get("error"):
        return f"### Could not process this image\n\n```\n{result['error']}\n```"
    cache = "replayed from cache (no API calls)" if result.get("from_cache") else "computed now"
    md = [f"## Route: **{result['route']}** — {cache}",
          "**Steps:** " + " → ".join(result["steps"]),
          f"<div class='setu-note'>{result['uncertainty_source']}</div>"]
    pal = result.get("palmira")
    if pal:
        tags = ", ".join(f"{k} ×{v}" for k, v in sorted((pal.get("tags") or {}).items()))
        md.append(f"**Palmira tagged:** {tags or 'nothing'} — "
                  f"**{pal.get('n_lines')}** character lines found")
    for line in result["lines"]:
        md.append(_branch_md(line))
    return "\n\n".join(md)


def overlay_of(result: dict):
    pal = result.get("palmira") or {}
    p = pal.get("overlay")
    return p if p and Path(p).exists() else None


def crops_of(result: dict):
    out = []
    for line in result.get("lines", []):
        for k, tag in (("crop", "crop"), ("binarized", "binarized")):
            p = line.get(k)
            if p and Path(p).exists():
                out.append((p, f"line {line['index'] + 1} — {tag}"))
    return out


# --------------------------------------------------------------------- app

def process(image, route_choice, samples, temperature, max_lines, do_modernize,
            progress=gr.Progress()):
    if image is None:
        return "Upload an image to begin.", None, []
    path = Path(image)

    auto, why = P.classify_input(path)
    route = auto if route_choice == "auto" else route_choice
    note = f"detected **{auto}** ({why})" + ("" if route == auto else f" — overridden to **{route}**")

    key_probe = P.image_key(path, path=route, n=int(samples), t=float(temperature),
                            m=int(max_lines), read=True, modern=bool(do_modernize)) \
        if route == "real" else \
        P.image_key(path, path="synthetic", t=1.5, k=5, u=0.9, modern=bool(do_modernize))
    is_cached = P.cached(key_probe) is not None
    if not is_cached and not ALLOW_LIVE:
        return ("This image is not cached and live calls are disabled. "
                "Pre-generate it, or enable live calls."), None, []

    def tick(msg):
        progress(0.5, desc=msg)

    ring = None
    read_fn = modern_fn = None
    if not is_cached:
        if route == "real":
            read_fn, ring = make_reader()
        else:
            from setu.label.vlm_label import KeyRing, collect_keys, load_env
            ring = KeyRing(collect_keys(load_env(REPO_ROOT / ".env"), "GEMINI_API_KEY"))
        if do_modernize:
            modern_fn = make_modernizer(ring)

    if route == "real":
        result = P.run_real(path, samples=int(samples), temperature=float(temperature),
                            max_lines=int(max_lines), read_fn=read_fn,
                            modernize_fn=modern_fn, progress=tick)
    else:
        result = P.run_synthetic(path, modernize_fn=modern_fn, progress=tick)

    return f"*{note}*\n\n" + render(result), overlay_of(result), crops_of(result)


def build() -> gr.Blocks:
    # Gradio 6 moved css from Blocks() to launch().
    with gr.Blocks(title="SETU — palm-leaf reading pipeline") as app:
        gr.Markdown(
            "# SETU — palm-leaf reading pipeline\n"
            "Upload a **page photograph** to run Palmira segmentation → line crops → "
            "binarization → vision-model ensemble → modernizer, or a **rendered line** to run "
            "the CRNN → soft bridge → modernizer. The route is chosen from the image and can "
            "be overridden.\n\n"
            "Both routes compare **B3** (argmax commits to one reading) against **B4** (the "
            "uncertainty is carried forward). They differ in where the uncertainty comes from, "
            "and the page states which is which — the CTC soft bridge is the project's own "
            "contribution; the ensemble on real crops is a weaker stand-in for it."
        )
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="Manuscript image", height=260)
                route = gr.Radio(["auto", "real", "synthetic"], value="auto", label="Pipeline")
                modern = gr.Checkbox(value=True, label="Run the modernizer (B3 vs B4)")
                with gr.Accordion("Real-path settings", open=False):
                    samples = gr.Slider(2, 7, value=5, step=1, label="Reads per line (ensemble size)")
                    temp = gr.Slider(0.0, 1.2, value=0.8, step=0.1, label="Reading temperature")
                    maxl = gr.Slider(1, 12, value=4, step=1, label="Lines to process")
                go = gr.Button("Run pipeline", variant="primary")
                gr.Markdown(
                    "<div class='setu-note'>A new image calls the vision model live; the free "
                    "tier allows <b>20 requests per key per day per model</b>, so a page at 5 "
                    "reads × 4 lines costs 20 of them. Anything processed before replays from "
                    "cache and spends nothing.</div>")
            with gr.Column(scale=2):
                out_md = gr.Markdown("Upload an image and press **Run pipeline**.")
                out_overlay = gr.Image(label="Palmira segmentation", height=260)
                out_crops = gr.Gallery(label="Line crops and binarized versions",
                                       columns=2, height=300)
        go.click(process, [img, route, samples, temp, maxl, modern],
                 [out_md, out_overlay, out_crops])
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-live", action="store_true",
                    help="refuse anything not already cached (safest for a graded demo)")
    args = ap.parse_args()
    global ALLOW_LIVE
    ALLOW_LIVE = not args.no_live
    print(f"Palmira worker: {'running' if P.worker_alive() else 'NOT running (real path will fail)'}")
    build().launch(server_port=args.port, share=args.share, inbrowser=False, css=CSS)


if __name__ == "__main__":
    main()

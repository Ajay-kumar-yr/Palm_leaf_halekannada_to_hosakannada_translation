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
crops our recogniser cannot read at all (0.696 CER), so the same
principle is shown with an ensemble over a borrowed reader, which
recovers 26.0%. The page says so rather than blurring the two.

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

# Theme taken from the Kannada Chandas Identifier, so the two projects
# read as one body of work: parchment ground, stone ink, amber accent,
# and the 8px amber gradient rule across the top of the page. Its own
# tokens, for reference:
#   parchment #fbf7f1   ink/stone-800 #292524   gold/amber-600 #d97706
#   amber-700 #b45309   amber-50 #fffbeb        amber-200 #fde68a
#   stone-200 #e7e5e4   stone-500 #78716c       red-800 #991b1b
# Fonts: Noto Sans Kannada for text, Tiro Kannada for display.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Kannada:wght@300;400;500;700&family=Tiro+Kannada:ital@0;1&display=swap');

:root {
  --setu-parchment:#fbf7f1; --setu-ink:#292524; --setu-gold:#d97706;
  --setu-amber-700:#b45309; --setu-amber-200:#fde68a; --setu-amber-50:#fffbeb;
  --setu-stone-200:#e7e5e4; --setu-stone-500:#78716c; --setu-red:#991b1b;
}

/* The page itself, and the gradient rule that identifies the family. */
gradio-app, .gradio-container, body {
  background:var(--setu-parchment) !important;
  color:var(--setu-ink) !important;
  font-family:"Noto Sans Kannada","Segoe UI",sans-serif !important;
}
.gradio-container {max-width:1180px !important; margin:0 auto !important;}
gradio-app::before {
  content:""; display:block; height:8px; width:100%;
  background:linear-gradient(90deg,var(--setu-amber-700),#f59e0b,var(--setu-amber-700));
}
::selection {background:var(--setu-amber-200);}
::-webkit-scrollbar {width:8px;}
::-webkit-scrollbar-track {background:#f1f1f1;}
::-webkit-scrollbar-thumb {background:#d6d3d1; border-radius:4px;}

/* Display type in the serif, as on the reference site. */
h1, h2 {font-family:"Tiro Kannada",Georgia,serif !important; color:#292524 !important;
        letter-spacing:-.01em;}
h1 {border-bottom:1px solid var(--setu-stone-200); padding-bottom:.4rem;}
h3, h4 {color:var(--setu-amber-700) !important; font-weight:700 !important;}
a {color:var(--setu-amber-700) !important;}

/* Cards: white on parchment, hairline stone border, soft corners. */
.block, .form, .panel, .gr-box, .gr-group {
  background:#fff !important; border:1px solid var(--setu-stone-200) !important;
  border-radius:.5rem !important;
}
.gr-panel, .tabs {box-shadow:0 1px 2px rgba(41,37,36,.05);}

/* Primary action in gold; the reference uses amber-600 with white text. */
button.primary, .gr-button-primary, button[variant="primary"] {
  background:var(--setu-gold) !important; border:1px solid var(--setu-amber-700) !important;
  color:#fff !important; font-weight:700 !important; border-radius:.5rem !important;
}
button.primary:hover, .gr-button-primary:hover {background:var(--setu-amber-700) !important;}
button.secondary, .gr-button-secondary {
  background:var(--setu-amber-50) !important; color:var(--setu-amber-700) !important;
  border:1px solid var(--setu-amber-200) !important; border-radius:.5rem !important;
}

/* Tables carry the B3-vs-B4 comparison, so they get the most attention. */
.prose table, table {border-collapse:collapse !important; width:100%;}
.prose th, .prose td, th, td {
  border:1px solid var(--setu-stone-200) !important; padding:.5rem .7rem !important;
  vertical-align:top;
}
.prose th, th {background:var(--setu-amber-50) !important; color:var(--setu-amber-700) !important;
               font-weight:700 !important; text-align:left !important;}
.prose tr:nth-child(even) td {background:#fafaf9 !important;}

/* The disclosure box: amber rule, parchment fill -- it says "read this". */
.setu-note {
  border-left:4px solid var(--setu-gold); padding:11px 14px;
  background:var(--setu-amber-50); color:#44403c;
  font-size:.88rem; line-height:1.6; border-radius:.5rem;
  border-top:1px solid var(--setu-stone-200);
  border-right:1px solid var(--setu-stone-200);
  border-bottom:1px solid var(--setu-stone-200);
}
.setu-note b {color:var(--setu-red);}

/* Kannada script: the reference pairs Tiro Kannada with generous leading. */
.kn {font-family:"Tiro Kannada","Noto Serif Kannada","Nirmala UI",serif;
     font-size:1.1rem; line-height:2.0; color:#1c1917;}
code, .font-mono {font-family:"IBM Plex Mono",Consolas,monospace !important;
                  background:#f5f5f4; padding:.1rem .3rem; border-radius:.25rem;}

/* Dark mode: force the same light palette. Gradio adds .dark from the
   browser's preference, and a half-themed dark page is what made the
   text invisible -- white type on parchment. Everything below states a
   colour explicitly rather than inheriting. */
.dark, .dark gradio-app, .dark .gradio-container, gradio-app.dark {
  background:var(--setu-parchment) !important; color:var(--setu-ink) !important;
}
.dark .block, .dark .form, .dark .panel, .dark .gr-box, .dark .gr-group,
.dark input, .dark textarea, .dark select {
  background:#fff !important; color:var(--setu-ink) !important;
  border-color:var(--setu-stone-200) !important;
}
.dark .prose, .dark .prose *, .dark label, .dark .gr-text, .dark span, .dark p,
.dark li, .dark td, .dark h1, .dark h2, .dark h3, .dark h4 {
  color:var(--setu-ink) !important;
}
.dark .prose th, .dark th {color:var(--setu-amber-700) !important;}

/* Labels and body copy, in either mode. The component label pills were
   white-on-black; make them amber-on-cream like the reference's chips. */
.prose, .prose p, .prose li, .prose td, label, .gr-text,
.markdown, .markdown p, .markdown li {color:var(--setu-ink) !important;}
.block-label, .label-wrap, .gr-block-label, span[data-testid="block-label"] {
  background:var(--setu-amber-50) !important; color:var(--setu-amber-700) !important;
  border:1px solid var(--setu-amber-200) !important; border-radius:.35rem !important;
  font-weight:600 !important;
}
.block-label svg, .label-wrap svg {color:var(--setu-amber-700) !important;
                                   fill:var(--setu-amber-700) !important;}
input, textarea, select {color:var(--setu-ink) !important; background:#fff !important;}
::placeholder {color:#a8a29e !important;}
.wrap.default, .empty, .svelte-1ipelgc {color:var(--setu-stone-500) !important;}

/* Branch diff. The reference site marks laghu green and guru red with a
   matching border; the same two badges mark which branch a word came
   from, so the colours mean the same thing across both projects. */
.b3-differs {background:#fee2e2; color:#991b1b; border:1px solid #fca5a5;
             border-radius:.25rem; padding:0 .22em;}
.b4-differs {background:#dcfce7; color:#15803d; border:1px solid #86efac;
             border-radius:.25rem; padding:0 .22em;}
.setu-legend {font-size:.8rem; color:var(--setu-stone-500); margin:.2rem 0 .5rem;}
.setu-legend .b3-differs, .setu-legend .b4-differs {font-weight:600; margin-right:.15rem;}

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

def _diff_words(old: str, new: str) -> tuple[str, str]:
    """Mark, word by word, where `new` departs from `old`.

    Returns (old_html, new_html): the words only `old` has are marked in
    red, the words only `new` has in green, so the two rows can be read
    against each other at a glance.

    Word-level, not character-level, and on whitespace, which is right
    for Kannada: the script is written with spaces between words, and a
    single changed consonant usually changes the whole word's meaning --
    which is the thing worth pointing at.

    **This marks difference, not correctness.** At demo time there is no
    ground truth in the pipeline, so a green word is what B4 produced,
    not proof that B4 is right. The two staged recovery lines are the
    cases where the gold IS known, and `demo_inputs/README.md` names
    them.
    """
    import difflib

    a, b = (old or "").split(), (new or "").split()
    if not a or not b:
        return (old or "—"), (new or "—")
    out_a, out_b = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "equal":
            out_a += a[i1:i2]
            out_b += b[j1:j2]
        else:
            if i1 != i2:
                out_a.append("<span class='b3-differs'>" + " ".join(a[i1:i2]) + "</span>")
            if j1 != j2:
                out_b.append("<span class='b4-differs'>" + " ".join(b[j1:j2]) + "</span>")
    return " ".join(out_a), " ".join(out_b)


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
    # The reading: argmax's string against the strongest reading B4 also
    # carried, so the contested character is visible in its word.
    alt = (line.get("rivals") or [{}])[1:2]
    read_b3, read_b4 = _diff_words(line["b3_text"] or "",
                                   (alt[0].get("text") if alt else "") or "")
    # The modern text: this is the one that matters, because a difference
    # here is a difference in the project's actual output.
    mod_b3, mod_b4 = _diff_words(line.get("b3_modern") or "", line.get("b4_modern") or "")

    legend = ("<div class='setu-legend'>"
              "<span class='b3-differs'>B3 only</span> "
              "<span class='b4-differs'>B4 only</span> "
              "&nbsp;marks where the branches differ. B4 differing is not by itself "
              "proof B4 is right — the pipeline has no ground truth at demo time."
              "</div>") if differ else ""

    return f"""
#### Line {line['index'] + 1} — {line['n_uncertain']}/{line['n_chars']} characters uncertain — {head}

{legend}

| | |
|---|---|
| **B3** reading (argmax commits) | <span class='kn'>{read_b3 or '—'}</span> |
| **B4** also carried | <span class='kn'>{read_b4 or '—'}</span> |
| **B3** modernized | <span class='kn'>{mod_b3 or '—'}</span> |
| **B4** modernized (uncertainty carried) | <span class='kn'>{mod_b4 or '—'}</span> |

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

def process(picked, image, route_choice, samples, temperature, max_lines, do_modernize,
            progress=gr.Progress()):
    # The picked file takes precedence and is used by its real path, so
    # the cache key matches what was pre-cached. An upload is a copy
    # Gradio re-encoded, and is only usable with live calls enabled.
    source = picked or image
    if not source:
        return "Pick a demo file, or upload an image, then press Run pipeline.", None, []
    path = Path(source)

    auto, why = P.classify_input(path)
    route = auto if route_choice == "auto" else route_choice
    note = f"detected **{auto}** ({why})" + ("" if route == auto else f" — overridden to **{route}**")

    # Same key the run functions store under -- never re-derived here,
    # or the --no-live gate drifts out of agreement with the cache and
    # refuses images that are sitting in it.
    key_probe = (P.real_key(path, samples, temperature, max_lines,
                            read=True, modern=bool(do_modernize))
                 if route == "real" else
                 P.synthetic_key(path, modern=bool(do_modernize)))
    is_cached = P.cached(key_probe) is not None
    if not is_cached and not ALLOW_LIVE:
        return ("This image is not cached and live calls are disabled. "
                "Pre-generate it, or enable live calls."), None, []

    def tick(msg):
        progress(0.5, desc=msg)

    # Build these ALWAYS, cached or not. They are closures -- constructing
    # one reads .env and makes no API call -- and the cache key records
    # whether they were supplied: `read=read_fn is not None`. Skipping them
    # on a cache hit therefore made `run_real` look up a *different* key
    # than the gate just probed, and replay a readings-free entry left over
    # from a segmentation-only run: every line "0/0 characters uncertain",
    # no modernizer, nothing to show. The gate decides whether a call is
    # allowed; it must not also change the key.
    ring = None
    read_fn = modern_fn = None
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


def make_theme():
    """Gradio draws many controls from its own theme tokens, so the palette
    has to be set here as well -- CSS alone leaves inputs and sliders blue.
    Gradio 6 takes `theme` at launch(), not on Blocks(), exactly like `css`;
    passing it to Blocks() is accepted with a warning and then ignored."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.GoogleFont("Noto Sans Kannada"), "Segoe UI", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "Consolas", "monospace"],
    ).set(
        # EVERY token needs its _dark twin. Gradio picks the dark set
        # from the browser's colour-scheme preference, and setting only
        # the light ones left white text and black label pills sitting on
        # the parchment this CSS forces -- the page was unreadable. The
        # demo is a light page in both modes, deliberately: it is shown
        # on a projector, not chosen by a visitor.
        body_background_fill="#fbf7f1", body_background_fill_dark="#fbf7f1",
        body_text_color="#292524", body_text_color_dark="#292524",
        body_text_color_subdued="#78716c", body_text_color_subdued_dark="#78716c",
        block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
        block_border_color="#e7e5e4", block_border_color_dark="#e7e5e4",
        block_label_background_fill="#fffbeb", block_label_background_fill_dark="#fffbeb",
        block_label_text_color="#b45309", block_label_text_color_dark="#b45309",
        block_title_text_color="#b45309", block_title_text_color_dark="#b45309",
        block_info_text_color="#78716c", block_info_text_color_dark="#78716c",
        panel_background_fill="#ffffff", panel_background_fill_dark="#ffffff",
        background_fill_primary="#ffffff", background_fill_primary_dark="#ffffff",
        background_fill_secondary="#fafaf9", background_fill_secondary_dark="#fafaf9",
        border_color_primary="#e7e5e4", border_color_primary_dark="#e7e5e4",
        border_color_accent="#fde68a", border_color_accent_dark="#fde68a",
        button_primary_background_fill="#d97706", button_primary_background_fill_dark="#d97706",
        button_primary_background_fill_hover="#b45309",
        button_primary_background_fill_hover_dark="#b45309",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        button_primary_border_color="#b45309", button_primary_border_color_dark="#b45309",
        button_secondary_background_fill="#fffbeb",
        button_secondary_background_fill_dark="#fffbeb",
        button_secondary_text_color="#b45309", button_secondary_text_color_dark="#b45309",
        input_background_fill="#ffffff", input_background_fill_dark="#ffffff",
        input_border_color="#e7e5e4", input_border_color_dark="#e7e5e4",
        input_placeholder_color="#a8a29e", input_placeholder_color_dark="#a8a29e",
        link_text_color="#b45309", link_text_color_dark="#b45309",
        checkbox_background_color="#ffffff", checkbox_background_color_dark="#ffffff",
        checkbox_background_color_selected="#d97706",
        checkbox_background_color_selected_dark="#d97706",
        checkbox_label_background_fill="#fafaf9",
        checkbox_label_background_fill_dark="#fafaf9",
        checkbox_label_text_color="#292524", checkbox_label_text_color_dark="#292524",
        table_text_color="#292524", table_text_color_dark="#292524",
    )


def demo_choices() -> list[tuple[str, str]]:
    """(label, path) for every staged demo file, in show order."""
    root = REPO_ROOT / "demo_inputs"
    out = []
    for folder in sorted(root.glob("0*")):
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                out.append((f"{folder.name[:2]} · {f.stem[:58]}", str(f)))
    return out


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
                # Pick a staged file rather than uploading it. An upload is
                # re-encoded by gr.Image to lossy webp, so the bytes -- and
                # therefore the cache key -- depend on how the file arrived;
                # pre-cached lines were refused in the browser under
                # --no-live for exactly that reason. A picked file is read
                # from disk unchanged, so it always hits.
                picked = gr.Dropdown(choices=demo_choices(), value=None,
                                     label="Demo file (pre-cached — use this)",
                                     info="From demo_inputs/. Replays offline.")
                img = gr.Image(type="filepath", height=260,
                               label="…or upload your own (needs live calls)")
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
        go.click(process, [picked, img, route, samples, temp, maxl, modern],
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
    build().launch(server_port=args.port, share=args.share, inbrowser=False,
                   css=CSS, theme=make_theme())


if __name__ == "__main__":
    main()

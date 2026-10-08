"""Render a demo_build run as one self-contained HTML page.

Everything is inlined — line images as data URIs, CSS, no scripts from
anywhere — so the page opens from a file, over a share link, or on a
machine with no Python, and needs no network at showtime (DEMO_PLAN.md).

Usage:
    python -m setu.demo.make_viewer --run runs/<ts>_demo_build
    python -m setu.demo.make_viewer --run ... --out demo.html --max-lines 10
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

PAGE_CSS = """
/* Layout: a conservator's line ledger -- one row per manuscript line,
   the leaf image full width, the two branches compared beneath it. */
:root {
  --ink:        #2b2317;   /* iron-gall brown, the text colour */
  --ink-soft:   #6b6052;
  --leaf:       #f6f3ec;   /* page ground */
  --leaf-edge:  #e4ded1;
  --card:       #fffdf8;
  --indigo:     #3c4a7a;   /* the accent: a manuscript pigment, not a UI blue */
  --indigo-dim: #8089ab;
  --keep:       #1f6b4f;   /* what the bridge kept */
  --drop:       #9a3d2e;   /* what argmax discarded */
  --font-ui: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  --font-kn: "Noto Serif Kannada", "Nirmala UI", serif;
  --font-mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #e8e2d6; --ink-soft: #9f978a; --leaf: #17150f; --leaf-edge: #322d23;
    --card: #211d16; --indigo: #9aa6d8; --indigo-dim: #5e678c;
    --keep: #63c49b; --drop: #e08a76; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --ink: #e8e2d6; --ink-soft: #9f978a; --leaf: #17150f; --leaf-edge: #322d23;
  --card: #211d16; --indigo: #9aa6d8; --indigo-dim: #5e678c;
  --keep: #63c49b; --drop: #e08a76; color-scheme: dark;
}
body { background: var(--leaf); color: var(--ink); font-family: var(--font-ui); }
.wrap { max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 32px 64px; }
h1 { font-size: clamp(1.5rem, 1.1rem + 1.6vw, 2.1rem); font-weight: 600; margin: 0 0 6px;
     letter-spacing: -0.015em; text-wrap: balance; }
.sub { color: var(--ink-soft); max-width: 62ch; line-height: 1.55; margin: 0 0 24px; }
.note { border-left: 3px solid var(--indigo); background: var(--card); padding: 12px 16px;
        margin: 0 0 28px; font-size: 0.86rem; line-height: 1.6; color: var(--ink-soft); }
.note strong { color: var(--ink); font-weight: 600; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
         gap: 1px; background: var(--leaf-edge); border: 1px solid var(--leaf-edge);
         border-radius: 6px; overflow: hidden; margin-bottom: 36px; }
.stat { background: var(--card); padding: 14px 16px; min-width: 0; }
.stat .v { font-family: var(--font-mono); font-size: 1.3rem; font-variant-numeric: tabular-nums;
           display: block; letter-spacing: -0.02em; }
.stat .k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em;
           color: var(--ink-soft); display: block; margin-top: 3px; }
.line { border: 1px solid var(--leaf-edge); border-radius: 8px; background: var(--card);
        margin-bottom: 22px; overflow: hidden; }
.line > header { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline;
                 padding: 10px 16px; border-bottom: 1px solid var(--leaf-edge); }
.line h2 { font-size: 0.82rem; font-weight: 600; margin: 0; font-family: var(--font-mono);
           letter-spacing: 0; }
.tag { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em;
       padding: 2px 7px; border-radius: 3px; border: 1px solid currentColor; }
.tag.held { color: var(--keep); }
.tag.seen { color: var(--ink-soft); }
.tag.differ { color: var(--indigo); border-color: var(--indigo); }
.strip { padding: 14px 16px; background: var(--leaf); border-bottom: 1px solid var(--leaf-edge);
         overflow-x: auto; }
.strip img { display: block; max-width: 100%; height: auto; min-width: 320px;
             border-radius: 3px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--leaf-edge); }
@media (max-width: 720px) { .cols { grid-template-columns: 1fr; } }
.col { background: var(--card); padding: 14px 16px; min-width: 0; }
.col h3 { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
          color: var(--ink-soft); margin: 0 0 8px; font-weight: 600; }
.col.bridge h3 { color: var(--indigo); }
.kn { font-family: var(--font-kn); font-size: 1.04rem; line-height: 1.95; margin: 0;
      word-break: break-word; }
.kn.modern { border-top: 1px dashed var(--leaf-edge); margin-top: 10px; padding-top: 10px; }
.lbl { font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.07em;
       color: var(--ink-soft); display: block; margin-bottom: 2px; }
.rivals { list-style: none; margin: 8px 0 0; padding: 0; }
.rivals li { display: flex; gap: 8px; align-items: baseline; padding: 3px 0;
             border-top: 1px dotted var(--leaf-edge); }
.rivals .p { font-family: var(--font-mono); font-size: 0.74rem; color: var(--keep);
             font-variant-numeric: tabular-nums; flex: 0 0 auto; }
.rivals .t { font-family: var(--font-kn); font-size: 0.92rem; line-height: 1.8;
             min-width: 0; word-break: break-word; }
.chars { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.chip { border: 1px solid var(--leaf-edge); border-radius: 4px; padding: 4px 7px;
        font-size: 0.74rem; display: flex; gap: 5px; align-items: baseline; }
.chip .c { font-family: var(--font-kn); font-size: 0.95rem; }
.chip .alt { color: var(--drop); font-family: var(--font-kn); }
.chip .pp { font-family: var(--font-mono); font-size: 0.68rem; color: var(--ink-soft);
            font-variant-numeric: tabular-nums; }
.truth { padding: 10px 16px; border-top: 1px solid var(--leaf-edge); font-size: 0.84rem; }
footer { margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--leaf-edge);
         color: var(--ink-soft); font-size: 0.78rem; line-height: 1.7; }
"""


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def render_line(e: dict, show_truth: bool) -> str:
    img = REPO_ROOT / e["image"]
    tags = [f'<span class="tag {"seen" if e["trained"] else "held"}">'
            f'{"seen in training" if e["trained"] else "held out"}</span>']
    if e.get("branches_differ"):
        tags.append('<span class="tag differ">branches differ</span>')
    u = e["uncertainty"]

    rivals = "".join(
        f'<li><span class="p">{r["prob"]:.2f}</span><span class="t">{esc(r["text"])}</span></li>'
        for r in e.get("rival_readings", [])[1:6]
    )
    chips = "".join(
        f'<span class="chip"><span class="c">{esc(c["alternatives"][0]["kannada"])}</span>'
        f'<span class="pp">{c["top1"]:.2f}</span>'
        + "".join(f'<span class="alt">{esc(a["kannada"])}</span>'
                  f'<span class="pp">{a["prob"]:.2f}</span>'
                  for a in c["alternatives"][1:3])
        + "</span>"
        for c in e.get("uncertain_chars", [])[:10]
    )

    b3_modern = (f'<p class="kn modern"><span class="lbl">modernized</span>{esc(e["b3_modern"])}</p>'
                 if e.get("b3_modern") else "")
    b4_modern = (f'<p class="kn modern"><span class="lbl">modernized</span>{esc(e["b4_modern"])}</p>'
                 if e.get("b4_modern") else "")
    truth = (f'<div class="truth"><span class="lbl">machine label (not human ground truth)'
             f'</span><span class="kn">{esc(e["label_text"])}</span></div>'
             if show_truth and e.get("label_text") else "")

    return f"""
<article class="line">
  <header><h2>{esc(e["crop"])}</h2>{"".join(tags)}
    <span class="tag seen">{u["n_uncertain"]}/{u["n_chars"]} chars uncertain</span></header>
  <div class="strip"><img src="{data_uri(img)}" alt="manuscript line {esc(e['crop'])}"></div>
  <div class="cols">
    <div class="col">
      <h3>B3 &middot; argmax</h3>
      <p class="kn"><span class="lbl">reading</span>{esc(e["b3_text"])}</p>
      {b3_modern}
    </div>
    <div class="col bridge">
      <h3>B4 &middot; soft bridge</h3>
      <p class="kn"><span class="lbl">reading</span>{esc(e["b3_text"])}</p>
      {("<span class='lbl'>readings it also kept</span><ul class='rivals'>" + rivals + "</ul>") if rivals else ""}
      {("<span class='lbl'>uncertain characters</span><div class='chars'>" + chips + "</div>") if chips else ""}
      {b4_modern}
    </div>
  </div>
  {truth}
</article>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="a runs/<ts>_demo_build folder")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--held-out-only", action="store_true")
    p.add_argument("--hide-truth", action="store_true", help="omit the machine labels")
    p.add_argument("--title", default="Palm Leaf Line Readings")
    args = p.parse_args()

    entries = json.loads((args.run / "demo.json").read_text(encoding="utf-8"))
    results = json.loads((args.run / "results.json").read_text(encoding="utf-8"))
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    if args.held_out_only:
        entries = [e for e in entries if not e["trained"]]
    if args.max_lines:
        entries = entries[: args.max_lines]
    if not entries:
        raise SystemExit("no entries to render")

    n_differ = results.get("n_branches_differ")
    cer = results.get("mean_cer_vs_machine_label")
    stats = [
        (f'{len(entries)}', "lines shown"),
        (f'{sum(1 for e in entries if not e["trained"])}', "held out from training"),
        (f'{results["mean_fraction_uncertain"]:.0%}', "characters flagged uncertain"),
        (f'{cer:.0%}' if cer is not None else "n/a", "CER vs machine label"),
    ]
    if n_differ is not None:
        stats.append((f"{n_differ}", "lines where branches differ"))

    body = f"""<title>{esc(args.title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400&family=Noto+Serif+Kannada:wght@400;600&display=swap">
<style>{PAGE_CSS}</style>
<div class="wrap">
<h1>{esc(args.title)}</h1>
<p class="sub">Real Hampi palm-leaf lines, segmented with Palmira and read by the project's
CRNN. Each line is shown twice: once as <strong>argmax</strong> commits to it, and once as the
<strong>soft bridge</strong> passes it on, carrying the readings the recogniser was still
weighing.</p>

<div class="note">
<strong>What this does and does not show.</strong> The recogniser was adapted on other lines of
these same pages, so this demonstrates adaptation to a known hand, not generalisation to unseen
manuscripts &mdash; on held-out <em>pages</em> it reads at 0.71 CER. Lines marked
&ldquo;held out&rdquo; were kept out of training; lines marked &ldquo;seen in training&rdquo; were
not. Accuracy is measured against <strong>machine labels from a vision model, not human ground
truth</strong>. The modernizer here is an LLM, not the project's own model, which is a documented
negative result. The quantified bridge claim is the frame-level one: on frozen-test frames where
the top-1 was wrong, the correct symbol was still in the top-5 <strong>80.6%</strong> of the time.
</div>

<div class="stats">{"".join(f'<div class="stat"><span class="v">{esc(v)}</span><span class="k">{esc(k)}</span></div>' for v, k in stats)}</div>

{"".join(render_line(e, not args.hide_truth) for e in entries)}

<footer>
Checkpoint <code>{esc(Path(config["checkpoint"]).parent.name)}</code> &middot;
bridge temperature {esc(config["temperature"])} &middot;
flagged below {esc(config["uncertain_below"])} top-1 &middot;
modernizer {esc(config.get("llm_model") or "not run")}.
Curated for the demo and never used in a reported number.
</footer>
</div>"""

    out = args.out or (args.run / "viewer.html")
    out.write_text(body, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"{len(entries)} lines -> {out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()

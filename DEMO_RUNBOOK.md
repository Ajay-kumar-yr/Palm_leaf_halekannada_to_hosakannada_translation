# Running the demo

Two processes, two Python environments. Start them in this order.
Laptop only — it needs WSL, the GPU, and both conda environments.

## 1. Palmira worker (conda env, Python 3.7)

Only the **real** path needs it. It loads the model once and then
watches a directory, because loading costs ~20 s and paying that per
upload makes the demo look broken.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate palmira
cd ~/Palmira                       # it needs configs/, pretrained/, predictor.py here
python -u /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/data/raw_corpus/palmira_worker.py
```

**Wait for `ready (pid …)` before starting the app.** Palmira takes
~40 s to load, and the app prints its worker banner once, at startup:
start them together and it says `NOT running` for the rest of the
session even though the real route works fine. The banner is a snapshot,
not a status line — `worker_alive()` is re-checked per job.

The app verifies `data/demo_jobs/worker_ready` **and that the pid in it
is alive**. Nothing deletes that file when the worker dies, so before
this check a worker killed yesterday was still reported as running, and
the real route would accept a job, wait out its 180 s timeout and fail
— after telling you the worker was up. If you kill the worker by hand,
delete `data/demo_jobs/worker_ready` too.

Two launcher scripts in the repo root do all of this, and matter for a
reason that is not obvious: **the worker and app must not be detached**
with `nohup`/`setsid`. When the launching command exits and nothing else
is running in the distro, WSL shuts the whole instance down a few
seconds later and takes the process with it — a 0-byte log and no
process, which looks exactly like a crash.

```bash
bash .launch_worker.sh    # hold this terminal open
bash .launch_app.sh       # and this one
```

## 2. The app (setu-venv, Python 3.14)

```bash
cd /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation
source ~/setu-venv/bin/activate
export PYTHONPATH=src
python -m setu.demo.app                 # http://127.0.0.1:7860
python -m setu.demo.app --no-live       # refuse anything not already cached
```

## Starting it, in short

One command up, Ctrl+C down:

```bash
bash /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/demo.sh --no-live
```

From a Windows terminal rather than a WSL shell, prefix it with
`wsl -d Ubuntu -e`. Drop `--no-live` only if you intend to spend API
quota on something not yet cached.

`demo.sh` starts the Palmira worker, **waits** for it to report ready,
then starts the app, prints the URL, and holds the terminal. Ctrl+C
stops both and deletes the ready file. If either process dies on its
own it says so and takes the other down, rather than leaving half a
demo running. Logs go to `.demo_logs/`.

It exists because three things have already gone wrong by hand:
starting the app before Palmira finished loading (the worker banner
then reads `NOT running` all session); detaching with `nohup`, which
dies when WSL tears the distro down seconds later; and killing the
worker without removing `data/demo_jobs/worker_ready`, after which the
app trusts a corpse and the real route waits out its 180 s timeout.

Then open <http://127.0.0.1:7860> and **pick from the "Demo file"
dropdown** — do not upload the staged files. `gr.Image` re-encodes
uploads to lossy webp (its `format` argument is ignored in Gradio 6),
so an uploaded copy is a different image as far as the cache is
concerned and `--no-live` will refuse it. The dropdown passes the
file's real path, so it always hits. Upload is for ad-hoc images and
needs live calls.

The dropdown is read **at startup**: re-stage `demo_inputs/` and you
must restart.

To start the two halves by hand instead — useful when debugging one of
them — `.launch_worker.sh` and `.launch_app.sh` do exactly one each,
in that order, waiting for `ready (pid …)` in between.

## What it does

Upload an image; the route is chosen from its shape and can be overridden.

| upload | route | steps |
|---|---|---|
| page photograph (≈4:3) | **real** | Palmira → line crops → U-Net binarization → vision-model ensemble (×N) → modernizer |
| rendered line (≥6:1) | **synthetic** | CRNN → per-frame CTC posterior → soft bridge → modernizer |
| real line crop (≥6:1) | routed to synthetic — **override to real**, which skips Palmira | |

Both routes end in the same comparison: **B3**, where argmax commits to
one reading, against **B4**, which carries the alternatives forward.

## Before the demo

- **Pre-run every image you intend to show.** Results are cached by a
  hash of the image plus the settings, so a shown image replays
  instantly and spends no quota. Free tier is **20 requests per key per
  day, per model**; a page at 5 reads × 4 lines costs 20 of them. A
  demo that calls out live can die in front of an examiner.
- Consider `--no-live` once everything is cached: it then cannot make a
  call at all.
- Check the worker is up before you start talking.

## What to say while it runs

The two routes are not the same claim, and the page says so:

- **Synthetic** — our CRNN, our CTC posterior, our soft bridge. This is
  the project's contribution, and where the recogniser actually works
  (1.53% CER). 80.6% of wrong top-1 frames still hold the correct
  symbol in the top-5.
- **Real** — our recogniser cannot read real crops (0.696 CER), so the
  reading is done by a vision model and the uncertainty comes from
  disagreement between repeated reads. Same principle, different
  mechanism, and weaker: 26.0% recovery.

Do not let the real route be described as "our system reads
manuscripts". It does not. What it shows is the pipeline and the
uncertainty mechanism operating on real data, on text that is roughly a
third wrong — and `RESULTS.md` §3 explains exactly why (94% of the
domain gap is letter shape).

## If something breaks

| symptom | cause |
|---|---|
| "Palmira worker is not running" | step 1 not started, or it crashed — check its terminal |
| real route hangs ~180 s then errors | worker died mid-job; restart it |
| `[modernizer unavailable: …]` | quota spent for that model today (resets ~9 h) or no key in `.env` |
| reading is nonsense on a synthetic line | it was resized; S1 images must go in unresized |
| a fix seems to have no effect | stale cache — the key covers the settings, but clear `data/demo_cache/pipeline/` if unsure |

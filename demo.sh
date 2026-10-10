#!/usr/bin/env bash
# SETU demo: one command up, Ctrl+C down.
#
#   bash demo.sh --no-live     # the graded demo: cannot call an API
#   bash demo.sh               # live: uncached images will call out
#
# From a Windows terminal instead of a WSL shell:
#   wsl -d Ubuntu -e bash /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/demo.sh --no-live
#
# Starts the Palmira worker, waits until it is genuinely ready, then
# starts the app, and shuts both down together on Ctrl+C. Three things
# it exists to get right, each of which has already gone wrong once:
#
#   * order -- Palmira takes ~40 s to load and the app prints its worker
#     banner once at startup, so starting them together makes the app
#     claim "NOT running" for the rest of the session;
#   * no detaching -- a nohup'd process dies when WSL tears the distro
#     down a few seconds after the launching command exits, leaving an
#     empty log and no process, which looks exactly like a crash;
#   * the ready file -- nothing else removes it, so a worker killed by
#     any other means leaves the app trusting a corpse, and the real
#     route waits out its full 180 s timeout before failing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$ROOT/.demo_logs"
READY="$ROOT/data/demo_jobs/worker_ready"
WORKER_LOG="$LOGS/worker.log"
APP_LOG="$LOGS/app.log"
PORT="${SETU_PORT:-7860}"
mkdir -p "$LOGS" "$(dirname "$READY")"

worker_pid=""
app_pid=""
stopping=0

say() { printf '\033[1;33m[setu]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setu]\033[0m %s\n' "$*" >&2; cleanup; exit 1; }

cleanup() {
    [ "$stopping" = 1 ] && return
    stopping=1
    echo
    say "stopping…"
    for pid in "$app_pid" "$worker_pid"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null
    done
    # The worker is a child of a subshell, and Palmira ignores a plain
    # TERM while it is inside a CUDA call, so give it a moment and then
    # insist -- otherwise the GPU stays occupied for the next run.
    for _ in 1 2 3 4 5 6; do
        sleep 0.5
        still=0
        for pid in "$app_pid" "$worker_pid"; do
            [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && still=1
        done
        [ "$still" = 0 ] && break
    done
    pkill -f "setu.demo.app" 2>/dev/null
    pkill -f "palmira_worker.py" 2>/dev/null
    sleep 1
    pkill -9 -f "setu.demo.app" 2>/dev/null
    pkill -9 -f "palmira_worker.py" 2>/dev/null
    rm -f "$READY"
    say "stopped. logs kept in .demo_logs/"
}
trap 'cleanup; exit 0' INT TERM

# --- refuse to start on top of something already running -------------
# The bracket keeps the pattern from matching the very shell that is
# running the check: `pgrep -f` searches whole command lines, so a
# plain pattern finds itself and the guard fires on an empty machine.
if pgrep -f "[s]etu\.demo\.app" >/dev/null 2>&1 || pgrep -f "[p]almira_worker\.py" >/dev/null 2>&1; then
    echo "[setu] a worker or app is already running:" >&2
    pgrep -af "[s]etu\.demo\.app|[p]almira_worker\.py" >&2
    echo "[setu] stop it with:  pkill -f setu.demo.app; pkill -f palmira_worker.py" >&2
    exit 1
fi
rm -f "$READY"   # any file here now is stale by definition

# --- 1. Palmira worker ------------------------------------------------
say "starting Palmira worker (loads the model once, ~40 s)…"
(
    # conda's activate.d hooks are not nounset-safe -- binutils' hook
    # reads $ADDR2LINE unconditionally, so under `set -u` the activation
    # aborts and the worker never starts. Relax it just for the
    # activation, which is what conda's own documentation advises.
    set +u
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh" || exit 1
    conda activate palmira || exit 1
    set -u
    cd "$HOME/Palmira" || exit 1       # needs configs/, pretrained/, predictor.py
    exec python -u "$ROOT/data/raw_corpus/palmira_worker.py"
) > "$WORKER_LOG" 2>&1 &
worker_pid=$!

for _ in $(seq 1 240); do
    grep -q "ready (pid" "$WORKER_LOG" 2>/dev/null && break
    grep -qiE "Traceback|ModuleNotFound|CondaError" "$WORKER_LOG" 2>/dev/null \
        && die "worker failed to start — see $WORKER_LOG"
    kill -0 "$worker_pid" 2>/dev/null || die "worker exited — see $WORKER_LOG"
    sleep 1
done
grep -q "ready (pid" "$WORKER_LOG" 2>/dev/null \
    || die "worker did not report ready within 4 minutes — see $WORKER_LOG"
say "worker ready ($(sed -n 's/.*ready (pid \([0-9]*\)).*/pid \1/p' "$WORKER_LOG" | tail -1))"

# --- 2. the app -------------------------------------------------------
say "starting the app…"
(
    cd "$ROOT" || exit 1
    export PYTHONPATH=src
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
    exec "$HOME/setu-venv/bin/python" -u -m setu.demo.app --port "$PORT" "$@"
) > "$APP_LOG" 2>&1 &
app_pid=$!

for _ in $(seq 1 180); do
    grep -q "Running on local URL" "$APP_LOG" 2>/dev/null && break
    grep -qiE "Traceback|Address already in use" "$APP_LOG" 2>/dev/null \
        && die "app failed to start — see $APP_LOG"
    kill -0 "$app_pid" 2>/dev/null || die "app exited — see $APP_LOG"
    sleep 1
done
grep -q "Running on local URL" "$APP_LOG" 2>/dev/null \
    || die "app did not bind within 3 minutes — see $APP_LOG"

case " $* " in
    *" --no-live "*) mode="--no-live: cached results only, no API calls possible" ;;
    *)               mode="LIVE: an uncached image will spend API quota" ;;
esac

cat <<BANNER

  SETU demo is up    http://127.0.0.1:$PORT
  $mode

  Pick from the "Demo file" dropdown — do not upload the staged files.
  Logs: .demo_logs/worker.log, .demo_logs/app.log

  Ctrl+C here stops both.

BANNER

# Hold the terminal. If either process dies on its own, say so and tidy
# up rather than leaving half the demo running.
while true; do
    kill -0 "$worker_pid" 2>/dev/null || { say "the worker exited — see $WORKER_LOG"; cleanup; exit 1; }
    kill -0 "$app_pid"    2>/dev/null || { say "the app exited — see $APP_LOG";      cleanup; exit 1; }
    sleep 2
done

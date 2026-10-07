#!/bin/bash
# Run WebFovea locally: start N Chrome instances with CDP enabled, then run the
# agent with one worker per browser.
#
#   export VLM_API_KEY=sk-ant-...                       # or ANTHROPIC_API_KEY
#   bash scripts/run_local.sh <tasks.json> <output_dir> [N]    # N parallel browsers, default 1, max 8
#   bash scripts/run_local.sh --kill                           # close the browsers this script started
#
# Optional env: MODEL, CHROME (browser binary), PORT_BASE (default 9222), PYTHON,
# WEBFOVEA_VIEWPORT (see scripts/launcher.py).
#
# Model and endpoint come from config.json (api_base / api_model); override the
# model with MODEL=... . The key is read from the environment only and is never
# written to disk or put on a command line.
#
# Browsers that are already running (local or remote) can be used directly with
# scripts/run.sh, which this script calls once the browsers are up.
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
PORT_BASE="${PORT_BASE:-9222}"
PROFILE_PREFIX="${TMPDIR:-/tmp}/webfovea-chrome-"

kill_chromes() {
    pkill -f "user-data-dir=${PROFILE_PREFIX}" 2>/dev/null && echo "closed WebFovea Chrome instances" \
        || echo "no WebFovea Chrome instances running"
}
[ "$1" = "--kill" ] && { kill_chromes; exit 0; }

USAGE="usage: bash scripts/run_local.sh <tasks.json> <output_dir> [N]"
INPUT="${1:?$USAGE}"
OUTPUT="${2:?$USAGE}"
N="${3:-1}"
[ "$N" -ge 1 ] && [ "$N" -le 8 ] || { echo "N must be 1..8"; exit 1; }
[ -f "$INPUT" ] || { echo "task file not found: $INPUT"; exit 1; }

export VLM_API_KEY="${VLM_API_KEY:-$ANTHROPIC_API_KEY}"
[ -n "$VLM_API_KEY" ] || { echo "set VLM_API_KEY (or ANTHROPIC_API_KEY) first"; exit 1; }

# Use Playwright's own Chromium (Chrome for Testing). A regular desktop Chrome
# may refuse browser-context management over CDP, and connect_over_cdp fails.
# Set CHROME=/path/to/chrome to override.
CHROME="${CHROME:-$("$PY" -c 'from playwright.sync_api import sync_playwright
with sync_playwright() as p: print(p.chromium.executable_path)' 2>/dev/null)}"
[ -x "$CHROME" ] || { echo "Chromium not found${CHROME:+ at $CHROME}; run: playwright install chromium"; exit 1; }

echo "== starting $N Chrome instance(s) on ports $PORT_BASE..$((PORT_BASE+N-1)) =="
URLS=()
for i in $(seq 0 $((N-1))); do
    PORT=$((PORT_BASE+i))
    URL="http://localhost:$PORT"
    URLS+=("$URL")
    if curl -s --noproxy '*' --max-time 2 "$URL/json/version" > /dev/null 2>&1; then
        echo "  :$PORT already running, reusing"
        continue
    fi
    # One profile per instance, otherwise the second process attaches to the
    # first and its CDP port never opens.
    # The last three flags matter for parallel runs: overlapping 1920x1080
    # windows occlude each other, Chrome stops producing frames for occluded
    # windows, and page.screenshot() then hangs until it times out.
    nohup "$CHROME" \
        --remote-debugging-port="$PORT" \
        --user-data-dir="${PROFILE_PREFIX}$PORT" \
        --no-first-run --no-default-browser-check \
        --window-size=1920,1080 \
        --disable-backgrounding-occluded-windows \
        --disable-renderer-backgrounding \
        --disable-background-timer-throttling \
        > "${PROFILE_PREFIX}$PORT.log" 2>&1 &
    disown
done

echo "== waiting for CDP =="
for u in "${URLS[@]}"; do
    for _ in $(seq 1 30); do
        curl -s --noproxy '*' --max-time 2 "$u/json/version" > /dev/null 2>&1 && break
        sleep 1
    done
    curl -s --noproxy '*' --max-time 2 "$u/json/version" > /dev/null 2>&1 \
        || { echo "  $u did not come up; see ${PROFILE_PREFIX}*.log"; exit 1; }
    echo "  $u OK"
done

mkdir -p "$OUTPUT"
# Record what was run next to the results, so a run can be attributed later.
{
    echo "date:   $(date '+%Y-%m-%dT%H:%M:%S')"
    echo "input:  $INPUT"
    echo "commit: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo none)$( [ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ] && echo ' (dirty)')"
    echo "model:  ${MODEL:-from config.json}"
    echo "chrome: $CHROME"
} > "$OUTPUT/run_info.txt"
cp "$REPO_DIR/config.json" "$OUTPUT/config_snapshot.json"

echo "== running: $INPUT -> $OUTPUT ($N worker(s)) =="
bash "$REPO_DIR/scripts/run.sh" "$INPUT" "$OUTPUT" "${URLS[@]}"

echo
echo "done. Chrome is still running for the next run; close it with: bash scripts/run_local.sh --kill"

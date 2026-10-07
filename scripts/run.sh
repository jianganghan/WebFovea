#!/bin/bash
# Run WebFovea against browsers that are already running.
#
#   bash scripts/run.sh <tasks.json> <output_dir> <cdp_url> [cdp_url ...]
#
# One worker per CDP URL. A URL may point at a local Chrome
# (http://localhost:9222) or at a remote browser; if the remote browser needs an
# access token, pass it as ?access_token=... in the URL.
# The model and endpoint come from config.json; the API key from VLM_API_KEY.
#
# To start local browsers and run in one step, use scripts/run_local.sh.
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"

if [ $# -lt 3 ]; then
    echo "usage: bash scripts/run.sh <tasks.json> <output_dir> <cdp_url> [cdp_url ...]"
    exit 1
fi

TASK_FILE="$1"
OUTPUT_DIR="$2"
shift 2
CDP_URLS=("$@")

[ -f "$TASK_FILE" ] || { echo "task file not found: $TASK_FILE"; exit 1; }

echo "tasks : $TASK_FILE"
echo "output: $OUTPUT_DIR"
echo "browsers (${#CDP_URLS[@]}):"
for u in "${CDP_URLS[@]}"; do
    echo "  ${u%%\?*}"      # never print an access token
done

# macOS: forked workers that touch CoreFoundation (via OpenCV) abort without this.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
# Keep localhost CDP traffic away from any system proxy / VPN client.
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"

"$PY" "$REPO_DIR/scripts/launcher.py" \
    --input "$TASK_FILE" \
    --output "$OUTPUT_DIR" \
    --cdp_url "${CDP_URLS[@]}" \
    ${MODEL:+--model "$MODEL"}

#!/bin/sh
# Worker container start: make sure the weights are on disk, then hand over.
#
# Runs scripts/ensure_weights.py (defaults fit a RunPod pod with the network volume at
# /workspace; override with RADAR_WEIGHTS_DIR and RADAR_WEIGHTS_FALLBACK), exports the
# directory it settled on as RADAR_WEIGHTS_RESOLVED, then runs the given command (the image's
# CMD is the pull worker, python3 -m radar_worker.pull), or exits 0 when there is none.
set -eu

APP_DIR="${RADAR_APP_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"

report="$("$PYTHON" "$APP_DIR/scripts/ensure_weights.py")" || {
    echo "$report"
    echo "entrypoint: weights not ready" >&2
    exit 1
}
echo "$report"
RADAR_WEIGHTS_RESOLVED="$(printf '%s' "$report" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["dir"])')"
export RADAR_WEIGHTS_RESOLVED
echo "entrypoint: weights in $RADAR_WEIGHTS_RESOLVED" >&2

if [ "$#" -gt 0 ]; then
    exec "$@"
fi
exit 0

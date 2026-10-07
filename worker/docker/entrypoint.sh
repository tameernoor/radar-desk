#!/bin/sh
# Worker container start: make sure the weights are on disk, then hand over.
#
# Started as root, it first hands the weights directory to the unprivileged user radar and runs
# everything below as that user. RADAR_RUN_AS_ROOT=1 keeps root (serverless on the RunPod volume
# writes results into folders the app made there). Started as another user (docker run --user),
# it skips the hand-over.
#
# Runs scripts/ensure_weights.py (defaults fit a RunPod pod with the network volume at
# /workspace; override with RADAR_WEIGHTS_DIR and RADAR_WEIGHTS_FALLBACK), exports the
# directory it settled on as RADAR_WEIGHTS_RESOLVED, then runs the given command (the image's
# CMD is the pull worker, python3 -m radar_worker.pull), or exits 0 when there is none.
set -eu

if [ "$(id -u)" = 0 ] && [ "${RADAR_RUN_AS_ROOT:-0}" != 1 ]; then
    dir="${RADAR_WEIGHTS_DIR:-/workspace/radar-weights}"
    if [ -d "$(dirname "$dir")" ]; then
        mkdir -p "$dir" 2>/dev/null || true
    fi
    # chown only when something is not radar's yet, so a filled volume costs one directory walk
    if [ -d "$dir" ] && [ -n "$(find "$dir" ! -user radar -print -quit 2>/dev/null)" ]; then
        chown -R radar:radar "$dir" 2>/dev/null \
            || echo "entrypoint: could not hand $dir to user radar; it is read-only for the worker" >&2
    fi
    export HOME=/home/radar
    exec setpriv --reuid=radar --regid=radar --init-groups -- "$0" "$@"
fi

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
echo "entrypoint: weights in $RADAR_WEIGHTS_RESOLVED as $(id -un 2>/dev/null || id -u)" >&2

if [ "$#" -gt 0 ]; then
    exec "$@"
fi
exit 0

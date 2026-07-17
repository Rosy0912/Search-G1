#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:-search_g1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
MANIFEST="${MANIFEST:-$ARTIFACT_ROOT/probes/live_${EXPERIMENT_NAME}/manifest.json}"
PORT="${PORT:-8002}"

exec python probe_server.py --manifest "$MANIFEST" --port "$PORT"

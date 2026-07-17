#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a Hugging Face model or checkpoint}"
: "${DATA_FILE:?Set DATA_FILE to a Search-G1 parquet split}"

DATASET_TAG="${DATASET_TAG:-nq}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-search_g1}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
LAYERS="${LAYERS:-21,24,27}"
CAP="${CAP:-300}"
OFFSET="${OFFSET:-0}"
GROUP_SIZE="${GROUP_SIZE:-5}"
GPU_MEM="${GPU_MEM:-0.45}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
DUMP_DIR="$ARTIFACT_ROOT/calibration/${EXPERIMENT_NAME}/round_0"
READOUT_DIR="$ARTIFACT_ROOT/readouts/${EXPERIMENT_NAME}/round_0"
MANIFEST="$ARTIFACT_ROOT/probes/live_${EXPERIMENT_NAME}/manifest.json"

export RETRIEVER_URL

python ts_probe_dump.py \
  --model "$MODEL_PATH" \
  --data "$DATA_FILE" \
  --tag "$DATASET_TAG" \
  --cap "$CAP" \
  --offset "$OFFSET" \
  --G "$GROUP_SIZE" \
  --layers "$LAYERS" \
  --gpu_mem "$GPU_MEM" \
  --out "$DUMP_DIR"

python ts_probe_train.py \
  --dumps "$DUMP_DIR" \
  --layers "$LAYERS" \
  --commit-position Xa \
  --commit-label flip \
  --save "$READOUT_DIR"

python - "$MODEL_PATH" "$READOUT_DIR" "$MANIFEST" <<'PY'
import json
import os
import sys

model, readout_dir, manifest = sys.argv[1:]
payload = {
    "round": 0,
    "encoder": os.path.abspath(model) if os.path.exists(model) else model,
    "commit": os.path.abspath(os.path.join(readout_dir, "commit_probe.pkl")),
    "boundary": os.path.abspath(os.path.join(readout_dir, "boundary_probe.pkl")),
}
os.makedirs(os.path.dirname(manifest), exist_ok=True)
tmp = manifest + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
os.replace(tmp, manifest)
print(f"wrote {manifest}")
PY

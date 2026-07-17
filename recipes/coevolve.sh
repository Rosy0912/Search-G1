#!/usr/bin/env bash
set -euo pipefail

: "${DATA_FILE:?Set DATA_FILE to the held-out calibration parquet source}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-search_g1}"
DATASET_TAG="${DATASET_TAG:-nq}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-verl_checkpoints}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
LAYERS="${LAYERS:-21,24,27}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-300}"
CALIBRATION_OFFSET="${CALIBRATION_OFFSET:-0}"
POLL_SECONDS="${POLL_SECONDS:-120}"

exec python ts_coevolve.py \
  --exps "$EXPERIMENT_NAME" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --work-dir "$ARTIFACT_ROOT/coevolution" \
  --manifest-root "$ARTIFACT_ROOT/probes" \
  --data "$DATA_FILE" \
  --tag "$DATASET_TAG" \
  --nq "$CALIBRATION_SIZE" \
  --holdout_offset "$CALIBRATION_OFFSET" \
  --layers "$LAYERS" \
  --commit-position Xa \
  --commit-label flip \
  --retriever "$RETRIEVER_URL" \
  --poll "$POLL_SECONDS"

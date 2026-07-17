#!/usr/bin/env bash
set -euo pipefail

: "${CORPUS_PATH:?Set CORPUS_PATH to a JSONL corpus}"
: "${INDEX_DIR:?Set INDEX_DIR to the output directory}"

python search_r1/search/index_builder.py \
  --retrieval_method "${RETRIEVER_NAME:-e5}" \
  --model_path "${RETRIEVER_MODEL:-intfloat/e5-base-v2}" \
  --corpus_path "$CORPUS_PATH" \
  --save_dir "$INDEX_DIR" \
  --use_fp16 \
  --max_length "${MAX_LENGTH:-256}" \
  --batch_size "${BATCH_SIZE:-512}" \
  --pooling_method "${POOLING_METHOD:-mean}" \
  --faiss_type "${FAISS_TYPE:-Flat}" \
  --save_embedding

#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_PATH:?Set DATASET_PATH to a prepared dataset directory}"
: "${INDEX_PATH:?Set INDEX_PATH to a FAISS index}"
: "${CORPUS_PATH:?Set CORPUS_PATH to a JSONL retrieval corpus}"

SPLIT="${SPLIT:-test}"
TOPK="${TOPK:-3}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"

python search_r1/search/retrieval.py \
  --retrieval_method e5 \
  --retrieval_topk "$TOPK" \
  --index_path "$INDEX_PATH" \
  --corpus_path "$CORPUS_PATH" \
  --dataset_path "$DATASET_PATH" \
  --data_split "$SPLIT" \
  --retrieval_model_path "$RETRIEVER_MODEL" \
  --retrieval_pooling_method mean \
  --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE:-512}"

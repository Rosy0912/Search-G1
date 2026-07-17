#!/usr/bin/env bash
set -euo pipefail

: "${INDEX_PATH:?Set INDEX_PATH to a FAISS index}"
: "${CORPUS_PATH:?Set CORPUS_PATH to a JSONL retrieval corpus}"

RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
TOPK="${TOPK:-3}"

extra_args=()
if [[ "${FAISS_GPU:-0}" == "1" ]]; then
  extra_args+=(--faiss_gpu)
fi

exec python -m search_r1.search.retrieval_server \
  --index_path "$INDEX_PATH" \
  --corpus_path "$CORPUS_PATH" \
  --topk "$TOPK" \
  --retriever_name "$RETRIEVER_NAME" \
  --retriever_model "$RETRIEVER_MODEL" \
  "${extra_args[@]}"

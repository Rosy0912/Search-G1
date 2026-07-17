#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8001}"

exec python cf_judge_server.py --model "$MODEL_PATH" --port "$PORT"

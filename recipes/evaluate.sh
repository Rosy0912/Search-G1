#!/usr/bin/env bash
set -euo pipefail

: "${CKPT:?Set CKPT to a model or actor checkpoint}"
: "${VAL_FILE:?Set VAL_FILE to an evaluation parquet}"

TRAIN_FILE="${TRAIN_FILE:-$VAL_FILE}"
TAG="${TAG:-search_g1_eval}"
VAL_N="${VAL_N:-null}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_TURNS="${MAX_TURNS:-4}"
N_GPU="${N_GPU:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.25}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ECO_USE_CF="${ECO_USE_CF:-0}"
export ECO_TRUST_VARIANT="${ECO_TRUST_VARIANT:-balanced}"
export ECO_TRUST_SIGNAL="${ECO_TRUST_SIGNAL:-probe}"
export CF_JUDGE_URL="${CF_JUDGE_URL:-http://127.0.0.1:8001/judge_batch}"

python -m verl.trainer.main_ppo_eco \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_data_num=null \
  data.val_data_num="$VAL_N" \
  data.train_batch_size=128 \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=500 \
  data.max_start_length=2048 \
  data.max_obs_length=500 \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$CKPT" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.n_agent=1 \
  actor_rollout_ref.rollout.temperature=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.state_masking=true \
  trainer.logger="['console']" \
  +trainer.val_only=true \
  +trainer.val_before_train=true \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="$N_GPU" \
  trainer.nnodes=1 \
  trainer.project_name=Search-G1-eval \
  trainer.experiment_name="$TAG" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  max_turns="$MAX_TURNS" \
  retriever.url="$RETRIEVER_URL" \
  retriever.topk=3

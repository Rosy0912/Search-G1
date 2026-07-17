#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a Hugging Face model or checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the training parquet}"
: "${VAL_FILE:?Set VAL_FILE to the validation parquet}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-search_g1}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
PROBE_URL="${PROBE_URL:-http://127.0.0.1:8002/judge_probe}"
N_GPU="${N_GPU:-4}"
VAL_N="${VAL_N:-200}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-250}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-25}"
MAX_TURNS="${MAX_TURNS:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
PPO_MICRO_BS="${PPO_MICRO_BS:-8}"
LOGPROB_MICRO_BS="${LOGPROB_MICRO_BS:-16}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-false}"
GRAD_OFFLOAD="${GRAD_OFFLOAD:-false}"
OPTIM_OFFLOAD="${OPTIM_OFFLOAD:-false}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PROBE_URL
export ECO_TRUST_VARIANT="${ECO_TRUST_VARIANT:-balanced}"
export ECO_TRUST_SIGNAL=probe
export ECO_PROBE_GATE="${ECO_PROBE_GATE:-1}"
export ECO_BAL_GROUND="${ECO_BAL_GROUND:-0.6}"
export ECO_BAL_NOSEARCH_BONUS="${ECO_BAL_NOSEARCH_BONUS:-0.5}"
export ECO_W_COST="${ECO_W_COST:-0.02}"
export ECO_WRONG_GROUND_SCALE="${ECO_WRONG_GROUND_SCALE:-0.0}"
export ECO_USE_CF="${ECO_USE_CF:-0}"

python -m verl.trainer.main_ppo_eco \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_data_num=null \
  data.val_data_num="$VAL_N" \
  data.train_batch_size=128 \
  data.val_batch_size=64 \
  data.max_prompt_length=4096 \
  data.max_response_length=500 \
  data.max_start_length=2048 \
  data.max_obs_length=500 \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BS" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.grad_offload="$GRAD_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPTIM_OFFLOAD" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="$LOGPROB_MICRO_BS" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.n_agent=5 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size="$LOGPROB_MICRO_BS" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$PARAM_OFFLOAD" \
  actor_rollout_ref.actor.state_masking=true \
  algorithm.no_think_rl=false \
  trainer.logger="['console']" \
  +trainer.val_only=false \
  +trainer.val_before_train=true \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="$N_GPU" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=Search-G1 \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs=15 \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
  max_turns="$MAX_TURNS" \
  retriever.url="$RETRIEVER_URL" \
  retriever.topk=3

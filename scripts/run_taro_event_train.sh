#!/usr/bin/env bash
set -euo pipefail
set -x

# ============================================================
# Formal Event-TARO training
# ============================================================
#
# Main comparison protocol:
#   - Qwen3-1.7B
#   - GRPO
#   - thinking enabled
#   - max response length = 8192
#   - Event-TARO: pilot=2, Gmin=4, Gmax=8
#   - prompt pool = 4
#   - total generation budget = 20 per training step
#   - commit window = 2
#   - adaptive window = 2
#   - lambda_time = 0.02
#
# We train for 200 optimizer/training steps, matching the existing
# Fixed-G=8 baseline in update count.
#
# Existing baseline:
#   train_batch_size=2, G=8
#   => 16 generated rollouts / step
#   => 200 steps = 3200 generated rollouts
#
# Event-TARO:
#   total_budget=20 / step
#   => step 160 = 3200 generated rollouts  (rollout-budget matched)
#   => step 200 = 4000 generated rollouts  (training-step matched)
#
# SAVE_FREQ=40 is chosen so the final two retained checkpoints are
# normally step 160 and step 200 when max_actor_ckpt_to_keep=2.
# ============================================================

TARO_ROOT="/root/autodl-tmp/taro"
VERL_ROOT="${TARO_ROOT}/verl"

MODEL_PATH="${TARO_ROOT}/models/Qwen3-1.7B"
TRAIN_FILE="${TARO_ROOT}/data/math/train.parquet"
VAL_FILE="${TARO_ROOT}/data/math/test.parquet"
REWARD_FILE="${VERL_ROOT}/taro_math_reward.py"
MAKESPAN_CSV="${TARO_ROOT}/analysis/taro_step1/makespan_v2_results.csv"

SEED="${SEED:-42}"
TOTAL_STEPS="${TOTAL_STEPS:-200}"

SAVE_FREQ="${SAVE_FREQ:-40}"
TEST_FREQ="${TEST_FREQ:-50}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"

COMMIT_WINDOW="${COMMIT_WINDOW:-2}"
ADAPTIVE_WINDOW="${ADAPTIVE_WINDOW:-2}"
MAX_INFLIGHT="${MAX_INFLIGHT:-16}"
ACTIVATION_EPSILON="${ACTIVATION_EPSILON:-1e-6}"
LAMBDA_TIME="${LAMBDA_TIME:-0.02}"

PILOT_G="${PILOT_G:-2}"
GMIN="${GMIN:-4}"
GMAX="${GMAX:-8}"
MIN_SELECTED="${MIN_SELECTED:-3}"
TOTAL_BUDGET="${TOTAL_BUDGET:-20}"
DEFAULT_PREDICTED_LENGTH="${DEFAULT_PREDICTED_LENGTH:-4096}"

RUN_NAME="${RUN_NAME:-qwen3_1.7b_taro_event_cw${COMMIT_WINDOW}_aw${ADAPTIVE_WINDOW}_seed${SEED}_${TOTAL_STEPS}steps}"

OUTPUT_ROOT="${TARO_ROOT}/outputs/taro_event_formal/${RUN_NAME}"
ROLLOUT_DIR="${OUTPUT_ROOT}/rollouts"
VAL_OUTPUT_DIR="${OUTPUT_ROOT}/validation"
TARO_LOG_DIR="${OUTPUT_ROOT}/taro_trace"
LOG_DIR="${OUTPUT_ROOT}/logs"
CKPT_DIR="${TARO_ROOT}/checkpoints/${RUN_NAME}"

mkdir -p \
  "${ROLLOUT_DIR}" \
  "${VAL_OUTPUT_DIR}" \
  "${TARO_LOG_DIR}" \
  "${LOG_DIR}" \
  "${CKPT_DIR}"

LOG_FILE="${LOG_DIR}/train.log"

# Do not delete output/checkpoint directories here.
# trainer.resume_mode=auto allows an interrupted run to resume.
exec > >(tee -a "${LOG_FILE}") 2>&1

# ============================================================
# Environment
# ============================================================

export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

# Explicitly register Megatron backend in verl.
export VERL_ENGINE_REGISTER_MEGATRON=1

export HYDRA_FULL_ERROR=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_DEDUP_LOGS=0

# Required by the validated environment.
export VLLM_USE_FLASHINFER_SAMPLER=0

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

unset NVTE_DEBUG
unset NVTE_DEBUG_LEVEL

cd "${VERL_ROOT}"

# ============================================================
# Experiment summary
# ============================================================

echo "============================================================"
echo "Formal Event-TARO GRPO Training"
echo "============================================================"
echo "RUN_NAME                 = ${RUN_NAME}"
echo "SEED                     = ${SEED}"
echo "TOTAL_STEPS              = ${TOTAL_STEPS}"
echo "SAVE_FREQ                = ${SAVE_FREQ}"
echo "TEST_FREQ                = ${TEST_FREQ}"
echo "VAL_MAX_SAMPLES          = ${VAL_MAX_SAMPLES}"
echo "VAL_BEFORE_TRAIN         = ${VAL_BEFORE_TRAIN}"
echo "PILOT_G                  = ${PILOT_G}"
echo "GMIN / GMAX              = ${GMIN} / ${GMAX}"
echo "MIN_SELECTED             = ${MIN_SELECTED}"
echo "TOTAL_BUDGET             = ${TOTAL_BUDGET}"
echo "COMMIT_WINDOW            = ${COMMIT_WINDOW}"
echo "ADAPTIVE_WINDOW          = ${ADAPTIVE_WINDOW}"
echo "MAX_INFLIGHT             = ${MAX_INFLIGHT}"
echo "ACTIVATION_EPSILON       = ${ACTIVATION_EPSILON}"
echo "LAMBDA_TIME              = ${LAMBDA_TIME}"
echo "DEFAULT_PREDICTED_LENGTH = ${DEFAULT_PREDICTED_LENGTH}"
echo "OUTPUT_ROOT              = ${OUTPUT_ROOT}"
echo "CKPT_DIR                 = ${CKPT_DIR}"
echo "============================================================"

# ============================================================
# Dependency / source sanity check
# ============================================================

python - <<'PY'
import os
import torch
import flash_attn
import transformer_engine
import math_verify

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("flash-attn:", flash_attn.__version__)
print("TransformerEngine:", transformer_engine.__version__)
print("math_verify:", math_verify)
print("GPU:", torch.cuda.get_device_name(0))

assert flash_attn.__version__ == "2.8.1", \
    f"Expected flash-attn 2.8.1, got {flash_attn.__version__}"

required_paths = [
    "/root/autodl-tmp/taro/verl/taro_math_reward.py",
    "/root/autodl-tmp/taro/verl/verl/experimental/agent_loop/taro_event_manager.py",
    "/root/autodl-tmp/taro/verl/verl/trainer/ppo/taro_event_online.py",
    "/root/autodl-tmp/taro/analysis/taro_step1/makespan_v2_results.csv",
]

for path in required_paths:
    assert os.path.exists(path), f"Required TARO file not found: {path}"

print("Dependency/source sanity check: PASS")
PY

# ============================================================
# Formal Event-TARO run
# ============================================================

python -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=ppo_megatron_trainer.yaml \
  \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  \
  data.train_batch_size=4 \
  data.train_max_samples=-1 \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.dataloader_num_workers=0 \
  \
  data.max_prompt_length=512 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  \
  data.shuffle=True \
  data.validation_shuffle=False \
  data.seed="${SEED}" \
  \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_torch_optimizer_for_cpu_offload=False \
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
  \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8704 \
  actor_rollout_ref.actor.data_loader_seed="${SEED}" \
  \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]' \
  actor_rollout_ref.actor.checkpoint.load_contents='["model","extra"]' \
  \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_fused_kernels=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.context_parallel_size=1 \
  actor_rollout_ref.actor.megatron.sequence_parallel=False \
  \
  actor_rollout_ref.actor.megatron.param_offload=False \
  actor_rollout_ref.actor.megatron.optimizer_offload=False \
  actor_rollout_ref.actor.megatron.grad_offload=False \
  \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  actor_rollout_ref.actor.megatron.use_remove_padding=True \
  \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
  actor_rollout_ref.actor.megatron.dist_ckpt_optim_fully_reshardable=True \
  actor_rollout_ref.actor.megatron.distrib_optim_fully_reshardable_mem_efficient=False \
  \
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.agent_loop.taro_event_manager.TAROEventAgentLoopManager \
  \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=640 \
  \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.max_model_len=8704 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2,4,8,16]' \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.n=1 \
  \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8704 \
  \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  \
  reward.custom_reward_function.path="${REWARD_FILE}" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers=1 \
  \
  ++trainer.taro_online.enabled=true \
  ++trainer.taro_online.pilot_g="${PILOT_G}" \
  ++trainer.taro_online.gmin="${GMIN}" \
  ++trainer.taro_online.gmax="${GMAX}" \
  ++trainer.taro_online.min_selected="${MIN_SELECTED}" \
  ++trainer.taro_online.activation_epsilon="${ACTIVATION_EPSILON}" \
  ++trainer.taro_online.total_budget="${TOTAL_BUDGET}" \
  ++trainer.taro_online.max_inflight="${MAX_INFLIGHT}" \
  ++trainer.taro_online.lambda_time="${LAMBDA_TIME}" \
  ++trainer.taro_online.default_predicted_length="${DEFAULT_PREDICTED_LENGTH}" \
  ++trainer.taro_online.seed="${SEED}" \
  ++trainer.taro_online.makespan_csv="${MAKESPAN_CSV}" \
  ++trainer.taro_online.log_dir="${TARO_LOG_DIR}" \
  ++trainer.taro_online.commit_window="${COMMIT_WINDOW}" \
  ++trainer.taro_online.adaptive_window="${ADAPTIVE_WINDOW}" \
  \
  trainer.rollout_data_dir="${ROLLOUT_DIR}" \
  +trainer.validation_data_dir="${VAL_OUTPUT_DIR}" \
  trainer.log_val_generations=8 \
  \
  trainer.project_name=taro \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.logger='["console"]' \
  \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.test_freq="${TEST_FREQ}" \
  \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=2 \
  \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_STEPS}"

echo "============================================================"
echo "Formal Event-TARO training finished."
echo "RUN_NAME       = ${RUN_NAME}"
echo "LOG_FILE       = ${LOG_FILE}"
echo "ROLLOUT_DIR    = ${ROLLOUT_DIR}"
echo "VAL_DIR        = ${VAL_OUTPUT_DIR}"
echo "TARO_TRACE_DIR = ${TARO_LOG_DIR}"
echo "CKPT_DIR       = ${CKPT_DIR}"
echo "============================================================"

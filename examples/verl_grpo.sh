#!/usr/bin/env bash
# GRPO on GSM8K: verl generates, ThunderSync's engine streams the backward.
#
# One node: TRAIN_GPUS train while ROLLOUT_GPUS generate. Needs CUDA, Python
# 3.12, and verl (with vLLM and TransferQueue) with ThunderSync installed after
# it; see "Training with verl" in the README. Prepare the data with verl's
# examples/data_preprocess/gsm8k.py. Extra arguments are Hydra overrides.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
BATCH=${BATCH:-64}

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=thundersync \
    trainer.v1.separate_async.parameter_sync_step=1 \
    trainer.v1.separate_async.num_warmup_batches=0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$TRAIN_GPUS" \
    trainer.logger='[console]' \
    trainer.total_epochs=1 \
    transfer_queue.enable=True \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.bypass_mode=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$BATCH" \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor._target_=thundersync.integrations.verl.engine.ThunderSyncActorConfig \
    actor_rollout_ref.actor.strategy=thundersync \
    +actor_rollout_ref.actor.thundersync.max_concurrent_groups=4 \
    +actor_rollout_ref.actor.thundersync.max_held_sources=8 \
    +actor_rollout_ref.actor.thundersync.direct_when_group_bound=True \
    +actor_rollout_ref.actor.thundersync.final_forest_max_tokens=16384 \
    +actor_rollout_ref.actor.thundersync.binary_rewards=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS" \
    "$@"

#!/usr/bin/env bash
# GRPO on GSM8K: verl generates, VeOmni trains each group as soon as it lands.
#
# One node: TRAIN_GPUS train under VeOmni (FSDP2, with expert parallelism EP
# for MoE models) while ROLLOUT_GPUS generate. Needs CUDA, Python 3.12, verl
# with VeOmni, vLLM and TransferQueue, and ThunderSync; see "Training with
# verl" in the README. Extra arguments are Hydra overrides.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
EP=${EP:-1}
BATCH=${BATCH:-64}

python3 -m verl.trainer.main_ppo \
    model_engine=veomni \
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
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$BATCH" \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.veomni.expert_parallel_size="$EP" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS" \
    "$@"

"""ThunderSync's streaming GRPO step as a verl v1 trainer.

``trainer.v1.trainer_mode=thundersync`` runs verl's separate-async topology,
rollout and training on their own GPUs, with one change to the step: a
trajectory is handed to the ThunderSync engine of the rank that owns its group
as soon as it lands in the TransferQueue, so its backward runs while the rest
of the batch is still generating. The optimizer still steps once, after every
group has closed, and the rollout receives the new policy before the next
batch is submitted. The update is the on-policy GRPO update, computed early.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils.debug import marked_timer
from verl.workers.engine_workers import ActorRolloutRefWorker

from thundersync.config import ConfigError
from thundersync.integrations.verl.engine import TRAJECTORY_FIELDS

__all__ = ["CONTRACT", "ThunderSyncTrainer", "ThunderSyncWorker", "check_contract"]

# How often the step polls the TransferQueue for landed trajectories. A
# close waits at most this long to start; the poll lists tags only.
POLL_INTERVAL_S = 0.25

# The settings under which verl's GRPO step is the update ThunderSync streams:
# one pass with no KL or entropy term, sequence log-probs summed over tokens,
# old log-probs taken from the rollout, and a policy that stays fixed until
# the whole batch has landed and been applied.
CONTRACT = {
    "algorithm.adv_estimator": "grpo",
    "algorithm.norm_adv_by_std_in_grpo": True,
    "algorithm.use_kl_in_reward": False,
    "algorithm.filter_groups.enable": False,
    "algorithm.rollout_correction.bypass_mode": True,
    "actor_rollout_ref.actor.strategy": "thundersync",
    "actor_rollout_ref.actor.ppo_epochs": 1,
    "actor_rollout_ref.actor.use_kl_loss": False,
    "actor_rollout_ref.actor.entropy_coeff": 0,
    "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum",
    "actor_rollout_ref.actor.optim.optimizer": "AdamW",
    "trainer.v1.separate_async.parameter_sync_step": 1,
    "trainer.v1.separate_async.num_warmup_batches": 0,
    "trainer.v1.separate_async.hybrid_rollout.enable_switch": False,
}


def check_contract(config) -> None:
    """Refuse a configuration whose update ThunderSync would not compute."""
    wrong = [key for key, value in CONTRACT.items() if OmegaConf.select(config, key) != value]
    ranks = config.trainer.nnodes * config.trainer.n_gpus_per_node
    problems = [f"{key}={CONTRACT[key]!r}" for key in wrong]
    if config.data.train_batch_size < ranks:
        problems.append(f"data.train_batch_size>={ranks} (one group per training rank)")
    if problems:
        raise ConfigError("ThunderSync's GRPO step requires " + ", ".join(problems))


class ThunderSyncWorker(ActorRolloutRefWorker):
    """verl's actor worker with the ThunderSync engine's streaming calls."""

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def close_trajectories(self, data) -> None:
        if data is not None:
            self.actor.engine.forward_backward_batch(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_step(self) -> dict[str, float]:
        engine = self.actor.engine
        metrics = {
            "actor/grad_norm": engine.optimizer_step(),
            "actor/lr": engine.lr_scheduler_step(),
        }
        stats = engine.step_stats
        return metrics | {
            f"thundersync/{key}": stats[key] for key in ("barrier_s", "allreduce_s", "step_total_s")
        }


@register_trainer("thundersync")
class ThunderSyncTrainer(PPOTrainerSeparateAsync):
    """Separate-async verl whose actor update streams in with the rollout."""

    def __init__(self, config):
        check_contract(config)
        super().__init__(config)

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        self.role_worker_mapping[Role.Actor] = ray.remote(ThunderSyncWorker)

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        # "gen" ends when the last trajectory lands; "update_actor" is only
        # the backward that was still queued then, and the collective step.
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, closing = self._stream(sample_batch_size)
            self.on_sample_end()
        with marked_timer("update_actor", timing_raw, color="red"):
            ray.get(closing)
            metrics.update(self.actor_rollout_wg.optimizer_step()[0])
        # verl's own GRPO advantages, for its data metrics; the update is applied.
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics)
        return batch

    def _stream(self, num_groups: int) -> tuple[KVBatchMeta, list]:
        """Close each trajectory of this step's groups on its group's rank as it lands.

        Groups are dealt round-robin over the ranks. A rank holds at most
        ``max_concurrent_groups`` groups open (each keeps parameter-sized
        statistics); a group past that bound waits for a slot, and one that has
        fully landed by then takes it first, since it needs no statistics.
        """

        size = self.config.actor_rollout_ref.rollout.n
        ranks = self.actor_rollout_wg.world_size
        limit = (
            OmegaConf.select(
                self.config,
                "actor_rollout_ref.actor.thundersync.max_concurrent_groups",
                default=None,
            )
            or math.inf
        )
        tags = tq.kv_list(partition_id="train").get("train", {})
        uids = sorted(
            key
            for key, tag in tags.items()
            if tag.get("is_prompt") and tag.get("global_steps") == self.global_steps
        )
        if len(uids) != num_groups:
            raise RuntimeError(
                f"step {self.global_steps} submitted {len(uids)} groups, expected {num_groups}"
            )
        owner = {uid: index % ranks for index, uid in enumerate(uids)}
        landed, waiting, sent, closing = {}, defaultdict(list), Counter(), []
        held = [set() for _ in range(ranks)]

        def shard(keys: list[str]) -> KVBatchMeta | None:
            """One rank's landed trajectories; None for a rank with none."""
            if not keys:
                return None
            return KVBatchMeta(
                keys=keys,
                tags=[landed[key] for key in keys],
                partition_id="train",
                fields=list(TRAJECTORY_FIELDS),
                extra_info={"rollout_n": size},
            )

        while sent.total() < size * len(uids):
            for key, tag in tags.items():
                uid = key.split("_")[0]
                if uid in owner and key not in landed and tag.get("status") == "success":
                    landed[key] = tag
                    waiting[uid].append(key)
            if failed := [uid for uid in uids if tags[uid].get("status") == "failure"]:
                raise RuntimeError(f"rollout failed for groups {failed}")
            shards = [[] for _ in range(ranks)]
            # Held groups first (they may free their slot), then fully landed ones.
            for uid in sorted(
                waiting,
                key=lambda uid: (uid not in held[owner[uid]], sent[uid] + len(waiting[uid]) < size),
            ):
                rank = owner[uid]
                if uid in held[rank] or len(held[rank]) < limit:
                    keys = waiting.pop(uid)
                    shards[rank] += keys
                    sent[uid] += len(keys)
                    if sent[uid] < size:
                        held[rank].add(uid)
                    else:
                        held[rank].discard(uid)
            if any(shards):
                closing += self.actor_rollout_wg.close_trajectories(list(map(shard, shards)))
            else:
                time.sleep(POLL_INTERVAL_S)
            tags = tq.kv_list(partition_id="train").get("train", {})
        tq.kv_clear(keys=uids, partition_id="train")
        batch = KVBatchMeta(partition_id="train", keys=list(landed), tags=list(landed.values()))
        return batch, closing

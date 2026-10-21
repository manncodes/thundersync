"""ThunderSync's streaming step as a verl v1 trainer.

``trainer.v1.trainer_mode=thundersync`` runs verl's separate-async topology,
rollout and training on their own GPUs, with one change to the step: the
actor update starts from what has landed in the TransferQueue while the rest
of the batch is still generating, and the optimizer steps once, after all of
it. The rollout receives the new policy before the next batch is submitted,
so the update is the on-policy one, computed early.

* With ThunderSync's engine (``actor.strategy=thundersync``), a trajectory
  goes to its group's rank as soon as its content or its reward lands, and
  that rank's admission runs the reward-linear backward.
* With any other verl engine, such as VeOmni or FSDP, a group goes through
  verl's own update pipeline (padding, old and reference log-probs,
  advantages, forward-backward) as soon as verl marks it finished, and the
  engine accumulates the pieces, each normalized by the whole batch.

``+trainer.v1.thundersync.defer=True`` holds every dispatch until the whole
batch has landed, the same step without the overlap, for comparison; the
``stream/`` metrics report how much of the step was dispatched early.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterator

import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils.debug import marked_timer
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker

from thundersync.config import ConfigError
from thundersync.integrations.verl.engine import COMPLETION_FIELDS, VERDICT_FIELDS

__all__ = [
    "CONTRACT",
    "STREAMED_LOSS_AGG_MODES",
    "THUNDERSYNC_ENGINE_CONTRACT",
    "ThunderSyncTrainer",
    "ThunderSyncWorker",
    "check_contract",
]

# How often the step polls the TransferQueue's tags for landed trajectories.
POLL_INTERVAL_S = 0.25

# The settings under which an update streamed in pieces is verl's one-update
# GRPO step: one pass, advantages that need only their own group, and a
# policy that stays fixed until the whole batch has landed and been applied.
CONTRACT = {
    "algorithm.adv_estimator": "grpo",
    "algorithm.use_kl_in_reward": False,
    "algorithm.filter_groups.enable": False,
    "algorithm.rollout_correction.rollout_is_batch_normalize": False,
    "actor_rollout_ref.actor.ppo_epochs": 1,
    "distillation.enabled": False,
    "trainer.v1.separate_async.parameter_sync_step": 1,
    "trainer.v1.separate_async.num_warmup_batches": 0,
    "trainer.v1.separate_async.hybrid_rollout.enable_switch": False,
}

# ThunderSync's engine computes its reward-linear GRPO: no KL or entropy term,
# advantages normalized by the group's standard deviation, sequence log-probs
# summed over tokens, and old log-probs taken from the rollout.
THUNDERSYNC_ENGINE_CONTRACT = {
    "algorithm.norm_adv_by_std_in_grpo": True,
    "algorithm.rollout_correction.bypass_mode": True,
    "actor_rollout_ref.actor.use_kl_loss": False,
    "actor_rollout_ref.actor.entropy_coeff": 0,
    "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum",
    "actor_rollout_ref.actor.optim.optimizer": "AdamW",
}

# The loss aggregations another engine computes the same piece by piece: each
# piece sums its own sequences and divides by the batch's sequence count, or
# by nothing. token-mean divides by the batch's token count, unknown until
# the batch has landed, and seq-mean-token-sum-norm by each micro-batch's
# padded width unless actor.loss_scale_factor fixes it.
STREAMED_LOSS_AGG_MODES = (
    "token-sum",
    "seq-mean-token-sum",
    "seq-mean-token-mean",
    "seq-mean-token-sum-norm",
)


def check_contract(config) -> None:
    """Refuse a configuration whose streamed update would not be verl's step."""
    actor = config.actor_rollout_ref.actor
    thundersync = actor.strategy == "thundersync"
    contract = CONTRACT | (THUNDERSYNC_ENGINE_CONTRACT if thundersync else {})
    problems = [
        f"{key}={value!r}"
        for key, value in contract.items()
        if OmegaConf.select(config, key) != value
    ]
    reward_model = config.reward.reward_model
    if reward_model.enable and not reward_model.enable_resource_pool:
        # A colocated reward model scores the batch only once it has landed.
        problems.append("reward.reward_model.enable_resource_pool=True")
    ranks = config.trainer.nnodes * config.trainer.n_gpus_per_node
    if thundersync and config.data.train_batch_size < ranks:
        problems.append(f"data.train_batch_size>={ranks} (one group per training rank)")
    mode = actor.loss_agg_mode
    if not thundersync and (
        mode not in STREAMED_LOSS_AGG_MODES
        or (mode == "seq-mean-token-sum-norm" and actor.loss_scale_factor is None)
    ):
        problems.append(
            f"actor_rollout_ref.actor.loss_agg_mode in {STREAMED_LOSS_AGG_MODES}"
            ", the last with actor.loss_scale_factor set"
        )
    if problems:
        raise ConfigError("ThunderSync's streamed step requires " + ", ".join(problems))


class ThunderSyncWorker(TinkerActorRolloutRefWorker):
    """verl's actor worker with the calls the streamed step makes.

    ``land`` feeds the ThunderSync engine's event loop; the inherited
    ``forward_backward`` accumulates complete groups on any other engine;
    ``optimizer_step`` ends the batch on either and advances the schedule.
    """

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def land(self, completions, verdicts) -> None:
        if completions is not None or verdicts is not None:
            self.actor.engine.land(completions, verdicts)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_step(self, optim_step_params=None) -> dict:
        metrics = super().optimizer_step(optim_step_params)
        engine = self.actor.engine
        return metrics | {"lr": engine.lr_scheduler_step()} | getattr(engine, "step_metrics", {})


@register_trainer("thundersync")
class ThunderSyncTrainer(PPOTrainerSeparateAsync):
    """Separate-async verl whose actor update streams in with the rollout."""

    def __init__(self, config):
        check_contract(config)
        super().__init__(config)
        self.streams_trajectories = config.actor_rollout_ref.actor.strategy == "thundersync"
        self.defer = OmegaConf.select(config, "trainer.v1.thundersync.defer", default=False)

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        # The role verl trains the actor under: Actor, or ActorRollout(Ref)
        # when the training GPUs also serve rollout.
        actor = (Role.Actor, Role.ActorRolloutRef, Role.ActorRollout)
        role = next(role for role in actor if role in self.role_worker_mapping)
        self.role_worker_mapping[role] = ray.remote(ThunderSyncWorker)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        # A streamed piece is padded to the data-parallel size, not to a mini-batch.
        return dp_size

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        # "gen" ends when the last trajectory lands; "update_actor" is only
        # the backward still queued then, and the step.
        stream = self._stream_trajectories if self.streams_trajectories else self._stream_groups
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, pending = stream(sample_batch_size, metrics)
            self.on_sample_end()
        with marked_timer("update_actor", timing_raw, color="red"):
            ray.get(pending)
            step = self.actor_rollout_wg.optimizer_step()[0]
        metrics.update({f"actor/{key}": value for key, value in step.items()})
        if self.streams_trajectories:
            # verl's own advantages, for its data metrics; the update is applied.
            with marked_timer("adv", timing_raw, color="brown"):
                batch = self._compute_advantage(batch, metrics)
        return batch

    def _stream_trajectories(self, num_groups: int, metrics: dict) -> tuple[KVBatchMeta, list]:
        """Land each trajectory's completion and verdict on its group's rank.

        Groups are dealt round-robin over the ranks. verl submits the whole
        batch at once, so ThunderSync's ``max_concurrent_groups`` bounds the
        groups a rank holds open here, where each keeps parameter-sized
        statistics: a group past the bound waits in the TransferQueue for a
        slot, and one whose every reward has landed by then goes first,
        since it closes at once. Each group must be ``rollout.n`` sessions
        of one output each.
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
        uids = self._step_uids(num_groups)
        total = size * len(uids)
        owner = {uid: index % ranks for index, uid in enumerate(uids)}
        content, verdicts, arrival = defaultdict(list), defaultdict(list), {}
        announced, scored, landed, sent, calls = set(), {}, Counter(), Counter(), []
        held = [set() for _ in range(ranks)]
        overlap = _Overlap()
        for tags, finished in self._polls(uids):
            for key, tag in tags.items():
                status, (uid, session, output) = tag.get("status"), _split(key)
                if output or not 0 <= session < size:
                    raise RuntimeError(f"{key}: not one of rollout.n={size} single-output sessions")
                if status in ("content", "success") and key not in announced:
                    announced.add(key)
                    content[uid].append(key)
                    arrival.setdefault(uid, len(arrival))
                if status == "success" and key not in scored:
                    scored[key] = tag
                    verdicts[uid].append(key)
                    landed[uid] += 1
            if short := sorted(uid for uid in finished if landed[uid] < size):
                raise RuntimeError(f"groups {short} finished short of rollout.n={size}")
            if len(scored) == total:
                overlap.landed()
            # Deferred, no group is admitted until the whole batch has landed.
            slots = 0 if self.defer and len(scored) < total else limit
            completions, binds = [[] for _ in range(ranks)], [[] for _ in range(ranks)]
            # Held groups first (they may free their slot), then fully scored
            # ones, each in arrival order.
            for uid in sorted(
                content.keys() | verdicts.keys(),
                key=lambda uid: (
                    uid not in held[owner[uid]],
                    sent[uid] + len(verdicts.get(uid, ())) < size,
                    arrival[uid],
                ),
            ):
                rank = owner[uid]
                if uid in held[rank] or len(held[rank]) < slots:
                    completions[rank] += content.pop(uid, [])
                    bound = verdicts.pop(uid, [])
                    binds[rank] += bound
                    sent[uid] += len(bound)
                    if sent[uid] < size:
                        held[rank].add(uid)
                    else:
                        held[rank].discard(uid)
            if any(completions) or any(binds):
                calls += self.actor_rollout_wg.land(
                    self._shards(completions, tags, COMPLETION_FIELDS, size),
                    self._shards(binds, tags, VERDICT_FIELDS, size),
                )
                overlap.dispatch(sum(map(len, completions)))
                if sent.total() == total:
                    break
            else:
                time.sleep(POLL_INTERVAL_S)
        metrics.update(overlap.metrics())
        tq.kv_clear(keys=uids, partition_id="train")
        batch = KVBatchMeta(partition_id="train", keys=list(scored), tags=list(scored.values()))
        return batch, calls

    def _stream_groups(self, num_groups: int, metrics: dict) -> tuple[KVBatchMeta, list]:
        """Accumulate each finished group on the engine once the last piece is done.

        A group is every trajectory verl stored for its prompt, as verl's own
        step takes it. Groups that finish while the engine is busy wait and
        go together, so a piece grows with the backlog.
        """

        uids = self._step_uids(num_groups)
        landed, members, done, ready = {}, defaultdict(list), set(), []
        pieces, futures, running = [], [], []
        aggregator, overlap = MetricsAggregator(), _Overlap()
        for tags, finished in self._polls(uids):
            for key, tag in tags.items():
                if tag.get("status") == "success" and key not in landed:
                    landed[key] = tag
                    members[_split(key)[0]].append(key)
            ready += sorted(finished - done)
            done |= finished
            if len(done) == len(uids):
                overlap.landed()
            busy = running and ray.wait(running, num_returns=len(running), timeout=0)[1]
            deferred = self.defer and len(done) < len(uids)
            if ready and not busy and not deferred:
                keys = [key for uid in ready for key in members[uid]]
                ready = []
                if keys:
                    pieces.append(self._accumulate(keys, [landed[key] for key in keys], aggregator))
                    running = self.actor_rollout_wg.forward_backward(pieces[-1]).futures
                    futures += running
                    overlap.dispatch(len(keys))
                if len(done) == len(uids):
                    break
            else:
                time.sleep(POLL_INTERVAL_S)
        metrics.update(aggregator.get_aggregated_metrics() | overlap.metrics())
        tq.kv_clear(keys=uids, partition_id="train")
        keys = [key for piece in pieces for key in piece.keys]
        tags = [tag for piece in pieces for tag in piece.tags]
        return KVBatchMeta(partition_id="train", keys=keys, tags=tags), futures

    def _accumulate(
        self, keys: list[str], tags: list[dict], aggregator: MetricsAggregator
    ) -> KVBatchMeta:
        """verl's update pipeline for complete groups, up to the forward-backward."""
        piece, metrics = KVBatchMeta(partition_id="train", keys=keys, tags=tags), {}
        piece = self._balance_batch(piece, metrics)
        # The policy is fixed until the batch steps, so the base computation
        # is the old policy for every piece; separate-async's own override
        # checkpoints the policy for steps that update more than once.
        piece = PPOTrainer._compute_old_log_prob(self, piece, metrics)
        if self.use_reference_policy:
            piece = self._compute_ref_log_prob(piece, metrics)
        piece = self._compute_advantage(piece, metrics)
        aggregator.add_step_metrics(metrics, sample_count=len(keys))
        actor, rollout = self.config.actor_rollout_ref.actor, self.config.actor_rollout_ref.rollout
        rollout_correction = actor.policy_loss.get("rollout_correction") or {}
        piece.extra_info = {
            # verl's loss divides by the whole batch's sequence count, so the
            # pieces sum to the batch's loss.
            "global_batch_size": actor.ppo_mini_batch_size * rollout.n,
            "calculate_entropy": actor.calculate_entropy or actor.entropy_coeff != 0,
            "score_centering": bool(rollout_correction.get("score_centering", False)),
            "distillation_use_topk": False,
            "distillation_only": False,
            "temperature": rollout.temperature,
        }
        return piece

    def _step_uids(self, num_groups: int) -> list[str]:
        """The step's prompt groups, as submitted to the rollout."""
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
        return uids

    @staticmethod
    def _polls(uids: list[str]) -> Iterator[tuple[dict[str, dict], set[str]]]:
        """Each poll's tags of the step's trajectories, and its finished groups.

        verl marks a group finished once every trajectory of it is stored,
        and failed if any of them failed, which raises.
        """
        members = set(uids)
        while True:
            tags = tq.kv_list(partition_id="train").get("train", {})
            status = {uid: tags[uid].get("status") for uid in uids}
            if failed := [uid for uid in uids if status[uid] == "failure"]:
                raise RuntimeError(f"rollout failed for groups {failed}")
            trajectories = {
                key: tag
                for key, tag in tags.items()
                if not tag.get("is_prompt") and key.rsplit("_", 2)[0] in members
            }
            yield trajectories, {uid for uid in uids if status[uid] == "finished"}

    @staticmethod
    def _shards(
        shards: list[list[str]], tags: dict[str, dict], fields: tuple[str, ...], size: int
    ) -> list[KVBatchMeta | None]:
        """One meta per rank over its keys; None for a rank with none."""
        return [
            KVBatchMeta(
                keys=keys,
                tags=[tags[key] for key in keys],
                partition_id="train",
                fields=list(fields),
                extra_info={"rollout_n": size},
            )
            if keys
            else None
            for keys in shards
        ]


class _Overlap:
    """How much of a step's training was dispatched before its rollout ended."""

    def __init__(self):
        self.start = time.perf_counter()
        self.first_dispatch = self.last_landing = math.nan
        self.dispatched = self.before_last_landing = 0

    def dispatch(self, trajectories: int) -> None:
        if math.isnan(self.first_dispatch):
            self.first_dispatch = time.perf_counter() - self.start
        self.dispatched += trajectories

    def landed(self) -> None:
        if math.isnan(self.last_landing):
            self.last_landing = time.perf_counter() - self.start
            self.before_last_landing = self.dispatched

    def metrics(self) -> dict[str, float]:
        return {
            "stream/first_dispatch_s": self.first_dispatch,
            "stream/last_landing_s": self.last_landing,
            "stream/dispatched_before_last_landing": self.before_last_landing
            / max(self.dispatched, 1),
        }


def _split(key: str) -> tuple[str, int, int]:
    """A trajectory key's group, session and output: verl's ``{uid}_{session}_{index}``."""
    uid, session, output = key.rsplit("_", 2)
    return uid, int(session), int(output)

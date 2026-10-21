"""ThunderSync as a verl training engine.

``actor_rollout_ref.actor.strategy=thundersync`` builds this engine where verl
would build FSDP: a full policy replica per rank (``load_trainer``) driven by
one ThunderSync event loop per rank. Landed trajectories enter the loop as
completions (content final) and verdicts (reward landed); the loop opens each
group at its first trajectory and serves the events through ThunderSync's
final-first admission, sweeping between them as ThunderSync's own driver
does. ``optimizer_step`` is the batch boundary: the loop drains the admission
and takes the step's one collective.

Replication is a requirement, not a choice: the reward-linear backward
differentiates with ``autograd.grad``, which bypasses FSDP's reduce-scatter.
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import convert_weight_keys
from verl.utils.torch_functional import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from verl.workers.config import FSDPActorConfig
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry

from thundersync.config import ConfigError

__all__ = [
    "COMPLETION_FIELDS",
    "LOOP_SETTINGS",
    "VERDICT_FIELDS",
    "RankLoop",
    "ThunderSyncActorConfig",
    "ThunderSyncEngine",
    "grpo_config",
]

# The fields of a verl agent-loop trajectory each event reads.
COMPLETION_FIELDS = ("uid", "session_id", "prompts", "responses", "response_mask")
VERDICT_FIELDS = ("uid", "session_id", "rm_scores")

# The event loop's settings, beside the ones ``GrpoTrainerConfig.from_namespace``
# reads (which include max_held_sources and prepare_pack_max, to size what
# the admission holds). Zero leaves a bound or the final forest off.
LOOP_SETTINGS = {
    # Differentiated sources a rank may hold unfolded; content prepares wait
    # for folds beyond it.
    "max_held_sources": 0,
    # Trajectories per packed branch rebuild.
    "prepare_pack_max": 0,
    # Once a rank's every reward has landed, run its remaining work as final
    # forest micro-batches of at most this many tokens (needs
    # direct_when_group_bound).
    "final_forest_max_tokens": 0,
    "final_forest_chunk": 1024,
    # Arena slabs reserved for known closes (needs pinned_nonblocking sources).
    "known_reserve_slabs": 0,
    # Idle ranks claim busy ranks' final work through the c10d store (needs
    # the final forest).
    "work_claims": False,
    # Every reward is 0 or 1: host statistics keep one parameter-sized sum.
    "binary_rewards": False,
}


@dataclass
class ThunderSyncActorConfig(FSDPActorConfig):
    """``dp_actor`` with ThunderSync's own settings.

    ``thundersync`` holds names ``GrpoTrainerConfig.from_namespace`` reads, for
    example ``rebuild_block_tokens``, ``source_offload_mode``,
    ``direct_when_group_bound`` or ``max_concurrent_groups``, and the event
    loop's ``LOOP_SETTINGS``; each one it leaves unset takes ThunderSync's
    default. Select it with ``actor_rollout_ref.actor._target_``.
    """

    thundersync: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.engine.thundersync = dict(self.thundersync)


@EngineRegistry.register(model_type="language_model", backend="thundersync")
class ThunderSyncEngine(BaseEngine):
    """A replicated policy trained by ``DataParallelThunderSync``.

    ``land`` takes any landed slice of a GRPO batch, down to one trajectory,
    as verl agent-loop rows; every trajectory of a group must reach the same
    rank. ``forward_backward_batch`` lands rows as completed and scored at
    once. ``optimizer_step`` ends the batch.
    """

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.step_metrics: dict[str, float] = {}
        self._saved: list[str] = []

    def initialize(self):
        from thundersync.grpo.runtime import load_trainer

        if self.model_config.lora_rank > 0:
            raise ConfigError("the verl engine trains every parameter; set model.lora_rank=0")
        optim = self.optimizer_config
        verl_settings = {
            "model_path": self.model_config.local_path,
            "learning_rate": optim.lr,
            "grad_clip": optim.clip_grad,
            "deterministic_algorithms": self.engine_config.full_determinism,
            "rebuild_block_tokens": 4096,
            # DataParallelThunderSync's own default; "none" keeps sources on
            # the device, which only the blocking mode allows.
            "source_offload_device": "cpu",
        }
        settings = getattr(self.engine_config, "thundersync", {})
        config = grpo_config(verl_settings, settings)
        device = torch.device(get_device_name(), get_device_id())
        self.module, self.optimizer, trainer = load_trainer(config, device)
        # load_trainer fixes GRPO's AdamW; verl's optimizer config owns the rest.
        for group in self.optimizer.param_groups:
            group.update(betas=tuple(optim.betas), weight_decay=optim.weight_decay)
        self.lr_scheduler = _lr_scheduler(self.optimizer, optim)
        loop_settings = {name: settings.get(name, value) for name, value in LOOP_SETTINGS.items()}
        self.loop = RankLoop(trainer, device, **loop_settings)

    @property
    def trainer(self):
        return self.loop.trainer

    def land(self, completions: TensorDict | None, verdicts: TensorDict | None) -> None:
        """Queue landed completions and verdicts on this rank's event loop."""
        data = next(rows for rows in (completions, verdicts) if rows is not None)
        self.loop.land(
            tu.get(data, "rollout_n"),
            _rows(completions, COMPLETION_FIELDS),
            _rows(verdicts, VERDICT_FIELDS),
        )

    def forward_backward_batch(self, data: TensorDict, loss_function=None, forward_only=False):
        """Land every row of ``data`` as completed and scored.

        The objective is ThunderSync's reward-linear GRPO; ``loss_function``
        is not consulted.
        """

        if forward_only:
            raise NotImplementedError(
                "ThunderSync trains one on-policy pass; its old log-probs are the rollout's"
            )
        self.land(data, data)
        return {}

    def optimizer_zero_grad(self):
        """Nothing to clear: ``optimizer_step`` leaves the gradients empty."""

    def optimizer_step(self) -> float:
        stats = self.loop.finish()
        self.step_metrics = {
            f"thundersync/{key}": stats[key] for key in ("barrier_s", "allreduce_s", "step_total_s")
        } | stats["loop_s"]
        return stats["global_grad_norm"]

    def lr_scheduler_step(self) -> float:
        self.lr_scheduler.step()
        return self.lr_scheduler.get_last_lr()[0]

    def train_mode(self, **kwargs):
        return BaseEngineCtx(self, "train", **kwargs)

    def eval_mode(self, **kwargs):
        return BaseEngineCtx(self, "eval", **kwargs)

    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    def get_per_tensor_param(self, **kwargs):
        return iter(convert_weight_keys(self.module.state_dict(), self.module).items()), None

    def get_data_parallel_size(self):
        return dist.get_world_size()

    def get_data_parallel_rank(self):
        return dist.get_rank()

    def get_data_parallel_group(self):
        return dist.group.WORLD

    def is_mp_src_rank_with_outputs(self):
        return True

    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs
    ):
        # The replicas are identical; each rank owns only its optimizer state.
        rank = dist.get_rank()
        os.makedirs(local_path, exist_ok=True)
        if rank == 0:
            torch.save(self.module.state_dict(), os.path.join(local_path, "model.pt"))
        state = {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(state, os.path.join(local_path, f"optimizer_rank_{rank}.pt"))
        dist.barrier()
        self._saved.append(local_path)
        if max_ckpt_to_keep:
            stale, self._saved = self._saved[:-max_ckpt_to_keep], self._saved[-max_ckpt_to_keep:]
            if rank == 0:
                for path in stale:
                    shutil.rmtree(path, ignore_errors=True)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **kwargs):
        device = next(self.module.parameters()).device
        model = torch.load(
            os.path.join(local_path, "model.pt"), map_location=device, weights_only=True
        )
        state = torch.load(
            os.path.join(local_path, f"optimizer_rank_{dist.get_rank()}.pt"),
            map_location=device,
            weights_only=True,
        )
        with torch.no_grad():
            self.module.load_state_dict(model)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        # The load wrote every parameter in place: re-arm the version guard.
        self.trainer.bw.abort()
        self.trainer._new_batch()


class RankLoop:
    """One rank's ThunderSync driver, on a thread of its own.

    ``land`` queues events and returns. The loop ingests every queued event
    (opening a group at its first trajectory, appending the trajectory,
    adding its completion, binding its verdict), then sweeps the step's
    admission: ready closes, content prepares, ready group closes, and with
    work claims a peer's final work when idle. It sweeps again without a new
    event while the admission reports deferred work. ``finish`` queues the
    batch boundary: the loop drains the admission, takes the trainer's step,
    and begins a new admission at the next step's first event.
    """

    def __init__(
        self,
        trainer,
        device: torch.device,
        *,
        max_held_sources: int = 0,
        prepare_pack_max: int = 0,
        final_forest_max_tokens: int = 0,
        final_forest_chunk: int = 1024,
        known_reserve_slabs: int = 0,
        work_claims: bool = False,
        binary_rewards: bool = False,
        poll_s: float = 0.02,
    ):
        self.trainer = trainer
        self._device = device
        self._poll_s = poll_s
        self._gates = {
            "max_held_sources": max_held_sources or None,
            "max_prepare_pack": prepare_pack_max or None,
            "in_flight_sources": lambda: self.trainer.held_source_count,
            "final_forest": bool(final_forest_max_tokens),
            "known_reserve_slabs": known_reserve_slabs,
            "events_waiting": lambda: not self._events.empty(),
        }
        if final_forest_max_tokens:
            trainer.configure_final_forest(
                max_tokens_per_microbatch=final_forest_max_tokens, chunk=final_forest_chunk
            )
        if known_reserve_slabs:
            trainer.reserve_known_slabs(known_reserve_slabs)
        if binary_rewards:
            trainer.configure_binary_rewards()
        self._claims = None
        if work_claims:
            from thundersync.scheduling.work_claims import StoreWorkClaims

            store = dist.distributed_c10d._get_default_store()
            self._claims = StoreWorkClaims(
                store, rank=dist.get_rank(), world_size=dist.get_world_size(), namespace="verl"
            )
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._admission = None
        self._groups: dict[str, tuple[int, list[int]]] = {}
        self._seconds: Counter[str] = Counter()
        self._failure: BaseException | None = None
        threading.Thread(target=self._run, name="thundersync-rank-loop", daemon=True).start()

    def land(self, size: int, completions: list[tuple], verdicts: list[tuple]) -> None:
        if self._failure is not None:
            raise RuntimeError("ThunderSync's event loop failed") from self._failure
        self._events.put((size, completions, verdicts))

    def finish(self) -> dict[str, Any]:
        boundary: Future = Future()
        self._events.put(boundary)
        return boundary.result()

    def _run(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        while True:
            event = self._next(block=True)
            if self._failure is not None:
                if isinstance(event, Future):
                    event.set_exception(self._failure)
                continue
            try:
                while isinstance(event, tuple):
                    self._ingest(*event)
                    event = self._next(block=False)
                if self._admission is not None:
                    self._sweep()
                if isinstance(event, Future):
                    event.set_result(self._boundary())
            except BaseException as error:
                self._failure = error
                if isinstance(event, Future):
                    event.set_exception(error)

    def _next(self, *, block: bool):
        deferred = self._failure is None and self._admission is not None
        timeout = self._poll_s if deferred and self._admission.deferred_work_reason else None
        try:
            return self._events.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def _ingest(self, size: int, completions: list[tuple], verdicts: list[tuple]) -> None:
        if self._admission is None:
            self._admission = self._new_admission(size)
        for uid, session, prompt, response, mask in completions:
            group = self._group(uid, prompt, size)
            trajectory = _trajectory(group, session, size)
            self.trainer.append_turn(group, trajectory, response, [bool(m) for m in mask])
            self._admission.add_completion(group, trajectory)
        for uid, session, scores in verdicts:
            group = self._groups[uid][0]
            self._admission.bind_verdict(group, _trajectory(group, session, size), sum(scores))

    def _new_admission(self, size: int):
        from thundersync.grpo.admission import FinalFirstAdmission
        from thundersync.grpo.claims import ClaimingFinalFirstAdmission

        options = {"group_size": size, "trainer_call": self._call, **self._gates}
        if self._claims is None:
            return FinalFirstAdmission(self.trainer, **options)
        return ClaimingFinalFirstAdmission(self.trainer, claims=self._claims, **options)

    def _group(self, uid: str, prompt: list[int], size: int) -> int:
        if uid not in self._groups:
            group = len(self._groups)
            self.trainer.open_group(group, prompt)
            self.trainer.register_group_trajectories(
                group, list(range(group * size, (group + 1) * size))
            )
            self._groups[uid] = group, prompt
        group, opened = self._groups[uid]
        if opened != prompt:
            raise ValueError(f"group {uid}: its trajectories disagree on the prompt")
        return group

    def _sweep(self) -> None:
        # Ready closes first (their folds free the budget prepares reuse),
        # then prepares, then group closes, as the admission orders its work.
        self._admission.drain_ready_closes()
        self._admission.flush_prepares()
        self._admission.close_ready_groups()

    def _boundary(self) -> dict[str, Any]:
        # A rank without events still takes the step, so its empty group
        # count fails every rank at once instead of stranding them in it.
        if (admission := self._admission) is not None:
            admission.drain_ready_closes()
            admission.flush_prepares(force=True)
            while admission.ready_close_backlog:
                admission.drain_ready_closes()
                admission.close_ready_groups()
            admission.drain_group_closes()
        stats = self.trainer.step()
        stats["loop_s"] = {f"thundersync/{key}_s": value for key, value in self._seconds.items()}
        self._admission = None
        self._groups.clear()
        self._seconds.clear()
        return stats

    def _call(self, phase: str, action, **metadata) -> None:
        """The admission's trainer call: run it and time it by phase and class."""
        started = time.perf_counter()
        action()
        label = metadata.get("job_class")
        self._seconds[f"{phase}/{label}" if label else phase] += time.perf_counter() - started


def grpo_config(defaults: dict[str, Any], overrides: dict[str, Any]):
    """``GrpoTrainerConfig.from_namespace`` over ``defaults`` updated by ``overrides``.

    ``from_namespace`` falls back to its default for any name it does not
    find, so a misspelt override would vanish; one that neither it nor the
    event loop reads is refused.
    """

    from thundersync.grpo.config import GrpoTrainerConfig

    read = set(LOOP_SETTINGS)

    class Settings(SimpleNamespace):
        def __getattribute__(self, name):
            read.add(name)
            return super().__getattribute__(name)

    GrpoTrainerConfig.from_namespace(Settings(**defaults))
    if unknown := overrides.keys() - read:
        raise ConfigError(f"ThunderSync has no setting {sorted(unknown)}")
    return GrpoTrainerConfig.from_namespace(SimpleNamespace(**defaults | overrides))


def _trajectory(group: int, session: int, size: int) -> int:
    if not 0 <= session < size:
        raise ValueError(f"session {session} is outside rollout.n={size}")
    return group * size + session


def _rows(data: TensorDict | None, fields: tuple[str, ...]) -> list[tuple]:
    """``fields`` of every row of ``data`` as Python values."""
    if data is None:
        return []
    columns = []
    for key in fields:
        value = tu.get(data, key)
        columns.append(
            value if isinstance(value, list) else [row.tolist() for row in value.unbind(0)]
        )
    return list(zip(*columns, strict=True))


def _lr_scheduler(optimizer: torch.optim.Optimizer, optim) -> torch.optim.lr_scheduler.LRScheduler:
    """verl's actor learning-rate schedule, as its FSDP engine builds it."""
    warmup = optim.lr_warmup_steps
    if warmup <= 0:
        warmup = int(optim.lr_warmup_steps_ratio * optim.total_training_steps)
    if optim.lr_scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, warmup)
    if optim.lr_scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            warmup,
            optim.total_training_steps,
            min_lr_ratio=optim.min_lr_ratio,
            num_cycles=optim.num_cycles,
            zero_indexed_step=optim.zero_indexed_step,
        )
    raise ConfigError(f"unsupported lr_scheduler_type {optim.lr_scheduler_type!r}")

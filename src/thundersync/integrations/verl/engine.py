"""ThunderSync as a verl training engine.

``actor_rollout_ref.actor.strategy=thundersync`` builds this engine where verl
would build FSDP: a full policy replica per rank (``load_trainer``) whose GRPO
backward runs per trajectory as each one lands, and whose ranks meet once per
batch, at ``optimizer_step``. Replication is a requirement, not a choice: the
reward-linear backward differentiates with ``autograd.grad``, which bypasses
FSDP's reduce-scatter.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
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

__all__ = ["TRAJECTORY_FIELDS", "ThunderSyncActorConfig", "ThunderSyncEngine", "grpo_config"]

# The fields of a verl agent-loop trajectory the engine reads.
TRAJECTORY_FIELDS = ("uid", "session_id", "prompts", "responses", "response_mask", "rm_scores")


@dataclass
class ThunderSyncActorConfig(FSDPActorConfig):
    """``dp_actor`` with ThunderSync's own settings.

    ``thundersync`` holds names ``GrpoTrainerConfig.from_namespace`` reads, for
    example ``rebuild_block_tokens``, ``source_offload_mode`` or
    ``max_concurrent_groups``; each one it leaves unset takes ThunderSync's
    default. Select it with ``actor_rollout_ref.actor._target_``.
    """

    thundersync: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.engine.thundersync = dict(self.thundersync)


@EngineRegistry.register(model_type="language_model", backend="thundersync")
class ThunderSyncEngine(BaseEngine):
    """A replicated policy trained by ``DataParallelThunderSync``.

    ``forward_backward_batch`` accepts any slice of a GRPO batch, down to a
    single trajectory, as verl agent-loop rows: it opens a group the first time
    one of its trajectories arrives, closes every trajectory with its reward,
    and closes a group with its last trajectory. Every trajectory of a group
    must reach the same rank. ``optimizer_step`` is the batch's only collective.
    """

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.step_stats: dict[str, Any] = {}
        self._groups: dict[str, tuple[int, list[int]]] = {}
        self._landed: Counter[int] = Counter()
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
        }
        config = grpo_config(verl_settings, getattr(self.engine_config, "thundersync", {}))
        device = torch.device(get_device_name(), get_device_id())
        self.module, self.optimizer, self.trainer = load_trainer(config, device)
        # load_trainer fixes GRPO's AdamW; verl's optimizer config owns the rest.
        for group in self.optimizer.param_groups:
            group.update(betas=tuple(optim.betas), weight_decay=optim.weight_decay)
        self.lr_scheduler = _lr_scheduler(self.optimizer, optim)

    def forward_backward_batch(self, data: TensorDict, loss_function=None, forward_only=False):
        """Close the trajectories in ``data`` and the groups they complete.

        The objective is ThunderSync's reward-linear GRPO; ``loss_function``
        is not consulted.
        """

        if forward_only:
            raise NotImplementedError(
                "ThunderSync trains one on-policy pass; its old log-probs are the rollout's"
            )
        size = tu.get(data, "rollout_n")
        closes, complete = [], []
        for uid, session, prompt, response, mask, scores in zip(
            *(_column(data, key) for key in TRAJECTORY_FIELDS), strict=True
        ):
            if not 0 <= session < size:
                raise ValueError(f"group {uid}: session {session} outside rollout.n={size}")
            group = self._group(uid, prompt, size)
            trajectory = group * size + session
            self.trainer.append_turn(group, trajectory, response, [bool(m) for m in mask])
            closes.append((trajectory, sum(scores)))
            self._landed[group] += 1
            if self._landed[group] == size:
                complete.append(group)
        self.trainer.close_trajectories(closes)
        for group in complete:
            self.trainer.close_group(group)
        return {}

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

    def optimizer_zero_grad(self):
        """Nothing to clear: ``optimizer_step`` leaves the gradients empty."""

    def optimizer_step(self) -> float:
        self.step_stats = self.trainer.step()
        self._groups.clear()
        self._landed.clear()
        return self.step_stats["global_grad_norm"]

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


def grpo_config(defaults: dict[str, Any], overrides: dict[str, Any]):
    """``GrpoTrainerConfig.from_namespace`` over ``defaults`` updated by ``overrides``.

    ``from_namespace`` falls back to its default for any name it does not
    find, so a misspelt override would vanish; one it never reads is refused.
    """

    from thundersync.grpo.config import GrpoTrainerConfig

    read = set()

    class Settings(SimpleNamespace):
        def __getattribute__(self, name):
            read.add(name)
            return super().__getattribute__(name)

    GrpoTrainerConfig.from_namespace(Settings(**defaults))
    if unknown := overrides.keys() - read:
        raise ConfigError(f"ThunderSync has no setting {sorted(unknown)}")
    return GrpoTrainerConfig.from_namespace(SimpleNamespace(**defaults | overrides))


def _column(data: TensorDict, key: str) -> list:
    """One field of every row as Python values."""
    value = tu.get(data, key)
    return value if isinstance(value, list) else [row.tolist() for row in value.unbind(0)]


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

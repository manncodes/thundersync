"""The verl integration: the engine computes GRPO, the step streams it.

* equality -- verl agent-loop rows fed to the engine in any arrival order and
  any chunking give the dense GRPO gradient: each trajectory alone, plain
  attention, population-std advantages, sequence log-probs summed over the
  generated tokens and summed over a group's samples;
* checkpoint -- a load restores the weights, optimizer and schedule, and re-arms
  the version guard the in-place load would otherwise trip;
* dispatch -- the step hands every landed trajectory to its group's rank
  exactly once, holds at most ``max_concurrent_groups`` groups open per
  rank, and admits a fully landed group first;
* contract -- verl's default configuration is refused key by key, and every
  key the contract names exists in verl's configuration schema.

DataParallelThunderSync needs CUDA and NCCL; the engine is driven here by a
one-process stand-in built from the run and backward ``load_trainer`` builds.
Skipped unless verl and TransferQueue are importable.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

verl = pytest.importorskip("verl")
pytest.importorskip("transfer_queue")

from omegaconf import OmegaConf  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402

from thundersync.config import ConfigError  # noqa: E402
from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
    register_stream_attention,
)
from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402
from thundersync.integrations.verl import trainer as verl_trainer  # noqa: E402
from thundersync.integrations.verl.engine import (  # noqa: E402
    TRAJECTORY_FIELDS,
    ThunderSyncEngine,
    grpo_config,
)

EPS = 1e-6
VOCAB = 97


def build_model(attention: str) -> torch.nn.Module:
    torch.manual_seed(0)
    config = AutoConfig.for_model(
        "qwen3",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=VOCAB,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(
        config, attn_implementation=attention, dtype=torch.float32
    ).eval()


class LocalTrainer:
    """DataParallelThunderSync's event API on one process, without the collective.

    The run and the backward are the ones ``load_trainer`` builds. Without an
    optimizer, ``step`` leaves the summed group gradients in ``.grad``.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None):
        self.model, self.optimizer = model, optimizer
        self._new_batch()

    def _new_batch(self):
        self.run = StreamingRun(
            self.model,
            boundary_cut=True,
            checkpoint=True,
            stream_turns=False,
            rebuild_block_tokens=16,
        )
        self.bw = RewardLinearBackward(self.run, self.model, eps=EPS)

    def open_group(self, group, prompt):
        self.run.open_group(group, prompt)

    def append_turn(self, group, trajectory, tokens, scored):
        self.run.append_turn(group, trajectory, tokens, scored)

    def register_group_trajectories(self, group, trajectories):
        self.bw.register_group_trajectories(group, trajectories)

    def close_trajectories(self, closes):
        self.bw.close_trajectories(closes)

    def close_group(self, group):
        self.bw.close_group(group)

    def step(self) -> dict:
        self.bw.assert_safe_to_step()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._new_batch()
        return {"global_grad_norm": 0.0}


def agent_loop_rows(groups: int, size: int, seed: int) -> list[dict]:
    """Rows as verl's agent loop stores them: a shared prompt per group, a
    response with tool tokens unmasked, the reward on the last token."""

    rng = random.Random(seed)
    rows = []
    for group in range(groups):
        prompt = [rng.randrange(VOCAB) for _ in range(rng.randint(5, 9))]
        for session in range(size):
            length = rng.randint(3, 10)
            scores = torch.zeros(length)
            scores[-1] = rng.random()
            rows.append(
                {
                    "uid": f"uid{group}",
                    "session_id": session,
                    "prompts": torch.tensor(prompt),
                    "responses": torch.tensor([rng.randrange(VOCAB) for _ in range(length)]),
                    "response_mask": torch.tensor([int(rng.random() > 0.3) for _ in range(length)]),
                    "rm_scores": scores,
                }
            )
    return rows


def dense_grpo_gradient(model: torch.nn.Module, rows: list[dict]) -> dict[str, torch.Tensor]:
    objective = 0.0
    for uid in {row["uid"] for row in rows}:
        members = [row for row in rows if row["uid"] == uid]
        rewards = torch.tensor(
            [float(row["rm_scores"].sum()) for row in members], dtype=torch.float64
        )
        mean, std = float(rewards.mean()), float(rewards.std(unbiased=False)) + EPS
        for row, reward in zip(members, rewards.tolist(), strict=True):
            prompt, response = row["prompts"], row["responses"]
            logits = model(torch.cat([prompt, response])[None]).logits[0, len(prompt) - 1 : -1]
            logprobs = logits.float().log_softmax(-1).gather(-1, response[:, None])[:, 0]
            scored = logprobs[row["response_mask"].bool()]
            objective = objective + (reward - mean) / std * scored.sum()
    objective.backward()
    return {name: parameter.grad for name, parameter in model.named_parameters()}


@pytest.mark.parametrize("seed", [0, 1])
def test_streamed_agent_loop_rows_give_the_dense_grpo_gradient(seed):
    register_stream_attention()
    size = 4
    rows = agent_loop_rows(groups=3, size=size, seed=seed)
    streamed, dense = build_model(STREAM_ATTENTION_NAME), build_model("eager")
    engine = ThunderSyncEngine(None, None, None, None)
    engine.trainer = LocalTrainer(streamed)

    rng = random.Random(seed)
    arrivals = rng.sample(rows, len(rows))
    while arrivals:
        take = rng.randint(1, 3)
        chunk, arrivals = arrivals[:take], arrivals[take:]
        data = tu.list_of_dict_to_tensordict(chunk)
        tu.assign_non_tensor(data, rollout_n=size)
        engine.forward_backward_batch(data)
    engine.optimizer_step()

    expected = dense_grpo_gradient(dense, rows)
    for name, parameter in streamed.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-3, atol=1e-5, msg=name)
    assert not engine._groups and not engine._landed


def test_a_group_whose_trajectories_disagree_on_the_prompt_is_refused():
    register_stream_attention()
    rows = agent_loop_rows(groups=1, size=2, seed=3)
    rows[1]["prompts"] = rows[1]["prompts"] + 1
    engine = ThunderSyncEngine(None, None, None, None)
    engine.trainer = LocalTrainer(build_model(STREAM_ATTENTION_NAME))
    data = tu.list_of_dict_to_tensordict(rows)
    tu.assign_non_tensor(data, rollout_n=2)
    with pytest.raises(ValueError, match="disagree on the prompt"):
        engine.forward_backward_batch(data)


def test_a_checkpoint_restores_the_step_and_rearms_the_version_guard(tmp_path):
    register_stream_attention()
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        model = build_model(STREAM_ATTENTION_NAME)
        engine = ThunderSyncEngine(None, None, None, None)
        engine.module = model
        engine.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, maximize=True)
        engine.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _: 1.0)
        engine.trainer = LocalTrainer(model, engine.optimizer)
        for step in range(3):
            data = tu.list_of_dict_to_tensordict(agent_loop_rows(groups=1, size=2, seed=step))
            tu.assign_non_tensor(data, rollout_n=2)
            engine.forward_backward_batch(data)
            engine.optimizer_step()
            engine.lr_scheduler_step()
            engine.save_checkpoint(str(tmp_path / f"step{step}"), max_ckpt_to_keep=2)
        assert sorted(path.name for path in tmp_path.iterdir()) == ["step1", "step2"]

        saved = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        engine.load_checkpoint(str(tmp_path / "step2"))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0, msg=name)
        assert engine.lr_scheduler.last_epoch == 3 and len(engine.optimizer.state) > 0

        data = tu.list_of_dict_to_tensordict(agent_loop_rows(groups=1, size=2, seed=9))
        tu.assign_non_tensor(data, rollout_n=2)
        engine.forward_backward_batch(data)
    finally:
        dist.destroy_process_group()


def test_thundersync_settings_reach_its_config_and_a_misspelt_one_is_refused():
    verl_settings = {"model_path": "model", "learning_rate": 1e-6, "rebuild_block_tokens": 4096}
    config = grpo_config(verl_settings, {"rebuild_block_tokens": 1024, "max_concurrent_groups": 3})
    assert (
        config.streaming.rebuild_block_tokens == 1024
        and config.streaming.statistic_open_groups == 3
    )
    with pytest.raises(ConfigError, match="rebuild_block_token"):
        grpo_config(verl_settings, {"rebuild_block_token": 1024})


class FakeQueue:
    """The TransferQueue calls the step makes; each poll lands one trajectory."""

    def __init__(self, groups: dict[str, str], landings: list[str]):
        self.tags = {
            uid: {"is_prompt": True, "global_steps": 1, "status": status}
            for uid, status in groups.items()
        }
        self.landings = list(landings)
        self.cleared: list[str] = []

    def kv_list(self, partition_id):
        if self.landings:
            self.tags[self.landings.pop(0)] = {"status": "success"}
        return {partition_id: dict(self.tags)}

    def kv_clear(self, keys, partition_id):
        self.cleared += keys


class FakeWorkers:
    world_size = 2

    def __init__(self):
        self.calls: list[list[list[str]]] = []

    def close_trajectories(self, shards):
        assert len(shards) == self.world_size
        for meta in filter(None, shards):
            assert meta.fields == list(TRAJECTORY_FIELDS) and meta.extra_info == {"rollout_n": 2}
            assert meta.tags == [{"status": "success"}] * len(meta.keys)
        self.calls.append([[] if meta is None else meta.keys for meta in shards])
        return []


def stream(monkeypatch, queue: FakeQueue, max_concurrent_groups=None):
    monkeypatch.setattr(verl_trainer, "tq", queue)
    monkeypatch.setattr(verl_trainer, "POLL_INTERVAL_S", 0)
    config = {
        "rollout": {"n": 2},
        "actor": {"thundersync": {"max_concurrent_groups": max_concurrent_groups}},
    }
    step = SimpleNamespace(
        config=OmegaConf.create({"actor_rollout_ref": config}),
        global_steps=1,
        actor_rollout_wg=FakeWorkers(),
    )
    batch, _ = verl_trainer.ThunderSyncTrainer._stream(step, num_groups=len(queue.tags))
    return batch, step.actor_rollout_wg.calls


def test_each_landed_trajectory_reaches_its_groups_rank_once_within_the_open_group_bound(
    monkeypatch,
):
    # Rank 0 owns u0, u2, u4. With one open group per rank, u2 and u4 wait for
    # u0. When it completes, the fully landed u4 goes first and closes in the
    # same call, so the partial u2 still gets the slot it leaves free.
    landings = ["u0_0_0", "u2_0_0", "u4_0_0", "u4_1_0", "u0_1_0", "u2_1_0"]
    landings += [f"u{group}_{session}_0" for group in (1, 3, 5) for session in (0, 1)]
    queue = FakeQueue(dict.fromkeys(["u0", "u1", "u2", "u3", "u4", "u5"], "running"), landings)
    batch, calls = stream(monkeypatch, queue, max_concurrent_groups=1)

    rank0 = [keys for keys, _ in calls if keys]
    assert rank0 == [["u0_0_0"], ["u0_1_0", "u4_0_0", "u4_1_0", "u2_0_0"], ["u2_1_0"]]
    dispatched = [(rank, key) for call in calls for rank, keys in enumerate(call) for key in keys]
    assert sorted(key for _, key in dispatched) == sorted(landings)
    assert all(rank == int(key[1]) % 2 for rank, key in dispatched)
    assert sorted(batch.keys) == sorted(landings) and queue.cleared == [
        "u0",
        "u1",
        "u2",
        "u3",
        "u4",
        "u5",
    ]


def test_unbounded_step_dispatches_each_trajectory_as_it_lands(monkeypatch):
    landings = ["u1_1_0", "u0_0_0", "u1_0_0", "u0_1_0"]
    _, calls = stream(monkeypatch, FakeQueue({"u0": "running", "u1": "running"}, landings))
    assert [key for call in calls for keys in call for key in keys] == landings


def test_a_failed_group_stops_the_step(monkeypatch):
    queue = FakeQueue({"u0": "running", "u1": "failure"}, ["u0_0_0"])
    with pytest.raises(RuntimeError, match="rollout failed for groups \\['u1'\\]"):
        stream(monkeypatch, queue)


def test_contract_names_verl_keys_and_refuses_verls_defaults():
    config = OmegaConf.load(
        Path(verl.__file__).parent / "trainer/config/_generated_ppo_trainer.yaml"
    )
    absent = [
        key
        for key in verl_trainer.CONTRACT
        if OmegaConf.select(config, key, default=KeyError) is KeyError
    ]
    assert not absent, f"verl's configuration has no {absent}"
    with pytest.raises(ConfigError) as refused:
        verl_trainer.check_contract(config)
    for key, value in verl_trainer.CONTRACT.items():
        assert (key in str(refused.value)) == (OmegaConf.select(config, key) != value), key
        OmegaConf.update(config, key, value)
    verl_trainer.check_contract(config)

"""The verl engine: ThunderSync's per-rank flow computes GRPO from verl rows.

* equality -- verl agent-loop rows landed on a rank's event loop give the
  dense GRPO gradient (each trajectory alone, plain attention,
  population-std advantages, sequence log-probs summed over the generated
  tokens and over a group's samples), whatever the arrival order and
  chunking, whether content and reward land together or apart, under a
  held-source budget, on the direct path with the final forest, and with
  binary rewards;
* failure -- an error on the loop thread reaches the step that waits on it;
* checkpoint -- a load restores the weights, optimizer and schedule, and
  re-arms the version guard the in-place load would otherwise trip;
* settings -- ThunderSync's settings reach its config, and a misspelt one is
  refused.

DataParallelThunderSync needs CUDA and NCCL; the loop drives a one-process
stand-in with its event, admission and final-forest calls, built from the run
and backward ``load_trainer`` builds. Skipped unless verl is importable.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("verl")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402

from thundersync.config import ConfigError  # noqa: E402
from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
    register_stream_attention,
)
from thundersync.grpo.data_parallel import DataParallelThunderSync  # noqa: E402
from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402
from thundersync.integrations.verl.engine import (  # noqa: E402
    RankLoop,
    ThunderSyncEngine,
    grpo_config,
)

EPS = 1e-6
VOCAB = 97
SIZE = 4


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
    """DataParallelThunderSync's event, admission and final-forest calls on one
    process. ``step`` keeps only the safety gate: without an optimizer the
    gradients stay summed over groups in ``.grad``."""

    def __init__(self, model, optimizer=None, *, direct_when_group_bound=False):
        self.model, self.optimizer, self.direct = model, optimizer, direct_when_group_bound
        self.forest, self.binary = None, False
        self._new_batch()

    def _new_batch(self):
        self.run = StreamingRun(
            self.model,
            boundary_cut=True,
            checkpoint=True,
            stream_turns=False,
            rebuild_block_tokens=16,
        )
        self.bw = RewardLinearBackward(
            self.run, self.model, eps=EPS, direct_when_group_bound=self.direct
        )
        if self.binary:
            self.bw.configure_binary_rewards()
        self._prompt_tokens, self._price_attention_weight = {}, None

    def __getattr__(self, name):
        return getattr(self.bw, name)

    def configure_binary_rewards(self):
        self.binary = True
        self.bw.configure_binary_rewards()

    def configure_final_forest(self, *, max_tokens_per_microbatch, chunk):
        self.forest = max_tokens_per_microbatch, chunk

    def open_group(self, group, prompt):
        self.run.open_group(group, prompt)
        self._prompt_tokens[group] = tuple(prompt)

    def append_turn(self, group, trajectory, tokens, scored):
        self.run.append_turn(group, trajectory, tokens, scored)

    def close_final_forest(self, closes):
        tokens, chunk = self.forest
        self.bw.close_final_forest(
            closes, prompt_tokens=self._prompt_tokens, max_tokens_per_microbatch=tokens, chunk=chunk
        )

    # Work claims: a final trajectory's payload, as the trainer offers it.
    final_work = DataParallelThunderSync.final_work
    _attention_weight = DataParallelThunderSync._attention_weight

    @property
    def final_forest_max_tokens(self):
        return self.forest[0]

    def record_work_claims(self, report):
        self.claims_report = report

    def step(self) -> dict:
        self.bw.assert_safe_to_step()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._new_batch()
        return dict.fromkeys(("global_grad_norm", "barrier_s", "allreduce_s", "step_total_s"), 0.0)


class SteppedLoop(RankLoop):
    """The rank loop without its thread: each landing is ingested and swept
    before the next arrives, the order a rank sees when events are sparse."""

    def _run(self):
        pass

    def land(self, size, completions, verdicts):
        self._ingest(size, completions, verdicts)
        self._sweep()

    def finish(self):
        return self._boundary()


def engine_on(
    model, optimizer=None, *, loop=SteppedLoop, direct_when_group_bound=False, **settings
):
    engine = ThunderSyncEngine(None, None, None, None)
    trainer = LocalTrainer(model, optimizer, direct_when_group_bound=direct_when_group_bound)
    engine.loop = loop(trainer, torch.device("cpu"), **settings)
    return engine


def agent_loop_rows(groups: int, seed: int, *, binary: bool = False) -> list[dict]:
    """Rows as verl's agent loop stores them: a shared prompt per group, a
    response with tool tokens unmasked, the reward on the last token."""

    rng = random.Random(seed)
    rows = []
    for group in range(groups):
        prompt = [rng.randrange(VOCAB) for _ in range(rng.randint(5, 9))]
        for session in range(SIZE):
            length = rng.randint(3, 10)
            scores = torch.zeros(length)
            scores[-1] = rng.randint(0, 1) if binary else rng.random()
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


def batch(rows: list[dict]):
    data = tu.list_of_dict_to_tensordict(rows) if rows else None
    if data is not None:
        tu.assign_non_tensor(data, rollout_n=SIZE)
    return data


def land(engine, rows: list[dict], seed: int, *, apart: bool) -> None:
    """Land every row in shuffled calls of one to three events. Together, a
    row's completion and verdict arrive in one call; apart, the verdict
    arrives in a later event than the completion."""

    rng = random.Random(seed)
    events = [index for index in range(len(rows)) for _ in range(1 + apart)]
    rng.shuffle(events)
    seen: set[int] = set()
    while events:
        take = rng.randint(1, 3)
        chunk, events = events[:take], events[take:]
        completions, verdicts = [], []
        for index in chunk:
            if index not in seen:
                seen.add(index)
                completions.append(rows[index])
                if not apart:
                    verdicts.append(rows[index])
            else:
                verdicts.append(rows[index])
        engine.land(batch(completions), batch(verdicts))


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


# flow -> (trainer options, loop settings, verdicts land apart, the work the
# admission must have run, by trainer-call phase and job class)
FLOWS = {
    "together": ({}, {}, False, {"trajectory_backward/known", "trajectory_backward/final"}),
    "content_first_under_a_budget": (
        {},
        {"max_held_sources": 2, "prepare_pack_max": 2},
        True,
        {"trajectory_backward/content", "reward_bind", "trajectory_backward/final"},
    ),
    # A one-source budget defers completions until every reward has landed.
    "final_forest": (
        {"direct_when_group_bound": True},
        {"max_held_sources": 1, "final_forest_max_tokens": 48, "final_forest_chunk": 16},
        True,
        {"trajectory_backward/final_forest"},
    ),
    "binary_rewards": ({}, {"binary_rewards": True}, True, {"trajectory_backward/content"}),
}


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_landed_rows_give_the_dense_grpo_gradient(flow, seed):
    register_stream_attention()
    trainer_options, settings, apart, work = FLOWS[flow]
    rows = agent_loop_rows(groups=3, seed=seed, binary=flow == "binary_rewards")
    streamed, dense = build_model(STREAM_ATTENTION_NAME), build_model("eager")
    engine = engine_on(streamed, **trainer_options, **settings)

    land(engine, rows, seed, apart=apart)
    engine.optimizer_step()

    expected = dense_grpo_gradient(dense, rows)
    for name, parameter in streamed.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-3, atol=1e-5, msg=name)
    ran = {key.removeprefix("thundersync/").removesuffix("_s") for key in engine.step_metrics}
    assert work <= ran, sorted(ran)


def test_the_loop_thread_gives_the_same_gradient():
    register_stream_attention()
    rows = agent_loop_rows(groups=3, seed=4)
    streamed, dense = build_model(STREAM_ATTENTION_NAME), build_model("eager")
    engine = engine_on(streamed, loop=RankLoop, max_held_sources=2)

    land(engine, rows, 4, apart=True)
    engine.optimizer_step()

    expected = dense_grpo_gradient(dense, rows)
    for name, parameter in streamed.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-3, atol=1e-5, msg=name)


def test_work_claims_run_every_final_trajectory_through_the_store():
    register_stream_attention()
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        rows = agent_loop_rows(groups=3, seed=5)
        streamed, dense = build_model(STREAM_ATTENTION_NAME), build_model("eager")
        engine = engine_on(
            streamed,
            direct_when_group_bound=True,
            max_held_sources=1,
            final_forest_max_tokens=48,
            final_forest_chunk=16,
            work_claims=True,
        )
        land(engine, rows, 5, apart=True)
        engine.optimizer_step()
        report = engine.trainer.claims_report
    finally:
        dist.destroy_process_group()

    expected = dense_grpo_gradient(dense, rows)
    for name, parameter in streamed.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-3, atol=1e-5, msg=name)
    assert report["done"] and report["offered"] > 0 and report["offered"] == report["own_claims"]


def test_an_error_on_the_loop_thread_reaches_the_step():
    register_stream_attention()
    rows = agent_loop_rows(groups=1, seed=3)
    rows[1]["prompts"] = rows[1]["prompts"] + 1
    engine = engine_on(build_model(STREAM_ATTENTION_NAME), loop=RankLoop)
    engine.land(batch(rows), batch(rows))
    with pytest.raises(ValueError, match="disagree on the prompt"):
        engine.optimizer_step()
    with pytest.raises(RuntimeError, match="event loop failed"):
        engine.land(batch(rows), None)


def test_a_checkpoint_restores_the_step_and_rearms_the_version_guard(tmp_path):
    register_stream_attention()
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        model = build_model(STREAM_ATTENTION_NAME)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, maximize=True)
        engine = engine_on(model, optimizer)
        engine.module, engine.optimizer = model, optimizer
        engine.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        for step in range(3):
            engine.forward_backward_batch(batch(agent_loop_rows(groups=1, seed=step)))
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

        engine.forward_backward_batch(batch(agent_loop_rows(groups=1, seed=9)))
        engine.optimizer_step()
    finally:
        dist.destroy_process_group()


def test_thundersync_settings_reach_its_config_and_a_misspelt_one_is_refused():
    verl_settings = {"model_path": "model", "learning_rate": 1e-6, "rebuild_block_tokens": 4096}
    overrides = {"rebuild_block_tokens": 1024, "max_concurrent_groups": 3, "work_claims": True}
    config = grpo_config(verl_settings, overrides)
    assert config.streaming.rebuild_block_tokens == 1024
    assert config.streaming.statistic_open_groups == 3
    with pytest.raises(ConfigError, match="rebuild_block_token"):
        grpo_config(verl_settings, {"rebuild_block_token": 1024})

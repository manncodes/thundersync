"""The verl trainer: what it streams, to whom, and under which settings.

* trajectories -- with ThunderSync's engine, every landed trajectory reaches
  its group's rank once as a completion and once as a verdict, a completion
  stored before its reward goes ahead of it, a rank holds at most
  ``max_concurrent_groups`` groups open, and a fully scored group goes first;
* groups -- with any other engine, each complete group goes through verl's
  update pipeline once, groups that complete while a piece runs go together,
  and every piece is normalized by the whole batch;
* content first -- the agent loop stores a trajectory's content before verl
  scores it;
* contracts -- verl's defaults are refused key by key, every contract key
  exists in verl's configuration, and the shipped launchers compose into
  configurations verl and the contract accept, for ThunderSync's engine and
  for VeOmni.

Skipped unless verl and TransferQueue are importable.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

verl = pytest.importorskip("verl")
pytest.importorskip("transfer_queue")

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from transfer_queue import KVBatchMeta  # noqa: E402
from verl.experimental.agent_loop import AgentLoopOutput  # noqa: E402
from verl.trainer.ppo.utils import need_critic, need_reference_policy  # noqa: E402
from verl.utils.config import omega_conf_to_dataclass, validate_config  # noqa: E402

from thundersync.config import ConfigError  # noqa: E402
from thundersync.integrations.verl import agent_loop as content_first  # noqa: E402
from thundersync.integrations.verl import trainer as verl_trainer  # noqa: E402
from thundersync.integrations.verl.engine import COMPLETION_FIELDS, VERDICT_FIELDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VERL_CONFIG = Path(verl.__file__).parent / "trainer" / "config"


class FakeQueue:
    """The TransferQueue calls the step makes; each listing lands one event."""

    def __init__(self, groups: dict[str, str], landings: list[tuple[str, str]]):
        self.tags = {
            uid: {"is_prompt": True, "global_steps": 1, "status": status}
            for uid, status in groups.items()
        }
        self.landings = list(landings)
        self.cleared: list[str] = []

    def kv_list(self, partition_id):
        if self.landings:
            key, status = self.landings.pop(0)
            self.tags[key] = {"status": status}
        return {partition_id: dict(self.tags)}

    def kv_clear(self, keys, partition_id):
        self.cleared += keys


class FakeWorkers:
    world_size = 2

    def __init__(self):
        self.lands: list[tuple[list[list[str]], list[list[str]]]] = []
        self.pieces: list[list[str]] = []

    def land(self, completions, verdicts):
        for shards, fields in ((completions, COMPLETION_FIELDS), (verdicts, VERDICT_FIELDS)):
            assert len(shards) == self.world_size
            for meta in filter(None, shards):
                assert meta.fields == list(fields) and meta.extra_info == {"rollout_n": 2}
        keys = [
            [[] if meta is None else meta.keys for meta in shards]
            for shards in (completions, verdicts)
        ]
        self.lands.append(tuple(keys))
        return [f"land{len(self.lands)}"]

    def forward_backward(self, piece):
        self.pieces.append(list(piece.keys))
        return SimpleNamespace(futures=[f"piece{len(self.pieces)}"])


def step_on(monkeypatch, queue: FakeQueue, **thundersync):
    """A trainer on ``queue`` without its constructor (no workers, no Ray)."""
    monkeypatch.setattr(verl_trainer, "tq", queue)
    monkeypatch.setattr(verl_trainer, "POLL_INTERVAL_S", 0)
    trainer = object.__new__(verl_trainer.ThunderSyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 3},
            "actor_rollout_ref": {
                "rollout": {"n": 2, "temperature": 1.0},
                "actor": {
                    "thundersync": thundersync,
                    "ppo_mini_batch_size": 3,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "policy_loss": {},
                },
            },
        }
    )
    trainer.global_steps, trainer.actor_rollout_wg = 1, FakeWorkers()
    return trainer


def owner(key: str) -> int:
    return int(key[1]) % 2


def test_each_trajectory_reaches_its_groups_rank_once_within_the_open_group_bound(monkeypatch):
    # Rank 0 owns u0, u2, u4. With one open group per rank, u2 and u4 wait for
    # u0. When it completes, the fully scored u4 goes first and closes in the
    # same call, so the partial u2 still gets the slot it leaves free.
    order = ["u0_0_0", "u2_0_0", "u4_0_0", "u4_1_0", "u0_1_0", "u2_1_0"]
    order += [f"u{group}_{session}_0" for group in (1, 3, 5) for session in (0, 1)]
    queue = FakeQueue(
        dict.fromkeys(["u0", "u1", "u2", "u3", "u4", "u5"], "running"),
        [(key, "success") for key in order],
    )
    trainer = step_on(monkeypatch, queue, max_concurrent_groups=1)
    batch, calls = trainer._stream_trajectories(6, {})

    lands = trainer.actor_rollout_wg.lands
    assert [completions[0] for completions, _ in lands if completions[0]] == [
        ["u0_0_0"],
        ["u0_1_0", "u4_0_0", "u4_1_0", "u2_0_0"],
        ["u2_1_0"],
    ]
    for kind in (0, 1):
        sent = [
            (rank, key) for call in lands for rank, keys in enumerate(call[kind]) for key in keys
        ]
        assert sorted(key for _, key in sent) == sorted(order)
        assert all(rank == owner(key) for rank, key in sent)
    assert sorted(batch.keys) == sorted(order) and len(calls) == len(lands)
    assert queue.cleared == ["u0", "u1", "u2", "u3", "u4", "u5"]


def test_a_completion_stored_before_its_reward_reaches_the_rank_first(monkeypatch):
    landings = [
        ("u0_0_0", "content"),
        ("u1_0_0", "content"),
        ("u1_1_0", "content"),
        ("u0_0_0", "success"),
        ("u0_1_0", "success"),
        ("u1_0_0", "success"),
        ("u1_1_0", "success"),
    ]
    trainer = step_on(monkeypatch, FakeQueue({"u0": "running", "u1": "running"}, landings))
    trainer._stream_trajectories(2, {})

    def first_call(kind: int, key: str) -> int:
        return next(
            index
            for index, call in enumerate(trainer.actor_rollout_wg.lands)
            if key in call[kind][owner(key)]
        )

    for key in ("u0_0_0", "u1_0_0", "u1_1_0"):
        assert first_call(0, key) < first_call(1, key), key
    assert first_call(0, "u0_1_0") == first_call(1, "u0_1_0")


def test_a_failed_group_stops_the_step(monkeypatch):
    queue = FakeQueue({"u0": "running", "u1": "failure"}, [("u0_0_0", "success")])
    with pytest.raises(RuntimeError, match=r"rollout failed for groups \['u1'\]"):
        step_on(monkeypatch, queue)._stream_trajectories(2, {})


def test_complete_groups_accumulate_once_and_wait_for_the_running_piece(monkeypatch):
    # u0 completes first; u2 and then u1 complete while its piece runs.
    order = ["u0_0_0", "u1_0_0", "u0_1_0", "u2_0_0", "u2_1_0", "u1_1_0"]
    queue = FakeQueue(
        dict.fromkeys(["u0", "u1", "u2"], "running"), [(key, "success") for key in order]
    )
    trainer = step_on(monkeypatch, queue)
    polls_running = iter([1, 1, 1, 0])
    monkeypatch.setattr(
        verl_trainer.ray,
        "wait",
        lambda refs, num_returns, timeout: ([], refs if next(polls_running, 0) else []),
    )
    accumulated = []

    def accumulate(keys, tags, aggregator):
        accumulated.append(keys)
        return KVBatchMeta(partition_id="train", keys=keys, tags=tags)

    trainer._accumulate = accumulate
    batch, futures = trainer._stream_groups(3, {})

    assert accumulated == [["u0_0_0", "u0_1_0"], ["u2_0_0", "u2_1_0", "u1_0_0", "u1_1_0"]]
    assert trainer.actor_rollout_wg.pieces == accumulated
    assert futures == ["piece1", "piece2"] and sorted(batch.keys) == sorted(order)


def test_a_piece_runs_verls_pipeline_and_is_normalized_by_the_whole_batch(monkeypatch):
    trainer = step_on(monkeypatch, FakeQueue({}, []))
    trainer.use_reference_policy, calls = True, []

    def stage(name):
        def run(*args):
            calls.append(name)
            return args[-2]

        return run

    for name in ("_balance_batch", "_compute_ref_log_prob", "_compute_advantage"):
        setattr(trainer, name, stage(name))
    monkeypatch.setattr(
        verl_trainer.PPOTrainer, "_compute_old_log_prob", stage("_compute_old_log_prob")
    )
    piece = trainer._accumulate(
        ["u0_0_0"], [{}], SimpleNamespace(add_step_metrics=lambda *_, **__: None)
    )

    assert calls == [
        "_balance_batch",
        "_compute_old_log_prob",
        "_compute_ref_log_prob",
        "_compute_advantage",
    ]
    assert piece.extra_info["global_batch_size"] == 6 and not piece.extra_info["calculate_entropy"]


def test_the_agent_loop_stores_content_before_verl_scores_it(monkeypatch):
    events = []

    async def put(**kwargs):
        events.append(("content", kwargs["keys"], kwargs["tags"], sorted(kwargs["fields"].keys())))

    async def postprocess(self, output, validate, **kwargs):
        events.append(("scored", validate))

    monkeypatch.setattr(content_first.tq, "async_kv_batch_put", put)
    monkeypatch.setattr(
        content_first.AgentLoopWorkerTQ.__ray_actor_class__, "_agent_loop_postprocess", postprocess
    )
    worker = object.__new__(content_first.ContentFirstAgentLoopWorker)
    output = AgentLoopOutput(
        prompt_ids=[1, 2], response_ids=[3, 4], response_mask=[1, 0], metrics={}
    )
    for validate in (False, True):
        asyncio.run(worker._agent_loop_postprocess(output, validate, uid="u0", session_id=3))

    assert events == [
        ("content", ["u0_3_0"], [{"status": "content"}], sorted(COMPLETION_FIELDS)),
        ("scored", False),
        ("scored", True),
    ]


def test_the_contract_names_verl_keys_and_refuses_verls_defaults():
    config = OmegaConf.load(VERL_CONFIG / "_generated_ppo_trainer.yaml")
    contract = verl_trainer.CONTRACT | verl_trainer.THUNDERSYNC_ENGINE_CONTRACT
    absent = [
        key for key in contract if OmegaConf.select(config, key, default=KeyError) is KeyError
    ]
    assert not absent, f"verl's configuration has no {absent}"
    OmegaConf.update(config, "actor_rollout_ref.actor.strategy", "thundersync")
    with pytest.raises(ConfigError) as refused:
        verl_trainer.check_contract(config)
    for key, value in contract.items():
        assert (key in str(refused.value)) == (OmegaConf.select(config, key) != value), key


def launcher_config(script: str):
    """verl's configuration as a shipped launcher composes it."""
    body = (ROOT / "examples" / script).read_text().split("python3 -m verl.trainer.main_ppo", 1)[1]
    words = shlex.split(body.replace("\\\n", " ").replace('"$@"', ""))
    defaults = {
        "MODEL_PATH": "model",
        "DATA_DIR": "data",
        "TRAIN_GPUS": "4",
        "ROLLOUT_GPUS": "4",
        "EP": "1",
        "BATCH": "64",
    }
    overrides = [re.sub(r"\$(\w+)", lambda match: defaults[match[1]], word) for word in words]
    with initialize_config_dir(config_dir=str(VERL_CONFIG), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


@pytest.mark.parametrize(
    ("script", "actor_config"),
    [("verl_grpo.sh", "ThunderSyncActorConfig"), ("verl_grpo_veomni.sh", "VeOmniActorConfig")],
)
def test_the_launchers_compose_into_accepted_configurations(script, actor_config):
    config = launcher_config(script)
    verl_trainer.check_contract(config)
    validate_config(
        config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config)
    )
    assert type(omega_conf_to_dataclass(config.actor_rollout_ref.actor)).__name__ == actor_config


def test_streaming_groups_refuses_token_mean():
    config = launcher_config("verl_grpo_veomni.sh")
    config.actor_rollout_ref.actor.loss_agg_mode = "token-mean"
    with pytest.raises(ConfigError, match="token-mean"):
        verl_trainer.check_contract(config)

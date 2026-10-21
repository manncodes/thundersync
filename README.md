# ThunderSync

[Paper](https://arxiv.org/abs/2610.05935) · [Blog](https://seilk.github.io/thundersync)

ThunderSync is a Python library for training language-model agents while their
rollouts are still in progress. It provides streaming backward execution for
group-relative policy optimization (GRPO) and on-policy distillation (OPD).

In GRPO, each completed trajectory can contribute a gradient before the rest
of its group finishes. In OPD, each generated action can contribute as soon as
its teacher scores arrive. Parameters stay fixed during a logical batch; the
optimizer runs after all required work has completed.

## Install

ThunderSync runs on Linux with Python 3.12 or newer. The rollout sandboxes
need Docker or uDocker; uDocker's network guard needs libseccomp. Install
PyTorch for your platform, then:

```bash
pip install .
```

Optional extras: `train` (LoRA and checkpoint loading through `peft` and
`safetensors`), `kernels` (Triton, flash-linear-attention and the cuDNN
frontend), `rollout` (vLLM) and `test`. The FA4 attention path additionally
needs a FlashAttention build that provides `flash_attn.cute`.

The examples use small, randomly initialized Qwen3 models and require no model
download or dataset. They run on CPU or an available CUDA device:

```bash
python examples/streaming_grpo.py
python examples/streaming_opd.py
```

The examples live in the source repository, not in the installed package.
Optional CUDA kernels are probed on the executing device; availability is determined
by an actual forward/backward trial.

## Integrating a training loop

The examples show one complete optimizer update. In an application, replace
their token sequences with your rollout events and preserve these boundaries:

1. Open the prompt group and append generated tokens with their scored masks.
2. For GRPO, close a trajectory when its reward becomes available. For OPD,
   submit each action with teacher log probabilities for its scored tokens.
3. Close trajectories and groups after their dependencies finish. OPD may
   replay deferred state adjoints at trajectory closure.
4. Call `assert_safe_to_step()` before the optimizer update. Publish the
   resulting policy only after that update completes.

The GRPO objective is maximized (`maximize=True` in the optimizer); the OPD
objective is minimized.

GRPO's reward-linear objective implements one fixed-snapshot semigradient
epoch with group population-standard-deviation normalization. It is not a
multi-epoch PPO trainer. OPD uses a detached teacher/student log-probability
advantage and normalizes by the scored-token count of the batch it accumulates.

Tensor dtype, accumulation order, batch partitioning and kernel choice can
affect finite-precision results.

## Training with verl

Installing ThunderSync registers a training engine and a v1 trainer with
[verl](https://github.com/volcengine/verl) through verl's `verl.plugins`
entry point. The trainer (`trainer.v1.trainer_mode=thundersync`) runs verl's
separate-async topology and starts the actor update from what has landed in
verl's TransferQueue while the rest of the batch is still generating. The
optimizer steps once, after the whole batch, and the next batch is generated
by the updated policy, so the update stays on-policy. How a landed piece is
trained depends on the engine:

- ThunderSync's engine (`actor_rollout_ref.actor.strategy=thundersync`) keeps
  a full policy replica on each training GPU. A trajectory reaches its group's
  rank when its content lands and when its reward does, and runs through
  ThunderSync's own per-rank flow: `FinalFirstAdmission` (content prepares,
  reward binds and final-first closes under the held-source budget), the
  final forest and cross-rank work claims when enabled, and
  `DataParallelThunderSync.step`.
- Any other verl engine, such as VeOmni (`model_engine=veomni`) or FSDP,
  keeps its sharding and its sequence and expert parallelism. A group goes
  through verl's own update pipeline as soon as verl marks its rollout
  finished, with every trajectory stored for it (padding to the
  data-parallel size, old and reference log-probs, advantages), and the
  engine accumulates its forward-backward, normalized by the whole batch. ThunderSync's reward-linear backward differentiates with
  `autograd.grad`, which bypasses a sharded engine's gradient reduction, so a
  sharded engine streams whole groups rather than single trajectories.

```bash
pip install -e /path/to/verl        # Python 3.12
pip install -e .                    # after verl, so transformers stays at 5.14
bash examples/verl_grpo.sh          # ThunderSync's engine
bash examples/verl_grpo_veomni.sh   # VeOmni
```

verl pins `transformers<5.13` and ThunderSync pins 5.14, so pip reports a
conflict when ThunderSync is installed second. verl's engine, worker and v1
trainer modules import and configure with 5.14; the rollout engine must accept
it too.

The trainer refuses settings under which the streamed update differs from
verl's one-update GRPO step (`CONTRACT` in
`thundersync.integrations.verl.trainer`): one epoch, advantages that need only
their own group, no dynamic sampling, no warmup batches, and rewards scored
as each trajectory lands (no colocated reward model). A sharded engine also
needs a `loss_agg_mode` it can normalize piece by piece
(`STREAMED_LOSS_AGG_MODES`): not `token-mean`, whose normalizer, the batch's
token count, is unknown until the batch has landed, and
`seq-mean-token-sum-norm` only with `actor.loss_scale_factor` set.
ThunderSync's engine also requires its objective's settings
(`THUNDERSYNC_ENGINE_CONTRACT`), and stops the step on a group it cannot
train: one that finishes short of `rollout.n` trajectories, or a session
with more than one output.

The `stream/` metrics report when the first piece was dispatched, when the
last group landed, and the share of the batch dispatched before then.
`+trainer.v1.thundersync.defer=True` holds every dispatch until the whole
batch has landed: the same step without the overlap, for comparison.

With `actor_rollout_ref.actor._target_` set to
`thundersync.integrations.verl.engine.ThunderSyncActorConfig`, ThunderSync's
settings go under `actor_rollout_ref.actor.thundersync`: the names
`GrpoTrainerConfig.from_namespace` reads and the event loop's `LOOP_SETTINGS`
(`max_held_sources`, `prepare_pack_max`, `final_forest_max_tokens`,
`known_reserve_slabs`, `work_claims`, `binary_rewards`). An unknown name is
refused. verl submits a whole batch at once, so `max_concurrent_groups` bounds
the groups each rank holds open, each of which keeps two parameter-sized
statistics on the host.

verl scores a trajectory before it stores it, so content and reward land
together. With `actor_rollout_ref.rollout.agent.agent_loop_manager_class` set
to `thundersync.integrations.verl.agent_loop.ContentFirstAgentLoopManager`,
the content is stored first and ThunderSync differentiates the trajectory
while its reward is computed, which pays off when rewards are slow, as with
graded tool use.

ThunderSync's update points in the direction of verl's GRPO update and is
n·√(n/(n−1)) times as large, where n is `rollout.n`: ThunderSync sums a group's
samples where verl averages them, and normalizes by the population standard
deviation rather than the sample one. Adam is invariant to that factor and
gradient clipping is not; multiply `actor.optim.clip_grad` by it to clip as
verl would. A sharded engine computes verl's update itself.

ThunderSync's engine trains every parameter (no LoRA) of a model whose
replica, gradients and optimizer state fit on one GPU;
`optimizer_state_sharding` shards the optimizer state across the training
ranks. Rewards must be computed as each trajectory finishes, as verl does for
rule-based rewards and for reward models with their own resource pool.

## Modules

| Module | Purpose |
| --- | --- |
| `thundersync.engine` | Streaming execution, causal segment forests, boundary adjoints and optimizer sharding. |
| `thundersync.grpo` | GRPO configuration, the reward-linear streaming objective and its data-parallel trainer. |
| `thundersync.opd` | Action-ready on-policy distillation objectives. |
| `thundersync.scheduling` | Source admission and cross-rank work claims. |
| `thundersync.rollout` | Agent rollout, bounded tool output and Docker/uDocker sandboxes. |
| `thundersync.accel` | Attention, log-probability, normalization and LoRA kernels. |
| `thundersync.integrations.verl` | The verl engine, streaming v1 trainer and content-first agent loop. |

Multi-process training uses PyTorch distributed. Applications supply their
own model, data, rollout service and device allocation.

## Environment variables

| Variable | Effect |
| --- | --- |
| `THUNDERSYNC_FA4` | Streaming attention kernel: `auto` (default), `require` or `off`. |
| `THUNDERSYNC_SANDBOX_SCRATCH` | Host scratch root for sandbox working trees; must be owned by the current user. |
| `THUNDERSYNC_DOCKER_BIN`, `THUNDERSYNC_UDOCKER_BIN` | Container CLI to invoke. |
| `THUNDERSYNC_DOCKER_SEED_DIR` | Where unpacked image working trees are cached; must be owned by the current user. |
| `UDOCKER_DIR`, `PROOT_TMP_DIR` | uDocker's image store and PRoot's temporary directory. |
| `THUNDERSYNC_OPERATOR_PROFILE_RANGES` | Profiler ranges around kernels: `off` (default), `kineto`, `nvtx` or `both`. |
| `THUNDERSYNC_GRPO_CLOSE_TRACE` | Record per-pack timing of the GRPO group close. |
| `THUNDERSYNC_GRPO_PROFILE_*` | Capture a `torch.profiler` window around selected close packs. |

The profiling variables are diagnostics and change no result.

## Citation

```bibtex
@article{kang2026thundersyncrl,
  title   = {ThunderSyncRL: Lossless Acceleration of Agentic Reinforcement Learning},
  author  = {Kang, Seil and Kang, Hangoo and Suresh, Tarun and Kim, Youngeun and
             Pimpalgaonkar, Shreyas and Hwang, Seong Jae and Mirhoseini, Azalia},
  journal = {arXiv preprint arXiv:2610.05935},
  year    = {2026}
}
```

## License

Apache-2.0.

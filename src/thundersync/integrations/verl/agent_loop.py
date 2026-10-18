"""verl's TransferQueue agent loop, storing each trajectory before scoring it.

ThunderSync differentiates a trajectory as soon as its content is final and
binds its reward when the reward lands. verl scores a trajectory before it
stores it, so both arrive at once; this loop stores the content first, tagged
``content``, then scores and stores the trajectory as verl does, which retags
it ``success``. Select it with
``actor_rollout_ref.rollout.agent.agent_loop_manager_class``; it pays off when
rewards are slow to compute, as with graded tool use.
"""

from __future__ import annotations

import ray
import torch
import transfer_queue as tq
from verl.experimental.agent_loop import AgentLoopOutput
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

__all__ = ["ContentFirstAgentLoopManager"]


class ContentFirstAgentLoopWorker(AgentLoopWorkerTQ.__ray_actor_class__):
    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        # An agent loop with several outputs per session is stored as verl does.
        if not validate and isinstance(output, AgentLoopOutput):
            content = {
                "uid": kwargs["uid"],
                "session_id": kwargs["session_id"],
                "prompts": torch.tensor(output.prompt_ids, dtype=torch.int64),
                "responses": torch.tensor(output.response_ids, dtype=torch.int64),
                "response_mask": torch.tensor(output.response_mask, dtype=torch.int64),
            }
            await tq.async_kv_batch_put(
                keys=[f"{kwargs['uid']}_{kwargs['session_id']}_0"],
                partition_id="train",
                fields=list_of_dict_to_tensordict([content]),
                tags=[{"status": "content"}],
            )
        return await super()._agent_loop_postprocess(output, validate, **kwargs)


class ContentFirstAgentLoopManager(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(ContentFirstAgentLoopWorker)

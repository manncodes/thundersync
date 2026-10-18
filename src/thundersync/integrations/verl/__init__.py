"""ThunderSync for verl: the ``thundersync`` engine backend and v1 trainer.

verl imports this package through its ``verl.plugins`` entry point, so every
verl process registers both. ``trainer.v1.trainer_mode=thundersync`` streams
the actor update into the rollout, on ThunderSync's engine
(``actor_rollout_ref.actor.strategy=thundersync``) or on any other verl
engine, such as VeOmni. The content-first agent loop
(``thundersync.integrations.verl.agent_loop``) is selected by name.
"""

from thundersync.integrations.verl import engine, trainer  # noqa: F401

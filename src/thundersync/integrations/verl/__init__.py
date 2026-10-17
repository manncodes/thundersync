"""ThunderSync for verl: the ``thundersync`` engine backend and v1 trainer.

verl imports this package through its ``verl.plugins`` entry point, so every
verl process registers both. Select them with
``actor_rollout_ref.actor.strategy=thundersync`` and
``trainer.v1.trainer_mode=thundersync``.
"""

from thundersync.integrations.verl import engine, trainer  # noqa: F401

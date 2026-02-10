"""Training loop scaffold.

Goal: isolate the optimizer/accelerate training loop.
"""

from __future__ import annotations


def train_loop(*_args, **_kwargs):
    """Placeholder training loop.

    The concrete loop still lives in train.py. This is a refactor target.
    """
    raise NotImplementedError(
        "Scaffold only. Migrate launch_training_task_with_accum_logging from train.py here."
    )

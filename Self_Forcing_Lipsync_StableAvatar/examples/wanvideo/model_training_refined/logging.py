"""Logging and checkpointing scaffold."""

from __future__ import annotations

from diffsynth.trainers.utils import ModelLogger


def build_model_logger(args):
    return ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_full_checkpoint_steps=getattr(args, "save_full_checkpoint_steps", None),
    )

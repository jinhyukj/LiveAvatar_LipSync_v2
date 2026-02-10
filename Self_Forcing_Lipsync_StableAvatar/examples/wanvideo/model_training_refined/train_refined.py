"""Refined entrypoint scaffold (non-invasive).

This file shows the target structure for a slimmer train.py without changing
any existing training code.
"""

from __future__ import annotations

from .cli import parse_args
from .data import build_train_dataset, build_val_dataset
from .modeling import build_model
from .logging import build_model_logger
from .validation import build_sync_evaluator
from .loop import train_loop


def main():
    args = parse_args()

    # Build datasets
    train_dataset = build_train_dataset(args)
    val_dataset = build_val_dataset(args)

    # Build model + logger
    model = build_model(args)
    model_logger = build_model_logger(args)

    # Optional sync metrics evaluator
    has_val_sets = bool(val_dataset) and len(val_dataset) > 0
    sync_evaluator = build_sync_evaluator(args, has_val_sets)
    _ = sync_evaluator  # Reserved for future loop signature.

    # Run training loop (placeholder)
    train_loop(train_dataset, model, model_logger, args, val_dataset=val_dataset)


if __name__ == "__main__":
    main()

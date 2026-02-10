"""Validation/sync-metrics scaffold."""

from __future__ import annotations

from examples.wanvideo.model_training.sync_metrics import SyncMetricsEvaluator


def build_sync_evaluator(args, has_val_sets: bool):
    if not getattr(args, "enable_sync_metrics", False):
        return None
    if not has_val_sets:
        return None
    try:
        return SyncMetricsEvaluator(
            syncnet_model_path=args.syncnet_model_path,
            device="cuda",
            temp_base_dir="/tmp/latentsync_sync_eval",
            s3fd_model_path=getattr(args, "s3fd_model_path", None),
        )
    except Exception:
        return None

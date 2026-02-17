"""
SyncNet evaluation module for audio-visual synchronization metrics.

This module provides self-contained SyncNet evaluation capabilities,
including face detection, tracking, and sync metric computation.
"""

from .syncnet.syncnet_eval import SyncNetEval
from .syncnet_detect import SyncNetDetector
from .eval_sync import syncnet_eval

__all__ = [
    "SyncNetEval",
    "SyncNetDetector",
    "syncnet_eval",
]

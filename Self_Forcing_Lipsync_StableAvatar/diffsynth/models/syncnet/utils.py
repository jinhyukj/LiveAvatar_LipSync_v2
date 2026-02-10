"""
Utility functions for SyncNet evaluation.
Replacements for latentsync.utils.util imports.
"""

import os


def red_text(text):
    """Return red-colored text for terminal output."""
    return f"\033[91m{text}\033[0m"


def check_model_and_download(model_path):
    """
    Check if model exists at the given path.

    Args:
        model_path: Path to the model file

    Returns:
        model_path if exists

    Raises:
        FileNotFoundError: If model doesn't exist

    Note:
        Original function in LatentSync downloads from URL if missing.
        For now, we just check existence.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please ensure the model checkpoint is available at this path."
        )
    return model_path

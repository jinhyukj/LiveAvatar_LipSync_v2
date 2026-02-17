"""
Utility functions for SyncNet evaluation.
"""

import os


def red_text(text):
    """Return red-colored text for terminal output."""
    return f"\033[91m{text}\033[0m"


def check_model_and_download(model_path):
    """
    Check if model exists at the given path.

    Raises:
        FileNotFoundError: If model doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please ensure the model checkpoint is available at this path."
        )
    return model_path

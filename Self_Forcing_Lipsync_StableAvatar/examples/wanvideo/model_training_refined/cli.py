"""CLI parsing scaffold.

Goal: isolate CLI argument definitions from the training logic.
Keep this file thin and declarative.
"""

from __future__ import annotations

from diffsynth.trainers.utils import wan_parser


def build_parser():
    """Return a parser. Extend here when migrating args from train.py.

    Note: this is a scaffold. The full argument list still lives in
    examples/wanvideo/model_training/train.py.
    """
    parser = wan_parser()
    # TODO: migrate extra add_argument() calls from train.py into this module.
    return parser


def parse_args():
    parser = build_parser()
    return parser.parse_args()

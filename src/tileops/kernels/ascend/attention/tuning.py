"""Finite launch/synchronization search for the already verified decode radix program."""

from .radix_select import topk_select_ascend

DEFAULT_CONFIG = {"H": 128, "PARALLEL_SCAN": True}
TUNING_CONFIGS = [
    DEFAULT_CONFIG,
    {"H": 64, "PARALLEL_SCAN": False},
    {"H": 128, "PARALLEL_SCAN": False},
    {"H": 256, "PARALLEL_SCAN": False},
]


def decode_select_prim(H: int = 128, PARALLEL_SCAN: bool = True):
    """Expose effective tuning parameters without changing the device algorithm.

    The output remains tensor argument 3; an AutoTuner compiles with out_idx=[3].
    Integer window inputs must be supplied explicitly; random integers are invalid.
    """
    return topk_select_ascend.get_tir(1, 16384, 2048, 1, 1, H=H, PARALLEL_SCAN=PARALLEL_SCAN)

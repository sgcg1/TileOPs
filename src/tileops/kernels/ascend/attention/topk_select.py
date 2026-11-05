"""Windowed exact radix selector on Ascend; initial public-interface integration."""

import torch

import tileops.backend
from tileops.kernels.attention.topk_select import TopKSelectCall, TopKSelectFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

from .radix_select import topk_select_ascend


class AscendTopKSelectKernel(Kernel, TopKSelectFwdInterface):
    """Exact selection for the frozen FP32 decode-b1 workload."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: TopKSelectCall):
        if (call.batch, call.seq_len, call.seq_len_kv, call.kv_group, call.topk) != (
            1, 1, 16384, 1, 2048
        ):
            return "initial integration serves only ds-v32-topk-decode-b1"
        if call.dtype != torch.float32 or call.out_dtype != torch.int32:
            return "requires float32 scores and int32 output"
        return None

    @classmethod
    def entry_for(cls, call: TopKSelectCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: TopKSelectCall):
        super().__init__()
        self.call = call
        # A single row uses one block; no assumption about 28 vs 32 available cores.
        self.program = topk_select_ascend(1, 16384, 2048, 1, 1)

    def forward(self, index_score, starts, ends):
        output = torch.empty((1, 1, 1, 2048), dtype=torch.int32, device=index_score.device)
        self.program(index_score.reshape(1, 16384, 1), starts.reshape(1),
                     ends.reshape(1), output.reshape(1, 2048))
        return output


tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend", AscendTopKSelectKernel)

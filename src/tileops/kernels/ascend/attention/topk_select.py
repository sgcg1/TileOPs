"""Windowed exact radix selector through the public Ascend interface."""

import torch

import tileops.backend
from tileops.kernels.attention.topk_select import TopKSelectCall, TopKSelectFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

from .radix_select import topk_select_ascend


class AscendTopKSelectKernel(Kernel, TopKSelectFwdInterface):
    """Exact FP32 selection for the public batch/query/group input contract."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: TopKSelectCall):
        if call.dtype != torch.float32 or call.out_dtype != torch.int32:
            return "requires float32 scores and int32 output"
        return None

    @classmethod
    def entry_for(cls, call: TopKSelectCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: TopKSelectCall):
        super().__init__()
        self.call = call
        self.rows = call.batch * call.seq_len * call.kv_group
        # Current development uses the 28-core PR card; workers stride over all rows.
        self.program = (
            topk_select_ascend(
                self.rows, call.seq_len_kv, call.topk, min(self.rows, 28), call.kv_group
            )
            if self.rows else None
        )

    def forward(self, index_score, starts, ends):
        call = self.call
        output = torch.empty(
            (call.batch, call.seq_len, call.kv_group, call.topk),
            dtype=torch.int32, device=index_score.device,
        )
        if self.program is not None:
            self.program(
                index_score.reshape(call.batch * call.seq_len, call.seq_len_kv, call.kv_group),
                starts.reshape(call.batch * call.seq_len),
                ends.reshape(call.batch * call.seq_len),
                output.reshape(self.rows, call.topk),
            )
        return output


tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend", AscendTopKSelectKernel)

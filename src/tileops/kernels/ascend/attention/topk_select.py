"""Windowed exact radix selector through the public Ascend interface."""

import torch

import tileops.backend
from tileops.kernels.attention.topk_select import TopKSelectCall, TopKSelectFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

from .radix_select import topk_select_ascend
from .simd_select import simd_workspace_bytes, topk_select_simd


class AscendTopKSelectKernel(Kernel, TopKSelectFwdInterface):
    """Exact FP32 selection for the public batch/query/group input contract."""

    devices = frozenset({"npu"})
    # Keep the measured general implementation until the SIMD candidate is validated.
    preferred_over = frozenset({"ascend_simd"})

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


class AscendTopKSelectSIMDKernel(Kernel, TopKSelectFwdInterface):
    """Resident SIMD candidate selected by resource constraints, not workload names."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: TopKSelectCall):
        if call.dtype != torch.float32 or call.out_dtype != torch.int32:
            return "requires float32 scores and int32 output"
        if not 0 < call.topk <= call.seq_len_kv < 65536:
            return "resident uint16 histogram requires 0 < K <= N < 65536"
        if simd_workspace_bytes(call.seq_len_kv, call.topk, call.kv_group) > 96 * 1024:
            return "resident SIMD plan exceeds its 96 KiB workspace budget"
        return None

    @classmethod
    def entry_for(cls, call: TopKSelectCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: TopKSelectCall):
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__()
        self.call = call
        self.rows = call.batch * call.seq_len * call.kv_group
        self.program = (
            topk_select_simd(
                self.rows, call.seq_len_kv, call.topk, call.kv_group, min(self.rows, 28)
            ) if self.rows else None
        )

    def forward(self, index_score, starts, ends):
        call = self.call
        output = torch.empty(
            (call.batch, call.seq_len, call.kv_group, call.topk),
            dtype=torch.int32, device=index_score.device,
        )
        if self.program is not None:
            ramp = torch.arange(64, dtype=torch.int32, device=index_score.device)
            self.program(
                index_score.view(torch.uint32).reshape(
                    call.batch * call.seq_len, call.seq_len_kv * call.kv_group
                ),
                starts.reshape(call.batch * call.seq_len),
                ends.reshape(call.batch * call.seq_len), ramp.view(torch.uint32),
                output.view(torch.uint32).reshape(self.rows, call.topk),
            )
        return output


tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend", AscendTopKSelectKernel)
tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend_simd", AscendTopKSelectSIMDKernel)

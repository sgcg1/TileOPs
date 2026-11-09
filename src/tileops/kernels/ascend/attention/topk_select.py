"""Windowed exact radix selector through the public Ascend interface."""

import torch

import tileops.backend
from tileops.kernels.attention.topk_select import TopKSelectCall, TopKSelectFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

from .hierarchical_select import pack_candidates, pack_shards, remap_candidates
from .radix_select import topk_select_ascend
from .simd_select import simd_workspace_bytes, topk_select_simd
from .streaming_select import streaming_workspace_bytes, topk_select_streaming


class AscendTopKSelectKernel(Kernel, TopKSelectFwdInterface):
    """Exact FP32 selection for the public batch/query/group input contract."""

    devices = frozenset({"npu"})
    # Keep the measured general implementation until the SIMD candidate is validated.
    preferred_over = frozenset({"ascend_simd", "ascend_hierarchical", "ascend_simd_compact"})

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
    compress_output = False
    output_tiles_per_vf = 1
    early_bucket = False

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
                self.rows, call.seq_len_kv, call.topk, call.kv_group, min(self.rows, 28),
                self.compress_output, self.output_tiles_per_vf, self.early_bucket,
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


class AscendTopKSelectCompactKernel(AscendTopKSelectSIMDKernel):
    """Same exact threshold selector with mask compression instead of double scan."""

    compress_output = True


class AscendTopKSelectGroupedCompactKernel(AscendTopKSelectCompactKernel):
    """Explicit experimental VF batch size; no automatic dispatch promotion."""

    def __init__(self, call: TopKSelectCall, tiles_per_vf=4, early_bucket=False):
        if tiles_per_vf not in (4, 8, 16):
            raise ValueError("tiles_per_vf must be 4, 8 or 16")
        self.output_tiles_per_vf = tiles_per_vf
        self.early_bucket = early_bucket
        super().__init__(call)


class AscendTopKSelectResidentKernel(AscendTopKSelectCompactKernel):
    """Resource-bounded resident selector with grouped output and bucket exit."""

    output_tiles_per_vf = 16
    early_bucket = True
    preferred_over = frozenset({
        "ascend", "ascend_simd", "ascend_simd_compact", "ascend_hierarchical",
    })


class AscendTopKSelectStreamingKernel(AscendTopKSelectSIMDKernel):
    """Explicit long-row experiment; reuse the public tensor binding contract."""

    @classmethod
    def refusal(cls, call: TopKSelectCall):
        if call.dtype != torch.float32 or call.out_dtype != torch.int32:
            return "requires float32 scores and int32 output"
        if not 0 < call.topk <= call.seq_len_kv < 2147483647:
            return "requires positive K <= N within int32 count range"
        if streaming_workspace_bytes(call.topk, call.kv_group) > 96 * 1024:
            return "streaming buffers exceed 96 KiB"
        return None

    def __init__(self, call: TopKSelectCall, block=0, fuse_keys=False, compact=False, second_bucket=False, fuse_compact_hist=False, *, worker_limit=32, skip_full_clear=False, skip_output_init=False, merge_hist_in_vf=False):
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        if block not in (0, 2048, 8192, 16384):
            raise ValueError("unsupported streaming block size")
        if streaming_workspace_bytes(call.topk, call.kv_group, block) > 96 * 1024:
            raise ValueError("requested streaming block exceeds 96 KiB")
        if worker_limit not in (28, 32):
            raise ValueError("streaming worker limit must be 28 or 32")
        Kernel.__init__(self)
        self.call = call
        self.rows = call.batch * call.seq_len * call.kv_group
        self.workers = min(self.rows, worker_limit)
        self.program = topk_select_streaming(
            self.rows, call.seq_len_kv, call.topk, call.kv_group, self.workers,
            block, fuse_keys, compact, second_bucket, fuse_compact_hist, skip_full_clear, skip_output_init, merge_hist_in_vf,
        ) if self.rows else None


class AscendTopKSelectHierarchicalKernel(Kernel, TopKSelectFwdInterface):
    """Experimental exact local-select/candidate-select route for few long rows."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: TopKSelectCall):
        if call.dtype != torch.float32 or call.out_dtype != torch.int32:
            return "requires float32 scores and int32 output"
        return cls.plan_refusal(call, 2)

    @staticmethod
    def plan_refusal(call, shards):
        if shards not in (2, 4, 8, 16, 28):
            return "unsupported shard count"
        n, k = call.seq_len_kv, call.topk
        if not 0 < k <= n:
            return "requires 0 < K <= N"
        length = ((n + shards * 64 - 1) // (shards * 64)) * 64
        local_k = min(k, length)
        candidates = shards * local_k
        if max(length, candidates) >= 65536:
            return "resident local/candidate selector requires length < 65536"
        if max(simd_workspace_bytes(length, local_k, 1),
               simd_workspace_bytes(candidates, k, 1)) > 96 * 1024:
            return "local/candidate selector exceeds 96 KiB workspace budget"
        return None

    @classmethod
    def entry_for(cls, call: TopKSelectCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: TopKSelectCall, shards=2):
        reason = self.plan_refusal(call, shards)
        if call.dtype != torch.float32 or call.out_dtype != torch.int32:
            reason = "requires float32 scores and int32 output"
        if reason is not None:
            raise ValueError(reason)
        super().__init__()
        self.call, self.shards = call, shards
        self.rows = call.batch * call.seq_len * call.kv_group
        self.length = ((call.seq_len_kv + shards * 64 - 1) // (shards * 64)) * 64
        self.local_k = min(call.topk, self.length)
        self.candidates = shards * self.local_k
        if self.rows:
            r, n, g, p, l, j = (self.rows, call.seq_len_kv, call.kv_group,
                                shards, self.length, self.local_k)
            w = min(r * p, 28)
            self.pack = pack_shards(r, n, g, p, l, w)
            self.local = topk_select_simd(r * p, l, j, 1, w)
            self.collect = pack_candidates(r, n, g, p, l, j, w)
            self.finish = topk_select_simd(r, self.candidates, call.topk, 1, min(r, 28))
            self.remap = remap_candidates(r, n, call.topk, self.candidates, min(r, 28))

    def forward(self, index_score, starts, ends):
        c, r, p, l, j = self.call, self.rows, self.shards, self.length, self.local_k
        options = dict(dtype=torch.int32, device=index_score.device)
        output = torch.empty((c.batch, c.seq_len, c.kv_group, c.topk), **options)
        if not r:
            return output
        packed = torch.empty((r * p, l), **options)
        lo, hi = torch.empty((r * p,), **options), torch.empty((r * p,), **options)
        local_ids = torch.empty((r * p, j), **options)
        scores = torch.empty((r, self.candidates), **options)
        mapping = torch.empty_like(scores)
        begin, end = torch.empty((r,), **options), torch.empty((r,), **options)
        final_ids = torch.empty((r, c.topk), **options)
        ramp = torch.arange(64, **options).view(torch.uint32)
        bits = index_score.view(torch.uint32).reshape(c.batch * c.seq_len, c.seq_len_kv * c.kv_group)
        self.pack(bits, starts.reshape(-1), ends.reshape(-1), packed.view(torch.uint32), lo, hi)
        self.local(packed.view(torch.uint32), lo, hi, ramp, local_ids.view(torch.uint32))
        self.collect(bits, lo, hi, local_ids.view(torch.uint32), scores.view(torch.uint32),
                     mapping.view(torch.uint32), begin, end)
        self.finish(scores.view(torch.uint32), begin, end, ramp, final_ids.view(torch.uint32))
        self.remap(final_ids.view(torch.uint32), mapping.view(torch.uint32),
                   output.view(torch.uint32).reshape(r, c.topk))
        return output


tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend", AscendTopKSelectKernel)
tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend_simd", AscendTopKSelectSIMDKernel)
tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend_hierarchical", AscendTopKSelectHierarchicalKernel)

tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend_simd_compact", AscendTopKSelectCompactKernel)
tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend_resident", AscendTopKSelectResidentKernel)

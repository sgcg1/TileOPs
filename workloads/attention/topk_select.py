import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase


def _score_keys(scores: torch.Tensor) -> torch.Tensor:
    """Exact FP32 order: NaNs first, signed zeros equal, no float comparisons."""
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    magnitude = bits & 0x7FFFFFFF
    keys = torch.where((bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    keys = torch.where(magnitude == 0, 0x80000000, keys)
    return torch.where(magnitude > 0x7F800000, 0xFFFFFFFF, keys)


class TopKSelectWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        kv_group: int,
        topk: int,
        in_dtype: torch.dtype,
        out_dtype: torch.dtype,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.randn(
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.kv_group,
            dtype=self.in_dtype,
            device=run_device(),
        )
        starts = torch.zeros(self.batch, self.seq_len, dtype=self.out_dtype, device=run_device())
        ends = (
            torch.ones(self.batch, self.seq_len, dtype=self.out_dtype, device=run_device())
            * self.seq_len_kv
        )
        return logits, starts, ends

    def ref_program(
        self, logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        # Window positions are ascending, so stable key sorting resolves ties by index.
        positions = (
            starts[:, :, None, None].long()
            + torch.arange(logits.shape[2], device=logits.device)[None, None, :, None]
        )
        valid = (positions < ends[:, :, None, None]) & (positions < logits.shape[2])
        window = logits.gather(2, positions.clamp(max=logits.shape[2] - 1).expand_as(logits))
        keys = _score_keys(window).masked_fill(~valid, -1)
        local = torch.argsort(keys, dim=2, descending=True, stable=True)[:, :, : self.topk, :]
        indexes = positions.expand_as(logits).gather(2, local)
        in_window = valid.expand_as(logits).gather(2, local)
        indexes = torch.where(in_window, indexes, logits.shape[2])
        return indexes.permute(0, 1, 3, 2).to(self.out_dtype)

    def verification(self, *inputs):
        from workloads.numerics import Custom

        def validate(got, expected):
            assert got.shape == expected.shape and got.dtype == expected.dtype
            scores, starts, ends = inputs
            padding = got == scores.shape[2]
            assert (
                padding | ((got >= starts[:, :, None, None]) & (got < ends[:, :, None, None]))
            ).all()
            required_padding = self.topk - (ends - starts).clamp(min=0, max=self.topk)
            assert torch.equal(
                padding.sum(-1), required_padding[:, :, None].expand_as(padding.sum(-1))
            )
            ordered = got.sort(-1).values
            assert (
                (ordered[..., 1:] != ordered[..., :-1]) | (ordered[..., 1:] == scores.shape[2])
            ).all(), "duplicate selected index"
            scores = torch.nn.functional.pad(_score_keys(scores).movedim(2, -1), (0, 1), value=-1)
            got_values = scores.gather(-1, got.long()).sort(-1).values
            expected_values = scores.gather(-1, expected.long()).sort(-1).values
            assert torch.equal(got_values, expected_values), "selected FP32 score keys differ"

        return Custom(validate, "selected values, unique indices and exact padding")


class TopKSelectCall(CallWorkload, TopKSelectWorkload):
    """A manifest call of TopKSelectFwdOp; the row's generators give the windows."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices = call.indices
        TopKSelectWorkload.__init__(
            self,
            indices["B"],
            indices["S"],
            indices["S_kv"],
            indices["G"],
            indices["top_k"],
            torch.float32,
            torch.int32,
        )

    gen_inputs = CallWorkload.gen_inputs

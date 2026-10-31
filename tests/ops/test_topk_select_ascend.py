"""Device correctness and framework integration for the Ascend selector."""

import pytest
import torch

pytestmark = pytest.mark.smoke

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("Ascend NPU required", allow_module_level=True)

import tileops.ascend  # noqa: E402, F401 -- optional NPU dependency checked above
from tileops.attention import TopKSelectFwdOp  # noqa: E402


def _reference(scores, starts, ends, k):
    b, s, length, groups = scores.shape
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    magnitude = bits & 0x7FFFFFFF
    keys = torch.where(bits & 0x80000000 != 0, ~bits & 0xFFFFFFFF, bits | 0x80000000)
    keys = torch.where(magnitude == 0, 0x80000000, keys)
    keys = torch.where(magnitude > 0x7F800000, 0xFFFFFFFF, keys)
    out = torch.full((b, s, groups, k), length, dtype=torch.int32)
    for batch in range(b):
        for query in range(s):
            start, end = int(starts[batch, query]), int(ends[batch, query])
            count = max(0, min(k, end - start))
            for group in range(groups):
                order = keys[batch, query, start:end, group].argsort(descending=True, stable=True)
                out[batch, query, group, :count] = (order[:count] + start).to(torch.int32)
    return out


@pytest.mark.parametrize(
    "shape,k,kind,strided",
    [
        ((1, 1, 1, 1), 1, "random", False),
        ((2, 3, 257, 3), 31, "random", False),
        ((1, 4, 257, 2), 128, "special", False),
        ((1, 3, 257, 1), 257, "special", False),
        ((1, 3, 4097, 2), 3073, "ties", False),
        ((1, 3, 4097, 1), 2048, "random", True),
        ((1, 2, 65536, 1), 1024, "random", False),
        ((1, 3, 8193, 2), 2048, "constant", False),
        ((1, 1, 65536, 1), 2048, "constant", False),
        ((1, 1, 257, 1), 193, "special", False),
        ((1, 1, 257, 1), 231, "special", False),
        ((1, 1, 8193, 1), 2048, "half_boundaries", False),
        ((1, 1, 8193, 1), 6144, "half_boundaries", False),
    ],
)
def test_ascend_topk(shape, k, kind, strided):
    torch.manual_seed(20261030)
    scores = torch.randn(shape)
    if kind == "ties":
        scores = scores.round()
    elif kind == "constant":
        scores.zero_()
    elif kind == "half_boundaries":
        anchors = torch.tensor(
            [
                0.0,
                2**-25,
                2**-24,
                2**-14,
                1 - 2**-12,
                1 + 2**-11,
                65504.0,
                65520.0,
                torch.inf,
                torch.nan,
            ],
            dtype=torch.float32,
        )
        anchors = torch.cat((anchors, -anchors))
        raw = torch.stack(
            (
                torch.nextafter(anchors, torch.full_like(anchors, -torch.inf)),
                anchors,
                torch.nextafter(anchors, torch.full_like(anchors, torch.inf)),
            ),
            dim=1,
        ).flatten()
        scores = raw.repeat((scores.numel() + raw.numel() - 1) // raw.numel())[
            : scores.numel()
        ].reshape(shape)
    elif kind == "special":
        raw = torch.tensor(
            [
                0x7F800001,
                -0x3FFFFF,
                0x7F800000,
                -0x800000,
                0,
                -0x80000000,
                1,
                -0x7FFFFFFF,
                0x3F800001,
                0x3F800000,
            ],
            dtype=torch.int32,
        ).view(torch.float32)
        scores = raw.repeat((scores.numel() + 9) // 10)[: scores.numel()].reshape(shape)
    starts = torch.zeros(shape[:2], dtype=torch.int32)
    ends = torch.full(shape[:2], shape[2], dtype=torch.int32)
    if shape[1] >= 3:
        starts[:, 1], ends[:, 1] = 3, min(shape[2], 9)
        starts[:, 2], ends[:, 2] = 7, 2  # Empty reversed window.
    expected = _reference(scores, starts, ends, k)
    device_scores = scores.npu()
    if strided:
        storage = torch.empty((*shape[:-1], shape[-1] * 2), device="npu", dtype=torch.float32)
        storage[..., ::2] = device_scores
        device_scores = storage[..., ::2]
    device_starts, device_ends = starts.npu(), ends.npu()
    original_bits = device_scores.cpu().contiguous().view(torch.int32).clone()
    op = TopKSelectFwdOp(k, target="ascend")
    for _ in range(3):
        actual = op(device_scores, device_starts, device_ends)
        assert actual.shape == expected.shape and actual.dtype == torch.int32
        assert actual.is_contiguous()
        assert actual.data_ptr() != device_scores.data_ptr()
        assert torch.equal(actual.cpu().sort(-1).values, expected.sort(-1).values)
    assert torch.equal(device_scores.cpu().contiguous().view(torch.int32), original_bits)


def test_ascend_compile_boundary():
    scores = torch.randn(1, 2, 257, 1, device="npu")
    starts = torch.zeros(1, 2, dtype=torch.int32, device="npu")
    ends = torch.full_like(starts, 257)
    op = TopKSelectFwdOp(31, target="ascend")
    compiled = torch.compile(op, backend="eager", fullgraph=True)
    expected = _reference(scores.cpu(), starts.cpu(), ends.cpu(), 31)
    actual = compiled(scores, starts, ends)
    assert torch.equal(actual.cpu().sort(-1).values, expected.sort(-1).values)

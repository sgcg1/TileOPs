"""General public-shape regressions for Ascend windowed TopK selection."""

import pytest
import torch

from tileops.attention import TopKSelectFwdOp
from workloads.attention.topk_select import TopKSelectWorkload
from workloads.device import run_device
from workloads.numerics import Request, verify


@pytest.mark.parametrize(
    "batch,queries,length,groups,topk,layout",
    [
        pytest.param(1, 1, 1, 1, 1, "contiguous", id="single-key", marks=pytest.mark.smoke),
        pytest.param(2, 3, 257, 3, 17, "contiguous", id="batch-and-groups", marks=pytest.mark.full),
        pytest.param(1, 29, 129, 1, 33, "contiguous", id="worker-stride-tail", marks=pytest.mark.full),
        pytest.param(2, 5, 513, 3, 65, "contiguous", id="group-row-stride-tail", marks=pytest.mark.full),
        pytest.param(1, 1, 127, 1, 126, "contiguous", id="almost-whole-row", marks=pytest.mark.full),
        pytest.param(1, 1, 128, 1, 128, "contiguous", id="whole-row", marks=pytest.mark.full),
        pytest.param(1, 1, 129, 1, 1, "contiguous", id="maximum-only", marks=pytest.mark.full),
        pytest.param(1, 2, 4097, 1, 3072, "contiguous", id="k-beyond-cache", marks=pytest.mark.full),
        pytest.param(1, 2, 65536, 2, 1024, "contiguous", id="prefill-axis-k1024", marks=pytest.mark.full),
        pytest.param(1, 3, 65536, 1, 2048, "contiguous", id="prefill-axis-k2048", marks=pytest.mark.full),
        pytest.param(2, 3, 257, 2, 17, "strided", id="noncontiguous-inputs", marks=pytest.mark.full),
        pytest.param(0, 3, 129, 2, 32, "contiguous", id="empty-batch", marks=pytest.mark.full),
        pytest.param(2, 0, 129, 2, 32, "contiguous", id="empty-queries", marks=pytest.mark.full),
        pytest.param(2, 3, 129, 0, 32, "contiguous", id="empty-groups", marks=pytest.mark.full),
    ],
)
def test_topk_select_ascend_public_shapes(batch, queries, length, groups, topk, layout):
    """Guard B/S/G row mapping, worker tail, legal K, empty axes and contiguous copies."""
    if torch.device(run_device()).type != "npu":
        pytest.skip("Ascend public-shape regression")
    device = run_device()
    storage_groups = groups * 2 if layout == "strided" else groups
    scores = torch.randn(batch, queries, length, storage_groups, device=device)
    if layout == "strided":
        scores = scores[..., ::2]
    windows = torch.empty(batch, queries, 2, dtype=torch.int32, device=device)
    for b in range(batch):
        for s in range(queries):
            mode = (b * queries + s) % 6
            if mode == 0:
                start, end = 0, length
            elif mode == 1:
                start, end = length, length
            elif mode == 2:
                start = min(1, length)
                end = min(length, start + topk - 1)
            elif mode == 3:
                start = (length - topk) // 2
                end = start + topk
            elif mode == 4:
                start, end = 0, min(length, topk + 1)
            else:
                # Both bounds satisfy the manifest; its interval/reference is empty.
                start, end = min(1, length), 0
            windows[b, s, 0], windows[b, s, 1] = start, end
    starts, ends = windows[..., 0], windows[..., 1]
    inputs = scores, starts, ends
    workload = TopKSelectWorkload(batch, queries, length, groups, topk, torch.float32, torch.int32)
    op = TopKSelectFwdOp(topk=topk)
    selected = []

    def run(*args):
        bits = args[0].cpu().contiguous().view(torch.int32).clone()
        output = op(*args)
        assert torch.equal(args[0].cpu().contiguous().view(torch.int32), bits)
        selected.append(output.cpu().sort(-1).values)
        return output

    for _ in range(3):
        checks = verify(workload.ref_program, inputs, evidence=workload.verification(*inputs),
                        requests={"tileops": Request(run, inputs, preserve_inputs=True)})
        assert checks["tileops"].checked_outputs == checks["tileops"].total_outputs == 1
        assert checks["tileops"].unchecked_reason is None
    assert len(selected) == 3
    assert torch.equal(selected[0], selected[1]) and torch.equal(selected[0], selected[2])

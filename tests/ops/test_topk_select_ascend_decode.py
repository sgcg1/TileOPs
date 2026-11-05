"""Boundary regressions for the frozen Ascend decode selector, through the public Op."""

import pytest
import torch

from tileops.attention import TopKSelectFwdOp
from workloads.attention.topk_select import TopKSelectWorkload
from workloads.device import run_device
from workloads.numerics import Request, verify


@pytest.mark.parametrize(
    "pattern,start,end",
    [
        pytest.param("random", 0, 16384, id="manifest-window", marks=pytest.mark.smoke),
        pytest.param("random", 16384, 16384, id="empty-end", marks=pytest.mark.full),
        pytest.param("random", 127, 128, id="single-at-vector-edge", marks=pytest.mark.full),
        pytest.param("random", 129, 2176, id="short-k-minus-one", marks=pytest.mark.full),
        pytest.param("random", 127, 2175, id="exact-k", marks=pytest.mark.full),
        pytest.param("random", 127, 2176, id="refine-k-plus-one", marks=pytest.mark.full),
        pytest.param("random", 14208, 16384, id="nonaligned-window-at-end", marks=pytest.mark.full),
        pytest.param("equal", 0, 16384, id="all-tied-overflow", marks=pytest.mark.full),
        pytest.param("quantized", 0, 16384, id="threshold-ties", marks=pytest.mark.full),
        pytest.param("adjacent", 0, 16384, id="fp32-boundary-precision", marks=pytest.mark.full),
        pytest.param("negative", 0, 16384, id="negative-key-order", marks=pytest.mark.full),
        pytest.param("zero", 0, 16384, id="signed-zero-ties", marks=pytest.mark.full),
    ],
)
def test_topk_select_ascend_decode_boundaries(pattern, start, end):
    """Guard short-window padding, refinement, full-key order and reproducible tie sets."""
    if torch.device(run_device()).type != "npu":
        pytest.skip("Ascend decode integration regression")
    generator = torch.Generator().manual_seed(20261105)
    values = torch.randn(16384, generator=generator)
    if pattern == "equal":
        values.fill_(1.0)
    elif pattern == "quantized":
        values = (torch.arange(16384) % 4).float()
    elif pattern == "adjacent":
        # All values round to the same coarse FP16 bucket, but differ in FP32.
        values = (torch.arange(16384, dtype=torch.int32) + 0x3F800000).view(torch.float32)
    elif pattern == "negative":
        values = -torch.arange(1, 16385, dtype=torch.float32)
    elif pattern == "zero":
        values = torch.zeros(16384)
        values[::2] = -0.0
    inputs = (
        values.reshape(1, 1, 16384, 1).to(run_device()),
        torch.tensor([[start]], dtype=torch.int32, device=run_device()),
        torch.tensor([[end]], dtype=torch.int32, device=run_device()),
    )
    workload = TopKSelectWorkload(1, 1, 16384, 1, 2048, torch.float32, torch.int32)
    op = TopKSelectFwdOp(topk=2048)
    selected = []

    def run(*args):
        score_bits = args[0].view(torch.int32).clone()
        output = op(*args)
        assert torch.equal(args[0].view(torch.int32), score_bits), "score bits modified"
        selected.append(output.detach().cpu().sort(-1).values)
        return output

    for _ in range(3):
        checks = verify(
            workload.ref_program, inputs, evidence=workload.verification(*inputs),
            requests={"tileops": Request(run, inputs, preserve_inputs=True)},
        )
        assert checks["tileops"].checked_outputs == checks["tileops"].total_outputs == 1
        assert checks["tileops"].unchecked_reason is None
    # The public contract permits output permutations, but requires the same selected set.
    assert len(selected) == 3
    assert torch.equal(selected[0], selected[1]) and torch.equal(selected[0], selected[2])

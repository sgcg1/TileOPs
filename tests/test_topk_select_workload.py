"""The shared reference must distinguish invalid padding from valid -inf."""

import pytest
import torch

from workloads.attention.topk_select import TopKSelectWorkload
from workloads.numerics import compare_outputs

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("start,end", [(2, 4), (4, 2), (2, 2), (0, 8)])
def test_reference_keeps_valid_negative_infinity(start, end):
    workload = TopKSelectWorkload(1, 1, 8, 1, 4, torch.float32, torch.int32)
    scores = torch.full((1, 1, 8, 1), -float("inf"))
    starts = torch.tensor([[start]], dtype=torch.int32)
    ends = torch.tensor([[end]], dtype=torch.int32)
    got = workload.ref_program(scores, starts, ends)
    selected = list(range(start, end))[:4]
    expected = torch.tensor(selected + [8] * (4 - len(selected)), dtype=torch.int32)
    assert torch.equal(got.flatten(), expected)
    compare_outputs(got, got, workload.verification(scores, starts, ends))


def test_reference_nan_zero_and_group_layout():
    workload = TopKSelectWorkload(1, 1, 8, 2, 4, torch.float32, torch.int32)
    values = torch.tensor([float("nan"), 1, float("nan"), float("inf"), 0, -0.0, -float("inf"), -1])
    scores = torch.stack((values, values.flip(0)), -1).reshape(1, 1, 8, 2)
    starts = torch.zeros((1, 1), dtype=torch.int32)
    ends = torch.full_like(starts, 8)
    got = workload.ref_program(scores, starts, ends)
    assert got.tolist() == [[[[0, 2, 3, 1], [5, 7, 4, 6]]]]
    compare_outputs(got, got, workload.verification(scores, starts, ends))
    wrong = got.clone()
    wrong[0, 0, 0, 0] = 4  # Replace a NaN with a finite score.
    with pytest.raises(AssertionError):
        compare_outputs(wrong, got, workload.verification(scores, starts, ends))


def test_verifier_rejects_missing_valid_negative_infinity():
    workload = TopKSelectWorkload(1, 1, 8, 1, 4, torch.float32, torch.int32)
    scores = torch.full((1, 1, 8, 1), -float("inf"))
    starts = torch.tensor([[2]], dtype=torch.int32)
    ends = torch.tensor([[4]], dtype=torch.int32)
    expected = workload.ref_program(scores, starts, ends)
    with pytest.raises(AssertionError):
        compare_outputs(
            torch.full_like(expected, 8), expected, workload.verification(scores, starts, ends)
        )


def test_reference_orders_nan_payloads_and_adjacent_fp32():
    # Positive/negative signaling/quiet NaNs tie; subnormal and adjacent values stay exact.
    bits = [
        0x7F800001,
        -0x3FFFFF,
        0x7FFFFFFF,
        0x3F800001,
        0x3F800000,
        1,
        0,
        -0x80000000,
        -0x7FFFFFFF,
        0x7F800000,
    ]
    scores = torch.tensor(bits, dtype=torch.int32).view(torch.float32).reshape(1, 1, 10, 1)
    starts = torch.tensor([[0]], dtype=torch.int32)
    ends = torch.tensor([[10]], dtype=torch.int32)
    workload = TopKSelectWorkload(1, 1, 10, 1, 6, torch.float32, torch.int32)
    got = workload.ref_program(scores, starts, ends)
    assert got.flatten().tolist() == [0, 1, 2, 9, 3, 4]
    compare_outputs(got, got, workload.verification(scores, starts, ends))
    wrong = got.clone()
    wrong[..., 4] = 5  # A positive subnormal cannot replace the larger adjacent FP32 value.
    with pytest.raises(AssertionError):
        compare_outputs(wrong, got, workload.verification(scores, starts, ends))

"""Mainline TileLang Ascend backend. Importing registers lazy builders only."""

from tileops.backend import register_detector, register_kernel_builder


def _build_topk(scores, starts, ends, *, top_k):
    from tileops.ascend.topk_select import AscendTopKSelect

    return AscendTopKSelect(scores, top_k)


register_detector(target="ascend", detect=lambda device: device.type == "npu")
register_kernel_builder(op="TopKSelectFwdOp", target="ascend", build_kernel=_build_topk)

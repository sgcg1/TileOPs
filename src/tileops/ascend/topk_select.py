"""Host adapter for the exact Ascend windowed selector."""

import torch


class AscendTopKSelect:
    def __init__(self, scores, top_k):
        if scores.device.type != "npu" or scores.dtype != torch.float32:
            raise ValueError("Ascend TopKSelect requires FP32 scores on an NPU")
        if "950" not in torch.npu.get_device_name(scores.device):
            raise ValueError("This Ascend TopKSelect backend requires Ascend 950")
        from tileops._kernels.attention.topk_select_ascend import topk_select_ascend

        self.batch, self.queries, self.length, self.groups = scores.shape
        self.top_k = top_k
        rows = self.batch * self.queries * self.groups
        # Conservative worker count; each worker reuses its UB across rows.
        # This controls scheduling, not the number of physical Vector cores.
        self.config = {"workers": min(rows, 32), "threads": 128, "candidate_capacity": 2048}
        self.program = (
            topk_select_ascend(rows, self.length, top_k, self.config["workers"], self.groups)
            if rows
            else None
        )

    def __call__(self, scores, starts, ends):
        output = torch.empty(
            (self.batch, self.queries, self.groups, self.top_k),
            dtype=torch.int32,
            device=scores.device,
        )
        if self.program is not None:
            self.program(
                scores.reshape(self.batch * self.queries, self.length, self.groups),
                starts.reshape(-1),
                ends.reshape(-1),
                output.reshape(-1, self.top_k),
            )
        return output

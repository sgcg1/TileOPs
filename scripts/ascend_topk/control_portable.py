"""Run same-process alternating controls without changing the deployed kernel."""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tileops._kernels.attention.topk_select_ascend import topk_select_ascend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    torch.manual_seed(20261030)
    torch.npu.manual_seed_all(20261030)
    x = torch.randn((args.rows, args.length, 1), device="npu", dtype=torch.float32)
    starts = torch.zeros(args.rows, device="npu", dtype=torch.int32)
    ends = torch.full_like(starts, args.length)
    programs = {}
    for version in ["v0.1.0", "v0.1.2"]:
        spec = importlib.util.spec_from_file_location(
            version.replace(".", "_"), args.support / f"kernel-{version}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        programs[version] = module.topk_select_ascend(
            args.rows, args.length, args.k, args.workers, 1
        )
    for threads in [128, 256]:
        programs[f"candidate-h{threads}"] = topk_select_ascend(
            args.rows, args.length, args.k, args.workers, 1, COARSE=True, H=threads
        )
    outputs = {name: torch.empty((args.rows, args.k), device="npu", dtype=torch.int32)
               for name in programs}
    sources = {}
    for name, program in programs.items():
        source = program.get_kernel_source()
        (args.output / f"{name}.cpp").write_text(source)
        sources[name] = hashlib.sha256(source.encode()).hexdigest()
        for _ in range(20):
            program(x, starts, ends, outputs[name])
    torch.npu.synchronize()
    reference = None
    for name, output in outputs.items():
        ids = output.long()
        assert bool(((ids >= 0) & (ids < args.length)).all().item())
        selected = ids.sort(dim=1).values
        assert bool((selected.diff(dim=1) != 0).all().item())
        threshold = x[:, :, 0].gather(1, ids).min(dim=1).values
        assert bool(((x[:, :, 0] > threshold[:, None]).sum(dim=1) < args.k).all().item())
        selected = selected.cpu()
        if reference is None:
            reference = selected
        else:
            assert torch.equal(reference, selected), name
    for name, program in programs.items():
        for _ in range(10):
            program(x, starts, ends, outputs[name])
    torch.npu.synchronize()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    samples = {name: [] for name in programs}
    names = list(programs)
    for iteration in range(50):
        for name in names if iteration % 2 == 0 else names[::-1]:
            begin.record()
            programs[name](x, starts, ends, outputs[name])
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000)
    result = {
        "shape": [args.rows, args.length, 1], "k": args.k, "workers": args.workers,
        "input_sha256": hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
        "physical_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "generated_source_sha256": sources, "event_samples_us": samples,
        "event_median_us": {name: statistics.median(v) for name, v in samples.items()},
        "correctness": "finite random input; bounds, unique, threshold, equal selected sets",
        "timing": "same process, alternating forward/reverse order; preallocated device events",
    }
    (args.output / "control.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["event_median_us"], indent=2), flush=True)


if __name__ == "__main__":
    main()

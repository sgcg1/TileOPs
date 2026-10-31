"""Measure preallocated TopK kernels and retain matched-input device evidence."""

import argparse
import csv
import hashlib
import importlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch_npu

from tileops._kernels.attention.topk_select_ascend import topk_select_ascend
from tileops.ascend.topk_select import AscendTopKSelect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--case",
        choices=["all", "decode", "small", "prefill-k1024", "prefill-k2048"],
        required=True,
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int, choices=[64, 128, 256])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--profile-samples", type=int, default=10)
    args = parser.parse_args()
    if args.case == "all":
        for name in ["decode", "small", "prefill-k1024", "prefill-k2048"]:
            args.case = name
            measure(args)
    else:
        measure(args)


def measure(args):
    rows, length, k = {
        "decode": (1, 16384, 2048),
        "small": (128, 65536, 2048),
        "prefill-k1024": (32768, 65536, 1024),
        "prefill-k2048": (32768, 65536, 2048),
    }[args.case]
    root = Path(os.environ["EVIDENCE"]) / args.version
    root.mkdir(parents=True, exist_ok=True)
    suffix = (
        args.case
        + (f"-w{args.workers}" if args.workers else "")
        + (f"-t{args.threads}" if args.threads else "")
    )
    torch.npu.set_device(0)
    torch.manual_seed(20261030)
    torch.npu.manual_seed_all(20261030)
    free, _ = torch.npu.mem_get_info()
    assert free > 1.25 * (rows * length * 4 + rows * k * 4) + 2**30
    scores = torch.randn((1, rows, length, 1), device="npu", dtype=torch.float32)
    starts = torch.zeros((rows,), device="npu", dtype=torch.int32)
    ends = torch.full_like(starts, length)
    output = torch.empty((rows, k), device="npu", dtype=torch.int32)
    adapter = AscendTopKSelect(scores, k)
    workers = args.workers or adapter.config["workers"]
    threads = args.threads or adapter.config["threads"]
    if args.workers or args.threads:
        program = topk_select_ascend(
            rows, length, k, workers, 1, COARSE=adapter.config["coarse"], H=threads
        )
    else:
        program = adapter.program
    input_view = scores.reshape(rows, length, 1)

    def run():
        program(input_view, starts, ends, output)

    run()
    torch.npu.synchronize()
    digest = hashlib.sha256()
    for first in range(0, rows, 128):
        x = scores.reshape(rows, length)[first : first + 128]
        digest.update(x.cpu().numpy().tobytes())
        ids = output[first : first + 128].long()
        assert bool(((ids >= 0) & (ids < length)).all().item())
        assert bool((ids.sort(dim=1).values.diff(dim=1) != 0).all().item())
        threshold = x.gather(1, ids).min(dim=1).values
        assert bool(((x > threshold[:, None]).sum(dim=1) < k).all().item())
    expected_hash = (
        "c59b9485e9750eaf04d4668934c0383510a15b2cc9365ef9c6c4b9eaf2bf7593"
        if args.case == "decode"
        else "0c435edd506b44a1cc41ea19a140811275f88c0129da89fb8ecb95e96e48e229"
        if rows == 32768
        else None
    )
    if expected_hash:
        assert digest.hexdigest() == expected_hash
    for _ in range(10):
        run()
    torch.npu.synchronize()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(args.samples):
        begin.record()
        run()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000)
    profile_dir = root / f"profile-{suffix}-{os.getpid()}"
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
    ) as prof:
        run()
        torch.npu.synchronize()
        prof.step()
        # Leave headroom for profiler startup; select a fixed chronological tail.
        for _ in range(args.profile_samples + 10):
            run()
            torch.npu.synchronize()
        prof.step()
    records = []
    for path in profile_dir.rglob("kernel_details.csv"):
        with path.open() as handle:
            records.extend(csv.DictReader(handle))
    assert args.profile_samples <= len(records) <= args.profile_samples + 10, (
        len(records),
        args.profile_samples,
    )
    records.sort(key=lambda row: float(row["Start Time(us)"].strip()))
    captured_records = len(records)
    records = records[-args.profile_samples :]
    kernel_samples = [float(row["Duration(us)"]) for row in records]
    data = {
        "version": args.version,
        "case": args.case,
        "shape": [1, rows, length, 1],
        "k": k,
        "workers": workers,
        "threads": threads,
        "coarse": adapter.config["coarse"],
        "input_sha256": digest.hexdigest(),
        "correctness": "all rows: bounds, uniqueness, exact FP32 threshold",
        "kernel_source_path": str(
            importlib.import_module("tileops._kernels.attention.topk_select_ascend").__file__
        ),
        "generated_source_sha256": hashlib.sha256(program.get_kernel_source().encode()).hexdigest(),
        "kernel_source_sha256": hashlib.sha256(
            Path(
                importlib.import_module("tileops._kernels.attention.topk_select_ascend").__file__
            ).read_bytes()
        ).hexdigest(),
        "event_samples_us": samples,
        "event_median_us": statistics.median(samples),
        "kernel_samples_us": kernel_samples,
        "kernel_median_us": statistics.median(kernel_samples),
        "kernel_names": sorted({row["Name"] for row in records}),
        "device": torch.npu.get_device_name(0),
        "physical_device": 1,
        "profile_path": str(profile_dir),
        "profile_captured_records": captured_records,
        "profile_launched_records": args.profile_samples + 10,
        "profile_selection": "last requested samples in device timestamp order",
        "timing": "preallocated raw kernel; 10 warmups; independent device events and profiling",
    }
    (root / (suffix + ".json")).write_text(json.dumps(data, indent=2))
    print(
        json.dumps({key: value for key, value in data.items() if "samples" not in key}), flush=True
    )


if __name__ == "__main__":
    main()

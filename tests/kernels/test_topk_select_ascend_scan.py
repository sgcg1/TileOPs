"""CPU arithmetic checks for the actual scan AST; no TileLang/NPU import required.

Run directly with Python. These checks do not validate VF lowering or synchronization.
"""

import ast
import contextlib
import copy
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

KERNEL = (
    Path(__file__).resolve().parents[2] / "src/tileops/_kernels/attention/topk_select_ascend.py"
)
TREE = ast.parse(KERNEL.read_text(encoding="utf-8"))
SCAN = next(
    node
    for node in ast.walk(TREE)
    if isinstance(node, ast.If)
    and isinstance(node.test, ast.Name)
    and node.test.id == "PARALLEL_SCAN"
)
REGIONS = {
    name: compile(ast.Module(body=body, type_ignores=[]), str(KERNEL), "exec")
    for name, body in [("parallel", SCAN.body), ("serial", SCAN.orelse)]
}
LANGUAGE = SimpleNamespace(
    SimtVF=lambda **kwargs: contextlib.nullcontext(),
    Parallel=range,
    serial=range,
    int32=int,
    alloc_var=lambda dtype: 0,
)


def run_scan(name, counts, quota, threads, reverse_parallel=False):
    language = SimpleNamespace(**vars(LANGUAGE))
    if reverse_parallel:
        language.Parallel = lambda count: reversed(range(count))
    values = {
        "T": language,
        "H": threads,
        "hist": list(counts) + [0],
        "state": [quota, 0, -1, 0, 0, 0, 0],
        "group_offsets": [0] * 16,
        "local_prefix": [0] * 256,
    }
    exec(REGIONS[name], values)
    return values["hist"][:256], values["state"]


class ScanArithmeticTest(unittest.TestCase):
    def check_counts(self, counts, quota, threads=128):
        # Independent oracle: a bucket owns [number strictly above, cumulative count).
        higher = [sum(counts[bucket + 1 :]) for bucket in range(256)]
        threshold = next((b for b in range(256) if higher[b] <= quota < higher[b] + counts[b]), -1)
        total = sum(counts)
        accepted = higher[threshold] if threshold >= 0 else total
        for name in REGIONS:
            hist, state = run_scan(name, counts, quota, threads)
            self.assertEqual(hist, higher, (name, quota))
            self.assertEqual(state[:5], [quota - accepted, 0, threshold, total, accepted])

    def test_all_single_bucket_overflows(self):
        for bucket in range(256):
            counts = [0] * 256
            counts[bucket] = 65536
            for quota in [1, 1024, 2048, 65535, 65536]:
                self.check_counts(counts, quota)

    def test_group_boundaries_and_threads(self):
        counts = [256] * 256
        for threads in [64, 128, 256]:
            for quota in [1, 255, 256, 4095, 4096, 4097, 32768, 65535, 65536]:
                self.check_counts(counts, quota, threads)

    def test_sparse_random_histograms(self):
        rng = random.Random(20261102)
        for _ in range(1000):
            counts = [rng.randrange(1025) if rng.randrange(4) == 0 else 0 for _ in range(256)]
            self.check_counts(counts, rng.randrange(1, sum(counts) + 2))

    def test_finished_quota(self):
        for name in REGIONS:
            hist, state = run_scan(name, [0] * 256, 0, 128)
            self.assertEqual(hist, [0] * 256)
            self.assertEqual(state[:5], [0, 0, -1, 0, 0])
        self.check_counts([0] * 256, 1)
        self.check_counts([1] * 256, 257)

    def test_parallel_iteration_order(self):
        rng = random.Random(16)
        for _ in range(100):
            counts = [rng.randrange(2049) for _ in range(256)]
            quota = rng.randrange(1, sum(counts))
            self.assertEqual(
                run_scan("parallel", counts, quota, 128),
                run_scan("parallel", counts, quota, 128, reverse_parallel=True),
            )

    def test_grouped_scan_configuration_guard(self):
        class KernelDefinitionReachedError(Exception):
            pass

        class LanguageBoundary:
            def __getattr__(self, name):
                raise KernelDefinitionReachedError

        function = copy.deepcopy(next(n for n in TREE.body if isinstance(n, ast.FunctionDef)))
        function.decorator_list = []
        namespace = {"T": LanguageBoundary()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(KERNEL), "exec"), namespace)
        build = namespace["topk_select_ascend"]
        for threads in [64, 256]:
            with self.assertRaisesRegex(AssertionError, "restricted to H=128"):
                build(1, 16384, 2048, 1, 1, H=threads, PARALLEL_SCAN=True)
        for threads, parallel in [(128, True), (128, False), (64, False), (256, False)]:
            with self.assertRaises(KernelDefinitionReachedError):
                build(1, 16384, 2048, 1, 1, H=threads, PARALLEL_SCAN=parallel)


if __name__ == "__main__":
    unittest.main()

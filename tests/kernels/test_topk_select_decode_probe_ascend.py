"""CPU merge AST and partition checks for the compile-only experimental module."""
import ast
import contextlib
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
TREE = ast.parse((ROOT / "src/tileops/_kernels/attention/topk_select_decode_probe_ascend.py").read_text())
MERGE = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "decode_merge")
MAIN = next(n for n in MERGE.body if isinstance(n, ast.FunctionDef))
CODE = compile(ast.Module(body=MAIN.body, type_ignores=[]), "merge-ast", "exec")
LANGUAGE = SimpleNamespace(Kernel=lambda *a: contextlib.nullcontext(),
                           SimtVF=lambda **kw: contextlib.nullcontext(), Parallel=range, serial=range,
                           alloc_shared=lambda shape, dtype: [0] * shape[0],
                           alloc_var=lambda dtype: 0, int32=int)


class Partial:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index[0]][index[1]]


class DecodeProbeTest(unittest.TestCase):
    def test_merge_against_independent_oracle(self):
        rng = random.Random(20261102)
        for p in [1, 2, 4]:
            cases = [[[rng.randrange(100) for _ in range(256)] for _ in range(p)],
                     [[0] * 256 for _ in range(p)], [[0] * 128 + [16384 // p] + [0] * 127 for _ in range(p)]]
            for values in cases:
                counts = [sum(row[b] for row in values) for b in range(256)]
                before = [sum(counts[b + 1:]) for b in range(256)]
                threshold = next((b for b in range(256) if before[b] <= 2048 < before[b] + counts[b]), -1)
                expected = [threshold, before[threshold] if threshold >= 0 else sum(counts),
                            counts[threshold] if threshold >= 0 else 0, sum(counts)]
                higher, meta = [-1] * 256, [-1] * 4
                exec(CODE, dict(T=LANGUAGE, P=p, K=2048, partial=Partial(values), higher=higher, meta=meta))
                self.assertEqual(higher, before)
                self.assertEqual(meta, expected)

    def test_partition_and_source_coarse_key(self):
        for p in [1, 2, 4]:
            indices = [w * (16384 // p) + t * 128 + lane for w in range(p)
                       for t in range(16384 // p // 128) for lane in range(128)]
            self.assertEqual(sorted(indices), list(range(16384)))
        production = ast.parse((ROOT / "src/tileops/_kernels/attention/topk_select_ascend.py").read_text())
        helper = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "coarse_byte")
        self.assertEqual(ast.dump(helper(TREE)), ast.dump(helper(production)))


if __name__ == "__main__":
    unittest.main()

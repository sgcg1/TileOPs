"""CPU merge AST and partition checks for the compile-only experimental module."""
import ast
import contextlib
from pathlib import Path
import random
import struct
import subprocess
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

    def test_staged_histogram_against_oracle(self):
        func = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "decode_histogram")
        main = next(n for n in func.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        class Atomic(ast.NodeTransformer):
            def visit_Call(self, node):
                if ast.unparse(node.func) == "T.atomic_add":
                    return ast.Call(func=ast.Name(id="add", ctx=ast.Load()),
                        args=[node.args[0].value, node.args[0].slice, node.args[1]], keywords=[])
                return self.generic_visit(node)
        module = ast.parse(ast.unparse(ast.Module(body=main.body, type_ignores=[])))
        code = compile(ast.fix_missing_locations(Atomic().visit(module)), "histogram", "exec")
        def coarse(value):
            try:
                bits = struct.unpack("H", struct.pack("e", value))[0]
            except OverflowError:
                bits = 0xfc00 if value < 0 else 0x7c00
            magnitude = bits & 32767
            key = 65535 if magnitude > 31744 else 32768 if magnitude == 0 else (~bits & 65535) if bits >> 15 else bits | 32768
            return key >> 8
        def add(array, i, value):
            array[i] += value
        class Output:
            def __init__(self, p):
                self.rows = [[-1] * 256 for _ in range(p)]
            def __setitem__(self, index, value):
                self.rows[index[0]][index[1]] = value
        rng = random.Random(42)
        cases = [[rng.uniform(-1, 1) for _ in range(512)], [1.] * 512,
                 [float("nan"), float("inf"), -float("inf"), 0., -0., 65520., -65520., 65504.] * 64]
        for values in cases:
            for start, end in [(0, 512), (9, 503), (11, 20), (20, 20), (30, 20)]:
                for p in [1, 2, 4]:
                    expected = [[0] * 256 for _ in range(p)]
                    for i in range(start, end):
                        expected[i // (512 // p)][coarse(values[i])] += 1
                    for reverse in [False, True]:
                        for staged in [False, True]:
                            output = Output(p)
                            language = SimpleNamespace(**vars(LANGUAGE))
                            language.Parallel = (lambda n: reversed(range(n))) if reverse else range
                            language.float32 = float
                            language.copy = lambda source, target: target.__setitem__(slice(None), source)
                            language.max, language.min = max, min
                            language.assume = lambda value: None
                            for worker in range(p):
                                language.Kernel = lambda *args, w=worker: contextlib.nullcontext(w)
                                exec(code, dict(T=language, N=512, P=p, STAGE_INPUT=staged,
                                    x=values, starts=[start], ends=[end], partial=output,
                                    coarse_byte=coarse, add=add))
                            self.assertEqual(output.rows, expected)

    def test_disabled_staging_matches_frozen_ast(self):
        class Disable(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == "STAGE_INPUT":
                    return []
                return self.generic_visit(node)
            def visit_IfExp(self, node):
                if ast.unparse(node.test) == "STAGE_INPUT":
                    return self.visit(node.orelse)
                return self.generic_visit(node)
        current = ast.parse(ast.unparse(TREE))
        func = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == "decode_histogram")
        self.assertEqual(func.args.args[-1].arg, "STAGE_INPUT")
        self.assertIs(func.args.defaults[-1].value, False)
        func.args.args.pop()
        func.args.defaults.pop()
        raw = subprocess.check_output(["git", "show", "9a5087dd:src/tileops/_kernels/attention/topk_select_decode_probe_ascend.py"], cwd=ROOT)
        self.assertEqual(ast.dump(Disable().visit(current)), ast.dump(ast.parse(raw)))

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

"""Execute scaled pipeline AST on CPU; no TileLang or NPU imports."""
import ast
import contextlib
import math
from pathlib import Path
import random
import struct
import subprocess
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/tileops/_kernels/attention/topk_select_decode_pipeline_ascend.py"
TREE = ast.parse(SOURCE.read_text())


class Lower(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant({16384: 256, 2048: 16, 2047: 15, 128: 8}.get(node.value, node.value)), node)

    def visit_Call(self, node):
        if ast.unparse(node.func) == "T.atomic_add":
            target = node.args[0]
            return ast.Call(func=ast.Name(id="add", ctx=ast.Load()),
                            args=[self.visit(target.value), self.visit(target.slice), self.visit(node.args[1])], keywords=[])
        return self.generic_visit(node)


def code(name):
    func = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    main = next(n for n in func.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    # Parse a copy so each scaled body has independent nodes.
    module = ast.parse(ast.unparse(ast.Module(body=main.body, type_ignores=[])))
    return compile(ast.fix_missing_locations(Lower().visit(module)), name, "exec")


PREPARE, COMPACT, FINISH = [code(name) for name in ["decode_prepare", "decode_compact", "decode_finish"]]


class Matrix:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return self.rows[key[0]][key[1]]

    def __setitem__(self, key, value):
        self.rows[key[0]][key[1]] = value


def score(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    magnitude = bits & 0x7fffffff
    return 0xffffffff if magnitude > 0x7f800000 else 0x80000000 if magnitude == 0 else (~bits & 0xffffffff) if bits >> 31 else bits | 0x80000000


def coarse(value):
    try:
        bits = struct.unpack("H", struct.pack("e", value))[0]
    except OverflowError:
        bits = 0xfc00 if value < 0 else 0x7c00
    magnitude = bits & 32767
    key = 65535 if magnitude > 31744 else 32768 if magnitude == 0 else (~bits & 65535) if bits >> 15 else bits | 32768
    return key >> 8


def run(values, start, end, p, reverse=False, stable=False):
    def add(array, i, value):
        old = array[i]
        array[i] += value
        return old

    base = dict(STABLE_TIE=stable, x=values, starts=[start], ends=[end], P=p, K=16,
                meta=[-1] * 4, higher=[-1] * 256, out=[-1] * 16,
                offsets=Matrix([[0, 0] for _ in range(p)]), candidate_scores=[0.] * 16,
                candidate_indices=[0] * 16, final_state=[-1, -1],
                add=add, score_key=score, coarse_byte=coarse)
    counts = [[0] * 256 for _ in range(p)]
    for i in range(max(0, start), min(256, end)):
        counts[i // (256 // p)][coarse(values[i])] += 1
    base["partial"] = Matrix(counts)
    language = SimpleNamespace(SimtVF=lambda **kw: contextlib.nullcontext(),
                               Kernel=lambda *args: contextlib.nullcontext(),
                               Parallel=(lambda n: reversed(range(n))) if reverse else range,
                               serial=range, alloc_shared=lambda shape, dtype: [0] * shape[0],
                               alloc_var=lambda dtype: 0, int32="int32", uint32="uint32",
                               Cast=lambda dtype, x: int(x) & 0xffffffff if dtype == "uint32" else int(x),
                               if_then_else=lambda c, a, b: a if c else b,
                               max=max, min=min, assume=lambda cond: None,
                               ceildiv=lambda a, b: (a + b - 1) // b)
    base["T"] = language
    exec(PREPARE, base)
    order = reversed(range(p)) if reverse else range(p)
    for worker in order:
        language.Kernel = lambda *args, w=worker: contextlib.nullcontext(w)
        exec(COMPACT, base)
    exec(FINISH, base)
    expected = sorted(range(start, end), key=lambda i: (score(values[i]), -i), reverse=True)[:16]
    expected += [256] * (16 - len(expected))
    return base, expected


class DecodePipelineTest(unittest.TestCase):
    def test_selection_distributions_windows_and_order(self):
        rng = random.Random(20261102)
        distributions = [[i * 10. for i in range(256)], [1.] * 256,
                         [rng.uniform(-1, 1) for _ in range(256)],
                         [float("nan"), float("inf"), -float("inf"), 0., -0., 65520., -65520., 65504.] * 32,
                         [1 + i * 1e-8 for i in range(256)]]
        small_boundary = [1000.] * 15 + [100.] * 10 + [0.] * 231
        rng.shuffle(small_boundary)
        distributions.append(small_boundary)
        seen = set()
        for values in distributions:
            for start, end in [(0, 256), (9, 210), (11, 20), (20, 20), (30, 20)]:
                for p in [2, 4]:
                    for reverse, stable in [(False, False), (True, False), (False, True), (True, True)]:
                        result, expected = run(values, start, end, p, reverse, stable)
                        if result["meta"][3] > 16 and result["meta"][1] < 16:
                            seen.add("small" if result["meta"][2] <= 16 else "overflow")
                        self.assertEqual(sorted(result["out"]), sorted(expected), (start, end, p))
                        self.assertEqual(result["final_state"], [0, min(16, max(0, end - start))])

        self.assertEqual(seen, {"small", "overflow"})

    def test_disabled_finish_matches_measured_r1_ast(self):
        class Disable(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == "STABLE_TIE":
                    return []
                return self.generic_visit(node)

            def visit_IfExp(self, node):
                if ast.unparse(node.test) == "STABLE_TIE":
                    return self.visit(node.orelse)
                return self.generic_visit(node)

        raw = subprocess.check_output(["git", "show", "03c8f916:src/tileops/_kernels/attention/topk_select_decode_pipeline_ascend.py"], cwd=ROOT)
        old = ast.parse(raw)
        current = ast.parse(SOURCE.read_text())
        finish = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == "decode_finish")
        self.assertEqual(finish.args.args[0].arg, "STABLE_TIE")
        self.assertIs(finish.args.defaults[0].value, False)
        finish.args.args.clear()
        finish.args.defaults.clear()
        self.assertEqual(ast.dump(Disable().visit(current)), ast.dump(old))

    def test_helpers_match_production_ast(self):
        prod = ast.parse((ROOT / "src/tileops/_kernels/attention/topk_select_ascend.py").read_text())
        for name in ["coarse_byte", "score_key"]:
            expected = ast.dump(next(n for n in ast.walk(prod) if isinstance(n, ast.FunctionDef) and n.name == name))
            for helper in (n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name):
                self.assertEqual(ast.dump(helper), expected)


if __name__ == "__main__":
    unittest.main()

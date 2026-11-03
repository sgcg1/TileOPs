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


def run(values, start, end, p, reverse=False, stable=False, adaptive=False, grouped=False, early=False, scan_width=128, stage_input=False, fuse_quota=False, skip_index_prefix=False, stage_overflow=False):
    def add(array, i, value):
        old = array[i]
        array[i] += value
        return old

    base = dict(STAGE_OVERFLOW=stage_overflow, SKIP_INDEX_PREFIX=skip_index_prefix, FUSE_QUOTA=fuse_quota, STAGE_INPUT=stage_input, SCAN_WIDTH={128: 8, 512: 32}[scan_width], EARLY_EXIT=early, GROUPED_OFFSETS=grouped, STABLE_TIE=stable, ADAPTIVE_TIE=adaptive, x=values, starts=[start], ends=[end], P=p, K=16,
                meta=[-1] * 4, higher=[-1] * 256, out=[-1] * 16,
                offsets=Matrix([[0, 0] for _ in range(p)]), candidate_scores=[0.] * 16,
                candidate_indices=[0] * 16, final_state=[-1, -1],
                add=add, score_key=score, coarse_byte=coarse)
    counts = [[0] * 256 for _ in range(p)]
    for i in range(max(0, start), min(256, end)):
        counts[i // (256 // p)][coarse(values[i])] += 1
    base["partial"] = Matrix(counts)
    calls = [0]
    def vf(**kw):
        calls[0] += 1
        return contextlib.nullcontext()
    base["vf_calls"] = calls
    language = SimpleNamespace(SimtVF=vf,
                               Kernel=lambda *args: contextlib.nullcontext(),
                               Parallel=(lambda n: reversed(range(n))) if reverse else range,
                               serial=range, alloc_shared=lambda shape, dtype: [0] * shape[0],
                               alloc_var=lambda dtype: 0, int32="int32", uint32="uint32", float32="float32",
                               copy=lambda source, target: target.__setitem__(slice(None), source),
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
        for boundary in [16, 17]:
            edge = [1000.] * 15 + [100.] * boundary + [0.] * (241 - boundary)
            rng.shuffle(edge)
            distributions.append(edge)
        seen = set()
        for values in distributions:
            for start, end in [(0, 256), (9, 210), (11, 20), (20, 20), (30, 20)]:
                for p in [2, 4]:
                    for reverse, stable, adaptive, grouped in [(a, b, c, g) for a in [False, True] for b, c in [(False, False), (True, False), (False, True)] for g in [False, True]]:
                        result, expected = run(values, start, end, p, reverse, stable, adaptive, grouped)
                        if result["meta"][3] > 16 and result["meta"][1] < 16:
                            seen.add("small" if result["meta"][2] <= 16 else "overflow")
                        self.assertEqual(sorted(result["out"]), sorted(expected), (start, end, p))
                        self.assertEqual(result["final_state"], [0, min(16, max(0, end - start))])

        self.assertEqual(seen, {"small", "overflow"})

    def test_early_exit_keeps_exact_sets(self):
        rng = random.Random(9)
        cases = [[rng.uniform(-1, 1) for _ in range(256)], [1.] * 256,
                 [1000.] * 15 + [100.] * 16 + [0.] * 225]
        for values in cases:
            for start, end in [(0, 256), (11, 20), (20, 20), (30, 20)]:
                for stable, adaptive in [(False, False), (True, False), (False, True)]:
                    old, expected = run(values, start, end, 4, True, stable, adaptive, True)
                    new, expected_new = run(values, start, end, 4, True, stable, adaptive, True, True)
                    self.assertEqual(sorted(new["out"]), sorted(expected_new))
                    self.assertEqual(sorted(old["out"]), sorted(new["out"]))
                    self.assertEqual(new["final_state"], old["final_state"])
                    self.assertLessEqual(new["vf_calls"][0], old["vf_calls"][0])
                    if end - start <= 16:
                        self.assertLess(new["vf_calls"][0], old["vf_calls"][0])

    def test_wide_compact_preserves_selection(self):
        rng = random.Random(11)
        cases = [[rng.uniform(-1, 1) for _ in range(256)], [1.] * 256,
                 [float("nan"), float("inf"), -float("inf"), 0., -0.] * 51 + [1.],
                 [1000.] * 15 + [100.] * 16 + [0.] * 225,
                 [1000.] * 15 + [100.] * 17 + [0.] * 224]
        for values in cases:
            for start, end in [(0, 256), (9, 210), (11, 20), (20, 20), (30, 20)]:
                for p in [2, 4]:
                    for reverse in [False, True]:
                        old, expected = run(values, start, end, p, reverse, adaptive=True, grouped=True, early=True)
                        for width in [128, 512]:
                            new, _ = run(values, start, end, p, reverse, adaptive=True, grouped=True, early=True, scan_width=width, stage_input=True)
                            self.assertEqual(sorted(new["out"]), sorted(expected))
                            self.assertEqual(sorted(new["out"]), sorted(old["out"]))
                            self.assertEqual(new["final_state"], old["final_state"])

    def test_fused_quota_preserves_sets_and_removes_vf(self):
        rng = random.Random(27)
        cases = [[rng.uniform(-1, 1) for _ in range(256)], [1.] * 256,
                 [float("nan"), float("inf"), -float("inf"), 0., -0.] * 51 + [1.],
                 [1000.] * 15 + [100.] * 16 + [0.] * 225,
                 [1000.] * 15 + [100.] * 17 + [0.] * 224]
        reduced = False
        for values in cases:
            for start, end in [(0, 256), (9, 210), (11, 20), (20, 20), (30, 20)]:
                for reverse in [False, True]:
                    for stable, adaptive in [(False, False), (True, False), (False, True)]:
                        for early in [False, True]:
                            old, expected = run(values, start, end, 4, reverse, stable, adaptive, True, early, stage_input=True)
                            new, _ = run(values, start, end, 4, reverse, stable, adaptive, True, early, stage_input=True, fuse_quota=True)
                            self.assertEqual(sorted(new["out"]), sorted(expected))
                            self.assertEqual(sorted(new["out"]), sorted(old["out"]))
                            self.assertEqual(new["final_state"], old["final_state"])
                            self.assertLessEqual(new["vf_calls"][0], old["vf_calls"][0])
                            reduced |= new["vf_calls"][0] < old["vf_calls"][0]
        self.assertTrue(reduced)

    def test_skip_constant_index_bytes(self):
        rng = random.Random(31)
        cases = [[rng.uniform(-1, 1) for _ in range(256)], [1.] * 256,
                 [float("nan"), float("inf"), -float("inf"), 0., -0.] * 51 + [1.],
                 [1000.] * 15 + [100.] * 16 + [0.] * 225,
                 [1000.] * 15 + [100.] * 17 + [0.] * 224]
        for values in cases:
            for start, end in [(0, 256), (9, 210), (11, 20), (20, 20), (30, 20)]:
                for reverse in [False, True]:
                    for stable, adaptive in [(False, False), (True, False), (False, True)]:
                        for early in [False, True]:
                            old, expected = run(values, start, end, 4, reverse, stable, adaptive, True, early, stage_input=True)
                            new, _ = run(values, start, end, 4, reverse, stable, adaptive, True, early, stage_input=True, skip_index_prefix=True)
                            self.assertEqual(sorted(new["out"]), sorted(expected))
                            self.assertEqual(new["final_state"], old["final_state"])
                            self.assertLessEqual(new["vf_calls"][0], old["vf_calls"][0])
                            if stable or (adaptive and new["meta"][2] > 16):
                                self.assertEqual(new["vf_calls"][0], old["vf_calls"][0])
                            elif not early:
                                self.assertEqual(old["vf_calls"][0] - new["vf_calls"][0], 16)

    def test_full_scale_index_prefix_is_constant(self):
        keys = [(~i) & 0xffffffff for i in range(16384)]
        self.assertEqual({key >> 16 for key in keys}, {0xffff})
        self.assertTrue(all((a & 0xffff) > (b & 0xffff) for a, b in zip(keys, keys[1:])))

    def test_overflow_staging_preserves_exact_selection(self):
        rng = random.Random(49)
        cases = [[rng.uniform(-1, 1) for _ in range(256)], [1.] * 256,
                 [float("nan"), float("inf"), -float("inf"), 0., -0.] * 51 + [1.],
                 [1000.] * 15 + [100.] * 16 + [0.] * 225,
                 [1000.] * 15 + [100.] * 17 + [0.] * 224]
        seen = set()
        for values in cases:
            for start, end in [(0, 256), (9, 210), (11, 20), (20, 20), (30, 20)]:
                for reverse in [False, True]:
                    old, expected = run(values, start, end, 4, reverse, adaptive=True, grouped=True, early=True, stage_input=True)
                    new, _ = run(values, start, end, 4, reverse, adaptive=True, grouped=True, early=True, stage_input=True, stage_overflow=True)
                    self.assertEqual(sorted(new["out"]), sorted(expected))
                    self.assertEqual(new["final_state"], old["final_state"])
                    seen.add(new["meta"][2] > 16)
                    if new["meta"][2] <= 16:
                        self.assertTrue(all(value == 0 for value in new["overflow_x"]))
        self.assertEqual(seen, {False, True})

    def test_prepare_offsets_against_full_scale_oracle(self):
        prepare = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "decode_prepare")
        main = next(n for n in prepare.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        actual = compile(ast.Module(body=main.body, type_ignores=[]), "full-prepare", "exec")
        rng = random.Random(7)
        for p in [1, 2, 4]:
            for grouped in [False, True]:
                values = [[rng.randrange(20) for _ in range(256)] for _ in range(p)]
                language = SimpleNamespace(Kernel=lambda *args: contextlib.nullcontext(),
                    SimtVF=lambda **kw: contextlib.nullcontext(), Parallel=range, serial=range,
                    alloc_shared=lambda shape, dtype: [0] * shape[0], alloc_var=lambda dtype: 0,
                    int32=int, max=max, min=min)
                meta, offsets = [0] * 4, Matrix([[0, 0] for _ in range(p)])
                exec(actual, dict(T=language, P=p, K=2048, GROUPED_OFFSETS=grouped,
                                  partial=Matrix(values), meta=meta, higher=[0] * 256,
                                  out=[0] * 2048, offsets=offsets))
                for worker in range(p):
                    self.assertEqual(offsets[worker, 0], sum(sum(row[meta[0] + 1:]) for row in values[:worker]))
                    self.assertEqual(offsets[worker, 1], sum(row[meta[0]] for row in values[:worker]) if meta[0] >= 0 else 0)

    def test_disabled_finish_matches_measured_r1_ast(self):
        class Disable(ast.NodeTransformer):
            def branch(self, nodes):
                result = []
                for node in nodes:
                    value = self.visit(node)
                    if value is not None:
                        result.extend(value if isinstance(value, list) else [value])
                return result

            def visit_Assert(self, node):
                if "ADAPTIVE_TIE" in ast.unparse(node):
                    return None
                return node
            def visit_If(self, node):
                if ast.unparse(node.test) in {"STABLE_TIE or ADAPTIVE_TIE", "GROUPED_OFFSETS", "STAGE_INPUT", "FUSE_QUOTA", "STAGE_OVERFLOW"}:
                    return self.branch(node.orelse)
                if ast.unparse(node.test) == "not SKIP_INDEX_PREFIX or round < 4 or round >= 6":
                    return self.branch(node.body)
                if ast.unparse(node.test) == "not FUSE_QUOTA":
                    return self.branch(node.body)
                if ast.unparse(node.test) == "not EARLY_EXIT or state[0] > 0":
                    return self.branch(node.body)
                return self.generic_visit(node)

            def visit_IfExp(self, node):
                if ast.unparse(node.test) in {"STABLE_TIE", "ADAPTIVE_TIE", "STAGE_INPUT", "STAGE_OVERFLOW"}:
                    return self.visit(node.orelse)
                return self.generic_visit(node)

        raw = subprocess.check_output(["git", "show", "03c8f916:src/tileops/_kernels/attention/topk_select_decode_pipeline_ascend.py"], cwd=ROOT)
        old = ast.parse(raw)
        current = ast.parse(SOURCE.read_text())
        prepare = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == "decode_prepare")
        self.assertEqual(prepare.args.args[-1].arg, "GROUPED_OFFSETS")
        self.assertIs(prepare.args.defaults[-1].value, False)
        prepare.args.args.pop()
        prepare.args.defaults.pop()
        compact = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == "decode_compact")
        compact.args.args.pop()
        compact.args.defaults.pop()
        compact.args.args.pop()
        compact.args.defaults.pop()
        compact.body = [n for n in compact.body if not (isinstance(n, ast.Assert) and "SCAN_WIDTH" in ast.unparse(n))]
        class Narrow(ast.NodeTransformer):
            def visit_Name(self, node):
                return ast.copy_location(ast.Constant(128), node) if node.id == "SCAN_WIDTH" else node
        Narrow().visit(compact)
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

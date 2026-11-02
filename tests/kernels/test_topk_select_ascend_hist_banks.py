"""CPU execution of the A2 bank index and reduction AST, without TileLang."""
import ast
import contextlib
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "src/tileops/_kernels/attention/topk_select_ascend.py"
TREE = ast.parse(SOURCE.read_text())
INDEX = next(n for n in ast.walk(TREE) if isinstance(n, ast.IfExp)
             and ast.unparse(n.test) == "HIST_BANKS == 1"
             and "lane % HIST_BANKS" in ast.unparse(n))
REDUCE = next(n for n in ast.walk(TREE) if isinstance(n, ast.If)
              and ast.unparse(n.test) == "HIST_BANKS > 1 and round == 0")
INDEX_CODE = compile(ast.Expression(INDEX), str(SOURCE), "eval")
REDUCE_CODE = compile(ast.Module(body=[REDUCE], type_ignores=[]), str(SOURCE), "exec")
LANGUAGE = SimpleNamespace(max=max, min=min, if_then_else=lambda c, a, b: a if c else b,
                           SimtVF=lambda **kw: contextlib.nullcontext(), Parallel=range,
                           serial=range, alloc_var=lambda dtype: 0, int32=int)


class BankHistogramTest(unittest.TestCase):
    def test_atomic_index_has_no_conditional_call(self):
        for node in ast.walk(TREE):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "T.atomic_add":
                self.assertFalse(any(isinstance(n, ast.Call)
                                     and ast.unparse(n.func) == "T.if_then_else"
                                     for n in ast.walk(node.args[0])))

    def test_parallel_extents_are_static(self):
        for node in ast.walk(TREE):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "T.Parallel":
                for extent in node.args:
                    names = {n.id for n in ast.walk(extent) if isinstance(n, ast.Name)}
                    self.assertLessEqual(names, {"H", "K", "HIST_BANKS"})
                    self.assertFalse(any(isinstance(n, ast.Call) for n in ast.walk(extent)))

    def test_bank_initialization_only_in_first_round(self):
        zero = next(n for n in ast.walk(TREE) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "HIST_BANKS > 1 and round == 0"
                    and isinstance(n.body[0], ast.For))
        code = compile(ast.Module(body=[zero], type_ignores=[]), str(SOURCE), "exec")
        for round_index in range(5):
            hist = [7] * 1281
            exec(code, dict(T=LANGUAGE, HIST_BANKS=4, round=round_index, hist=hist))
            self.assertEqual(hist[:257], [7] * 257)
            self.assertEqual(hist[257:], [0 if round_index == 0 else 7] * 1024)

    def test_first_round_reduction_and_overflow(self):
        rng = random.Random(20261102)
        for digits in [[128] * 65536, [0, 255] * 8192,
                       [rng.randrange(256) for _ in range(16384)]]:
            for window in [(0, len(digits)), (9, len(digits) - 11), (12, 12)]:
                hist = [0] * 1281
                expected = [0] * 256
                for i in range(*window):
                    digit = digits[i]
                    index = eval(INDEX_CODE, dict(T=LANGUAGE, HIST_BANKS=4,
                                                  round=0, lane=i % 128, digit=digit))
                    self.assertTrue(257 <= index < 1281)
                    hist[index] += 1
                    expected[digit] += 1
                exec(REDUCE_CODE, dict(T=LANGUAGE, HIST_BANKS=4, round=0, H=128, hist=hist))
                self.assertEqual(hist[:256], expected)
                self.assertEqual(hist[256], 0)

    def test_later_rounds_and_default_use_original_buckets(self):
        for banks in [1, 4]:
            for round_index in range(5):
                if banks == 4 and round_index == 0:
                    continue
                for digit in range(256):
                    self.assertEqual(eval(INDEX_CODE, dict(T=LANGUAGE, HIST_BANKS=banks,
                                                          round=round_index, lane=127, digit=digit)), digit)
                hist = [42] * 1281
                exec(REDUCE_CODE, dict(T=LANGUAGE, HIST_BANKS=banks,
                                      round=round_index, H=128, hist=hist))
                self.assertEqual(hist, [42] * 1281)


if __name__ == "__main__":
    unittest.main()

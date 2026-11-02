"""Execute the actual cached compact AST on CPU, including overflow counting."""

import ast
import copy
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "src/tileops/_kernels/attention/topk_select_ascend.py"
TREE = ast.parse(SOURCE.read_text())
ARM = next(n for n in ast.walk(TREE) if isinstance(n, ast.If)
           and ast.unparse(n.test) == "CACHE_COARSE and round == 0" and len(n.body) == 1
           and isinstance(n.body[0], ast.For))


class Atomics(ast.NodeTransformer):
    def visit_Call(self, node):
        if ast.unparse(node.func) == "T.atomic_add":
            target = node.args[0]
            return ast.Call(func=ast.Name(id="add", ctx=ast.Load()),
                            args=[target.value, target.slice, node.args[1]], keywords=[])
        return self.generic_visit(node)


CODE = compile(ast.fix_missing_locations(Atomics().visit(
    ast.Module(body=copy.deepcopy(ARM.body), type_ignores=[]))), str(SOURCE), "exec")


class Array(dict):
    pass


class CacheCompactTest(unittest.TestCase):
    def test_windows_groups_and_overflow(self):
        for equal in [False, True]:
            for reverse in [False, True]:
                rng = random.Random(4)
                n, start, end, quota = 4096, 9, 4000, 1024
                coarse = [128 if equal else rng.randrange(256) for _ in range(n)]
                counts = [coarse[start:end].count(b) for b in range(256)]
                before = [sum(counts[b + 1:]) for b in range(256)]
                threshold = next(b for b in range(256) if before[b] <= quota < before[b] + counts[b])
                state = [quota - before[threshold], 0, threshold, end - start, before[threshold], 0, 0]
                output, keys, indices = Array(), Array(), Array()

                class Input:
                    def __getitem__(self, index):
                        batch, i, group = index
                        self_test.assertEqual((batch, group), (0, 1))
                        self_test.assertEqual(coarse[i], threshold)
                        return i * 3

                def add(array, index, value):
                    old = array[index]
                    array[index] += value
                    return old

                self_test = self
                language = SimpleNamespace(serial=range, Parallel=(lambda h: reversed(range(h))) if reverse else range,
                                           ceildiv=lambda a, b: (a + b - 1) // b,
                                           Cast=lambda dtype, v: v, int32=int, max=max, min=min)
                env = dict(T=language, N=n, H=128, G=2, row=1, K=quota, M=2048,
                           starts=[start], ends=[end], state=state, coarse_cache=coarse,
                           hist=before[:], out=output, cached_key=keys, cached_index=indices,
                           x=Input(), score_key=lambda v: v, add=add)
                exec(CODE, env)
                self.assertEqual(set(output.values()), {i for i in range(start, end) if coarse[i] > threshold})
                self.assertEqual(state[6], counts[threshold])
                self.assertEqual(len(indices), min(2048, counts[threshold]))
                for slot, i in indices.items():
                    self.assertEqual(coarse[i], threshold)
                    self.assertEqual(keys[slot], i * 3)


if __name__ == "__main__":
    unittest.main()

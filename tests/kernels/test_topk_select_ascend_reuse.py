"""Check the default-off A0 source against the frozen measured baseline on CPU."""

import ast
import copy
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/tileops/_kernels/attention/topk_select_ascend.py"


class DisableReuse(ast.NodeTransformer):
    def visit_If(self, node):
        if ast.unparse(node.test) == "COARSE and REUSE_COARSE":
            return []
        return self.generic_visit(node)

    def visit_IfExp(self, node):
        if ast.unparse(node.test) == "COARSE and REUSE_COARSE":
            return self.visit(node.orelse)
        return self.generic_visit(node)


class ReuseSourceTest(unittest.TestCase):
    def test_disabled_path_matches_frozen_baseline(self):
        baseline = ast.parse(subprocess.check_output(
            ["git", "show", "7247ad4f355c62a8679dcdc8fdf182c0b219db5e:" + SOURCE], cwd=ROOT
        ))
        candidate = ast.parse((ROOT / SOURCE).read_text())
        function = next(n for n in candidate.body if isinstance(n, ast.FunctionDef))
        self.assertEqual(function.args.args[-1].arg, "REUSE_COARSE")
        self.assertIs(function.args.defaults[-1].value, False)
        function.args.args.pop()
        function.args.defaults.pop()
        disabled = DisableReuse().visit(copy.deepcopy(candidate))
        self.assertEqual(ast.dump(disabled), ast.dump(baseline))


if __name__ == "__main__":
    unittest.main()

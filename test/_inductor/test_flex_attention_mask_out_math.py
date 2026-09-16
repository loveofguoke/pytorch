"""CPU checks of expressions extracted from the NPU mask-out templates.

These verify scalar/tile math, not Triton compilation or device execution.
"""

import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TEMPLATE = Path(__file__).resolve().parents[2] / (
    "torch_npu/_inductor/kernel/flexattention_template.py"
)


class Tile(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype)


def exp(x):
    return np.exp(x).view(Tile)


class TestMaskOutMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"))
        cls.templates = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        cls.tl = SimpleNamespace(math=SimpleNamespace(exp=exp), where=np.where)

    def test_partial_probability_retains_fp32_for_score_gradient(self):
        source = self.templates["flex_attention_backward_dkdv_only_source"]
        source = source.split("def bwd_dkdv_block_mn(", 1)[1]
        expression = re.search(r"^    pT = (.+)$", source, re.MULTILINE)[1]
        scores = np.array([[0.001, -0.001]], dtype=np.float32)
        lse = np.array([np.log(np.exp(scores).sum())], dtype=np.float32)
        p = eval(expression, {"tl": self.tl, "qkT": scores,
                              "lse": lse, "MATMUL_PRECISION": np.float16})
        self.assertEqual(p.dtype, np.float32)
        expected = np.exp(scores - lse[:, None])
        dp_minus_delta = np.array([[0.75, -0.75]], dtype=np.float32)
        np.testing.assert_array_equal(p * dp_minus_delta, expected * dp_minus_delta)
        self.assertGreater(
            np.max(np.abs(expected - expected.astype(np.float16).astype(np.float32))),
            0,
        )

    def test_full_dq_excludes_padded_keys(self):
        source = self.templates["flex_attention_backward_qmajor_dq_source"]
        source = source.split("if HAS_FULL_BLOCKS:", 1)[1]
        expression = re.search(r"qk = (tl.where\(offs_n.+)\n", source)[1]
        scores = np.array([[2.0, 0.0]], dtype=np.float32)
        masked = eval(expression, {"tl": self.tl, "qk": scores,
                                   "offs_n": np.array([0, 1]), "KV_LEN": 1})
        p = np.exp(masked - np.float32(2.0))
        np.testing.assert_array_equal(p, [[1.0, 0.0]])

    def test_full_forward_excludes_padding_from_softmax_denominator(self):
        source = self.templates["compute_forward_block_mn_full"]
        expression = re.search(
            r"if CHECK_BLOCK_BOUNDARY:\n        post_mod_scores = (.+)", source
        )[1]
        scores = np.array([[2.0, 0.0]], dtype=np.float32)
        masked = eval(expression, {
            "tl": self.tl, "post_mod_scores": scores,
            "offs_n": np.array([[0, 1]]), "KV_LEN": 1,
        })
        probabilities = np.exp(masked - masked.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        np.testing.assert_array_equal(probabilities, [[1.0, 0.0]])
        # Zero-filled padded V must not dilute a valid value of 3.
        np.testing.assert_array_equal(probabilities @ np.array([[3.0], [0.0]]), [[3.0]])

    def test_full_dkdv_excludes_padded_queries_and_keys(self):
        source = self.templates["flex_attention_backward_dkdv_only_source"]
        source = source.split("def bwd_dkdv_full_block_mn(", 1)[1]
        expression = re.search(r"qkT = (tl.where\(\n.*?\n    \))", source, re.S)[1]
        scores = np.zeros((2, 2), dtype=np.float32)
        masked = eval(expression, {"tl": self.tl, "qkT": scores,
                                   "valid_m": np.array([True, False]),
                                   "offs_n1": np.array([0, 1]), "KV_LEN": 1})
        np.testing.assert_array_equal(np.exp(masked), [[1.0, 0.0], [0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()

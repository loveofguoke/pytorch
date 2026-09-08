import functools

import torch
import torch_npu
import torch_npu._inductor  # noqa: F401
from torch._inductor import metrics
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing import FileCheck
from torch_npu.testing.testcase import TestCase, run_tests


class TestFlexAttention(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        metrics.reset()

    def test_epilogue_fused(self):
        @torch.compile
        def f(q, k, v):
            return flex_attention(q, k, v).cos()

        q, k, v = (
            torch.randn(1, 8, 1024, 64, device="npu") for _ in range(3)
        )
        _, code = run_and_get_code(f, q, k, v)

        # FileCheck().check("triton_tem_fused").check_not("poi_fused_cos").run(
        #     code[0]
        # )
        accessed_bytes = 1 * 8 * 1024 * 64 * torch.float32.itemsize
        num_accesses = 6
        # TODO: Get rid of this fudge factor
        # We need this fudge factor for now as we write the extraneous logsumexp
        num_accesses += 1
        self.assertLess(metrics.num_bytes_accessed, accessed_bytes * num_accesses)

    def test_kernel_options_argument_is_respected(self):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device="npu",
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        _, code = run_and_get_code(
            torch.compile(flex_attention),
            q,
            k,
            v,
            kernel_options={"BLOCK_M": 16},
        )

        FileCheck().check("BLOCK_M : tl.constexpr = 16").run(code[0])

    def test_mask_mod_can_index_captured_tensor(self):
        seq_len = 128
        allowed = torch.tril(
            torch.ones(seq_len, seq_len, device="npu", dtype=torch.bool)
        )

        def mask_mod(_batch, _head, q_idx, kv_idx):
            return allowed[q_idx, kv_idx]

        block_mask = create_block_mask(
            mask_mod,
            B=1,
            H=1,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device="npu",
        )
        compiled_inputs = tuple(
            torch.randn(
                1, 2, seq_len, 64, device="npu", requires_grad=True
            )
            for _ in range(3)
        )
        reference_inputs = tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in compiled_inputs
        )

        actual = torch.compile(flex_attention)(
            *compiled_inputs, block_mask=block_mask
        )
        expected = flex_attention(*reference_inputs, block_mask=block_mask)
        actual.sum().backward()
        expected.sum().backward()

        self.assertRtolEqual(actual, expected)
        for actual_input, expected_input in zip(
            compiled_inputs, reference_inputs
        ):
            self.assertRtolEqual(actual_input.grad, expected_input.grad)


if __name__ == "__main__":
    run_tests()

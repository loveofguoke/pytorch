import pytest
import torch
import torch_npu

from torch.nn.attention.flex_attention import create_block_mask, flex_attention


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
def test_scalar_shift_mask_stays_pointwise_with_forward_and_backward():
    torch.npu.set_device(0)
    torch.manual_seed(61)
    q, k, v = [torch.randn(1, 2, 128, 64, device="npu", requires_grad=True) for _ in range(3)]

    def mask_mod(b, h, q, k):
        return (q >> 1) >= (k >> 1)

    mask = create_block_mask(mask_mod, 1, 2, 128, 128, device="npu")
    actual = torch.compile(flex_attention, fullgraph=True)(q, k, v, block_mask=mask)
    positions = torch.arange(128, device="npu")
    allowed = (positions[:, None] >> 1) >= (positions[None, :] >> 1)
    scores = q @ k.transpose(-2, -1) / 8
    expected = scores.masked_fill(~allowed, float("-inf")).softmax(-1) @ v
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    gradient = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), gradient)
    expected_grads = torch.autograd.grad(expected, (q, k, v), gradient)
    for got, ref in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

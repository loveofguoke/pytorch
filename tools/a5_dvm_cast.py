"""Reproduce a DVM cast on the GLM compact-RoPE strided input layout."""

import argparse
import json

import torch
import torch_npu
from torch_npu._inductor import dvm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contiguous", "strided"), required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(61)
    base = torch.randn(256, 96, generator=generator).bfloat16().to("npu")
    value = base[:, 64:].unsqueeze(1)
    if args.mode == "contiguous":
        value = value.contiguous()
    expected = value.cpu().float()
    output = torch.empty((256, 1, 32), device="npu", dtype=torch.float32)
    print(json.dumps({
        "mode": args.mode, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "shape": list(value.shape), "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "pointer_mod_32": value.data_ptr() % 32,
        "pointer_mod_256": value.data_ptr() % 256,
    }), flush=True)

    @dvm.kernel(ktype="vector", dyn_shape=False)
    def copy(k):
        loaded = k.view_load([256, 1, 32], [-1, -1, 1], dvm.bfloat16)
        converted = k.cast(loaded, dvm.float32)
        k.store(converted, dvm.float32)

    copy.run(value, output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    print(json.dumps({"status": "passed", "mode": args.mode}), flush=True)


if __name__ == "__main__":
    main()

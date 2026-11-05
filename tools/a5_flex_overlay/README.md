# A5 isolated FlexAttention deployment

This deploys only the empty-full-block metadata normalization from committed
`edbee7956d364ed794361eb9fb6510f35294f390`. The currently installed A5 package
differs substantially from this checkout, so replacing its whole kernel module
would be incorrect. The patch preserves `HAS_FULL_BLOCKS=False` and argument
positions. Newer NPU templates read full-metadata strides even with full blocks
disabled, so unused slots use independent zero-element tensors with matching
rank and dtype. Aliasing partial metadata is not valid here: template bodies
hard-code argument names that IR deduplication can rename. No nonzero data
buffer is allocated. `HAS_FULL_BLOCKS=False` still prevents kernel
loads through those slots; no blocks are promoted to full and the predicate is
unchanged. The original commit contains a compiled
captured-mask forward/QKV-gradient regression.

```bash
bash tools/a5_flex_overlay/prepare.sh /absolute/site-packages/torch_npu /fresh/overlay
TORCHINDUCTOR_CACHE_DIR=/fresh/overlay/inductor-cache \
TRITON_CACHE_DIR=/fresh/overlay/triton-cache \
PYTHONPATH=/fresh/overlay:$PYTHONPATH python ...
```

The destination must not exist. The installed package is copied, not changed.
`git apply --check` rejects an incompatible or already-fixed source. Record the
source package version, repository commit, overlay path and printed kernel hash
with each run. This is a diagnostic source overlay, not an official wheel or a
claim that the vendor build already includes the fix.

Use fresh compiler caches for each source overlay and backend. Changing these
Python lowering files without changing the wheel version does not reliably
invalidate existing generated code. The A5 DVM regression otherwise reused a
pre-fix static cast; a fresh cache regenerated the dynamic kernel and passed.

For the separately reproduced DVM fixed-shape/symbolic-stride failure, add
`--dvm-dynamic-strides` as the third argument. This also applies the narrow
`graph_build.is_fx_dynamic` correction: symbolic tensor strides require the
dynamic DVM kernel even when every dimension is concrete. Otherwise generated
`view_load(..., [-1, -1, 1], ...)` runs in a static kernel with missing runtime
stride metadata, causing the observed A5 MTE alignment exception. No RoPE math,
view-fusion setting, recomputation policy, or installed file is changed.

`tools/a5_dvm_cast.py` reproduces the generated BF16-to-FP32 cast independently
of GLM. In fresh A5 processes, static execution with runtime stride slots fails;
`--dynamic-strides` passes both contiguous and `(96, 32, 1)` strided inputs with
exact output equality. End-to-end DVM training must still pass before accepting
the overlay; this small probe alone is not model or distributed validation.

`--dvm-compat` additionally applies the Flex-only scalar right-shift lowering.
The GLM mask binary search uses a nonnegative integer midpoint, expressed as
`(lower + upper) >> 1`. Without this lowering DVM's global fallback creates a
buffer inside a pointwise Flex subgraph, which cannot contain such buffers.
The override maps only this subgraph operation to `ops.bitwise_right_shift`;
it does not change global fallback policy or attention math. The device
forward/QKV-gradient oracle and two-step single-rank GLM local-compile gate
pass with this overlay. Distributed validation is separate; this is not
whole-step GraphTrainer coverage.

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
PYTHONPATH=/fresh/overlay:$PYTHONPATH python ...
```

The destination must not exist. The installed package is copied, not changed.
`git apply --check` rejects an incompatible or already-fixed source. Record the
source package version, repository commit, overlay path and printed kernel hash
with each run. This is a diagnostic source overlay, not an official wheel or a
claim that the vendor build already includes the fix.

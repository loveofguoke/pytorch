#!/usr/bin/env bash
# Deploy only the already-reviewed edbee7956d3 empty-full-block hunk.
set -euo pipefail
SOURCE=${1:?installed torch_npu package directory required}
OUTPUT=${2:?fresh overlay parent directory required}
PATCH=$(cd "$(dirname "$0")" && pwd)/empty_full_blocks.patch
test -d "$SOURCE" && test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
cp -a "$SOURCE" "$OUTPUT/torch_npu"
git -C "$OUTPUT" apply --check --unsafe-paths "$PATCH"
git -C "$OUTPUT" apply --unsafe-paths "$PATCH"
sha256sum "$OUTPUT/torch_npu/_inductor/kernel/flex_attention.py"

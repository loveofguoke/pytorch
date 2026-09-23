"""DVM load/view_load selection based on real tensor strides."""

from __future__ import annotations

from typing import Sequence

import torch
from torch._inductor.virtualized import V

from . import config as dvm_config
from .op_emitter import load, view_load


def _is_non_overlapping_and_dense(
    shape: Sequence[int], stride: Sequence[int]
) -> bool:
    """Return whether a concrete strided layout has neither overlap nor gaps."""
    if any(size == 0 for size in shape):
        return True

    dimensions = sorted(
        (step, size)
        for size, step in zip(shape, stride)
        if size > 1
    )
    expected_stride = 1
    for step, size in dimensions:
        if step != expected_stride:
            return False
        expected_stride *= size
    return True


def apply_dvm_input_layouts(
    call_args: Sequence[str],
    cont_flag_input: Sequence[bool],
    need_trans_input: Sequence[bool] = (),
) -> list[str]:
    """Materialize or transpose DVM inputs exactly as codegen requested."""
    updated_args = list(call_args)
    for index, skip_contiguous in enumerate(cont_flag_input):
        if not skip_contiguous:
            updated_args[index] += ".contiguous()"
    for index, transpose in enumerate(need_trans_input):
        if transpose:
            updated_args[index] += ".mT"
    return updated_args


def patch_gm_placeholder_strides_from_codegen_args(
    gm: torch.fx.GraphModule,
    arg_names: Sequence[str],
) -> None:
    """Patch placeholder meta['val'] with Inductor buffer layout strides at codegen time."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for node, name in zip(placeholders, arg_names):
        val = node.meta.get("val")
        if isinstance(val, (torch.SymInt, torch.SymFloat)):
            continue
        if not isinstance(val, torch.Tensor):
            continue
        buf = V.graph.try_get_buffer(name)
        if buf is None:
            continue
        layout_stride = tuple(buf.get_stride())
        if layout_stride == tuple(val.stride()):
            continue
        node.meta["val"] = torch.empty_strided(
            val.shape,
            layout_stride,
            dtype=val.dtype,
            device=val.device,
        )


def codegen_maybe_view_load(
    shape: Sequence,
    stride: Sequence,
    dtype: torch.dtype,
    *,
    is_symbolic: bool,
) -> tuple[str, bool]:
    """Return (expr, skip_cont).

    skip_cont=False means the caller must manually materialize a non-contiguous
    input with .contiguous() before launching the DVM kernel.
    """
    if dvm_config.view_fusion_level == 0:
        return load(shape, dtype), False

    if dvm_config.view_fusion_level == 2:
        return view_load(shape, stride, dtype), True

    if is_symbolic:
        return load(shape, dtype), False

    if (
        stride[-1] == 1
        and shape[-1] != 1
        and _is_non_overlapping_and_dense(shape, stride)
    ):
        return view_load(shape, stride, dtype), True

    return load(shape, dtype), False

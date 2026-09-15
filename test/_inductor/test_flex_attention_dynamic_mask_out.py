import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOWERING_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flex_attention.py"
TEMPLATE_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flexattention_template.py"


class TestFlexAttentionDynamicMaskOutSource(unittest.TestCase):
    def test_backward_masks_sparse_query_rows_to_local_query_length(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        self.assertIn("valid_m = offs_m1 < Q_LEN", template)
        self.assertIn(
            "if IS_DIVISIBLE and not GUARD_SPARSE_Q_ROWS:",
            template,
        )
        self.assertIn(
            'lse = tl.load(LSE + offs_m1, mask=valid_m, other=float("-inf"))',
            template,
        )
        self.assertIn(
            "tl.load(DELTA + offs_m1, mask=valid_m, other=0.0)",
            template,
        )
        self.assertIn(
            "if GUARD_SPARSE_Q_ROWS:",
            template,
        )
        self.assertIn(
            "mask_mod_output = mask_mod_output & valid_m[:, None]",
            template,
        )

    def test_cp_backward_uses_largest_supported_kv_tiles(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "asymmetric_q_kv = not V.graph.sizevars.evaluate_expr(\n"
            "            sympy.Eq(seq_len_q, seq_len_kv)\n"
            "        )",
            lowering,
        )
        self.assertIn(
            'supported_bwd_dq_configs,\n                "BLOCK_N2"',
            lowering,
        )
        self.assertIn(
            'supported_bwd_dkdv_configs,\n                "BLOCK_N1"',
            lowering,
        )
        self.assertIn(
            'kernel_options["GUARD_SPARSE_Q_ROWS"] = asymmetric_q_kv',
            lowering,
        )
        self.assertIn(
            '"No numerically supported asymmetric flex attention "',
            lowering,
        )

    def test_mask_out_uses_largest_supported_risky_axis_tiles(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "dict_configs = _keep_largest_supported_block_configs(",
            lowering,
        )
        self.assertIn(
            'supported_bwd_dq_configs,\n                "BLOCK_N2"',
            lowering,
        )
        self.assertIn(
            'supported_bwd_dkdv_configs,\n                "BLOCK_N1"',
            lowering,
        )

    def test_largest_block_filter_preserves_nonempty_candidate_sets(self):
        tree = ast.parse(LOWERING_PATH.read_text(encoding="utf-8"))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_keep_largest_supported_block_configs"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[helper], type_ignores=[]),
                str(LOWERING_PATH),
                "exec",
            ),
            namespace,
        )
        keep_largest = namespace["_keep_largest_supported_block_configs"]

        self.assertEqual(keep_largest([], [], "BLOCK_M"), [])
        for sparse_block_size in (16, 32, 64, 128, 256):
            blocks = [
                block
                for block in (128, 64, 32, 16)
                if block <= sparse_block_size
                and sparse_block_size % block == 0
            ]
            configs = [
                {"BLOCK_M": block_m, "BLOCK_N": block_n}
                for block_m in blocks
                for block_n in blocks
            ]
            selected = keep_largest(configs, configs, "BLOCK_M")
            self.assertTrue(selected)
            self.assertEqual(
                {config["BLOCK_M"] for config in selected},
                {max(blocks)},
            )

        supported_configs = [
            {"BLOCK_M": block_m, "BLOCK_N": 128}
            for block_m in (128, 64, 32, 16)
        ]
        unsafe_explicit_config = [{"BLOCK_M": 32, "BLOCK_N": 128}]
        self.assertEqual(
            keep_largest(
                unsafe_explicit_config,
                supported_configs,
                "BLOCK_M",
            ),
            [],
        )
        safe_explicit_config = [{"BLOCK_M": 128, "BLOCK_N": 32}]
        self.assertEqual(
            keep_largest(
                safe_explicit_config,
                supported_configs,
                "BLOCK_M",
            ),
            safe_explicit_config,
        )

    def test_backward_preserves_divisible_self_attention_loads(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        self.assertIn("lse = tl.load(LSE + offs_m1)\n", template)
        self.assertIn("Di = tl.load(DELTA + offs_m1)\n", template)

    def test_split_backward_keeps_explicit_mutation_outputs(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            '{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DQ",',
            template,
        )
        self.assertIn(
            '{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DV", "DK",',
            template,
        )
        self.assertNotIn("broadcasted_grad_key_accum = dkdv_result", lowering)
        self.assertNotIn("grad_query = dq_result", lowering)

    def test_tasklist_unsplit_query_bounds_use_index_dtype(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        # The dynamic task-list kernel uses the same loop for split and
        # unsplit work items.  Keep the unsplit zero in the same scalar type as
        # q_hi so Triton does not change the loop index type across branches.
        self.assertEqual(
            template.count(
                "if is_split == 0:\n"
                "                q_begin = tl.zeros([], dtype=INDEX_DTYPE)"
            ),
            1,
        )
        self.assertEqual(
            template.count(
                "if is_split == 0:\n"
                "                    full_q_begin = tl.zeros([], dtype=INDEX_DTYPE)"
            ),
            1,
        )

    def test_backward_masks_missing_partial_block_only_for_cp_rows(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "safe_partial_block_idx = tl.maximum(partial_block_idx, 0)",
            template,
        )
        self.assertIn(
            "if GUARD_SPARSE_Q_ROWS:\n"
            "            # CP can produce sparse rows outside the local Q shard.",
            template,
        )
        self.assertIn(
            "mask=valid_partial_block,\n                other=False,",
            template,
        )
        self.assertIn(
            "else:\n"
            "            # Preserve the original self-attention codegen.",
            template,
        )
        self.assertIn(
            "mask_mod_output = mask_mod_output & (partial_block_idx >= 0)",
            template,
        )

    def test_mask_out_cache_loads_restore_boolean_dtype(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        # The compact mask is deliberately stored as int8. Every consumer
        # must convert the loaded byte back to a predicate before passing it
        # to tl.where; newer Triton versions reject integer conditions.
        self.assertIn(
            "arg_SPARSE_MASK + flat_blk * SPARSE_MASK_STRIDE_BLK + mask_offsets\n"
            "            ) != 0",
            template,
        )
        self.assertIn(
            "mask=valid_partial_block,\n"
            "                other=False,\n"
            "            ) != 0",
            template,
        )
        self.assertIn(
            "mask_mod_output = tl.load(mask_base + mask_offsets) != 0",
            template,
        )

    def test_optional_full_block_strides_are_guarded_during_rendering(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        dq_guarded_strides = """{% if HAS_FULL_BLOCKS %}
    stride_full_kv_num_blks_z = {{stride(\"FULL_KV_NUM_BLKS\", 0)}}
    stride_full_kv_num_blks_h = {{stride(\"FULL_KV_NUM_BLKS\", 1)}}
    stride_full_kv_num_blks_m = {{stride(\"FULL_KV_NUM_BLKS\", 2)}}
    stride_full_kv_idx_z = {{stride(\"FULL_KV_IDX\", 0)}}
    stride_full_kv_idx_h = {{stride(\"FULL_KV_IDX\", 1)}}
    stride_full_kv_idx_m = {{stride(\"FULL_KV_IDX\", 2)}}
    stride_full_kv_idx_blk = {{stride(\"FULL_KV_IDX\", 3)}}
{% endif %}"""
        tasklist_guarded_strides = """{% if HAS_FULL_BLOCKS %}
    stride_full_q_num_blks_z = {{stride(\"FULL_Q_NUM_BLKS\", 0)}}
    stride_full_q_num_blks_h = {{stride(\"FULL_Q_NUM_BLKS\", 1)}}
    stride_full_q_idx_z = {{stride(\"FULL_Q_IDX\", 0)}}
    stride_full_q_idx_h = {{stride(\"FULL_Q_IDX\", 1)}}
    stride_full_q_idx_n = {{stride(\"FULL_Q_IDX\", 2)}}
{% endif %}"""

        self.assertIn(dq_guarded_strides, template)
        self.assertIn(tasklist_guarded_strides, template)

    def test_full_q_blocks_use_their_own_layout_in_dkdv(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'stride_full_q_idx_n = {{stride("FULL_Q_IDX", 2)}}', template
        )
        self.assertIn(
            "sparse_full_q_idx_offset = sparse_hz_offset * "
            "stride_full_q_idx_h + pid_mask * stride_full_q_idx_n",
            template,
        )
        self.assertIn(
            "q_indices = FULL_Q_IDX + sparse_full_q_idx_offset", template
        )
        self.assertIn(
            "sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + "
            "sparse_full_q_num_blks_offset)",
            template,
        )

    def test_direct_backward_supplies_public_kernel_option_defaults(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        backward = lowering.rsplit(
            "def flex_attention_backward(*args, **kwargs):", 1
        )[1]
        normalization = backward.split("fwd_placeholder_inps =", 1)[0]

        self.assertIn(
            'kernel_options.setdefault("PRESCALE_QK", False)',
            normalization,
        )
        self.assertIn(
            'kernel_options.setdefault("WRITE_DQ", True)',
            normalization,
        )

    def test_short_query_uses_community_flex_decoding(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        self.assertIn("flex_decoding_template,", lowering)
        self.assertIn("def _use_flex_decoding(", lowering)
        self.assertIn("def _create_npu_flex_decoding_kernel(*args):", lowering)
        self.assertIn(
            "flex_decoding_template.maybe_append_choice(",
            lowering,
        )
        self.assertIn('"flex_decoding",', lowering)
        self.assertIn(
            'cur_kernel_options.setdefault("USE_TMA", bool(torch.xpu.is_available()))',
            lowering,
        )

    def test_npu_decoding_uses_community_template(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("flex_decoding_npu", template)
        self.assertNotIn("flex_decoding_npu_source", template)
        self.assertIn(
            "flex_decoding_template = _wrap_upstream_template(", template
        )
        self.assertIn("_upstream_flex_decoding_template", template)

    @unittest.skip("temporarily disabled pending decoding dispatch update")
    def test_decoding_dispatch_precedes_mask_out_dispatch(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        forward = lowering.split(
            "def _register_npu_inductor_flex_attention():", 1
        )[1]
        self.assertLess(
            forward.index("_use_flex_decoding("),
            forward.index("configured_mask_out = bool("),
        )

    def test_forced_decoding_rejects_unsupported_inputs(self):
        lowering = LOWERING_PATH.read_text(encoding="utf-8")
        self.assertIn('backend = kernel_options.get("BACKEND", "AUTO")', lowering)
        self.assertIn(
            'if backend == "TRITON_DECODE" and not use_flex_decoding:', lowering
        )
        self.assertIn(
            "BACKEND='TRITON_DECODE' was specified but flex_decoding cannot be used",
            lowering,
        )


if __name__ == "__main__":
    unittest.main()

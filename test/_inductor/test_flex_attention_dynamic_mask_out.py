import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOWERING_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flex_attention.py"
TEMPLATE_PATH = REPO_ROOT / "torch_npu/_inductor/kernel/flexattention_template.py"


class TestFlexAttentionDynamicMaskOutSource(unittest.TestCase):
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

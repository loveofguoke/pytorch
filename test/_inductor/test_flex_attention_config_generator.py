import unittest

from torch._inductor import config as inductor_config

from torch_npu._inductor.kernel.flex_attention_config_generator import (
    FlexAttentionConfigGenerator,
    FlexMode,
)


class TestFlexAttentionConfigGenerator(unittest.TestCase):
    def _generate(self, mode=FlexMode.BWDDKDV, kernel_options=None):
        return FlexAttentionConfigGenerator(
            sparse_q_block_size=128,
            sparse_kv_block_size=128,
            mode=mode,
            kernel_options=kernel_options,
        ).generate_configs()

    def test_default_autotune_omits_unsafe_block_n_for_backward_modes(self):
        with inductor_config.patch(
            {
                "max_autotune": True,
                "max_autotune_flex_search_space": "DEFAULT",
            }
        ):
            for mode, block_n_key in (
                (FlexMode.BWD, "BLOCK_N1"),
                (FlexMode.BWDDQ, "BLOCK_N2"),
                (FlexMode.BWDDKDV, "BLOCK_N1"),
            ):
                with self.subTest(mode=mode):
                    configs = self._generate(mode)
                    self.assertNotIn(
                        128, {config[block_n_key] for config in configs}
                    )
            self.assertEqual(len(self._generate(FlexMode.FWD)), 16)

    def test_exhaustive_autotune_keeps_block_n_128(self):
        with inductor_config.patch(
            {
                "max_autotune": True,
                "max_autotune_flex_search_space": "EXHAUSTIVE",
            }
        ):
            configs = self._generate()

        self.assertEqual(len(configs), 16)
        self.assertIn(128, {config["BLOCK_N1"] for config in configs})

    def test_explicit_block_n_128_overrides_default_search_space(self):
        with inductor_config.patch(
            {
                "max_autotune": True,
                "max_autotune_flex_search_space": "DEFAULT",
            }
        ):
            configs = self._generate(kernel_options={"BLOCK_N1": 128})

        self.assertEqual(len(configs), 4)
        self.assertEqual({config["BLOCK_N1"] for config in configs}, {128})

    def test_fused_backward_keeps_block_n_required_by_explicit_block_m(self):
        with inductor_config.patch(
            {
                "max_autotune": True,
                "max_autotune_flex_search_space": "DEFAULT",
            }
        ):
            configs = self._generate(
                FlexMode.BWD, {"BLOCK_M1": 128, "BLOCK_N2": 128}
            )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["BLOCK_N1"], 128)

    def test_non_autotune_order_is_unchanged(self):
        with inductor_config.patch(
            {
                "max_autotune": False,
                "max_autotune_flex_search_space": "DEFAULT",
            }
        ):
            configs = self._generate()

        self.assertEqual(len(configs), 16)
        self.assertEqual(configs[0]["BLOCK_N1"], 128)


if __name__ == "__main__":
    unittest.main()

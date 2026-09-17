import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "torch_npu/_inductor/config.py"
SELECT_ALGORITHM_PATH = REPO_ROOT / "torch_npu/_inductor/select_algorithm.py"


def _load_precompile_thread_function():
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_obtain_precompile_thread_num"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "inductor_config": SimpleNamespace(compile_threads=32),
        "log": SimpleNamespace(info=lambda *args, **kwargs: None),
        "os": os,
        "sys": SimpleNamespace(exit=lambda code: None),
    }
    exec(compile(module, str(CONFIG_PATH), "exec"), namespace)
    return namespace["_obtain_precompile_thread_num"], namespace


class TestPrecompileControls(unittest.TestCase):
    def test_default_worker_count_is_bounded(self):
        get_worker_count, namespace = _load_precompile_thread_function()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            os, "cpu_count", return_value=128
        ):
            namespace["inductor_config"].compile_threads = 32
            self.assertEqual(get_worker_count(), 4)

        with patch.dict(os.environ, {}, clear=True), patch.object(
            os, "cpu_count", return_value=4096
        ):
            namespace["inductor_config"].compile_threads = 2
            self.assertEqual(get_worker_count(), 32)

        with patch.dict(os.environ, {}, clear=True), patch.object(
            os, "cpu_count", return_value=1
        ):
            namespace["inductor_config"].compile_threads = 32
            self.assertEqual(get_worker_count(), 1)

    def test_environment_override_is_preserved(self):
        get_worker_count, _ = _load_precompile_thread_function()

        with patch.dict(os.environ, {"TORCHNPU_PRECOMPILE_THREADS": "7"}):
            self.assertEqual(get_worker_count(), 7)

    def test_precompile_matches_upstream_triton_dispatch(self):
        tree = ast.parse(SELECT_ALGORITHM_PATH.read_text(encoding="utf-8"))
        precompile = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "precompile"
        )
        wait_on_futures = next(
            node
            for node in ast.walk(precompile)
            if isinstance(node, ast.FunctionDef) and node.name == "wait_on_futures"
        )

        precompile_source = ast.unparse(precompile)
        self.assertIn("c.kernel_hash_key()", precompile_source)
        self.assertIn("async_compile.triton", precompile_source)
        self.assertIn("async_compile.use_process_pool()", precompile_source)
        self.assertNotIn("c.hash_key() in seen_choices", precompile_source)

        wait_source = ast.unparse(wait_on_futures)
        self.assertIn("choice.mark_failed()", wait_source)
        self.assertIn("as_completed", wait_source)


if __name__ == "__main__":
    unittest.main()

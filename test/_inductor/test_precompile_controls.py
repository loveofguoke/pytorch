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

    def test_precompile_timeout_is_degraded_instead_of_propagated(self):
        tree = ast.parse(SELECT_ALGORITHM_PATH.read_text(encoding="utf-8"))
        wait_on_futures = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "wait_on_futures"
        )

        timeout_handlers = [
            handler
            for handler in ast.walk(wait_on_futures)
            if isinstance(handler, ast.ExceptHandler)
            and isinstance(handler.type, ast.Name)
            and handler.type.id == "TimeoutError"
        ]
        self.assertEqual(len(timeout_handlers), 1)

        handler_calls = {
            node.func.id
            for node in ast.walk(timeout_handlers[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_log_autotune_error", handler_calls)


if __name__ == "__main__":
    unittest.main()

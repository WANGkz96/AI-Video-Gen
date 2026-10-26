"""Exercise the scoped VAE policy without importing GPU/model dependencies."""
import ast
import os
import unittest
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class VaeCudnnPolicyTests(unittest.TestCase):
    def setUp(self):
        source = (Path(__file__).resolve().parents[1] / "scripts/run_longcat_avatar_batch.py").read_text(encoding="utf-8")
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_configure_vae_cudnn")
        self.enabled = False
        self.capability = (12, 0)

        @contextmanager
        def flags(*, enabled):
            previous = self.enabled
            self.enabled = enabled
            try:
                yield
            finally:
                self.enabled = previous

        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: self.capability),
                                backends=SimpleNamespace(cudnn=SimpleNamespace(flags=flags)))
        namespace = {"Any": object, "os": os, "torch": torch, "wraps": wraps}
        exec(compile(ast.Module(body=[function], type_ignores=[]), source, "exec"), namespace)
        self.configure = namespace["_configure_vae_cudnn"]
        self.vae = SimpleNamespace(encode=lambda x: (x, self.enabled), decode=lambda x: (x, self.enabled))

    def test_only_vae_calls_enable_cudnn_and_restore_audio_policy(self):
        with patch.dict(os.environ, {"LONGCAT_VAE_CUDNN": "1"}):
            self.configure(self.vae)
        self.assertEqual(self.vae.encode("input"), ("input", True))
        self.assertEqual(self.vae.decode("latent"), ("latent", True))
        self.assertFalse(self.enabled)

    def test_failure_also_restores_previous_policy(self):
        def fail(_):
            self.assertTrue(self.enabled)
            raise ValueError("decode failed")
        self.vae.decode = fail
        with patch.dict(os.environ, {"LONGCAT_VAE_CUDNN": "1"}):
            self.configure(self.vae)
        with self.assertRaisesRegex(ValueError, "decode failed"):
            self.vae.decode(None)
        self.assertFalse(self.enabled)

    def test_explicit_opt_out_preserves_historical_fallback(self):
        original = self.vae.decode
        with patch.dict(os.environ, {"LONGCAT_VAE_CUDNN": "0"}):
            self.configure(self.vae)
        self.assertIs(self.vae.decode, original)

    def test_other_gpu_architectures_keep_their_policy(self):
        self.capability = (9, 0)
        original = self.vae.decode
        self.configure(self.vae)
        self.assertIs(self.vae.decode, original)


if __name__ == "__main__":
    unittest.main()

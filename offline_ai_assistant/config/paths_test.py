"""
Tests for offline_ai_assistant.config.paths.

Uses unittest and temporary directories for secure path resolution and SafePathResolver.
"""

import os
import tempfile
import unittest
from pathlib import Path

from offline_ai_assistant.config.paths import (
    resolve_under,
    validate_path_under,
    SafePathResolver,
)


class TestResolveUnder(unittest.TestCase):
    def test_returns_path_when_under_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            candidate = base / "sub" / "file.txt"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.touch()
            result = resolve_under(candidate, base)
            self.assertIsNotNone(result)
            self.assertEqual(result, candidate.resolve())

    def test_returns_path_when_equal_to_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result = resolve_under(base, base)
            self.assertIsNotNone(result)
            self.assertEqual(result, base.resolve())

    def test_returns_none_when_path_escapes_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Path that resolves outside base (e.g. sibling or parent)
            escape = base / ".." / "other"
            result = resolve_under(escape, base)
            self.assertIsNone(result)

    def test_returns_none_when_absolute_outside_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Another temp dir is never under base
            with tempfile.TemporaryDirectory() as other:
                candidate = Path(other) / "file.txt"
                result = resolve_under(candidate, base)
                self.assertIsNone(result)

    def test_returns_resolved_path_for_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            candidate = base / "a" / "b" / "c"
            candidate.mkdir(parents=True, exist_ok=True)
            result = resolve_under(candidate, base)
            self.assertIsNotNone(result)
            self.assertTrue(result.is_absolute())
            self.assertEqual(result, candidate.resolve())

    def test_accepts_string_like_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            candidate = base / "ok"
            candidate.touch()
            result = resolve_under(candidate, base)
            self.assertIsNotNone(result)


class TestValidatePathUnder(unittest.TestCase):
    def test_returns_true_and_path_when_under_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            candidate = base / "sub" / "f"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.touch()
            valid, out = validate_path_under(candidate, base)
            self.assertTrue(valid)
            self.assertIn("f", out)

    def test_returns_false_when_not_under_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with tempfile.TemporaryDirectory() as other:
                candidate = Path(other) / "file"
                valid, msg = validate_path_under(candidate, base)
                self.assertFalse(valid)
                self.assertIn("not under", msg.lower())

    def test_must_exist_false_does_not_require_existence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            candidate = base / "nonexistent" / "file.txt"
            valid, out = validate_path_under(candidate, base, must_exist=False)
            self.assertTrue(valid)

    def test_must_exist_true_returns_false_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            candidate = base / "missing.txt"
            valid, msg = validate_path_under(candidate, base, must_exist=True)
            self.assertFalse(valid)
            self.assertIn("exist", msg.lower())

    def test_must_exist_true_returns_true_when_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            candidate = base / "exists.txt"
            candidate.touch()
            valid, out = validate_path_under(candidate, base, must_exist=True)
            self.assertTrue(valid)
            self.assertIn("exists.txt", out)


class TestSafePathResolver(unittest.TestCase):
    def setUp(self):
        self._ud = tempfile.mkdtemp()
        self._md = tempfile.mkdtemp()
        self._dd = tempfile.mkdtemp()
        self.user_data = Path(self._ud)
        self.models = Path(self._md)
        self.docs = Path(self._dd)
        self.resolver = SafePathResolver(
            user_data_dir=self.user_data,
            models_dir=self.models,
            docs_dir=self.docs,
        )

    def tearDown(self):
        import shutil
        for d in (self._ud, self._md, self._dd):
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

    def test_init_resolves_paths(self):
        self.assertEqual(self.resolver.user_data_dir, self.user_data.resolve())
        self.assertEqual(self.resolver.models_dir, self.models.resolve())
        self.assertEqual(self.resolver.docs_dir, self.docs.resolve())

    def test_resolve_docs_path_under_docs_returns_path(self):
        p = self.docs / "sub" / "doc.pdf"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        result = self.resolver.resolve_docs_path(p)
        self.assertIsNotNone(result)
        self.assertEqual(result, p.resolve())

    def test_resolve_docs_path_outside_returns_none(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                result = self.resolver.resolve_docs_path(Path(f.name))
                self.assertIsNone(result)
            finally:
                os.unlink(f.name)

    def test_resolve_models_path_under_models_returns_path(self):
        p = self.models / "model.gguf"
        p.touch()
        result = self.resolver.resolve_models_path(p)
        self.assertIsNotNone(result)
        self.assertEqual(result, p.resolve())

    def test_resolve_models_path_under_user_data_returns_path(self):
        p = self.user_data / "models" / "model.gguf"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        result = self.resolver.resolve_models_path(p)
        self.assertIsNotNone(result)
        self.assertEqual(result, p.resolve())

    def test_resolve_models_path_outside_returns_none(self):
        with tempfile.TemporaryDirectory() as other:
            p = Path(other) / "x.gguf"
            result = self.resolver.resolve_models_path(p)
            self.assertIsNone(result)

    def test_resolve_llm_model_path_same_as_models_path(self):
        p = self.models / "llama.gguf"
        p.touch()
        self.assertIsNotNone(self.resolver.resolve_llm_model_path(p))
        self.assertEqual(
            self.resolver.resolve_llm_model_path(p),
            self.resolver.resolve_models_path(p),
        )

    def test_safe_join_docs_returns_path_under_docs(self):
        p = self.resolver.safe_join_docs("folder", "file.pdf")
        self.assertEqual(p, self.resolver.docs_dir / "folder" / "file.pdf")
        self.assertTrue(str(p).startswith(str(self.resolver.docs_dir)))

    def test_safe_join_models_returns_path_under_models(self):
        p = self.resolver.safe_join_models("llama-2", "model.gguf")
        self.assertEqual(p, self.resolver.models_dir / "llama-2" / "model.gguf")
        self.assertTrue(str(p).startswith(str(self.resolver.models_dir)))


if __name__ == "__main__":
    unittest.main()

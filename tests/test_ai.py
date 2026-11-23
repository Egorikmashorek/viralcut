"""ИИ-помощник: провайдеры, ключи и советы по клипам."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from viralcut import ai, settings


def _conf(key: str = "sk-test-1234567890", provider: str = "deepseek") -> None:
    settings.update({"ai_provider": provider, "ai_key": key, "ai_base": "", "ai_model": ""})


class PromptTest(unittest.TestCase):
    def test_prompt_contains_clip_data(self):
        project = {"title": "Разбор недели", "language": "ru", "duration": 120,
                   "clips": [{"start": 0, "end": 20, "score": 9}]}
        clip = {"start": 1.0, "end": 9.0, "hook": "Почему это работает",
                "words": [{"word": "Почему"}, {"word": "это"}, {"word": "работает"}]}
        messages = ai.build_prompt(project, clip)
        self.assertEqual(messages[0]["role"], "system")
        user = messages[1]["content"]
        for piece in ("Разбор недели", "ru", "Почему это работает", "1.0–9.0"):
            self.assertIn(piece, user)

    def test_prompt_without_clip_asks_general(self):
        messages = ai.build_prompt({"title": "Пусто"}, None)
        self.assertIn("общие советы", messages[1]["content"])

    def test_long_transcript_truncated(self):
        words = [{"word": "слово"} for _ in range(5000)]
        messages = ai.build_prompt({"title": "Длинно"}, {"start": 0, "end": 100, "words": words})
        self.assertLessEqual(len(messages[1]["content"]), ai.MAX_TRANSCRIPT_CHARS + 800)


class ProviderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "settings.json"
        self.patch = mock.patch.object(settings, "SETTINGS_FILE", self.tmp)
        self.patch.start()
        os.environ.pop("VIRALCUT_AI_KEY", None)

    def tearDown(self):
        self.patch.stop()

    def test_deepseek_is_default(self):
        status = ai.status()
        self.assertEqual(status["provider"], "deepseek")
        self.assertEqual(status["base"], "https://api.deepseek.com/v1")
        self.assertEqual(status["model"], "deepseek-chat")

    def test_provider_order_starts_with_deepseek(self):
        keys = [item["key"] for item in ai.status()["providers"]]
        self.assertEqual(keys[0], "deepseek")
        self.assertIn("polza", keys)

    def test_polza_base_url(self):
        settings.update({"ai_provider": "polza"})
        status = ai.status()
        self.assertEqual(status["base"], "https://polza.ai/api/v1")
        self.assertEqual(status["model"], "openai/gpt-4o-mini")

    def test_switching_provider_resets_model(self):
        settings.update({"ai_provider": "polza", "ai_model": "openai/gpt-4o"})
        settings.update({"ai_provider": "deepseek"})
        self.assertEqual(ai.status()["model"], "deepseek-chat")

    def test_key_not_exposed(self):
        _conf()
        self.assertTrue(ai.configured())
        self.assertEqual(ai.status()["key_hint"][:3], "sk-")
        public = settings.public(settings.get())
        self.assertEqual(public["ai_key"], "")
        self.assertTrue(public["ai_key_set"])

    def test_key_from_env(self):
        settings.update({"ai_key": ""})
        with mock.patch.dict(os.environ, {"VIRALCUT_AI_KEY": "sk-env-1"}):
            self.assertEqual(ai.key(), "sk-env-1")
            self.assertTrue(ai.status()["from_env"])


class RequestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "settings.json"
        self.patch = mock.patch.object(settings, "SETTINGS_FILE", self.tmp)
        self.patch.start()
        _conf(provider="custom")
        settings.update({"ai_base": "http://127.0.0.1:9", "ai_model": "test-model"})

    def tearDown(self):
        self.patch.stop()

    def _fake(self, payload: dict, status: int = 200):
        class Response:
            def __init__(self, data: bytes):
                self.data = data
            def read(self):
                return self.data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        body = json.dumps(payload).encode()
        if status >= 400:
            error = ai.urllib.error.HTTPError("http://x", status, "err", {}, None)
            error.read = lambda: body  # type: ignore[assignment]
            raise error
        return Response(body)

    def test_advice_success(self):
        payload = {"choices": [{"message": {"content": "1) Хук: тест"}}]}
        with mock.patch.object(ai.urllib.request, "urlopen", return_value=self._fake(payload)):
            result = ai.advice({"title": "Тест", "clips": []})
        self.assertEqual(result["advice"], "1) Хук: тест")
        self.assertEqual(result["model"], "test-model")

    def test_bad_key_message(self):
        with mock.patch.object(ai.urllib.request, "urlopen", side_effect=lambda *a, **k: self._fake({}, 401)):
            with self.assertRaises(ai.AIError) as ctx:
                ai.advice({"title": "Тест", "clips": []})
        self.assertIn("Ключ не подошёл", str(ctx.exception))

    def test_no_key_message(self):
        settings.update({"ai_key": ""})
        with self.assertRaises(ai.AIError) as ctx:
            ai.advice({"title": "Тест", "clips": []})
        self.assertIn("Не задан ключ", str(ctx.exception))

    def test_empty_answer(self):
        with mock.patch.object(ai.urllib.request, "urlopen",
                               return_value=self._fake({"choices": [{"message": {"content": "  "}}]})):
            with self.assertRaises(ai.AIError):
                ai.advice({"title": "Тест", "clips": []})


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "settings.json"
        self.patch = mock.patch.object(settings, "SETTINGS_FILE", self.tmp)
        self.patch.start()
        ai._MODELS_CACHE.update({"ts": 0.0, "base": "", "data": []})

    def tearDown(self):
        self.patch.stop()
        ai._MODELS_CACHE.update({"ts": 0.0, "base": "", "data": []})

    def test_catalog_parsed_and_filtered(self):
        payload = {"data": [
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini", "type": "chat", "owned_by": "openai"},
            {"id": "openai/whisper", "name": "Whisper", "type": "audio"},
        ]}
        class Response:
            def read(self): return json.dumps(payload).encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with mock.patch.object(ai.urllib.request, "urlopen", return_value=Response()):
            result = ai.models(force=True, provider="polza")
        ids = [item["id"] for item in result["models"]]
        self.assertEqual(ids, ["openai/gpt-4o-mini"])

    def test_fallback_without_key(self):
        _conf(key="")
        settings.update({"ai_provider": "deepseek"})
        error = ai.urllib.error.HTTPError("http://x", 401, "err", {}, None)
        with mock.patch.object(ai.urllib.request, "urlopen", side_effect=error):
            result = ai.models(force=True, provider="deepseek")
        self.assertTrue(result.get("fallback"))
        self.assertEqual([m["id"] for m in result["models"]], ["deepseek-chat", "deepseek-reasoner"])


if __name__ == "__main__":
    unittest.main()

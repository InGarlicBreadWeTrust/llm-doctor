"""Unit tests for llm_doctor.py - mock HTTP, no real endpoint.

Run:
    python3 -m unittest -v tests/test_llm_doctor.py
"""
import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
     "llm_doctor",
     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "llm_doctor.py"))
ld = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ld)


class TestNormEndpoint(unittest.TestCase):
    def test_bare_hostport_gets_http(self):
        self.assertEqual(ld.norm_endpoint("192.0.2.1:11434"),
                          "http://192.0.2.1:11434")

    def test_already_http_untouched(self):
        self.assertEqual(ld.norm_endpoint("http://localhost:11434"),
                          "http://localhost:11434")

    def test_https_preserved(self):
        self.assertEqual(ld.norm_endpoint("https://example.com"),
                          "https://example.com")


class TestProbeKind(unittest.TestCase):
    def test_v1_suffix_forces_openai(self):
        kind, _ = ld.probe_kind("http://localhost:1234/v1")
        self.assertEqual(kind, "chat_completions")

    def test_api_suffix_forces_ollama(self):
        kind, _ = ld.probe_kind("http://localhost:11434/api")
        self.assertEqual(kind, "ollama_native")

    def test_unreachable_defaults_to_ollama(self):
         # RFC 5737 reserved IP; will not connect, falls through to default
        kind, _ = ld.probe_kind("http://192.0.2.1:11434")
        self.assertEqual(kind, "ollama_native")


class TestUrlBuilders(unittest.TestCase):
    def test_ollama_native_chat(self):
        self.assertEqual(ld.chat_url("http://h:11434", "ollama_native"),
                          "http://h:11434/api/chat")

    def test_openai_chat(self):
        self.assertEqual(ld.chat_url("http://h:11434/v1", "chat_completions"),
                          "http://h:11434/v1/chat/completions")

    def test_list_url_ollama(self):
        self.assertEqual(ld.list_url("http://h:11434", "ollama_native"),
                          "http://h:11434/api/tags")

    def test_list_url_openai(self):
        self.assertEqual(ld.list_url("http://h:11434/v1", "chat_completions"),
                           "http://h:11434/v1/models")

    def test_no_double_v1_when_base_has_v1(self):
        # regression: a user passing --endpoint http://h/v1 must not get /v1/v1
        self.assertEqual(ld.chat_url("http://h:8000/v1", "chat_completions"),
                           "http://h:8000/v1/chat/completions")
        self.assertEqual(ld.list_url("http://h:8000/v1", "chat_completions"),
                           "http://h:8000/v1/models")

    def test_no_double_api_for_ollama(self):
        self.assertEqual(ld.chat_url("http://h:11434/api", "ollama_native"),
                           "http://h:11434/api/chat")
        self.assertEqual(ld.list_url("http://h:11434/api", "ollama_native"),
                           "http://h:11434/api/tags")


class TestMakePrompt(unittest.TestCase):
    def test_zero_is_hi(self):
        self.assertEqual(ld.make_prompt(0), "hi")
        self.assertEqual(ld.make_prompt(-1), "hi")

    def test_1_token_ends_with_pong(self):
        p = ld.make_prompt(10)
        self.assertTrue(p.endswith("pong"))
        self.assertGreater(len(p), 10)

    def test_no_secret_words(self):
        p = ld.make_prompt(200)
        self.assertNotIn("secret", p.lower())
        self.assertNotIn("password", p.lower())
        self.assertNotIn("token", p.lower())


class TestCheckPing(unittest.TestCase):
    def test_unreachable_host_returns_false(self):
         # RFC 5737 reserved IP - guaranteed unreachable
        result = ld.check_ping("192.0.2.1", 11434, 3)
        self.assertFalse(result)


class TestContextScalingTooFew(unittest.TestCase):
    def test_no_fit_message_when_all_fail(self):
        orig = ld._stream_ttfb
        ld._stream_ttfb = lambda *a, **k: (0.0, 0.0, False)
        try:
            ld.check_context_scaling("http://h", "ollama_native",
                                     "model", base_ctx=8, mults=(1, 2, 4, 8))
        finally:
            ld._stream_ttfb = orig


if __name__ == "__main__":
    unittest.main()

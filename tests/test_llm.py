"""Tests for the provider-agnostic LLM wrapper (P0-4)."""

import os
import pytest


class TestLlmCall:
    def test_returns_text_and_token_stats(self, mock_openai_response):
        from app.llm import llm
        text, tokens = llm("test prompt", model="gpt-4o-mini")
        assert isinstance(text, str)
        assert len(text) > 0
        assert "prompt_tokens" in tokens
        assert "completion_tokens" in tokens
        assert "total_tokens" in tokens

    def test_passes_model_and_messages_to_completion(self, mocker):
        import app.llm
        mock = mocker.patch("app.llm.completion")
        app.llm.llm("test", model="gpt-4o")
        mock.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
        )

    def test_raises_on_api_error(self, mocker):
        import app.llm
        mocker.patch("app.llm.completion", side_effect=RuntimeError("API error"))
        with pytest.raises(RuntimeError, match="API error"):
            app.llm.llm("test")

    def test_model_from_env_when_not_passed(self, mocker, monkeypatch):
        import app.llm
        monkeypatch.setenv("LLM_MODEL", "ollama/llama3")
        mock = mocker.patch("app.llm.completion")
        app.llm.llm("ping")
        mock.assert_called_once()
        assert mock.call_args.kwargs["model"] == "ollama/llama3"


class TestProviderSwap:
    def test_swap_model_via_env_only(self, mocker, monkeypatch):
        """Changing LLM_MODEL must not require touching call sites."""
        import app.llm
        for model in ("anthropic/claude-sonnet-4-6", "ollama/llama3"):
            monkeypatch.setenv("LLM_MODEL", model)
            mock = mocker.patch("app.llm.completion")
            app.llm.llm("hello")
            assert mock.call_args.kwargs["model"] == model

    def test_ollama_no_api_key_needed(self, monkeypatch, mocker):
        """Ollama path requires no provider key: no client is constructed."""
        import app.llm
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("LLM_MODEL", "ollama/llama3")
        mock = mocker.patch("app.llm.completion")
        app.llm.llm("ping")
        assert mock.call_args.kwargs["model"] == "ollama/llama3"


class TestCost:
    def test_known_model_priced(self):
        from app.llm import calculate_cost
        assert calculate_cost("gpt-4o-mini", {"prompt_tokens": 1_000_000, "completion_tokens": 0}) == pytest.approx(0.15)

    def test_unknown_provider_model_counts_zero(self):
        from app.llm import calculate_cost
        assert calculate_cost("ollama/llama3", {"prompt_tokens": 100}) == 0.0


class TestEnvExample:
    def test_env_example_documents_llm_model(self):
        from pathlib import Path
        content = Path(".env.example").read_text()
        assert "LLM_MODEL" in content
        assert "OLLAMA" in content.upper() or "ollama" in content

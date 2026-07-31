"""Provider-agnostic LLM wrapper (litellm).

Model is chosen via the LLM_MODEL env var (e.g. gpt-4o-mini, anthropic/claude-sonnet-4-6,
ollama/llama3). Provider keys come from their standard env vars (OPENAI_API_KEY,
ANTHROPIC_API_KEY, ...). Swap providers by changing LLM_MODEL only.
"""

import os

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
}

DEFAULT_MODEL = "gpt-4o-mini"


def _resolve_model(model: str | None) -> str:
    return model or os.getenv("LLM_MODEL", DEFAULT_MODEL)


def llm(prompt: str, model: str | None = None) -> tuple[str, dict]:
    """Call the configured LLM. Returns (output_text, token_usage_dict)."""
    response = completion(
        model=_resolve_model(model),
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    tokens = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return text, tokens


def llm_stream(prompt: str, model: str | None = None):
    response = completion(
        model=_resolve_model(model),
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def calculate_cost(model: str, tokens: dict) -> float:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # provider-agnostic (claude/ollama/etc.): pricing not tracked, count as 0
        return 0.0
    return pricing["input"] * tokens.get("prompt_tokens", 0) + pricing["output"] * tokens.get("completion_tokens", 0)

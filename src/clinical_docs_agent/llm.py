"""Provider-agnostic LLM client: messages in, a validated Pydantic object out.

Pick the backend with environment variables (no code changes):
    LLM_PROVIDER=ollama     (default; free, local; needs `ollama serve`)
    LLM_PROVIDER=anthropic  (needs ANTHROPIC_API_KEY and `pip install -e '.[anthropic]'`)
    LLM_MODEL=<model id>    (optional; overrides the provider default)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {
    "ollama": "qwen2.5:3b",
    "anthropic": "claude-opus-5-5",
}


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, system: str, user: str, schema: type[T]) -> T: ...


class OllamaClient:
    """Local models through Ollama's native /api/chat, using its JSON-schema `format` for structured output."""

    provider = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, num_ctx: int = 16384, timeout: float = 300):
        self.model = model or DEFAULT_MODELS["ollama"]
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx = num_ctx
        self.timeout = timeout

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema.model_json_schema(),
            "stream": False,
            # num_ctx: Ollama's default context is small and silently truncates long prompts.
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            raise LLMError(f"Ollama returned {e.code}: {e.read().decode(errors='replace')[:300]}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Can't reach Ollama at {self.host}. Is `ollama serve` running? ({e.reason})") from e
        return schema.model_validate_json(data["message"]["content"])


class AnthropicClient:
    """Claude through the official SDK; `messages.parse` validates the reply against the Pydantic schema."""

    provider = "anthropic"

    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("Install the Anthropic SDK: pip install -e '.[anthropic]'") from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or DEFAULT_MODELS["anthropic"]

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except self._anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise LLMError("Network error reaching the Anthropic API") from e
        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer")
        if response.parsed_output is None:
            raise LLMError(f"No structured output (stop_reason={response.stop_reason})")
        return response.parsed_output


def get_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    provider = (provider or os.environ.get("LLM_PROVIDER") or "ollama").lower()
    model = model or os.environ.get("LLM_MODEL") or None
    if provider == "ollama":
        return OllamaClient(model=model)
    if provider == "anthropic":
        return AnthropicClient(model=model)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use 'ollama' or 'anthropic'.")

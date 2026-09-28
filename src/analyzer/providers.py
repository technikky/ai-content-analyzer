"""LLM providers.

``ScriptedLLM`` is not a testing afterthought -- it is what makes the LLM analyzers
testable at all. Every LLM code path in this package (prompt construction, response
parsing, repair, schema validation, failure handling) is exercised in CI against scripted
responses, including deliberately malformed ones. Without it those paths would be
unexercised until a user hit them in production with an API key.
"""

from __future__ import annotations

import os
import time
from typing import Any, Protocol, runtime_checkable

DEFAULT_TIMEOUT_S = 60.0


@runtime_checkable
class LLM(Protocol):
    """A text-completion model."""

    @property
    def name(self) -> str: ...

    def complete(self, prompt: str, *, max_tokens: int = 800) -> str:
        """Return the completion. Raises on transport failure rather than returning text.

        Returning an error string would let it be parsed as a model answer, which is the
        worst available outcome: a pipeline would score a transport failure as a bad answer.
        """
        ...


class ProviderError(RuntimeError):
    """Raised for provider configuration and transport failures."""


class ScriptedLLM:
    """Replays fixed responses, for tests and for offline demonstration."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        default: str | None = None,
        name: str = "scripted",
        fail_first_n: int = 0,
    ) -> None:
        self._responses = list(responses or [])
        self._default = default
        self._name = name
        self._fail_first_n = fail_first_n
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(self, prompt: str, *, max_tokens: int = 800) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) <= self._fail_first_n:
            raise ProviderError(f"scripted transient failure {len(self.prompts)}")
        if self._responses:
            return self._responses.pop(0)
        if self._default is not None:
            return self._default
        raise ProviderError("ScriptedLLM has no response left to return")


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProviderError(
            f"{name} is not set. Copy .env.example to .env and fill it in, or export "
            f"{name} in your shell."
        )
    return value


def _require_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ProviderError(
            "httpx is required for hosted providers. Install with: "
            "pip install 'ai-content-analyzer[providers]'"
        ) from exc
    return httpx


class OpenAIChat:
    """Chat-completions client for the OpenAI API or a compatible endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.0,
        base_url: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._model = model
        # Zero by default: an analysis pipeline that returns different entities on a
        # re-run cannot be regression-tested.
        self._temperature = temperature
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return self._model

    def complete(self, prompt: str, *, max_tokens: int = 800) -> str:  # pragma: no cover
        httpx = _require_httpx()
        api_key = _require_env("OPENAI_API_KEY")
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=self._timeout_s,
        )
        if response.status_code >= 400:
            raise ProviderError(
                f"OpenAI API returned {response.status_code}: {response.text[:300]}"
            )
        try:
            return str(response.json()["choices"][0]["message"]["content"])
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"unexpected OpenAI response: {response.text[:300]}") from exc


class AnthropicMessages:
    """Messages-API client for Anthropic models."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        temperature: float = 0.0,
        base_url: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        api_version: str = "2023-06-01",
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._base_url = (
            base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        ).rstrip("/")
        self._timeout_s = timeout_s
        self._api_version = api_version

    @property
    def name(self) -> str:
        return self._model

    def complete(self, prompt: str, *, max_tokens: int = 800) -> str:  # pragma: no cover
        httpx = _require_httpx()
        api_key = _require_env("ANTHROPIC_API_KEY")
        response = httpx.post(
            f"{self._base_url}/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": self._api_version},
            json={
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": self._temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self._timeout_s,
        )
        if response.status_code >= 400:
            raise ProviderError(
                f"Anthropic API returned {response.status_code}: {response.text[:300]}"
            )
        try:
            blocks = response.json()["content"]
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"unexpected Anthropic response: {response.text[:300]}") from exc


class RetryingLLM:
    """Wraps a provider with bounded retries.

    Retries transport failures only. A response that parsed but violated the schema is not
    retried here: that is a prompt problem, and retrying it silently would hide how often
    it happens.
    """

    def __init__(self, inner: LLM, *, max_retries: int = 2, backoff_s: float = 0.5) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        self._inner = inner
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self.attempts = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def complete(self, prompt: str, *, max_tokens: int = 800) -> str:
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self.attempts += 1
            try:
                return self._inner.complete(prompt, max_tokens=max_tokens)
            except Exception as exc:
                last = exc
                if attempt < self._max_retries and self._backoff_s > 0:
                    time.sleep(self._backoff_s * (2**attempt))
        raise ProviderError(f"provider failed after {self._max_retries + 1} attempts: {last}")


PROVIDERS = ("openai", "anthropic", "scripted")


def build_llm(name: str, model: str | None = None, **kwargs: Any) -> LLM:
    key = name.strip().lower()
    if key == "openai":
        return OpenAIChat(model or "gpt-4o-mini", **kwargs)
    if key == "anthropic":
        return AnthropicMessages(model or "claude-sonnet-5", **kwargs)
    if key == "scripted":
        return ScriptedLLM(**kwargs)
    raise ProviderError(f"unknown provider {name!r}; expected one of {', '.join(PROVIDERS)}")

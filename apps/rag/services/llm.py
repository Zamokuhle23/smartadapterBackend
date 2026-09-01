"""
LLM provider adapter.

- OpenAIChatProvider: any OpenAI-compatible /chat/completions endpoint
  (OpenAI, Azure, Groq, Ollama with OpenAI shim, etc.) via LLM_BASE_URL.
- OfflineTutorProvider: extractive fallback used in development when no API key
  is configured. It composes an answer from the retrieved syllabus context so
  the full chat flow is testable end-to-end with zero external calls.
"""

import json
from urllib import request as http_request

from django.conf import settings


class BaseChatProvider:
    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError


class OpenAIChatProvider(BaseChatProvider):
    def chat(self, messages: list[dict], model: str | None = None,
             max_tokens: int | None = None) -> str:
        payload = {"model": model or settings.LLM_MODEL, "messages": messages}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
        }
        # OpenRouter-recommended attribution headers (harmless elsewhere)
        if settings.LLM_APP_URL:
            headers["HTTP-Referer"] = settings.LLM_APP_URL
        if settings.LLM_APP_TITLE:
            headers["X-Title"] = settings.LLM_APP_TITLE
        req = http_request.Request(
            f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with http_request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def stream(self, messages: list[dict], model: str | None = None):
        """Stream assistant turns token-by-token (SSE) as a generator of str deltas.

        Enables the "talk while thinking" voice experience instead of waiting for a
        complete answer. Falls back to a single `chat()` chunk if the endpoint does
        not support streaming.
        """
        payload = {
            "model": model or settings.LLM_MODEL,
            "messages": messages,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
        }
        if settings.LLM_APP_URL:
            headers["HTTP-Referer"] = settings.LLM_APP_URL
        if settings.LLM_APP_TITLE:
            headers["X-Title"] = settings.LLM_APP_TITLE
        req = http_request.Request(
            f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with http_request.urlopen(req, timeout=180) as resp:
                buffer = ""
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        buffer = line[len("data:"):].strip()
                    else:
                        buffer = line
                    if buffer == "[DONE]":
                        return
                    if not buffer.startswith("{"):
                        continue
                    try:
                        delta = json.loads(buffer)["choices"][0].get("delta", {}).get("content", "")
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
                    if delta:
                        yield delta
        except Exception:  # noqa: BLE001 - fall back to one-shot on any streaming failure
            yield self.chat(messages, model=model)


class AzureOpenAIChatProvider(BaseChatProvider):
    """
    Azure OpenAI chat provider.

    Azure uses `api-key` auth (not Bearer), a deployment URL of the form
        https://<resource>.openai.azure.com/openai/deployments/<deployment>/
    and requires the `api-version` query param. Set in env:
        LLM_PROVIDER=azure
        AZURE_OPENAI_API_KEY=...
        LLM_BASE_URL=https://<resource>.openai.azure.com
        LLM_MODEL=<deployment-name>            # e.g. gpt-4o-mini (deployment name)
    """

    def _chat_url(self) -> str:
        base = settings.LLM_BASE_URL.rstrip("/")
        version = getattr(settings, "AZURE_OPENAI_API_VERSION", "2024-06-01")
        return (f"{base}/openai/deployments/{settings.LLM_MODEL}/chat/completions"
                f"?api-version={version}")

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "api-key": settings.LLM_API_KEY,
        }

    def _request(self, url: str, payload: dict) -> http_request.Request:
        return http_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

    def chat(self, messages: list[dict], model: str | None = None,
             max_tokens: int | None = None) -> str:
        del model  # Azure routes on the deployment name in the URL
        payload: dict = {"messages": messages}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        with http_request.urlopen(self._request(self._chat_url(), payload), timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def stream(self, messages: list[dict], model: str | None = None):
        """Stream completion deltas (SSE). Falls back to one-shot on failure."""
        del model
        payload: dict = {"messages": messages, "stream": True}
        try:
            req = self._request(self._chat_url(), payload)
            with http_request.urlopen(req, timeout=180) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    buffer = line[5:].strip() if line.startswith("data:") else line.strip()
                    if buffer == "[DONE]":
                        return
                    if not buffer.startswith("{"):
                        continue
                    try:
                        delta = json.loads(buffer)["choices"][0].get("delta", {}).get("content", "")
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
                    if delta:
                        yield delta
        except Exception:  # noqa: BLE001 - fall back to one-shot on any streaming failure
            yield self.chat(messages)


class OfflineTutorProvider(BaseChatProvider):
    """
    Deterministic study-notes style answer built from the retrieved chunks.
    Clearly labelled so nobody mistakes it for a real LLM in production.
    """

    def chat(self, messages: list[dict], model: str | None = None,
             max_tokens: int | None = None) -> str:
        del model, max_tokens  # unused; offline provider ignores model/token choices
        context = ""
        for msg in reversed(messages):
            if msg["role"] == "system":
                context = msg["content"]
                break
        question = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                question = msg["content"]
                break
        return (
            "[offline mode - set OPENROUTER_API_KEY for full AI tutoring]\n\n"
            f"Regarding your question: \"{question[:300]}\"\n\n"
            "Here is what your syllabus says that is relevant:\n\n"
            f"{context[:2000]}"
        )


def get_chat_provider() -> BaseChatProvider:
    if not settings.LLM_API_KEY:
        return OfflineTutorProvider()
    if settings.LLM_PROVIDER == "azure":
        return AzureOpenAIChatProvider()
    return OpenAIChatProvider()

"""Bulk tagging LLM: Muse Spark (free) via OpenCode Zen, with fallback.

Primary: opencode.ai Zen Responses API, model muse-spark-1.3-contributor-free
(FREE). Contributor tier trains on prompts, so this channel carries ONLY
public exam/corpus text - never student data or PII.
Fallback: any free OpenAI-compatible chat model (default minimax-m3:free
on OpenRouter), same prompt, JSON out.

Zen is IP-restricted in some networks (Cloudflare 1010); the fallback keeps
local dev working where Zen is unreachable.
"""

import json
from urllib import request as http_request

from django.conf import settings


class TaggingError(Exception):
    pass


def _zen_base() -> str:
    return getattr(settings, "OPENCODE_BASE_URL",
                   "https://opencode.ai/zen/v1").rstrip("/")


def _zen_model() -> str:
    return (getattr(settings, "TAGGING_ZEN_MODEL", "") or
            "muse-spark-1.3-contributor-free")


def _fallback_model() -> str:
    return (getattr(settings, "TAGGING_MODEL", "") or
            "minimax/minimax-m3:free")


def _zen_chat(prompt_text: str) -> str:
    """One Responses-API call. Raises TaggingError on any failure."""
    key = getattr(settings, "OPENCODE_API_KEY", "")
    if not key:
        raise TaggingError("OPENCODE_API_KEY not configured")
    payload = {"model": _zen_model(), "input": prompt_text}
    req = http_request.Request(
        f"{_zen_base()}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with http_request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise TaggingError(f"Zen call failed: {exc}") from exc
    texts = []
    for item in data.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    text = "\n".join(texts).strip()
    if not text:
        raise TaggingError("Zen returned no text")
    return text


def _fallback_chat(prompt_text: str) -> str:
    """OpenRouter chat/completions with a free model. Raises TaggingError."""
    from apps.rag.services.llm import get_chat_provider

    try:
        return get_chat_provider().chat(
            [{"role": "user", "content": prompt_text}],
            model=_fallback_model(),
        )
    except Exception as exc:
        raise TaggingError(f"Fallback tagging failed: {exc}") from exc


def tagging_chat(prompt_text: str, attempts: int = 4) -> str:
    """Tagging completion, trying free providers in order.

    Retries with backoff: free tiers rate-limit bulk runs, and a dropped
    page would stay untagged (tag_pages only retries untagged pages on a
    fresh run, so surviving transient 429s here matters).
    """
    import time

    last: TaggingError | None = None
    for attempt in range(max(1, attempts)):
        try:
            return _zen_chat(prompt_text)
        except TaggingError as exc:
            last = exc
            try:
                return _fallback_chat(prompt_text)
            except TaggingError as exc2:
                last = exc2
        if attempt < attempts - 1:
            time.sleep(5 * (attempt + 1))
    raise last if last else TaggingError("tagging failed")

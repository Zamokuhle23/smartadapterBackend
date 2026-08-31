"""Verify the configured LLM provider works: sends one tiny chat request.

Usage:
    python scripts/check_llm.py
Set OPENROUTER_API_KEY in backend/.env first.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.conf import settings  # noqa: E402

from apps.rag.services.llm import get_chat_provider  # noqa: E402


def main():
    print(f"Base URL : {settings.LLM_BASE_URL}")
    print(f"Model    : {settings.LLM_MODEL}")
    if not settings.LLM_API_KEY:
        print("API key  : NOT SET - paste your key into backend/.env (OPENROUTER_API_KEY=...)")
        sys.exit(1)
    masked = settings.LLM_API_KEY[:8] + "..." + settings.LLM_API_KEY[-4:]
    print(f"API key  : {masked}")
    provider = get_chat_provider()
    print(f"Provider : {type(provider).__name__}")
    reply = provider.chat(
        [
            {"role": "system", "content": "You are FundzaAI, an EGCSE tutor. Be brief."},
            {"role": "user", "content": "Reply with exactly: FundzaAI connection OK"},
        ]
    )
    print(f"Reply    : {reply.strip()[:200]}")
    print("\nLLM CONNECTION OK")


if __name__ == "__main__":
    main()

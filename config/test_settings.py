"""Test settings: force an in-memory SQLite database for hermetic tests,
regardless of DATABASE_URL in .env."""
import os
from config.settings import *  # noqa: F401,F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
# No external LLM / network in tests.
get_chat_provider = None  # placeholder - not used; keep default behavior
CELERY_TASK_ALWAYS_EAGER = True
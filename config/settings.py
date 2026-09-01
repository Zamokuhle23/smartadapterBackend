"""
Django settings for the FundzaAI backend.

Environment-driven configuration:
- DATABASE_URL: postgres://... (recommended, enables pgvector) or unset -> SQLite for quick dev
- REDIS_URL: redis://... (Celery broker + Channels layer) or unset -> in-memory fallbacks
- OPENROUTER_API_KEY (or OPENAI_API_KEY) / LLM_BASE_URL / LLM_MODEL:
  real tutor responses via any OpenAI-compatible endpoint; unset -> offline extractive mock
- EMBEDDING_PROVIDER: "hash" (offline default) or "openai" (needs EMBEDDINGS_* settings)
"""

import os
from datetime import timedelta
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"

# Never leave a known key or open hosts in production.
if DEBUG:
    SECRET_KEY = SECRET_KEY or "dev-insecure-key-change-me"
else:
    if not SECRET_KEY:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG != 1")
    if not os.getenv("DJANGO_ALLOWED_HOSTS"):
        raise ImproperlyConfigured("DJANGO_ALLOWED_HOSTS must be set when DJANGO_DEBUG != 1")

ALLOWED_HOSTS = [h.strip() for h in os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",")]
if not DEBUG and "*" in ALLOWED_HOSTS:
    ALLOWED_HOSTS = []  # no wildcard hosts in production; set the real domain

# Secure cookie / HTTPS guardrails - opt in via DJANGO_FORCE_HTTPS=1 (TLS terminated at the edge).
if os.getenv("DJANGO_FORCE_HTTPS", "0") == "1":
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_SSL_REDIRECT = True

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third party
    "rest_framework",
    "channels",
    # FundzaAI apps
    "apps.accounts",
    "apps.syllabus",
    "apps.rag",
    "apps.tutoring",
    "apps.progress",
    "apps.quiz",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# ---------------------------------------------------------------------------
# Database - PostgreSQL (+pgvector) when DATABASE_URL is set; SQLite for dev.
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres"):
    try:
        import dj_database_url

        DATABASES = {"default": dj_database_url.parse(DATABASE_URL, conn_max_age=600)}
    except ImportError:
        # Minimal manual parse: postgres://user:pass@host:port/dbname
        from urllib.parse import urlsplit

        parts = urlsplit(DATABASE_URL)
        DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": parts.path.lstrip("/"),
                "USER": parts.username or "",
                "PASSWORD": parts.password or "",
                "HOST": parts.hostname or "localhost",
                "PORT": str(parts.port or 5432),
            }
        }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_USER_MODEL = "accounts.User"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Africa/Mbabane"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# DRF / JWT
# ---------------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": ("rest_framework.permissions.IsAuthenticated",),
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 50,
    # Global throttling baseline - tailored per-view via ScopedRateThrottle below.
    "DEFAULT_THROTTLE_CLASSES": (
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ),
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/hour",
        "user": "1000/hour",
        # "auth": login/register attempts (brute-force guard)
        "auth": "20/hour",
        # "llm": expensive LLM-backed endpoints (quiz gen / grading / exam)
        "llm": "60/hour",
    },
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(hours=12),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=14),
}

# ---------------------------------------------------------------------------
# Channels - Redis channel layer when available, in-memory fallback for dev.
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "")
if REDIS_URL:
    CHANNEL_LAYERS = {
        "default": {
            "BACKEND": "channels_redis.core.RedisChannelLayer",
            "CONFIG": {"hosts": [REDIS_URL]},
        }
    }
else:
    CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}

# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------
CELERY_BROKER_URL = REDIS_URL or "memory://"
CELERY_RESULT_BACKEND = REDIS_URL or None
CELERY_TASK_ALWAYS_EAGER = os.getenv("CELERY_TASK_ALWAYS_EAGER", "0") == "1"

# ---------------------------------------------------------------------------
# RAG / LLM providers (OpenRouter-compatible by default; Azure OpenAI supported)
# ---------------------------------------------------------------------------
# Provider: "openai" (OpenAI / OpenRouter / any OpenAI-compatible chat endpoint)
#           "azure" (Azure OpenAI - uses api-key header + deployment URL)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
if LLM_PROVIDER in ("azure", "azure_ai", "azure_sdk"):
    LLM_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
else:
    LLM_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "stealth/ox-alpha")

# Separate provider/model for high-volume tutor CHAT (cheap by default) vs.
# the primary provider used for question generation + grading (quality-critical).
# If CHAT_LLM_PROVIDER is unset it falls back to LLM_PROVIDER (single provider).
CHAT_LLM_PROVIDER = os.getenv("CHAT_LLM_PROVIDER", "").lower() or LLM_PROVIDER
CHAT_LLM_MODEL = os.getenv("CHAT_LLM_MODEL", "") or LLM_MODEL
if CHAT_LLM_PROVIDER in ("azure", "azure_ai", "azure_sdk"):
    CHAT_LLM_API_KEY = os.getenv("CHAT_AZURE_OPENAI_API_KEY", "") or os.getenv("AZURE_OPENAI_API_KEY", "")
else:
    CHAT_LLM_API_KEY = os.getenv("CHAT_OPENAI_API_KEY", "") or LLM_API_KEY

# Azure OpenAI only - the deployed model name and API version.
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
# Recommended OpenRouter attribution headers (no-op for Azure)
LLM_APP_URL = os.getenv("LLM_APP_URL", "https://fundza.ai")
LLM_APP_TITLE = os.getenv("LLM_APP_TITLE", "FundzaAI")
# Embeddings: "hash" = offline hasher (fallback); "local" = in-process
# sentence-transformers model; "openai" = OpenAI-compatible /embeddings endpoint.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "hash")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "")   # E5 models need "query: "
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "")  # E5: "passage: "
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "https://api.openai.com/v1")
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))


# FundzaAI — Deployment Checklist

Pre-production hardening, in order:

## 1. Environment
- [ ] `DJANGO_SECRET_KEY` set to a long random value (never commit it)
- [ ] `DJANGO_DEBUG=0`
- [ ] `DJANGO_ALLOWED_HOSTS=api.fundza.example` (your domain)
- [ ] `DATABASE_URL=postgres://...` (pgvector image: `pgvector/pgvector:pg16`)
- [ ] `REDIS_URL=redis://...` (real broker for Celery + Channels)
- [ ] `OPENROUTER_API_KEY` set on the server only

## 2. Database
- [ ] `python manage.py migrate`
- [ ] `python manage.py seed_syllabi`
- [ ] Enable pgvector extension: `CREATE EXTENSION vector;`
- [ ] Switch retriever to VectorField KNN (see apps/rag/services/retriever.py TODO) once on Postgres

## 3. Serving
- [ ] ASGI server: `daphne -b 0.0.0.0 -p 8000 config.asgi:application` (needed for WebSockets)
      or uvicorn behind nginx
- [ ] TLS via nginx/Caddy → `https://` + `wss://` (Android blocks cleartext by default in release)
- [ ] Celery worker running: `celery -A config worker -l info`

## 4. Android release build
- [ ] Point release `API_BASE_URL` / `WS_BASE_URL` at the real domain (app/build.gradle.kts)
- [ ] Remove `android:usesCleartextTraffic="true"` from the manifest
- [ ] Move JWT storage to EncryptedSharedPreferences
- [ ] Signed AAB via Play Console

## 5. Operations
- [ ] Backups of Postgres (learner mastery data is irreplaceable)
- [ ] Log aggregation + Sentry (or similar) for the backend
- [ ] LLM spend alerts on OpenRouter
- [ ] Rotate/restrict the OpenRouter key per environment

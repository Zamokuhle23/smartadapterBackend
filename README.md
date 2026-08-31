# FundzaAI 🇸🇿📚

**"One AI, Many Syllabuses"** — a personalized AI study assistant for Eswatini students,
aligned to ECESWA syllabuses. Priority levels: **EGCSE** (Forms 4–5) and **JC** (Forms 1–3),
with AS/A-Levels as future expansion.

See [`PLAN.md`](./PLAN.md) for the full product & technical plan.

## How it works
1. **Dynamic syllabus system** — syllabus shells (JC/EGCSE with real ECESWA subject codes)
   are seeded via `seed_syllabi`; admins upload any document (PDF/TXT/DOCX) against a
   syllabus and it is automatically chunked → embedded → indexed for RAG. No retraining.
2. **Personalized tutor** — every chat reply is grounded in *that student's* syllabus via
   RAG and shaped by their learner profile (language: English/siSwati/mix; Socratic vs
   direct style; pace) and their weakest objectives from the mastery model.
3. **Learner model** — each attempt updates a Bayesian Knowledge Tracing (BKT) estimate
   per learning objective; the dashboard surfaces weak subjects/objectives and
   personalized next-step recommendations.
4. **Realtime chat** — Django Channels WebSocket at `ws/chat/<session_id>/`.

## Quickstart (local dev — no Docker needed)
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

copy .env.example .env                              # defaults work for SQLite dev
python manage.py migrate
python manage.py seed_syllabi                       # JC + EGCSE shells w/ ECESWA codes
python manage.py createsuperuser
python manage.py runserver
```
Admin: http://localhost:8000/admin/ · API root: http://localhost:8000/api/

> Dev fallbacks: without Redis, Celery runs in eager/in-memory mode and Channels uses an
> in-memory layer; without an LLM key, the tutor answers extractively from uploaded docs
> (`[offline mode]` label). Everything is exercisable end-to-end with zero external services.

## Full stack (Postgres + pgvector + Redis)
```bash
docker compose up --build
```

## API overview
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/auth/token/` | POST | JWT login |
| `/api/auth/register/` | POST | Student self-signup (returns JWT pair) |
| `/api/quiz/generate/` | POST | LLM-generate MCQs grounded in the subject's RAG corpus |
| `/api/quiz/next/?subject_id=` | GET | Adaptive next question (weakest objectives first) |
| `/api/quiz/answer/` | POST | Grade answer + update BKT mastery |

| `/api/me/profile/` | GET/PATCH | Learner profile (personalization prefs) |
| `/api/syllabi/?level=EGCSE` | GET | List syllabuses |
| `/api/subjects/?syllabus=` | GET | Subjects (ECESWA codes) |
| `/api/topics/?subject=` | GET | Topic tree + objectives |
| `/api/documents/` | POST (staff, multipart) | Upload syllabus doc → auto-ingest |
| `/api/documents/{id}/reingest/` | POST (staff) | Re-run ingestion |
| `/api/my-subjects/` | GET/POST/DELETE | Student's subject workspaces ("projects") |
| `/api/workspace/{subject_id}/` | GET | Subject bundle: mastery, recommendations, chats, latest session |
| `/api/chat-sessions/` | POST/GET | Create/list tutoring sessions |
| `ws/chat/<id>/?token=<jwt>` | WS | Real-time tutor chat (JWT via query string — mobile-friendly) |
| `/api/progress/attempt/` | POST | Record attempt → BKT mastery update |
| `/api/progress/dashboard/` | GET | Subject summary + recommendations |

## Project layout
```
config/        Django project (settings, urls, asgi+channels, celery)
apps/accounts  Custom user + LearnerProfile (personalization preferences)
apps/syllabus  Versioned syllabuses, subjects (ECESWA codes), topics, objectives,
               document upload + ingestion pipeline (Celery tasks), seed command
apps/rag       DocumentChunk store, embeddings (OpenAI-compatible / offline hasher),
               retriever (pgvector-ready), LLM adapter (OpenAI-compatible / offline)
apps/tutoring  ChatSession/Message models, WS consumer, personalized orchestrator
apps/progress  MasteryEvent/MasteryRecord, BKT engine, dashboard analytics
```

## Roadmap
- Phase 1 MVP ✅ (this scaffold): ingestion, RAG tutor chat, BKT tracking, dashboards
- Phase 2: prerequisite weakness tracing, FSRS spaced review, diagnostic quiz generator
- Phase 3: Android app (Kotlin + Compose, Room offline-first, Retrofit), teacher views
- Phase 4: mock exam simulator with ECESWA-style grading

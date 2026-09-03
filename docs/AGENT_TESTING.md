# SmartAdapter (FundzaAI) — Automated Agent Testing Guide

This document is a **behavior + API spec** intended for automated agents (or other AI
testers) to exercise the app end-to-end **without manual testing**. It describes the
system's capabilities, the exact REST/WebSocket contracts, expected behaviors, and
edge cases, so a driving agent can script realistic flows and assert correct
behaviour.

> Stack: Django + DRF + Django Channels. PostgreSQL + **pgvector** (production / VM),
> SQLite for local dev. Android (Kotlin/Compose) client. LLM = Azure AI Foundry
> (Phi-4-mini / Phi-4-reasoning) or OpenRouter; offline `OfflineTutorProvider` fallback
> when no key is set.

---

## 1. What the system does (capability overview)

A tutor app for **Eswatini JC + EGCSE** students. Capabilities a tester can drive:

1. **Auth** — register + JWT login/refresh.
2. **Syllabus browsing** — levels → syllabi → subjects → topics/objectives.
3. **Workspace ("my subjects")** — enroll in subjects, per-subject mastery + recommendations.
4. **Practice** — adaptive questions grounded in the syllabus + real ECESWA/IGCSE past
   papers; generate, answer (MCQ + structured), auto-graded.
5. **Exam simulation** — sit a full paper (blueprint of topic weightings), lazily
   generated questions, running score.
6. **Tutoring chat (text + voice)** — WebSocket, RAG-grounded, personalization from
   `LearnerProfile`.
7. **Student-fact memory** — durable facts auto-extracted and injected when relevant.
8. **Subtopic chat threading** — messages routed/grouped into syllabus subtopics;
   main chat + ordered thread list; thread-scoped history continuity.
9. **Smart Practice/Exam buttons** — chat replies asking for practice/exam render
   actionable buttons in the app (and expose the data needed).

---

## 2. Conventions for a driving agent

- **Base URL**: `API_BASE_URL` (e.g. `http://4.222.216.8/api/`). Append trailing slash.
- **Auth**: `Authorization: Bearer <access_token>` for all requests except auth endpoints.
- **REST**: DRF JSON. Pagination envelope: `{"count": N, "results": [...]}`.
- Missing token → `401 {"detail":"Authentication credentials were not provided."}`.
- **Enrollment gating**: generation/answering/exam-start/workspace restricted to
  subjects the student is **enrolled in**. Unenrolled → `403`.
- **LLM endpoints** are rate-limited (scope `llm`) and may take **15–60s**. Use long timeouts.

---

## 3. Auth flow

### POST `/api/auth/register/` (public)
Body:
```json
{"username": "tester123", "password": "StrongPass99", "level": "EGCSE"}
```
→ `200` `{"access": "...", "refresh": "...", "username": "tester123"}` and creates a
`LearnerProfile`.
- Invalid/missing password (<8 chars) → `400`.
- Duplicate username → `400`.
- **Assert profile auto-created**: later `GET /api/me/profile/` succeeds.

### POST `/api/auth/token/`
`{"username": "...", "password": "..."}` → `200` tokens. Wrong password → `401`.

### POST `/api/auth/token/refresh/`
`{"refresh": "..."}` → `200` `{"access": "..."}`.

### GET / PATCH `/api/me/profile/`
Returns/updates learner profile. Field validation.

---

## 4. Syllabus

| Endpoint | Notes |
|---|---|
| `GET /api/syllabi/?level=EGCSE` | List published syllabi |
| `GET /api/subjects/?syllabus=<id>` | Subjects for a syllabus |
| `GET /api/topics/?subject=<id>` | Topic tree (strands + subtopics + objectives) |

Assert: `subjects/count > 0`; topics include `objectives`.

---

## 5. Workspace / enrollment / mastery

### POST `/api/my-subjects/`
`{"subject_id": <id>}` (+ `"tier"` if subject.tiers_available) → `201`.
Tiered subject without tier → `400`; unknown → `400`; invalid tier → `400`.

### GET `/api/my-subjects/` → enrolled subjects.

### GET `/api/workspace/<subject_id>/`
→ subject, `avg_mastery`, `objectives_tracked`, `recommendations`,
`latest_session_id`, `sessions`.
Unknown subject → `400`; not enrolled → `403`; recommendations weakest-first.

### POST `/api/progress/attempt/`
`{"objective_id": <id>, "correct": true}` → `200` mastery. Non-enrolled subject → `403`.
---

## 6. Practice

### GET `/api/quiz/next/?subject_id=<id>`
→ next adaptive `QuestionDto`, or `404` when bank empty. Must be enrolled (else `403`).

`QuestionDto` (never leaks answer):
```json
{
  "id": 1, "question_text": "...", "options": ["..."],
  "difficulty": 2, "topic_title": "...", "format": "mcq|structured",
  "marks": 1, "paper_label": "...", "source_year": null,
  "source": "igcse|egcse", "adapted_from_past_paper": false,
  "figure_urls": []
}
```
**Assert invariants**: no `correct_index`, `explanation`, or `marking_guidance`.

### POST `/api/quiz/generate/`
`{"subject_id": <id>, "count": 1}` → list of questions. `503` if no LLM.

### POST `/api/quiz/answer/`
MCQ `{"question_id": <id>, "selected_index": 0}` → `AnswerQuizResponse`
`{correct, correct_index?, explanation?, mastery?, awarded_marks?, max_marks?, feedback?}`.
Structured `{"question_id": <id>, "answer_text": "..."}`.
- Not enrolled → `403`; out-of-range `selected_index` → `400`; wrong-tier → `403`.

---

## 7. Exam simulation

- `POST /api/quiz/exam/start/` `{"subject_id": <id>, "paper": 1}` → `201` `ExamStateDto`
  (`id`, `title`, `total_questions`, `status`, `sections`).
- `POST /api/quiz/exam/<id>/next/` → generated `QuestionDto` or `204` when done.
- `GET /api/quiz/exam/<id>/` → state with answered + running score.
- Answering uses `/api/quiz/answer/`.

Assert: `paper` clamped 1..4; not enrolled → `403`.

---

## 8. Tutoring chat (WebSocket)

### REST
- `POST /api/chat-sessions/` `{"syllabus": <id>, "subject": <id?>}` → session.
- `GET /api/chat-sessions/` → list.
- `GET /api/chat-sessions/<id>/` → session + messages. `?topic=main` or `?topic=<id>`
  returns only that thread's messages.
- `GET /api/chat-sessions/<id>/threads/` → ordered `[main, sub1, sub2, ...]`.

### WebSocket
`ws://HOST/ws/chat/<session_id>/?token=<access>`
- Text: `{"content": "..."}` → reply `{"role":"tutor","content":...,"meta":{...}}`.
- Voice: binary PCM 16k/mono, then `{"kind":"voice_end"}`; server streams
  `{"kind":"token"|"audio"|"done", ...}`. Barge-in: `{"kind":"voice_cancel"}`.
- Auth fail → close `4401`; bad session → `4404`.

**Assert**: every message carries `topic_id` (null for main chat). A syllabus-bound
question gets a non-null `topic_id`, and `/threads/` then shows that subtopic.
---

## 9. Subtopic threading (key smartadapter behavior)

The single most distinctive feature. Drive and assert it like this:

1. In main chat, ask about a **specific subtopic** (e.g. "How do I simplify algebraic
   fractions?").
2. **Assert**: the tutor message returns a non-null `topic_id`; the message is stored
   under that subtopic.
3. `GET .../threads/` now lists **Main chat** + the auto-created **Algebra /
   Algebraic fractions** thread (named from the subtopic title), ordered.
4. Ask a **follow-up** in that thread (e.g. "so answer is x=3 — what next?").
   **Assert** the thread's own prior context is present, and messages from other
   subtopics/main are NOT injected (thread-scoped history).
5. Unclear / off-syllabus / greeting → stays in **main chat** (`topic_id == null`).
6. Each distinct subtopic yields at most its own thread; if the subject has N
   subtopics, the list has ≤ N subtopic entries (+ main).

**Continuity rule**: within a thread, history is `created_at`-ordered (oldest →
newest), then the new user message follows — i.e. the new question comes *below* the
prior context, exactly like continuing that chat.

### Memory (student-fact) behaviour
- Say "I'm not good at maths, how can I improve?" → facts are extracted after the
  reply. Assert `MemoryEntry` rows exist for the student.
- Later ask where the fact is relevant → the system prompt includes remembered facts;
  the tutor must NOT ask you to restate them.
- Facts are `always_on` (broad traits, low threshold) vs `situational` (retrieved
  when on-topic, higher threshold).

---

## 10. Smart Practice/Exam buttons (data contract)

The app analyzes the tutor reply's `content`:
- `practice` / `exercise` / `test` → **Practice** button.
- `exam` / `paper` → **Exam** button.
- Only when a valid `subject_id` is known.

For automated agents:
- "give me a practice test" → tutor reply mentions practice intent.
- "how would this look in a real exam paper?" → exam intent.
- App navigates to `practice/<subjectId>` or `exam/<subjectId>`.

(Practice topic pre-selection from a thread is a follow-up; backend accepts
`topics=`/`objective_id=` filters on `quiz/next` and `quiz/generate`.)

---

## 11. ASCII diagrams / figures

- **Scenario 1 (reuse image)**: `figure_required=true` → `figure_urls` non-empty
  (real past-paper diagram PNGs).
- **Scenario 2 (draw fresh)**: `figure_required=false` → the question text contains a
  monospace ASCII diagram in a ```` ```ascii ```` block. The app renders it
  monospaced. Assert the glyphs + a self-consistent labelled shape, and that it
  differs across regenerations (fresh diagram ⇒ "unlimited questions").

Assert `QuestionDto.figure_urls` (Scenario 1) vs ASCII in `question_text`
(Scenario 2).

---

## 12. pgvector RAG behaviour

- Documents (syllabus + past papers) ingested into `DocumentChunk`; embeddings in
  `embedding_vec` (Postgres) + `embedding` (JSON).
- `retrieve(syllabus, query)` returns top-k chunks; HNSW cosine on Postgres,
  Python-cosine fallback on SQLite.
- Assert: tutor replies carry `meta.retrieved_chunk_ids` (RAG-grounded).
- Figures: past-paper corpus yields `rag_documentfigure` rows; `figure_required=true`
  questions map to real figure PNGs (sizes in KBs, not 58-byte blanks).
---

## 13. Recommended automated test scenario (happy-path end-to-end)

1. Register a fresh user; assert tokens + profile.
2. `GET /api/syllabi/?level=EGCSE` → first syllabus; `GET /api/subjects/` → pick
   Mathematics (6880).
3. `POST /api/my-subjects/` → enroll.
4. `GET /api/workspace/<id>/` → assert subject info + no sessions yet.
5. Open tutoring: `POST /api/chat-sessions/` → session id.
6. WebSocket connect; ask a syllabus question; assert tutor reply + `topic_id`.
7. `GET .../threads/` → assert the subtopic thread appeared.
8. Follow-up in that thread; assert continuity (prior thread context present).
9. Ask for a practice test; assert practice intent; then `GET /api/quiz/next/`,
   `POST /api/quiz/answer/` (MCQ), assert score/mastery.
10. `POST /api/quiz/exam/start/` Paper 1; `next/` a few times; `answer/`; assert
    running score.
11. `PATCH /api/me/profile/` change learning style; assert persisted.

Each step asserts HTTP codes, envelope shape, and the invariants in the relevant section.

---

## 14. Edge cases the agent should probe

- Unauthenticated request → `401`.
- Non-enrolled subject for generate/answer/exam/workspace → `403`.
- Unknown ids (`subject_id=999999`, `workspace/999999/`) → `400`.
- MCQ `selected_index` out of range → `400`.
- Structured answer missing `answer_text` → `400`.
- Weak/short password → `400`.
- Duplicate enroll (update_or_create) → `201`, not an error.
- Question payload never leaks `correct_index`/`explanation`/`marking_guidance`.
- Voice: send binary + `voice_end`, assert `{"kind":"transcript"|"token"|"audio"|"done"}`.
- Streaming chat: assert token events before `done` (voice path).
- Offline mode: when no LLM key, tutor returns a labelled `[offline mode ...]` reply.

---

## 15. Test tooling hints

- Unit/regression: `python manage.py test --settings config.test_settings` (hermetic
  in-memory SQLite). Existing suites: accounts, syllabus, quiz, progress, tutoring
  (memory + thread routing).
- Live E2E: drive the REST/WS flows in §13 against the deployed base URL.
- RAG/pdf correctness: verify `reembed_chunks` + ingestion produced real (non-blank)
  figure images and pgvector rows.

---

*This spec is derived from the actual backend implementation and the built Android
client. When endpoints/behavior change, keep this document in sync so agents stay
green.*
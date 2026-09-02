"""Student-fact memory.

Persists durable facts a student states during tutoring, embeds them, and
retrieves the topically-relevant ones on demand. This lets the tutor "remember"
across a long conversation (and across sessions) without sending full history:
only a small number of relevant facts are injected per turn.

Retrieval is threshold-gated:
  - always_on facts use a LOW threshold (usually injected - global traits).
  - situational facts use a HIGHER threshold (only when clearly relevant).
"""

from django.conf import settings

from apps.tutoring.models import MemoryEntry

# Tunable knobs (per the refined scheme).
ALWAYS_ON_THRESHOLD = 0.10
SITUATIONAL_THRESHOLD = 0.55
MEMORY_LIMIT = 4
ALWAYS_ON_LIMIT = 2
PROMOTE_IMPORTANCE = 8


def _embed(text: str) -> list[float]:
    from apps.rag.services.embeddings import get_embedder
    try:
        return get_embedder().embed_query(text)
    except Exception:
        return []


def _cosine(a, b) -> float:
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def user_memory_count(student) -> int:
    return MemoryEntry.objects.filter(student=student).count()


def relevant_memory(student, user_text: str, limit: int = MEMORY_LIMIT) -> list[MemoryEntry]:
    """Return up to `limit` facts relevant to `user_text`, threshold-gated."""
    qvec = _embed(user_text)
    if not qvec:
        return list(MemoryEntry.objects.filter(student=student)[:limit])

    entries = list(MemoryEntry.objects.filter(student=student))
    if not entries:
        return []

    scored_always: list[tuple[float, MemoryEntry]] = []
    scored_sit: list[tuple[float, MemoryEntry]] = []
    for e in entries:
        ev = e.embedding or _embed(e.fact)
        sim = _cosine(qvec, ev)
        if e.kind == MemoryEntry.Kind.ALWAYS_ON:
            if sim >= ALWAYS_ON_THRESHOLD:
                scored_always.append((sim, e))
        else:
            if sim >= SITUATIONAL_THRESHOLD:
                scored_sit.append((sim, e))

    scored_always.sort(key=lambda t: (-t[1].importance, -t[0]))
    scored_sit.sort(key=lambda t: t[0], reverse=True)

    chosen = scored_always[:ALWAYS_ON_LIMIT] + scored_sit[: max(0, limit - ALWAYS_ON_LIMIT)]
    return [e for _sim, e in chosen[:limit]]
EXTRACT_PROMPT = """You are a memory librarian for a tutoring app. From the conversation
(most recent student message + the tutor reply), extract any NEW, durable, factual
statements about the STUDENT that should be remembered - e.g. their level, subject
weaknesses, goals, exam dates, preferred study style, or recurring struggles.

Rules:
- Only durable facts that remain true over time. Ignore one-off questions, greetings,
  or statements already fully captured by an existing memory.
- Use third-person phrasing (\"Student is weak at Maths fractions\").
- Keep each fact short (under 90 chars) and self-contained.
- Rate importance 1-10. Score >=8 for broad, always-relevant traits (e.g. overall
  weakness in a subject); lower for specific one-off details.

Return ONLY valid JSON: {\"memories\": [{\"fact\": \"...\", \"importance\": 7}, ...]}
If nothing worth remembering, return {\"memories\": []}.
"""


def _fact_fingerprint(text: str) -> str:
    """Stable de-dup key: lowercased first ~20 words."""
    import re
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(words[:20])


def upsert_memory(student, fact: str, kind: str = MemoryEntry.Kind.SITUATIONAL, importance: int = 5) -> None:
    """Insert or update a memory fact (de-duplicated by fingerprint)."""
    fact = fact.strip()
    if not fact:
        return
    if kind == MemoryEntry.Kind.SITUATIONAL and importance >= PROMOTE_IMPORTANCE:
        kind = MemoryEntry.Kind.ALWAYS_ON
    fp = _fact_fingerprint(fact)
    for e in MemoryEntry.objects.filter(student=student).all():
        if _fact_fingerprint(e.fact) == fp:
            e.fact = fact
            e.importance = max(e.importance, importance)
            e.kind = MemoryEntry.Kind.ALWAYS_ON if e.importance >= PROMOTE_IMPORTANCE else kind
            e.embedding = _embed(fact)
            e.save()
            return
    if user_memory_count(student) >= 60:
        old = MemoryEntry.objects.filter(student=student, kind=MemoryEntry.Kind.SITUATIONAL).first()
        if old:
            old.delete()
    MemoryEntry.objects.create(student=student, fact=fact, kind=kind, importance=importance, embedding=_embed(fact))


def extract_facts(student, user_text: str, reply_text: str) -> int:
    """Ask the LLM to extract durable facts and upsert them. Returns count added."""
    from apps.rag.services.llm import get_chat_provider

    if not user_text.strip() or not reply_text.strip():
        return 0
    transcript = f"STUDENT: {user_text}\nTUTOR: {reply_text[:2000]}"
    try:
        provider = get_chat_provider("")
        raw = provider.chat(
            [
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": transcript},
            ]
        )
    except Exception:
        return 0
    memories = _parse_memories(raw)
    n = 0
    for m in memories:
        upsert_memory(student, m["fact"], MemoryEntry.Kind.SITUATIONAL, int(m.get("importance", 5)))
        n += 1
    return n


def _parse_memories(raw: str) -> list[dict]:
    import json
    import re

    if not raw:
        return []
    raw_clean = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_clean)
    if m:
        raw_clean = m.group(1).strip()
    try:
        data = json.loads(raw_clean)
    except Exception:
        obj = re.search(r"\{.*\}", raw_clean, re.DOTALL)
        if not obj:
            return []
        try:
            data = json.loads(obj.group(0))
        except Exception:
            return []
    mems = data.get("memories", []) if isinstance(data, dict) else []
    out = []
    for item in mems:
        if isinstance(item, dict) and item.get("fact"):
            out.append({"fact": str(item["fact"]).strip(), "importance": int(item.get("importance", 5))})
    return out
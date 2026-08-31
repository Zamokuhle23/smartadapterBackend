"""
Tutor orchestrator: turns a student message into a personalized, syllabus-grounded reply.

Personalization inputs gathered here:
1. LearnerProfile - language preference, learning style, pace
2. Progress data - weakest objectives for this subject become extra guidance
3. RAG context   - retrieved chunks from THIS student's syllabus only
"""

from apps.progress.services.dashboard import weak_objectives_for_message
from apps.rag.services.llm import get_chat_provider
from apps.rag.services.retriever import retrieve

SYSTEM_TEMPLATE = """You are FundzaAI, a warm, encouraging expert tutor for Eswatini students.

The student is studying:
- Syllabus: {syllabus_name} ({syllabus_level}), version {version}{subject_line}
- Form/level: {form_level}{tier_line}

STRICT RULES:
- Teach ONLY content within this syllabus's scope. If asked something outside it,
  gently redirect and explain that it is beyond their current syllabus.
- If a Curriculum tier is shown, teach ONLY that tier's content and papers: Core =
  lower tier (papers 1/2, grades C-G); Extended = higher tier (papers 3/4, A*-E).
- Ground explanations in the SYLLABUS CONTEXT below. If it is insufficient,
  say what you know but flag it as general knowledge.
- Adapt to the learner profile below.

LEARNER PROFILE (personalization):
- Language: {language}
- Learning style: {style} ({style_hint})
- Pace: {pace}

KNOWN WEAKNESSES (only those matching this topic; may be empty):
{weaknesses}

SYLLABUS CONTEXT (retrieved for this question):
{context}

Respond in clear steps, use worked examples when helpful, and end with one short
check-understanding question tailored to this student."""

STYLE_HINTS = {
    "socratic": "Guide with questions first; let the student do each step before revealing it.",
    "direct": "Explain directly and clearly, step by step, then check understanding.",
    "auto": "Start direct; if the student struggles twice on a topic, switch to guided questioning.",
}


def _tier_line(student, subject) -> str:
    """Formatted curriculum tier for the system prompt ('' if un-tiered)."""
    from apps.syllabus.services.subject_map import tier_for, tier_label

    tier = tier_for(student, subject)
    if not tier:
        return ""
    label = tier_label(tier)
    return "\n- Curriculum tier: " + (label or tier)


def _format_weaknesses(student, subject, user_text: str) -> str:
    """Topic-matched weaknesses for THIS question (may be empty for off-topic)."""
    from apps.syllabus.services.subject_map import tier_for

    tier = tier_for(student, subject)
    rows = weak_objectives_for_message(student, subject, user_text, limit=3, tier=tier)
    if not rows:
        return ""
    return "\n".join(f"- {statement} (mastery {mastery:.0%})" for statement, mastery in rows)


def build_messages(session, user_text: str, history_limit: int = 8) -> list[dict]:
    profile = session.student.profile if hasattr(session.student, "profile") else None
    language = getattr(profile, "preferred_language", "en") or "en"
    style = getattr(profile, "learning_style", "auto") or "auto"
    pace = getattr(profile, "pace", "normal") or "normal"
    language_desc = {"en": "English", "ss": "siSwati", "mix": "English with siSwati clarifications where useful"}[language]

    chunks = retrieve(session.syllabus, user_text, subject=session.subject)
    context = (
        "\n---\n".join(f"[chunk {c.ordinal}] {c.text}" for c in chunks)
        or "(no indexed syllabus text matched; answer from general knowledge within scope)"
    )

    weaknesses = _format_weaknesses(session.student, session.subject, user_text)

    system = SYSTEM_TEMPLATE.format(
        syllabus_name=session.syllabus.name,
        syllabus_level=session.syllabus.get_level_display(),
        version=session.syllabus.version,
        subject_line=f"\n- Subject: {session.subject.name} ({session.subject.code})" if session.subject else "",
        form_level=getattr(session.student, "form_level", None) or "-",
        tier_line=_tier_line(session.student, session.subject),
        language=language_desc,
        style=style,
        style_hint=STYLE_HINTS.get(style, ""),
        pace=pace,
        weaknesses=weaknesses,
        context=context,
    )

    recent = [
        {"role": "user" if m.role == "user" else "assistant", "content": m.content}
        for m in session.messages.all().order_by("-created_at")[:history_limit]
    ][::-1]

    return [{"role": "system", "content": system}, *recent, {"role": "user", "content": user_text}], [c.id for c in chunks]


def generate_reply(session, user_text: str) -> tuple[str, dict]:
    """Full turn: persist user message, produce grounded tutor reply."""
    messages, chunk_ids = build_messages(session, user_text)
    provider = get_chat_provider()
    reply_text = provider.chat(messages)
    meta = {
        "retrieved_chunk_ids": chunk_ids,
        "provider": type(provider).__name__,
        "model": getattr(type(provider), "model", "") or "",
    }
    return reply_text, meta

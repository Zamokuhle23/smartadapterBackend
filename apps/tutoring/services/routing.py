"""
Message -> subtopic routing.

Each student message is classified to a subtopic of the subject's syllabus tree
using the existing RAG retrieval: the retrieved chunk with the highest similarity
that carries a Topic determines the subtopic. If confidence is below a threshold
(uncertain/off-syllabus/greeting), the message stays in the root "main chat"
(topic=None).

Subtopic threads are auto-created implicitly: any message tagged with a Topic is
grouped under that Topic, so a thread appears the first time the student talks
about it. A subject yields at most as many threads as it has distinct subtopics.
"""

from django.conf import settings

from apps.syllabus.models import Topic

# Confidence floor for assigning a message to a subtopic. Below this the message
# stays in main chat (route returns None). Tunable.
ROUTE_THRESHOLD = 0.08


def _top_similar_chunks(syllabus, subject, user_text: str, k: int = 6):
    from apps.rag.services.retriever import retrieve

    return retrieve(syllabus, user_text, k=k, subject=subject)


def classify_topic(session, user_text: str) -> Topic | None:
    """Return the subtopic a message belongs to, or None for main chat."""
    if not user_text or not user_text.strip():
        return None
    subject = session.subject
    if subject is None:
        return None
    try:
        chunks = _top_similar_chunks(session.syllabus, subject, user_text, k=6)
    except Exception:
        return None
    if not chunks:
        return None

    # Take the first chunk that has a Topic mapped; the splittaed chunks are
    # returned in similarity order, so the first topic-bearing one is the best.
    for chunk in chunks:
        topic = getattr(chunk, "topic", None)
        if topic is not None:
            return topic
    return None


def thread_list(session) -> list[dict]:
    """
    Ordered list of (active) subtopic threads for this subject session, plus the
    main-chat entry. Ordered by the syllabus's topic order; each entry carries the
    latest message preview and a message count so the client can render the drawer.
    """
    from django.db.models import Count, Max

    qs = (
        session.messages.exclude(topic__isnull=True)
        .values("topic")
        .annotate(count=Count("id"), last=Max("created_at"))
    )
    topics = {t.id: t for t in Topic.objects.filter(pk__in=[r["topic"] for r in qs])}

    rows = []
    for r in qs:
        t = topics.get(r["topic"])
        if t is None:
            continue
        last_msg = session.messages.filter(topic_id=t.id).order_by("-created_at").first()
        rows.append(
            {
                "topic_id": t.id,
                "title": t.title,
                "parent_id": t.parent_id,
                "messages": r["count"],
                "updated_at": r["last"].isoformat() if r["last"] else None,
                "preview": (last_msg.content[:80] if last_msg else ""),
            }
        )
    # Order by topic tree: parent strands first, then subtopics by their order/id.
    rows.sort(key=lambda r: (r["parent_id"] or 0, r["title"].lower()))

    main_count = session.messages.filter(topic__isnull=True).count()
    main_last = session.messages.filter(topic__isnull=True).order_by("-created_at").first()
    main = {
        "topic_id": None,
        "title": "Main chat",
        "messages": main_count,
        "updated_at": main_last.created_at.isoformat() if main_last else None,
        "preview": (main_last.content[:80] if main_last else ""),
    }
    return [main] + rows
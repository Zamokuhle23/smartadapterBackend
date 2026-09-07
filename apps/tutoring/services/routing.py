"""
Message -> subtopic routing.

Each student message is classified to a subtopic of the subject's Topic tree
(built from PageTopic page labels by the build_topic_trees command). Scoring
is lexical: the message's content stems are matched against each subtopic's
title plus its source page labels, so "factorise x^2 - 9" lands in the
Factorisation thread even when the embedding retriever returns noisy chunks.

If confidence is below the threshold (uncertain/off-syllabus/greeting), the
message stays in the caller's current thread (follow-ups like "show me the
steps" continue where the student is) or in the root "main chat"
(topic=None) when there is no current thread.

Subtopic threads are auto-created implicitly: any message tagged with a Topic
is grouped under that Topic, so a thread appears the first time the student
talks about it. A later message about an existing subtopic is appended to
that thread.
"""

import re

from apps.syllabus.models import Topic

# Minimum lexical score for assigning a message to a subtopic. The thread
# title itself is worth 3 per shared stem, page-label variants 1 each, so a
# single shared thread-title word (score 3) routes while stray single-word
# label matches (score 1) do not. Tunable.
ROUTE_THRESHOLD = 2

_STOP = set(
    "how do i the a an is are was were be been to of and or in on for with "
    "what when where which who whom whose can could would should you your me "
    "my we our they their this that these those it its as at by from not no "
    "yes ok okay hi hello hey thanks thank please give get got make made show "
    "tell explain another more other such like just very really thing answer "
    "question example".split()
)

_SOCIAL = (
    "hi", "hello", "hey", "thanks", "thank you", "good morning",
    "good afternoon", "good evening", "bye", "who are you",
    "what can you do",
)


def _stems(text: str) -> set[str]:
    return {t[:6] for t in re.findall(r"[a-z]{3,}", (text or "").lower())
            if t not in _STOP}


def _is_social(user_text: str) -> bool:
    lowered = (user_text or "").lower().strip()
    lowered = re.sub(r"[^a-z ]", "", lowered).strip()
    if len(lowered) < 4:
        return True
    return any(lowered == s or lowered.startswith(s + " ") for s in _SOCIAL)


def classify_topic(session, user_text: str, fallback=None) -> Topic | None:
    """Return the subtopic a message belongs to, a fallback, or None (main).

    `fallback` is the Topic of the thread the student is currently viewing;
    lexically empty follow-ups ("show the steps") continue there instead of
    bouncing to main chat.
    """
    if not user_text or not user_text.strip():
        return fallback
    subject = session.subject
    if subject is None:
        return fallback
    if _is_social(user_text):
        return fallback
    try:
        topic = _best_topic(subject, user_text)
    except Exception:
        return fallback
    return topic if topic is not None else fallback


def _best_topic(subject, user_text: str) -> Topic | None:
    from apps.quiz.models import PageTopic

    from apps.syllabus.services.topic_labels import canonical_key

    user = _stems(user_text)
    if not user:
        return None
    # Aggregate source labels per canonical topic key.
    labels_by_key: dict[str, set[str]] = {}
    for label in PageTopic.objects.filter(
        document__subject=subject
    ).values_list("label", flat=True):
        key = canonical_key(label)
        if key:
            labels_by_key.setdefault(key, set()).add(label)
    if not labels_by_key:
        return None
    best_key, best_score = None, 0
    for key, labels in labels_by_key.items():
        title_hit = len(user & _stems(key))
        label_hit = len(user & _stems(" ".join(labels)))
        score = 3 * title_hit + label_hit
        if score > best_score:
            best_key, best_score = key, score
    if best_key is None or best_score < ROUTE_THRESHOLD:
        return None
    # Resolve the canonical key to its Topic row (title may differ in case).
    for topic in Topic.objects.filter(subject=subject):
        if canonical_key(topic.title) == best_key:
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
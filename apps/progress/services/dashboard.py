"""
Analytics over the mastery data: what is this student weak at, and what should
they do next? These functions feed both the dashboard API and the tutor prompt.
"""

import re

from ..models import MasteryRecord
from apps.syllabus.models import Topic

# Split into lowercased alphanumeric word tokens; drop stop-words-like noise.
_TOKEN_RE = re.compile(r"[a-z][a-z]{1,}")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "how", "what", "why",
    "can", "you", "me", "my", "about", "please", "explain", "is", "are", "am",
    "a", "an", "to", "of", "on", "in", "it", "do", "does", "help", "would",
    "could", "should", "i", "have", "has", "was", "were", "be", "been",
}


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def weak_objectives_for_message(student, subject, user_text: str, limit=3, tier: str = ""):
    """
    Topic-matched weak objectives.

    Returns [(statement, mastery), ...] for the student's weakest objectives that
    fall under a topic the user's message is clearly about. Returns () when the
    message is off-syllabus (e.g. "suggest a study timetable") or doesn't map to a
    topic, so the tutor doesn't inject irrelevant weaknesses.
    """
    msg_tokens = _tokens(user_text)
    if not msg_tokens:
        return []

    # 1. Which of this subject's topics does the message talk about?
    matched_topic_ids = set()
    for topic in Topic.objects.filter(subject=subject):
        tt = _tokens(topic.title)
        if not tt:
            continue
        coverage = len(tt & msg_tokens) / len(tt)
        if coverage >= 0.5:
            matched_topic_ids.add(topic.id)

    if not matched_topic_ids:
        return []  # off-topic / meta question -> no weaknesses

    # 2. Weak objectives whose topic tree sits under a matched topic.
    weak = list(
        MasteryRecord.objects.filter(student=student, mastery__lt=0.6)
        .select_related("objective__topic")
        .order_by("mastery")
    )
    result = []
    for record in weak:
        objective = record.objective
        topic = objective.topic
        if topic is None:
            continue
        # Skip objectives outside the student's tier (e.g. extended-only for a
        # core student).
        if tier and getattr(objective, "tier", "") and objective.tier not in ("", tier):
            continue
        # Walk up the topic tree; if any ancestor is a matched topic, attach it.
        node = topic
        matched = False
        while node is not None:
            if node.id in matched_topic_ids:
                matched = True
                break
            node = node.parent
        if matched:
            result.append((objective.statement, record.mastery))
        if len(result) >= limit:
            break
    return result


def weakest_objectives(student, subject=None, limit=5):
    """
    Return [(statement, mastery), ...] for the student's lowest-mastery objectives,
    optionally restricted to one Subject. Used by the tutor orchestrator to
    personalize every reply around known gaps.
    """
    qs = (
        MasteryRecord.objects.filter(student=student)
        .filter(mastery__lt=0.6)
        .select_related("objective__topic")
        .order_by("mastery")
    )
    if subject is not None:
        # Objectives whose topic tree belongs to this subject
        objective_ids = [
            r.objective_id
            for r in qs
            if _objective_subject(r.objective) == subject
        ]
        qs = qs.filter(objective_id__in=objective_ids or [None])
    return [
        (r.objective.statement, r.mastery)
        for r in qs[:limit]
    ]


def _objective_subject(objective):
    topic = objective.topic
    while topic is not None and topic.parent_id is not None:
        topic = topic.parent
    return getattr(topic, "subject", None)


def subject_summary(student):
    """Average mastery grouped by subject for dashboard charts."""
    records = MasteryRecord.objects.filter(student=student).select_related(
        "objective__topic__subject__syllabus"
    )
    buckets: dict[int, dict] = {}
    for record in records:
        subject = _objective_subject(record.objective)
        if subject is None:
            continue
        entry = buckets.setdefault(
            subject.id,
            {
                "subject_id": subject.id,
                "subject": subject.name,
                "code": subject.code,
                "syllabus_level": subject.syllabus.level,
                "mastery_values": [],
                "attempts": 0,
            },
        )
        entry["mastery_values"].append(record.mastery)
        entry["attempts"] += record.attempts

    result = []
    for entry in buckets.values():
        values = entry.pop("mastery_values")
        result.append(
            {
                **entry,
                "avg_mastery": round(sum(values) / len(values), 3) if values else 0.0,
                "objectives_tracked": len(values),
            }
        )
    result.sort(key=lambda item: item["avg_mastery"])  # weakest subjects first
    return result


def study_recommendations(student, limit=5):
    """
    Next-best-action list: practice the weakest objectives first.
    Phase 2 will extend this with prerequisite tracing and FSRS review due dates.
    """
    recommendations = []
    for statement, mastery in weakest_objectives(student, limit=limit):
        action = "practice" if mastery < 0.35 else "review"
        recommendations.append(
            {
                "objective_statement": statement,
                "mastery": mastery,
                "recommended_action": action,
                "reason": (
                    "Low mastery detected - targeted practice recommended"
                    if action == "practice"
                    else "Mastery fading - a quick review will consolidate it"
                ),
            }
        )
    return recommendations

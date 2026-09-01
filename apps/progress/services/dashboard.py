"""Analytics over the mastery data (dashboard + tutor prompts)."""

import re

from django.db.models import Avg, Count, Sum

from ..models import MasteryRecord
from apps.syllabus.models import Subject, Topic

_TOKEN_RE = re.compile(r"[a-z][a-z]{1,}")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "how", "what", "why",
    "can", "you", "me", "my", "about", "please", "explain", "is", "are", "am",
    "a", "an", "to", "of", "on", "in", "it", "do", "does", "help", "would",
    "could", "should", "i", "have", "has", "was", "were", "be", "been",
}


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _resolve_subject(objective):
    """Subject of an objective's topic tree (legacy-row backfill only)."""
    topic = objective.topic
    while topic is not None and topic.parent_id is not None:
        topic = topic.parent
    return getattr(topic, "subject", None)


def _ensure_subject(record):
    """Backfill a pre-migration MasteryRecord (subject NULL) from its objective tree."""
    if record.subject_id is None and record.objective_id:
        subject = _resolve_subject(record.objective)
        if subject is not None:
            MasteryRecord.objects.filter(pk=record.pk).update(subject=subject)


def _backfill_for_student(student):
    """Set the denormalised subject on any legacy NULL-subject rows (bounded, one-time)."""
    for record in MasteryRecord.objects.filter(student=student, subject__isnull=True).iterator(200):
        _ensure_subject(record)


def weak_objectives_for_message(student, subject, user_text, limit=3, tier=""):
    """Topic-matched weak objectives; empty when the message is off-syllabus."""
    msg_tokens = _tokens(user_text)
    if not msg_tokens:
        return []

    matched_topic_ids = set()
    for topic in Topic.objects.filter(subject=subject):
        tt = _tokens(topic.title)
        if tt and len(tt & msg_tokens) / len(tt) >= 0.5:
            matched_topic_ids.add(topic.id)

    if not matched_topic_ids:
        return []

    _backfill_for_student(student)
    weak = (
        MasteryRecord.objects.filter(student=student, subject=subject, mastery__lt=0.6)
        .select_related("objective__topic")
        .order_by("mastery")
    )
    result = []
    for record in weak:
        objective = record.objective
        topic = objective.topic
        if topic is None:
            continue
        if tier and getattr(objective, "tier", "") and objective.tier not in ("", tier):
            continue
        node = topic
        while node is not None:
            if node.id in matched_topic_ids:
                result.append((objective.statement, record.mastery))
                break
            node = node.parent
        if len(result) >= limit:
            break
    return result


def weakest_objectives(student, subject=None, limit=5):
    """[(statement, mastery)] for the student's weakest objectives, SQL-scoped."""
    _backfill_for_student(student)
    qs = MasteryRecord.objects.filter(student=student).select_related("objective__topic")
    if subject is not None:
        qs = qs.filter(subject=subject)
    qs = qs.filter(mastery__lt=0.6).order_by("mastery")
    return [(r.objective.statement, r.mastery) for r in qs[:limit]]


def subject_summary(student):
    """Average mastery grouped by subject for dashboard charts - one SQL query."""
    _backfill_for_student(student)
    rows = (
        MasteryRecord.objects.filter(student=student, subject__isnull=False)
        .values("subject_id", "subject__name", "subject__code", "subject__syllabus__level")
        .annotate(
            avg_mastery=Avg("mastery"),
            attempts=Sum("attempts"),
            objectives_tracked=Count("id"),
        )
    )
    result = [
        {
            "subject_id": r["subject_id"],
            "subject": r["subject__name"],
            "code": r["subject__code"],
            "syllabus_level": r["subject__syllabus__level"],
            "avg_mastery": round(r["avg_mastery"] or 0.0, 3),
            "attempts": r["attempts"] or 0,
            "objectives_tracked": r["objectives_tracked"],
        }
        for r in rows
    ]
    result.sort(key=lambda item: item["avg_mastery"])
    return result


def study_recommendations(student, limit=5):
    """Next-best-action list: practice the weakest objectives first."""
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
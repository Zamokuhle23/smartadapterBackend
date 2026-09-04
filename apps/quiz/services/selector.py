"""
Adaptive question selection: serve the weakest objectives first, avoid recent
repeats, and fall back gracefully.
"""

from django.db.models import Max, Q

from apps.progress.models import MasteryRecord

from ..models import QuizAttempt, QuizQuestion


def _expanded_topic_ids(topic_ids) -> list[int]:
    """Expand selected topic ids to include child subtopics.

    A selection may contain a strand (top-level grouping); we translate it into
    all of its subtopics because learning objectives hang off subtopics.
    """
    from apps.syllabus.models import Topic

    ids = set(int(i) for i in topic_ids if str(i).isdigit())
    if not ids:
        return []
    children = Topic.objects.filter(parent_id__in=ids).values_list("id", flat=True)
    return list(ids | set(children))


def _scope_qs(subject, topic_ids=None, expanded=None, titles=None, objective_ids=None):
    """Base query for this subject, restricted to the selected objectives/topics."""
    qs = QuizQuestion.objects.filter(subject=subject)
    if objective_ids:
        return qs.filter(objective_id__in=objective_ids)
    if not topic_ids:
        return qs
    if expanded is None:
        expanded = _expanded_topic_ids(topic_ids)
    if not expanded:
        return qs.none()
    if titles is None:
        from apps.syllabus.models import Topic

        titles = list(Topic.objects.filter(id__in=expanded).values_list("title", flat=True))
    scope = Q(objective__topic_id__in=expanded) | (
        Q(objective__isnull=True) & Q(topic_title__in=titles)
    )
    return qs.filter(scope)


def next_question_for(student, subject, topic_ids=None, objective_ids=None,
                      exclude_ids=None) -> QuizQuestion | None:
    """Pick the next practice question. IDs in exclude_ids (already shown this
    session, e.g. skipped) are never returned; when nothing fresh remains,
    returns None so the caller grows the bank instead of looping one row."""
    expanded = _expanded_topic_ids(topic_ids) if topic_ids else None
    titles = None
    if expanded:
        from apps.syllabus.models import Topic

        titles = list(Topic.objects.filter(id__in=expanded).values_list("title", flat=True))
    qs = _scope_qs(subject, topic_ids, expanded, titles, objective_ids)
    if exclude_ids:
        qs = qs.exclude(id__in=list(exclude_ids))
    if not qs.exists():
        return None

    attempted_ids = list(
        QuizAttempt.objects.filter(student=student, question__subject=subject)
        .values_list("question_id", flat=True)
    )

    # 1. Unattempted questions on the student's weakest objectives (within selection)
    weak_qs = MasteryRecord.objects.filter(student=student, objective__topic__subject=subject)
    if expanded:
        weak_qs = weak_qs.filter(objective__topic_id__in=expanded)
    if objective_ids:
        weak_qs = weak_qs.filter(objective_id__in=objective_ids)
    weak_objective_ids = list(
        weak_qs.order_by("mastery").values_list("objective_id", flat=True)[:5]
    )
    if weak_objective_ids:
        q = (
            qs.exclude(id__in=attempted_ids)
            .filter(objective_id__in=weak_objective_ids)
            .order_by("?")
            .first()
        )
        if q:
            return q

    # 2. Any unattempted question within selection
    q = qs.exclude(id__in=attempted_ids).order_by("?").first()
    if q:
        return q

    # 3. Least-recently attempted (recycle once the bank is exhausted)
    return (
        qs.annotate(
            last_attempt=Max("attempts__created_at", filter=Q(attempts__student=student))
        )
        .order_by("last_attempt", "?")
        .first()
    )


"""
Adaptive question selection: serve the weakest objectives first, avoid recent
repeats, and fall back gracefully.
"""

from django.db.models import Max, Q

from apps.progress.models import MasteryRecord

from ..models import QuizAttempt, QuizQuestion



def next_question_for(student, subject) -> QuizQuestion | None:
    qs = QuizQuestion.objects.filter(subject=subject)
    if not qs.exists():
        return None

    attempted_ids = list(
        QuizAttempt.objects.filter(student=student, question__subject=subject)
        .values_list("question_id", flat=True)
    )

    # 1. Unattempted questions on the student's weakest objectives
    weak_objective_ids = list(
        MasteryRecord.objects.filter(
            student=student, objective__topic__subject=subject
        )
        .order_by("mastery")
        .values_list("objective_id", flat=True)[:5]
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

    # 2. Any unattempted question
    q = qs.exclude(id__in=attempted_ids).order_by("?").first()
    if q:
        return q

    # 3. Least-recently attempted (recycle once the bank is exhausted)
    return (
        QuizQuestion.objects.filter(subject=subject)
        .annotate(
            last_attempt=Max("attempts__created_at", filter=Q(attempts__student=student))
        )
        .order_by("last_attempt", "?")
        .first()
    )


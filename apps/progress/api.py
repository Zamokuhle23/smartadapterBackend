from django.db.models import Avg, Count
from rest_framework import permissions, serializers
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.syllabus.models import LearningObjective, Subject
from apps.tutoring.models import ChatSession

from .models import MasteryEvent, MasteryRecord
from .services.bkt import update_mastery
from .services.dashboard import study_recommendations, subject_summary, weakest_objectives



class AttemptSerializer(serializers.Serializer):
    objective_id = serializers.IntegerField()
    correct = serializers.BooleanField()
    latency_ms = serializers.IntegerField(required=False, allow_null=True)
    hints_used = serializers.IntegerField(default=0, min_value=0)


class RecordAttemptView(APIView):
    """POST an attempt; updates the student's BKT mastery for that objective."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = AttemptSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            objective = LearningObjective.objects.get(pk=data["objective_id"])
        except LearningObjective.DoesNotExist:
            return Response({"detail": "Unknown objective_id"}, status=400)

        MasteryEvent.objects.create(
            student=request.user,
            objective=objective,
            correct=data["correct"],
            latency_ms=data.get("latency_ms"),
            hints_used=data.get("hints_used", 0),
        )

        record, _created = MasteryRecord.objects.get_or_create(
            student=request.user,
            objective=objective,
            defaults={"mastery": update_mastery(None, data["correct"])},
        )
        record.attempts += 1
        if data["correct"]:
            record.correct_count += 1
        record.mastery = update_mastery(record.mastery, data["correct"])
        record.save(update_fields=["attempts", "correct_count", "mastery"])

        return Response(
            {
                "objective_id": objective.id,
                "mastery": record.mastery,
                "attempts": record.attempts,
            }
        )


class WorkspaceView(APIView):
    """
    GET /api/workspace/<subject_id>/

    Everything about ONE subject workspace in a single call:
    subject info, mastery summary, personalized recommendations, the student's
    chats within this subject, and the latest session id ("continue where you
    left off").
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, subject_id: int):
        try:
            subject = Subject.objects.select_related("syllabus").get(pk=subject_id)
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject"}, status=400)

        records = MasteryRecord.objects.filter(
            student=request.user, objective__topic__subject=subject
        )
        agg = records.aggregate(avg=Avg("mastery"), tracked=Count("pk"))

        sessions = ChatSession.objects.filter(student=request.user, subject=subject)[:10]
        latest = sessions.first()

        return Response(
            {
                "subject": {
                    "id": subject.id,
                    "name": subject.name,
                    "code": subject.code,
                    "level": subject.syllabus.level,
                },
                "avg_mastery": round(agg["avg"] or 0.0, 3),
                "objectives_tracked": agg["tracked"],
                "recommendations": [
                    {"objective_statement": statement, "mastery": mastery}
                    for statement, mastery in weakest_objectives(request.user, subject=subject, limit=3)
                ],
                "latest_session_id": latest.id if latest else None,
                "sessions": [
                    {
                        "id": s.id,
                        "title": s.title,
                        "updated_at": s.updated_at.isoformat(),
                    }
                    for s in sessions
                ],
            }
        )


class DashboardView(APIView):
    """Per-subject mastery summary + personalized next-step recommendations."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response(
            {
                "subjects": subject_summary(request.user),
                "recommendations": study_recommendations(request.user),
                "diagnostic_complete": getattr(
                    getattr(request.user, "profile", None), "diagnostic_complete", False
                ),
            }
        )

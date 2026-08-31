from rest_framework import permissions, serializers
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.progress.models import MasteryEvent, MasteryRecord
from apps.progress.services.bkt import update_mastery
from apps.syllabus.models import LearningObjective, Subject

from .models import ExamSession, QuizAttempt, QuizQuestion
from .services.generator import (
    QuizGenerationError,
    generate_questions,
    grade_structured_answer,
    next_exam_question,
    start_exam_session,
)
from .services.selector import next_question_for


class QuestionPublicSerializer(serializers.ModelSerializer):
    """Never leaks correct_index/explanation/marking_guidance to the client."""
    figure_urls = serializers.SerializerMethodField()

    class Meta:
        model = QuizQuestion
        fields = (
            "id",
            "question_text",
            "options",
            "difficulty",
            "topic_title",
            "format",
            "marks",
            "paper_label",
            "source_year",
            "source",
            "adapted_from_past_paper",
            "figure_urls",
        )

    def get_figure_urls(self, obj):
        """Absolute URLs of the question's diagrams/figures, ready for image loading."""
        if not obj.figures.exists():
            return []
        request = self.context.get("request")
        urls = []
        for figure in obj.figures.all()[:6]:
            path = figure.image.url if figure.image else ""
            if not path:
                continue
            urls.append(request.build_absolute_uri(path) if request else path)
        return urls


class GenerateQuizView(APIView):
    """POST {subject_id, count?, difficulty?, objective_id?} -> new MCQs."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        subject_id = request.data.get("subject_id")
        try:
            subject = Subject.objects.get(pk=subject_id)
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=400)

        objective = None
        if request.data.get("objective_id"):
            objective = LearningObjective.objects.filter(
                pk=request.data["objective_id"]
            ).first()

        from apps.syllabus.services.subject_map import tier_for

        tier = tier_for(request.user, subject)
        try:
            questions = generate_questions(
                subject,
                count=request.data.get("count", 3),
                difficulty=request.data.get("difficulty"),
                objective=objective,
                tier=tier,
            )
        except QuizGenerationError as exc:
            return Response({"detail": str(exc)}, status=503)

        return Response(
            QuestionPublicSerializer(questions, many=True, context={"request": self.request}).data
        )


class NextQuestionView(APIView):
    """GET ?subject_id=N -> the next best practice question (adaptive)."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            subject = Subject.objects.get(pk=request.query_params.get("subject_id"))
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=400)
        question = next_question_for(request.user, subject)
        if question is None:
            return Response({"detail": "no_questions"}, status=404)
        return Response(QuestionPublicSerializer(question, context={"request": self.request}).data)


class AnswerQuizView(APIView):
    """POST {question_id, selected_index? | answer_text?, latency_ms?} -> grade + BKT update."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            question = QuizQuestion.objects.select_related("objective").get(
                pk=request.data.get("question_id")
            )
        except QuizQuestion.DoesNotExist:
            return Response({"detail": "Unknown question_id"}, status=400)

        latency_ms = request.data.get("latency_ms")
        correct = False
        selected = None
        awarded = max_marks = None
        feedback = ""
        explanation = question.explanation

        if question.format == QuizQuestion.Format.STRUCTURED:
            answer_text = (request.data.get("answer_text") or "").strip()
            if not answer_text:
                return Response({"detail": "answer_text required"}, status=400)
            try:
                awarded, max_marks, feedback = grade_structured_answer(question, answer_text)
            except QuizGenerationError as exc:
                return Response({"detail": str(exc)}, status=503)
            # Half marks or more counts as a correct event for the BKT model.
            correct = awarded >= max_marks * 0.5
            explanation = explanation or feedback
        else:
            try:
                selected = int(request.data.get("selected_index"))
            except (TypeError, ValueError):
                return Response({"detail": "selected_index required"}, status=400)
            selected = max(0, selected)
            correct = selected == question.correct_index

        QuizAttempt.objects.create(
            student=request.user,
            question=question,
            selected_index=selected,
            answer_text=request.data.get("answer_text") or "",
            awarded_marks=awarded,
            feedback=feedback,
            correct=correct,
            latency_ms=latency_ms,
        )

        mastery = None
        if question.objective is not None:
            MasteryEvent.objects.create(
                student=request.user,
                objective=question.objective,
                correct=correct,
                latency_ms=latency_ms,
            )
            record, _created = MasteryRecord.objects.get_or_create(
                student=request.user, objective=question.objective
            )
            record.attempts += 1
            if correct:
                record.correct_count += 1
            record.mastery = update_mastery(record.mastery, correct)
            record.save(update_fields=["attempts", "correct_count", "mastery"])
            mastery = record.mastery

        return Response(
            {
                "correct": correct,
                "correct_index": question.correct_index,
                "explanation": explanation,
                "mastery": mastery,
                "awarded_marks": awarded,
                "max_marks": int(max_marks) if max_marks is not None else None,
                "feedback": feedback,
            }
        )


# ---------------------------------------------------------------------------
# Exam simulation
# ---------------------------------------------------------------------------


def _session_response(session: ExamSession, request=None) -> dict:
    questions = QuizQuestion.objects.filter(id__in=session.question_ids)
    by_id = {q.id: q for q in questions}
    ordered = [by_id[qid] for qid in session.question_ids if qid in by_id]
    attempts = QuizAttempt.objects.filter(
        student=session.student, question_id__in=session.question_ids
    ).order_by("created_at")
    latest_per_question = {}
    for attempt in attempts:
        latest_per_question[attempt.question_id] = attempt
    score_awarded = sum((a.awarded_marks or 0) for a in latest_per_question.values())
    return {
        "id": session.id,
        "title": session.title,
        "paper_label": f"Paper {session.paper_number}",
        "tier": session.plan.get("tier", ""),
        "duration_minutes": session.duration_minutes,
        "status": session.status,
        "total_questions": session.total_questions,
        "answered": len(latest_per_question),
        "score_awarded": round(score_awarded, 1),
        "score_possible": sum(q.marks for q in ordered),
        "sections": session.plan.get("sections", []),
        "questions": QuestionPublicSerializer(
            ordered, many=True, context={"request": request} if request else {}
        ).data,
    }


class StartExamView(APIView):
    """POST {subject_id, paper} -> simulated exam sitting following the syllabus blueprint."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        subject_id = request.data.get("subject_id")
        try:
            subject = Subject.objects.select_related("syllabus").get(pk=subject_id)
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=400)

        try:
            paper = int(request.data.get("paper", 1))
        except (TypeError, ValueError):
            paper = 1
        paper = max(1, min(4, paper))

        from apps.syllabus.services.subject_map import tier_for

        tier = tier_for(request.user, subject)
        session = start_exam_session(request.user, subject, paper, tier=tier)
        return Response(_session_response(session, self.request), status=201)


class ExamStateView(APIView):
    """GET /api/quiz/exam/<id>/ -> sitting status, answered questions and running score."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        session = ExamSession.objects.filter(pk=pk, student=request.user).first()
        if session is None:
            return Response({"detail": "Not found"}, status=404)
        return Response(_session_response(session, self.request))


class ExamNextView(APIView):
    """POST /api/quiz/exam/<id>/next/ -> generate + return the next exam question."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        session = ExamSession.objects.filter(pk=pk, student=request.user).first()
        if session is None:
            return Response({"detail": "Not found"}, status=404)
        if len(session.question_ids) >= session.total_questions:
            return Response(None, status=204)
        try:
            question = next_exam_question(session)
        except QuizGenerationError as exc:
            return Response({"detail": str(exc)}, status=503)
        if question is None:
            return Response(None, status=204)
        return Response(QuestionPublicSerializer(question, context={"request": self.request}).data)

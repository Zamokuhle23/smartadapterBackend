from rest_framework import permissions, serializers
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from django.db import models

from apps.progress.models import MasteryEvent, MasteryRecord
from apps.progress.services.bkt import update_mastery
from apps.syllabus.models import (
    Enrollment,
    LearningObjective,
    Subject,
    SyllabusDocument,
)

from .models import CropAttempt, ExamSession, PageTopic, PaperAttempt, QuestionAnchor, QuestionCrop, QuizAttempt, QuizQuestion
from .services.generator import (
    QuizGenerationError,
    extract_keys,
    find_mark_scheme,
    generate_questions,
    grade_drawing,
    grade_structured_answer,
    grade_text,
    ms_excerpt_for,
    next_exam_question,
    start_exam_session,
)
from .services.selector import next_question_for


def _parse_id_list(values) -> list[int] | None:
    """Parse integer query params that arrive either comma-separated or repeated."""
    ids: list[int] = []
    for raw in values:
        for part in str(raw).split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
    return ids or None


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
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "llm"

    def post(self, request):
        subject_id = request.data.get("subject_id")
        try:
            subject = Subject.objects.select_related("syllabus").get(pk=subject_id)
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=400)

        if Enrollment.objects.filter(student=request.user, subject=subject).first() is None:
            return Response(
                {"detail": "Enroll in this subject before generating questions"},
                status=403,
            )

        objective = None
        if request.data.get("objective_id"):
            objective = LearningObjective.objects.filter(
                pk=request.data["objective_id"]
            ).first()

        # Optional topic restriction: [1,5,9] -> only generate for these topics.
        topics = request.data.get("topics") or []
        if isinstance(topics, (int, str)):
            topics = [topics]
        topic_ids = [int(t) for t in topics if str(t).isdigit()] or None

        # Optional objective restriction (finest): pick exactly these learning objectives.
        objectives = request.data.get("objectives") or request.data.get("objective_ids") or []
        if isinstance(objectives, (int, str)):
            objectives = [objectives]
        objective_ids = [int(o) for o in objectives if str(o).isdigit()] or None

        from apps.syllabus.services.subject_map import tier_for

        tier = tier_for(request.user, subject)
        try:
            questions = generate_questions(
                subject,
                count=request.data.get("count", 3),
                difficulty=request.data.get("difficulty"),
                objective=objective,
                tier=tier,
                topic_ids=topic_ids,
                objective_ids=objective_ids,
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
        if Enrollment.objects.filter(student=request.user, subject=subject).first() is None:
            return Response(
                {"detail": "Enroll in this subject before practising"},
                status=403,
            )
        # Optional topic restriction: ?topics=1,5,9 OR ?topics=1&topics=5&topics=9.
        # Retrofit (Android) sends a List<Int> @Query as repeated params, so use
        # getlist() and split each value on commas to cover both encodings.
        topic_ids = _parse_id_list(request.query_params.getlist("topics"))
        objective_ids = _parse_id_list(request.query_params.getlist("objectives"))
        # IDs already shown this session (answered or skipped): never repeat them.
        exclude_ids = _parse_id_list(request.query_params.getlist("exclude"))
        question = next_question_for(
            request.user, subject,
            topic_ids=topic_ids if not objective_ids else None,
            objective_ids=objective_ids,
            exclude_ids=exclude_ids,
        )
        if question is None:
            return Response({"detail": "no_questions"}, status=404)
        if QuizAttempt.objects.filter(
            student=request.user, question=question
        ).exists():
            # Bank exhausted for this student: the selector is recycling an
            # already-answered question. Grow the bank so practice keeps
            # serving fresh questions instead of looping on one.
            question = self._grow_and_repick(
                request, subject, topic_ids, objective_ids, exclude_ids,
                fallback=question,
            )
        return Response(QuestionPublicSerializer(question, context={"request": self.request}).data)

    @staticmethod
    def _grow_and_repick(request, subject, topic_ids, objective_ids,
                         exclude_ids, fallback):
        """Generate fresh questions, then re-pick. Falls back to the recycled
        question when generation fails (offline LLM, all items malformed)."""
        from apps.syllabus.services.subject_map import tier_for

        try:
            generate_questions(
                subject,
                count=3,
                tier=tier_for(request.user, subject),
                topic_ids=topic_ids,
                objective_ids=objective_ids,
            )
        except QuizGenerationError:
            return fallback
        fresh = next_question_for(
            request.user, subject,
            topic_ids=topic_ids if not objective_ids else None,
            objective_ids=objective_ids,
            exclude_ids=exclude_ids,
        )
        return fresh if fresh is not None else fallback


class AnswerQuizView(APIView):
    """POST {question_id, selected_index? | answer_text?, latency_ms?} -> grade + BKT update."""

    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "llm"

    def post(self, request):
        try:
            question = QuizQuestion.objects.select_related("objective", "subject").get(
                pk=request.data.get("question_id")
            )
        except QuizQuestion.DoesNotExist:
            return Response({"detail": "Unknown question_id"}, status=400)

        # Only enrolled students may answer a subject's questions, and only
        # within their curriculum tier (mirrors the generator's filtering).
        enrollment = Enrollment.objects.filter(
            student=request.user, subject=question.subject
        ).first()
        if enrollment is None:
            return Response({"detail": "Not enrolled in this question's subject"}, status=403)
        if question.subject.tiers_available and enrollment.tier:
            obj_tier = question.objective.tier if question.objective else ""
            if obj_tier and obj_tier != enrollment.tier:
                return Response(
                    {"detail": "Question belongs to a different curriculum tier"}, status=403
                )

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
            # Bounds-check against the real option list - a junk index is invalid,
            # not merely wrong.
            if not (0 <= selected < len(question.options)):
                return Response({"detail": "selected_index out of range"}, status=400)
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
                student=request.user,
                objective=question.objective,
                defaults={"subject": question.subject},
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
        if Enrollment.objects.filter(student=request.user, subject=subject).first() is None:
            return Response(
                {"detail": "Enroll in this subject before sitting an exam"},
                status=403,
            )

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
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "llm"

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


# --------------------------------------------------------------------------
# Smart practice: question-by-question mix of text-only paper parts + AI variants
# --------------------------------------------------------------------------


def _enrolled_or_404(user, subject_id):
    try:
        subject = Subject.objects.select_related("syllabus").get(pk=subject_id)
    except Subject.DoesNotExist:
        return None, Response({"detail": "Unknown subject_id"}, status=400)
    if Enrollment.objects.filter(student=user, subject=subject).first() is None:
        return None, Response(
            {"detail": "Enroll in this subject before practising"}, status=403)
    return subject, None


class SmartStartView(APIView):
    """POST {subject_id, topics?, count?} -> new smart practice session."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        from apps.syllabus.services.subject_map import tier_for

        subject, err = _enrolled_or_404(request.user, request.data.get("subject_id"))
        if err:
            return err
        topics = request.data.get("topics") or []
        if isinstance(topics, str):
            topics = [topics]
        topics = [t.strip() for t in topics if t.strip()]
        try:
            count = int(request.data.get("count", 20))
        except (TypeError, ValueError):
            count = 20
        from .services.smart import start_smart_session
        session = start_smart_session(
            request.user, subject, topics, count=count,
            tier=tier_for(request.user, subject))
        return Response({"session_id": session.id, "total_questions": session.total_questions},
                        status=201)


class SmartNextView(APIView):
    """POST /api/quiz/smart/<id>/next/ -> next item (anchor or variant)."""

    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "llm"

    def post(self, request, pk):
        from apps.syllabus.services.subject_map import tier_for

        from .models import PracticeSession
        from .services.smart import serve_next

        session = PracticeSession.objects.filter(pk=pk, student=request.user).first()
        if session is None:
            return Response({"detail": "Not found"}, status=404)
        tier = tier_for(request.user, session.subject)
        item = serve_next(session, tier=tier)
        if item is None:
            return Response(None, status=204)
        return Response(item)


class SmartAnswerView(APIView):
    """POST {index, answer_text? | selected_index? | drawing?, latency_ms?}."""

    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "llm"

    def post(self, request, pk):
        from apps.syllabus.services.subject_map import tier_for

        from .models import PracticeSession
        from .services.smart import answer_item

        session = PracticeSession.objects.filter(pk=pk, student=request.user).first()
        if session is None:
            return Response({"detail": "Not found"}, status=404)
        try:
            index = int(request.data.get("index"))
        except (TypeError, ValueError):
            return Response({"detail": "index required"}, status=400)
        result, code = answer_item(
            session, index, request.data,
            tier=tier_for(request.user, session.subject))
        if code != 200:
            return Response(result, status=code)
        return Response(result)


class SmartSummaryView(APIView):
    """GET /api/quiz/smart/<id>/summary/ -> compiled answers for review."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        from .models import PracticeSession
        from .services.smart import session_summary

        session = PracticeSession.objects.filter(pk=pk, student=request.user).first()
        if session is None:
            return Response({"detail": "Not found"}, status=404)
        return Response(session_summary(session))


class ExamDurationsView(APIView):
    """GET ?subject_id=N -> per-paper durations/weights for timed sittings.

    Cached sources only (never builds): the ExamProposition if present, else
    cached per-paper blueprints, else defaults (Paper 1 = 45 min, rest 120).
    Papers listed are the subject's real past papers, so the app can time an
    exam by the paper's written duration. A subject with no past papers
    returns an empty list (the app shows "No papers yet").
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        from .models import ExamBlueprint, ExamProposition

        try:
            subject = Subject.objects.get(pk=request.query_params.get("subject_id"))
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=400)
        if Enrollment.objects.filter(student=request.user, subject=subject).first() is None:
            return Response(
                {"detail": "Enroll in this subject first"},
                status=403,
            )
        numbers = sorted({
            n for n in SyllabusDocument.objects.filter(
                subject=subject, doc_type=SyllabusDocument.DocType.PAST_PAPER,
            ).values_list("paper_number", flat=True) if n
        })

        prop = {
            p.get("paper_number"): p
            for p in (ExamProposition.objects.filter(
                subject=subject).first().data.get("papers", [])
                if ExamProposition.objects.filter(subject=subject).exists() else [])
        }
        blueprints = {
            b.paper_number: b.data
            for b in ExamBlueprint.objects.filter(subject=subject)
        }
        out = []
        for n in numbers:
            entry = prop.get(n) or {}
            duration = entry.get("duration_minutes")
            if not duration and n in blueprints:
                try:
                    duration = int(blueprints[n].get("duration_minutes") or 0) or None
                except (TypeError, ValueError):
                    duration = None
            out.append({
                "paper_number": n,
                "label": entry.get("label") or f"Paper {n}",
                "duration_minutes": duration or (45 if n == 1 else 120),
                "weight_pct": entry.get("weight_pct"),
            })
        return Response({"papers": out})


# --------------------------------------------------------------------------
# Past-paper crops: exact scanned questions (text + diagram + table as one).
# --------------------------------------------------------------------------

class CropPublicSerializer(serializers.ModelSerializer):
    """A crop serialises like a question so the app reuses its UI.

    MCQ crops carry generic A-D options (the options live in the image);
    structured crops carry none. Grading keys come from the mark scheme.
    """
    options = serializers.SerializerMethodField()
    image_urls = serializers.SerializerMethodField()
    topic_title = serializers.SerializerMethodField()
    paper_label = serializers.SerializerMethodField()
    source_year = serializers.SerializerMethodField()
    source = serializers.SerializerMethodField()

    class Meta:
        model = QuestionCrop
        fields = (
            "id", "q_number", "stable_key", "format", "marks",
            "correct_index", "options", "image_urls", "ocr_text",
            "topic_title", "paper_label", "source_year", "source",
        )

    def get_options(self, obj):
        return ["A", "B", "C", "D"] if obj.format == "mcq" else []

    def get_image_urls(self, obj):
        request = self.context.get("request")
        urls = []
        for img in obj.images.all():
            if not img.image:
                continue
            path = img.image.url
            urls.append(request.build_absolute_uri(path) if request else path)
        return urls

    def _doc(self, obj):
        return obj.document

    def get_topic_title(self, obj):
        return f"Past paper Q{obj.q_number}"

    def get_paper_label(self, obj):
        paper = self._doc(obj).paper_number
        return f"Paper {paper}" if paper else "Past paper"

    def get_source_year(self, obj):
        return self._doc(obj).year

    def get_source(self, obj):
        return self._doc(obj).source


class NextCropView(APIView):
    """GET ?subject_id=N&exclude=1,2 -> next past-paper crop (never repeats).

    Serves approved/auto crops; MCQ crops only once keyed (correct_index).
    404 when nothing fresh remains (the app falls back to text questions).
    Crop answers never touch the BKT mastery model.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            subject = Subject.objects.get(pk=request.query_params.get("subject_id"))
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=400)
        if Enrollment.objects.filter(student=request.user, subject=subject).first() is None:
            return Response(
                {"detail": "Enroll in this subject before practising"},
                status=403,
            )
        exclude_ids = _parse_id_list(request.query_params.getlist("exclude"))
        qs = QuestionCrop.objects.filter(
            document__subject=subject,
            # NEEDS_QC crops are still real exam content (boundary review
            # pending); the QC flag gates trust, not serving.
            status__in=(QuestionCrop.Status.AUTO,
                        QuestionCrop.Status.APPROVED,
                        QuestionCrop.Status.NEEDS_QC),
        ).exclude(
            # Unkeyed MCQ crops are show-only (no correct answer to mark
            # against); structured ones grade by LLM either way.
            models.Q(format="mcq", correct_index__isnull=True),
        )
        if exclude_ids:
            qs = qs.exclude(id__in=list(exclude_ids))
        crop = qs.order_by("?").first()
        if crop is None:
            return Response({"detail": "no_questions"}, status=404)
        return Response(CropPublicSerializer(crop, context={"request": self.request}).data)


class CropAnswerView(APIView):
    """POST {crop_id, selected_index? | answer_text?} -> grade (no BKT)."""

    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "llm"

    def post(self, request):
        try:
            crop = QuestionCrop.objects.select_related("document").get(
                pk=request.data.get("crop_id"))
        except QuestionCrop.DoesNotExist:
            return Response({"detail": "Unknown crop_id"}, status=400)
        if Enrollment.objects.filter(
            student=request.user, subject=crop.document.subject
        ).first() is None:
            return Response({"detail": "Not enrolled in this question's subject"},
                            status=403)

        latency_ms = request.data.get("latency_ms")
        awarded = max_marks = None
        feedback = ""
        explanation = ""
        correct = False
        selected = None

        if crop.format == "mcq":
            if crop.correct_index is None:
                return Response({"detail": "This crop is not keyed yet"},
                                status=400)
            try:
                selected = int(request.data.get("selected_index"))
            except (TypeError, ValueError):
                return Response({"detail": "selected_index required"}, status=400)
            if not (0 <= selected < 4):
                return Response({"detail": "selected_index out of range"}, status=400)
            correct = selected == crop.correct_index
            max_marks = int(crop.marks or 1)
            awarded = float(max_marks) if correct else 0.0
        else:
            answer_text = (request.data.get("answer_text") or "").strip()
            if not answer_text:
                return Response({"detail": "answer_text required"}, status=400)
            try:
                awarded, max_marks, feedback = grade_text(
                    question_text=crop.ocr_text or f"Past paper Q{crop.q_number}",
                    guidance=crop.marking_guidance or "(none supplied)",
                    marks=crop.marks or 1,
                    answer_text=answer_text,
                )
            except QuizGenerationError as exc:
                return Response({"detail": str(exc)}, status=503)
            correct = awarded >= max_marks * 0.5
            explanation = feedback

        CropAttempt.objects.create(
            student=request.user,
            crop=crop,
            selected_index=selected,
            answer_text=request.data.get("answer_text") or "",
            awarded_marks=awarded,
            correct=correct,
            latency_ms=latency_ms,
        )
        return Response(
            {
                "correct": correct,
                "correct_index": crop.correct_index if crop.format == "mcq" else None,
                "explanation": explanation,
                "mastery": None,
                "awarded_marks": awarded,
                "max_marks": int(max_marks) if max_marks is not None else None,
                "feedback": feedback,
            }
        )


# --------------------------------------------------------------------------
# Interactive paper: page anchors + redact zones for the annotation layer.
# --------------------------------------------------------------------------


class PaperAnchorsView(APIView):
    """GET /api/quiz/paper/<doc_id>/anchors/ -> page geometry for the app.

    The app renders the original PDF itself, paints the redact zones over
    (barcodes, marginalia, admin furniture) and opens the answer overlay at
    tapped anchor bboxes. All bboxes are PDF points; multiply by the render
    zoom for screen pixels.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, doc_id):
        from apps.quiz.services.cropper import (
            answer_lines, content_crop, redact_zones)
        from apps.syllabus.models import SyllabusDocument

        try:
            doc = SyllabusDocument.objects.select_related("subject").get(
                pk=doc_id)
        except SyllabusDocument.DoesNotExist:
            return Response({"detail": "Unknown document"}, status=404)
        import pymupdf

        try:
            pdf = pymupdf.open(doc.file.path)
        except Exception:  # noqa: BLE001
            return Response({"detail": "Paper file unavailable"}, status=404)
        anchors = list(QuestionAnchor.objects.filter(
            document=doc).order_by("page_number", "qid").values(
            "qid", "page_number", "bbox", "kind", "confidence", "status"))
        pages = {}
        raster_line_cache = {}
        with pdf:
            for page in pdf:
                pno = page.number + 1
                try:
                    drawings = page.get_drawings()
                except Exception:  # noqa: BLE001
                    drawings = []
                page_anchors = [a for a in anchors
                                if a["page_number"] == pno]
                for a in page_anchors:
                    try:
                        a["lines"] = answer_lines(
                            page, a["bbox"], drawings, raster_line_cache
                        )
                    except Exception:  # noqa: BLE001
                        a["lines"] = []
                pages[pno] = {
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "questions": page_anchors,
                    "redact": redact_zones(page),
                    "crop": content_crop(page),
                }
        return Response({
            "document": doc.id,
            "title": doc.title,
            "subject": doc.subject.code if doc.subject else None,
            "pdf_url": (request.build_absolute_uri(doc.file.url)
                        if doc.file else None),
            "pages": pages,
        })


class PaperPagePartsView(APIView):
    """GET /api/quiz/paper/<doc_id>/page/<page_no>/parts/ -> one page's
    anchors with readable text plus an LLM variant per part.

    The app renders follow-up (continuation) pages as a text list instead of
    the PDF bitmap: each part shows its variant question (same skill, new
    wording, never referencing a diagram), falling back to the cleaned
    scraped text. Variants generate once per page, then cache forever.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, doc_id, page_no):
        from apps.quiz.services.cropper import anchor_marks, anchor_text, clean_part_text
        from apps.quiz.services.smart import generate_part_variants
        from apps.syllabus.models import SyllabusDocument

        try:
            doc = SyllabusDocument.objects.select_related("subject").get(
                pk=doc_id)
        except SyllabusDocument.DoesNotExist:
            return Response({"detail": "Unknown document"}, status=404)
        if doc.subject is not None and Enrollment.objects.filter(
            student=request.user, subject=doc.subject
        ).first() is None:
            return Response({"detail": "Not enrolled in this subject"},
                            status=403)
        anchors = list(QuestionAnchor.objects.filter(
            document=doc, page_number=page_no).order_by("qid"))
        try:
            variants = generate_part_variants(doc, page_no)
        except Exception:  # noqa: BLE001 - fall back to cleaned text
            variants = {}
        out = []
        for anchor in anchors:
            try:
                text = anchor_text(doc.file.path, page_no, anchor.bbox)
            except Exception:  # noqa: BLE001
                text = ""
            marks = anchor_marks(text) or anchor.marks or 2
            variant = variants.get(anchor.qid)
            out.append({
                "qid": anchor.qid,
                "kind": anchor.kind,
                "marks": marks,
                "text": text,
                "clean_text": clean_part_text(text),
                "variant": (
                    {"id": variant.id,
                     "question_text": variant.question_text,
                     "marks": variant.marks}
                    if variant is not None else None),
            })
        return Response(out)


class PaperAnswerView(APIView):
    """POST {doc_id, qid, answer_text?, selected_index?, drawing?, latency_ms?}.

    Grades one tapped paper anchor: text answers via the LLM against the
    mark scheme (resolved inline once, cached on the anchor); A-D letters
    against the cached correct_index; hand drawings via vision grading.
    History only, never BKT. drawing is a base64 PNG (max ~2 MB).
    """

    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "llm"

    def post(self, request):
        from apps.quiz.services.cropper import anchor_marks, anchor_text
        from apps.syllabus.models import SyllabusDocument

        try:
            doc = SyllabusDocument.objects.select_related("subject").get(
                pk=request.data.get("doc_id"))
            anchor = QuestionAnchor.objects.get(
                document=doc, qid=str(request.data.get("qid") or ""))
        except (SyllabusDocument.DoesNotExist, QuestionAnchor.DoesNotExist):
            return Response({"detail": "Unknown paper or question"}, status=400)
        if doc.subject is not None and Enrollment.objects.filter(
            student=request.user, subject=doc.subject
        ).first() is None:
            return Response({"detail": "Not enrolled in this subject"},
                            status=403)

        try:
            text = anchor_text(doc.file.path, anchor.page_number, anchor.bbox)
        except Exception:  # noqa: BLE001
            text = ""
        marks = anchor_marks(text) or anchor.marks or 2
        self._ensure_keys(doc, anchor, text)

        # Answering the displayed variant (follow-up text form): grade
        # against the variant's own marking guidance, not the anchor's.
        variant = None
        variant_qid = request.data.get("variant_question_id")
        if variant_qid is not None:
            try:
                variant = QuizQuestion.objects.select_related("objective").get(
                    pk=int(variant_qid), source_anchor=anchor)
            except (QuizQuestion.DoesNotExist, TypeError, ValueError):
                return Response({"detail": "Unknown variant question"},
                                status=400)

        latency_ms = request.data.get("latency_ms")
        awarded = max_marks = None
        feedback = ""
        correct = False
        drawing_b64 = ""
        selected = request.data.get("selected_index")
        if selected is not None and anchor.kind != "drawing":
            if anchor.correct_index is None:
                return Response({"detail": "This part is not keyed yet"},
                                status=400)
            try:
                selected = int(selected)
            except (TypeError, ValueError):
                return Response({"detail": "selected_index required"}, status=400)
            if selected not in (0, 1, 2, 3):
                return Response({"detail": "selected_index out of range"},
                                status=400)
            correct = selected == anchor.correct_index
            max_marks = int(marks)
            awarded = float(max_marks) if correct else 0.0
        else:
            answer_text = (request.data.get("answer_text") or "").strip()
            drawing_b64 = request.data.get("drawing") or ""
            if isinstance(drawing_b64, str) and drawing_b64.startswith("data:"):
                drawing_b64 = drawing_b64.split(",", 1)[-1]
            if len(drawing_b64) > 3_000_000:
                return Response({"detail": "drawing too large"}, status=400)
            if not answer_text and not drawing_b64:
                return Response({"detail": "answer_text required"}, status=400)
            try:
                if drawing_b64:
                    awarded, max_marks, feedback = grade_drawing(
                        question_text=text or f"Paper Q{anchor.qid}",
                        guidance=anchor.marking_guidance or "(none supplied)",
                        marks=marks,
                        image_b64=drawing_b64,
                    )
                else:
                    guidance = (variant.marking_guidance
                                if variant is not None and variant.marking_guidance
                                else anchor.marking_guidance or "(none supplied)")
                    awarded, max_marks, feedback = grade_text(
                        question_text=(
                            variant.question_text if variant is not None
                            else text or f"Paper Q{anchor.qid}"),
                        guidance=guidance,
                        marks=marks,
                        answer_text=answer_text,
                    )
            except QuizGenerationError as exc:
                return Response({"detail": str(exc)}, status=503)
            correct = awarded >= max_marks * 0.5

        attempt = PaperAttempt.objects.create(
            student=request.user,
            anchor=anchor,
            answer_text=request.data.get("answer_text") or "",
            awarded_marks=awarded,
            correct=correct,
            latency_ms=latency_ms,
        )
        if variant is not None:
            # The variant counts as an attempt on its own question too, so
            # mastery tracks concept grasp (and variant rotation counting).
            QuizAttempt.objects.create(
                student=request.user,
                question=variant,
                answer_text=request.data.get("answer_text") or "",
                awarded_marks=awarded,
                feedback=feedback,
                correct=correct,
                latency_ms=latency_ms,
            )
            if variant.objective is not None:
                MasteryEvent.objects.create(
                    student=request.user, objective=variant.objective,
                    correct=correct, latency_ms=latency_ms)
                record, _created = MasteryRecord.objects.get_or_create(
                    student=request.user, objective=variant.objective,
                    defaults={"subject": variant.subject})
                record.attempts += 1
                if correct:
                    record.correct_count += 1
                record.mastery = update_mastery(record.mastery, correct)
                record.save(update_fields=["attempts", "correct_count", "mastery"])
        if drawing_b64:
            from django.core.files.base import ContentFile
            import base64

            try:
                attempt.drawing.save(
                    f"{anchor.qid}.png",
                    ContentFile(base64.b64decode(drawing_b64)), save=True)
            except Exception:  # noqa: BLE001 - grade stands, image optional
                pass
        return Response(
            {
                "correct": correct,
                "correct_index": anchor.correct_index,
                "explanation": feedback,
                "mastery": None,
                "awarded_marks": awarded,
                "max_marks": int(max_marks) if max_marks is not None else None,
                "feedback": feedback,
                "model_answer": (
                    variant.marking_guidance if variant is not None
                    and variant.marking_guidance
                    else anchor.marking_guidance or ""),
            }
        )

    @staticmethod
    def _ensure_keys(doc, anchor, text):
        """Resolve mark-scheme keys once per anchor (cached forever)."""
        if anchor.marking_guidance:
            return
        ms = find_mark_scheme(doc.subject, doc.year, doc.paper_number)
        if ms is None:
            return
        base = "".join(ch for ch in anchor.qid if ch.isdigit())
        for qid in (anchor.qid, base):
            if not qid:
                continue
            excerpt = ms_excerpt_for(ms, qid)
            if not excerpt:
                continue
            keys = extract_keys(text, qid, excerpt)
            if not keys:
                continue
            anchor.marks = keys["marks"]
            anchor.correct_index = keys["correct_index"]
            anchor.marking_guidance = keys["marking_guidance"]
            anchor.save(update_fields=["marks", "correct_index",
                                       "marking_guidance"])
            return


class PaperExplainView(APIView):
    """POST {doc_id, qid} -> on-demand explanation for one paper question.

    Uses the cached mark-scheme guidance plus supporting syllabus /
    examiner-report chunks. Best-effort: never 500s on LLM failure.
    """

    permission_classes = [permissions.IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "llm"

    def post(self, request):
        from apps.quiz.services.cropper import anchor_text
        from apps.quiz.services.generator import explain_question
        from apps.syllabus.models import SyllabusDocument

        try:
            doc = SyllabusDocument.objects.select_related(
                "subject", "subject__syllabus").get(
                pk=request.data.get("doc_id"))
            anchor = QuestionAnchor.objects.get(
                document=doc, qid=str(request.data.get("qid") or ""))
        except (SyllabusDocument.DoesNotExist, QuestionAnchor.DoesNotExist):
            return Response({"detail": "Unknown paper or question"}, status=400)
        if doc.subject is not None and Enrollment.objects.filter(
            student=request.user, subject=doc.subject
        ).first() is None:
            return Response({"detail": "Not enrolled in this subject"},
                            status=403)
        try:
            text = anchor_text(doc.file.path, anchor.page_number, anchor.bbox)
        except Exception:  # noqa: BLE001
            text = ""
        PaperAnswerView._ensure_keys(doc, anchor, text)
        notes = self._supporting_notes(doc, text)
        try:
            explanation = explain_question(
                text or f"Paper Q{anchor.qid}",
                anchor.marking_guidance or "",
                notes,
            )
        except Exception:  # noqa: BLE001
            explanation = ""
        return Response({
            "explanation": explanation,
            "model_answer": anchor.marking_guidance or "",
        })

    @staticmethod
    def _supporting_notes(doc, question_text: str) -> str:
        """Top syllabus/examiner-report chunks for this question (RAG)."""
        from apps.quiz.services.generator import _chunk_source_line
        from apps.rag.services.retriever import retrieve

        if doc.subject is None or not (question_text or "").strip():
            return ""
        try:
            chunks = retrieve(
                doc.subject.syllabus, question_text[:1000], k=6,
                subject=doc.subject,
            )
        except Exception:  # noqa: BLE001
            return ""
        bits = []
        for chunk in chunks:
            doc_type = getattr(chunk.document, "doc_type", "")
            if doc_type not in ("syllabus", "notes"):
                continue
            bits.append(
                f"{_chunk_source_line(chunk)}\n{(chunk.text or '')[:800]}")
            if len(bits) >= 3:
                break
        return "\n\n".join(bits)


# --------------------------------------------------------------------------
# Practice sourcing from tagged pages: distinct labels + anchor feed.
# --------------------------------------------------------------------------


class PageTopicsView(APIView):
    """GET ?subject_id=N -> distinct page labels (the practice pick-list).

    Normalised: case-insensitive dedupe (most common casing wins) and
    non-content labels (blank pages, instructions, formula sheets,
    working space) excluded. Only labels on pages that actually carry
    question anchors in a past paper are listed - otherwise the picker
    offers topics that practice/pages/?topics=<label> can never serve
    (404 no_questions) and the learner hits a dead end.
    """

    permission_classes = [permissions.IsAuthenticated]

    JUNK_LABELS = ("blank", "instruction", "working space", "formula",
                   "calculator", "answer booklet", "answer lines",
                   "examiner", "copyright")


    def get(self, request):
        try:
            subject = Subject.objects.get(pk=request.query_params.get("subject_id"))
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=400)
        from django.db.models import Count, F

        rows = list(PageTopic.objects.filter(
            document__subject=subject,
            document__doc_type=SyllabusDocument.DocType.PAST_PAPER,
            # Same-page anchor must exist: the topic filter in
            # PracticePagesView joins anchors to same-page labels, so a
            # label without anchored pages could never return content.
            document__anchors__page_number=F("page_number"),
        ).values("label").annotate(
            pages=Count("id", distinct=True)).order_by("-pages"))
        grouped = {}
        for row in rows:
            key = (row["label"] or "").strip().lower()
            if not key or any(j in key for j in self.JUNK_LABELS):
                continue
            if key not in grouped:
                grouped[key] = {"label": row["label"].strip(), "pages": 0}
            grouped[key]["pages"] += row["pages"]
        return Response(sorted(grouped.values(), key=lambda r: -r["pages"]))


class NextAnchorView(APIView):
    """GET ?subject_id=N&topics=a,b&exclude=1,2 -> one tappable anchor.

    Topics match PageTopic labels (exact, case-insensitive). 404 when
    nothing fresh remains (the app falls back to text questions).
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            subject = Subject.objects.get(pk=request.query_params.get("subject_id"))
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=400)
        if Enrollment.objects.filter(student=request.user, subject=subject).first() is None:
            return Response(
                {"detail": "Enroll in this subject before practising"},
                status=403,
            )
        topics = [t.strip() for t in
                  (request.query_params.get("topics") or "").split(",")
                  if t.strip()]
        exclude_ids = _parse_id_list(request.query_params.getlist("exclude"))
        qs = QuestionAnchor.objects.filter(document__subject=subject)
        if topics:
            from django.db.models import F as _F
            from django.db.models import Q as _Q

            label_q = _Q()
            for label in topics:
                # Same-join conditions: the label must sit on the anchor's
                # own page, not merely somewhere in the document.
                label_q |= _Q(
                    document__page_topics__label__iexact=label,
                    document__page_topics__page_number=_F("page_number"),
                )
            qs = qs.filter(label_q)
        if exclude_ids:
            qs = qs.exclude(id__in=list(exclude_ids))
        anchor = qs.order_by("?").first()
        if anchor is None:
            return Response({"detail": "no_questions"}, status=404)
        doc = anchor.document
        is_mcq = anchor.correct_index is not None
        starts_mid, context_pages, continued_pages = question_context(doc, anchor.page_number)
        return Response({
            "id": anchor.id,
            "doc_id": doc.id,
            "qid": anchor.qid,
            "page_number": anchor.page_number,
            "starts_mid_question": starts_mid,
            "context_pages": context_pages,
            "continued_pages": continued_pages,
            "bbox": anchor.bbox,
            "kind": anchor.kind,
            "format": "mcq" if is_mcq else "structured",
            "marks": anchor.marks or 2,
            "label": _anchor_label(anchor),
            "pdf_url": (request.build_absolute_uri(doc.file.url)
                        if doc.file else None),
            "paper_label": (f"Paper {doc.paper_number}"
                            if doc.paper_number else "Past paper"),
            "source_year": doc.year,
            "source": doc.source,
        })


def _anchor_label(anchor) -> str:
    topic = PageTopic.objects.filter(
        document=anchor.document, page_number=anchor.page_number).first()
    return topic.label if topic else ""


def _qid_prefix(qid: str) -> str:
    """Leading digits of a question id ("10b" -> "10", "4" -> "4")."""
    out = []
    for ch in (qid or ""):
        if ch.isdigit():
            out.append(ch)
        else:
            break
    return "".join(out)


def _is_bare_number(qid: str) -> bool:
    """A bare question head ("5") vs a lettered part ("5a", "(b)")."""
    q = (qid or "").strip().strip("()")
    return bool(q) and q.isdigit()


def _question_key(qid: str) -> str:
    """Group anchors of one exam question: numeric prefix ("10b" -> "10"),
    else the whole normalized qid (bare "(b)" slices match each other)."""
    q = (qid or "").strip()
    prefix = _qid_prefix(q)
    if prefix:
        return f"#{prefix}"
    return q.strip("()").lower() or q.lower()


def question_context(document, page_number: int,
                     _anchor_cache: dict | None = None, span: int = 3,
                     ) -> tuple[bool, list[int], list[int]]:
    """Group split questions with their stem, both directions.

    Returns (starts_mid_question, context_pages, continued_pages):
    - context_pages: earlier pages of the same question(s), ascending
      (stem first: problem statement, then the parts that build on it), so
      a part like 5b never appears without 5/5a above it.
    - continued_pages: later pages holding the same question(s), ascending,
      so the client can label "continued on next page".
    - starts_mid_question: True when every question on this page already
      started on an earlier page (a pure continuation slice).

    Pages group by question key over a contiguous run (gap = different
    question). A bare-number slice ("5" alone, no lettered parts) counts as
    its question's continuation just like "5b" does.
    """
    if _anchor_cache is None:
        qids = list(QuestionAnchor.objects.filter(
            document=document, page_number=page_number
        ).values_list("qid", flat=True))
        _anchor_cache = {(document.id, page_number): qids}

    def keys_on(doc_id, page):
        key = (doc_id, page)
        if key not in _anchor_cache:
            _anchor_cache[key] = list(QuestionAnchor.objects.filter(
                document_id=doc_id, page_number=page
            ).values_list("qid", flat=True))
        return {_question_key(q) for q in _anchor_cache[key]}

    keys_here = keys_on(document.id, page_number)
    if not keys_here:
        return False, [], []
    context: set[int] = set()
    continued: set[int] = set()
    started_earlier: dict[str, bool] = {}
    for key in keys_here:
        # Backward run: the stem chain (nearest pages first, stored ascending).
        page = page_number - 1
        back: list[int] = []
        while page >= 1 and len(back) < span:
            if key in keys_on(document.id, page):
                back.append(page)
                page -= 1
            else:
                break
        context.update(back)
        started_earlier[key] = bool(back)
        # Forward run: pages this question spills onto.
        page = page_number + 1
        fwd = 0
        while fwd < span:
            if key in keys_on(document.id, page):
                continued.add(page)
                page += 1
                fwd += 1
            else:
                break
    # "Mid" only when every question on this page continues from an earlier
    # page AND at least one does (all([]) is True - that would wrongly flag a
    # page of orphan parts whose stem is genuinely absent).
    starts_mid = bool(started_earlier) and all(started_earlier.values())
    return starts_mid, sorted(context), sorted(continued)


def _prefetch_anchor_cache(documents_pages: dict[int, set[int]],
                           span: int = 3) -> dict:
    """One query per document for anchors in [min_page - span, max_page + span].

    The window covers neighbours on both sides so backward stem chains and
    forward continuations both resolve from cache.
    """
    cache: dict = {}
    for doc_id, pages in documents_pages.items():
        lo = max(1, min(pages) - span)
        hi = max(pages) + span
        rows = (QuestionAnchor.objects.filter(
            document_id=doc_id, page_number__gte=lo, page_number__lte=hi)
            .values_list("page_number", "qid"))
        for page, qid in rows:
            cache.setdefault((doc_id, page), []).append(qid)
    return cache


class PracticePagesView(APIView):
    """GET ?subject_id=N&topics=a,b&limit=20 -> page queue for practice.

    Serves topic-filtered pages as WHOLE real-paper runs, each run in the
    paper's own page order (no cross-paper shuffling), so a page that begins
    mid-question is always preceded in the queue by its stem page. A run is
    never started on a mid-question page (its stem would be missing), so no
    question ever "spawns" as a bare part without its context. Statement-only
    stem pages stay read_only (nothing answerable, no submit).
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            subject = Subject.objects.get(pk=request.query_params.get("subject_id"))
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=400)
        if Enrollment.objects.filter(student=request.user, subject=subject).first() is None:
            return Response(
                {"detail": "Enroll in this subject before practising"},
                status=403,
            )
        topics = [t.strip() for t in
                  (request.query_params.get("topics") or "").split(",")
                  if t.strip()]
        try:
            limit = max(1, min(50, int(request.query_params.get("limit", 20))))
        except (TypeError, ValueError):
            limit = 20
        pages = (QuestionAnchor.objects.filter(
            document__subject=subject,
            document__doc_type=SyllabusDocument.DocType.PAST_PAPER)
            .values("document_id", "page_number").distinct())
        if topics:
            from django.db.models import F as _F
            from django.db.models import Q as _Q

            label_q = _Q()
            for label in topics:
                label_q |= _Q(document__page_topics__label__iexact=label,
                              document__page_topics__page_number=_F("page_number"))
            pages = pages.filter(label_q)
        # DISTINCT can duplicate pairs when the anchor join fans out (one row
        # per anchor qid), so collect and dedupe pairs explicitly.
        pairs = set()
        for r in pages:
            pairs.add((r["document_id"], r["page_number"]))
        if not pairs:
            return Response({"detail": "no_questions"}, status=404)
        doc_ids = {d for d, _ in pairs}
        docs = {d.id: d for d in SyllabusDocument.objects.filter(
            id__in=doc_ids).select_related("subject")}
        # Prefetch anchors around the selected pages so stem/continuation
        # detection stays at one query per document instead of per page.
        doc_pages: dict[int, set[int]] = {}
        for d, p in pairs:
            doc_pages.setdefault(d, set()).add(p)
        anchor_cache = _prefetch_anchor_cache(doc_pages)
        if topics:
            # A topic page that begins mid-question would be dropped by
            # the orphan trimmer below (its stem page wasn't selected),
            # turning a servable label into a 404. Pull the stem pages
            # into the run so the question ships with its context. (The
            # prefetch window already covers these neighbours, so this
            # stays cache-backed.)
            for doc_id, selected in list(doc_pages.items()):
                doc = docs.get(doc_id)
                if doc is None:
                    continue
                for pno in list(selected):
                    starts_mid, context_pages, _ = question_context(
                        doc, pno, anchor_cache)
                    if starts_mid:
                        doc_pages[doc_id] |= set(context_pages)
        # Prefetch page labels for the selected pages in one query.
        label_map: dict[tuple[int, int], str] = {}
        for doc_id, page_no, lab in PageTopic.objects.filter(
                document_id__in=set(doc_pages)).values_list(
                "document_id", "page_number", "label"):
            label_map.setdefault((doc_id, page_no), lab)

        def page_entry(doc, page_no, kept: set[int]) -> dict:
            starts_mid, context_pages, continued_pages = question_context(
                doc, page_no, anchor_cache)
            qids = anchor_cache.get((doc.id, page_no), [])
            # Statement-only stem page: every anchor is a bare head whose
            # lettered parts live on later pages, so nothing is answerable
            # here - the client shows it read-only with no submit button.
            # (A bare head with no lettered parts anywhere stays answerable.)
            read_only = False
            if qids and all(_is_bare_number(q) for q in qids):
                keys = {_question_key(q) for q in qids}
                page = page_no + 1
                for _ in range(3):
                    later = anchor_cache.get((doc.id, page), [])
                    if any(_question_key(q) in keys and not _is_bare_number(q)
                           for q in later):
                        read_only = True
                        break
                    page += 1
            # "continued" only counts when the later page is itself in the
            # served run - i.e. the next screen genuinely carries the rest.
            continued_pages = [p for p in continued_pages if p in kept]
            return {
                "doc_id": doc.id,
                "page_number": page_no,
                "starts_mid_question": starts_mid,
                "context_pages": [p for p in context_pages if p in kept],
                "continued_pages": continued_pages,
                # Statement-only stem page (bare head, question continues
                # later): no answerable parts here, client shows it
                # read-only with no submit button.
                "read_only": read_only,
                "label": label_map.get((doc.id, page_no), ""),
                "pdf_url": (request.build_absolute_uri(doc.file.url)
                            if doc.file else None),
                "paper_label": (f"Paper {doc.paper_number}"
                                if doc.paper_number else "Past paper"),
                "source_year": doc.year,
                "source": doc.source,
            }

        def doc_run(doc_id: int) -> list[dict]:
            """Sorted page run for one paper, minus any orphan start."""
            doc = docs.get(doc_id)
            if doc is None:
                return []
            sel = sorted(doc_pages[doc_id])
            kept = set(sel)
            run = [page_entry(doc, pno, kept) for pno in sel]
            # Never start a run on a page that begins mid-question: its stem
            # page is not in this run, so the parts would spawn with no
            # context. (Later mid pages are fine - their stems precede them.)
            while run and run[0]["starts_mid_question"]:
                run.pop(0)
            return run

        out: list[dict] = []
        for doc_id in sorted(doc_pages):
            if len(out) >= limit:
                break
            run = doc_run(doc_id)
            if not run:
                continue
            if len(out) + len(run) > limit:
                # Never split a run across the limit (would orphan the tail);
                # a lone oversized run is capped at the limit.
                if not out:
                    out = run[:limit]
                break
            out.extend(run)
        if not out:
            return Response({"detail": "no_questions"}, status=404)
        return Response({"pages": out})

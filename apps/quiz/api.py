from rest_framework import permissions, serializers
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from django.db import models

from apps.progress.models import MasteryEvent, MasteryRecord
from apps.progress.services.bkt import update_mastery
from apps.syllabus.models import Enrollment, LearningObjective, Subject

from .models import CropAttempt, ExamSession, PaperAttempt, QuestionAnchor, QuestionCrop, QuizAttempt, QuizQuestion
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
        from apps.quiz.services.cropper import redact_zones
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
        with pdf:
            for page in pdf:
                pno = page.number + 1
                pages[pno] = {
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "questions": [a for a in anchors
                                  if a["page_number"] == pno],
                    "redact": redact_zones(page),
                }
        return Response({
            "document": doc.id,
            "title": doc.title,
            "subject": doc.subject.code if doc.subject else None,
            "pdf_url": (request.build_absolute_uri(doc.file.url)
                        if doc.file else None),
            "pages": pages,
        })


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
                    awarded, max_marks, feedback = grade_text(
                        question_text=text or f"Paper Q{anchor.qid}",
                        guidance=anchor.marking_guidance or "(none supplied)",
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

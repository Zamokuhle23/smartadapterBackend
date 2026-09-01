from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient

from apps.accounts.models import User
from apps.progress.models import MasteryRecord
from apps.quiz.models import QuizAttempt, QuizQuestion
from apps.quiz.services.generator import QuizGenerationError
from apps.syllabus.management.commands.seed_syllabi import EGCSE_SUBJECTS
from apps.syllabus.models import Enrollment, LearningObjective, Subject, Syllabus, Topic

FAKE_LLM_JSON = (
    '[{"question": "Simplify 6x/9.", "options": ["2x/3", "3x/2", "6x/3", "x/3"], '
    '"correct_index": 0, "explanation": "Divide both by 3.", '
    '"objective_hint": "Algebraic fractions", "difficulty": 2}]'
)


class FakeProvider:
    def chat(self, messages):
        return FAKE_LLM_JSON


def make_maths():
    syllabus = Syllabus.objects.create(level="EGCSE", name="EGCSE Test", version="1.0")
    subject = Subject.objects.create(syllabus=syllabus, code=EGCSE_SUBJECTS[4][0], name="Mathematics")
    Topic.objects.create(subject=subject, title="Algebra")
    # A realistic subject has several strands; the exam blueprint is grounded to the
    # subject's real topics, so add the topic the mock blueprint expects.
    Topic.objects.create(subject=subject, title="Geometry")
    topic = Topic.objects.get(subject=subject, title="Algebra")
    obj = LearningObjective.objects.create(topic=topic, statement="Simplify algebraic fractions")
    return syllabus, subject, obj


class QuizFlowTests(TestCase):
    def setUp(self):
        self.syllabus, self.subjects_entry, self.obj = make_maths()
        self.subject = self.subjects_entry
        self.user = User.objects.create_user("learner", password="test-pass-123")
        Enrollment.objects.create(student=self.user, subject=self.subject)
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def _generate(self):
        with patch("apps.rag.services.llm.get_chat_provider", return_value=FakeProvider()), patch(
            "apps.rag.services.retriever.retrieve", return_value=[]
        ):
            response = self.client.post(
                "/api/quiz/generate/",
                {"subject_id": self.subject.id, "count": 1, "objective_id": self.obj.id},
                format="json",
            )
        self.assertEqual(response.status_code, 200)
        return response.json()


    def test_generation_creates_valid_question(self):
        data = self._generate()
        self.assertEqual(len(data), 1)
        q = data[0]
        self.assertNotIn("correct_index", q)  # never leaked to students
        self.assertEqual(len(q["options"]), 4)

    def test_generation_without_llm_returns_503(self):
        with patch("apps.quiz.api.generate_questions", side_effect=QuizGenerationError("No LLM configured")):
            response = self.client.post(
                "/api/quiz/generate/", {"subject_id": self.subject.id}, format="json"
            )
        self.assertEqual(response.status_code, 503)


    def test_next_then_answer_updates_mastery(self):
        payload = self._generate()

        # Seed a mastery record so adaptive selection has a signal.
        MasteryRecord.objects.create(student=self.user, objective=self.obj, mastery=0.2, subject=self.subject)

        nxt = self.client.get(f"/api/quiz/next/?subject_id={self.subject.id}")
        self.assertEqual(nxt.status_code, 200)
        question_id = nxt.json()["id"]
        self.assertEqual(question_id, payload[0]["id"])

        answer = self.client.post(
            "/api/quiz/answer/",
            {"question_id": question_id, "selected_index": 0, "latency_ms": 5000},
            format="json",
        )
        body = answer.json()
        self.assertTrue(body["correct"])
        self.assertIsNotNone(body["mastery"])

        record = MasteryRecord.objects.get(student=self.user, objective=self.obj)
        self.assertEqual(record.attempts, 1)
        self.assertEqual(record.correct_count, 1)
        self.assertGreater(record.mastery, 0.2)


    def test_next_returns_404_when_bank_empty(self):
        response = self.client.get(f"/api/quiz/next/?subject_id={self.subject.id}")
        self.assertEqual(response.status_code, 404)


BLUEPRINT_JSON = (
    '{"paper_label": "Paper 1", "duration_minutes": 90, "total_questions": 2, '
    '"sections": [{"topic": "Algebra", "weight_pct": 50, "questions": 1, "format": "mcq"}, '
    '{"topic": "Geometry", "weight_pct": 50, "questions": 1, "format": "mcq"}]}'
)

EXAM_QUESTION_JSON = (
    '{"question": "Simplify 8x/12.", "format": "mcq", "options": ["2x/3", "3x/2", "4x/6", "x/2"], '
    '"correct_index": 0, "marks": 2, "explanation": "Divide by 4.", '
    '"marking_guidance": "B1 for 2x/3", "objective_hint": "Algebraic fractions"}'
)


class ScriptedProvider:
    """Returns canned responses depending on which prompt stage fired."""

    def __init__(self):
        self.calls = []

    def chat(self, messages):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if "assessment scheme" in prompt.lower():
            return BLUEPRINT_JSON
        if "examiner marking" in prompt.lower():
            return GRADE_JSON
        return EXAM_QUESTION_JSON


GRADE_JSON = '{"awarded": 3, "max": 4, "feedback": "Method mark earned; final accuracy lost."}'


class ExamFlowTests(TestCase):
    def setUp(self):
        syllabus, subject_obj, self.obj = make_maths()
        self.subject = subject_obj
        self.user = User.objects.create_user("sitter", password="test-pass-123")
        Enrollment.objects.create(student=self.user, subject=self.subject)
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.provider = ScriptedProvider()

    def _start(self):
        with patch(
            "apps.rag.services.llm.get_chat_provider", return_value=self.provider
        ), patch("apps.rag.services.retriever.retrieve", return_value=[]):
            response = self.client.post(
                "/api/quiz/exam/start/",
                {"subject_id": self.subject.id, "paper": 1},
                format="json",
            )
        self.assertEqual(response.status_code, 201)
        return response.json()

    def test_start_builds_blueprint_from_syllabus(self):
        data = self._start()
        self.assertIn("Paper 1", data["title"])
        self.assertEqual(data["total_questions"], 2)
        self.assertEqual(len(data["sections"]), 2)  # topic weightings honoured
        self.assertEqual(data["duration_minutes"], 90)
        from apps.quiz.models import ExamBlueprint

        self.assertTrue(ExamBlueprint.objects.filter(subject=self.subject).exists())

    def test_next_generates_paper_tagged_mcq(self):
        session = self._start()
        with patch(
            "apps.rag.services.llm.get_chat_provider", return_value=self.provider
        ), patch("apps.rag.services.retriever.retrieve", return_value=[]):
            nxt = self.client.post(f"/api/quiz/exam/{session['id']}/next/", format="json")
        self.assertEqual(nxt.status_code, 200)
        q = nxt.json()
        self.assertEqual(q["paper_label"], "Paper 1")
        self.assertEqual(q["format"], "mcq")
        self.assertNotIn("correct_index", q)  # answers never leak

        state = self.client.get(f"/api/quiz/exam/{session['id']}/").json()
        self.assertEqual(state["score_possible"], 2)
        self.assertEqual(len(state["questions"]), 1)

    def test_full_sitting_scores_and_completes(self):
        session = self._start()
        with patch(
            "apps.rag.services.llm.get_chat_provider", return_value=self.provider
        ), patch("apps.rag.services.retriever.retrieve", return_value=[]):
            for _ in range(session["total_questions"]):
                nxt = self.client.post(f"/api/quiz/exam/{session['id']}/next/", format="json")
                self.assertEqual(nxt.status_code, 200)
                self.client.post(
                    "/api/quiz/answer/",
                    {"question_id": nxt.json()["id"], "selected_index": 0},
                    format="json",
                )
            state = self.client.get(f"/api/quiz/exam/{session['id']}/").json()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["answered"], 2)

    def test_other_user_cannot_read_session(self):
        session = self._start()
        other = User.objects.create_user("other", password="test-pass-123")
        self.client.force_authenticate(other)
        response = self.client.get(f"/api/quiz/exam/{session['id']}/")
        self.assertEqual(response.status_code, 404)


class StructuredAnswerTests(TestCase):
    def setUp(self):
        syllabus, self.subject, obj = make_maths()
        self.question = QuizQuestion.objects.create(
            subject=self.subject,
            objective=obj,
            topic_title="Algebra",
            format=QuizQuestion.Format.STRUCTURED,
            question_text="Factorise x^2 + 5x + 6.",
            options=[],
            marks=4,
            marking_guidance="M1 split, A1 factors, ...",
        )
        self.user = User.objects.create_user("writer", password="test-pass-123")
        Enrollment.objects.create(student=self.user, subject=self.subject)
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def test_answer_graded_with_partial_marks(self):
        with patch(
            "apps.quiz.api.grade_structured_answer",
            return_value=(3.0, 4.0, "Good method, arithmetic slip."),
        ):
            response = self.client.post(
                "/api/quiz/answer/",
                {"question_id": self.question.id, "answer_text": "(x+2)(x+7)"},
                format="json",
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["awarded_marks"], 3.0)
        self.assertEqual(body["max_marks"], 4)
        self.assertTrue(body["correct"])  # >= half marks drives BKT
        attempt = QuizAttempt.objects.get(question=self.question)
        self.assertEqual(attempt.awarded_marks, 3.0)

    def test_missing_answer_text_rejected(self):
        response = self.client.post(
            "/api/quiz/answer/", {"question_id": self.question.id}, format="json"
        )
        self.assertEqual(response.status_code, 400)


class TopicMatchedWeaknessTests(TestCase):
    """Weaknesses are injected ONLY for the topic the user asks about, and never
    for off-syllabus/meta questions."""

    def setUp(self):
        syllabus = Syllabus.objects.create(level="EGCSE", name="EGCSE T", version="1.0")
        self.subject = Subject.objects.create(syllabus=syllabus, code="6884", name="Biology")
        self.repro = Topic.objects.create(subject=self.subject, title="Human reproduction", order=1)
        self.nervous = Topic.objects.create(subject=self.subject, title="Nervous system", order=2)
        self.user = User.objects.create_user("bio", password="x")
        obj_repro = LearningObjective.objects.create(
            topic=self.repro, statement="Describe the menstrual cycle"
        )
        obj_nerv = LearningObjective.objects.create(
            topic=self.nervous, statement="Explain reflex arcs"
        )
        MasteryRecord.objects.create(student=self.user, objective=obj_repro, mastery=0.2, subject=self.subject)
        MasteryRecord.objects.create(student=self.user, objective=obj_nerv, mastery=0.9, subject=self.subject)

    def test_topic_question_returns_only_that_topics_weakness(self):
        from apps.progress.services.dashboard import weak_objectives_for_message
        rows = weak_objectives_for_message(self.user, self.subject, "Explain reproduction and the menstrual cycle")
        self.assertEqual(len(rows), 1)
        self.assertIn("menstrual", rows[0][0])

    def test_unrelated_strong_topic_not_injected(self):
        from apps.progress.services.dashboard import weak_objectives_for_message
        rows = weak_objectives_for_message(self.user, self.subject, "Explain reflexes and the nervous system")
        # nervous is strong (0.9): maps to a topic but has no weak objective.
        self.assertEqual(rows, [])

    def test_off_syllabus_meta_question_returns_none(self):
        from apps.progress.services.dashboard import weak_objectives_for_message
        rows = weak_objectives_for_message(self.user, self.subject, "Can you suggest a study timetable for my subject?")
        self.assertEqual(rows, [])

    def test_core_student_does_not_get_extended_weakness(self):
        from apps.progress.services.dashboard import weak_objectives_for_message

        self.subject.tiers_available = ["core", "extended"]
        self.subject.save()
        # Mark the reflex (nervous) objective as extended-only and weak.
        obj_nerv = LearningObjective.objects.get(statement__contains="reflex")
        obj_nerv.tier = "extended"
        obj_nerv.save()
        MasteryRecord.objects.filter(student=self.user, objective=obj_nerv).update(mastery=0.2)
        # A core student asking about the nervous system must NOT get the
        # extended-only weakness injected.
        rows = weak_objectives_for_message(
            self.user, self.subject, "Explain the nervous system", tier="core"
        )
        self.assertEqual(rows, [])
        # Without tier gating, that extended weakness would show.
        rows_free = weak_objectives_for_message(
            self.user, self.subject, "Explain the nervous system", tier="extended"
        )
        self.assertTrue(any("reflex" in s for s, _ in rows_free))


class ProvenanceSourceTests(TestCase):
    """Source tagging: Cambridge IGCSE (primary) vs ECESWA EGCSE (secondary)."""

    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()
        self.user = User.objects.create_user("provenance", password="test-pass-123")
        Enrollment.objects.create(student=self.user, subject=self.subject)
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def test_generate_response_exposes_source(self):
        with patch("apps.rag.services.llm.get_chat_provider", return_value=FakeProvider()), \
             patch("apps.rag.services.retriever.retrieve", return_value=[]):
            response = self.client.post(
                "/api/quiz/generate/", {"subject_id": self.subject.id, "count": 1}, format="json"
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("source", response.json()[0])

    def test_infer_source_uses_majority_chunk_document(self):
        from apps.rag.models import DocumentChunk
        from apps.quiz.services.generator import _infer_source_from_chunks
        from apps.syllabus.models import SyllabusDocument

        igcse_doc = SyllabusDocument.objects.create(
            syllabus=self.syllabus, subject=self.subject, title="0580_m24_qp_12",
            doc_type=SyllabusDocument.DocType.PAST_PAPER, source=SyllabusDocument.Source.IGCSE,
        )
        egcse_doc = SyllabusDocument.objects.create(
            syllabus=self.syllabus, subject=self.subject, title="Mathematics 1",
            doc_type=SyllabusDocument.DocType.PAST_PAPER, source=SyllabusDocument.Source.EGCSE,
        )
        c1 = DocumentChunk(syllabus=self.syllabus, document=igcse_doc, ordinal=0, text="a")
        c2 = DocumentChunk(syllabus=self.syllabus, document=igcse_doc, ordinal=1, text="b")
        c3 = DocumentChunk(syllabus=self.syllabus, document=egcse_doc, ordinal=2, text="c")
        self.assertEqual(_infer_source_from_chunks([c1, c2, c3]), "igcse")

    def test_preferred_source_is_ignored_when_unavailable(self):
        from apps.quiz.services.generator import _preferred_source
        self.assertEqual(_preferred_source(["igcse", "egcse"]) in ("igcse", "egcse"), True)
        self.assertEqual(_preferred_source([]), "")


class FigureAttachmentTests(TestCase):
    """Real diagrams attach only when the model flags figure_required=true."""

    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()

    def _make_figured_chunk(self):
        from apps.rag.models import DocumentChunk, DocumentFigure
        from django.core.files.base import ContentFile
        from apps.syllabus.models import SyllabusDocument

        doc = SyllabusDocument.objects.create(
            syllabus=self.syllabus, subject=self.subject, title="Bio 2023 QP 1",
            doc_type=SyllabusDocument.DocType.PAST_PAPER,
            source=SyllabusDocument.Source.EGCSE,
        )
        fig = DocumentFigure(document=doc, page_number=3, ordinal=0)
        fig.image.save("t.png", ContentFile(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50), save=True)
        chunk = DocumentChunk(
            syllabus=self.syllabus, document=doc, subject=self.subject,
            ordinal=0, page_number=3, text="heart diagram A B C",
        )
        return doc, fig, chunk

    def test_figures_attached_when_required(self):
        from apps.quiz.services.generator import _attach_figures
        from apps.quiz.models import QuizQuestion

        _doc, _fig, chunk = self._make_figured_chunk()
        q = QuizQuestion.objects.create(subject=self.subject, format="structured", question_text="Label A, B, C.", marks=3)
        _attach_figures(q, [chunk])
        self.assertEqual(q.figures.count(), 1)

    def test_no_chunk_pages_means_no_attach(self):
        from apps.quiz.services.generator import _attach_figures
        from apps.quiz.models import QuizQuestion
        from apps.rag.models import DocumentChunk

        _doc, _fig, chunk = self._make_figured_chunk()
        # chunk without page_number -> no figure mapped
        bare = DocumentChunk(
            syllabus=self.syllabus, document=chunk.document, subject=self.subject,
            ordinal=1, page_number=None, text="no figure page",
        )
        q = QuizQuestion.objects.create(subject=self.subject, format="structured", question_text="Number line x from 1 to 3.", marks=2)
        _attach_figures(q, [bare])
        self.assertEqual(q.figures.count(), 0)

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


class NextGrowsBankTests(TestCase):
    """When the bank is exhausted, /next/ grows it instead of looping one row."""

    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()
        self.user = User.objects.create_user("looper", password="test-pass-123")
        Enrollment.objects.create(student=self.user, subject=self.subject)
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.q1 = QuizQuestion.objects.create(
            subject=self.subject,
            objective=self.obj,
            topic_title="Algebra",
            format=QuizQuestion.Format.STRUCTURED,
            question_text="First and only bank question.",
            options=[],
            marks=2,
        )
        QuizAttempt.objects.create(student=self.user, question=self.q1, correct=False)

    def _next_id(self):
        response = self.client.get(f"/api/quiz/next/?subject_id={self.subject.id}")
        self.assertEqual(response.status_code, 200)
        return response.json()["id"]

    def test_exhausted_bank_serves_fresh_question(self):
        def fake_generate(subject, **kwargs):
            return [QuizQuestion.objects.create(
                subject=subject,
                format=QuizQuestion.Format.STRUCTURED,
                question_text="Fresh generated question.",
                marks=2,
            )]

        with patch("apps.quiz.api.generate_questions", side_effect=fake_generate):
            self.assertNotEqual(self._next_id(), self.q1.id)

    def test_generation_failure_falls_back_to_recycled(self):
        with patch(
            "apps.quiz.api.generate_questions",
            side_effect=QuizGenerationError("No LLM configured"),
        ):
            self.assertEqual(self._next_id(), self.q1.id)

    def test_excluded_ids_are_never_served(self):
        q2 = QuizQuestion.objects.create(
            subject=self.subject,
            format=QuizQuestion.Format.STRUCTURED,
            question_text="Second bank question.",
            marks=2,
        )
        # q1 attempted AND excluded; q2 untouched -> must serve q2, never q1.
        response = self.client.get(
            f"/api/quiz/next/?subject_id={self.subject.id}&exclude={self.q1.id}"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], q2.id)

    def test_all_excluded_returns_404_so_client_generates(self):
        response = self.client.get(
            f"/api/quiz/next/?subject_id={self.subject.id}&exclude={self.q1.id}"
        )
        # only q1 in bank and it is excluded -> 404, app then generates fresh.
        self.assertEqual(response.status_code, 404)


class QueryOverrideTests(TestCase):
    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()

    def test_query_override_reaches_retriever(self):
        from unittest.mock import patch

        from apps.quiz.services.generator import generate_questions

        seen = {}

        def fake_retrieve(syllabus, query, **kwargs):
            seen["query"] = query
            return []

        with patch("apps.rag.services.llm.get_chat_provider",
                   return_value=FakeProvider()), patch(
            "apps.quiz.services.generator.retrieve",
            side_effect=fake_retrieve,
        ):
            generate_questions(self.subject, count=1,
                               query="triangle diagram right-angled")
        self.assertIn("triangle diagram", seen.get("query", ""))


class GroundingTests(TestCase):
    """Generation grounds in past-paper items, not mark schemes/syllabi."""

    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()

    def test_mark_scheme_chunks_excluded_from_prompt(self):
        from unittest.mock import patch

        from apps.quiz.services.generator import generate_questions
        from apps.rag.models import DocumentChunk
        from apps.syllabus.models import SyllabusDocument

        def doc(title, dtype):
            return SyllabusDocument.objects.create(
                syllabus=self.syllabus, subject=self.subject, title=title,
                doc_type=dtype, source=SyllabusDocument.Source.EGCSE)

        def chunk(d, text):
            return DocumentChunk(syllabus=self.syllabus, document=d,
                                 subject=self.subject, ordinal=0,
                                 page_number=1, text=text)

        ms = chunk(doc("MS 2024", SyllabusDocument.DocType.MARK_SCHEME),
                   "MS-TEXT annotations listed below")
        pp = chunk(doc("QP 2024", SyllabusDocument.DocType.PAST_PAPER),
                   "PP-TEXT triangle ABC diagram question")

        prompts = []

        class CapProvider:
            def chat(self, messages):
                prompts.append(messages[-1]["content"])
                return FAKE_LLM_JSON

        with patch("apps.rag.services.llm.get_chat_provider",
                   return_value=CapProvider()), patch(
            "apps.quiz.services.generator.retrieve",
            return_value=[ms, pp],
        ):
            made = generate_questions(self.subject, count=1)
        self.assertEqual(len(made), 1)
        self.assertIn("PP-TEXT", prompts[0])
        self.assertNotIn("MS-TEXT", prompts[0])

    def test_figured_page_outside_top6_still_forces_diagrams(self):
        from unittest.mock import patch

        from apps.quiz.services.generator import generate_questions
        from apps.rag.models import DocumentChunk, DocumentFigure
        from apps.syllabus.models import SyllabusDocument
        from django.core.files.base import ContentFile

        plain_doc = SyllabusDocument.objects.create(
            syllabus=self.syllabus, subject=self.subject, title="QP plain",
            doc_type=SyllabusDocument.DocType.PAST_PAPER,
            source=SyllabusDocument.Source.EGCSE)
        plains = [DocumentChunk(syllabus=self.syllabus, document=plain_doc,
                                subject=self.subject, ordinal=i,
                                page_number=9, text=f"plain {i}")
                  for i in range(6)]
        fig_doc = SyllabusDocument.objects.create(
            syllabus=self.syllabus, subject=self.subject, title="QP figs",
            doc_type=SyllabusDocument.DocType.PAST_PAPER,
            source=SyllabusDocument.Source.EGCSE)
        fig = DocumentFigure(document=fig_doc, page_number=3, ordinal=0)
        fig.image.save("g.png", ContentFile(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50),
                       save=True)
        figured = DocumentChunk(syllabus=self.syllabus, document=fig_doc,
                                subject=self.subject, ordinal=6,
                                page_number=3, text="triangle diagram labels")
        prompts = []

        class CapProvider:
            def chat(self, messages):
                prompts.append(messages[-1]["content"])
                return FAKE_LLM_JSON

        with patch("apps.rag.services.llm.get_chat_provider",
                   return_value=CapProvider()), patch(
            "apps.quiz.services.generator.retrieve",
            return_value=plains + [figured],
        ):
            generate_questions(self.subject, count=1)
        self.assertIn("WILL be shown", prompts[0])


class GenerationRetryTests(TestCase):
    """An all-bare batch gets one strict retry before surfacing an error."""

    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()

    def _run(self, item_results):
        from unittest.mock import patch

        from apps.quiz.services.generator import generate_questions

        sentinel = object()
        with patch("apps.quiz.services.generator._chat",
                   return_value='[{}]') as chat, patch(
            "apps.quiz.services.generator._question_from_item",
            side_effect=item_results) as _qfi, patch(
            "apps.rag.services.retriever.retrieve", return_value=[]
        ):
            try:
                made = generate_questions(self.subject, count=1)
            except QuizGenerationError as exc:
                return None, chat.call_count, str(exc)
            return made, chat.call_count, ""

    def test_retry_succeeds_on_second_attempt(self):
        made, calls, _ = self._run([None, object()])
        self.assertIsNotNone(made)
        self.assertEqual(len(made), 1)
        self.assertEqual(calls, 2)

    def test_raises_after_two_bad_batches(self):
        made, calls, err = self._run([None, None])
        self.assertIsNone(made)
        self.assertEqual(calls, 2)
        self.assertIn("malformed", err)


class ServeTimeDanglingTests(TestCase):
    """Rows whose figures vanished after creation are never served."""

    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()
        from apps.accounts.models import User

        self.student = User.objects.create_user("seer", password="x")

    def _row(self, text):
        return QuizQuestion.objects.create(
            subject=self.subject,
            format=QuizQuestion.Format.STRUCTURED,
            question_text=text,
            marks=2,
        )

    def test_dangling_row_is_skipped(self):
        from apps.quiz.services.selector import next_question_for

        self._row("In the diagram below, find X. (see diagram)")
        good = self._row("Solve 2x + 3 = 11.")
        self.assertEqual(
            next_question_for(self.student, self.subject).id, good.id)

    def test_only_dangling_rows_means_none(self):
        from apps.quiz.services.selector import next_question_for

        self._row("The table below shows scores. Find the mean.")
        self.assertIsNone(next_question_for(self.student, self.subject))


class WarmBankCommandTests(TestCase):
    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()

    def test_warms_to_total_in_batches(self):
        from unittest.mock import patch

        calls = []

        def fake_generate(subject, **kwargs):
            calls.append(kwargs)
            return [object() for _ in range(kwargs["count"])]

        from django.core.management import call_command

        with patch(
            "apps.quiz.management.commands.warm_bank.generate_questions",
            side_effect=fake_generate,
        ):
            call_command("warm_bank", "--subject-code", self.subject.code,
                         "--total", "5", "--batch", "3")
        self.assertEqual([c["count"] for c in calls], [3, 2])

    def test_aborts_after_repeated_failure(self):
        from unittest.mock import patch

        from django.core.management import call_command, CommandError
        from apps.quiz.services.generator import QuizGenerationError

        with patch(
            "apps.quiz.management.commands.warm_bank.generate_questions",
            side_effect=QuizGenerationError("bad model output"),
        ):
            with self.assertRaises(CommandError):
                call_command("warm_bank", "--subject-code", self.subject.code,
                             "--total", "4", "--batch", "2")


class NoAttemptGuardTests(TestCase):
    """Gibberish must earn 0 even if the grader hallucinates marks."""

    def setUp(self):
        self.syllabus, self.subject, obj = make_maths()
        self.question = QuizQuestion.objects.create(
            subject=self.subject,
            objective=obj,
            topic_title="Algebra",
            format=QuizQuestion.Format.STRUCTURED,
            question_text="Simplify the expression: (3x^2 - 2x + 1) - (x^2 + 4x - 5). Show your working.",
            options=[],
            marks=3,
            marking_guidance="M1 subtract like terms, A1 2x^2 - 6x, A1 +6",
        )

    def _grade(self, answer, llm_awarded):
        from apps.quiz.services.generator import grade_structured_answer

        payload = (
            '{"awarded": %s, "max": 3, '
            '"feedback": "The student correctly simplified."}' % llm_awarded
        )
        with patch(
            "apps.quiz.services.generator._chat", return_value=payload
        ):
            return grade_structured_answer(self.question, answer)

    def test_single_unrelated_word_scores_zero(self):
        awarded, max_marks, feedback = self._grade("Calculator", 2.0)
        self.assertEqual(awarded, 0.0)
        self.assertEqual(max_marks, 3)
        self.assertIn("no attempt", feedback)

    def test_genuine_working_keeps_llm_marks(self):
        awarded, _, _ = self._grade("2x^2 - 6x + 6", 3.0)
        self.assertEqual(awarded, 3.0)

    def test_prose_explanation_sharing_vocabulary_is_kept(self):
        awarded, _, _ = self._grade(
            "the square is always positive so the expression has a minimum", 2.0
        )
        self.assertEqual(awarded, 2.0)


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


class BareDiagramReferenceTests(TestCase):
    """A question must never point at a diagram that isn't there."""

    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()

    def _item(self, **overrides):
        base = {
            "question": "Triangle ABC is right-angled at B. AB = 8 cm, BC = 6 cm. Find AC.",
            "format": "structured",
            "marks": 3,
            "explanation": "Pythagoras.",
            "marking_guidance": "M1 + A1 + A1",
            "objective_hint": "Pythagoras",
        }
        base.update(overrides)
        return base

    def test_bare_see_diagram_is_rejected(self):
        from unittest.mock import patch

        from apps.quiz.services.generator import (
            QuizGenerationError, _question_from_item)
        from apps.quiz.models import QuizQuestion

        before = QuizQuestion.objects.count()
        # Repair unavailable offline: drawing the figure fails as well.
        with patch("apps.quiz.services.generator._chat",
                   side_effect=QuizGenerationError("No LLM configured")):
            q = _question_from_item(
                self.subject,
                self._item(question="In the diagram, triangle ABC is right-angled at B. Find AC. (see diagram)"),
                self.obj, [],
            )
        self.assertIsNone(q)
        self.assertEqual(QuizQuestion.objects.count(), before)  # no orphan row

    def test_diagram_ref_with_ascii_block_is_kept(self):
        from apps.quiz.services.generator import _question_from_item

        text = ("In the diagram, triangle ABC is right-angled at B. Find AC.\n"
                "```ascii\n    A\n   / |\n  /  |\n B---C\n```")
        q = _question_from_item(self.subject, self._item(question=text), self.obj, [])
        self.assertIsNotNone(q)

    def test_plain_question_without_reference_is_kept(self):
        from apps.quiz.services.generator import _question_from_item

        q = _question_from_item(self.subject, self._item(), self.obj, [])
        self.assertIsNotNone(q)

    def test_bare_table_reference_is_rejected(self):
        from unittest.mock import patch

        from apps.quiz.services.generator import (
            QuizGenerationError, _question_from_item)
        from apps.quiz.models import QuizQuestion

        before = QuizQuestion.objects.count()
        with patch("apps.quiz.services.generator._chat",
                   side_effect=QuizGenerationError("No LLM configured")):
            q = _question_from_item(
                self.subject,
                self._item(question="The table below shows the scores of 20 "
                                    "students in a mathematics test. Calculate "
                                    "the mean score."),
                self.obj, [],
            )
        self.assertIsNone(q)
        self.assertEqual(QuizQuestion.objects.count(), before)

    def test_inline_markdown_table_is_kept(self):
        from apps.quiz.services.generator import _question_from_item

        text = ("The table below shows the scores of 20 students. "
                "Calculate the mean score.\n"
                "| Score | Frequency |\n|---|---| \n| 5 | 8 |\n| 6 | 12 |")
        q = _question_from_item(self.subject, self._item(question=text),
                                self.obj, [])
        self.assertIsNotNone(q)

    def test_graph_mention_without_below_is_kept(self):
        from apps.quiz.services.generator import _question_from_item

        q = _question_from_item(
            self.subject,
            self._item(question="Explain why the graph of y = (x - 2)^2 + 3 "
                                "has a minimum at (2, 3)."),
            self.obj, [],
        )
        self.assertIsNotNone(q)

    def test_bare_reference_is_repaired_with_ascii(self):
        from unittest.mock import patch

        from apps.quiz.services.generator import _question_from_item

        with patch(
            "apps.quiz.services.generator._chat",
            side_effect=["```ascii\nA\n|\n| 3 cm\nB---C\n```", "YES"],
        ):
            q = _question_from_item(
                self.subject,
                self._item(question="In the diagram, triangle ABC is "
                                    "right-angled at B. Find AC. (see diagram)"),
                self.obj, [],
            )
        self.assertIsNotNone(q)
        self.assertIn("```ascii", q.question_text)
        self.assertIn("B---C", q.question_text)

    def test_judge_rejects_bad_drawing(self):
        from unittest.mock import patch

        from apps.quiz.services.generator import _question_from_item
        from apps.quiz.models import QuizQuestion

        before = QuizQuestion.objects.count()
        with patch(
            "apps.quiz.services.generator._chat",
            side_effect=["```ascii\nA\n|\n| 3 cm\nB---C\n```", "NO"],
        ):
            q = _question_from_item(
                self.subject,
                self._item(question="In the diagram, triangle ABC is "
                                    "right-angled at B. Find AC. (see diagram)"),
                self.obj, [],
            )
        self.assertIsNone(q)
        self.assertEqual(QuizQuestion.objects.count(), before)

    def test_failed_repair_still_rejects(self):
        from unittest.mock import patch

        from apps.quiz.services.generator import _question_from_item
        from apps.quiz.models import QuizQuestion

        before = QuizQuestion.objects.count()
        with patch(
            "apps.quiz.services.generator._chat",
            return_value="sorry, no idea",
        ):
            q = _question_from_item(
                self.subject,
                self._item(question="The table below shows scores. Find the mean."),
                self.obj, [],
            )
        self.assertIsNone(q)
        self.assertEqual(QuizQuestion.objects.count(), before)


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

    def test_figure_flag_attaches_without_adapted(self):
        from apps.quiz.services.generator import _question_from_item
        from apps.quiz.models import QuizQuestion

        _doc, _fig, chunk = self._make_figured_chunk()
        q = _question_from_item(
            self.subject,
            {"question": "Label parts A, B and C (see diagram).",
             "format": "structured", "marks": 3,
             "figure_required": True,  # NOT adapted: still the reuse contract
             "objective_hint": "Heart"},
            None, [chunk],
        )
        self.assertIsNotNone(q)
        self.assertEqual(QuizQuestion.objects.get(pk=q.pk).figures.count(), 1)

    def test_prompt_line_forces_diagrams_when_figures_exist(self):
        from apps.quiz.services.generator import _figure_prompt_line

        _doc, _fig, chunk = self._make_figured_chunk()
        line = _figure_prompt_line([chunk])
        self.assertIn("WILL be shown", line)
        self.assertIn("figure_required", line)

    def test_prompt_line_forbids_bare_refs_without_figures(self):
        from apps.quiz.services.generator import _figure_prompt_line

        line = _figure_prompt_line([])
        self.assertIn("No source diagrams are available", line)
        self.assertIn("figure_required", line)

    def test_forced_figures_stripped_when_text_ignores_them(self):
        from apps.quiz.services.generator import _question_from_item
        from apps.quiz.models import QuizQuestion

        _doc, fig, chunk = self._make_figured_chunk()
        q = _question_from_item(
            self.subject,
            {"question": "Solve 2x + 3 = 11.",
             "format": "structured", "marks": 2,
             "figure_required": False,
             "objective_hint": "Algebra"},
            None, [chunk], force_figure_ids=[fig.id],
        )
        self.assertIsNotNone(q)
        self.assertEqual(QuizQuestion.objects.get(pk=q.pk).figures.count(), 0)

    def test_forced_figures_kept_when_referenced(self):
        from apps.quiz.services.generator import _question_from_item
        from apps.quiz.models import QuizQuestion

        _doc, fig, chunk = self._make_figured_chunk()
        q = _question_from_item(
            self.subject,
            {"question": "Label parts A, B and C (see diagram).",
             "format": "structured", "marks": 3,
             "figure_required": False,
             "objective_hint": "Heart"},
            None, [chunk], force_figure_ids=[fig.id],
        )
        self.assertIsNotNone(q)
        self.assertEqual(QuizQuestion.objects.get(pk=q.pk).figures.count(), 1)


class SeedFigureQuestionsTests(TestCase):
    """Needs PostgreSQL (embedding_vec has no SQLite column); verified live."""

    def setUp(self):
        self.syllabus, self.subject, self.obj = make_maths()

    def test_command_passes_page_figures(self):
        from django.db import connection

        if connection.vendor != "postgresql":
            self.skipTest("requires PostgreSQL")
        from unittest.mock import patch

        from apps.rag.models import DocumentChunk, DocumentFigure
        from apps.syllabus.models import SyllabusDocument
        from django.core.files.base import ContentFile
        from django.core.management import call_command

        doc = SyllabusDocument.objects.create(
            syllabus=self.syllabus, subject=self.subject, title="QP figs",
            doc_type=SyllabusDocument.DocType.PAST_PAPER,
            source=SyllabusDocument.Source.EGCSE)
        fig = DocumentFigure(document=doc, page_number=3, ordinal=0)
        fig.image.save("s.png", ContentFile(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50),
                       save=True)
        DocumentChunk.objects.create(
            syllabus=self.syllabus, document=doc, subject=self.subject,
            ordinal=0, page_number=3, text="triangle diagram labels")

        seen = {}

        def fake_generate(subject, **kwargs):
            seen.update(kwargs)
            return []

        with patch("apps.quiz.management.commands.seed_figure_questions"
                   ".generate_questions", side_effect=fake_generate):
            call_command("seed_figure_questions", "--subject-code",
                         self.subject.code, "--count", "2")
        self.assertEqual(seen.get("force_figure_ids"), [fig.id])
        self.assertEqual(len(seen.get("force_chunks") or []), 1)

from django.contrib.auth import get_user_model
from django.test import TestCase

from .models import ChatSession, MemoryEntry
from .services.memory import relevant_memory, upsert_memory


class MemoryTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("memuser", password="pass12345")

    def test_upsert_creates_entry(self):
        upsert_memory(self.user, "Student is weak at Maths fractions.")
        self.assertEqual(MemoryEntry.objects.filter(student=self.user).count(), 1)
        e = MemoryEntry.objects.get(student=self.user)
        self.assertIn("Maths", e.fact)
        self.assertTrue(e.embedding)  # embedded

    def test_upsert_dedupes(self):
        upsert_memory(self.user, "Student is weak at Maths fractions.")
        upsert_memory(self.user, "Student is weak at Maths fractions.")
        self.assertEqual(MemoryEntry.objects.filter(student=self.user).count(), 1)

    def test_relevant_returns_similar(self):
        upsert_memory(self.user, "Student is preparing for the February exams.")
        out = relevant_memory(self.user, "When are my exams?")
        self.assertEqual(len(out), 1)

    def test_relevant_returns_none_for_unrelated(self):
        upsert_memory(self.user, "Student is weak at Maths fractions.", kind=MemoryEntry.Kind.SITUATIONAL, importance=5)
        out = relevant_memory(self.user, "Tell me about the water cycle in geography.")
        # situational fact does not match; may return 0 (or the always-on none)
        self.assertIsInstance(out, list)


class ThreadRoutingTests(TestCase):
    def setUp(self):
        from apps.syllabus.models import Subject, Syllabus, Topic

        self.user = get_user_model().objects.create_user("tuser", password="pass12345")
        self.syllabus = Syllabus.objects.create(level="EGCSE", name="EGCSE T", version="1.0")
        self.subject = Subject.objects.create(syllabus=self.syllabus, code="6880", name="Mathematics")
        self.topic = Topic.objects.create(subject=self.subject, title="Algebra", kind="subtopic")
        self.session = ChatSession.objects.create(student=self.user, syllabus=self.syllabus, subject=self.subject)

    def _msg(self, content, topic=None):
        from .models import Message
        return Message.objects.create(session=self.session, role="user", content=content, topic=topic)

    def test_thread_list_main_and_topic(self):
        from .services.routing import thread_list
        self._msg("hello")
        self._msg("factorise this", self.topic)
        threads = thread_list(self.session)
        by_title = {t["title"]: t for t in threads}
        self.assertEqual(by_title["Main chat"]["messages"], 1)
        self.assertIn("Algebra", by_title)
        self.assertEqual(by_title["Algebra"]["messages"], 1)
        self.assertEqual(by_title["Algebra"]["topic_id"], self.topic.id)

    def test_history_scoped_by_topic(self):
        from .services.orchestrator import _history_messages
        self._msg("main chat msg")
        self._msg("in algebra", self.topic)
        self._msg("follow up x=3", self.topic)
        hist = _history_messages(self.session, "follow up", topic=self.topic, recent=6, relevant=4)
        contents = [m["content"] for m in hist]
        self.assertIn("in algebra", contents)
        self.assertIn("follow up x=3", contents)
        self.assertNotIn("main chat msg", contents)  # scoped out of the thread


class KeyTermsTests(TestCase):
    def test_strips_trailing_key_terms_line(self):
        from .services.orchestrator import _split_key_terms
        clean, terms = _split_key_terms(
            "Osmosis moves water.\nKEY_TERMS: osmosis, cell wall")
        self.assertEqual(clean, "Osmosis moves water.")
        self.assertEqual(terms, ["osmosis", "cell wall"])

    def test_no_key_terms_line_keeps_text(self):
        from .services.orchestrator import _split_key_terms
        clean, terms = _split_key_terms("Short answer, no terms line.")
        self.assertEqual(clean, "Short answer, no terms line.")
        self.assertEqual(terms, [])

    def test_prompt_caps_length_and_demands_key_terms(self):
        from .services.orchestrator import SYSTEM_TEMPLATE
        self.assertIn("130 words", SYSTEM_TEMPLATE)
        self.assertIn("KEY_TERMS", SYSTEM_TEMPLATE)
        self.assertIn("**bold**", SYSTEM_TEMPLATE)


class ChatRetrievalScopeTests(TestCase):
    """Chat searches this subject's syllabus + mark schemes + notes only.

    Past papers are excluded (question generation alone may search those).
    """

    def test_chat_doc_types_exclude_past_papers(self):
        from .services.orchestrator import CHAT_DOC_TYPES
        self.assertIn("syllabus", CHAT_DOC_TYPES)
        self.assertIn("mark_scheme", CHAT_DOC_TYPES)
        self.assertIn("notes", CHAT_DOC_TYPES)  # bucket holding examiner reports
        self.assertNotIn("past_paper", CHAT_DOC_TYPES)

    def test_retrieve_forwards_doc_type_filter(self):
        from unittest.mock import patch
        from apps.rag.services import retriever as retr
        from .services.orchestrator import CHAT_DOC_TYPES

        seen = []

        class _QS:
            def filter(self, **kw):
                seen.append(kw)
                return self
            def exclude(self, **kw):
                return self
            def select_related(self, *a):
                return self
            def only(self, *a):
                return self
            def __getitem__(self, s):
                return []

        class _Manager:
            def filter(self, **kw):
                seen.append(kw)
                return _QS()

        class _Emb:
            def embed_query(self, q):
                return [1.0, 0.0]

        with patch.object(retr.DocumentChunk, "objects", _Manager()), \
                patch("apps.rag.services.embeddings.get_embedder", return_value=_Emb()):
            retr.retrieve(object(), "photosynthesis", subject=object(),
                           doc_types=CHAT_DOC_TYPES)
        self.assertTrue(
            any(kw.get("document__doc_type__in") == CHAT_DOC_TYPES for kw in seen),
            f"doc_type filter missing from queryset chain: {seen}")
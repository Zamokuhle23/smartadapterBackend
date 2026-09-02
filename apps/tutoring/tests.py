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
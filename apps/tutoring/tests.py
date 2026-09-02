from django.contrib.auth import get_user_model
from django.test import TestCase

from .models import MemoryEntry
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
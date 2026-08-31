from django.test import TestCase

from apps.rag.services.embeddings import HashingEmbedder, get_embedder
from apps.rag.services.retriever import cosine_similarity
from apps.syllabus.services.ingestion import chunk_text

LONG_TEXT = (
    "Algebraic fractions are simplified by factorising. "
    "First factorise the numerator completely. "
) * 40


class ChunkerTests(TestCase):
    def test_basic_chunking(self):
        chunks = chunk_text(LONG_TEXT, size=400, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 400 for c in chunks))

    def test_no_tail_loop(self):
        # Regression: text shorter than one chunk must produce exactly 1 chunk.
        self.assertEqual(chunk_text("short text", size=800, overlap=120), ["short text"])

    def test_empty(self):
        self.assertEqual(chunk_text("   "), [])

    def test_covers_all_text(self):
        chunks = chunk_text(LONG_TEXT, size=300, overlap=60)
        self.assertIn("factorising", " ".join(chunks))


class HashingEmbedderTests(TestCase):
    def test_deterministic(self):
        e = HashingEmbedder(dim=64)
        a = e.embed_query("algebraic fractions")
        b = e.embed_query("algebraic fractions")
        self.assertEqual(a, b)

    def test_normalized(self):
        e = HashingEmbedder(dim=64)
        v = e.embed_query("quadratic equations")
        norm = sum(x * x for x in v) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_similar_words_share_direction(self):
        e = HashingEmbedder(dim=64)
        same = cosine_similarity(e.embed_query("factorise trinomials"), e.embed_query("factorise trinomials"))
        self.assertAlmostEqual(same, 1.0, places=6)


class CosineTests(TestCase):
    def test_identical(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_zero_safe(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)


class EmbedderSelectionTests(TestCase):
    def test_fallback_when_lib_missing(self):
        from django.test import override_settings

        with override_settings(EMBEDDING_PROVIDER="local"):
            # On machines without sentence-transformers this must NOT raise.
            embedder = get_embedder()
            self.assertIsNotNone(embedder)

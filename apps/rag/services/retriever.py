"""
Retrieval service: semantic search over DocumentChunk rows of one syllabus.

On PostgreSQL (+pgvector) it uses the `embedding_vec` column and an HNSW index
for fast KNN cosine search. On SQLite (dev) it falls back to the portable JSON
`embedding` column, ranked with cosine similarity in Python.
"""

import math

from django.conf import settings
from django.db import connection

from ..models import DocumentChunk


def cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _pgvector_retrieve(syllabus, qvec, k, subject=None):
    """SQL KNN search over the pgvector column (PostgreSQL only)."""
    from pgvector.django import CosineDistance

    qs = DocumentChunk.objects.filter(syllabus=syllabus, embedding_vec__isnull=False)
    if subject is not None:
        qs = qs.filter(subject=subject)
    return list(
        qs.select_related("document")
        .only(
            "id",
            "ordinal",
            "page_number",
            "text",
            "embedding_vec",
            "subject_id",
            "document__title",
            "document__doc_type",
            "document__paper_number",
            "document__year",
            "document__source",
        )
        .order_by(CosineDistance("embedding_vec", qvec))[:k]
    )


def retrieve(syllabus, query: str, k: int | None = None, subject=None):
    """
    Return up to k chunks from `syllabus` most similar to `query`,
    optionally restricted to one subject. Returns list[DocumentChunk].

    Uses pgvector SQL search on PostgreSQL, Python cosine on SQLite.
    """
    from .embeddings import get_embedder

    top_k = k or settings.RAG_TOP_K
    qvec = get_embedder().embed_query(query)

    if connection.vendor == "postgresql":
        return _pgvector_retrieve(syllabus, qvec, top_k, subject)

    # SQLite / portable path.
    qs = DocumentChunk.objects.filter(syllabus=syllabus).exclude(embedding__isnull=True)
    if subject is not None:
        qs = qs.filter(subject=subject)
    chunks = list(
        qs.select_related("document")
        .only(
            "id",
            "ordinal",
            "page_number",
            "text",
            "embedding",
            "subject_id",
            "document__title",
            "document__doc_type",
            "document__paper_number",
            "document__year",
            "document__source",
        )[:5000]
    )
    if not chunks:
        return []
    # Dimension guard: skip chunks embedded by a different model/space.
    chunks = [c for c in chunks if c.embedding and len(c.embedding) == len(qvec)]
    if not chunks:
        return []
    scored = [(cosine_similarity(qvec, c.embedding), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]

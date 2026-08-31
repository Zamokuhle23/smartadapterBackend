"""
Retrieval service: semantic search over DocumentChunk rows of one syllabus.

Production path (Postgres + pgvector): add a VectorField column and use
`embedding <-> query_vec` KNN ordering - see the TODO at the bottom.
Current implementation loads candidate chunks for the syllabus and ranks
them with cosine similarity in Python (exact, portable, fine for MVP scale).
"""

import math

from django.conf import settings

from ..models import DocumentChunk


def cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def retrieve(syllabus, query: str, k: int | None = None, subject=None):
    """
    Return up to k chunks from `syllabus` most similar to `query`,
    optionally restricted to one subject. Returns list[DocumentChunk].

    Chunks carry their source document (select_related) so callers can
    distinguish past-paper items from syllabus/notes text.
    """
    from .embeddings import get_embedder

    top_k = k or settings.RAG_TOP_K
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
    qvec = get_embedder().embed_query(query)
    # Dimension guard: skip chunks embedded by a different model/space
    # (e.g. leftovers from before a model switch) instead of corrupting ranking.
    chunks = [c for c in chunks if c.embedding and len(c.embedding) == len(qvec)]
    if not chunks:
        return []

    scored = [(cosine_similarity(qvec, c.embedding), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]


# TODO(pgvector): on PostgreSQL, migrate to
#   embedding = VectorField(dimensions=settings.EMBEDDING_DIM)
#   index = HnswIndex(fields=["embedding"], m=16, ef_construction=64)
# and replace the ranking above with:
#   qs.annotate(distance=CosineDistance("embedding", qvec)).order_by("distance")[:top_k]

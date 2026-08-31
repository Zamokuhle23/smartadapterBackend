"""
Retrieval quality eval - measures top-k hit rate for the CURRENT embedding provider.

Requires a seeded syllabus with at least one ingested document (smoke_test.py's
sample doc is enough). Golden questions below map a student-style question to a
keyword that MUST appear in a good retrieved chunk.

Usage:
    set EMBEDDING_PROVIDER=local  (or hash)
    python scripts/eval_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from apps.rag.services.embeddings import get_embedder  # noqa: E402
from apps.rag.services.retriever import retrieve  # noqa: E402
from apps.syllabus.models import Subject, Syllabus  # noqa: E402

# question -> keywords that a relevant chunk should contain (any-of)
GOLDEN = [
    ("How do I simplify an algebraic fraction like 6x/9?", ["algebraic fraction", "factorise", "common"]),
    ("What are the steps to factorise a quadratic trinomial?", ["trinomial", "factorise", "middle term"]),
    ("How do I solve a quadratic equation with the formula?", ["quadratic formula", "discriminant", "roots"]),
    ("What does the discriminant tell me?", ["discriminant", "nature", "roots"]),
    ("How do I add two algebraic fractions?", ["common denominator", "add"]),
]


def main():
    syllabus = Syllabus.objects.filter(level="EGCSE").first()
    if syllabus is None:
        print("No EGCSE syllabus found - run seed_syllabi + smoke_test first.")
        sys.exit(1)
    maths = Subject.objects.filter(syllabus=syllabus, code="6880").first()
    if maths is None or not maths.documents.exists():
        print("No ingested documents for EGCSE Mathematics - run smoke_test.py first.")
        sys.exit(1)

    embedder = get_embedder()
    print(f"Provider: {type(embedder).__name__}")

    hits = 0
    total = 0
    for question, keywords in GOLDEN:
        chunks = retrieve(syllabus, question, k=3, subject=maths)
        joined = " ".join(c.text.lower() for c in chunks)
        hit = any(k in joined for k in keywords)
        hits += int(hit)
        total += 1
        mark = "HIT " if hit else "MISS"
        print(f"  [{mark}] {question[:60]} -> {len(chunks)} chunks")

    print(f"\nTop-3 hit rate: {hits}/{total} = {hits / total:.0%}")


if __name__ == "__main__":
    main()

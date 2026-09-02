"""
Re-embed all stored chunks with the CURRENT embedding provider.

Required after switching EMBEDDING_PROVIDER or EMBEDDING_MODEL_NAME, because a
new model produces a completely different vector space - old vectors would be
meaningless next to new ones.

Usage:
    python manage.py reembed_chunks [--syllabus ID]
"""

from django.core.management.base import BaseCommand

from apps.rag.models import DocumentChunk
from apps.rag.services.embeddings import get_embedder

BATCH = 64


class Command(BaseCommand):
    help = "Re-embed stored chunks with the current embedding provider/model"

    def add_arguments(self, parser):
        parser.add_argument("--syllabus", type=int, default=None)

    def handle(self, *args, **options):
        embedder = get_embedder()
        self.stdout.write(f"Embedder: {type(embedder).__name__}")

        qs = DocumentChunk.objects.exclude(text="").order_by("pk")
        if options["syllabus"]:
            qs = qs.filter(syllabus_id=options["syllabus"])

        total = qs.count()
        if total == 0:
            self.stdout.write(self.style.WARNING("No chunks found."))
            return

        from django.db import connection

        is_postgres = connection.vendor == "postgresql"
        done = 0
        while done < total:
            batch = list(qs[done : done + BATCH])
            vectors = embedder.embed_texts([c.text for c in batch])
            for chunk, vec in zip(batch, vectors):
                chunk.embedding = vec
                if is_postgres:
                    chunk.embedding_vec = vec
            fields = ["embedding"] + (["embedding_vec"] if is_postgres else [])
            DocumentChunk.objects.bulk_update(batch, fields, batch_size=BATCH)
            done += len(batch)
            self.stdout.write(f"  {done}/{total}")

        self.stdout.write(
            self.style.SUCCESS(
                f"Re-embedded {done} chunks."
                + (" pgvector column updated." if is_postgres else " (SQLite: JSON only).")
            )
        )

"""Verify stored chunk embeddings: dimensions + provider sanity."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from apps.rag.models import DocumentChunk  # noqa: E402

chunks = DocumentChunk.objects.filter(embedding__isnull=False)
print(f"Embedded chunks in DB: {chunks.count()}")
for c in chunks[:3]:
    print(f"  chunk {c.pk}: dims={len(c.embedding)} sample={[round(x, 4) for x in c.embedding[:3]]}")

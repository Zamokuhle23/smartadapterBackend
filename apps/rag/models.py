from django.db import models

from apps.syllabus.models import Subject, Syllabus, SyllabusDocument, Topic
from pgvector.django import VectorField


class DocumentChunk(models.Model):
    """
    One retrievable passage of a syllabus document, with its embedding.

    `embedding` is a portable JSON list of floats so development works on SQLite
    too. On PostgreSQL a real pgvector `embedding_vec` column is available for
    fast HNSW KNN search - see apps/rag/services/retriever.py.
    """

    syllabus = models.ForeignKey(Syllabus, on_delete=models.CASCADE, related_name="chunks")
    document = models.ForeignKey(
        SyllabusDocument, null=True, blank=True, on_delete=models.CASCADE, related_name="chunks"
    )
    subject = models.ForeignKey(Subject, null=True, blank=True, on_delete=models.SET_NULL)
    topic = models.ForeignKey(Topic, null=True, blank=True, on_delete=models.SET_NULL)
    ordinal = models.PositiveIntegerField(default=0)
    page_number = models.PositiveSmallIntegerField(null=True, blank=True)  # 1-based PDF page (figures)
    text = models.TextField()
    embedding = models.JSONField(null=True, blank=True)
    # pgvector column (PostgreSQL only). NULL/unused on SQLite; backfilled on
    # Postgres by reembed_chunks so HNSW KNN search works.
    embedding_vec = VectorField(blank=True, null=True)

    class Meta:
        indexes = [models.Index(fields=["syllabus", "subject"])]

    def __str__(self):
        return f"Chunk<{self.pk} doc={self.document_id} ord={self.ordinal}>"


def figure_upload_to(instance, filename: str) -> str:
    """Keyed filenames so a file found anywhere resolves to its paper."""
    key = (instance.stable_key or "").strip() or "unkeyed"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:100]
    return f"figures/{safe}.png"


class DocumentFigure(models.Model):
    """
    An image extracted from a syllabus document's page (e.g. a Biology diagram or
    Physics graph). Generated questions whose source chunk points at this page can
    carry the figure so students see the real diagram, not just a text description.

    Identity rules (see stable_key):
    - physical uniqueness: one figure = one (document, page, position) triple;
    - stable_key is assigned once and reused across re-ingestions, so questions
      keep pointing at the same picture;
    - caption describes WHAT the figure shows (searchable, editable in admin).
    """

    document = models.ForeignKey(
        SyllabusDocument, on_delete=models.CASCADE, related_name="figures"
    )
    page_number = models.PositiveSmallIntegerField(null=True, blank=True)  # 1-based PDF page
    ordinal = models.PositiveSmallIntegerField(default=0)  # figure index within the page
    stable_key = models.CharField(
        max_length=120, unique=True, blank=True, default="",
        help_text="Canonical traceable id, e.g. IGCSE-0580-2024-MJ-P1-p12-f2",
    )
    bbox = models.JSONField(
        null=True, blank=True,
        help_text="Region rect [x0, y0, x1, y1] in PDF points; used to "
                  "match figures across re-ingestions so keys stay stable",
    )
    image = models.ImageField(upload_to=figure_upload_to)
    caption = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ("page_number", "ordinal")
        constraints = [
            models.UniqueConstraint(
                fields=["document", "page_number", "ordinal"],
                name="unique_figure_per_page_position",
            )
        ]

    def __str__(self):
        return self.stable_key or f"Figure<doc={self.document_id} p{self.page_number}:{self.ordinal}>"

    def save(self, *args, **kwargs):
        if not self.stable_key:
            from apps.syllabus.services.figure_keys import figure_key_for

            try:
                base = figure_key_for(self.document, self.page_number,
                                      self.ordinal)
            except Exception:  # noqa: BLE001 - never block a save on keys
                base = (f"FIG-{self.document_id or 0}"
                        f"-p{self.page_number or 0}-f{self.ordinal}")
            key, i = base, 2
            while (DocumentFigure.objects.filter(stable_key=key)
                   .exclude(pk=self.pk).exists()):
                key = f"{base}-{i}"
                i += 1
            self.stable_key = key
        super().save(*args, **kwargs)

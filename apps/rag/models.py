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


class DocumentFigure(models.Model):
    """
    An image extracted from a syllabus document's page (e.g. a Biology diagram or
    Physics graph). Generated questions whose source chunk points at this page can
    carry the figure so students see the real diagram, not just a text description.
    """

    document = models.ForeignKey(
        SyllabusDocument, on_delete=models.CASCADE, related_name="figures"
    )
    page_number = models.PositiveSmallIntegerField(null=True, blank=True)  # 1-based PDF page
    ordinal = models.PositiveSmallIntegerField(default=0)  # figure index within the page
    image = models.ImageField(upload_to="figures/%Y/%m/")
    caption = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ("page_number", "ordinal")

    def __str__(self):
        return f"Figure<doc={self.document_id} p{self.page_number}:{self.ordinal}>"

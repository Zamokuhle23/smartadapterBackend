"""
Seed figure-bearing questions deterministically: pick figured pages from the
subject's past papers, generate one question per page with those figures
pre-attached. The model is told the figures WILL be shown; text still
decides (unreferenced figures are stripped, dangling refs repaired/dropped).

Usage:
    python manage.py seed_figure_questions --subject-code 6880 --count 10
"""

import random

from django.core.management.base import BaseCommand, CommandError

from apps.quiz.services.generator import QuizGenerationError, generate_questions
from apps.rag.models import DocumentChunk, DocumentFigure
from apps.syllabus.models import Subject, SyllabusDocument


class Command(BaseCommand):
    help = "Generate questions bound to real past-paper figures."

    def add_arguments(self, parser):
        parser.add_argument("--subject-code", required=True)
        parser.add_argument("--count", type=int, default=10)

    def handle(self, *args, **options):
        try:
            subject = Subject.objects.get(code=options["subject_code"])
        except Subject.DoesNotExist:
            raise CommandError(
                f"No subject with code {options['subject_code']}")
        from django.db.models import Count

        single_pages = (
            DocumentFigure.objects.filter(
                document__subject=subject,
                document__doc_type=SyllabusDocument.DocType.PAST_PAPER,
            ).values("document_id", "page_number").annotate(
                n=Count("id")).filter(n=1))
        pages = sorted((r["document_id"], r["page_number"]) for r in single_pages)
        # Only single-figure pages: with several figures we cannot tell
        # which one the text will describe.
        if not pages:
            raise CommandError("No figured past-paper pages for this subject")
        random.shuffle(pages)
        created = with_figs = 0
        for document_id, page in pages:
            if created >= options["count"]:
                break
            chunks = list(DocumentChunk.objects.filter(
                document_id=document_id, page_number=page).order_by("ordinal")[:8])
            if not chunks:
                continue
            fig_ids = list(DocumentFigure.objects.filter(
                document_id=document_id, page_number=page
            ).order_by("ordinal").values_list("id", flat=True)[:6])
            try:
                made = generate_questions(
                    subject, count=1,
                    force_chunks=chunks, force_figure_ids=fig_ids)
            except QuizGenerationError as exc:
                self.stderr.write(f"page {page} of doc {document_id}: {exc}")
                continue
            for q in made:
                created += 1
                if q.figures.exists():
                    with_figs += 1
                    self.stdout.write(
                        f"+ Q{q.id} figs="
                        f"{[f.stable_key for f in q.figures.all()[:2]]}")
        self.stdout.write(self.style.SUCCESS(
            f"Seeded {created} questions, {with_figs} carrying figures"))

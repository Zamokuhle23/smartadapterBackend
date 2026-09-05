"""
Detect tappable question/part anchors for one past-paper document.

Idempotent: existing (document, qid) rows are skipped. Sub-parts ((a), (b))
become their own anchors (4a, 4b); questions without parts get one anchor.

Usage:
    python manage.py anchor_paper --doc-id 471 [--max 5]
"""

from django.core.management.base import BaseCommand, CommandError

from apps.quiz.models import QuestionAnchor
from apps.quiz.services.cropper import detect_parts, detect_questions, part_kind
from apps.syllabus.models import SyllabusDocument


class Command(BaseCommand):
    help = "Detect interaction anchors for a past-paper document."

    def add_arguments(self, parser):
        parser.add_argument("--doc-id", type=int, required=True)
        parser.add_argument("--max", type=int, default=0)

    def handle(self, *args, **options):
        try:
            doc = SyllabusDocument.objects.get(pk=options["doc_id"])
        except SyllabusDocument.DoesNotExist:
            raise CommandError(f"No document {options['doc_id']}")
        import pymupdf

        pdf = pymupdf.open(doc.file.path)
        widths = {p.number + 1: float(p.rect.width) for p in pdf}
        made = 0
        for q in detect_questions(pdf):
            if options["max"] and made >= options["max"]:
                break
            made += self._store(pdf, doc, q, widths)
        self.stdout.write(self.style.SUCCESS(
            f"Stored {made} new anchors for {doc.title[:50]}"))

    def _store(self, pdf, doc, q, widths) -> int:
        made = 0
        for pno, top, bottom in q["pages"]:
            page = pdf[pno - 1]
            parts = detect_parts(page, top, bottom)
            if not parts:
                kind = part_kind(page, top, bottom)
                made += self._save(doc, q["number"], pno, top, bottom,
                                   widths[pno], kind,
                                   0.85 if q["confident"] else 0.55)
                continue
            bounds = [p[1] for p in parts] + [bottom]
            for (suffix, start), end in zip(parts, bounds[1:]):
                made += self._save(doc, f"{q['number']}{suffix}", pno,
                                   start, end, widths[pno],
                                   part_kind(page, start, end),
                                   0.8 if q["confident"] else 0.5)
        return made

    @staticmethod
    def _save(doc, qid, pno, top, bottom, width, kind, confidence) -> int:
        if QuestionAnchor.objects.filter(document=doc, qid=qid).exists():
            return 0
        QuestionAnchor.objects.create(
            document=doc, qid=qid, page_number=pno,
            bbox=[0.0, round(top, 1), round(width, 1), round(bottom, 1)],
            kind=kind, confidence=confidence,
            status=(QuestionAnchor.Status.AUTO if confidence >= 0.7
                    else QuestionAnchor.Status.NEEDS_QC),
        )
        return 1

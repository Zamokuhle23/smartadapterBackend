"""
Purge anchors that sit on syllabus pages inside past-paper PDFs.

ECESWA specimen booklets bundle the full syllabus (aims, objectives,
appendices) with the specimen paper in one file. Syllabus objective
numbers ("10.4", "(a) define homeostasis…") seed bogus anchors that then
surface syllabus pages in practice. A anchor is purged only when BOTH hold:
  * its page carries syllabus markers (aims / all-learners / appendix /
    scheme of assessment / %-teaching-time / assessment objectives), AND
  * its own text carries no question markers ([n] marks, Fig., Total,
    answer blanks, candidate instructions).

Real paper pages never carry syllabus markers, so real questions are safe.

Usage:
    python manage.py purge_syllabus_anchors            # dry run
    python manage.py purge_syllabus_anchors --apply    # delete
"""

import re

from django.core.management.base import BaseCommand

from apps.quiz.models import QuestionAnchor
from apps.quiz.services.cropper import anchor_text
from apps.syllabus.models import SyllabusDocument

SYL_MARK = re.compile(
    r"all learners should be able to|the aims of the syllabus|"
    r"scheme of assessment|% of teaching time|assessment objectives|"
    r"appendix \d",
    re.IGNORECASE)
Q_MARK = re.compile(
    r"\[\d{1,2}\]|\[Total|Fig\.|Turn over|CANDIDATE NUMBER|CENTRE NUMBER|"
    r"\.{10,}|_{10,}|2 hours|1 hour",
    re.IGNORECASE)


class Command(BaseCommand):
    help = "Purge syllabus-page anchors from past-paper PDFs."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true")

    def handle(self, *args, **options):
        import pymupdf

        apply = options["apply"]
        docs = SyllabusDocument.objects.filter(
            doc_type=SyllabusDocument.DocType.PAST_PAPER).order_by("id")
        total_bad = total_kept = 0
        for doc in docs:
            anchors = list(QuestionAnchor.objects.filter(
                document=doc).order_by("page_number", "qid"))
            if not anchors:
                continue
            try:
                pdf = pymupdf.open(doc.file.path)
            except Exception:  # noqa: BLE001
                continue
            try:
                pages = {}
                for i, page in enumerate(pdf):
                    pages[i + 1] = page.get_text("text")
            finally:
                pdf.close()
            if not any(SYL_MARK.search(t) for t in pages.values()):
                continue  # no syllabus content anywhere: nothing to do
            bad_ids = []
            for a in anchors:
                page_text = pages.get(a.page_number, "")
                if not SYL_MARK.search(page_text):
                    continue
                try:
                    text = anchor_text(doc.file.path, a.page_number, a.bbox)
                except Exception:  # noqa: BLE001
                    text = ""
                if not Q_MARK.search(text or ""):
                    bad_ids.append(a.id)
            kept = len(anchors) - len(bad_ids)
            total_bad += len(bad_ids)
            total_kept += kept
            if bad_ids:
                self.stdout.write(
                    f"  {'APPLY' if apply else 'PLAN'} doc {doc.id} "
                    f"{doc.title[:45]!r}: purge {len(bad_ids)}, keep {kept}")
                if apply:
                    QuestionAnchor.objects.filter(id__in=bad_ids).delete()
        self.stdout.write(self.style.SUCCESS(
            f"{'Purged' if apply else 'Would purge'} {total_bad} anchors "
            f"({total_kept} kept)."))

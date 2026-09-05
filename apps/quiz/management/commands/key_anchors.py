"""
Resolve grading keys for paper anchors from their mark schemes.

Same linkage as key_crops, but for tappable anchors: the anchor's bbox text
is graded against the mark-scheme excerpt for its question number. Cached on
the anchor forever.

Usage:
    python manage.py key_anchors --subject-code 6880 [--limit 20]
"""

from django.core.management.base import BaseCommand, CommandError

from apps.quiz.models import QuestionAnchor
from apps.quiz.services.cropper import anchor_marks, anchor_text
from apps.quiz.services.generator import (
    extract_keys,
    find_mark_scheme,
    ms_excerpt_for,
)
from apps.syllabus.models import Subject


class Command(BaseCommand):
    help = "Resolve grading keys for paper anchors."

    def add_arguments(self, parser):
        parser.add_argument("--subject-code", required=True)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **options):
        try:
            subject = Subject.objects.get(code=options["subject_code"])
        except Subject.DoesNotExist:
            raise CommandError(
                f"No subject with code {options['subject_code']}")
        from django.db.models import Q

        anchors = list(QuestionAnchor.objects.filter(
            document__subject=subject,
        ).filter(
            Q(marking_guidance="") | Q(marking_guidance__isnull=True),
        ).select_related("document").order_by("id"))
        if options["limit"]:
            anchors = anchors[:options["limit"]]
        done = skipped = 0
        for anchor in anchors:
            if self._resolve(anchor):
                done += 1
            else:
                skipped += 1
        self.stdout.write(self.style.SUCCESS(
            f"Keyed {done} anchors ({skipped} skipped)"))

    def _resolve(self, anchor) -> bool:
        doc = anchor.document
        try:
            text = anchor_text(doc.file.path, anchor.page_number, anchor.bbox)
        except Exception:  # noqa: BLE001
            return False
        if not text.strip():
            return False
        ms = find_mark_scheme(doc.subject, doc.year, doc.paper_number)
        if ms is None:
            return False
        base = "".join(ch for ch in anchor.qid if ch.isdigit())
        for qid in (anchor.qid, base):
            if not qid:
                continue
            excerpt = ms_excerpt_for(ms, qid)
            if not excerpt:
                continue
            keys = extract_keys(text, qid, excerpt)
            if not keys:
                continue
            anchor.marks = anchor_marks(text) or keys["marks"]
            anchor.correct_index = keys["correct_index"]
            anchor.marking_guidance = keys["marking_guidance"]
            anchor.save(update_fields=["marks", "correct_index",
                                       "marking_guidance"])
            return True
        return False

"""
Resolve grading keys for question crops from their mark schemes.

For each unkeyed crop: find the matching mark-scheme document (same subject,
year, paper), slice the excerpt around the question number, and ask the LLM
once for {format, marks, correct_index?, marking_guidance}. Results are cached
on the crop forever.

Usage:
    python manage.py key_crops --subject-code 6880 [--limit 20]
"""

from django.core.management.base import BaseCommand, CommandError

from apps.quiz.models import QuestionCrop
from apps.quiz.services.generator import (
    extract_keys,
    find_mark_scheme,
    ms_excerpt_for,
)
from apps.syllabus.models import Subject


class Command(BaseCommand):
    help = "Resolve grading keys for crops from mark schemes."

    def add_arguments(self, parser):
        parser.add_argument("--subject-code", required=True)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **options):
        try:
            subject = Subject.objects.get(code=options["subject_code"])
        except Subject.DoesNotExist:
            raise CommandError(
                f"No subject with code {options['subject_code']}")
        crops = list(QuestionCrop.objects.filter(
            document__subject=subject,
            marking_guidance="",
        ).select_related("document").order_by("id"))
        if options["limit"]:
            crops = crops[:options["limit"]]
        done = skipped = 0
        for crop in crops:
            if self._resolve(crop):
                done += 1
            else:
                skipped += 1
        self.stdout.write(self.style.SUCCESS(
            f"Keyed {done} crops ({skipped} skipped)"))

    def _resolve(self, crop) -> bool:
        doc = crop.document
        ms = find_mark_scheme(doc.subject, doc.year, doc.paper_number)
        if ms is None:
            return False
        excerpt = ms_excerpt_for(ms, crop.q_number)
        if not excerpt:
            return False
        keys = extract_keys(crop.ocr_text, crop.q_number, excerpt)
        if not keys:
            return False
        crop.format = keys["format"]
        crop.marks = keys["marks"]
        crop.correct_index = keys["correct_index"]
        crop.marking_guidance = keys["marking_guidance"]
        crop.save(update_fields=["format", "marks", "correct_index",
                                 "marking_guidance"])
        return True

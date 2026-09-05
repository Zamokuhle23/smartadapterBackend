"""
Resolve grading keys for question crops from their mark schemes.

For each unkeyed crop: find the matching mark-scheme document (same subject,
year, paper), slice the excerpt around the question number, and ask the LLM
once for {format, marks, correct_index?, marking_guidance}. Results are cached
on the crop forever.

Usage:
    python manage.py key_crops --subject-code 6880 [--limit 20]
"""

import re

from django.core.management.base import BaseCommand, CommandError

from apps.quiz.models import QuestionCrop
from apps.quiz.services.generator import _chat, _extract_json_object
from apps.syllabus.models import Subject, SyllabusDocument


KEY_PROMPT = """You link an exam question to its mark scheme.

QUESTION (OCR of the printed question):
{question}

MARK SCHEME EXCERPT (same paper, question {number}):
{ms}

Reply with ONLY valid JSON, no fences:
{{"format": "mcq" or "structured",
  "marks": <total marks, integer>,
  "correct_index": <0-3 for MCQ with options A-D in the image, else null>,
  "marking_guidance": "<model answer + mark breakdown>"}}"""


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
        ms = SyllabusDocument.objects.filter(
            subject=doc.subject, doc_type=SyllabusDocument.DocType.MARK_SCHEME,
            year=doc.year, paper_number=doc.paper_number,
        ).first()
        if ms is None:
            return False
        excerpt = self._ms_excerpt(ms, crop.q_number)
        if not excerpt:
            return False
        try:
            raw = _chat([
                {"role": "system", "content": (
                    "You read mark schemes precisely. Output ONLY valid JSON.")},
                {"role": "user", "content": KEY_PROMPT.format(
                    question=(crop.ocr_text or "")[:2000],
                    number=crop.q_number, ms=excerpt[:3000])},
            ])
            item = _extract_json_object(raw)
        except Exception:  # noqa: BLE001 - leave unkeyed, retry later
            return False
        fmt = str(item.get("format", "structured")).lower()
        if fmt not in ("mcq", "structured"):
            return False
        try:
            marks = max(1, min(25, int(item.get("marks", 1))))
        except (TypeError, ValueError):
            return False
        correct = None
        if fmt == "mcq":
            try:
                correct = int(item.get("correct_index"))
            except (TypeError, ValueError):
                return False
            if correct not in (0, 1, 2, 3):
                return False
        crop.format = fmt
        crop.marks = marks
        crop.correct_index = correct
        crop.marking_guidance = str(item.get("marking_guidance", ""))[:2000]
        crop.save(update_fields=["format", "marks", "correct_index",
                                 "marking_guidance"])
        return True

    @staticmethod
    def _ms_excerpt(ms, q_number: str) -> str:
        """Slice ~2000 chars of mark-scheme text around the question number."""
        from apps.rag.models import DocumentChunk

        texts = list(DocumentChunk.objects.filter(
            document=ms).order_by("ordinal").values_list("text", flat=True))
        blob = "\n".join(texts)
        if not blob.strip():
            return ""
        m = re.search(r"(?mi)^\s*" + re.escape(q_number) + r"\b", blob)
        if not m:
            return ""
        return blob[max(0, m.start() - 200):m.start() + 2000]

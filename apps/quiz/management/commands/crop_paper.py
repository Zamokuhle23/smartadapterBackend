"""
Detect + persist whole-question crops for one past-paper document.

Idempotent: existing (document, q_number) rows are skipped, so re-running
fills gaps without duplicating. Crops render at ~144 DPI.

Usage:
    python manage.py crop_paper --doc-id 123 [--max 5]
"""

from io import BytesIO

from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError

from apps.quiz.models import QuestionCrop, QuestionCropImage
from apps.quiz.services.cropper import crop_question, detect_questions
from apps.syllabus.models import SyllabusDocument
from apps.syllabus.services.figure_keys import crop_key_for

ZOOM = 2.0


class Command(BaseCommand):
    help = "Crop whole questions from a past-paper PDF."

    def add_arguments(self, parser):
        parser.add_argument("--doc-id", type=int, required=True)
        parser.add_argument("--max", type=int, default=0,
                            help="Stop after N questions (0 = all)")

    def handle(self, *args, **options):
        try:
            doc = SyllabusDocument.objects.select_related("subject").get(
                pk=options["doc_id"])
        except SyllabusDocument.DoesNotExist:
            raise CommandError(f"No document {options['doc_id']}")
        import pymupdf

        pdf = pymupdf.open(doc.file.path)
        questions = detect_questions(pdf)
        made = skipped = 0
        renders = {}
        for q in questions:
            if options["max"] and made >= options["max"]:
                break
            if QuestionCrop.objects.filter(document=doc,
                                           q_number=q["number"]).exists():
                skipped += 1
                continue
            for pno, _, _ in q["pages"]:
                if pno not in renders:
                    pix = pdf[pno - 1].get_pixmap(
                        matrix=pymupdf.Matrix(ZOOM, ZOOM))
                    from PIL import Image

                    renders[pno] = Image.open(
                        BytesIO(pix.tobytes("png"))).convert("RGB")
            datas = crop_question(renders, q, ZOOM)
            if not datas:
                continue
            ocr = self._ocr_text(pdf, q)
            crop = QuestionCrop.objects.create(
                document=doc, q_number=q["number"],
                stable_key=self._unique_key(doc, q["number"]),
                pages=[{"page": p, "top": t, "bottom": b}
                       for p, t, b in q["pages"]],
                ocr_text=ocr,
                confidence=0.85 if q["confident"] else 0.55,
                status=(QuestionCrop.Status.AUTO if q["confident"]
                        else QuestionCrop.Status.NEEDS_QC),
            )
            for i, data in enumerate(datas):
                pno = q["pages"][i][0] if i < len(q["pages"]) else 0
                img = QuestionCropImage(crop=crop, page_number=pno, sort=i)
                img.image.save(f"p{pno}.png", ContentFile(data), save=True)
            made += 1
            self.stdout.write(
                f"  Q{q['number']}: {len(datas)} slice(s) "
                f"conf={crop.confidence} [{crop.status}]")
        self.stdout.write(self.style.SUCCESS(
            f"Cropped {made} questions ({skipped} already present)"))

    @staticmethod
    def _ocr_text(pdf, question) -> str:
        """Native PDF text inside the question bounds (no OCR engine needed)."""
        import pymupdf

        parts = []
        for pno, top, bottom in question["pages"]:
            page = pdf[pno - 1]
            clip = pymupdf.Rect(0, max(0, top),
                                page.rect.width, min(page.rect.height, bottom))
            parts.append(page.get_text("text", clip=clip).strip())
        return "\n".join(p for p in parts if p)[:4000]

    @staticmethod
    def _unique_key(doc, number: str) -> str:
        key, i = crop_key_for(doc, number), 2
        while QuestionCrop.objects.filter(stable_key=key).exists():
            key = f"{crop_key_for(doc, number)}-{i}"
            i += 1
        return key

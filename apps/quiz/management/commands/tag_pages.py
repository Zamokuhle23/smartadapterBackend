"""
Label every page of a subject's past papers with one subtopic.

Groups each page's chunk texts and asks the tagging model once per page:
{"subtopic": "...", "confidence": 0.0-1.0}. Idempotent (tagged pages are
skipped); free-tier friendly, one call per page.

Usage:
    python manage.py tag_pages --subject-code 6880 [--limit-pages 50]
"""

from django.core.management.base import BaseCommand, CommandError

from apps.quiz.models import PageTopic
from apps.rag.models import DocumentChunk
from apps.rag.services.tagging import TaggingError, tagging_chat
from apps.syllabus.models import Subject, SyllabusDocument


TAG_PROMPT = """Label this exam-paper page with ONE subtopic.

SUBJECT: {subject}
PAGE TEXT:
{text}

Reply with ONLY valid JSON, no fences:
{{"subtopic": "<short label, max 6 words, syllabus vocabulary>",
  "confidence": <0.0-1.0>}}"""


class Command(BaseCommand):
    help = "Label past-paper pages with subtopics."

    def add_arguments(self, parser):
        parser.add_argument("--subject-code", required=True)
        parser.add_argument("--limit-pages", type=int, default=0)
        parser.add_argument("--max-chars", type=int, default=1500)
        parser.add_argument("--source", default="",
                            help="Only tag docs of this source (egcse/igcse)")

    def handle(self, *args, **options):
        try:
            subject = Subject.objects.get(code=options["subject_code"])
        except Subject.DoesNotExist:
            raise CommandError(
                f"No subject with code {options['subject_code']}")
        docs = list(SyllabusDocument.objects.filter(
            subject=subject,
            doc_type=SyllabusDocument.DocType.PAST_PAPER,
            **({"source": options["source"]} if options["source"] else {}),
        ).order_by("id"))
        done = skipped = failed = 0
        for doc in docs:
            pages = self._page_texts(doc, options["max_chars"])
            for page, text in pages.items():
                if options["limit_pages"] and done >= options["limit_pages"]:
                    break
                if PageTopic.objects.filter(
                        document=doc, page_number=page).exists():
                    skipped += 1
                    continue
                label, conf = self._tag(subject.name, text)
                if label is None:
                    failed += 1
                    self.stderr.write(f"  p{page} of doc {doc.id}: no label")
                    continue
                PageTopic.objects.create(
                    document=doc, page_number=page, label=label,
                    confidence=conf)
                done += 1
                if done % 10 == 0:
                    self.stdout.write(f"  {done} pages tagged")
                # Gentle pace for free-tier rate limits.
                import time as _time
                _time.sleep(3.0)
            if options["limit_pages"] and done >= options["limit_pages"]:
                break
        self.stdout.write(self.style.SUCCESS(
            f"Tagged {done} pages ({skipped} already done, {failed} failed)"))

    @staticmethod
    def _page_texts(doc, max_chars: int) -> dict:
        """Concatenated chunk text per page (cap chars, skip empties)."""
        pages = {}
        # Explicit columns only: embedding_vec is PostgreSQL-only, and this
        # must also run on SQLite dev/test databases.
        chunks = DocumentChunk.objects.filter(
            document=doc).order_by("page_number", "ordinal").values(
            "page_number", "ordinal", "text")
        for chunk in chunks:
            if chunk["page_number"] is None:
                continue
            pages.setdefault(chunk["page_number"], []).append(
                chunk["text"] or "")
        out = {}
        for page, texts in pages.items():
            blob = "\n".join(texts)[:max_chars].strip()
            if blob:
                out[page] = blob
        return out

    @staticmethod
    def _tag(subject_name: str, text: str):
        """Returns (label, confidence) or (None, 0.0)."""
        import json

        try:
            raw = tagging_chat(TAG_PROMPT.format(subject=subject_name,
                                                 text=text))
            item = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            label = str(item.get("subtopic", "")).strip()[:120]
            conf = float(item.get("confidence", 0.5))
        except (TaggingError, ValueError, KeyError, TypeError,
                AttributeError):
            return None, 0.0
        if not label:
            return None, 0.0
        return label, max(0.0, min(1.0, conf))

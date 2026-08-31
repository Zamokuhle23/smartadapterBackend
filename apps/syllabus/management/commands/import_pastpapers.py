"""
Bulk-ingest the downloaded Cambridge IGCSE + ECESWA EGCSE past-paper corpus.

IGCSE is the PRIMARY source (5,300+ papers); EGCSE is the SECONDARY source
(295 papers) used for local alignment. Every document is tagged `source`
so practice/exam generation can weight (and label) provenance - see
apps/syllabus/services/subject_map.py.

Usage:
    python manage.py import_pastpapers --source igcse
    python manage.py import_pastpapers --source egcse
    python manage.py import_pastpapers --source igcse --limit 20   # smoke test

Defaults read from the standard Resource/ layout:
    C:\\work\\FundzaAI\\Resource\\IGCSE_Papers
    C:\\work\\FundzaAI\\Resource\\EGCSE_Papers
"""

import os
import re

from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError

from apps.syllabus.models import SyllabusDocument
from apps.syllabus.services.ingestion import process_document
from apps.syllabus.services.subject_map import (
    egcse_code_from_filename,
    egcse_subject_by_code,
    egcse_subject_for_igcse_code,
)


class Command(BaseCommand):
    help = "Ingest Cambridge IGCSE / ECESWA EGCSE past papers with source tagging"

    def add_arguments(self, parser):
        parser.add_argument(
            "--source", choices=["igcse", "egcse"], required=True,
            help="Which corpus to ingest (igcse = Cambridge primary, egcse = ECESWA secondary)",
        )
        parser.add_argument("--root", default=None, help="Override the Resource folder")
        parser.add_argument("--limit", type=int, default=0, help="Stop after N files (smoke test)")

    def handle(self, *args, **options):
        source = SyllabusDocument.Source(options["source"])
        default_root = (
            os.path.join(self._content_root(), "past_papers", "igcse")
            if options["source"] == "igcse"
            else os.path.join(self._content_root(), "past_papers", "egcse")
        )
        root = options["root"] or default_root
        if not os.path.isdir(root):
            raise CommandError(f"Not a folder: {root}")

        files = []
        for r, _dirs, names in os.walk(root):
            for name in sorted(names):
                if name.lower().endswith(".pdf"):
                    files.append(os.path.join(r, name))
        if not files:
            raise CommandError(f"No PDFs found under {root}")
        self.stdout.write(self.style.WARNING(f"Found {len(files)} PDFs under {root}"))

        ok = skipped = failed = 0
        for i, path in enumerate(files, start=1):
            if options["limit"] and i > options["limit"]:
                break
            name = os.path.basename(path)
            resolved = self._resolve(path, options["source"])
            if resolved[0] is None:
                skipped += 1
                self.stdout.write(self.style.WARNING(f"  SKIP {name}: {resolved[1]}"))
                continue
            subject, provenance, title = resolved
            doc_title = f"{title} [{subject.code}]"
            if SyllabusDocument.objects.filter(subject=subject, title=doc_title).exists():
                skipped += 1  # idempotent re-run
                continue
            try:
                doc = SyllabusDocument(
                    syllabus=subject.syllabus,
                    subject=subject,
                    title=doc_title,
                    source=source,
                    **provenance,
                )
                with open(path, "rb") as fh:
                    doc.file.save(name, ContentFile(fh.read()), save=True)
                n = process_document(doc)
                ok += 1
                if ok % 25 == 0 or i <= 3:
                    self.stdout.write(self.style.SUCCESS(f"  OK  {title} -> {subject.code}: {n} chunks"))
            except Exception as exc:  # noqa: BLE001 - keep going
                failed += 1
                self.stdout.write(self.style.ERROR(f"  FAIL {name}: {exc}"))

        self.stdout.write(
            self.style.SUCCESS(f"Done. {ok} ingested, {skipped} skipped, {failed} failed (source={source}).")
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _content_root() -> str:
        """Canonical content root inside the backend (backend/content)."""
        from django.conf import settings
        return os.path.join(str(settings.BASE_DIR), "content")

    def _resolve(self, path: str, source_flag: str):
        """Return (Subject, provenance dict, title) or (None, reason)."""
        name = os.path.splitext(os.path.basename(path))[0]
        from .ingest_folder import classify_filename  # local import avoids a cycle

        provenance = classify_filename(name)
        if provenance["doc_type"] == SyllabusDocument.DocType.SYLLABUS:
            return None, "syllabus/notes file (only past papers + mark schemes ingested here)"
        # Only ingest past papers and mark schemes (the question-generation corpus).
        if provenance["doc_type"] not in (
            SyllabusDocument.DocType.PAST_PAPER,
            SyllabusDocument.DocType.MARK_SCHEME,
        ):
            return None, f"not a question paper / mark scheme ({provenance['doc_type']})"

        if source_flag == "igcse":
            m = re.match(r"(\d{4})", name)
            subject = egcse_subject_for_igcse_code(m.group(1)) if m else None
            if subject is None:
                return None, "no local EGCSE mapping for this Cambridge code (or unmapped)"
            return subject, provenance, name

        # egcse: infer the subject from the filename itself
        code = egcse_code_from_filename(name)
        subject = egcse_subject_by_code(code) if code else None
        if subject is None:
            return None, "could not infer an EGCSE subject from filename"
        return subject, provenance, name
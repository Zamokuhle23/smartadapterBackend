"""
Import the official syllabus PDFs into the RAG corpus, one document per subject.

Reads from backend/content/syllabi/{egcse,igcse}, resolves each PDF to its
subject (EGCSE by filename, or IGCSE ICT for the sole Cambridge-syllabus subject),
ingests it as a SYLLABUS document (source-tagged), and - when --seed-topics is
passed - seeds a best-effort Topic tree for that subject.

Usage:
    python manage.py import_syllabi --source egcse
    python manage.py import_syllabi --source igcse --seed-topics
    python manage.py import_syllabi              # scans both, seeds topics
"""

import os
import re

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand

from apps.syllabus.models import Subject, SyllabusDocument, Topic
from apps.syllabus.services.topic_seeder import seed_topics_for_subject
from apps.syllabus.services import subject_map as sm


class Command(BaseCommand):
    help = "Ingest official syllabus PDFs (EGCSE + IGCSE ICT) with optional topic seeding"

    def add_arguments(self, parser):
        parser.add_argument(
            "--source", choices=["egcse", "igcse", "all"], default="all",
            help="Which syllabus set to ingest (egcse / igcse / all)",
        )
        parser.add_argument(
            "--seed-topics", action="store_true",
            help="Best-effort create Topic rows from each syllabus's numbered headings",
        )

    def handle(self, *args, **options):
        content = os.path.join(str(settings.BASE_DIR), "content", "syllabi")
        sets = []
        if options["source"] in ("egcse", "all"):
            sets.append(("egcse", os.path.join(content, "egcse")))
        if options["source"] in ("igcse", "all"):
            sets.append(("igcse", os.path.join(content, "igcse")))

        total_ok = total_fail = total_topic = 0
        for source, folder in sets:
            if not os.path.isdir(folder):
                self.stdout.write(self.style.WARNING(f"Skipping missing {folder}"))
                continue
            ok, fail, topics = self._ingest_set(source, folder, options["seed_topics"])
            total_ok += ok
            total_fail += fail
            total_topic += topics

        self.stdout.write(self.style.SUCCESS(
            f"Done. {total_ok} syllabi ingested, {total_fail} failed, {total_topic} topics seeded."
        ))

    # ------------------------------------------------------------------
    def _ingest_set(self, source: str, folder: str, seed_topics: bool):
        source_enum = SyllabusDocument.Source(source)
        files = sorted(
            f for f in os.listdir(folder) if f.lower().endswith(".pdf")
        )
        ok = fail = topics_added = 0
        for name in files:
            subject = self._resolve_subject(source, name)
            if subject is None:
                self.stdout.write(self.style.WARNING(f"  SKIP {name}: no matching subject"))
                fail += 1
                continue
            title = os.path.splitext(name)[0]
            doc_title = f"{title} [{subject.code}]"
            if SyllabusDocument.objects.filter(subject=subject, title=doc_title).exists():
                self.stdout.write(self.style.HTTP_INFO(f"  keep {name} (already ingested)"))
            else:
                doc = SyllabusDocument(
                    syllabus=subject.syllabus,
                    subject=subject,
                    title=doc_title,
                    doc_type=SyllabusDocument.DocType.SYLLABUS,
                    source=source_enum,
                    year=self._years(title),
                )
                try:
                    with open(os.path.join(folder, name), "rb") as fh:
                        doc.file.save(name, ContentFile(fh.read()), save=True)
                    from apps.syllabus.services.ingestion import process_document

                    n = process_document(doc)
                    ok += 1
                    self.stdout.write(self.style.SUCCESS(f"  OK  {title} -> {subject.code}: {n} chunks"))
                except Exception as exc:  # noqa: BLE001 - keep going
                    fail += 1
                    self.stdout.write(self.style.ERROR(f"  FAIL {name}: {exc}"))
            if seed_topics:
                created = seed_topics_for_subject(subject)
                if created:
                    topics_added += len(created)
                    self.stdout.write(
                        self.style.SUCCESS(f"      topics: {', '.join(created)}")
                    )
        return ok, fail, topics_added

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_subject(source: str, filename: str):
        stem = os.path.splitext(filename)[0]
        # Prefer the human-name rules (handles both "EGCSE Mathematics ..." and
        # the IGCSE ICT syllabus, which has no leading exam code).
        code = sm.egcse_code_from_filename(stem)
        if code is None:
            m = re.match(r"(\d{4})", stem)  # fallback: Cambridge code prefix
            code = m.group(1) if m else None
        return sm.egcse_subject_by_code(code) if code else None

    @staticmethod
    def _years(title: str) -> int | None:
        m = re.search(r"(20\d{2})\s*-\s*(20\d{2})", title)
        return int(m.group(1)) if m else None
"""
Backfill figures + chunk page-numbers for already-ingested documents.

Documents ingested before figure support were chunked from the whole PDF with no
page mapping and no extracted images. Re-running process_document re-extracts text
per page, records each chunk's page_number, stores the page's figures as
DocumentFigure rows, and re-embeds (using the current EMBEDDING_PROVIDER).

Usage:
    python manage.py backfill_figures --subject 6884        # Biology
    python manage.py backfill_figures --subject 6884 --limit 3   # smoke test
    python manage.py backfill_figures --all                 # everything (slow)
    python manage.py backfill_figures --document-id 12 13   # specific docs
"""

from django.core.management.base import BaseCommand, CommandError

from apps.syllabus.models import Syllabus, Subject, SyllabusDocument
from apps.syllabus.services.ingestion import process_document


class Command(BaseCommand):
    help = "Re-ingest documents to store page-based figures and chunk page numbers"

    def add_arguments(self, parser):
        parser.add_argument("--subject", default=None, help="EGCSE subject code, e.g. 6884")
        parser.add_argument("--all", action="store_true", help="Process every syllabus document")
        parser.add_argument("--document-id", nargs="*", type=int, default=[], help="Specific doc ids")
        parser.add_argument("--limit", type=int, default=0, help="Cap how many docs to process")

    def handle(self, *args, **options):
        qs = SyllabusDocument.objects.filter(status=SyllabusDocument.Status.READY)
        if options["subject"]:
            qs = qs.filter(subject__code=options["subject"])
        elif options["document_id"]:
            qs = qs.filter(id__in=options["document_id"])
        elif not options["all"]:
            raise CommandError("Pass --subject CODE, --all, or --document-id id...")
        qs = qs.order_by("id")
        if options["limit"]:
            qs = qs[: options["limit"]]

        total = qs.count()
        self.stdout.write(self.style.WARNING(f"Backfilling {total} document(s)..."))
        ok = failed = 0
        for doc in qs:
            try:
                doc.refresh_from_db()
                n = process_document(doc)
                figures = doc.figures.count()
                ok += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  OK {doc.title}: {n} chunks, {figures} figures"
                    )
                )
            except Exception as exc:  # noqa: BLE001 - keep going
                failed += 1
                self.stdout.write(self.style.ERROR(f"  FAIL {doc.title}: {exc}"))
        self.stdout.write(self.style.SUCCESS(f"Done. {ok} ok, {failed} failed."))
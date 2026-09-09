"""
Backfill QuestionAnchor.requires_figure for every past-paper anchor.

Replaceable (requires_figure=False) anchors are text-only parts whose answer
does not depend on a diagram/table/picture; they feed smart practice as
text-form items or AI variants. Figure-dependent anchors stay in page mode.

Usage:
    python manage.py backfill_anchor_flags [--doc-id N] [--limit 1000]
"""

from django.core.management.base import BaseCommand, CommandError

from apps.quiz.models import QuestionAnchor
from apps.quiz.services.cropper import anchor_requires_figure
from apps.syllabus.models import SyllabusDocument


class Command(BaseCommand):
    help = "Backfill QuestionAnchor.requires_figure (text-only replaceability)."

    def add_arguments(self, parser):
        parser.add_argument("--doc-id", type=int, default=None)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **options):
        qs = QuestionAnchor.objects.filter(
            document__doc_type=SyllabusDocument.DocType.PAST_PAPER)
        if options["doc_id"]:
            if not SyllabusDocument.objects.filter(pk=options["doc_id"]).exists():
                raise CommandError(f"No document {options['doc_id']}")
            qs = qs.filter(document_id=options["doc_id"])
        total = qs.count()
        if options["limit"]:
            qs = qs[: options["limit"]]
        updated = 0
        for anchor in qs.iterator(chunk_size=200):
            doc = anchor.document
            try:
                value = anchor_requires_figure(doc, anchor)
            except Exception as exc:  # noqa: BLE001 - keep going, flag conservative
                self.stderr.write(
                    f"skip doc={doc.id} {anchor.qid}: {exc}")
                continue
            if anchor.requires_figure != value:
                anchor.requires_figure = value
                anchor.save(update_fields=["requires_figure"])
                updated += 1
        self.stdout.write(self.style.SUCCESS(
            f"Backfilled {updated}/{total} anchors "
            f"({total - updated} already correct)."))
"""Decompose cached mark-scheme guidance into atomic criteria (offline packs).

Idempotent: anchors that already have criteria are skipped. Rows the LLM
cannot split exactly are reported, never stored half-done.

Usage:
    python manage.py extract_criteria --subject 6880 --limit 40
    python manage.py extract_criteria --anchor 123
"""

from django.core.management.base import BaseCommand, CommandError

from apps.quiz.models import QuestionAnchor
from apps.quiz.services.atomic_criteria import extract_criteria
from apps.quiz.services.cropper import anchor_text
from apps.syllabus.models import Subject


class Command(BaseCommand):
    help = "Extract atomic marking criteria from cached guidance."

    def add_arguments(self, parser):
        parser.add_argument("--subject", default=None,
                            help="Subject code, e.g. 6880")
        parser.add_argument("--anchor", type=int, default=0,
                            help="Single anchor id")
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **options):
        if options["anchor"]:
            try:
                qs = [QuestionAnchor.objects.select_related("document").get(
                    pk=options["anchor"])]
            except QuestionAnchor.DoesNotExist:
                raise CommandError("No such anchor")
        elif options["subject"]:
            try:
                subject = Subject.objects.get(code=options["subject"])
            except Subject.DoesNotExist:
                raise CommandError("No such subject")
            qs = list(QuestionAnchor.objects.filter(
                document__subject=subject, kind="text",
            ).exclude(marking_guidance="").order_by("id"))
        else:
            raise CommandError("Pass --subject CODE or --anchor ID")
        if options["limit"]:
            qs = qs[: options["limit"]]
        done = skipped = failed = 0
        for anchor in qs:
            if anchor.marking_criteria:
                skipped += 1
                continue
            try:
                text = anchor_text(anchor.document.file.path,
                                   anchor.page_number, anchor.bbox)
            except Exception:  # noqa: BLE001
                text = ""
            criteria = extract_criteria(
                text, anchor.marking_guidance, anchor.marks)
            if criteria is None:
                failed += 1
                self.stdout.write(
                    f"  anchor {anchor.id} ({anchor.qid}): needs review")
                continue
            anchor.marking_criteria = criteria
            anchor.save(update_fields=["marking_criteria"])
            done += 1
        self.stdout.write(self.style.SUCCESS(
            f"Done. {done} extracted, {skipped} already done, "
            f"{failed} need review."))

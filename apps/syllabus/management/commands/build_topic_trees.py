"""Build Topic trees from PageTopic page labels and link chunks to topics.

PageTopic labels are free text (one per document page). This command groups
them into canonical subtopics per subject, creates one Topic row each
(idempotent), and backfills DocumentChunk.topic via (document, page_number)
so chat routing (classify_topic) and thread grouping can work.

Canonical form: text before the first ':' / ';', then before the first
',' / '&' / '+' / '/' / ' and '. Grouping is case-insensitive; the most
common original variant becomes the display title. Obvious junk pages
(blank pages, copyright notices) are skipped.
"""

from collections import Counter, defaultdict

from django.core.management.base import BaseCommand
from django.db import transaction

from apps.syllabus.services.topic_labels import canonical_key


class Command(BaseCommand):
    help = "Build Topic trees from PageTopic labels and backfill chunk topics."

    def add_arguments(self, parser):
        parser.add_argument("--subject", dest="subject_code", default=None,
                            help="Only process one subject code, e.g. 6880")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        from apps.quiz.models import PageTopic
        from apps.rag.models import DocumentChunk
        from apps.syllabus.models import Subject, Topic

        code = options["subject_code"]
        subjects = Subject.objects.order_by("code")
        if code:
            subjects = subjects.filter(code=code)

        total_topics = 0
        total_linked = 0
        for subject in subjects:
            pts = list(
                PageTopic.objects.filter(document__subject=subject)
                .select_related("document")
                .order_by("document_id", "page_number")
            )
            if not pts:
                continue
            groups: dict[str, list] = defaultdict(list)
            for pt in pts:
                key = canonical_key(pt.label)
                if not key:
                    continue
                groups[key].append(pt)

            if options["dry_run"]:
                self.stdout.write(
                    f"{subject.code}: DRY-RUN would create ~{len(groups)} topics "
                    f"over {len(pts)} pages"
                )
                continue
            with transaction.atomic():
                # One Topic per canonical label; display title = most common variant.
                topic_by_key: dict[str, Topic] = {}
                for key, items in groups.items():
                    variants = Counter(
                        (p.label or "").strip() for p in items if (p.label or "").strip()
                    )
                    title = variants.most_common(1)[0][0][:300]
                    topic, created = Topic.objects.get_or_create(
                        subject=subject, title=title,
                        defaults={"kind": Topic.Kind.SUBTOPIC},
                    )
                    topic_by_key[key] = topic
                    if created:
                        total_topics += 1

                # Backfill chunks: one UPDATE per (document, page) group.
                page_topic: dict[tuple, int] = {}
                for key, items in groups.items():
                    tid = topic_by_key[key].id
                    for pt in items:
                        page_topic[(pt.document_id, pt.page_number)] = tid
                by_topic: dict[int, list] = defaultdict(list)
                for (doc_id, page), tid in page_topic.items():
                    by_topic[tid].append((doc_id, page))
                for tid, pairs in by_topic.items():
                    doc_ids = {d for d, _ in pairs}
                    for doc_id in doc_ids:
                        pages = [p for d, p in pairs if d == doc_id]
                        n = DocumentChunk.objects.filter(
                            document_id=doc_id, page_number__in=pages,
                            topic__isnull=True,
                        ).update(topic_id=tid)
                        total_linked += n

            self.stdout.write(
                f"{subject.code}: {len(groups)} topics, "
                f"{sum(len(v) for v in groups.values())} pages mapped"
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"done: {total_topics} new topics, {total_linked} chunks linked"
            )
        )

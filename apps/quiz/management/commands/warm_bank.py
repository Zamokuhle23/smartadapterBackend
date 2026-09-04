"""
Prefill the shared question bank so the first practice question is instant.

The bank is shared across students (attempts are per-student), so warming once
per subject makes every student's first question a bank hit instead of a
60-second LLM wait. Background prefetch in the app then keeps it topped up.

Usage:
    python manage.py warm_bank --subject-code 6880 --total 12 --batch 3
    python manage.py warm_bank --subject-code 6880 --topic-ids 4,9 --total 6
"""

from django.core.management.base import BaseCommand, CommandError

from apps.quiz.services.generator import QuizGenerationError, generate_questions
from apps.syllabus.models import Subject


class Command(BaseCommand):
    help = "Generate questions ahead of time so practice starts instantly."

    def add_arguments(self, parser):
        parser.add_argument("--subject-code", required=True,
                            help="ECESWA subject code, e.g. 6880")
        parser.add_argument("--total", type=int, default=12,
                            help="Target number of bank questions to add")
        parser.add_argument("--batch", type=int, default=3,
                            help="Questions per LLM call (1-10)")
        parser.add_argument("--difficulty", type=int, default=2)
        parser.add_argument("--topic-ids", default="",
                            help="Comma-separated topic ids to scope to")

    def handle(self, *args, **options):
        try:
            subject = Subject.objects.get(code=options["subject_code"])
        except Subject.DoesNotExist:
            raise CommandError(
                f"No subject with code {options['subject_code']}")
        topic_ids = [int(t) for t in str(options["topic_ids"]).split(",")
                     if t.strip().isdigit()] or None
        total = max(1, options["total"])
        batch = max(1, min(10, options["batch"]))
        created = 0
        failures = 0
        while created < total:
            if failures >= 3:
                raise CommandError(
                    f"Aborting after 3 consecutive failures "
                    f"({created}/{total} created)")
            try:
                made = generate_questions(
                    subject,
                    count=min(batch, total - created),
                    difficulty=options["difficulty"],
                    topic_ids=topic_ids,
                )
            except QuizGenerationError as exc:
                failures += 1
                self.stderr.write(f"batch failed ({exc}); continuing")
                continue
            failures = 0
            created += len(made)
            self.stdout.write(f"+{len(made)} ({created}/{total})")
        self.stdout.write(
            self.style.SUCCESS(f"Bank warmed: {created} questions "
                               f"for {subject.code}"))

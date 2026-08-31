"""
Seed the ECESWA syllabus shells (JC + EGCSE) with their published subjects so
the system boots with real "textbooks on the shelf" ready for document uploads.

Usage: python manage.py seed_syllabi
"""

from django.core.management.base import BaseCommand

from apps.syllabus.models import Subject, Syllabus

JC_SUBJECTS = [
    ("101", "English Language"),
    ("120", "English Literature"),
    ("202", "French"),
    ("207", "SiSwati"),
    ("309", "Mathematics"),
    ("414", "Science"),
    ("516", "Agriculture"),
    ("519", "Additional Mathematics"),
    ("520", "Bookkeeping and Accounts"),
    ("521", "Business Studies"),
    ("524", "Development Studies"),
    ("527", "Geography"),
    ("530", "History"),
    ("533", "Religious Education"),
    ("537", "Design and Technology"),
    ("540", "Consumer Science"),
]

EGCSE_SUBJECTS = [
    ("6870", "First Language SiSwati"),
    ("6871", "SiSwati as a Second Language"),
    ("6873", "English Language"),
    ("6875", "Literature in English"),
    ("6880", "Mathematics"),
    ("6882", "Agriculture"),
    ("6884", "Biology"),
    ("6888", "Physical Science"),
    ("6890", "Geography"),
    ("6891", "History"),
    ("6893", "Religious Education"),
    ("6896", "Accounting"),
    ("6897", "Business Studies"),
    ("6899", "Economics"),
    ("6902", "Design and Technology"),
    ("6904", "Fashion and Fabrics"),
    ("6905", "Food and Nutrition"),
    # No EGCSE ICT exists - Eswatini students sit Cambridge IGCSE ICT.
    ("0417", "Information and Communication Technology"),
]


class Command(BaseCommand):
    help = "Create JC and EGCSE syllabus shells with ECESWA subjects"

    def handle(self, *args, **options):
        created = 0
        for level, name, subjects in (
            (Syllabus.Level.JC, "Junior Certificate", JC_SUBJECTS),
            (Syllabus.Level.EGCSE, "Eswatini General Certificate of Secondary Education", EGCSE_SUBJECTS),
        ):
            syllabus, was_created = Syllabus.objects.get_or_create(
                level=level,
                name=name,
                version="1.0",
                defaults={
                    "status": Syllabus.Status.PUBLISHED,
                    "description": (
                        "ECESWA national syllabus. Administered by the Examinations "
                        "Council of Eswatini."
                    ),
                },
            )
            if was_created:
                created += 1
            for code, subject_name in subjects:
                Subject.objects.get_or_create(
                    syllabus=syllabus, code=code, defaults={"name": subject_name}
                )
            self.stdout.write(
                self.style.SUCCESS(f"{syllabus}: {len(subjects)} subjects ensured")
            )
        self.stdout.write(self.style.SUCCESS(f"Done. {created} syllabi created."))

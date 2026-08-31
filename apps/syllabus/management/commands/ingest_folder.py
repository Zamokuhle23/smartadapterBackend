"""
Bulk-ingest a local folder of syllabus documents.

Drop ECESWA syllabus PDFs / notes / past papers into backend/content/<subject>/,
then run:

    python manage.py ingest_folder content --subject 6880

Every .pdf/.txt/.md/.docx found (recursively) is ingested against that subject's
syllabus and becomes available to the tutor's RAG immediately.
"""

import os
import re

from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError

from apps.syllabus.models import Subject, SyllabusDocument
from apps.syllabus.services.ingestion import process_document

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


def classify_filename(name: str) -> dict:
    """
    Infer provenance from the filename so past papers feed question generation:
      '6880_paper2_2021.pdf'          -> past_paper, paper 2, year 2021
      'Maths P1 2023 mark scheme.pdf' -> mark_scheme, paper 1, year 2023
      'EGCSE-syllabus.pdf'            -> syllabus
      '0580_m24_qp_12.pdf'           -> Cambridge-style past_paper, paper 1, 2024
      '0620_w17_ms_21.pdf'           -> Cambridge-style mark_scheme, paper 2, 2017
    """
    lowered = name.lower()
    info = {"doc_type": SyllabusDocument.DocType.NOTES, "paper_number": None, "year": None}

    # ---- Cambridge/IGCSE style: <code>_<session><yy>_<qp|ms>_<variant>.pdf ----
    cam = re.search(
        r"(?P<code>\d{4})\s*_\s*(?P<sess>[wms])(?P<yy>\d{2})\s*_\s*(?P<kind>qp|ms)(?:\s*_\s*(?P<var>\d{2}))?",
        lowered,
    )
    if cam:
        if cam["kind"] == "qp":
            info["doc_type"] = SyllabusDocument.DocType.PAST_PAPER
        else:
            info["doc_type"] = SyllabusDocument.DocType.MARK_SCHEME
        info["year"] = 2000 + int(cam["yy"])
        if cam["var"]:
            variant = int(cam["var"])
            info["paper_number"] = max(1, variant // 10)
        return info

    # ---- Human/ECESWA style ----
    if "mark scheme" in lowered or "markscheme" in lowered or "m s" in lowered or re.search(r"\bms\b", lowered):
        info["doc_type"] = SyllabusDocument.DocType.MARK_SCHEME
    elif "syllab" in lowered:
        info["doc_type"] = SyllabusDocument.DocType.SYLLABUS
    elif "paper" in lowered or re.search(r"\bp[1-4]\b", lowered) or "question paper" in lowered or "past" in lowered:
        info["doc_type"] = SyllabusDocument.DocType.PAST_PAPER

    m = re.search(r"(?:paper|pp|p)[\s_-]*([1-4])", lowered)
    if m:
        info["paper_number"] = int(m.group(1))

    y = re.search(r"\b(19[89]\d|20\d{2})\b", name)
    if y:
        info["year"] = int(y.group(1))

    # Bare "<SUBJECT> N" (e.g. "BIOLOGY 2", "PHYSICAL SCIENCE 4") = a question paper.
    # Only applies when nothing else was detected and there is no marking-scheme marker.
    if info["doc_type"] == SyllabusDocument.DocType.NOTES:
        trail = re.search(r"\b([1-5])\s*$", lowered)
        if trail:
            info["doc_type"] = SyllabusDocument.DocType.PAST_PAPER
            info["paper_number"] = int(trail.group(1))

    return info


class Command(BaseCommand):
    help = "Ingest all supported documents in a folder against one subject"

    def add_arguments(self, parser):
        parser.add_argument("folder")
        parser.add_argument("--subject", required=True, help="ECESWA subject code, e.g. 6880")
        parser.add_argument(
            "--source",
            choices=["igcse", "egcse"],
            default="egcse",
            help="Provenance tag: 'igcse' (Cambridge, primary) or 'egcse' (ECESWA, secondary)",
        )

    def handle(self, *args, **options):
        subject = Subject.objects.filter(code=options["subject"]).select_related("syllabus").first()
        if subject is None:
            raise CommandError(f"No subject with code {options['subject']} - run seed_syllabi first.")
        source = SyllabusDocument.Source(options["source"])

        folder = options["folder"]
        if not os.path.isdir(folder):
            raise CommandError(f"Not a folder: {folder}")

        files = []
        for root, _dirs, names in os.walk(folder):
            for name in sorted(names):
                if os.path.splitext(name)[1].lower() in SUPPORTED:
                    files.append(os.path.join(root, name))

        if not files:
            raise CommandError(f"No supported documents ({', '.join(sorted(SUPPORTED))}) in {folder}")

        ok = failed = 0
        for path in files:
            title = os.path.splitext(os.path.basename(path))[0]
            provenance = classify_filename(title)
            doc = SyllabusDocument(
                syllabus=subject.syllabus,
                subject=subject,
                title=f"{title} [{subject.code}]",
                source=source,
                **provenance,
            )
            with open(path, "rb") as fh:
                doc.file.save(os.path.basename(path), ContentFile(fh.read()), save=True)
            try:
                n = process_document(doc)
                ok += 1
                self.stdout.write(self.style.SUCCESS(f"  OK  {title}: {n} chunks"))
            except Exception as exc:  # noqa: BLE001 - keep going through bad files
                failed += 1
                self.stdout.write(self.style.ERROR(f"  FAIL {title}: {exc}"))

        self.stdout.write(
            self.style.SUCCESS(f"Done. {ok} ingested, {failed} failed, for {subject} ({source}).")
        )

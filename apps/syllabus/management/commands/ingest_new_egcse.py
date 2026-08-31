"""
Ingest newly-downloaded EGCSE PDFs, adding ONLY documents we don't yet have.

Two-layer dedup:
  1. Fingerprint dedup  - (subject, year, paper, doc_type) matches a stored doc.
  2. Content dedup      - for past papers & mark schemes, compare extracted text
     (token-set Jaccard) against every existing EGCSE doc. This catches files the
     messy legacy corpus has with a DIFFERENT filename (e.g. "BIOLOGY 1" vs the
     clean "EGCSE Biology 2020 Question Paper 1"), which fingerprint matching
     cannot see.

Kept files are copied into the canonical content/past_papers/egcse/ tree and
ingested with provenance source=egcse. One-pass, idempotent.

Usage: python manage.py ingest_new_egcse --dir <downloads_folder>
       python manage.py ingest_new_egcse --dir <folder> --limit 5   # smoke test
"""

import os
import re
import shutil
from collections import Counter

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError

from apps.syllabus.models import SyllabusDocument
from apps.syllabus.services.ingestion import extract_text, process_document
from apps.syllabus.services.subject_map import egcse_code_from_filename, egcse_subject_by_code

# Simple tokenizer: keep stable content words (3+ chars, alnum) lowercased.
_TOKEN_RE = re.compile(r"[a-z][a-z'\d]{2,}")


def tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
class Command(BaseCommand):
    help = "Ingest new EGCSE PDFs, adding only documents not already stored (content dedup)"

    def add_arguments(self, parser):
        parser.add_argument("--dir", required=True, help="Folder of newly downloaded EGCSE PDFs")
        parser.add_argument("--limit", type=int, default=0, help="Process at most N (smoke test)")

    def handle(self, *args, **options):
        src = options["dir"]
        if not os.path.isdir(src):
            raise CommandError(f"Not a folder: {src}")

        existing = {}
        for d in SyllabusDocument.objects.filter(source=SyllabusDocument.Source.EGCSE):
            key = (d.subject.code, d.doc_type)
            try:
                existing.setdefault(key, []).append((d.pk, tokens(extract_text(d.file.path))))
            except Exception:  # noqa: BLE001
                existing.setdefault(key, []).append((d.pk, set()))
        self.stdout.write(self.style.WARNING(
            f"Loaded {sum(len(v) for v in existing.values())} existing EGCSE docs."
        ))

        target_root = os.path.join(str(settings.BASE_DIR), "content", "past_papers", "egcse")
        os.makedirs(target_root, exist_ok=True)

        files = sorted(f for f in os.listdir(src) if f.lower().endswith(".pdf"))
        if options["limit"]:
            files = files[: options["limit"]]

        copied = ingested = dup_content = dup_fp = skipped = failed = 0
        dup_list = []
        new_by_type = []

        for name in files:
            fp = self._classify(name)
            code = fp["subject_code"]
            subject = egcse_subject_by_code(code) if code else None
            if subject is None:
                skipped += 1
                self.stdout.write(self.style.WARNING(f"  SKIP {name}: no subject"))
                continue

            fp_key = (code, fp["year"], fp["paper_number"], fp["doc_type"])
            if self._fingerprint_exists(fp_key):
                dup_fp += 1
                dup_list.append(("fingerprint", name))
                continue

            if fp["doc_type"] in (SyllabusDocument.DocType.PAST_PAPER, SyllabusDocument.DocType.MARK_SCHEME):
                try:
                    toks = tokens(extract_text(os.path.join(src, name)))
                except Exception as exc:  # noqa: BLE001
                    self.stdout.write(self.style.WARNING(f"  FAIL-extract {name}: {exc}"))
                    continue
                if not toks:
                    dup_content += 1
                    dup_list.append(("empty", name))
                    continue
                for _pk, etoks in existing.get((code, fp["doc_type"]), []):
                    if jaccard(toks, etoks) >= 0.55:
                        dup_found = True
                        break
                else:
                    dup_found = False
                if dup_found:
                    dup_content += 1
                    dup_list.append(("content", name))
                    continue

            sub_dir = os.path.join(target_root, code)
            os.makedirs(sub_dir, exist_ok=True)
            dst = os.path.join(sub_dir, name)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(src, name), dst)
                copied += 1
            doc_title = f"{os.path.splitext(name)[0]} [{code}]"
            if not SyllabusDocument.objects.filter(subject=subject, title=doc_title).exists():
                doc = SyllabusDocument(
                    syllabus=subject.syllabus, subject=subject, title=doc_title,
                    doc_type=fp["doc_type"], source=SyllabusDocument.Source.EGCSE,
                    year=fp["year"], paper_number=fp["paper_number"],
                )
                try:
                    with open(dst, "rb") as fh:
                        doc.file.save(name, ContentFile(fh.read()), save=True)
                    n = process_document(doc)
                    ingested += 1
                    existing.setdefault((code, fp["doc_type"]), []).append(
                        (doc.pk, tokens(extract_text(dst)))
                    )
                except Exception as exc:  # noqa: BLE001 - never let one bad PDF kill the batch
                    doc.status = SyllabusDocument.Status.FAILED
                    doc.error = str(exc)[:200]
                    doc.save(update_fields=["status", "error"])
                    failed += 1
                    self.stdout.write(self.style.ERROR(f"  INGEST-FAIL {name}: {exc}"))
            else:
                skipped += 1
            new_by_type.append(fp["doc_type"])

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(
            f"Done. INGESTED {ingested}, copied {copied}, dup-fingerprint {dup_fp}, "
            f"dup-content {dup_content}, skipped {skipped}."
        ))
        by = Counter(new_by_type)
        self.stdout.write("New by type: " + ", ".join(f"{k}={v}" for k, v in by.items()))
        if dup_list:
            self.stdout.write(self.style.WARNING("Kept out (dup/empty):"))
            for why, n in dup_list[:40]:
                self.stdout.write(f"   [{why}] {n}")
    @staticmethod
    def _classify(name: str) -> dict:
        stem = os.path.splitext(name)[0]
        low = stem.lower()
        if "question paper" in low:
            ctype = SyllabusDocument.DocType.PAST_PAPER
        elif "mark scheme" in low:
            ctype = SyllabusDocument.DocType.MARK_SCHEME
        elif "syllabus" in low:
            ctype = SyllabusDocument.DocType.SYLLABUS
        else:
            ctype = SyllabusDocument.DocType.NOTES
        m = re.search(r"(20\d{2})", stem)
        year = int(m.group(1)) if m else None
        pm = re.search(r"(?:paper\s*)(\d)(?!\d)", low)
        paper = int(pm.group(1)) if pm else None
        return {
            "subject_code": egcse_code_from_filename(stem),
            "doc_type": ctype, "year": year, "paper_number": paper,
        }

    @staticmethod
    def _fingerprint_exists(fp_key) -> bool:
        code, year, paper, ctype = fp_key
        qs = SyllabusDocument.objects.filter(
            subject__code=code, doc_type=ctype, source=SyllabusDocument.Source.EGCSE
        )
        if year is None:
            qs = qs.filter(year__isnull=True)
        else:
            qs = qs.filter(year=year)
        if paper is None:
            qs = qs.filter(paper_number__isnull=True)
        else:
            qs = qs.filter(paper_number=paper)
        return qs.exists()
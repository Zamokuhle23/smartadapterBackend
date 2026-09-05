"""
Fix EGCSE document labels from CONTENT, not filenames.

The EGCSE download batch has unreliable filenames (question-paper names on
examiner reports, wrong subjects/years). ECESWA papers carry a rigid header
(`6871/01/O/N/2020`, `6882/02/SPECIMEN`, `Agriculture (6882)`), so parse
p1+p2 text and correct subject/year/paper_number/session/doc_type/title.

Usage:
    python manage.py fix_egcse_labels            # dry run, prints plan
    python manage.py fix_egcse_labels --apply    # apply + drop stale anchors
"""

import re

from django.core.management.base import BaseCommand

from apps.quiz.models import PageTopic, QuestionAnchor
from apps.syllabus.models import Subject, SyllabusDocument
from apps.syllabus.services.subject_map import EGCSE_SUBJECT_NAMES_BY_CODE

CODE_LINE = re.compile(
    r"(?m)^[^\S\n]*(\d{4})/(\d{1,2})/(SPECIMEN|[A-Z]/[A-Z]/(20\d{2}))")
# Bare code/paper without session ("6897/01", "6902/02") + month-year line.
BARE_CODE = re.compile(r"(?m)^[^\S\n]*(\d{4})/(\d{1,2})\s*$")
MONTH_YEAR = re.compile(
    r"(October\s*/\s*November|February\s*/\s*March|May\s*/\s*June)\s*(20\d{2})",
    re.IGNORECASE)
MONTH_SESSION = {"october": "ON", "february": "FM", "may": "MJ"}
NAME_CODE = re.compile(r"([A-Z][A-Za-z &'\-.]{3,50}?)\s*\((\d{4})\)")
REPORT_FOR = re.compile("examination reports?\s*(?:for\s*)?[-\u2013\u2014]?\s*(20\d{2})",
                        re.IGNORECASE)
SPECIMEN_YEAR = re.compile(r"specimen\s*(20\d{2})", re.IGNORECASE)
COPYRIGHT_YEAR = re.compile("\u00a9\s*ECESWA\s*(20\d{2})")
TOC_CODE = re.compile(r"subject code:\s*(\d{4})", re.IGNORECASE)
MS_MARK = re.compile(r"mark scheme", re.IGNORECASE)
REPORT_MARK = re.compile(
    r"examination reports?|examiner'?s? reports?|\breports?\s*-\s*20\d{2}",
    re.IGNORECASE)
TITLE_BRACKET = re.compile(r"\[\d{3,4}[A-Za-z]?\]\s*$")


def parse_head(text: str) -> dict:
    head = text[:3000]
    out = {"code": None, "paper": None, "year": None, "session": "",
           "kind": None, "name": ""}
    m = CODE_LINE.search(head)
    if m:
        out["code"] = m.group(1)
        try:
            out["paper"] = int(m.group(2))
        except ValueError:
            pass
        rest = m.group(3)
        if rest == "SPECIMEN":
            out["session"] = ""  # no specimen choice; leave blank
        else:
            bits = rest.split("/")
            sess = (bits[0] + bits[1]).upper() if len(bits) >= 3 else ""
            out["session"] = sess if sess in ("ON", "MJ", "FM") else ""
            out["year"] = int(m.group(4))
    nm = NAME_CODE.search(head)
    if nm:
        out["name"] = nm.group(1).strip()
        if not out["code"]:
            out["code"] = nm.group(2)
    if not out["code"]:
        bm = BARE_CODE.search(head)
        if bm and bm.group(1) in EGCSE_SUBJECT_NAMES_BY_CODE:
            out["code"] = bm.group(1)
            try:
                out["paper"] = int(bm.group(2))
            except ValueError:
                pass
            my = MONTH_YEAR.search(head)
            if my:
                out["year"] = int(my.group(2))
                for key, sess in MONTH_SESSION.items():
                    if my.group(1).lower().startswith(key):
                        out["session"] = sess
    if not out["code"]:
        # Bare "SUBJECT NAME NNNN" (no parens) -> resolve via name table.
        upper = " ".join(head.split())
        for code, name in EGCSE_SUBJECT_NAMES_BY_CODE.items():
            if re.search(r"\b" + re.escape(name.upper()) + r"\s+" + code
                         + r"\b", upper):
                out["code"] = code
                out["name"] = name
                break
    if not out["code"]:
        tm = TOC_CODE.search(head)
        if tm:
            out["code"] = tm.group(1)
    if not out["year"]:
        rm = REPORT_FOR.search(head)
        ym = re.search(r"\bYEAR\s*(20\d{2})", head)
        sm = SPECIMEN_YEAR.search(head)
        cm = COPYRIGHT_YEAR.search(head)
        if rm:
            out["year"] = int(rm.group(1))
        elif ym and REPORT_MARK.search(head):
            out["year"] = int(ym.group(1))
        elif sm:
            out["year"] = int(sm.group(1))
            out["session"] = out["session"] or ""
        elif cm:
            out["year"] = int(cm.group(1))
    low = head.lower()
    if MS_MARK.search(head):
        out["kind"] = SyllabusDocument.DocType.MARK_SCHEME
    elif REPORT_MARK.search(head):
        out["kind"] = "report"
    elif "question paper" in low or "read these instructions" in low \
            or "centre number" in low or "candidate name" in low:
        out["kind"] = SyllabusDocument.DocType.PAST_PAPER
    return out


class Command(BaseCommand):
    help = "Relabel EGCSE docs from content headers (dry run by default)."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true")

    def handle(self, *args, **options):
        import pymupdf

        apply = options["apply"]
        docs = SyllabusDocument.objects.filter(
            source=SyllabusDocument.Source.EGCSE).order_by("id")
        subjects = {s.code: s for s in Subject.objects.all()}
        changed = 0
        for doc in docs:
            try:
                pdf = pymupdf.open(doc.file.path)
                head = pdf[0].get_text("text")
                if len(pdf) > 1:
                    head += "\n" + pdf[1].get_text("text")
                pdf.close()
            except Exception as exc:  # noqa: BLE001
                self.stdout.write(f"  UNREADABLE {doc.id}: {exc}")
                continue
            p = parse_head(head)
            code_ok = bool(p["code"] and p["code"] in subjects)
            if not code_ok and p["kind"] != "report":
                self.stdout.write(
                    f"  NOCODE {doc.id} [{doc.subject.code}] {doc.title[:50]}")
                continue
            # Syllabi stay untouched: their filenames are the authority and
            # their text legitimately contains specimen excerpts.
            if doc.doc_type == SyllabusDocument.DocType.SYLLABUS:
                continue
            low_title = doc.title.lower()
            file_says_report = "report" in low_title
            file_says_ms = "mark scheme" in low_title
            file_says_qp = "question paper" in low_title
            updates = {}
            want_type = doc.doc_type
            if p["kind"] == "report":
                # Report markers are distinctive: filename QP/MS on top of
                # report content means a mislabeled file.
                want_type = SyllabusDocument.DocType.NOTES
            elif file_says_report and p["kind"] in (
                    SyllabusDocument.DocType.PAST_PAPER,
                    SyllabusDocument.DocType.MARK_SCHEME):
                # "Examiner Report" filename but clean paper content.
                want_type = p["kind"]
            elif file_says_ms and p["kind"] == \
                    SyllabusDocument.DocType.PAST_PAPER:
                # "Mark Scheme" filename but question-paper content, no MS
                # markers anywhere in the head.
                want_type = SyllabusDocument.DocType.PAST_PAPER
            elif file_says_qp and p["kind"] == \
                    SyllabusDocument.DocType.MARK_SCHEME:
                # "Question Paper" filename but mark-scheme content.
                want_type = SyllabusDocument.DocType.MARK_SCHEME
            if want_type != doc.doc_type:
                updates["doc_type"] = want_type
            final_type = updates.get("doc_type", doc.doc_type)
            if code_ok and p["code"] != doc.subject.code:
                updates["subject"] = p["code"]
            if p["year"] and p["year"] != doc.year and (
                    final_type != SyllabusDocument.DocType.NOTES
                    or p["kind"] == "report"):
                updates["year"] = p["year"]
            if p["paper"] and final_type in (
                    SyllabusDocument.DocType.PAST_PAPER,
                    SyllabusDocument.DocType.MARK_SCHEME) and \
                    p["paper"] != doc.paper_number:
                updates["paper_number"] = p["paper"]
            if final_type == SyllabusDocument.DocType.NOTES and \
                    doc.paper_number is not None:
                updates["paper_number"] = None
            if p["session"] and p["session"] != (doc.session or "") and \
                    final_type != SyllabusDocument.DocType.NOTES:
                updates["session"] = p["session"]
            if code_ok and p["code"] != doc.subject.code:
                stem = TITLE_BRACKET.sub("", doc.title).rstrip()
                updates["title"] = f"{stem} [{p['code']}]"
            # Cross-check the printed subject name against the code table.
            want_name = EGCSE_SUBJECT_NAMES_BY_CODE.get(p["code"], "")
            norm_name = " ".join(p["name"].split()).lower()
            if p["name"] and want_name and norm_name not in (
                    want_name.lower(), "egcse"):
                self.stdout.write(
                    self.style.WARNING(
                        f"  NAME-MISMATCH {doc.id}: header says "
                        f"{p['name']!r} but code {p['code']}={want_name}"))
            if not updates:
                continue
            changed += 1
            desc = ", ".join(f"{k}={v}" for k, v in updates.items())
            self.stdout.write(f"  {'APPLY' if apply else 'PLAN'} {doc.id} "
                              f"{doc.title[:45]!r} -> {desc}")
            if not apply:
                continue
            old_type = doc.doc_type
            if "subject" in updates:
                doc.subject = subjects[updates["subject"]]
            if "year" in updates:
                doc.year = updates["year"]
            if "paper_number" in updates:
                doc.paper_number = updates["paper_number"]
            if "session" in updates:
                doc.session = updates["session"]
            if "doc_type" in updates:
                doc.doc_type = updates["doc_type"]
            if "title" in updates:
                doc.title = updates["title"]
            doc.save()
            if old_type == SyllabusDocument.DocType.PAST_PAPER and \
                    doc.doc_type != old_type:
                na, _ = QuestionAnchor.objects.filter(
                    document=doc).delete()
                nt, _ = PageTopic.objects.filter(document=doc).delete()
                self.stdout.write(f"    dropped {na} anchors, {nt} topics")
        self.stdout.write(self.style.SUCCESS(
            f"{'Applied' if apply else 'Planned'} changes on {changed} docs."))

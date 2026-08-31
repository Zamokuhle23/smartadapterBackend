"""
Subject-resolution and source-priority tables for the Cambridge IGCSE
(PRIMARY) and ECESWA EGCSE (SECONDARY) past-paper corpus.

Strategy:
  * The IGCSE corpus is ~18x larger (5,300+ vs 295 papers).
  * EGCSE is a local derivative of IGCSE: pupils who master Cambridge
    IGCSE questions will excel at EGCSE.
  * Practice/questions therefore favour IGCSE provenance ~70% of the time
    and EGCSE ~30%, with every question clearly tagged.

Filename conventions handled here:
  IGCSE:  <subject_code>_<session><yy>_<qp|ms>_<variant>.pdf
          e.g. 0580_m24_qp_12.pdf, 0620_w17_ms_21.pdf
  EGCSE:  human title with subject name + paper number + (MS) marker
          e.g. "BIOLOGY 2", "Biology PAPER 3 - MS"
"""

import re

from apps.syllabus.models import Subject, Syllabus, SyllabusDocument

# Probability that a generated practice question is drawn from each source
# when that source has past-paper chunks available.
SOURCE_PRIORITY = {
    SyllabusDocument.Source.IGCSE: 0.7,  # primary - Cambridge side
    SyllabusDocument.Source.EGCSE: 0.3,  # secondary - local alignment
}

# IGCSE subject code -> EGCSE subject NAME (resolved against the local syllabus).
# Several Cambridge codes collapse into EGCSE's combined subjects (e.g. the
# combined "Physical Science" paper absorbs separate Chemistry/Physics).
IGCSE_TO_EGCSE_NAME = {
    "0580": "Mathematics",
    "4024": "Mathematics",  # older Cambridge IGCSE code for the same syllabus
    "0606": "Mathematics",  # Additional Mathematics - closest EGCSE match
    "0610": "Biology",
    "0620": "Physical Science",  # Chemistry is examined within Physical Science
    "0625": "Physical Science",  # Physics likewise
    "0653": "Physical Science",  # Combined Science
    "0500": "English Language",
    "0510": "English Language",  # English as a Second Language
    "0460": "Geography",
    "0470": "History",
    "0450": "Business Studies",
    "0452": "Accounting",
    "0648": "Food and Nutrition",
    "0445": "Design and Technology",
    "0600": "Agriculture",
    # ICT: Eswatini students sit Cambridge IGCSE ICT (no EGCSE version exists).
    "0417": "Information and Communication Technology",
    "3015": "Information and Communication Technology",  # older ICT spec, same subject
    # Codes below are present in the corpus but have no subject home yet:
    "0520": None,  # French
    "0474": None,  # (bundled with Agriculture folder but different subject)
}

# EGCSE subject names keyed by code, matching the seeded rows so we can
# resolve an IGCSE code -> local Subject row without hard-coding numeric codes.
# (Ordered by code.)
EGCSE_SUBJECT_NAMES_BY_CODE = {
    "6870": "First Language SiSwati",
    "6871": "SiSwati as a Second Language",
    "6873": "English Language",
    "6875": "Literature in English",
    "6880": "Mathematics",
    "6882": "Agriculture",
    "6884": "Biology",
    "6888": "Physical Science",
    "6890": "Geography",
    "6891": "History",
    "6893": "Religious Education",
    "6896": "Accounting",
    "6897": "Business Studies",
    "6899": "Economics",
    "6902": "Design and Technology",
    "6904": "Fashion and Fabrics",
    "6905": "Food and Nutrition",
}

# Ordered regex -> EGCSE code used to classify EGCSE filenames.
# Specific phrases must come before generic ones (e.g. "Physical Science"
# before "Science", "English Language" before "English").
EGCSE_FILENAME_RULES = [
    (re.compile(r"information and communication|information\s*&\s*communication|\bict\b"), "0417"),
    (re.compile(r"second\s*language.*siswati|siswati.*second|siswati\s*(as\s*a\s*)?second"), "6871"),
    (re.compile(r"first\s*language.*siswati|siswati\s*(as\s*a\s*)?first"), "6870"),
    (re.compile(r"physical\s*science|physics|chemistry"), "6888"),
    (re.compile(r"english\s*language|english\s*lang"), "6873"),
    (re.compile(r"literature"), "6875"),
    (re.compile(r"geograph|geo\b"), "6890"),
    (re.compile(r"history"), "6891"),
    (re.compile(r"religious|religion"), "6893"),
    (re.compile(r"accounting|accounts|bookkeeping"), "6896"),
    (re.compile(r"business"), "6897"),
    (re.compile(r"econom"), "6899"),
    (re.compile(r"design|d\s*and\s*t\b|d&t|dt\b"), "6902"),
    (re.compile(r"fashion|fabrics|ff\b"), "6904"),
    (re.compile(r"food|nutrition|fn\b"), "6905"),
    (re.compile(r"mathematics|maths|mathematical|math\b|add\s*maths?"), "6880"),
    (re.compile(r"biology|bio\b"), "6884"),
    (re.compile(r"agric"), "6882"),
    (re.compile(r"siswati"), "6870"),
]


def egcse_code_from_filename(name: str) -> str | None:
    """Infer the EGCSE subject code from a (possibly messy) local filename."""
    lowered = name.lower()
    for pattern, code in EGCSE_FILENAME_RULES:
        if pattern.search(lowered):
            return code
    return None


def egcse_subject_for_igcse_code(igcse_code: str) -> Subject | None:
    """Resolve a Cambridge IGCSE subject code to the matching local Subject row."""
    egcse_name = IGCSE_TO_EGCSE_NAME.get(igcse_code)
    if not egcse_name:
        return None
    return (
        Subject.objects.filter(syllabus__level=Syllabus.Level.EGCSE, name=egcse_name)
        .select_related("syllabus")
        .first()
    )


def egcse_subject_by_code(code: str) -> Subject | None:
    return Subject.objects.filter(code=code).select_related("syllabus").first()


# Education tiers, following the ECESWA/Cambridge convention where a tiered subject
# splits its papers: Core -> the lower paper numbers, Extended -> the higher ones.
TIER_PAPERS = {
    Subject.Tier.CORE: {1, 2},
    Subject.Tier.EXTENDED: {3, 4},
}


def tier_papers(tier: str) -> set[int]:
    """Return the set of paper numbers that belong to a student's tier."""
    if tier in Subject.Tier.CORE:
        return TIER_PAPERS[Subject.Tier.CORE]
    if tier in Subject.Tier.EXTENDED:
        return TIER_PAPERS[Subject.Tier.EXTENDED]
    return set()


def tier_label(tier: str) -> str:
    if tier == Subject.Tier.CORE:
        return "Core curriculum"
    if tier == Subject.Tier.EXTENDED:
        return "Extended curriculum"
    return ""


def tier_for(student, subject) -> str:
    """The student's tier for a subject, from their enrollment ('' if un-tiered)."""
    from apps.syllabus.models import Enrollment

    if subject.tiers_available:
        enrollment = Enrollment.objects.filter(student=student, subject=subject).first()
        return (enrollment.tier if enrollment else "") or ""
    return ""
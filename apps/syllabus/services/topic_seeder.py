"""
Best-effort topic seeding from an official syllabus PDF.

The real source of truth for a subject's skill tree is its official syllabus.
This module scans a syllabus document's text for top-level numbered section
headings (e.g. "1 Number", "2 Algebra", "3 Geometry") and materialises them as
Topic rows under the subject. It is deliberately conservative - it only accepts
top-level numbered headings, filters syllabus boilerplate, and never fabricates
LearningObjectives. RAG still carries the full syllabus content regardless;
this just gives the adaptive engine a usable topic index.

If a syllabus doesn't number its sections this way, no topics are added rather
than adding noise - the app still works (practice/exam don't require Topic rows;
they read the syllabus corpus via RAG).
"""

import re

from apps.syllabus.models import SyllabusDocument, Topic

# Matching: "16   Statistics and Probability" or "3 Algebra" or "2. Number"
HEADING_RE = re.compile(r"^\s*(\d{1,2})\s*[\.\)]?\s+([A-Z][A-Za-z ,&'()/:\-]{3,80})\s*$")
# Sub-headings like "10.1 Financial statements" are NOT top-level topics.
SUB_RE = re.compile(r"\d{1,2}\.\d")

# Boilerplate words that are section markers, not topics.
NOISE = {
    "introduction", "contents", "content", "assessment", "assessment objectives",
    "aims", "objectives", "syllabus", "paper", "papers", "examination",
    "cambridge", "university", "notes", "front", "back", "index", "glossary",
    "appendix", "acknowledgements", "update", "version", "learning outcomes",
    "subject", "school", "teachers", "scheme of assessment", "grading",
    "progression", "support", "resources", "access", "recording", "reporting",
    "equality", "safeguarding", "privacy", "what", "how", "why",
}

# Outcome/descriptor fragments common in prose-style syllabi - these are
# learning-outcome lines or grading descriptors, not top-level topics.
DESCRIPTOR_MARKERS = (
    "such as", "should be able", "all learners", "the oral test", "completely limited",
    "highly effective", "little relevance", "hard to understand", "fluent",
    "no relevance", "use of an electronic", "use of mathematical",
    "and mathematical tables", "words such as", "sample", "notes for teachers",
    "specimen", "mark scheme", "continued", "cont.",
)
# Titles that start like a verb/outcome clause rather than a topic heading.
VERB_STARTS = (
    "use of ", "add and subtract", "draw ", "identify ", "recognise ", "describe ",
    "use words", "state ", "give an ", "given that ", "find the ", "calculate ",
)


def _is_descriptor(title: str) -> bool:
    cheap = title.strip()
    klow = cheap.lower()
    if cheap.endswith(":") or cheap.endswith(";"):
        return True
    if "," in cheap:  # e.g. "Draw, recognise and describe..."
        return True
    if len(cheap.split()) > 12:
        return True
    if any(m in klow for m in DESCRIPTOR_MARKERS):
        return True
    if klow.startswith(VERB_STARTS):
        return True
    return False


def seed_topics_for_subject(subject, max_topics: int = 40) -> list[str]:
    """Create Topic rows from the subject's most recent syllabus document."""
    doc = (
        SyllabusDocument.objects.filter(
            subject=subject, doc_type=SyllabusDocument.DocType.SYLLABUS
        )
        .order_by("-created_at")
        .first()
    )
    if doc is None:
        return []
    from .ingestion import extract_text

    try:
        text = extract_text(doc.file.path)
    except Exception:  # noqa: BLE001 - a bad PDF shouldn't block the subject
        return []

    found = []  # ordered unique titles
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        m = HEADING_RE.match(line)
        if not m or SUB_RE.search(line):
            continue
        title = " ".join(m.group(2).split())
        key = title.lower()
        if len(title) < 4:
            continue
        if _is_descriptor(title):
            continue
        if any(word in key for word in NOISE):
            continue
        # Require at least one space => a real phrase, not a section word.
        if " " not in key:
            continue
        if key in seen:
            continue
        seen.add(key)
        found.append(title)
        if len(found) >= max_topics:
            break

    created = []
    for order, title in enumerate(found, start=1):
        if Topic.objects.filter(subject=subject, title=title).exists():
            continue
        Topic.objects.create(subject=subject, title=title, order=order)
        created.append(title)
    return created
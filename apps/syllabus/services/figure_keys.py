"""Figure identity: canonical traceable keys + exam-session parsing.

Canonical key: {SOURCE}-{SUBJECT}-{YEAR}-{SESSION?}-{PAPER?}-p{PAGE}-f{ORD}
e.g. IGCSE-0580-2024-MJ-P1-p12-f2  /  EGCSE-6880-2022-P1-p3-f1
"""

import re

SESSION_LETTERS = {"m": "MJ", "s": "ON", "w": "FM"}


def parse_session(name: str) -> str:
    """Exam session within the year: MJ (May/June), ON (Oct/Nov), FM (Feb/March).

    Reads Cambridge filename codes (_m24_, _s23_, _w17_) and plain-text
    variants (M/J, O/N, May June...). Returns "" when unknown (ECESWA human
    titles carry no session).
    """
    lowered = (name or "").lower()
    cam = re.search(r"_(?P<sess>[wms])(?P<yy>\d{2})_", lowered)
    if cam:
        return SESSION_LETTERS[cam.group("sess")]
    if re.search(r"\bm\s*/\s*j\b|may\s*(/|and\s*|&)?.{0,3}june\b", lowered):
        return "MJ"
    if re.search(r"\bo\s*/\s*n\b|oct\w*\s*(/|and\s*|&)?.{0,5}nov\w*\b", lowered):
        return "ON"
    if re.search(r"\bf\s*/\s*m\b|feb\w*\s*(/|and\s*|&)?.{0,5}mar\w*\b", lowered):
        return "FM"
    return ""


def build_figure_key(source: str, subject_code: str, year, session,
                     paper, page, ordinal) -> str:
    """Assemble the canonical key, omitting unknown segments (never blank)."""
    parts = [(source or "").upper() or "UNK",
             str(subject_code or "UNK"),
             str(year) if year else "XXXX"]
    if session:
        parts.append(session)
    if paper:
        parts.append(f"P{paper}")
    parts.append(f"p{page or 0}")
    parts.append(f"f{ordinal}")
    return "-".join(parts)


def figure_key_for(document, page, ordinal) -> str:
    """Key for one figure from its document row (session may be blank)."""
    subject = getattr(document, "subject", None)
    return build_figure_key(
        source=getattr(document, "source", ""),
        subject_code=getattr(subject, "code", None),
        year=getattr(document, "year", None),
        session=getattr(document, "session", "") or parse_session(
            getattr(document, "title", "")),
        paper=getattr(document, "paper_number", None),
        page=page,
        ordinal=ordinal,
    )


def crop_key_for(document, q_number: str) -> str:
    """Key for one cropped question, e.g. IGCSE-0580-2024-MJ-P1-Q6."""
    subject = getattr(document, "subject", None)
    parts = [(getattr(document, "source", "") or "").upper() or "UNK",
             str(getattr(subject, "code", None) or "UNK")]
    if getattr(document, "year", None):
        parts.append(str(document.year))
    session = getattr(document, "session", "") or parse_session(
        getattr(document, "title", ""))
    if session:
        parts.append(session)
    if getattr(document, "paper_number", None):
        parts.append(f"P{document.paper_number}")
    parts.append(f"Q{q_number}")
    return "-".join(parts)

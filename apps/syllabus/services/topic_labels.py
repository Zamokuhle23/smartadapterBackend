"""Canonical subtopic keys shared by topic-tree building and chat routing.

A PageTopic label like "Algebra: Factorisation, Expansion" canonicalises to
"algebra" so page variants collapse into one thread per subtopic.
"""

import re

JUNK_KEYS = {
    "blank page", "blank pages", "copyright", "copyright policy",
    "cover page", "contents page", "this page is blank",
}


def canonical_key(label: str) -> str:
    """Lowercased canonical key for a page label ("" when unusable)."""
    s = (label or "").strip()
    if not s:
        return ""
    s = re.split(r"[:;]", s, maxsplit=1)[0]
    s = re.split(r",|&|\+|/|(?<=\w) and (?=\w)", s, maxsplit=1)[0]
    s = re.sub(r"\s+", " ", s).strip().rstrip(".").lower()
    if not s or s in JUNK_KEYS:
        return ""
    return s[:120]

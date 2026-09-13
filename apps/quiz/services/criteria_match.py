"""Deterministic keyword/numeric matching of answers to atomic criteria.

No LLM: a criterion is awarded when ANY of its accept strings matches -
either all numbers in the accept appear in the answer (numeric, order-free,
tolerant) or the full normalized accept text appears. Criteria without
accept values abstain (None) for LLM/teacher handling.

Same function runs server-side (reference/parity) and is mirrored in
Kotlin on device (offline calculation marking).
"""

import math
import re

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list:
    out = []
    for tok in NUMBER_RE.findall(text or ""):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:  # noqa: BLE001 - ignore unparsable tokens
            continue
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _nospace(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def match_criterion(answer: str, criterion: dict):
    """True/False, or None when the criterion has nothing matchable."""
    accepts = criterion.get("accept") or []
    if not accepts:
        return None
    ans_nums = _numbers(answer)
    ans_norm = _norm(answer)
    ans_nospace = _nospace(answer)
    for accept in accepts:
        a = str(accept).strip()
        if not a:
            continue
        a_nums = _numbers(a)
        if a_nums and all(
                any(_close(x, y) for y in ans_nums) for x in a_nums):
            return True
        # Spaceless containment catches equations despite spacing
        # ("8 - 4x = 10" vs "8-4x=10"); minus signs are operators here,
        # not negative markers, which plain number parsing mishandles.
        if _nospace(a) and _nospace(a) in ans_nospace:
            return True
        if _norm(a) and _norm(a) in ans_norm:
            return True
    return False


def match_all(answer: str, criteria: list):
    """Returns (awards, total, abstained). awards = {id: bool}."""
    awards = {}
    abstained = []
    total = 0
    for c in criteria:
        hit = match_criterion(answer, c)
        if hit is None:
            abstained.append(c["id"])
            continue
        awards[c["id"]] = hit
        if hit:
            total += c["max_marks"]
    return awards, total, abstained

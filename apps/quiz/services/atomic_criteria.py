"""One-off decomposition of cached mark-scheme guidance into atomic criteria.

Each text anchor's marking_guidance (free prose, sometimes with M1/B1 codes)
becomes [{id: "M1", criterion: "...", max_marks: 1}] stored on
anchor.marking_criteria. Offline devices match student answers against these
instead of re-deriving structure; the sum of max_marks must equal the
anchor's marks or the row is left for human review (never silently stored).

LLM: the app's default chat provider (Azure on the VM). Strict validation
after parsing - a failed row returns None.
"""

from apps.quiz.services.generator import _chat, _extract_json_array


CRITERIA_PROMPT = """You decompose an examiner's mark-scheme guidance into atomic marking criteria.

QUESTION:
{question}

MARKING GUIDANCE:
{guidance}

TOTAL MARKS: {marks}

Rules:
- Output one item per independently awardable point. One check per item -
  split "correct substitution and simplification" into two items.
- "id" runs M1, M2, ... in order. "max_marks" is a positive integer.
- The max_marks values MUST sum to exactly {marks}.
- "criterion" states the observable evidence in the student's answer
  (<= 30 words). Never add points, facts, or methods absent from the guidance.
- If the guidance is vague or the total cannot be split exactly, output [].

Reply with ONLY a valid JSON array, no fences:
[{{"id": "M1", "criterion": "...", "max_marks": 1}}]"""


def extract_criteria(question_text: str, guidance: str, marks: int) -> list | None:
    """Returns validated criteria or None (leave for review)."""
    try:
        marks = int(marks)
    except (TypeError, ValueError):
        return None
    if marks < 1 or not (guidance or "").strip():
        return None
    try:
        raw = _chat([
            {"role": "system", "content": (
                "You output atomic marking criteria. ONLY the JSON array, no prose.")},
            {"role": "user", "content": CRITERIA_PROMPT.format(
                question=(question_text or "").strip()[:2000],
                guidance=guidance.strip()[:4000],
                marks=marks,
            )},
        ])
        items = _extract_json_array(raw)
    except Exception:  # noqa: BLE001 - unparseable, leave for review
        return None
    if not isinstance(items, list) or not items:
        return None
    out = []
    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            return None
        crit = str(item.get("criterion", "")).strip()[:300]
        try:
            mm = int(item.get("max_marks") or 0)
        except (TypeError, ValueError):
            return None
        if not crit or mm < 1:
            return None
        out.append({"id": f"M{i}", "criterion": crit, "max_marks": mm})
    if sum(c["max_marks"] for c in out) != marks:
        return None
    return out

"""
Smart practice: a question-by-question mix of text-only past-paper parts and
AI variants, sampled by exam paper contribution % and blueprint topic weights.

Rules implemented here:
- Only replaceable anchors (requires_figure=False, kind=text) enter the pool.
- "Seen" means SUBMITTED (a PaperAttempt exists). Scrolling/generating never
  counts. Unseen anchors are served as their original text; seen anchors
  rotate to AI variants.
- A source anchor is dropped after `variant_limit` distinct variants are
  attempted, so the student moves on to other papers instead of looping.
- Variants feed mastery (BKT) via their learning objective; original paper
  parts stay history-only.
"""

import random

from django.db.models import F
from django.utils import timezone

from apps.progress.models import MasteryEvent, MasteryRecord
from apps.progress.services.bkt import update_mastery
from apps.syllabus.models import Topic
from apps.syllabus.services.topic_labels import canonical_key

from ..models import PageTopic, PaperAttempt, PracticeSession, QuestionAnchor, QuizAttempt, QuizQuestion
from .cropper import anchor_marks, anchor_text, clean_part_text
from .generator import (
    QuizGenerationError,
    _chat,
    _extract_json_array,
    extract_keys,
    find_mark_scheme,
    generate_questions,
    grade_drawing,
    grade_structured_answer,
    grade_text,
    ms_excerpt_for,
    _is_bare_diagram_reference,
    get_exam_proposition,
)


def _paper_slots(subject, count: int, tier: str = "") -> list[int]:
    """Ordered list of paper numbers for `count` slots, sampled by weight."""
    prop = get_exam_proposition(subject, tier=tier)
    papers = prop.get("papers") or [{"paper_number": 1, "weight_pct": 100}]
    nums = [p["paper_number"] for p in papers]
    weights = [max(0.0, p.get("weight_pct", 0.0)) for p in papers]
    if sum(weights) <= 0:
        weights = [1.0] * len(nums)
    return random.choices(nums, weights=weights, k=count)


def _labels_to_topics(subject, labels: list[str]) -> list[Topic]:
    if not labels:
        return list(Topic.objects.filter(subject=subject))
    keys = {canonical_key(l) for l in labels if canonical_key(l)}
    return [t for t in Topic.objects.filter(subject=subject) if canonical_key(t.title) in keys]


def _pool_for_paper(subject, paper_number: int, topics: list[Topic]) -> list[QuestionAnchor]:
    """Text-only, replaceable anchors of one paper, optionally topic-scoped."""
    qs = QuestionAnchor.objects.filter(
        document__subject=subject,
        document__paper_number=paper_number,
        document__doc_type="past_paper",
        kind="text",
        requires_figure=False,
    )
    if topics:
        labels = set(PageTopic.objects.filter(
            document__subject=subject
        ).values_list("label", flat=True))
        matching = {lab for lab in labels if canonical_key(lab) in {
            canonical_key(t.title) for t in topics}}
        if not matching:
            return []
        qs = qs.filter(
            document__page_topics__label__in=list(matching),
            document__page_topics__page_number=F("page_number"),
        )
    return list(qs[:400])


def _pool_any_paper(subject, topics: list[Topic], exclude_paper: int | None = None,
                    ) -> list[QuestionAnchor]:
    """Unseen-anchor fallback across papers (keeps real questions flowing)."""
    qs = QuestionAnchor.objects.filter(
        document__subject=subject,
        document__doc_type="past_paper",
        kind="text",
        requires_figure=False,
    )
    if exclude_paper is not None:
        qs = qs.exclude(document__paper_number=exclude_paper)
    if topics:
        labels = set(PageTopic.objects.filter(
            document__subject=subject
        ).values_list("label", flat=True))
        matching = {lab for lab in labels if canonical_key(lab) in {
            canonical_key(t.title) for t in topics}}
        if not matching:
            return []
        qs = qs.filter(
            document__page_topics__label__in=list(matching),
            document__page_topics__page_number=F("page_number"),
        )
    return list(qs[:400])


def _seen_anchor_ids(student) -> set[int]:
    return set(PaperAttempt.objects.filter(student=student).values_list("anchor_id", flat=True))


def _variant_count(student, anchor_id: int) -> int:
    return QuizAttempt.objects.filter(
        student=student, question__source_anchor_id=anchor_id).count()


def _pick_anchor(pool, student, limit: int) -> QuestionAnchor | None:
    """Only UNSEEN anchors are served as their original text. Seen anchors
    rotate to AI variants (handled by the caller), never re-shown verbatim.
    """
    seen = _seen_anchor_ids(student)
    unseen = [a for a in pool if a.id not in seen]
    return random.choice(unseen) if unseen else None


def _topics_for_anchor(anchor) -> list[Topic]:
    label = PageTopic.objects.filter(
        document=anchor.document, page_number=anchor.page_number
    ).values_list("label", flat=True).first()
    return _labels_to_topics(anchor.document.subject, [label]) if label else []


def _generate_text_variant(subject, anchor: QuestionAnchor | None,
                           tier: str, topic_labels: list[str] | None = None) -> QuizQuestion | None:
    """A text-only AI question on the anchor's topic (never its diagram).

    Uses the topic as the generation scope (not the anchor text), so follow-up
    parts of diagram stems become a topic question answerable without the
    figure. Any generated question still referencing a figure is dropped.
    """
    if anchor is not None:
        topics = _topics_for_anchor(anchor)
    else:
        topics = _labels_to_topics(subject, topic_labels or [])
    topic_ids = [t.id for t in topics] or None
    # Retry: figure-heavy topics often come back diagram-tied; fresh samples
    # usually yield a clean text-only question within a few tries.
    for _ in range(3):
        try:
            questions = generate_questions(subject, count=3, tier=tier, topic_ids=topic_ids)
        except QuizGenerationError:
            continue
        for q in questions:
            if q.figures.exists():
                q.figures.clear()
            if _is_bare_diagram_reference(q):
                q.delete()
                continue
            if anchor is not None:
                q.source_anchor = anchor
                q.save(update_fields=["source_anchor"])
            return q
    return None


PART_VARIANT_PROMPT = """You are an experienced Eswatini examiner writing text-only
practice variants of real past-paper parts. Rewrite EACH part below as a fresh
question testing the SAME skill and command words, with DIFFERENT numbers,
names and context where applicable.

HARD RULES:
- Text only: NEVER mention or require any diagram, figure, table, graph,
  image or data not printed in the question itself.
- Keep the same marks as the original part (shown in brackets).
- Reply with ONLY a JSON array, no fences:
  [{{"qid": "4a", "question": "...", "marks": 2, "marking_guidance": "..."}}]
  marking_guidance = short model answer + how marks are awarded.

SUBJECT: {subject_name} ({subject_code}){topic_line}
PARTS:
{parts}"""


def generate_part_variants(doc, page_no: int) -> dict[str, QuizQuestion]:
    """One LLM call producing text-only variants for a page's text parts.

    Results persist as QuizQuestion rows linked by source_anchor, so repeat
    views are instant. Parts that already have a variant are skipped.
    Returns {qid: QuizQuestion}.
    """
    anchors = list(QuestionAnchor.objects.filter(
        document=doc, page_number=page_no, kind="text").order_by("qid"))
    have = {q.source_anchor_id: q for q in QuizQuestion.objects.filter(
        source_anchor__in=[a.id for a in anchors])}
    todo = []
    for anchor in anchors:
        if anchor.id in have:
            continue
        try:
            text = anchor_text(doc.file.path, page_no, anchor.bbox)
        except Exception:  # noqa: BLE001
            continue
        cleaned = clean_part_text(text)
        if not cleaned:
            continue
        marks = anchor_marks(text) or anchor.marks or 2
        todo.append((anchor, cleaned, marks))
    made: dict[str, QuizQuestion] = {
        a.qid: have[a.id] for a in anchors if a.id in have}
    if not todo:
        return made
    label = PageTopic.objects.filter(
        document=doc, page_number=page_no).values_list(
        "label", flat=True).first() or ""
    parts_block = "\n---\n".join(
        f"[{a.qid}] ({marks} marks)\n{cleaned}"
        for a, cleaned, marks in todo)
    try:
        raw = _chat([
            {"role": "system", "content": (
                "You write syllabus-accurate exam questions. Output ONLY valid JSON.")},
            {"role": "user", "content": PART_VARIANT_PROMPT.format(
                subject_name=doc.subject.name if doc.subject else "",
                subject_code=doc.subject.code if doc.subject else "",
                topic_line=f"\nTOPIC: {label}" if label else "",
                parts=parts_block[:6000],
            )},
        ])
        items = _extract_json_array(raw)
    except QuizGenerationError:
        return made
    by_qid = {a.qid: a for a, _, _ in todo}
    objective = None
    if label and doc.subject is not None:
        from apps.syllabus.models import LearningObjective
        objective = LearningObjective.objects.filter(
            topic__subject=doc.subject, topic__title__iexact=label).first()
    for item in items:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("qid", ""))
        anchor = by_qid.get(qid)
        question_text = str(item.get("question", "")).strip()
        if anchor is None or not question_text:
            continue
        if _is_bare_diagram_reference_text(question_text):
            continue
        try:
            marks = max(1, min(25, int(item.get("marks") or 2)))
        except (TypeError, ValueError):
            marks = 2
        made[qid] = QuizQuestion.objects.create(
            subject=doc.subject,
            objective=objective,
            topic_title=label,
            format=QuizQuestion.Format.STRUCTURED,
            question_text=question_text,
            marks=marks,
            marking_guidance=str(item.get("marking_guidance", "")),
            adapted_from_past_paper=True,
            source_anchor=anchor,
        )
    return made


def _is_bare_diagram_reference_text(text: str) -> bool:
    """Figure-reference check for not-yet-persisted variant text."""
    from .generator import _BARE_REFERENCE_RE
    return bool(_BARE_REFERENCE_RE.search(text or ""))


def _anchor_slice(anchor: QuestionAnchor) -> dict:
    return {"doc_id": anchor.document_id, "page_number": anchor.page_number,
            "bbox": list(anchor.bbox)}


def _anchor_item(anchor: QuestionAnchor, index: int, paper_label: str) -> dict:
    text = anchor_text(anchor.document.file.path, anchor.page_number, anchor.bbox)
    marks = anchor_marks(text) or anchor.marks or 2
    return {
        "index": index, "kind": "anchor", "anchor_id": anchor.id,
        "question_id": None, "label": anchor.qid,
        "paper_number": anchor.document.paper_number or 1,
        "paper_label": paper_label, "marks": marks, "text": text,
        "format": "mcq" if anchor.correct_index is not None else "structured",
        "slice": _anchor_slice(anchor), "answered": False,
    }


def _serialize_question(question: QuizQuestion) -> dict:
    return {
        "id": question.id,
        "question_text": question.question_text,
        "options": question.options,
        "format": question.format,
        "marks": question.marks,
        "topic_title": question.topic_title,
        "paper_label": question.paper_label,
        "source_year": question.source_year,
        "figure_urls": [],
        "adapted_from_past_paper": question.adapted_from_past_paper,
        "source": question.source,
    }


def _variant_item(question: QuizQuestion, anchor: QuestionAnchor | None,
                  index: int, paper_number: int, paper_label: str) -> dict:
    return {
        "index": index, "kind": "variant",
        "anchor_id": anchor.id if anchor else None,
        "question_id": question.id,
        "label": anchor.qid if anchor else f"Q{index + 1}",
        "paper_number": paper_number, "paper_label": paper_label,
        "marks": question.marks, "question": _serialize_question(question),
        "slice": _anchor_slice(anchor) if anchor else None, "answered": False,
    }


def start_smart_session(student, subject, topics: list[str],
                        count: int = 20, tier: str = "") -> PracticeSession:
    count = max(1, min(50, int(count or 20)))
    slots = _paper_slots(subject, count, tier=tier)
    return PracticeSession.objects.create(
        student=student, subject=subject, topics=topics,
        total_questions=count, plan=slots)


def _paper_label(subject, paper_number: int) -> str:
    prop = get_exam_proposition(subject)
    for p in prop.get("papers", []):
        if p.get("paper_number") == paper_number:
            return p.get("label") or f"Paper {paper_number}"
    return f"Paper {paper_number}"


def serve_next(session: PracticeSession, tier: str = "") -> dict | None:
    """Produce the next item (anchor or variant) for the session."""
    if session.status == PracticeSession.Status.COMPLETED:
        return None
    index = len(session.items)
    if index >= session.total_questions:
        session.status = PracticeSession.Status.COMPLETED
        session.completed_at = timezone.now()
        session.save(update_fields=["status", "completed_at"])
        return None
    paper_number = int(session.plan[index]) if index < len(session.plan) else 1
    paper_label = _paper_label(session.subject, paper_number)
    topics = _labels_to_topics(session.subject, session.topics)
    pool = _pool_for_paper(session.subject, paper_number, topics)
    anchor = _pick_anchor(pool, session.student, session.variant_limit)
    if anchor is not None:
        item = _anchor_item(anchor, index, paper_label)
    else:
        # No fresh anchor in this paper's pool: prefer an unseen anchor from
        # any other paper (real questions first, no LLM cost); then a variant
        # of a seen anchor still under its limit; only then give up.
        # (A syllabus-only variant is generated only when a seen source or
        # an empty pool forces it - see below.)
        other = _pick_anchor(
            _pool_any_paper(session.subject, topics, exclude_paper=paper_number),
            session.student, session.variant_limit)
        if other is not None:
            item = _anchor_item(other, index, _paper_label(
                session.subject, other.document.paper_number or 1))
        else:
            seen = _seen_anchor_ids(session.student)
            eligible_sources = [
                a for a in pool if a.id in seen
                and _variant_count(session.student, a.id) < session.variant_limit
            ]
            source = random.choice(eligible_sources) if eligible_sources else None
            question = _generate_text_variant(
                session.subject, source, tier, topic_labels=session.topics)
            if question is not None:
                item = _variant_item(question, source, index, paper_number, paper_label)
            else:
                item = {
                    "index": index, "kind": "variant", "anchor_id": None,
                "question_id": None, "label": f"Q{index + 1}",
                "paper_number": paper_number, "paper_label": paper_label,
                "marks": 0, "question": None, "slice": None, "answered": False,
                "error": "Could not prepare a question - try again.",
            }
    items = list(session.items)
    items.append(item)
    session.items = items
    session.save(update_fields=["items"])
    return item


# ---------------------------------------------------------------------------
# Grading (mirrors PaperAnswerView / AnswerQuizView)
# ---------------------------------------------------------------------------


def _ensure_anchor_keys(doc, anchor, text: str):
    if anchor.marking_guidance:
        return
    ms = find_mark_scheme(doc.subject, doc.year, doc.paper_number)
    if ms is None:
        return
    base = "".join(ch for ch in anchor.qid if ch.isdigit())
    for qid in (anchor.qid, base):
        if not qid:
            continue
        excerpt = ms_excerpt_for(ms, qid)
        if not excerpt:
            continue
        keys = extract_keys(text, qid, excerpt)
        if not keys:
            continue
        anchor.marks = keys["marks"]
        anchor.correct_index = keys["correct_index"]
        anchor.marking_guidance = keys["marking_guidance"]
        anchor.save(update_fields=["marks", "correct_index", "marking_guidance"])
        return


def _answer_anchor(student, item: dict, data: dict) -> dict:
    anchor = QuestionAnchor.objects.select_related("document").filter(pk=item["anchor_id"]).first()
    if anchor is None:
        return {"_error": "Unknown question"}
    doc = anchor.document
    try:
        text = anchor_text(doc.file.path, anchor.page_number, anchor.bbox)
    except Exception:  # noqa: BLE001
        text = ""
    marks = anchor_marks(text) or anchor.marks or 2
    _ensure_anchor_keys(doc, anchor, text)

    awarded = max_marks = None
    feedback = ""
    correct = False
    selected = data.get("selected_index")
    if selected is not None and anchor.kind != "drawing":
        if anchor.correct_index is None:
            return {"_error": "This part is not keyed yet"}
        try:
            selected = int(selected)
        except (TypeError, ValueError):
            return {"_error": "selected_index required"}
        if selected not in (0, 1, 2, 3):
            return {"_error": "selected_index out of range"}
        correct = selected == anchor.correct_index
        max_marks = int(marks)
        awarded = float(max_marks) if correct else 0.0
        answer_text = data.get("answer_text") or ""
    else:
        answer_text = (data.get("answer_text") or "").strip()
        drawing_b64 = data.get("drawing") or ""
        if isinstance(drawing_b64, str) and drawing_b64.startswith("data:"):
            drawing_b64 = drawing_b64.split(",", 1)[-1]
        if not answer_text and not drawing_b64:
            return {"_error": "answer_text required"}
        try:
            if drawing_b64:
                awarded, max_marks, feedback = grade_drawing(
                    question_text=text or f"Paper Q{anchor.qid}",
                    guidance=anchor.marking_guidance or "(none supplied)",
                    marks=marks, image_b64=drawing_b64)
            else:
                awarded, max_marks, feedback = grade_text(
                    question_text=text or f"Paper Q{anchor.qid}",
                    guidance=anchor.marking_guidance or "(none supplied)",
                    marks=marks, answer_text=answer_text)
        except QuizGenerationError as exc:
            return {"_error": str(exc)}
        correct = awarded >= max_marks * 0.5

    attempt = PaperAttempt.objects.create(
        student=student, anchor=anchor, answer_text=answer_text,
        awarded_marks=awarded, correct=correct,
        latency_ms=data.get("latency_ms"))
    return {
        "correct": correct,
        "correct_index": anchor.correct_index,
        "explanation": feedback,
        "mastery": None,
        "awarded_marks": awarded,
        "max_marks": int(max_marks) if max_marks is not None else None,
        "feedback": feedback,
        "model_answer": anchor.marking_guidance or "",
        "attempt_id": attempt.id,
    }


def _answer_variant(student, item: dict, data: dict, tier: str) -> dict:
    question = QuizQuestion.objects.select_related("objective", "subject").filter(
        pk=item["question_id"]).first()
    if question is None:
        return {"_error": "Unknown question"}

    correct = False
    selected = None
    awarded = max_marks = None
    feedback = ""
    explanation = question.explanation

    if question.format == QuizQuestion.Format.STRUCTURED:
        answer_text = (data.get("answer_text") or "").strip()
        if not answer_text:
            return {"_error": "answer_text required"}
        try:
            awarded, max_marks, feedback = grade_structured_answer(question, answer_text)
        except QuizGenerationError as exc:
            return {"_error": str(exc)}
        correct = awarded >= max_marks * 0.5
        explanation = explanation or feedback
    else:
        try:
            selected = int(data.get("selected_index"))
        except (TypeError, ValueError):
            return {"_error": "selected_index required"}
        if not (0 <= selected < len(question.options)):
            return {"_error": "selected_index out of range"}
        correct = selected == question.correct_index

    attempt = QuizAttempt.objects.create(
        student=student, question=question,
        selected_index=selected,
        answer_text=data.get("answer_text") or "",
        awarded_marks=awarded, feedback=feedback, correct=correct,
        latency_ms=data.get("latency_ms"))

    mastery = None
    if question.objective is not None:
        MasteryEvent.objects.create(
            student=student, objective=question.objective,
            correct=correct, latency_ms=data.get("latency_ms"))
        record, _created = MasteryRecord.objects.get_or_create(
            student=student, objective=question.objective,
            defaults={"subject": question.subject})
        record.attempts += 1
        if correct:
            record.correct_count += 1
        record.mastery = update_mastery(record.mastery, correct)
        record.save(update_fields=["attempts", "correct_count", "mastery"])
        mastery = record.mastery

    return {
        "correct": correct,
        "correct_index": question.correct_index,
        "explanation": explanation,
        "mastery": mastery,
        "awarded_marks": awarded,
        "max_marks": int(max_marks) if max_marks is not None else None,
        "feedback": feedback,
        "model_answer": question.marking_guidance or "",
        "attempt_id": attempt.id,
    }


def answer_item(session: PracticeSession, index: int, data: dict,
                tier: str = "") -> tuple[dict, int]:
    """Grade one item (anchor -> PaperAttempt; variant -> QuizAttempt+BKT)."""
    item = next((it for it in session.items if it.get("index") == index), None)
    if item is None:
        return {"detail": "Unknown item"}, 404
    if item.get("answered"):
        return {"detail": "Already answered"}, 400

    result = (_answer_anchor(session.student, item, data)
              if item.get("kind") == "anchor"
              else _answer_variant(session.student, item, data, tier))
    if result.get("_error"):
        return {"detail": result["_error"]}, 400
    result.pop("_error", None)

    item["answered"] = True
    items = list(session.items)
    session.items = items
    session.save(update_fields=["items"])
    return result, 200


def session_summary(session: PracticeSession) -> dict:
    """Compiled answers: ordered rows + per-paper and total marks."""
    rows = []
    by_paper: dict[int, dict] = {}
    total_awarded = total_possible = 0.0
    for item in session.items:
        row = {
            "index": item.get("index"),
            "label": item.get("label"),
            "paper_number": item.get("paper_number"),
            "paper_label": item.get("paper_label"),
            "marks": item.get("marks"),
            "answered": item.get("answered", False),
            "your_answer": "",
            "awarded": None,
            "max": None,
            "correct": None,
            "model_answer": "",
            "feedback": "",
        }
        if item.get("answered"):
            attempt = _attempt_for_item(session.student, item)
            if attempt:
                row["your_answer"] = getattr(attempt, "answer_text", "") or ""
                row["awarded"] = attempt.awarded_marks
                row["max"] = item.get("marks")
                row["correct"] = attempt.correct
                row["feedback"] = getattr(attempt, "feedback", "") or ""
                q = getattr(attempt, "question", None)
                if q is not None:
                    row["model_answer"] = q.marking_guidance or q.explanation or ""
                elif getattr(attempt, "anchor", None) is not None:
                    row["model_answer"] = attempt.anchor.marking_guidance or ""
                awarded = attempt.awarded_marks or 0.0
                mx = item.get("marks") or 0
                total_awarded += awarded
                total_possible += mx
                pn = item.get("paper_number") or 1
                pp = by_paper.setdefault(pn, {
                    "paper_label": item.get("paper_label"),
                    "awarded": 0.0, "possible": 0})
                pp["awarded"] += awarded
                pp["possible"] += mx
        rows.append(row)
    return {
        "session_id": session.id,
        "status": session.status,
        "total_awarded": round(total_awarded, 1),
        "total_possible": int(total_possible),
        "answered": sum(1 for r in rows if r["answered"]),
        "per_paper": [
            {"paper_number": pn, **pp} for pn, pp in sorted(by_paper.items())
        ],
        "items": rows,
    }


def _attempt_for_item(student, item: dict):
    if item.get("kind") == "anchor":
        return PaperAttempt.objects.filter(
            student=student, anchor_id=item.get("anchor_id")
        ).order_by("-created_at").first()
    return QuizAttempt.objects.filter(
        student=student, question_id=item.get("question_id")
    ).order_by("-created_at").first()
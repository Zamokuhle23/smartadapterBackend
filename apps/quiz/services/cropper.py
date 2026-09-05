"""Question-boundary detection + cropping for past-paper pages.

Pipeline (digital PDFs: PyMuPDF gives text + boxes + style natively):
  page dict (words/spans with size, bold, bbox)
    -> top-level anchors  ("6.", "10)") + sub-anchors ((a), (ii))
    -> regions (anchor .. next anchor, multi-page spans)
    -> crops (render once, slice, confidence)

Confidence: bold/large sequential anchors score high; regex-only hits get
flagged for QC. Nothing here needs OCR (tesseract) on digital PDFs.
"""

import re

TOP_ANCHOR = re.compile(r"^\s*(\d{1,2})(?:[\.\)]|\s{2,}|\s*$)")
LOOSE_ANCHOR = re.compile(r"^\s*(\d{1,2})\s+\S")
FOOTER = re.compile(r"©|specimen|turn over|^\s*page\s*\d+", re.IGNORECASE)
SUB_ANCHOR = re.compile(r"^\s*(\(?[a-z]\)|\(?[ivx]+\)|\([0-9]+\))")


def _line_text(line) -> str:
    return "".join(s.get("text", "") for s in line.get("spans", []))


def _line_size(line) -> float:
    sizes = [s.get("size", 0) for s in line.get("spans", []) if s.get("text", "").strip()]
    return max(sizes) if sizes else 0.0


def _line_bold(line) -> bool:
    return any((s.get("flags", 0) & 16) and s.get("text", "").strip()
               for s in line.get("spans", []))


def page_lines(page):
    """Flat lines with geometry: [(x0, top, text, size, bold)]."""
    out = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            text = _line_text(line)
            if not text.strip():
                continue
            x0, top, x1, bottom = line.get("bbox", (0, 0, 0, 0))
            out.append({"x0": x0, "top": top, "x1": x1, "bottom": bottom,
                        "text": text, "size": _line_size(line),
                        "bold": _line_bold(line)})
    out.sort(key=lambda ln: (round(ln["top"]), ln["x0"]))
    return out


def detect_anchors(lines, page_width: float, expected=None):
    """Top-level question starts: number + (bold|large) + left zone + sequence.

    expected threads across pages so numbering stays strictly sequential;
    returns (anchors, expected).
    """
    counts = {}
    for ln in lines:
        counts[ln["size"]] = counts.get(ln["size"], 0) + 1
    body_size = max(counts, key=lambda s: counts[s]) if counts else 10.0
    anchors = []
    for ln in lines:
        m = TOP_ANCHOR.match(ln["text"])
        loose = False
        if not m:
            # Single-space variant ("8 The function..."): only with bold,
            # left and exact sequence (kills "6 cm" false positives).
            m = LOOSE_ANCHOR.match(ln["text"])
            loose = True
        if not m:
            continue
        num = int(m.group(1))
        strong = ln["bold"] or ln["size"] >= body_size * 1.08
        left = ln["x0"] < page_width * 0.35
        if not left:
            continue
        if expected is None:
            if strong and not loose:
                anchors.append({"number": str(num), "top": ln["top"],
                                "confident": ln["bold"]})
                expected = num + 1
        elif num == expected and (strong or not loose):
            anchors.append({"number": str(num), "top": ln["top"],
                            "confident": strong and not loose})
            expected = num + 1
        elif (strong and not loose and expected is not None
                and expected < num <= expected + 3):
            # Papers occasionally skip a number: follow with low confidence
            # so one gap doesn't swallow the rest of the paper.
            anchors.append({"number": str(num), "top": ln["top"],
                            "confident": False})
            expected = num + 1
    return anchors, expected


def detect_questions(doc):
    """Full-paper detection: [{number, pages: [(pno, top, bottom)], confident}]."""
    questions = []
    open_q = None
    expected = None
    page_lines_cache = {}
    for page in doc:
        pno = page.number + 1
        height = float(page.rect.height)
        lines = page_lines(page)
        page_lines_cache[pno] = lines
        anchors, expected = detect_anchors(lines, page.rect.width, expected)
        if open_q is not None:
            if anchors:
                # Previous question spills onto this page: close it here.
                open_q["pages"].append((pno, 0.0, anchors[0]["top"]))
                questions.append(open_q)
                open_q = None
            else:
                # Whole page belongs to the spilling question.
                open_q["pages"].append((pno, 0.0, height))
                continue
        for i, a in enumerate(anchors):
            bottom = anchors[i + 1]["top"] if i + 1 < len(anchors) else height
            q = {"number": a["number"],
                 "pages": [(pno, a["top"], bottom)],
                 "confident": a["confident"]}
            # Last anchor runs to page bottom: may continue next page.
            if i == len(anchors) - 1:
                open_q = q
            else:
                questions.append(q)
    if open_q is not None:
        questions.append(open_q)
    # Trim footer furniture (copyright / specimen / turn-over strips) from
    # page-bottom slices, then cap spans and drop slivers.
    for q in questions:
        trimmed = []
        for pno, top, bottom in q["pages"]:
            lines = page_lines_cache.get(pno, [])
            for ln in reversed(lines):
                if ln["top"] < top:
                    continue
                if FOOTER.search(ln["text"]):
                    bottom = min(bottom, ln["top"])
                else:
                    break
            trimmed.append((pno, top, bottom))
        q["pages"] = trimmed[:3]
    kept = []
    for q in questions:
        total = sum(b - t for _, t, b in q["pages"])
        if total >= 40 or len(q["pages"]) > 1:
            kept.append(q)
    return kept


def crop_question(page_images, question, zoom: float):
    """Slice page renders into question crops. Returns [PNG bytes]."""
    from io import BytesIO

    out = []
    for pno, top, bottom in question["pages"]:
        img = page_images[pno]
        w, h = img.size
        box = (0, max(0, int(top * zoom)),
               w, min(h, int(bottom * zoom)))
        if box[3] <= box[1]:
            continue
        buf = BytesIO()
        img.crop(box).save(buf, format="PNG")
        out.append(buf.getvalue())
    return out
